#!/usr/bin/env python3
"""Bounded native Isaac Gym/TacSL physics+camera probe without task dependencies."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physics", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--contactworld-camera", action="store_true",
                        help="Check released front-camera conventions against live matrices and known box/ground geometry")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"physics_requested": args.physics, "python": sys.version,
              "status": "running", "stages": [], "pid": os.getpid()}

    def stage(name, **details):
        report["stages"].append({"stage": name, **details})
        (args.output / "probe.json").write_text(json.dumps(report, indent=2)+"\n")
        print(name, details, flush=True)

    stage("import_isaacgym_begin")
    from isaacgym import gymapi
    import numpy as np
    from PIL import Image
    gym = gymapi.acquire_gym()
    stage("import_isaacgym_pass", native_tactile_api=[x for x in dir(gym) if "sdf" in x.lower() or "compliant" in x.lower()])
    sim = None
    try:
        params = gymapi.SimParams()
        params.dt = 1/60
        params.substeps = 2
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        params.physx.use_gpu = args.physics == "gpu"
        params.use_gpu_pipeline = False  # native CPU readback; GPU tensor pipeline tested separately
        params.physx.num_position_iterations = 8
        stage("create_sim_begin")
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError("create_sim returned None")
        stage("create_sim_pass")
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
        env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,2), 1)
        asset = gym.create_box(sim, .1, .1, .1, gymapi.AssetOptions())
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(.5 if args.contactworld_camera else 0,0,.5)
        actor = gym.create_actor(env, asset, pose, "falling_box", 0, 0)
        camera = gymapi.CameraProperties()
        width, height = (256, 256) if args.contactworld_camera else (320, 240)
        camera.width, camera.height = width, height
        camera.horizontal_fov = 75
        camera.enable_tensors = False
        handle = gym.create_camera_sensor(env, camera)
        if handle < 0:
            raise RuntimeError("camera creation failed")
        if args.contactworld_camera:
            gym.set_camera_transform(handle, env, gymapi.Transform(
                gymapi.Vec3(.68, 0, .15), gymapi.Quat(-.258819045, 0, .965925826, 0)))
        else:
            gym.set_camera_location(handle, env, gymapi.Vec3(1,1,.8), gymapi.Vec3(0,0,.1))
        gym.prepare_sim(sim)
        stage("physics_begin")
        for _ in range(args.steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        position = [float(states['pose']['p'][name][0]) for name in ('x','y','z')]
        if not np.isfinite(position).all() or not .025 < position[2] < .10:
            raise RuntimeError("Falling-box physics invalid: %s" % position)
        stage("physics_pass", box_position=position)
        stage("camera_begin")
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        rgba = np.asarray(gym.get_camera_image(sim, env, handle, gymapi.IMAGE_COLOR)).reshape(height,width,4)
        depth = np.asarray(gym.get_camera_image(sim, env, handle, gymapi.IMAGE_DEPTH)).reshape(height,width)
        valid = np.isfinite(depth) & (depth < 0)
        if rgba[..., :3].std() < 1 or not valid.any():
            raise RuntimeError("Empty/invalid camera output")
        Image.fromarray(rgba[..., :3]).save(args.output / "rgb.png")
        np.savez_compressed(args.output / "camera.npz", rgba=rgba, depth=depth,
                            view=np.asarray(gym.get_camera_view_matrix(sim, env, handle)),
                            projection=np.asarray(gym.get_camera_proj_matrix(sim, env, handle)))
        stage("camera_pass", valid_depth_fraction=float(valid.mean()))
        if args.contactworld_camera:
            from check_contactworld_geometry import rotation
            view = np.asarray(gym.get_camera_view_matrix(sim, env, handle))
            projection = np.asarray(gym.get_camera_proj_matrix(sim, env, handle))
            inv_view = np.linalg.inv(view)
            body_r = rotation([-.258819045, 0, .965925826, 0])
            r = body_r @ np.array([[0,0,-1],[-1,0,0],[0,1,0]])
            t = np.array([.68, 0, .15])
            # Isaac Gym returns row-vector view matrices.
            rotation_error = float(np.abs(inv_view[:3,:3] - r.T).max())
            translation_error = float(np.abs(inv_view[3,:3] - t).max())
            expected_focal = width/2/np.tan(np.deg2rad(75)/2)
            focal = np.array([projection[0,0]*width/2, projection[1,1]*height/2])
            v, u = np.indices(depth.shape)
            z = -depth[valid]
            optical = np.column_stack([(u[valid]-width/2)*z/focal[0],
                                       (v[valid]-height/2)*z/focal[1], z])
            h = np.column_stack([optical, np.ones(len(z))])
            stored = (h @ inv_view.T)[:,:3]  # Exact upstream conversion, before base transform.
            correction = r @ np.diag([1,-1,-1]) @ r
            corrected = stored @ correction.T + t
            h[:,:3] *= [1,-1,-1]
            world = (h @ inv_view)[:,:3]
            conversion_error = float(np.abs(corrected-world).max())
            # Depth samples lie at pixel centers; upstream uses integer corners.
            h[:,0] += .5*z/focal[0]
            h[:,1] -= .5*z/focal[1]
            center_world = (h @ inv_view)[:,:3]
            def surface_p99(points):
                local = points-np.asarray(position)
                q = np.abs(local)-.05
                box_sdf = np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
                error = np.minimum(np.abs(points[:,2]), np.abs(box_sdf))
                return float(np.quantile(error[z < 1.0],.99))
            integer_surface_p99 = surface_p99(world)
            center_surface_p99 = surface_p99(center_world)
            checks = dict(rotation_max_error=rotation_error, translation_max_error_m=translation_error,
                          focal_px=focal.tolist(), expected_focal_px=float(expected_focal),
                          corrected_vs_live_world_max_error_m=conversion_error,
                          known_surface_depth_limit_m=1.0,
                          integer_pixel_surface_error_p99_m=integer_surface_p99,
                          center_pixel_surface_error_p99_m=center_surface_p99,
                          point_count=len(world), view=view.tolist(), projection=projection.tolist())
            if (rotation_error > 1e-5 or translation_error > 1e-5 or
                    np.abs(focal-expected_focal).max() > 1e-3 or conversion_error > 1e-4 or center_surface_p99 > .0001):
                stage("camera_geometry_failed", **checks)
                raise RuntimeError("Live camera/known geometry checks failed")
            np.savez_compressed(args.output / "geometry.npz", world_integer_pixels=world,
                                source_stored=stored, corrected=corrected, world_pixel_centers=center_world)
            stage("camera_geometry_pass", **checks)
        report["status"] = "passed"
        stage("completed")
    except Exception as exc:
        report["status"] = "failed"
        stage("exception", type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if sim is not None:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
