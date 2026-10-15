#!/usr/bin/env python3
"""Check Isaac Lab tiled RGB/depth rendering independently of UniVTAC/UIPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unproject_univtac_depth import unproject_opengl, write_binary_ply

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from pxr import Gf, UsdGeom


def main() -> None:
    result = {"pass": False, "steps": 0}
    camera = None
    sim = None
    try:
        stage_utils.create_new_stage()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/Ground", ground)
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)
        cube = prim_utils.create_prim("/World/Box", "Cube", translation=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3))
        UsdGeom.Cube(cube).CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        stage_utils.update_stage()
        cfg = TiledCameraCfg(
            prim_path="/World/Camera",
            width=320,
            height=240,
            update_period=0,
            data_types=["rgb", "depth"],
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        )
        camera = TiledCamera(cfg)
        sim.reset()
        for step in range(15):
            sim.step()
            camera.update(1 / 120)
            result["steps"] = step + 1
        rgb = camera.data.output["rgb"]
        depth = camera.data.output["depth"]
        intrinsic = camera.data.intrinsic_matrices
        pose_w_opengl = torch.cat((camera.data.pos_w, camera.data.quat_w_opengl), dim=-1)[0]
        finite = torch.isfinite(depth)
        points, pixel_indices = unproject_opengl(
            depth[0].cpu().numpy(), intrinsic[0].cpu().numpy(), pose_w_opengl.cpu().numpy(), stride=2
        )
        center_index = (240 // 2) * 320 + (320 // 2)
        center_pos = np.flatnonzero(pixel_indices == center_index)
        assert center_pos.size == 1
        center_world = points[center_pos[0]]
        result.update({
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "rgb_mean": float(rgb.float().mean()),
            "depth_finite_fraction": float(finite.float().mean()),
            "depth_finite_min": float(depth[finite].min()) if finite.any() else None,
            "depth_finite_max": float(depth[finite].max()) if finite.any() else None,
            "intrinsic": intrinsic[0].cpu().tolist(),
            "pose_w_opengl": pose_w_opengl.cpu().tolist(),
            "pointcloud_count_stride2": int(len(points)),
            "center_world_xyz_m": center_world.tolist(),
        })
        assert tuple(rgb.shape) == (1, 240, 320, 3)
        assert tuple(depth.shape) == (1, 240, 320, 1)
        assert result["rgb_mean"] > 1.0
        assert result["depth_finite_fraction"] > 0.01
        assert abs(float(center_world[2]) - 0.8) < 0.05, center_world
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(rgb[0].cpu().numpy()).save(out / "rgb.png")
        np.save(out / "depth.npy", depth[0].cpu().numpy())
        rgb_sample = rgb[0].cpu().numpy().reshape(-1, 3)[pixel_indices]
        write_binary_ply(out / "cloud_stride2.ply", points, rgb_sample)
        result["pass"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if camera is not None:
            del camera
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close()


if __name__ == "__main__":
    main()
