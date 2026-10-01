#!/usr/bin/env python3
"""Validate Torch CUDA, cuDNN, Gym GPU physics tensors and camera tensors.

This is a native box probe, not a USB/TacSL task success claim.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'stages': [], 'scope': 'native box GPU interoperability'}
    start = time.monotonic()

    def stage(name, **details):
        row = {'name': name, 'elapsed_s': time.monotonic()-start, **details}
        report['stages'].append(row)
        (args.output/'probe.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    gym = sim = None
    try:
        from isaacgym import gymapi
        import torch
        import numpy as np
        stage('imports', torch=torch.__version__, cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(), arches=torch.cuda.get_arch_list(),
              gpu=torch.cuda.get_device_name())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        x = torch.arange(9, device='cuda', dtype=torch.float32).reshape(3, 3)
        assert torch.allclose((torch.sin(x)+x*x).cpu(), torch.sin(x.cpu())+x.cpu()**2)
        a = x@x.T + torch.eye(3, device='cuda')
        # NumPy provides an independent CPU reference; this minimal Torch build
        # does not include CPU LAPACK.
        np.testing.assert_allclose(torch.linalg.inv(a).cpu().numpy(),
                                   np.linalg.inv(a.cpu().numpy()), atol=1e-4, rtol=1e-4)
        torch.manual_seed(0)
        inp, kernel = torch.randn(1, 3, 16, 16), torch.randn(4, 3, 3, 3)
        expected = torch.nn.functional.conv2d(inp, kernel)
        actual = torch.nn.functional.conv2d(inp.cuda(), kernel.cuda()).cpu()
        assert torch.allclose(actual, expected, atol=2e-4, rtol=1e-4)
        stage('torch_ops_and_convolution_passed')
        from isaacgym import gymtorch
        stage('gymtorch_import_passed')
        gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt = 1/60
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        params.use_gpu_pipeline = True
        params.physx.use_gpu = True
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
        assert sim is not None
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
        asset = gym.create_box(sim, .1, .1, .1, gymapi.AssetOptions())
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, .4)
        gym.create_actor(env, asset, pose, 'box', 0, 0)
        props = gymapi.CameraProperties()
        props.width = props.height = 128
        props.enable_tensors = True
        cam = gym.create_camera_sensor(env, props)
        gym.set_camera_location(cam, env, gymapi.Vec3(.8, .8, .7), gymapi.Vec3(0, 0, .05))
        gym.prepare_sim(sim)
        state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        assert state.is_cuda
        stage('gpu_root_state_acquired', shape=list(state.shape), device=str(state.device))
        for _ in range(90):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        assert torch.isfinite(state).all()
        assert .04 < state[0, 2].item() < .08
        stage('gpu_physics_passed', box_z_m=state[0, 2].item())
        color = gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_COLOR))
        depth = gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(sim, env, cam, gymapi.IMAGE_DEPTH))
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)
        try:
            rgb, d = color.clone().cpu().numpy(), depth.clone().cpu().numpy()
        finally:
            gym.end_access_image_tensors(sim)
        assert color.is_cuda and depth.is_cuda
        assert rgb[..., :3].std() > 0
        assert (np.isfinite(d) & (d < 0)).any()
        np.savez_compressed(args.output/'camera.npz', rgba=rgb, depth=d)
        stage('gpu_camera_tensors_passed', rgb_shape=list(rgb.shape), depth_shape=list(d.shape))
        report['status'] = 'passed'
        stage('completed')
    except Exception as exc:
        report['status'] = 'failed'
        stage('exception', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if gym is not None and sim is not None:
            gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
