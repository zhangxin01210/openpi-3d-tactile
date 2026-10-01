#!/usr/bin/env python3
"""Single USB or Peg reset/step/sensor smoke in a private ContactWorld checkout.

Imports only the requested USB/Peg task (skips the unrelated all-task registry). Does not train
or collect perturbed demonstrations, and does not measure policy success.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', choices=('usb', 'peg'), default='usb')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--demo', type=Path, help='Run complete action replay from an exported episode NPZ')
    p.add_argument('--clear-reset-actuation', action='store_true')
    p.add_argument('--staged-reset', action='store_true')
    p.add_argument('--restore-dt', type=float)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--white-gel', action='store_true')
    p.add_argument('--white-mounts', action='store_true')
    p.add_argument('--socket-away', action='store_true')
    p.add_argument('--estimated-velocity', action='store_true')
    p.add_argument('--pose-feedback', action='store_true')
    p.add_argument('--hold-after', type=int, default=0, help='Zero-action hold steps after a demo replay')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'scope': f'single {args.task} smoke; no policy evaluation', 'stages': []}
    start = time.monotonic()

    def stage(name, **info):
        row = {'name': name, 'elapsed_s': time.monotonic()-start, **info}
        report['stages'].append(row)
        (args.output/'probe.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)

    env = None
    try:
        from isaacgym import gymapi
        import torch
        import numpy as np
        import hydra
        from omegaconf import OmegaConf
        code = args.source/'thirdparty/manifeel-isaacgymenvs'
        sys.path.insert(0, str(code))
        import isaacgymenvs
        tasks = types.ModuleType('isaacgymenvs.tasks')
        tasks.__path__ = [str(code/'isaacgymenvs/tasks')]
        sys.modules['isaacgymenvs.tasks'] = tasks
        isaacgymenvs.tasks = tasks
        if args.task == 'usb':
            from isaacgymenvs.tasks.tacsl.tacsl_task_usb import TacSLTaskUSB as TaskClass
        else:
            from isaacgymenvs.tasks.tacsl.tacsl_task_peg import TacSLTaskPeg as TaskClass
        stage('task_imported', torch=torch.__version__)
        # Mirror the official eval_planner.py Hydra composition layout.
        cfg_root = args.output/'hydra'
        cfg_root.mkdir(exist_ok=True)
        manifeel = args.source/'thirdparty/manifeel'
        for config in (manifeel/'manifeel/config').glob('*.yaml'):
            link = cfg_root/config.name
            if not link.exists(): link.symlink_to(config.resolve())
        for name, target in [('task', manifeel/'manifeel/config/task'), ('assets', manifeel/'assets')]:
            link = cfg_root/name
            if not link.exists(): link.symlink_to(target.resolve())
        with hydra.initialize_config_dir(config_dir=str(cfg_root.resolve()), version_base='1.1'):
            cfg = hydra.compose(config_name=f'isaacgym_config_{args.task}', overrides=['num_envs=1', 'headless=true'])
            cfg.capture_video = False
            cfg.force_render = False
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            (args.output/'task_config.json').write_text(json.dumps(task_cfg, indent=2)+'\n')
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            stage('creating_task')
            env = TaskClass(cfg=task_cfg, rl_device='cuda:0', sim_device='cuda:0',
                              graphics_device_id=0, headless=True,
                              virtual_screen_capture=False, force_render=False)
            stage('task_created')
            env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
            stage('reset_idx_passed')
            if args.demo:
                from replay_contactworld_episode import replay
                replay(env, args.demo, args.output, clear_actuation=args.clear_reset_actuation, staged_reset=args.staged_reset, restore_dt=args.restore_dt, start_frame=args.start_frame, white_gel=args.white_gel, white_mounts=args.white_mounts, socket_away=args.socket_away, estimated_velocity=args.estimated_velocity, pose_feedback=args.pose_feedback, hold_after=args.hold_after)
                report['status'] = 'completed_execution_not_validated_reproduction'
                stage('complete_episode_executed', caveat='See replay.json for trajectory agreement; not a success-rate evaluation')
                return
            snapshots, forces = {}, []
            for step in range(args.steps):
                obs, _, _, _ = env.step(torch.zeros((1, env.num_actions), device=env.device))
                obs = obs['obs']
                for key, value in obs.items():
                    if torch.is_tensor(value):
                        assert torch.isfinite(value).all(), key
                        if step == args.steps-1:
                            snapshots[key] = value.detach().cpu().numpy()
                forces.append({key: float(value.abs().max().item()) for key, value in obs.items()
                               if 'force_field' in key})
            stage('zero_action_steps_passed', steps=args.steps, force_maxima=forces)
            images = env.get_camera_image_tensors_dict()
            for name, value in images.items():
                snapshots['camera_'+name] = value.detach().cpu().numpy()
            for name, handle in env.camera_handles_list[0].items():
                snapshots['view_'+name] = np.array(env.gym.get_camera_view_matrix(env.sim, env.env_ptrs[0], handle))
                snapshots['projection_'+name] = np.array(env.gym.get_camera_proj_matrix(env.sim, env.env_ptrs[0], handle))
            np.savez_compressed(args.output/'sensors.npz', **snapshots)
            stats = {}
            for key, array in snapshots.items():
                finite = np.isfinite(array)
                stats[key] = {'shape': list(array.shape), 'finite_fraction': float(finite.mean()),
                              'min_finite': float(array[finite].min()) if finite.any() else None,
                              'max_finite': float(array[finite].max()) if finite.any() else None}
            assert 'front' in snapshots
            assert snapshots['front'].std() > 0
            assert any('force_field' in key for key in snapshots)
            assert any('depth' in key for key in images)
            report['status'] = 'passed'
            stage('sensors_saved', statistics=stats)
    except Exception as exc:
        report['status'] = 'failed'
        stage('exception', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
