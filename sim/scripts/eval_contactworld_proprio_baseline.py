#!/usr/bin/env python3
"""Closed-loop random-reset evaluation of a q/dq/time-only ContactWorld policy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation


def describe_pose(env):
    obs = env.obs_dict
    plug_pos = obs["plug_pos"][0].detach().cpu().numpy()
    socket_pos = obs["socket_pos_gt"][0].detach().cpu().numpy()
    plug_quat = obs["plug_quat"][0].detach().cpu().numpy()
    socket_quat = obs["socket_quat"][0].detach().cpu().numpy()
    plug_axis = Rotation.from_quat(plug_quat).apply([0., 0., 1.])
    socket_axis = Rotation.from_quat(socket_quat).apply([0., 0., 1.])
    axis_error = np.rad2deg(np.arccos(np.clip(abs(np.dot(plug_axis, socket_axis)), 0, 1)))
    keypoint_error = float((env.keypoints_plug[0] - env.keypoints_socket[0]).norm(dim=-1).mean().item())
    delta = plug_pos-socket_pos
    return {"source_success": bool(env._check_success()[0].item()),
            "keypoint_error_mm": keypoint_error*1000,
            "plug_socket_xyz_error_mm": float(np.linalg.norm(delta)*1000),
            "plug_socket_xy_error_mm": float(np.linalg.norm(delta[:2])*1000),
            "plug_socket_z_mm": float(delta[2]*1000),
            "unsigned_axis_error_deg": float(axis_error),
            "socket_xy_m": socket_pos[:2].tolist()}


def strict_pose(pose):
    # Diagnostic only: source criterion plus tighter root alignment and depth.
    return (pose["source_success"] and pose["plug_socket_xyz_error_mm"] < 2.
            and pose["plug_socket_z_mm"] < 5.
            and pose["unsigned_axis_error_deg"] < 5.)


def video_frame(env):
    front = env.obs_dict["front"][0].detach().cpu().numpy()
    wrist = env.obs_dict["wrist"][0].detach().cpu().numpy()
    return np.uint8(np.clip(np.concatenate((front, wrist), axis=1)*255, 0, 255))


def prepare_environment(args, torch, np):
    from isaacgym import gymapi
    import hydra
    from omegaconf import OmegaConf
    code = args.source / "thirdparty/manifeel-isaacgymenvs"
    sys.path.insert(0, str(code))
    import isaacgymenvs
    tasks = types.ModuleType("isaacgymenvs.tasks")
    tasks.__path__ = [str(code / "isaacgymenvs/tasks")]
    sys.modules["isaacgymenvs.tasks"] = tasks
    isaacgymenvs.tasks = tasks
    if args.task == "usb":
        from isaacgymenvs.tasks.tacsl.tacsl_task_usb import TacSLTaskUSB as TaskClass
    else:
        from isaacgymenvs.tasks.tacsl.tacsl_task_peg import TacSLTaskPeg as TaskClass
    cfg_root = args.output / "hydra"
    cfg_root.mkdir(parents=True, exist_ok=True)
    manifeel = args.source / "thirdparty/manifeel"
    for config in (manifeel / "manifeel/config").glob("*.yaml"):
        link = cfg_root/config.name
        if not link.exists():
            link.symlink_to(config.resolve())
    for name, target in (("task", manifeel / "manifeel/config/task"),
                         ("assets", manifeel / "assets")):
        link = cfg_root/name
        if not link.exists():
            link.symlink_to(target.resolve())
    np.random.seed(args.seed_base)
    torch.manual_seed(args.seed_base)
    with hydra.initialize_config_dir(config_dir=str(cfg_root.resolve()), version_base="1.1"):
        cfg = hydra.compose(config_name="isaacgym_config_"+args.task,
                            overrides=["num_envs=1", "headless=true"])
        cfg.capture_video = False
        cfg.force_render = False
        task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
        (args.output / "task_config.json").write_text(json.dumps(task_cfg, indent=2)+"\n")
        env = TaskClass(cfg=task_cfg, rl_device="cuda:0", sim_device="cuda:0",
                        graphics_device_id=0, headless=True,
                        virtual_screen_capture=False, force_render=False)
    for name in ("elastomer_left", "elastomer_right", "mount_left", "mount_right"):
        body = env.gym.find_actor_rigid_body_index(env.env_ptrs[0], env.actor_handles["franka"],
                                                    name, gymapi.DOMAIN_ACTOR)
        env.gym.set_rigid_body_color(env.env_ptrs[0], env.actor_handles["franka"], body,
                                     gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", choices=("usb", "peg"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--video-first", type=int, default=1)
    parser.add_argument("--limit-this-run", type=int, help="Run at most this many new seeds, then checkpoint")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacgym import gymapi  # must precede torch import
    import torch
    from contactworld_proprio_policy import load_policy, act
    torch.set_num_threads(4)
    model, bundle = load_policy(args.policy, "cuda:0")
    expected_task = "insertion_" + args.task
    if bundle["task"] != expected_task:
        raise ValueError("Policy task differs from simulator task")
    policy_sha = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    summary = {"scope": "Random-reset closed-loop q/dq/time baseline; no demonstration reference or goal inputs",
               "task": args.task, "policy": str(args.policy), "policy_sha256": policy_sha,
               "seed_base": args.seed_base, "trials_requested": args.trials,
               "max_steps": args.max_steps,
               "strict_pose_definition": "source proximity plus plug/socket root <2mm, root z <5mm, unsigned axis <5deg, then 10 zero-action hold steps; still a proxy, not independent collision verification",
               "results": []}
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if (saved.get("policy_sha256") != policy_sha or saved.get("task") != args.task or
            saved.get("seed_base") != args.seed_base or saved.get("trials_requested") != args.trials or
            saved.get("max_steps") != args.max_steps):
            raise ValueError("Existing evaluation summary uses a different configuration")
        summary = saved
    completed = len(summary["results"])
    if [r["seed"] for r in summary["results"]] != list(range(args.seed_base, args.seed_base+completed)):
        raise ValueError("Existing seed prefix is not contiguous")
    stop_index = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
    env = None
    try:
        env = prepare_environment(args, torch, np)
        for trial_index in range(completed, stop_index):
            seed = args.seed_base + trial_index
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
            env.refresh_all_tensors()
            env.compute_observations()
            initial = describe_pose(env)
            trial_dir = args.output / f"seed_{seed}"
            trial_dir.mkdir(exist_ok=True)
            video = None
            if trial_index < args.video_first:
                video = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", "512x256", "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     str(trial_dir / "front_wrist.mp4")], stdin=subprocess.PIPE)
            history = []
            source_ever = initial["source_success"]
            strict_streak = 0
            held_strict = False
            stop_reason = "step_budget"
            try:
                for step in range(args.max_steps):
                    q = env.obs_dict["dof_pos"][0]
                    dq = env.obs_dict["dof_vel"][0]
                    action = act(model, bundle, q, dq, step)
                    if not torch.isfinite(action).all():
                        stop_reason = "nonfinite_action"
                        break
                    env.step(action[None])
                    pose = describe_pose(env)
                    source_ever |= pose["source_success"]
                    strict_streak = strict_streak+1 if strict_pose(pose) else 0
                    history.append({"step": step, **pose, "action": action.detach().cpu().tolist()})
                    if video is not None:
                        video.stdin.write(video_frame(env).tobytes())
                    if strict_streak >= 5:
                        hold_poses = []
                        for _ in range(10):
                            env.step(torch.zeros((1, 6), device=env.device))
                            held_pose = describe_pose(env)
                            hold_poses.append(held_pose)
                            if video is not None:
                                video.stdin.write(video_frame(env).tobytes())
                        held_strict = all(strict_pose(p) for p in hold_poses)
                        history[-1]["hold_poses"] = hold_poses
                        if held_strict:
                            stop_reason = "held_strict_pose"
                            break
                        strict_streak = 0
                    if not np.isfinite(pose["keypoint_error_mm"]):
                        stop_reason = "nonfinite_state"
                        break
            finally:
                if video is not None:
                    video.stdin.close()
                    if video.wait() != 0:
                        raise RuntimeError("ffmpeg failed")
            final = describe_pose(env)
            result = {"seed": seed, "initial": initial, "final": final,
                      "steps": len(history), "source_ever": bool(source_ever),
                      "held_strict_pose": bool(held_strict), "stop_reason": stop_reason,
                      "trajectory": str(trial_dir / "trajectory.json"),
                      "video": str(trial_dir / "front_wrist.mp4") if video is not None else None}
            (trial_dir / "trajectory.json").write_text(json.dumps({"result": result, "rows": history}, indent=2)+"\n")
            summary["results"].append(result)
            summary_path.write_text(json.dumps(summary, indent=2)+"\n")
            print(args.task, seed, "steps", len(history), "ever", source_ever,
                  "final", final["source_success"], "held", held_strict,
                  "error_mm", round(final["keypoint_error_mm"], 2), flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
