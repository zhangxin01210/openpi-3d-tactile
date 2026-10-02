#!/usr/bin/env python3
"""Paired full-task USB rollouts for ContactWorld OpenPI modality checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
from html import escape
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess

from eval_contactworld_proprio_baseline import describe_pose
from eval_contactworld_proprio_baseline import prepare_environment
from eval_contactworld_proprio_baseline import video_frame
import numpy as np

CONFIGS = {
    "pi0_cw_usb_01_rgb": (False, False),
    "pi0_cw_usb_02_rgb_pc": (True, False),
    "pi0_cw_usb_03_rgb_ff": (False, True),
    "pi0_cw_usb_04_rgb_pc_ff": (True, True),
    "pi0_cw_usb_05_ff_summary": (True, True),
    "pi0_cw_usb_06_ff_ee3d_proxy": (True, True),
    "pi0_cw_usb_07_tacrgb": (True, False),
    "pi0_cw_usb_08_tacdepth": (True, False),
    "pi0_cw_usb_09_prefix": (True, True),
    "pi0_cw_usb_10_split": (True, True),
    "pi0_cw_usb_11_pcnoise": (True, True),
}
GEOMETRY = Path(__file__).resolve().parents[1] / "reports/2026-10-01/contactworld/runtime_geometry.json"


def obs_array(env, key):
    return env.obs_dict[key][0].detach().cpu().numpy().copy()


def snapshot(env, rotation, translation, *, config=None):
    cloud = obs_array(env, "pointcloud")[:, :3].astype(np.float32) @ rotation.T + translation
    force = obs_array(env, "tactile_force_field_right").astype(np.float32)
    state = np.concatenate((obs_array(env, "dof_pos"), obs_array(env, "dof_vel"))).astype(np.float32)
    front = np.rint(np.clip(obs_array(env, "front"), 0, 1) * 255).astype(np.uint8)
    wrist = np.rint(np.clip(obs_array(env, "wrist"), 0, 1) * 255).astype(np.uint8)
    if (cloud.shape != (1024, 3) or force.shape != (10, 14, 3) or state.shape != (18,) or
        not all(np.isfinite(a).all() for a in (cloud, force, state))):
        raise ValueError("Invalid simulator observation")
    result = {"front": front, "wrist": wrist, "cloud_xyz": cloud, "force_grid": force,
              "state": state, "ee_pose": np.concatenate((obs_array(env, "ee_pos"),
                                                           obs_array(env, "ee_quat"))).astype(np.float32)}
    if config == "pi0_cw_usb_07_tacrgb":
        result["tactile_rgb"] = np.rint(np.clip(obs_array(env, "tactile_rgb_right"), 0, 1) * 255).astype(np.uint8)
    if config == "pi0_cw_usb_08_tacdepth":
        result["tactile_depth"] = obs_array(env, "tactile_depth_right").astype(np.float32)
    return result


def policy_input(snap, *, cloud, force, config=None):
    request = {"observation/front_rgb": snap["front"], "observation/wrist_rgb": snap["wrist"],
               "observation/state": snap["state"]}
    visual = tactile = None
    if cloud:
        visual = {"xyz_m": snap["cloud_xyz"], "rgb": np.zeros((1024, 3), np.uint8),
                  "rgb_valid": np.zeros(1024, bool), "point_mask": np.ones(1024, bool)}
    if force:
        yy, xx = np.meshgrid(np.linspace(-1, 1, 10, dtype=np.float32),
                             np.linspace(-1, 1, 14, dtype=np.float32), indexing="ij")
        xyz = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(140, 3)
        values = snap["force_grid"].reshape(140, 3)
        if config == "pi0_cw_usb_06_ff_ee3d_proxy":
            pose = snap["ee_pose"]
            x, y, z, w = pose[3:]
            rotation = np.array([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ], dtype=np.float32)
            xyz = (xyz * 0.01) @ rotation.T + pose[:3]
            values = values @ rotation.T
        tactile = {"xyz_m": xyz, "force": values, "force_norm": np.linalg.norm(values, axis=-1),
                   "finger_id": np.zeros(140, np.int32), "taxel_id": np.arange(140, dtype=np.int32),
                   "point_mask": np.ones(140, bool)}
    if cloud or force:
        request["spatial"] = {"visual": visual, "tactile": tactile}
    if config == "pi0_cw_usb_07_tacrgb":
        request["observation/tactile_rgb"] = snap["tactile_rgb"]
    elif config == "pi0_cw_usb_08_tacdepth":
        request["observation/tactile_depth"] = snap["tactile_depth"]
    return request


def physical_state(env):
    keys = ("ee_pos", "ee_quat", "plug_pos", "plug_quat", "socket_pos_gt", "socket_quat", "dof_pos", "dof_vel")
    return {key: obs_array(env, key).astype(float).tolist() for key in keys}


def metrics(env):
    force = obs_array(env, "tactile_force_field_right")
    mag = np.linalg.norm(force, axis=-1)
    contact = env.contact_force_pairwise[0, env.plug_body_id_env, env.socket_body_id_env].detach().cpu().numpy()
    return {**describe_pose(env),
            "ee_plug_distance_mm": float(np.linalg.norm(obs_array(env, "ee_pos") - obs_array(env, "plug_pos")) * 1000),
            "gripper_width_mm": float(obs_array(env, "dof_pos")[-2:].sum() * 1000),
            "plug_socket_contact_force_xyz": contact.astype(float).tolist(),
            "tacff_mean_norm": float(mag.mean()), "tacff_max_norm": float(mag.max()),
            "tacff_active_fraction": float((mag > 1e-4).mean())}


def save_index(root, summary):
    cards = []
    for result in summary["results"]:
        seed = result["seed"]
        title = (f"seed {seed} · ever {result['source_ever']} · final {result['source_final']} · "
                 f"first {result['first_success_step']} · min error {result['min_keypoint_error_mm']:.1f} mm")
        cards.append(f'<section><h2>{escape(title)}</h2><video controls preload="metadata" '
                     f'src="seed_{seed}/front_wrist.mp4"></video><p>'
                     f'<a href="seed_{seed}/trajectory.json">Trajectory</a> · '
                     f'<a href="seed_{seed}/inputs.npz">Sensor and policy inputs</a></p></section>')
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld OpenPI USB</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto;background:#16191d;color:#eee}"
        "video{width:100%;max-width:768px}a{color:#9bd}section{border-top:1px solid #555;padding:1rem 0}</style>"
        "</head><body><h1>ContactWorld USB · " + escape(summary["policy_config"]) + "</h1>"
        "<p>Random resets, paired by seed. Source success tests keypoint proximity; review contact, pose, and video "
        "before calling it physical insertion. All sensors are saved for diagnosis, but each policy receives only "
        "its configured modalities.</p>" + "".join(cards) + "</body></html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--limit-this-run", type=int, help="Run a bounded batch and leave resumable results")
    parser.add_argument("--fresh-env-per-trial", action="store_true",
                        help="Destroy/recreate the simulator for each seed to remove rollout-history effects")
    args = parser.parse_args()
    if args.trials < 1 or args.max_steps < 1 or not 1 <= args.replan_every <= 16:
        parser.error("Invalid trial, step or replan count")
    if not (args.checkpoint / "_CHECKPOINT_METADATA").is_file():
        parser.error("Incomplete Orbax checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacgym import gymapi  # noqa: F401  Must precede torch import.
    import torch

    torch.set_num_threads(4)
    geometry = json.loads(GEOMETRY.read_text())
    if geometry["status"] != "passed":
        raise ValueError("Cloud geometry audit did not pass")
    rotation = np.asarray(geometry["A"], dtype=np.float32)
    translation = np.asarray(geometry["t"], dtype=np.float32)
    use_cloud, use_force = CONFIGS[args.config]
    cloud_noise_m = 0.002 if args.config == "pi0_cw_usb_11_pcnoise" else 0.0
    summary = {"policy_config": args.config, "checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_metadata_sha256": hashlib.sha256((args.checkpoint / "_CHECKPOINT_METADATA").read_bytes()).hexdigest(),
               "seed_base": args.seed_base, "trials_requested": args.trials,
               "replan_every": args.replan_every, "max_steps": args.max_steps,
               "cloud_noise_m": cloud_noise_m,
               "fresh_env_per_trial": args.fresh_env_per_trial,
               "source_success_definition": "env._check_success(): mean plug/socket keypoint error below close_error_thresh",
               "primary_metric": "source_ever", "secondary_metrics": ["source_final", "max_consecutive_source_steps"],
               "results": []}
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        for key in ("policy_config", "checkpoint", "checkpoint_metadata_sha256", "seed_base",
                    "trials_requested", "replan_every", "max_steps", "fresh_env_per_trial", "cloud_noise_m"):
            if previous[key] != summary[key]:
                raise ValueError(f"Existing summary differs on {key}")
        summary = previous
    completed = len(summary["results"])
    if [r["seed"] for r in summary["results"]] != list(range(args.seed_base, args.seed_base + completed)):
        raise ValueError("Existing results are not a contiguous seed prefix")
    stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
    args.task = "usb"
    env = None
    try:
        if not args.fresh_env_per_trial:
            env = prepare_environment(args, torch, np)
        for trial in range(completed, stop):
            seed = args.seed_base + trial
            if args.fresh_env_per_trial:
                init_args = copy.copy(args)
                init_args.seed_base = seed
                env = prepare_environment(init_args, torch, np)
            dt = float(env.cfg_base.sim.dt) * int(env.cfg_task.env.controlFrequencyInv)
            summary["action_dt_s"] = dt
            summary["video_fps"] = 1 / dt
            np.random.seed(seed)
            torch.manual_seed(seed)
            cloud_noise_rng = np.random.default_rng(314159 + seed)
            env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
            env.refresh_all_tensors()
            env.compute_observations()
            initial = metrics(env)
            initial_state = physical_state(env)
            initial_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
            trial_dir = args.output / f"seed_{seed}"
            trial_dir.mkdir(exist_ok=True)
            video = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "512x256", "-r", f"{1 / dt:.6f}", "-i", "-", "-an", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(trial_dir / "front_wrist.mp4")],
                stdin=subprocess.PIPE)
            rows = []
            sensors = []
            chunks = []
            first_success = None
            streak = max_streak = 0
            ever = bool(initial["source_success"])
            reason = "step_budget"
            try:
                video.stdin.write(video_frame(env).tobytes())
                with Client(str(args.socket), family="AF_UNIX", authkey=b"contactworld-rgb") as connection:
                    for step in range(args.max_steps):
                        if step % args.replan_every == 0:
                            snap = snapshot(env, rotation, translation, config=args.config)
                            if cloud_noise_m:
                                snap["cloud_xyz_clean"] = snap["cloud_xyz"].copy()
                                snap["cloud_xyz"] += cloud_noise_rng.normal(
                                    0.0, cloud_noise_m, snap["cloud_xyz"].shape).astype(np.float32)
                            connection.send(policy_input(snap, cloud=use_cloud, force=use_force,
                                                         config=args.config))
                            response = connection.recv()
                            if "error" in response:
                                raise RuntimeError(response["error"])
                            actions = np.asarray(response["actions"], dtype=np.float32)
                            if actions.shape != (16, 6) or not np.isfinite(actions).all():
                                raise ValueError("Invalid policy action chunk")
                            sensors.append((step, snap))
                            chunks.append(actions.copy())
                            latency = float(response["infer_ms"])
                        action = actions[step % args.replan_every]
                        env.step(torch.as_tensor(action[None], device=env.device))
                        current = metrics(env)
                        ever |= bool(current["source_success"])
                        if current["source_success"]:
                            streak += 1
                            if first_success is None:
                                first_success = step
                        else:
                            streak = 0
                        max_streak = max(max_streak, streak)
                        rows.append({"step": step, "time_s": (step + 1) * dt,
                                     "action_raw": action.tolist(),
                                     "action_sim_clipped": np.clip(action, -env.clip_actions, env.clip_actions).tolist(),
                                     "replan": step % args.replan_every == 0,
                                     "inference_ms": latency if step % args.replan_every == 0 else None,
                                     **current, **physical_state(env)})
                        video.stdin.write(video_frame(env).tobytes())
                        if not np.isfinite(current["keypoint_error_mm"]):
                            reason = "nonfinite_state"
                            break
            finally:
                video.stdin.close()
                if video.wait() != 0:
                    raise RuntimeError("ffmpeg failed")
            final = metrics(env)
            extra_inputs = {}
            for key in ("tactile_rgb", "tactile_depth", "cloud_xyz_clean"):
                if key in sensors[0][1]:
                    extra_inputs[key] = np.stack([s[key] for _, s in sensors])
            np.savez_compressed(trial_dir / "inputs.npz",
                                steps=np.array([step for step, _ in sensors], np.int32),
                                front=np.stack([s["front"] for _, s in sensors]),
                                wrist=np.stack([s["wrist"] for _, s in sensors]),
                                state=np.stack([s["state"] for _, s in sensors]),
                                cloud_xyz=np.stack([s["cloud_xyz"] for _, s in sensors]),
                                force_grid=np.stack([s["force_grid"] for _, s in sensors]),
                                ee_pose=np.stack([s["ee_pose"] for _, s in sensors]),
                                action_chunks=np.stack(chunks), **extra_inputs)
            result = {"seed": seed, "steps": len(rows), "initial": initial, "final": final,
                      "initial_state_sha256": initial_hash, "initial_physical_state": initial_state,
                      "source_ever": ever, "source_final": bool(final["source_success"]),
                      "first_success_step": first_success, "max_consecutive_source_steps": max_streak,
                      "min_keypoint_error_mm": min([initial["keypoint_error_mm"]] +
                                               [r["keypoint_error_mm"] for r in rows]),
                      "stop_reason": reason, "video": str(trial_dir / "front_wrist.mp4")}
            (trial_dir / "trajectory.json").write_text(json.dumps({"result": result, "rows": rows}, indent=2) + "\n")
            summary["results"].append(result)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            save_index(args.output, summary)
            print(f"trial {trial + 1}/{args.trials} {args.config} seed={seed} "
                  f"ever={ever} final={final['source_success']} first={first_success} "
                  f"min_mm={result['min_keypoint_error_mm']:.1f}", flush=True)
            if args.fresh_env_per_trial:
                env.gym.destroy_sim(env.sim)
                env = None
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
