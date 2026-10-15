#!/usr/bin/env python3
"""Resumable paired USB rollouts for the 13 bilateral V3 OpenPI checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess

import numpy as np

from contactworld_v3_eval_inputs import CONFIGS, obs_array, policy_input, snapshot, tactile_summary
from eval_contactworld_proprio_baseline import describe_pose, prepare_environment, video_frame


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def physical_state(env):
    keys = ("ee_pos", "ee_quat", "plug_pos", "plug_quat", "socket_pos_gt", "socket_quat",
            "dof_pos", "dof_vel")
    return {key: obs_array(env, key).astype(float).tolist() for key in keys}


def metrics(env):
    pair = env.contact_force_pairwise[
        0, env.plug_body_id_env, env.socket_body_id_env].detach().cpu().numpy()
    local = np.stack([obs_array(env, "tactile_force_field_" + side)
                      for side in ("left", "right")]).astype(np.float32)
    norms = np.linalg.norm(local, axis=-1)
    return {
        **describe_pose(env),
        "ee_plug_distance_mm": float(np.linalg.norm(
            obs_array(env, "ee_pos") - obs_array(env, "plug_pos")) * 1000),
        "gripper_width_mm": float(obs_array(env, "dof_pos")[-2:].sum() * 1000),
        "plug_socket_contact_force_xyz": pair.astype(float).tolist(),
        "tacff_left_active_fraction": float((norms[0] > 1e-4).mean()),
        "tacff_right_active_fraction": float((norms[1] > 1e-4).mean()),
        "tacff_left_max_norm": float(norms[0].max()),
        "tacff_right_max_norm": float(norms[1].max()),
        "tacff_left_channel_abs_mean": np.abs(local[0]).mean(axis=(0, 1)).tolist(),
        "tacff_right_channel_abs_mean": np.abs(local[1]).mean(axis=(0, 1)).tolist(),
    }, local


def save_index(root, summary):
    cards = []
    for result in summary["results"]:
        seed = result["seed"]
        title = ("seed %d · ever %s · final %s · first %s · min %.1f mm" % (
            seed, result["source_ever"], result["source_final"],
            result["first_success_step"], result["min_keypoint_error_mm"]))
        cards.append('<section><h2>%s</h2><video controls preload="metadata" '
                     'src="seed_%d/front_wrist.mp4"></video><p>'
                     '<a href="seed_%d/trajectory.json">Trajectory</a> · '
                     '<a href="seed_%d/inputs.npz">All sensor inputs</a></p></section>' % (
                         escape(title), seed, seed, seed))
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld V3</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto;background:#16191d;color:#eee}"
        "video{width:100%;max-width:768px}a{color:#9bd}section{border-top:1px solid #555;padding:1rem 0}</style>"
        "</head><body><h1>ContactWorld USB V3 · " + escape(summary["policy_config"]) + "</h1>"
        "<p>Identical seed schedule across models. Success is the original source keypoint threshold; "
        "contact, pose and videos are separate diagnostics. Raw paired force fields and fused clouds "
        "are archived even if this policy does not use them.</p>" + "".join(cards) + "</body></html>",
        encoding="utf-8")


def save_inputs(path, snapshots, chunks, step_fields):
    if not snapshots:
        raise ValueError("No inference snapshots captured")
    names = ("front", "wrist", "front_depth", "wrist_depth", "front_view", "front_projection",
             "wrist_view", "wrist_projection", "state", "cloud_xyz", "cloud_camera", "cloud_uv",
             "base_pose_world", "taxel_xyz_base", "force_base", "force_local")
    arrays = {key: np.stack([item[key] for _, item in snapshots]) for key in names}
    arrays.update(steps=np.asarray([step for step, _ in snapshots], np.int32),
                  action_chunks=np.stack(chunks),
                  force_local_post_step=np.stack(step_fields))
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--limit-this-run", type=int)
    parser.add_argument("--input-only", action="store_true",
                        help="Capture one reset's actual online V3 inputs without loading a policy")
    args = parser.parse_args()
    args.task = "usb"
    args.dual_tactile = True
    if args.trials < 1 or args.max_steps < 1 or not 1 <= args.replan_every <= 16:
        parser.error("Invalid trial/step/replan count")
    if args.limit_this_run is not None and args.limit_this_run < 1:
        parser.error("--limit-this-run must be positive")
    if not args.input_only and (args.socket is None or args.checkpoint is None or
                                not (args.checkpoint / "_CHECKPOINT_METADATA").is_file()):
        parser.error("A complete checkpoint and policy socket are required")
    if not args.source.is_dir():
        parser.error("Missing patched ContactWorld checkout: " + str(args.source))
    patched = args.source / ("thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/"
                             "tacsl_task_usb.py")
    bilateral_line = "force_keys = ['tactile_force_field_left', 'tactile_force_field_right']"
    if not patched.is_file() or bilateral_line not in patched.read_text():
        parser.error("V3 requires the isolated bilateral TacFF simulator patch")
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacgym import gymapi  # noqa: F401
    # Importing Isaac Gym must precede importing torch.
    import torch
    torch.set_num_threads(4)
    checkpoint_hash = (hashlib.sha256((args.checkpoint / "_CHECKPOINT_METADATA").read_bytes()).hexdigest()
                       if args.checkpoint else None)
    summary = {
        "policy_config": args.config, "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_metadata_sha256": checkpoint_hash, "seed_base": args.seed_base,
        "trials_requested": args.trials, "max_steps": args.max_steps,
        "replan_every": args.replan_every, "source": str(args.source.resolve()),
        "source_success_definition": "env._check_success(): source plug/socket keypoint threshold",
        "primary_metric": "source_ever", "results": [],
        "input_contract": (
            "V3 bilateral LeRobot: front+wrist RGB; fused base 4096 XYZ; "
            "ordered left/right 140-taxel local/base force"
        ),
    }
    summary_path = args.output / "summary.json"
    if summary_path.exists() and not args.input_only:
        previous = json.loads(summary_path.read_text())
        for key in ("policy_config", "checkpoint", "checkpoint_metadata_sha256", "seed_base",
                    "trials_requested", "max_steps", "replan_every", "source", "input_contract"):
            if previous.get(key) != summary[key]:
                raise ValueError("Existing summary differs on " + key)
        summary = previous
    completed = len(summary["results"])
    if [r["seed"] for r in summary["results"]] != list(range(args.seed_base, args.seed_base + completed)):
        raise ValueError("Existing results are not a contiguous seed prefix")
    stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
    # The runner invokes this script once per seed, since Isaac Gym could not
    # safely destroy and recreate repeated simulator instances in one process.
    if not args.input_only and stop - completed != 1:
        parser.error("Invoke with --limit-this-run 1 through the V3 runner")
    seed = args.seed_base + completed
    args.seed_base = seed
    env = None
    try:
        env = prepare_environment(args, torch, np)
        dt = float(env.cfg_base.sim.dt) * int(env.cfg_task.env.controlFrequencyInv)
        summary["action_dt_s"] = dt
        summary["video_fps"] = 1 / dt
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
        env.refresh_all_tensors()
        env.compute_observations()
        initial, _ = metrics(env)
        initial_state = physical_state(env)
        initial_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
        first = snapshot(env)
        if args.input_only:
            request = policy_input(first, args.config)
            np.savez_compressed(args.output / "online_input_audit.npz",
                                **{key: first[key] for key in (
                                    "front", "wrist", "front_depth", "wrist_depth", "front_view",
                                    "front_projection", "wrist_view", "wrist_projection", "state",
                                    "cloud_xyz", "cloud_camera",
                                    "cloud_uv", "base_pose_world", "taxel_xyz_base", "force_base", "force_local")})
            write_json(args.output / "online_input_audit.json", {
                "seed": seed, "config": args.config, "initial_state_sha256": initial_hash,
                "cloud_counts": first["cloud_counts"], "tactile_summary": tactile_summary(first),
                "policy_visual": request.get("spatial", {}).get("visual") is not None,
                "policy_tactile": request.get("spatial", {}).get("tactile") is not None,
            })
            print("INPUT_AUDIT_PASS", args.output / "online_input_audit.npz", flush=True)
            return
        trial_dir = args.output / ("seed_%d" % seed)
        trial_dir.mkdir(exist_ok=True)
        video = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "512x256", "-r", "%.6f" % (1 / dt), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(trial_dir / "front_wrist.mp4")],
            stdin=subprocess.PIPE)
        rows, snapshots, chunks, step_fields = [], [], [], []
        first_success = None
        streak = max_streak = 0
        ever = bool(initial["source_success"])
        reason = "step_budget"
        try:
            video.stdin.write(video_frame(env).tobytes())
            with Client(str(args.socket), family="AF_UNIX", authkey=b"contactworld-rgb") as connection:
                for step in range(args.max_steps):
                    if step % args.replan_every == 0:
                        snap = first if step == 0 else snapshot(env)
                        connection.send(policy_input(snap, args.config))
                        response = connection.recv()
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        actions = np.asarray(response["actions"], dtype=np.float32)
                        if actions.shape != (16, 6) or not np.isfinite(actions).all():
                            raise ValueError("Invalid policy action chunk")
                        snapshots.append((step, snap))
                        chunks.append(actions.copy())
                        latency = float(response["infer_ms"])
                        tactile = tactile_summary(snap)
                    action = actions[step % args.replan_every]
                    env.step(torch.as_tensor(action[None], device=env.device))
                    current, field = metrics(env)
                    step_fields.append(field)
                    ever |= bool(current["source_success"])
                    if current["source_success"]:
                        streak += 1
                        if first_success is None:
                            first_success = step
                    else:
                        streak = 0
                    max_streak = max(max_streak, streak)
                    rows.append({
                        "step": step, "time_s": (step + 1) * dt,
                        "action_raw": action.tolist(),
                        "action_sim_clipped": np.clip(action, -env.clip_actions, env.clip_actions).tolist(),
                        "replan": step % args.replan_every == 0,
                        "inference_ms": latency if step % args.replan_every == 0 else None,
                        "tactile_pre_action": tactile if step % args.replan_every == 0 else None,
                        **current, **physical_state(env),
                    })
                    video.stdin.write(video_frame(env).tobytes())
                    if not np.isfinite(current["keypoint_error_mm"]):
                        reason = "nonfinite_state"
                        break
        finally:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        final, _ = metrics(env)
        save_inputs(trial_dir / "inputs.npz", snapshots, chunks, step_fields)
        result = {
            "seed": seed, "steps": len(rows), "initial": initial, "final": final,
            "initial_state_sha256": initial_hash, "initial_physical_state": initial_state,
            "source_ever": ever, "source_final": bool(final["source_success"]),
            "first_success_step": first_success, "max_consecutive_source_steps": max_streak,
            "min_keypoint_error_mm": min([initial["keypoint_error_mm"]] +
                                         [r["keypoint_error_mm"] for r in rows]),
            "stop_reason": reason, "video": str(trial_dir / "front_wrist.mp4"),
        }
        write_json(trial_dir / "trajectory.json", {
            "result": result, "cloud_counts_initial": first["cloud_counts"], "rows": rows,
        })
        summary["results"].append(result)
        write_json(summary_path, summary)
        save_index(args.output, summary)
        print("trial %d/%d %s seed=%d ever=%s final=%s min_mm=%.1f" % (
            completed + 1, args.trials, args.config, seed, ever,
            final["source_success"], result["min_keypoint_error_mm"]), flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
