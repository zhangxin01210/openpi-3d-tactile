#!/usr/bin/env python3
"""Closed-loop ContactWorld USB RGB rollout with a separate OpenPI policy process."""

from __future__ import annotations

import argparse
from html import escape
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess

from eval_contactworld_proprio_baseline import describe_pose
from eval_contactworld_proprio_baseline import prepare_environment
from eval_contactworld_proprio_baseline import strict_pose
from eval_contactworld_proprio_baseline import video_frame
import numpy as np


def observation(env):
    def image(name):
        value = env.obs_dict[name][0].detach().cpu().numpy()
        return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)

    state = np.concatenate((env.obs_dict["dof_pos"][0].detach().cpu().numpy(),
                            env.obs_dict["dof_vel"][0].detach().cpu().numpy())).astype(np.float32)
    if state.shape != (18,) or not np.isfinite(state).all():
        raise ValueError("Invalid 18D joint state")
    return {"observation/front_rgb": image("front"),
            "observation/wrist_rgb": image("wrist"),
            "observation/state": state}


def write_index(output, summary):
    cards = []
    for row in summary["results"]:
        seed = row["seed"]
        label = " · ".join([
            f"seed {seed}: {row['steps']} steps",
            f"source criterion {row['source_ever']}",
            f"strict hold {row['held_strict_pose']}",
            f"final error {row['final']['keypoint_error_mm']:.1f} mm",
        ])
        cards.append(f'<section><h2>{escape(label)}</h2><video controls preload="metadata" '
                     f'src="seed_{seed}/front_wrist.mp4"></video><p><a href="seed_{seed}/trajectory.json">'
                     'Trajectory and metrics</a></p></section>')
    (output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld π0 RGB rollouts</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto;background:#16191d;color:#eee}"
        "video{width:100%;max-width:768px}a{color:#9bd}section{border-top:1px solid #555;padding:1rem 0}</style>"
        "</head><body><h1>ContactWorld π0 RGB closed-loop rollouts</h1>"
        "<p>Front (left), wrist (right). Random resets, no demonstration actions. Source success and strict pose "
        "are diagnostic proxies, not independent physical insertion verification.</p>"
        + "".join(cards) + "</body></html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint for result provenance")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--replan-every", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.replan_every <= 16:
        parser.error("--replan-every must be in [1, 16]")
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacgym import gymapi  # noqa: F401  Must precede torch import.
    import torch

    torch.set_num_threads(4)
    env = None
    summary = {"policy_config": "pi0_cw_usb_01_rgb", "checkpoint": str(args.checkpoint.resolve()),
               "seed_base": args.seed_base,
               "replan_every": args.replan_every, "max_steps": args.max_steps,
               "scope": "random-reset closed-loop RGB policy review; source/strict pose are proxies",
               "results": []}
    try:
        # Reuse the already validated USB scene configuration and white gel/mount colors.
        args.task = "usb"
        env = prepare_environment(args, torch, np)
        with Client(str(args.socket), family="AF_UNIX", authkey=b"contactworld-rgb") as connection:
            for trial in range(args.trials):
                seed = args.seed_base + trial
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
                env.refresh_all_tensors()
                env.compute_observations()
                initial = describe_pose(env)
                trial_dir = args.output / f"seed_{seed}"
                trial_dir.mkdir(exist_ok=True)
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", "512x256", "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(trial_dir / "front_wrist.mp4")],
                    stdin=subprocess.PIPE,
                )
                rows = []
                actions = None
                source_ever = initial["source_success"]
                strict_streak = 0
                held_strict = False
                stop_reason = "step_budget"
                try:
                    proc.stdin.write(video_frame(env).tobytes())
                    for step in range(args.max_steps):
                        if step % args.replan_every == 0:
                            request = observation(env)
                            connection.send(request)
                            response = connection.recv()
                            if "error" in response:
                                raise RuntimeError(response["error"])
                            actions = np.asarray(response["actions"], dtype=np.float32)
                            latency = float(response["infer_ms"])
                        action = actions[step % args.replan_every]
                        if not np.isfinite(action).all():
                            stop_reason = "nonfinite_action"
                            break
                        env.step(torch.as_tensor(action[None], device=env.device))
                        pose = describe_pose(env)
                        source_ever |= pose["source_success"]
                        strict_streak = strict_streak + 1 if strict_pose(pose) else 0
                        rows.append({"step": step, "action_raw": action.tolist(),
                                     "action_sim_clipped": np.clip(action, -env.clip_actions, env.clip_actions).tolist(),
                                     "inference_ms": latency if step % args.replan_every == 0 else None, **pose})
                        proc.stdin.write(video_frame(env).tobytes())
                        if step % 20 == 0:
                            print(f"seed {seed} step {step}: error={pose['keypoint_error_mm']:.1f}mm "
                                  f"source={pose['source_success']}", flush=True)
                        if strict_streak >= 5:
                            hold = []
                            for _ in range(10):
                                env.step(torch.zeros((1, 6), device=env.device))
                                hold.append(describe_pose(env))
                                proc.stdin.write(video_frame(env).tobytes())
                            held_strict = all(strict_pose(p) for p in hold)
                            rows[-1]["hold_poses"] = hold
                            if held_strict:
                                stop_reason = "held_strict_pose"
                                break
                            strict_streak = 0
                        if not np.isfinite(pose["keypoint_error_mm"]):
                            stop_reason = "nonfinite_state"
                            break
                finally:
                    proc.stdin.close()
                    if proc.wait() != 0:
                        raise RuntimeError("ffmpeg video encoder failed")
                final = describe_pose(env)
                result = {"seed": seed, "steps": len(rows), "initial": initial, "final": final,
                          "source_ever": bool(source_ever), "held_strict_pose": bool(held_strict),
                          "min_keypoint_error_mm": min([initial["keypoint_error_mm"]] +
                                                   [row["keypoint_error_mm"] for row in rows]),
                          "stop_reason": stop_reason,
                          "video": str(trial_dir / "front_wrist.mp4")}
                (trial_dir / "trajectory.json").write_text(json.dumps({"result": result, "rows": rows}, indent=2) + "\n")
                summary["results"].append(result)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                write_index(args.output, summary)
                print(f"trial complete: {result}", flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
