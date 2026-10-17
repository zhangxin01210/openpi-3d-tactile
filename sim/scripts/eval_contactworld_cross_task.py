#!/usr/bin/env python3
"""Resumable closed-loop evaluation for the three ContactWorld task families.

This is the cross-task counterpart of ``eval_contactworld_pi0_v3.py``.  It
uses the same 40 paired random resets, 180-step budget, four-step replanning,
RGB/cloud/tactile capture and per-seed trajectory files, while selecting the
task class and action dimension from the cross-task contract.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from contactworld_cross_task_common import TASKS, get_task
from contactworld_cross_task_eval_inputs import CONFIGS, policy_input, snapshot
from eval_contactworld_proprio_baseline import describe_pose, video_frame
from probe_contactworld_cross_task import _enable_bilateral, _stage_released_task_assets


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def physical_state(env) -> dict[str, list[float]]:
    keys = ("ee_pos", "ee_quat", "plug_pos", "plug_quat", "socket_pos_gt", "socket_quat",
            "dof_pos", "dof_vel")
    return {key: np.asarray(env.obs_dict[key][0].detach().cpu()).astype(float).tolist()
            for key in keys if key in env.obs_dict}


def metrics(env) -> tuple[dict, np.ndarray]:
    pair = env.contact_force_pairwise[
        0, env.plug_body_id_env, env.socket_body_id_env].detach().cpu().numpy()
    local = np.stack([
        np.asarray(env.obs_dict["tactile_force_field_" + side][0].detach().cpu())
        for side in ("left", "right")
    ]).astype(np.float32)
    norms = np.linalg.norm(local, axis=-1)
    return {
        **describe_pose(env),
        "ee_plug_distance_mm": float(np.linalg.norm(
            np.asarray(env.obs_dict["ee_pos"][0].detach().cpu()) -
            np.asarray(env.obs_dict["plug_pos"][0].detach().cpu())) * 1000),
        "gripper_width_mm": float(np.asarray(env.obs_dict["dof_pos"][0].detach().cpu())[-2:].sum() * 1000),
        "plug_socket_contact_force_xyz": pair.astype(float).tolist(),
        "tacff_left_active_fraction": float((norms[0] > 1e-4).mean()),
        "tacff_right_active_fraction": float((norms[1] > 1e-4).mean()),
        "tacff_left_max_norm": float(norms[0].max()),
        "tacff_right_max_norm": float(norms[1].max()),
        "tacff_left_channel_abs_mean": np.abs(local[0]).mean(axis=(0, 1)).tolist(),
        "tacff_right_channel_abs_mean": np.abs(local[1]).mean(axis=(0, 1)).tolist(),
    }, local


def save_index(root: Path, summary: dict) -> None:
    cards = []
    for result in summary["results"]:
        seed = result["seed"]
        title = ("seed %d · ever %s · final %s · first %s · min %.1f mm" % (
            seed, result["source_ever"], result["source_final"],
            result["first_success_step"], result["min_keypoint_error_mm"]))
        cards.append(
            '<section><h2>%s</h2><video controls preload="metadata" '
            'src="seed_%d/front_wrist.mp4"></video><p>'
            '<a href="seed_%d/trajectory.json">Trajectory</a> · '
            '<a href="seed_%d/inputs.npz">All sensor inputs</a></p></section>' % (
                escape(title), seed, seed, seed))
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld cross-task evaluation</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto;background:#16191d;color:#eee}"
        "video{width:100%;max-width:768px}a{color:#9bd}section{border-top:1px solid #555;padding:1rem 0}</style>"
        "</head><body><h1>ContactWorld cross-task · " + escape(summary["policy_config"]) + "</h1>"
        "<p>Task: " + escape(summary["task"]) + " · prompt: " + escape(summary["prompt"]) +
        " · action dim: %d · paired seeds: %d · replan every %d control steps.</p>" % (
            summary["action_dim"], summary["trials_requested"], summary["replan_every"]) +
        "<p>Videos show synchronized front and wrist views. Trajectories and NPZ files retain RGB, "
        "fused cloud, bilateral base tactile fields and executed policy actions.</p>" +
        "".join(cards) + "</body></html>", encoding="utf-8")


def save_inputs(path: Path, snapshots: list[tuple[int, dict]], chunks: list[np.ndarray],
                step_fields: list[np.ndarray]) -> None:
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


def prepare_environment(args, torch, np):
    from isaacgym import gymapi  # noqa: F401
    import hydra
    from omegaconf import OmegaConf

    spec = get_task(args.task)
    code = args.source / "thirdparty/manifeel-isaacgymenvs"
    sys.path.insert(0, str(code))
    import isaacgymenvs
    tasks = types.ModuleType("isaacgymenvs.tasks")
    tasks.__path__ = [str(code / "isaacgymenvs/tasks")]
    sys.modules["isaacgymenvs.tasks"] = tasks
    isaacgymenvs.tasks = tasks
    module = __import__("isaacgymenvs.tasks.tacsl.tacsl_task_" + spec.hydra_config,
                        fromlist=[spec.task_class])
    task_class = getattr(module, spec.task_class)
    _enable_bilateral(task_class)

    cfg_root = args.output / "hydra"
    cfg_root.mkdir(parents=True, exist_ok=True)
    manifeel = args.source / "thirdparty/manifeel"
    _stage_released_task_assets(args.source)
    for config in (manifeel / "manifeel/config").glob("*.yaml"):
        link = cfg_root / config.name
        if not link.exists():
            link.symlink_to(config.resolve())
    for name, target in (("task", manifeel / "manifeel/config/task"),
                         ("assets", manifeel / "assets")):
        link = cfg_root / name
        if not link.exists():
            link.symlink_to(target.resolve())
    # The task constructor calls hydra.compose again while loading the
    # environment YAML (see TacSLEnvInsertion._get_env_yaml_params).
    # Keep Hydra initialized through construction; leaving this context
    # after composing the outer config causes ``GlobalHydra is not
    # initialized`` before the first trial starts.
    with hydra.initialize_config_dir(config_dir=str(cfg_root.resolve()), version_base="1.1"):
        cfg = hydra.compose(config_name="isaacgym_config_" + spec.hydra_config,
                            overrides=["num_envs=1", "headless=true"])
        cfg.capture_video = False
        cfg.force_render = False
        task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
        task_cfg["env"].setdefault("obsDims", {})["tactile_force_field_left"] = [10, 14, 3]
        # The released Valve task is a screwing task.  Its shared TacSL YAML
        # defaults to insertion, which would bypass the official terminal
        # screw primitive and evaluate the wrong success condition.
        if args.task == "valve":
            task_cfg["env"]["task_type"] = "screwing"
        (args.output / "task_config.json").write_text(json.dumps(task_cfg, indent=2) + "\n")
        np.random.seed(args.seed_base)
        torch.manual_seed(args.seed_base)
        return task_class(cfg=task_cfg, rl_device="cuda:0", sim_device="cuda:0",
                          graphics_device_id=0, headless=True,
                          virtual_screen_capture=False, force_render=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--limit-this-run", type=int)
    args = parser.parse_args()
    spec = get_task(args.task)
    if not args.config.startswith(f"pi0_cw3_{args.task}_"):
        parser.error("--config task does not match --task")
    if args.trials < 1 or args.max_steps < 1 or not 1 <= args.replan_every <= 16:
        parser.error("Invalid trial/step/replan count")
    if args.limit_this_run is not None and args.limit_this_run < 1:
        parser.error("--limit-this-run must be positive")
    if not (args.checkpoint / "_CHECKPOINT_METADATA").is_file():
        parser.error("Incomplete checkpoint: " + str(args.checkpoint))
    if not args.source.is_dir():
        parser.error("Missing patched ContactWorld checkout: " + str(args.source))
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacgym import gymapi  # noqa: F401
    import torch
    torch.set_num_threads(4)
    checkpoint_hash = hashlib.sha256((args.checkpoint / "_CHECKPOINT_METADATA").read_bytes()).hexdigest()
    summary = {
        "task": args.task, "prompt": spec.prompt, "action_dim": spec.action_dim,
        "policy_config": args.config, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": checkpoint_hash, "seed_base": args.seed_base,
        "trials_requested": args.trials, "max_steps": args.max_steps,
        "replan_every": args.replan_every, "source": str(args.source.resolve()),
        "source_success_definition": "task env _check_success()",
        "primary_metric": "source_ever", "results": [],
        "input_contract": "cross-task V3 bilateral: front+wrist RGB; fused base 4096 XYZ; ordered left/right 140-taxel base force",
    }
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if int(previous.get("trials_requested", 0)) > args.trials:
            raise ValueError("Existing summary has more trials than requested")
        for key in ("task", "policy_config", "checkpoint", "checkpoint_metadata_sha256", "seed_base",
                    "max_steps", "replan_every", "source", "action_dim"):
            if previous.get(key) != summary[key]:
                raise ValueError("Existing summary differs on " + key)
        summary = previous
        # A short smoke run can be extended in place to the full trial count.
        summary["trials_requested"] = args.trials
    completed = len(summary["results"])
    if [r["seed"] for r in summary["results"]] != list(range(args.seed_base, args.seed_base + completed)):
        raise ValueError("Existing results are not a contiguous seed prefix")
    stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
    if stop - completed != 1:
        parser.error("The cross-task launcher invokes this evaluator one seed at a time")
    seed = args.seed_base + completed
    env = None
    try:
        env = prepare_environment(args, torch, np)
        if int(env.num_actions) != spec.action_dim:
            raise ValueError(f"Environment action dim {env.num_actions} != task contract {spec.action_dim}")
        dt = float(env.cfg_base.sim.dt) * int(env.cfg_task.env.controlFrequencyInv)
        summary["action_dt_s"], summary["video_fps"] = dt, 1 / dt
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
        env.refresh_all_tensors(); env.compute_observations()
        initial, _ = metrics(env)
        initial_state = physical_state(env)
        initial_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
        first = snapshot(env)
        trial_dir = args.output / f"seed_{seed}"
        trial_dir.mkdir(exist_ok=True)
        video = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "512x256", "-r", "%.6f" % (1 / dt), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(trial_dir / "front_wrist.mp4")
        ], stdin=subprocess.PIPE)
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
                        connection.send(policy_input(snap, args.config, spec.prompt))
                        response = connection.recv()
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        actions = np.asarray(response["actions"], dtype=np.float32)
                        expected = (16, spec.action_dim)
                        if actions.shape != expected or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid policy action chunk: {actions.shape}, expected {expected}")
                        snapshots.append((step, snap)); chunks.append(actions.copy())
                        latency = float(response["infer_ms"])
                        tactile = {
                            side: {
                                "force_local_abs_mean": np.abs(snap["force_local"][index]).mean(axis=(0, 1)).tolist(),
                                "force_base_abs_mean": np.abs(snap["force_base"][index]).mean(axis=0).tolist(),
                            } for index, side in enumerate(("left", "right"))
                        }
                    action = actions[step % args.replan_every]
                    env.step(torch.as_tensor(action[None], device=env.device))
                    current, field = metrics(env); step_fields.append(field)
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
                                 "tactile_pre_action": tactile if step % args.replan_every == 0 else None,
                                 **current, **physical_state(env)})
                    video.stdin.write(video_frame(env).tobytes())
                    if not np.isfinite(current["keypoint_error_mm"]):
                        reason = "nonfinite_state"; break
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
            "min_keypoint_error_mm": min([initial["keypoint_error_mm"]] + [r["keypoint_error_mm"] for r in rows]),
            "stop_reason": reason, "video": str(trial_dir / "front_wrist.mp4"),
        }
        write_json(trial_dir / "trajectory.json", {"result": result, "cloud_counts_initial": first["cloud_counts"], "rows": rows})
        summary["results"].append(result)
        write_json(summary_path, summary)
        save_index(args.output, summary)
        print(json.dumps({"task": args.task, "config": args.config, "seed": seed,
                          "ever": ever, "final": result["source_final"],
                          "steps": len(rows)}, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

