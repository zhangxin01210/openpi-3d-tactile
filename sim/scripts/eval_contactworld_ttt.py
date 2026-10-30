#!/usr/bin/env python3
"""Closed-loop evaluation of one ContactWorld TTT, Zeva, or RGB/state policy."""

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

from contactworld_cross_task_common import TASKS as CROSS_TASKS, TaskSpec
from contactworld_v3_eval_inputs import obs_array
from eval_contactworld_proprio_baseline import describe_pose, video_frame
from probe_contactworld_cross_task import _enable_bilateral, _stage_released_task_assets


USB = TaskSpec(
    key="usb", source_group="insertion_usb", archive=None,
    hydra_config="usb", task_class="TacSLTaskUSB", action_dim=6,
    prompt="Insert the USB plug into the socket.",
    output_dataset="contactworld_usb_v3_bilateral",
)
TTT_TASKS = {"usb": USB, **CROSS_TASKS}
VALVE_SUCCESS_ROTATION_RAD = -np.pi / 2


def get_task(name: str) -> TaskSpec:
    return TTT_TASKS[name]


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _rgb(value) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype == np.uint8:
        return value.copy()
    return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)


def snapshot_ttt(env) -> dict[str, np.ndarray]:
    """Capture only the current inputs consumed by ContactWorld TTT v0/v1."""
    images = env.get_camera_image_tensors_dict()
    result = {
        role: _rgb(images[role][0].detach().cpu().numpy())
        for role in ("front", "wrist")
    }
    for role in ("front", "wrist"):
        observed = _rgb(obs_array(env, role))
        if result[role].shape != (256, 256, 3) or np.max(np.abs(
                result[role].astype(np.int16) - observed.astype(np.int16))) > 1:
            raise ValueError("Direct camera and policy RGB disagree: " + role)
    result["state"] = np.r_[obs_array(env, "dof_pos"), obs_array(env, "dof_vel")].astype(np.float32)
    result["force_local"] = np.stack([
        obs_array(env, "tactile_force_field_" + side) for side in ("left", "right")
    ]).astype(np.float32)
    if result["state"].shape != (18,) or result["force_local"].shape != (2, 10, 14, 3):
        raise ValueError("Invalid ContactWorld TTT observation shape")
    if not np.isfinite(result["state"]).all() or not np.isfinite(result["force_local"]).all():
        raise ValueError("Nonfinite ContactWorld TTT observation")
    return result


def policy_input(snap: dict[str, np.ndarray], prompt: str, *, reset: bool,
                 runtime: str, executed_actions: list[np.ndarray] | None = None,
                 fixed_noise_seed: int | None = None,
                 tactile_frames: list[np.ndarray] | None = None) -> dict:
    if runtime in ("zeva_base", "zeva", "zeva_tactile"):
        result = {
            "observation/front_rgb": snap["front"],
            "observation/wrist_rgb": snap["wrist"],
            "observation/state": snap["state"],
            "prompt": prompt,
            "reset_zeva_state": reset,
            "executed_actions": (
                np.stack(executed_actions).astype(np.float32, copy=False)
                if executed_actions else None
            ),
            "__fixed_noise_seed__": fixed_noise_seed,
        }
        if runtime == "zeva_tactile":
            result.update({
                "observation/tactile_force": snap["force_local"],
                "tactile_force_frames": (
                    np.stack(tactile_frames).astype(np.float32, copy=False)
                    if tactile_frames else None
                ),
            })
        return result
    result = {
        "observation.images.front": snap["front"],
        "observation.images.wrist": snap["wrist"],
        "observation.state": snap["state"],
        "prompt": prompt,
    }
    if runtime != "baseline":
        result["observation/tactile_force"] = snap["force_local"]
        result["reset_tactile_memory"] = reset
    return result


def checkpoint_metadata_bytes(checkpoint: Path, runtime: str) -> bytes:
    if runtime not in ("zeva_base", "zeva", "zeva_tactile"):
        return (checkpoint / "_CHECKPOINT_METADATA").read_bytes()
    latest = checkpoint / "policy/checkpoints/latest.json"
    selected = json.loads(latest.read_text())["checkpoint"]
    if Path(selected).name != selected:
        raise ValueError("Invalid Zeva latest.json checkpoint reference")
    return (latest.parent / selected / "manifest.json").read_bytes()


def physical_state(env) -> dict[str, list[float]]:
    keys = ("ee_pos", "ee_quat", "plug_pos", "plug_quat", "socket_pos_gt", "socket_quat",
            "dof_pos", "dof_vel")
    return {key: np.asarray(env.obs_dict[key][0].detach().cpu()).astype(float).tolist()
            for key in keys if key in env.obs_dict}


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for Isaac Gym's scalar-last quaternion convention."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=np.float64)


def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    v = np.asarray(vector, dtype=np.float64)
    return _quat_multiply_xyzw(
        _quat_multiply_xyzw(q, np.r_[v, 0.0]), np.r_[-q[:3], q[3]])[:3]


def signed_rotation_about_axis(current: np.ndarray, initial: np.ndarray,
                               axis_world: np.ndarray) -> float:
    """Return shortest relative rotation projected onto a normalized world axis."""
    current = np.asarray(current, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    current /= np.linalg.norm(current)
    initial /= np.linalg.norm(initial)
    relative = _quat_multiply_xyzw(current, np.r_[-initial[:3], initial[3]])
    if relative[3] < 0:
        relative = -relative
    vector_norm = np.linalg.norm(relative[:3])
    if vector_norm < 1e-12:
        return 0.0
    rotvec = relative[:3] / vector_norm * (2 * np.arctan2(vector_norm, relative[3]))
    axis_world = np.asarray(axis_world, dtype=np.float64)
    axis_world /= np.linalg.norm(axis_world)
    return float(np.dot(rotvec, axis_world))


def metrics(env, task: str, initial_plug_quat: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    pair = env.contact_force_pairwise[
        0, env.plug_body_id_env, env.socket_body_id_env].detach().cpu().numpy()
    local = np.stack([
        np.asarray(env.obs_dict["tactile_force_field_" + side][0].detach().cpu())
        for side in ("left", "right")
    ]).astype(np.float32)
    norms = np.linalg.norm(local, axis=-1)
    result = {
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
    }
    # The released valve task's screwing branch reports success whenever the
    # valve is upright and centered, which is already true at reset.  Preserve
    # that native value as source_success, but use actual valve rotation for
    # the benchmark outcome.  All 101 non-degenerate released demonstrations
    # rotate the valve by at least 1.87 rad in the negative socket-axis
    # direction, so pi/2 is a conservative completion threshold.
    if task == "valve":
        if initial_plug_quat is None:
            valve_rotation_rad = 0.0
        else:
            plug_quat = np.asarray(env.obs_dict["plug_quat"][0].detach().cpu())
            socket_quat = np.asarray(env.obs_dict["socket_quat"][0].detach().cpu())
            socket_axis = _quat_rotate_xyzw(socket_quat, np.asarray([0.0, 0.0, 1.0]))
            valve_rotation_rad = signed_rotation_about_axis(
                plug_quat, initial_plug_quat, socket_axis)
        result["valve_rotation_rad"] = valve_rotation_rad
        result["valve_rotation_deg"] = float(np.rad2deg(valve_rotation_rad))
        result["benchmark_success"] = bool(
            result["source_success"] and valve_rotation_rad <= VALVE_SUCCESS_ROTATION_RAD)
    else:
        result["benchmark_success"] = bool(result["source_success"])
    return result, local


def tactile_summary(force: np.ndarray) -> dict:
    return {
        side: {
            "abs_mean": np.abs(force[index]).mean(axis=(0, 1)).tolist(),
            "max_norm": float(np.linalg.norm(force[index], axis=-1).max()),
            "active_fraction": float((np.linalg.norm(force[index], axis=-1) > 1e-4).mean()),
        }
        for index, side in enumerate(("left", "right"))
    }


def prepare_environment(args, torch):
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
    with hydra.initialize_config_dir(config_dir=str(cfg_root.resolve()), version_base="1.1"):
        cfg = hydra.compose(config_name="isaacgym_config_" + spec.hydra_config,
                            overrides=["num_envs=1", "headless=true"])
        cfg.capture_video = False
        cfg.force_render = False
        task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
        task_cfg["env"].setdefault("obsDims", {})["tactile_force_field_left"] = [10, 14, 3]
        if args.task == "valve":
            task_cfg["env"]["task_type"] = "screwing"
        (args.output / "task_config.json").write_text(json.dumps(task_cfg, indent=2) + "\n")
        return task_class(cfg=task_cfg, rl_device="cuda:0", sim_device="cuda:0",
                          graphics_device_id=0, headless=True,
                          virtual_screen_capture=False, force_render=False)


def save_index(root: Path, summary: dict) -> None:
    cards = []
    for result in summary["results"]:
        seed = result["seed"]
        title = "seed %d · ever %s · final %s · min %.1f mm" % (
            seed, result["benchmark_ever"], result["benchmark_final"],
            result["min_keypoint_error_mm"])
        cards.append(
            '<section><h2>%s</h2><video controls preload="metadata" '
            'src="seed_%d/front_wrist.mp4"></video><p><a href="seed_%d/trajectory.json">'
            'Trajectory</a> · <a href="seed_%d/inputs.npz">Inputs</a></p></section>' %
            (escape(title), seed, seed, seed))
    (root / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld TactileTTT</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto;background:#16191d;color:#eee}"
        "video{width:100%;max-width:768px}a{color:#9bd}</style></head><body><h1>" +
        escape(summary["policy_config"]) + "</h1>" + "".join(cards) + "</body></html>", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TTT_TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=16)
    parser.add_argument("--limit-this-run", type=int, default=1)
    parser.add_argument("--ttt-runtime", choices=(
        "online", "noop", "baseline", "zeva_base", "zeva", "zeva_tactile"),
        default="online")
    args = parser.parse_args()
    spec = get_task(args.task)
    if args.ttt_runtime == "baseline":
        if args.config != "pi05_cw_baseline_" + args.task:
            parser.error("baseline --config does not match --task")
    elif args.ttt_runtime == "zeva_base":
        if args.config != "pi05_zeva_base_" + args.task:
            parser.error("Zeva-base --config does not match --task")
        if args.replan_every != 4:
            parser.error("Zeva-base comparison requires --replan-every=4")
    elif args.ttt_runtime == "zeva":
        if args.config != "pi05_zeva_" + args.task:
            parser.error("Zeva --config does not match --task")
        if args.replan_every != 4:
            parser.error("Zeva requires --replan-every=4 for its online CTE cadence")
    elif args.ttt_runtime == "zeva_tactile":
        if args.config != "pi05_zeva_tactile_" + args.task:
            parser.error("Zeva tactile --config does not match --task")
        if args.replan_every != 4:
            parser.error("Zeva tactile requires --replan-every=4 for its online CTE cadence")
    elif not args.config.startswith("pi05_tactile_ttt_") or not args.config.endswith("_cw_" + args.task):
        parser.error("TTT --config does not match --task")
    if not 1 <= args.replan_every <= 16 or args.max_steps < 1 or args.trials < 1:
        parser.error("Invalid rollout counts")
    if args.ttt_runtime in ("zeva_base", "zeva", "zeva_tactile"):
        if not (args.checkpoint / "policy/checkpoints/latest.json").is_file():
            parser.error("Incomplete Zeva checkpoint package")
    elif not (args.checkpoint / "params/_METADATA").is_file():
        parser.error("Incomplete checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)

    from isaacgym import gymapi  # noqa: F401
    import torch
    torch.set_num_threads(4)
    metadata_hash = hashlib.sha256(
        checkpoint_metadata_bytes(args.checkpoint, args.ttt_runtime)).hexdigest()
    summary = {
        "task": args.task, "prompt": spec.prompt, "action_dim": spec.action_dim,
        "policy_config": args.config, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": metadata_hash, "seed_base": args.seed_base,
        "trials_requested": args.trials, "max_steps": args.max_steps,
        "replan_every": args.replan_every, "source": str(args.source.resolve()),
        "ttt_runtime": args.ttt_runtime,
        "source_success_definition": "task env _check_success()",
        "benchmark_success_definition": (
            "valve: native upright+centered AND signed valve rotation about the socket axis <= -pi/2; "
            "other tasks: task env _check_success()"
        ),
        "primary_metric": "benchmark_ever", "results": [],
        "input_contract": (
            "current front+wrist RGB and 18D state"
            if args.ttt_runtime in ("baseline", "zeva_base") else
            "current front+wrist RGB and 18D state; online Wan-VAE/CTE state; "
            "actual four-action transition blocks; episode-0 PIM support"
            if args.ttt_runtime == "zeva" else
            "current front+wrist RGB and 18D state; every-control-step bilateral "
            "2x10x14x3 local force history (30-frame rolling window); online "
            "Wan-VAE/CTE state; actual four-action transition blocks; episode-0 PIM support"
            if args.ttt_runtime == "zeva_tactile" else
            "current front+wrist RGB, 18D state, bilateral 2x10x14x3 local TacFF"
        ),
    }
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        for key in ("task", "policy_config", "checkpoint", "checkpoint_metadata_sha256",
                    "seed_base", "max_steps", "replan_every", "source", "action_dim"):
            if previous.get(key) != summary[key]:
                raise ValueError("Existing summary differs on " + key)
        if previous.get("ttt_runtime", "online") != summary["ttt_runtime"]:
            raise ValueError("Existing summary differs on ttt_runtime")
        summary = previous
        summary.setdefault("ttt_runtime", "online")
        summary["trials_requested"] = args.trials
    completed = len(summary["results"])
    if completed >= args.trials:
        print("COMPLETE", completed, flush=True)
        return
    if args.limit_this_run != 1:
        parser.error("Invoke one seed per simulator process")
    seed = args.seed_base + completed
    env = None
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = prepare_environment(args, torch)
        if int(env.num_actions) != spec.action_dim:
            raise ValueError(f"Environment action dim {env.num_actions} != {spec.action_dim}")
        dt = float(env.cfg_base.sim.dt) * int(env.cfg_task.env.controlFrequencyInv)
        summary["action_dt_s"], summary["video_fps"] = dt, 1 / dt
        env.reset_idx(torch.arange(1, device=env.device, dtype=torch.long))
        env.refresh_all_tensors(); env.compute_observations()
        initial_plug_quat = np.asarray(env.obs_dict["plug_quat"][0].detach().cpu()).copy()
        initial, _ = metrics(env, args.task, initial_plug_quat)
        initial_state = physical_state(env)
        initial_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest()
        first = snapshot_ttt(env)
        trial_dir = args.output / f"seed_{seed}"
        trial_dir.mkdir(exist_ok=True)
        video = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "512x256", "-r", "%.6f" % (1 / dt), "-i", "-", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(trial_dir / "front_wrist.mp4")
        ], stdin=subprocess.PIPE)
        rows, snapshots, chunks, step_fields = [], [], [], []
        executed_since_replan: list[np.ndarray] = []
        tactile_since_replan: list[np.ndarray] = []
        first_success = None
        streak = max_streak = 0
        ever = bool(initial["benchmark_success"])
        reason = "step_budget"
        try:
            video.stdin.write(video_frame(env).tobytes())
            with Client(str(args.socket), family="AF_UNIX", authkey=b"contactworld-ttt") as connection:
                for step in range(args.max_steps):
                    if step % args.replan_every == 0:
                        snap = first if step == 0 else snapshot_ttt(env)
                        if args.ttt_runtime in ("zeva", "zeva_tactile"):
                            if step and len(executed_since_replan) != args.replan_every:
                                raise ValueError("Zeva transition block is not exactly four actions")
                        if args.ttt_runtime == "zeva_tactile":
                            if step and len(tactile_since_replan) != args.replan_every:
                                raise ValueError("Zeva tactile block is not exactly four force frames")
                        connection.send(policy_input(
                            snap, spec.prompt, reset=(step == 0), runtime=args.ttt_runtime,
                            executed_actions=executed_since_replan,
                            fixed_noise_seed=seed * 1000 + step,
                            tactile_frames=tactile_since_replan))
                        response = connection.recv()
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        actions = np.asarray(response["actions"], dtype=np.float32)
                        if actions.shape != (16, spec.action_dim) or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid policy action chunk {actions.shape}")
                        if args.ttt_runtime in ("online", "noop") and bool(response.get("memory_reset")) != (step == 0):
                            raise ValueError("Policy memory reset acknowledgement mismatch")
                        if args.ttt_runtime in ("zeva_base", "zeva", "zeva_tactile") and bool(
                                response.get("zeva_reset")) != (step == 0):
                            raise ValueError("Zeva state reset acknowledgement mismatch")
                        if response.get("ttt_runtime") != args.ttt_runtime:
                            raise ValueError("Policy TTT runtime acknowledgement mismatch")
                        snapshots.append((step, snap)); chunks.append(actions.copy())
                        executed_since_replan.clear()
                        tactile_since_replan.clear()
                        latency = float(response["infer_ms"])
                        tactile = tactile_summary(snap["force_local"])
                    action = actions[step % args.replan_every]
                    env.step(torch.as_tensor(action[None], device=env.device))
                    if args.ttt_runtime in ("zeva", "zeva_tactile"):
                        executed_since_replan.append(action.copy())
                    current, field = metrics(env, args.task, initial_plug_quat); step_fields.append(field)
                    if args.ttt_runtime == "zeva_tactile":
                        tactile_since_replan.append(field.copy())
                    ever |= bool(current["benchmark_success"])
                    if current["benchmark_success"]:
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
                        "memory_reset": (
                            (step == 0) if step % args.replan_every == 0 and
                            args.ttt_runtime in ("online", "noop") else None
                        ),
                        "tactile_pre_action": tactile if step % args.replan_every == 0 else None,
                        **current, **physical_state(env),
                    })
                    video.stdin.write(video_frame(env).tobytes())
                    if not np.isfinite(current["keypoint_error_mm"]):
                        reason = "nonfinite_state"; break
        finally:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        final, _ = metrics(env, args.task, initial_plug_quat)
        np.savez_compressed(
            trial_dir / "inputs.npz",
            steps=np.asarray([step for step, _ in snapshots], np.int32),
            front=np.stack([snap["front"] for _, snap in snapshots]),
            wrist=np.stack([snap["wrist"] for _, snap in snapshots]),
            state=np.stack([snap["state"] for _, snap in snapshots]),
            force_local=np.stack([snap["force_local"] for _, snap in snapshots]),
            action_chunks=np.stack(chunks), force_local_post_step=np.stack(step_fields),
        )
        result = {
            "seed": seed, "steps": len(rows), "initial": initial, "final": final,
            "initial_state_sha256": initial_hash, "initial_physical_state": initial_state,
            "benchmark_ever": ever, "benchmark_final": bool(final["benchmark_success"]),
            "source_ever": bool(initial["source_success"] or any(row["source_success"] for row in rows)),
            "source_final": bool(final["source_success"]),
            "first_success_step": first_success, "max_consecutive_source_steps": max_streak,
            "min_keypoint_error_mm": min([initial["keypoint_error_mm"]] +
                                         [row["keypoint_error_mm"] for row in rows]),
            "stop_reason": reason, "video": str(trial_dir / "front_wrist.mp4"),
        }
        write_json(trial_dir / "trajectory.json", {"result": result, "rows": rows})
        summary["results"].append(result)
        write_json(summary_path, summary); save_index(args.output, summary)
        print(json.dumps({"task": args.task, "config": args.config, "seed": seed,
                          "ever": ever, "final": result["benchmark_final"],
                          "min_mm": result["min_keypoint_error_mm"]}), flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
