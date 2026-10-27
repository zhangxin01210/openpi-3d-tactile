#!/usr/bin/env python3
"""Run a trained XHand Pi0 checkpoint on causal perception packets; never commands a robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = {
    "pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1": "checkpoints/pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1/m2_b8_20k_1026/19999",
    "pi0_xhand_spatial_dtcr_v1": "checkpoints/pi0_xhand_spatial_dtcr_v1/dtcr_b8_20k_1026/19999",
    "pi0_xhand_spatial_dtcr_independent_v1": "checkpoints/pi0_xhand_spatial_dtcr_independent_v1/dtcr_independent_b8_20k_1026/19999",
    "pi0_xhand_spatial_hand_scene_oo_object_centric_v1": "checkpoints/pi0_xhand_spatial_hand_scene_oo_object_centric_v1/m2_oc_b8_20k_1026/19999",
}
OBJECT_FIELDS = (
    "manipulated_xyz_m", "manipulated_mask", "manipulated_source", "manipulated_age", "manipulated_confidence",
    "target_xyz_m", "target_mask", "target_source", "target_age", "target_confidence",
)
HAND_FIELDS = (
    "manipulated_relation_valid", "target_relation_valid", "manipulated_geometry_quality", "target_geometry_quality",
    "grasp_reference_force_norm", "grasp_short_delta_force_norm", "grasp_contact_valid", "grasp_quality",
    "funnel_mask", "funnel_mask_valid", "funnel_mask_quality", "funnel_mask_hand_offset_xy",
)
CAUSAL_FIELDS = (
    "ema_before_force_norm", "ema_after_force_norm", "current_minus_ema_force_norm",
    "current_minus_previous_force_norm", "reference_valid", "previous_valid", "sample_valid", "alpha",
)


def _geometry_from_npz(source: np.lib.npyio.NpzFile) -> dict[str, object]:
    prefix = "object_geometry__"
    missing = [name for name in OBJECT_FIELDS if prefix + name not in source.files]
    missing += ["hand_scene__" + name for name in HAND_FIELDS if prefix + "hand_scene__" + name not in source.files]
    if missing:
        raise KeyError(f"Perception packet misses training geometry fields: {missing}")
    result = {name: np.asarray(source[prefix + name]) for name in OBJECT_FIELDS}
    hand_scene = {name: np.asarray(source[prefix + "hand_scene__" + name]) for name in HAND_FIELDS}
    causal_keys = [prefix + "hand_scene__causal_tactile__" + name for name in CAUSAL_FIELDS]
    if all(key in source.files for key in causal_keys):
        hand_scene["causal_tactile"] = {
            name: np.asarray(source[key]) for name, key in zip(CAUSAL_FIELDS, causal_keys, strict=True)
        }
    result["hand_scene"] = hand_scene
    return result


def _frame_stream(frames_dir: Path, *, follow: bool, idle_timeout_s: float):
    seen: set[Path] = set()
    last_frame = -1
    last_packet_at = time.monotonic()
    while True:
        paths = [path for path in sorted(frames_dir.glob("frame_*.npz")) if path not in seen]
        if paths:
            for path in paths:
                suffix = path.stem.removeprefix("frame_")
                if not suffix.isdigit():
                    raise ValueError(f"Unexpected frame packet name: {path.name}")
                frame_index = int(suffix)
                if frame_index <= last_frame:
                    raise ValueError(f"Policy frame indices must increase strictly: {path.name}")
                seen.add(path)
                last_frame = frame_index
                last_packet_at = time.monotonic()
                yield path
            continue
        if not follow:
            if not seen:
                raise FileNotFoundError(f"No perception frame packets under {frames_dir}")
            return
        if idle_timeout_s > 0 and time.monotonic() - last_packet_at >= idle_timeout_s:
            return
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--config", choices=tuple(DEFAULT_CHECKPOINTS), default="pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--follow", action="store_true", help="Consume atomic frame packets as the perception process writes them.")
    parser.add_argument("--idle-timeout-s", type=float, default=0.0, help="Stop a followed stream after this many seconds without a packet; 0 waits until interrupted.")
    parser.add_argument("--prompt", default="Move the funnel over the target flask and insert it.")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or (root / DEFAULT_CHECKPOINTS[args.config])).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {checkpoint}; refusing to use base weights")
    if args.idle_timeout_s < 0:
        raise ValueError("--idle-timeout-s must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when set")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dry-run output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    train_config = get_config(args.config)
    if train_config.model.action_dim < 18:
        raise ValueError(f"Configured Pi0 action dimension {train_config.model.action_dim} cannot carry 18 XHand joints")
    policy = create_trained_policy(train_config, checkpoint)
    timings = []
    spatial_timings = []
    input_prepare_timings = []
    model_timings = []
    output_transform_timings = []
    index_path = output_dir / "actions.jsonl"
    with index_path.open("x", encoding="utf-8") as index:
        for frame_number, path in enumerate(
            _frame_stream(frames_dir, follow=args.follow, idle_timeout_s=args.idle_timeout_s)
        ):
            if args.max_frames is not None and frame_number >= max(0, args.max_frames):
                break
            with np.load(path, allow_pickle=False) as source:
                required = ("front_rgb", "left_rgb", "right_rgb", "front_depth_raw", "robot_state", "timestamp_s", "frame_index")
                absent = [key for key in required if key not in source.files]
                if absent:
                    raise KeyError(f"{path}: missing policy raw inputs {absent}")
                geometry = _geometry_from_npz(source)
                frame_index = int(np.asarray(source["frame_index"]).reshape(()))
                timestamp = float(np.asarray(source["timestamp_s"]).reshape(()))
                observation = {
                    "observation.images.cam_front": np.asarray(source["front_rgb"]),
                    "observation.images.cam_left": np.asarray(source["left_rgb"]),
                    "observation.images.cam_right": np.asarray(source["right_rgb"]),
                    "observation.depths.cam_front": np.asarray(source["front_depth_raw"]),
                    "observation.state": np.asarray(source["robot_state"], dtype=np.float32),
                    "object_geometry": geometry,
                    "frame_index": frame_index,
                    "timestamp_s": timestamp,
                    "prompt": args.prompt,
                }
                if "episode_index" in source.files:
                    observation["episode_index"] = int(np.asarray(source["episode_index"]).reshape(()))
            start = time.perf_counter()
            prediction = policy.infer(observation)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timing = prediction.get("policy_timing", {})
            spatial_ms = float(timing.get("input_transform_ms", 0.0))
            input_prepare_ms = float(timing.get("model_input_prepare_ms", 0.0))
            model_ms = float(timing.get("infer_ms", 0.0))
            output_transform_ms = float(timing.get("output_transform_ms", 0.0))
            actions = np.asarray(prediction["actions"], dtype=np.float32)
            expected_shape = (int(train_config.model.action_horizon), 18)
            if actions.shape != expected_shape:
                raise ValueError(f"Policy returned {actions.shape}; expected {expected_shape} from this XHand config")
            if not np.isfinite(actions).all():
                raise FloatingPointError(f"Pi0 output contains NaN/Inf at frame {frame_index}")
            with (output_dir / f"actions_{frame_index:06d}.npz").open("xb") as out:
                np.savez_compressed(out, actions=actions)
            row = {
                "frame_index": frame_index,
                "timestamp_s": timestamp,
                "action_shape": list(actions.shape),
                "first_action_18d": actions[0].tolist(),
                "elapsed_ms": elapsed_ms,
                "spatial_latency_ms": spatial_ms,
                "spatial_preprocess_and_repack_ms": spatial_ms,
                "model_input_prepare_ms": input_prepare_ms,
                "model_inference_ms": model_ms,
                "output_transform_ms": output_transform_ms,
                "policy_latency_ms": elapsed_ms,
                "dry_run": True,
                "robot_command_sent": False,
            }
            index.write(json.dumps(row, ensure_ascii=False) + "\n")
            index.flush()
            timings.append(elapsed_ms)
            spatial_timings.append(spatial_ms)
            input_prepare_timings.append(input_prepare_ms)
            model_timings.append(model_ms)
            output_transform_timings.append(output_transform_ms)
            print(f"[DRY-RUN] frame={frame_index} actions={actions.shape} elapsed_ms={elapsed_ms:.1f}", flush=True)
            if args.max_frames is not None and len(timings) >= args.max_frames:
                break
    if not timings:
        raise ValueError("No perception frame selected for policy inference")
    summary = {
        "status": "pass",
        "config": args.config,
        "checkpoint": str(checkpoint),
        "frames": len(timings),
        "stream_mode": "follow" if args.follow else "replay",
        "action_horizon": int(train_config.model.action_horizon),
        "action_dim_out": 18,
        "mean_inference_ms": float(np.mean(timings)),
        "p95_inference_ms": float(np.percentile(timings, 95)),
        "mean_spatial_preprocess_and_repack_ms": float(np.mean(spatial_timings)),
        "mean_model_input_prepare_ms": float(np.mean(input_prepare_timings)),
        "mean_model_inference_ms": float(np.mean(model_timings)),
        "mean_output_transform_ms": float(np.mean(output_transform_timings)),
        "dry_run": True,
        "robot_command_sent": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
