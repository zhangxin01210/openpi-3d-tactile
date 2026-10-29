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



def _ground_truth_actions(dataset_root: Path, episode: int) -> dict[int, np.ndarray]:
    """Read original absolute arm6/hand12 training labels from LeRobot parquet."""
    import pyarrow.parquet as pq
    info=json.loads((dataset_root/"meta"/"info.json").read_text(encoding="utf-8"))
    names=info["features"]["action"].get("names")
    if names is not None and (len(names)!=18 or
                              not all(str(names[i]).startswith("arm_joint_") for i in range(6)) or
                              not all(str(names[i]).startswith("hand_joint_") for i in range(6,18))):
        raise ValueError("Original dataset action ordering is not arm 6 + hand 12")
    chunk=int(info.get("chunks_size",1000))
    path=dataset_root/info["data_path"].format(episode_index=episode,
                                               episode_chunk=episode//chunk)
    parquet=pq.ParquetFile(path)
    if not {"action","frame_index"}.issubset(parquet.schema_arrow.names):
        raise KeyError(f"Original dataset parquet lacks action/frame_index: {path}")
    result={}
    for batch in parquet.iter_batches(columns=["frame_index","action"],batch_size=256):
        ids=batch.column(0).to_pylist()
        vectors=batch.column(1).to_pylist()
        for idx, vec in zip(ids,vectors,strict=True):
            action=np.asarray(vec,dtype=np.float32)
            if action.shape!=(18,) or not np.isfinite(action).all():
                raise ValueError(f"GT frame={idx} has invalid 18-dim action {action.shape}")
            idx=int(idx)
            if idx in result:
                raise ValueError(f"Duplicate dataset action frame {idx}")
            result[idx]=action
    if not result:
        raise ValueError(f"No action rows in dataset episode {episode}")
    return result


def _gt_comparison(prediction: np.ndarray, frame: int, ground_truth: dict[int,np.ndarray],
                   robot_state: np.ndarray | None = None):
    """Compare first step and valid future steps using recorded-frame alignment."""
    truth=ground_truth.get(frame)
    if truth is None:
        raise KeyError(f"Episode lacks ground-truth action for frame {frame}")
    predicted=np.asarray(prediction,dtype=np.float32)
    error=np.abs(predicted[0]-truth)
    valid=[(i,ground_truth[frame+i]) for i in range(len(predicted))
           if frame+i in ground_truth]
    if not valid:
        raise ValueError("No aligned labels")
    indices=np.asarray([i for i,_ in valid],dtype=np.int64)
    gt_chunk=np.stack([x for _,x in valid])
    chunk_error=np.abs(predicted[indices]-gt_chunk)
    def stat(a):
        return float(np.mean(a))
    # Crucial: because actions are ABSOLUTE targets, simply holding the
    # current arm/hand positions can look "close" to GT. Report the hold
    # baseline so a close match is not confused with learned motion.
    baseline={}
    if robot_state is not None:
        state=np.asarray(robot_state,dtype=np.float32)
        if state.shape!=(1972,):
            raise ValueError(f"Cannot compute hold baseline from state {state.shape}")
        # XHand positions are interleaved with motor current/torque:
        # xhand_policy.extract_joint_state() uses state[28:52:2].
        hold=np.concatenate([state[:6],state[28:52:2]])
        baseline={
            "hold_first_arm6_mae_rad":float(np.abs(hold[:6]-truth[:6]).mean()),
            "hold_first_hand12_mae_rad":float(np.abs(hold[6:]-truth[6:]).mean()),
        }
    return {
        **baseline,
        "gt_first_action_18d": truth.tolist(),
        "first_action_mae_rad":stat(error),
        "first_arm6_mae_rad":stat(error[:6]),
        "first_hand12_mae_rad":stat(error[6:]),
        "chunk_compared_steps":len(valid),
        "chunk_mae_rad":stat(chunk_error),
        "chunk_arm6_mae_rad":stat(chunk_error[:,:6]),
        "chunk_hand12_mae_rad":stat(chunk_error[:,6:]),
    }



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
    parser.add_argument("--prompt", default="Pick up the glass funnel, align its stem with the opening of the small beaker, and insert it until it is fully seated.")
    parser.add_argument("--gt-dataset", type=Path,
                        help="Original LeRobot dataset root for comparison against recorded 18D actions")
    parser.add_argument("--gt-episode", type=int, default=23)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Infer every Nth packet, e.g. 16 matches 16-step robot replanning")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or (root / DEFAULT_CHECKPOINTS[args.config])).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {checkpoint}; refusing to use base weights")
    if args.idle_timeout_s < 0:
        raise ValueError("--idle-timeout-s must be non-negative")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >=1")
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
    gt_actions = (_ground_truth_actions(args.gt_dataset.expanduser().resolve(),args.gt_episode)
                  if args.gt_dataset is not None else None)
    gt_rows=[]
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
            if frame_number % args.frame_stride:
                continue
            if args.max_frames is not None and len(timings) >= args.max_frames:
                break
            with np.load(path, allow_pickle=False) as source:
                required = ("front_rgb", "left_rgb", "right_rgb", "front_depth_native", "front_depth_color_aligned", "robot_state", "timestamp_s", "frame_index")
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
                    "observation.depths.cam_front": np.asarray(source["front_depth_native"]),
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
            if gt_actions is not None:
                comparison=_gt_comparison(actions,frame_index,gt_actions,
                                          robot_state=observation["observation.state"])
                row["ground_truth"]=comparison
                gt_rows.append({"frame_index":frame_index,**comparison})
                print(f"[GT] frame={frame_index} first_arm6_mae={comparison['first_arm6_mae_rad']:.4f}rad "
                      f"first_hand12_mae={comparison['first_hand12_mae_rad']:.4f}rad "
                      f"chunk_mae={comparison['chunk_mae_rad']:.4f}rad",flush=True)
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
        "input_frame_stride":args.frame_stride,
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
    if gt_rows:
        def aggregate(key):
            return float(np.mean([r[key] for r in gt_rows]))
        summary["ground_truth"]={
            "episode":args.gt_episode,
            "frames_compared":len(gt_rows),
            "first_arm6_mae_rad":aggregate("first_arm6_mae_rad"),
            "first_hand12_mae_rad":aggregate("first_hand12_mae_rad"),
            "first_action_mae_rad":aggregate("first_action_mae_rad"),
            "hold_first_arm6_mae_rad":aggregate("hold_first_arm6_mae_rad"),
            "hold_first_hand12_mae_rad":aggregate("hold_first_hand12_mae_rad"),
            "chunk_mae_rad":aggregate("chunk_mae_rad"),
            "chunk_arm6_mae_rad":aggregate("chunk_arm6_mae_rad"),
            "chunk_hand12_mae_rad":aggregate("chunk_hand12_mae_rad"),
            "meaning":"Open-loop action-label fit only; NOT a robot collision/success test.",
        }
        (output_dir/"ground_truth_comparison.json").write_text(
            json.dumps(gt_rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
