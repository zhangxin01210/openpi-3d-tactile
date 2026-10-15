#!/usr/bin/env python3
"""Audit UniVTAC HDF5 episodes without importing Isaac Sim.

The released UniVTAC files store joint/EEF observations; action labels are
defined by the upstream loader as the next row of those trajectories. This
script makes that convention explicit and reports whether camera depth is
actually present in the downloaded archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from unproject_univtac_depth import unproject_opengl


def _dataset_schema(group: h5py.Group, prefix: str = "") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            result[path] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "bytes": int(item.nbytes),
            }
        else:
            result.update(_dataset_schema(item, path))
    return result


def _stream_lengths(dataset: h5py.Dataset) -> dict[str, int]:
    values = dataset[:]
    if values.dtype.kind not in {"S", "O", "U"}:
        return {"count": int(len(values)), "nonempty": int(len(values))}
    lengths = np.asarray([len(x) for x in values], dtype=np.int64)
    return {
        "count": int(len(lengths)),
        "nonempty": int(np.count_nonzero(lengths)),
        "min_bytes": int(lengths.min()) if lengths.size else 0,
        "max_bytes": int(lengths.max()) if lengths.size else 0,
    }


def _camera_geometry(root: h5py.File, name: str, frames: int) -> dict[str, Any]:
    if frames < 1:
        return {"pass": False, "reason": "empty episode"}
    prefix = f"observation/{name}"
    required = (f"{prefix}/depth", f"{prefix}/intrinsic", f"{prefix}/pose_w_opengl")
    if any(path not in root for path in required):
        return {"pass": False, "reason": "camera depth/intrinsic/world pose missing"}
    depth = root[required[0]]
    intrinsic = root[required[1]]
    pose = root[required[2]]
    expected = (
        depth.ndim == 4 and depth.shape[0] == frames and depth.shape[-1] == 1
        and depth.shape[1] > 0 and depth.shape[2] > 0,
        intrinsic.shape == (frames, 3, 3),
        pose.shape == (frames, 7),
    )
    if not all(expected):
        return {
            "pass": False, "reason": "unexpected camera calibration or depth shape",
            "shapes": [list(x.shape) for x in (depth, intrinsic, pose)],
        }
    intrinsics = np.asarray(intrinsic[:])
    poses = np.asarray(pose[:])
    intrinsic_valid = bool(
        np.isfinite(intrinsics).all()
        and np.all(intrinsics[:, 0, 0] > 0)
        and np.all(intrinsics[:, 1, 1] > 0)
    )
    quaternion_norm_error = np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1.0)
    pose_valid = bool(np.isfinite(poses).all() and np.all(quaternion_norm_error < 0.01))
    valid_depth_fractions = []
    for start in range(0, frames, 16):
        chunk = np.asarray(depth[start:start + 16])
        valid = np.isfinite(chunk) & (chunk > 0)
        valid_depth_fractions.extend(valid.mean(axis=(1, 2, 3)).tolist())
    samples = []
    for index in sorted({0, frames // 2, frames - 1}):
        d = np.asarray(depth[index])
        k = np.asarray(intrinsic[index])
        p = np.asarray(pose[index])
        valid = np.isfinite(d) & (d > 0)
        cloud, _ = unproject_opengl(d, k, p, stride=4, max_depth_m=3.0)
        samples.append({
            "frame": index,
            "valid_depth_fraction": float(valid.mean()),
            "pointcloud_count_stride4": int(len(cloud)),
            "pose_quaternion_norm": float(np.linalg.norm(p[3:])),
            "pointcloud_bounds_m": [cloud.min(axis=0).tolist(), cloud.max(axis=0).tolist()] if len(cloud) else None,
        })
    return {
        "pass": intrinsic_valid and pose_valid and min(valid_depth_fractions) > 0.01 and all(
            sample["valid_depth_fraction"] > 0.01
            and sample["pointcloud_count_stride4"] > 100
            and abs(sample["pose_quaternion_norm"] - 1.0) < 0.01
            for sample in samples
        ),
        "all_intrinsics_valid": intrinsic_valid,
        "all_poses_valid": pose_valid,
        "max_quaternion_norm_error": float(quaternion_norm_error.max()),
        "all_frame_valid_depth_fraction_min": float(min(valid_depth_fractions)),
        "all_frame_valid_depth_fraction_max": float(max(valid_depth_fractions)),
        "samples": samples,
    }


def audit_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as root:
        schema = _dataset_schema(root)
        length = int(root["step"].shape[0])
        steps = np.asarray(root["step"])
        joint = np.asarray(root["embodiment/joint"])
        ee = np.asarray(root["embodiment/ee"])
        press_left = np.asarray(root["tactile/left_tactile/press_depth"])
        press_right = np.asarray(root["tactile/right_tactile/press_depth"])
        marker_left = np.asarray(root["tactile/left_tactile/marker"])
        marker_right = np.asarray(root["tactile/right_tactile/marker"])

        result: dict[str, Any] = {
            "file": str(path),
            "frames": length,
            "all_dataset_frame_counts_match": all(
                item["shape"] and item["shape"][0] == length for item in schema.values()
            ),
            "step_deltas": {
                "min": int(np.diff(steps).min()) if length > 1 else None,
                "max": int(np.diff(steps).max()) if length > 1 else None,
                "all_positive": bool(np.all(np.diff(steps) > 0)),
            },
            "schema": schema,
            "required_present": {
                "joint": "embodiment/joint" in schema,
                "ee": "embodiment/ee" in schema,
                "head_rgb": "observation/head/rgb" in schema,
                "wrist_rgb": "observation/wrist/rgb" in schema,
                "head_depth": "observation/head/depth" in schema,
                "wrist_depth": "observation/wrist/depth" in schema,
                "head_intrinsic": "observation/head/intrinsic" in schema,
                "wrist_intrinsic": "observation/wrist/intrinsic" in schema,
                "head_pose_w_opengl": "observation/head/pose_w_opengl" in schema,
                "wrist_pose_w_opengl": "observation/wrist/pose_w_opengl" in schema,
                "left_press_depth": "tactile/left_tactile/press_depth" in schema,
                "right_press_depth": "tactile/right_tactile/press_depth" in schema,
            },
            "derived_action_contract": {
                "source": "next row of embodiment/joint and embodiment/ee",
                "valid_action_frames": max(length - 1, 0),
                "joint_action_dim": int(joint.shape[-1]),
                "ee_action_dim": int(ee.shape[-1]),
                "joint_delta_abs_max": float(np.abs(np.diff(joint, axis=0)).max(initial=0.0)),
                "ee_delta_abs_max": float(np.abs(np.diff(ee, axis=0)).max(initial=0.0)),
            },
            "tactile": {
                "left_press_depth_min": float(press_left.min(initial=0.0)),
                "left_press_depth_max": float(press_left.max(initial=0.0)),
                "left_press_depth_nonzero_fraction": float(np.count_nonzero(press_left) / press_left.size),
                "right_press_depth_min": float(press_right.min(initial=0.0)),
                "right_press_depth_max": float(press_right.max(initial=0.0)),
                "right_press_depth_nonzero_fraction": float(np.count_nonzero(press_right) / press_right.size),
                "left_marker_shape": list(marker_left.shape),
                "right_marker_shape": list(marker_right.shape),
            },
            "compressed_streams": {
                "head_rgb": _stream_lengths(root["observation/head/rgb"]),
                "wrist_rgb": _stream_lengths(root["observation/wrist/rgb"]),
                "left_tactile_rgb": _stream_lengths(root["tactile/left_tactile/rgb"]),
                "right_tactile_rgb": _stream_lengths(root["tactile/right_tactile/rgb"]),
            },
            "camera_geometry": {
                name: _camera_geometry(root, name, length) for name in ("head", "wrist")
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Episode directory, e.g. data/isaac51/insert_hole")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episode_root = args.root.expanduser().resolve()
    files = sorted(episode_root.glob("hdf5/*.hdf5"))
    incomplete = sorted(str(p) for p in episode_root.glob("hdf5/*.incomplete"))
    if not files and not incomplete:
        raise SystemExit(f"No HDF5 files found under {episode_root / 'hdf5'}")

    episodes = [audit_episode(path) for path in files]
    required_keys = set(episodes[0]["required_present"]) if episodes else set()
    schema_signatures = {
        json.dumps(
            {
                path: {
                    "dtype_kind": value["dtype"][0] if value["dtype"] else "",
                    "tail_shape": value["shape"][1:],
                }
                for path, value in episode["schema"].items()
            },
            sort_keys=True,
        )
        for episode in episodes
    }
    aggregate: dict[str, Any] = {
        "dataset_root": str(episode_root),
        "episode_count": len(episodes),
        "incomplete_file_count": len(incomplete),
        "incomplete_files": incomplete,
        "all_episode_schemas_match": len(schema_signatures) <= 1,
        "all_dataset_frame_counts_match": bool(episodes) and all(
            episode["all_dataset_frame_counts_match"] for episode in episodes
        ),
        "all_required_flags": {
            key: bool(episodes) and all(e["required_present"].get(key, False) for e in episodes)
            for key in sorted(required_keys)
        },
        "episodes": episodes,
    }
    aggregate["data_gate"] = {
        "pass": bool(episodes) and not incomplete and aggregate["all_dataset_frame_counts_match"] and all(
            aggregate["all_required_flags"].get(key, False)
            for key in ("joint", "ee", "head_rgb", "wrist_rgb", "left_press_depth", "right_press_depth")
        ),
        "reason": "Synchronized RGB, tactile, and embodiment columns are present with matching frame counts.",
    }
    aggregate["pointcloud_gate"] = {
        "pass": bool(episodes) and not incomplete and aggregate["all_dataset_frame_counts_match"] and all(
            aggregate["all_required_flags"].get(key, False)
            for key in (
                "head_depth", "wrist_depth", "head_intrinsic", "wrist_intrinsic",
                "head_pose_w_opengl", "wrist_pose_w_opengl",
            )
        ) and all(
            episode["camera_geometry"][name]["pass"]
            for episode in episodes for name in ("head", "wrist")
        ),
        "reason": "Metric world-frame clouds require synchronized camera depth, intrinsics, and world poses.",
    }
    if aggregate["pointcloud_gate"]["pass"]:
        aggregate["pointcloud_gate"]["reason"] += " All episodes passed sampled reconstruction checks."
    else:
        aggregate["pointcloud_gate"]["reason"] += " One or more episodes lack these fields or failed reconstruction checks."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({k: aggregate[k] for k in (
        "episode_count", "incomplete_file_count", "all_required_flags", "data_gate", "pointcloud_gate"
    )}, indent=2))


if __name__ == "__main__":
    main()
