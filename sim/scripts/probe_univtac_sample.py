#!/usr/bin/env python3
"""Extract one synchronized UniVTAC RGB-D/tactile training-contract sample.

This is a data-contract probe, not a LeRobot exporter or a tactile-force
adapter. ``joint_target_next`` and ``ee_target_next`` follow the upstream
loader's next-observation convention; they are not recorded motor commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from unproject_univtac_depth import unproject_opengl


def decode_rgb(encoded: bytes) -> np.ndarray:
    # UniVTAC passes RGB numeric values to cv2.imencode without swapping them.
    # cv2.imdecode returns those numeric values, so this array is source RGB.
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decode failed")
    return image


def extract(episode: Path, row: int, *, stride: int = 4) -> dict[str, np.ndarray]:
    with h5py.File(episode, "r") as h5:
        frames = len(h5["step"])
        if not 0 <= row < frames - 1:
            raise ValueError(f"row must be in [0, {frames - 2}], got {row}")
        if int(h5["step"][row + 1] - h5["step"][row]) != 2:
            raise ValueError("Expected adjacent saved rows to be two physics steps apart")

        result: dict[str, np.ndarray] = {
            "row": np.asarray(row, dtype=np.int64),
            "sim_step": np.asarray(h5["step"][row], dtype=np.int64),
            "dt_s": np.asarray(2 / 120, dtype=np.float32),
            "joint_observation": np.asarray(h5["embodiment/joint"][row], dtype=np.float32),
            "ee_observation": np.asarray(h5["embodiment/ee"][row], dtype=np.float32),
            "joint_target_next": np.asarray(h5["embodiment/joint"][row + 1], dtype=np.float32),
            "ee_target_next": np.asarray(h5["embodiment/ee"][row + 1], dtype=np.float32),
        }
        for camera in ("head", "wrist"):
            prefix = f"observation/{camera}"
            rgb = decode_rgb(h5[f"{prefix}/rgb"][row])
            depth = np.asarray(h5[f"{prefix}/depth"][row], dtype=np.float32)
            intrinsic = np.asarray(h5[f"{prefix}/intrinsic"][row], dtype=np.float32)
            pose = np.asarray(h5[f"{prefix}/pose_w_opengl"][row], dtype=np.float32)
            xyz, pixels = unproject_opengl(depth, intrinsic, pose, stride=stride, max_depth_m=3.0)
            if rgb.shape[:2] != depth.shape[:2] or not len(xyz):
                raise ValueError(f"{camera} RGB/depth shape mismatch or empty cloud")
            result[f"{camera}_rgb"] = rgb
            result[f"{camera}_depth_m"] = depth
            result[f"{camera}_intrinsic"] = intrinsic
            result[f"{camera}_pose_w_opengl"] = pose
            result[f"{camera}_xyz_w_m"] = xyz
            result[f"{camera}_point_rgb"] = rgb.reshape(-1, 3)[pixels]
        for side in ("left", "right"):
            prefix = f"tactile/{side}_tactile"
            result[f"{side}_tactile_rgb"] = decode_rgb(h5[f"{prefix}/rgb"][row])
            result[f"{side}_press_depth_mm"] = np.asarray(h5[f"{prefix}/press_depth"][row], dtype=np.float32)
            result[f"{side}_marker"] = np.asarray(h5[f"{prefix}/marker"][row], dtype=np.float32)
            result[f"{side}_pose"] = np.asarray(h5[f"{prefix}/pose"][row], dtype=np.float32)
    # Isaac depth uses inf for pixels with no geometry; those pixels are masked
    # by unproject_opengl and should remain visible in the raw depth array.
    if any(not np.isfinite(value).all() for key, value in result.items()
           if value.dtype.kind == "f" and not key.endswith("_depth_m")):
        raise ValueError("Sample contains non-finite numeric values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    sample = extract(args.episode, args.row, stride=args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **sample)
    summary = {
        "source": str(args.episode.resolve()),
        "row": int(sample["row"]),
        "sim_step": int(sample["sim_step"]),
        "dt_s": float(sample["dt_s"]),
        "camera_world_cloud_points": {name: len(sample[f"{name}_xyz_w_m"]) for name in ("head", "wrist")},
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in sample.items()},
        "action_semantics": "next saved joint/EEF observation; no independently recorded motor command",
        "tactile_semantics": "RGB, marker and gel press depth in mm; no force vector or peg/socket contact truth",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("row", "sim_step", "dt_s", "camera_world_cloud_points")}))


if __name__ == "__main__":
    main()
