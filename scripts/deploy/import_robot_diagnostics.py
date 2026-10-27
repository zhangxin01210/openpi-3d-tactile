#!/usr/bin/env python3
"""Adapt the existing UR7e/XHand diagnostic observation files to replay packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("diagnostic_run", type=Path, help="One run directory created by the existing deploy client.")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run = a.diagnostic_run.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "recording", "interrupted", "error"}:
        raise ValueError(f"Unrecognized old diagnostic run status: {manifest.get('status')!r}")
    inputs = sorted((run / "inference").glob("query_*_input.npz"))
    if not inputs:
        raise FileNotFoundError(f"No inference/query_*_input.npz files under {run}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty replay directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    required = {
        "front_rgb": "rgb_cam_front",
        "left_rgb": "rgb_cam_left",
        "right_rgb": "rgb_cam_right",
        "front_depth_raw": "depth_cam_front",
        "robot_state": "state",
        "frame_index": "frame_index",
        "timestamp_s": "timestamp_s",
    }
    frames = []
    previous = -1
    for path in inputs:
        with np.load(path, allow_pickle=False) as source:
            absent = [key for key in required.values() if key not in source.files]
            if absent:
                raise KeyError(f"{path} is missing observation arrays required by the perception adapter: {absent}")
            values = {target: np.asarray(source[key]).copy() for target, key in required.items()}
        frame = int(np.asarray(values["frame_index"]).reshape(()))
        if frame <= previous:
            raise ValueError(f"Recorded frame indices are not strictly increasing: {frame} after {previous}")
        previous = frame
        for name in ("front_rgb", "left_rgb", "right_rgb"):
            image = values[name]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{path}: {name} must be uint8 HxWx3 RGB, got {image.shape} {image.dtype}")
        shapes = {tuple(values[name].shape[:2]) for name in ("front_rgb", "left_rgb", "right_rgb")}
        if len(shapes) != 1:
            raise ValueError(f"{path}: all three RGB images must share resolution, got {sorted(shapes)}")
        if values["front_depth_raw"].shape != values["front_rgb"].shape[:2]:
            raise ValueError(f"{path}: front depth and RGB are not aligned: {values['front_depth_raw'].shape}")
        state = np.asarray(values["robot_state"], dtype=np.float32).reshape(-1)
        if state.shape != (1972,) or not np.isfinite(state).all():
            raise ValueError(f"{path}: expected finite UR7e+XHand robot state [1972], got {state.shape}")
        values["robot_state"] = state
        if not np.isfinite(float(np.asarray(values["timestamp_s"]).reshape(()))):
            raise ValueError(f"{path}: timestamp_s is not finite")
        frames.append((frame, values, path.name))

    for frame, values, _source_name in frames:
        target = out / f"frame_{frame:06d}.npz"
        tmp = target.with_suffix(".npz.tmp")
        with tmp.open("xb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
        tmp.replace(target)
    summary = {
        "status": "PASS",
        "source_diagnostic_run": str(run),
        "source_queries": len(frames),
        "first_frame_index": frames[0][0],
        "last_frame_index": frames[-1][0],
        "output_replay_dir": str(out),
        "actions_imported": False,
        "action_commands_sent_by_adapter": False,
        "note": "One packet per old policy query. Missing depth, image schema, state size, or frame order is an error; this adapter never synthesizes observations.",
    }
    (out / "import_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
