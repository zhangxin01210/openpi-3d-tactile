#!/usr/bin/env python3
"""Measure raw front-depth stability on a user-selected stationary planar ROI."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_frames(value):
    if ":" in value:
        parts = [int(part) for part in value.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("--frames range must be start:stop[:step], stop exclusive")
        frames = list(range(*parts))
    else:
        frames = [int(part) for part in value.split(",") if part.strip()]
    if len(frames) < 3 or min(frames) < 0 or len(set(frames)) != len(frames):
        raise ValueError("--frames requires at least three unique nonnegative indices")
    if len(frames) > 300:
        raise ValueError("Use at most 300 frames to keep the ROI stack bounded")
    return frames


def fit_plane(depth_m, intrinsics, x0, y0):
    """Robust approximate point-to-plane distances in depth optical meters."""
    v, u = np.nonzero(np.isfinite(depth_m) & (depth_m > 0))
    if len(u) < 100:
        return {"count": int(len(u)), "abs_median_mm": None, "abs_p90_mm": None}
    if len(u) > 20000:
        sample = np.linspace(0, len(u) - 1, 20000).astype(int)
        u, v = u[sample], v[sample]
    z = depth_m[v, u].astype(np.float64)
    x = (u + x0 - intrinsics.cx) * z / intrinsics.fx
    y = (v + y0 - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))
    keep = np.ones(len(points), bool)
    for _ in range(2):
        center = np.mean(points[keep], axis=0)
        _, _, basis = np.linalg.svd(points[keep] - center, full_matrices=False)
        distance = (points - center) @ basis[-1]
        keep = np.abs(distance) <= np.percentile(np.abs(distance), 90)
    center = np.mean(points[keep], axis=0)
    _, _, basis = np.linalg.svd(points[keep] - center, full_matrices=False)
    distance = (points - center) @ basis[-1]
    mm = np.abs(distance[keep]) * 1000
    return {
        "count": int(keep.sum()),
        "abs_median_mm": float(np.median(mm)),
        "abs_p90_mm": float(np.percentile(mm, 90)),
    }


def temporal_metrics(stack):
    """Compare every frame against per-pixel temporal median, ignoring depth holes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reference = np.nanmedian(stack, axis=0)
    diff = stack - reference[None]
    valid = np.isfinite(diff)
    per_frame = []
    for frame_diff, frame_valid in zip(diff, valid, strict=True):
        values = frame_diff[frame_valid] * 1000
        per_frame.append({
            "common_pixels": int(len(values)),
            "drift_median_mm": float(np.median(values)) if len(values) else None,
            "temporal_abs_p90_mm": float(np.percentile(np.abs(values), 90)) if len(values) else None,
        })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        temporal_mad = np.nanmedian(np.abs(diff), axis=0) * 1000
    reliable = np.count_nonzero(valid, axis=0) >= 3
    return per_frame, temporal_mad, reliable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", required=True, help="Comma list or Python-style start:stop[:step]")
    parser.add_argument("--roi", required=True, help="x0,y0,x1,y1 in raw depth pixels; x1/y1 exclusive")
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = parse_frames(args.frames)
    roi = [int(part) for part in args.roi.split(",")]
    if len(roi) != 4:
        raise ValueError("--roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = roi
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        raise ValueError("Invalid ROI or depth range")
    if (x1 - x0) * (y1 - y0) * len(frames) > 30_000_000:
        raise ValueError("ROI and frame count are too large; use a smaller stationary patch")
    dataset = RawSpatialDataset(args.dataset.expanduser().resolve(), episode=args.episode)
    pre = SpatialPreprocessor.from_repo_root(repo_root=REPO_ROOT, config=make_baseline_config().with_camera_roles("front"))
    camera = pre.calibration_bundle.cameras["front"]
    arrays, timestamps = {}, {}
    for row in dataset.iter_rows(selected=frames, depth_roles=("front",)):
        frame = int(row["frame_index"])
        depth = np.asarray(row["observation.depths.cam_front"])
        if x1 > depth.shape[1] or y1 > depth.shape[0]:
            raise ValueError(f"ROI {roi} exceeds raw depth shape {depth.shape}")
        crop = depth[y0:y1, x0:x1].astype(np.float32) * camera.depth_scale_m_per_unit
        crop[(crop < args.min_depth_m) | (crop > args.max_depth_m)] = np.nan
        arrays[frame] = crop
        timestamps[frame] = float(row["timestamp"])
    stack = np.stack([arrays[frame] for frame in frames])
    temporal, mad, reliable = temporal_metrics(stack)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame_rows = []
    for frame, values in zip(frames, temporal, strict=True):
        plane = fit_plane(arrays[frame], camera.depth_intrinsics, x0, y0)
        frame_rows.append({"frame": frame, "timestamp_s": timestamps[frame],
                           "valid_pixels": int(np.isfinite(arrays[frame]).sum()),
                           **values, "plane_abs_median_mm": plane["abs_median_mm"],
                           "plane_abs_p90_mm": plane["abs_p90_mm"]})
    image = np.zeros((*mad.shape, 3), np.uint8)
    scale = np.zeros(mad.shape, np.uint8)
    scale[reliable] = np.clip(mad[reliable] * (255 / 20), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scale, cv2.COLORMAP_TURBO)
    image[reliable] = color[reliable]
    if not cv2.imwrite(str(output / "temporal_mad_mm_clipped20.png"), image):
        raise RuntimeError("Could not write temporal MAD image")
    (output / "report.json").write_text(json.dumps({
        "dataset": str(args.dataset), "episode": args.episode, "frames": frames, "roi_xyxy": roi,
        "raw_depth_scale_m_per_unit": camera.depth_scale_m_per_unit,
        "temporally_reliable_pixels": int(reliable.sum()),
        "temporal_mad_median_mm": float(np.median(mad[reliable])) if np.any(reliable) else None,
        "note": "Temporal stability/plane flatness only; neither establishes absolute depth accuracy. ROI must remain stationary, planar, and unobstructed.",
        "results": frame_rows,
    }, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"Wrote {output / 'report.json'} and {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
