#!/usr/bin/env python3
"""Read-only pilot: q2 geometric gaps and XHand contact-signal observability.

This is NOT an accuracy oracle, contact detector or insertion-success labeler.
Requires the repository's own derived readers plus numpy/scipy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.spatial_dataset.derived import SpatialDerivedDataset


def parse_episodes(text: str, available: tuple[int, ...]) -> list[int]:
    if text.lower() == "all":
        return list(available)
    result = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"Reversed episode range: {part}")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    absent = sorted(result - set(available))
    if absent:
        raise ValueError(f"Episodes not present in q2 sidecar: {absent}")
    return sorted(result)


def finite_points(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    return xyz[mask & np.isfinite(xyz).all(axis=1)]


def point_distance_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Nearest *surface-sample* separation; does not diagnose physical contact."""
    if len(a) == 0 or len(b) == 0:
        return math.nan, math.nan
    distance, _ = cKDTree(b).query(a, k=1, workers=1)
    return float(np.min(distance)), float(np.percentile(distance, 10))


def stats_safe(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {"count": len(x), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def json_number(x: float | None) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def process(root: Path, spatial_version: str, geo_version: str, episodes_arg: str,
            stride: int, output: Path) -> dict:
    spatial = SpatialDerivedDataset(root, version=spatial_version)
    geo = ObjectGeometryDerivedDataset(root, version=geo_version)
    episodes = parse_episodes(episodes_arg, geo.episode_ids)
    if stride < 1:
        raise ValueError("stride must be positive")
    output.mkdir(parents=True, exist_ok=True)
    per_ep = defaultdict(list)
    records = []
    missing_spatial = []
    for ep in episodes:
        prev_frame, prev_center, prev_time = None, None, None
        all_frames = geo.frame_ids(ep)
        for frame in all_frames[::stride]:
            try:
                g = geo.get(episode_index=ep, frame_index=frame)
                s = spatial.get(episode_index=ep, frame_index=frame)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                missing_spatial.append({"episode": ep, "frame": frame, "error": repr(exc)})
                continue
            funnel = finite_points(g.manipulated_xyz_m, g.manipulated_mask)
            flask = finite_points(g.target_xyz_m, g.target_mask)
            touch = np.asarray(s.tactile_force_norm, dtype=np.float32)
            if touch.shape != (600,) or not np.isfinite(touch).all():
                raise ValueError(f"Invalid tactile force: episode {ep}, frame {frame}: {touch.shape}")
            hs = g.hand_scene
            delta = np.asarray(hs.grasp_short_delta_force_norm, dtype=np.float32) if hs is not None else None
            if delta is not None and delta.shape != (600,):
                raise ValueError(f"Invalid grasp delta at episode {ep}, frame {frame}")
            min_gap, p10_gap = point_distance_stats(funnel, flask)
            tactile_xyz = finite_points(s.tactile_xyz_m, np.ones(600, bool))
            hand_gap, _ = point_distance_stats(tactile_xyz, funnel)
            center = np.mean(funnel, axis=0) if len(funnel) else None
            center_step_m = math.nan
            center_speed_m_s = math.nan
            if center is not None and prev_center is not None and prev_frame is not None:
                center_step_m = float(np.linalg.norm(center - prev_center))
                if prev_time is not None and s.timestamp_s > prev_time:
                    center_speed_m_s = center_step_m / (s.timestamp_s - prev_time)
            row = {
                "episode": int(ep), "frame": int(frame), "timestamp_s": float(s.timestamp_s),
                "progress_fraction": float(frame / max(1, all_frames[-1])),
                "funnel_valid_points": int(len(funnel)), "flask_valid_points": int(len(flask)),
                "funnel_relation_valid": bool(hs.manipulated_relation_valid) if hs else False,
                "flask_relation_valid": bool(hs.target_relation_valid) if hs else False,
                "geometry_quality_structural_funnel": float(hs.manipulated_geometry_quality) if hs else 0.,
                "geometry_quality_structural_flask": float(hs.target_geometry_quality) if hs else 0.,
                "funnel_flask_nearest_surface_m": min_gap,
                "funnel_flask_p10_nearest_surface_m": p10_gap,
                "hand_funnel_nearest_surface_m": hand_gap,
                "funnel_centroid_shift_since_last_sample_m": center_step_m,
                "funnel_centroid_speed_m_s": center_speed_m_s,
                "tactile_force_mean": float(np.mean(touch)),
                "tactile_force_max": float(np.max(touch)),
                "tactile_short_delta_mean_abs": float(np.mean(np.abs(delta))) if delta is not None else math.nan,
                "tactile_short_delta_max_abs": float(np.max(np.abs(delta))) if delta is not None else math.nan,
            }
            for finger in range(5):
                selection = np.asarray(s.finger_id) == finger
                if not selection.any():
                    raise ValueError(f"No taxels for finger {finger}, episode {ep} frame {frame}")
                row[f"finger_{finger}_force_mean"] = float(np.mean(touch[selection]))
                if delta is not None:
                    row[f"finger_{finger}_short_delta_mean_abs"] = float(np.mean(np.abs(delta[selection])))
            records.append(row)
            per_ep[ep].append(row)
            prev_frame, prev_center, prev_time = frame, center, float(s.timestamp_s)
        print(f"episode={ep:03d} sampled={len(per_ep[ep])} / {len(all_frames)}", flush=True)

    if records:
        path = output / "per_frame.csv"
        fields = list(records[0])
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    else:
        path = None

    summaries = []
    for ep, rows in per_ep.items():
        def top(metric: str, reverse: bool, n: int = 6):
            valid = [r for r in rows if np.isfinite(r.get(metric, math.nan))]
            valid.sort(key=lambda r: r[metric], reverse=reverse)
            return [{"frame": x["frame"], "value": x[metric]} for x in valid[:n]]

        late = [r for r in rows if r["progress_fraction"] >= 0.8]
        summary = {
            "episode": ep, "sampled_frames": len(rows),
            "valid_both_count": sum(r["funnel_valid_points"] > 0 and r["flask_valid_points"] > 0 for r in rows),
            "nearest_surface_distance_all_m": stats_safe([r["funnel_flask_nearest_surface_m"] for r in rows]),
            "nearest_surface_distance_last_20pct_m": stats_safe([r["funnel_flask_nearest_surface_m"] for r in late]),
            "tactile_delta_last_20pct": stats_safe([r["tactile_short_delta_mean_abs"] for r in late]),
            "largest_tactile_changes": top("tactile_short_delta_mean_abs", True),
            "largest_funnel_centroid_speed": top("funnel_centroid_speed_m_s", True),
            "smallest_funnel_flask_surface_distances": top("funnel_flask_nearest_surface_m", False),
        }
        summaries.append(summary)
    report = {
        "disclaimer": (
            "Surface distance is between sampled CAD surfaces, not signed penetration or physical contact. "
            "Final 20% is temporal position, not an insertion label. "
            "Tactile on XHand fingertips senses hand-funnel contact, not flask contact directly. "
            "Sidecar confidence/geometry_quality indicate structural validity, not metric accuracy. "
            "No inference about causation or teacher accuracy is supported by this report."
        ),
        "root": str(root.resolve()), "spatial_version": spatial_version, "geometry_version": geo_version,
        "stride": stride, "sampled_frames": len(records), "episodes": summaries,
        "missing_spatial_count": len(missing_spatial), "missing_spatial_examples": missing_spatial[:20],
        "csv": str(path) if path else None,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}, {path}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("data/1018_peg_all"))
    p.add_argument("--spatial-version", default="v1_front_candidate")
    p.add_argument("--geometry-version", default="v1_hand_scene_q2")
    p.add_argument("--episodes", default="23,38,65", help="e.g. 23,38,65 or 0-69 or all")
    p.add_argument("--stride", type=int, default=5, help="1 scans every frame")
    p.add_argument("--output", type=Path, default=Path("scratch/probe_geometry_tactile_signals"))
    args = p.parse_args()
    process(args.dataset_root, args.spatial_version, args.geometry_version,
            args.episodes, args.stride, args.output)


if __name__ == "__main__":
    main()
