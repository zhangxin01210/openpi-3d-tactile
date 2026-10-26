#!/usr/bin/env python3
"""Audit a few Hand/Funnel/Flask frames without creating labels or rewriting data.

Example (computer 4):

    PYTHONPATH=src python scripts/object_geometry/audit_object_centric_geometry.py \
      --dataset-root data/1018_peg_all --object-version v1_hand_scene_q2 \
      --spatial-version v1_front_candidate --episodes 23,38,65 --frames-per-episode 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.object_geometry.coordinate_diagnostic import frame_diagnostic
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.spatial_dataset.derived import SpatialDerivedDataset


def _parse_episodes(value: str | None) -> set[int] | None:
    if value is None:
        return None
    result: set[int] = set()
    for item in value.split(","):
        bounds = item.split("-", 1)
        if len(bounds) == 1:
            result.add(int(bounds[0]))
        else:
            result.update(range(int(bounds[0]), int(bounds[1]) + 1))
    return result


def _sample_frames(frame_ids: list[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("frames-per-episode must be positive")
    if len(frame_ids) <= count:
        return frame_ids
    indices = np.linspace(0, len(frame_ids) - 1, count, dtype=np.int64)
    return [frame_ids[int(index)] for index in indices]


def audit(
    *,
    dataset_root: Path,
    spatial_version: str,
    object_version: str,
    episodes: set[int] | None,
    frames_per_episode: int,
    max_abs_m: float,
    max_jump_m: float,
) -> dict[str, Any]:
    if max_abs_m <= 0 or max_jump_m <= 0:
        raise ValueError("max_abs_m and max_jump_m must be positive")
    spatial = SpatialDerivedDataset(dataset_root, version=spatial_version)
    geometry = ObjectGeometryDerivedDataset(dataset_root, version=object_version)
    selected = set(geometry.episode_ids) if episodes is None else set(episodes)
    unknown = selected - set(geometry.episode_ids)
    if unknown:
        raise ValueError(f"Episodes absent from object geometry sidecar: {sorted(unknown)}")

    records: list[dict[str, Any]] = []
    coordinate_warnings = 0
    jump_records = 0
    jump_warnings = 0
    previous_centroids: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    for episode in sorted(selected):
        frame_ids = sorted(frame for ep, frame in geometry._records if int(ep) == int(episode))  # noqa: SLF001
        for frame in _sample_frames(frame_ids, frames_per_episode):
            spatial_sample = spatial.get(episode_index=episode, frame_index=frame)
            geometry_sample = geometry.get(episode_index=episode, frame_index=frame)
            current = frame_diagnostic(
                hand_xyz=spatial_sample.tactile_xyz_m,
                hand_mask=np.isfinite(spatial_sample.tactile_xyz_m).all(axis=-1),
                funnel_xyz=geometry_sample.manipulated_xyz_m,
                funnel_mask=geometry_sample.manipulated_mask,
                flask_xyz=geometry_sample.target_xyz_m,
                flask_mask=geometry_sample.target_mask,
                max_abs_m=max_abs_m,
            )
            current.update({"episode_index": episode, "frame_index": frame})
            if current["coordinate_warning"]:
                coordinate_warnings += 1

            hand_centroid = current["hand"]["centroid_m"]
            funnel_centroid = current["funnel"]["centroid_m"]
            flask_centroid = current["flask"]["centroid_m"]
            previous = previous_centroids.get(episode)
            if previous is not None and all(value is not None for value in (hand_centroid, funnel_centroid, flask_centroid)):
                previous_frame, previous_hand, previous_funnel, previous_flask = previous
                current["frame_delta"] = frame - previous_frame
                current["centroid_delta_m"] = {
                    "hand": float(np.linalg.norm(np.asarray(hand_centroid) - previous_hand)),
                    "funnel": float(np.linalg.norm(np.asarray(funnel_centroid) - previous_funnel)),
                    "flask": float(np.linalg.norm(np.asarray(flask_centroid) - previous_flask)),
                }
                current["centroid_jump_warning"] = any(
                    delta > max_jump_m for delta in current["centroid_delta_m"].values()
                )
                if current["centroid_jump_warning"]:
                    jump_warnings += 1
                jump_records += 1
            if all(value is not None for value in (hand_centroid, funnel_centroid, flask_centroid)):
                previous_centroids[episode] = (
                    frame,
                    np.asarray(hand_centroid),
                    np.asarray(funnel_centroid),
                    np.asarray(flask_centroid),
                )
            records.append(current)

    return {
        "coordinate_frame": "base_link",
        "units": "meters",
        "spatial_version": spatial_version,
        "object_geometry_version": object_version,
        "episodes": sorted(selected),
        "sampled_frame_count": len(records),
        "coordinate_warning_count": coordinate_warnings,
        "centroid_jump_comparisons": jump_records,
        "centroid_jump_warning_count": jump_warnings,
        "max_jump_m": max_jump_m,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spatial-version", default="v1_front_candidate")
    parser.add_argument("--object-version", default="v1_hand_scene_q2")
    parser.add_argument("--episodes")
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--max-abs-m", type=float, default=10.0)
    parser.add_argument("--max-jump-m", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        dataset_root=args.dataset_root.expanduser().resolve(),
        spatial_version=args.spatial_version,
        object_version=args.object_version,
        episodes=_parse_episodes(args.episodes),
        frames_per_episode=args.frames_per_episode,
        max_abs_m=args.max_abs_m,
        max_jump_m=args.max_jump_m,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.expanduser().resolve().write_text(text + "\n")
        print(json.dumps({"output": str(args.output.expanduser().resolve()), "sampled_frame_count": report["sampled_frame_count"]}))


if __name__ == "__main__":
    main()
