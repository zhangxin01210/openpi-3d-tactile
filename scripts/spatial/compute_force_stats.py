"""Compute training-set tactile force scale candidates for spatial/v1.

This script is intentionally standalone.  It is not imported by model code and
does not affect training unless a user copies one of the reported values into a
StructuredSpatialEncoderConfig(force_scale=...).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute P50/P90/P95/P99/max for tactile_force_norm in spatial/v1."
    )
    parser.add_argument(
        "spatial_root",
        type=Path,
        help="Path to a spatial sidecar root, for example data/press_0828_17/spatial/v1.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=(50.0, 90.0, 95.0, 99.0),
    )
    return parser.parse_args()


def iter_tactile_force_norm_arrays(spatial_root: Path):
    manifest_path = spatial_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["storage"]["episodes"]

    for episode in episodes:
        episode_root = spatial_root / episode["path"]
        episode_manifest = json.loads((episode_root / "manifest.json").read_text(encoding="utf-8"))
        for shard in episode_manifest["shards"]:
            force_file = shard["files"]["tactile_force_norm"]["file"]
            yield np.load(episode_root / force_file, mmap_mode="r")


def main() -> None:
    args = parse_args()
    spatial_root = args.spatial_root.expanduser().resolve()
    if not spatial_root.is_dir():
        raise FileNotFoundError(spatial_root)

    values = []
    total_values = 0
    for array in iter_tactile_force_norm_arrays(spatial_root):
        finite = np.asarray(array[np.isfinite(array)], dtype=np.float32)
        values.append(finite)
        total_values += int(finite.size)

    if not values:
        raise RuntimeError(f"No tactile_force_norm shards found under {spatial_root}")

    force_norm = np.concatenate(values)
    result = {
        "spatial_root": str(spatial_root),
        "count": total_values,
        "percentiles": {
            f"p{percentile:g}": float(np.percentile(force_norm, percentile))
            for percentile in args.percentiles
        },
        "max": float(np.max(force_norm)),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
