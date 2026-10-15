#!/usr/bin/env python3
"""Audit every UniVTAC LeRobot row through the on-demand world-cloud join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.models.spatial_pi0_config import StructuredSpatialPi0Config
from openpi.training import config
from openpi.training import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    model_config = StructuredSpatialPi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        use_visual=True, use_tactile=False, visual_points=2048,
    )
    settings = config.LeRobotUniVTACDataConfig(
        repo_id=str(root),
        pointcloud=config.UniVTACPointcloudDataConfig(dataset_root=str(root)),
    ).create(root / "unused_assets", model_config)
    dataset = data_loader.create_torch_dataset(settings, model_config.action_horizon, model_config)
    xyz_min = np.full(3, np.inf)
    xyz_max = np.full(3, -np.inf)
    episode_counts: dict[int, int] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        xyz = sample["spatial"]["visual"]["xyz_m"]
        rgb = sample["spatial"]["visual"]["rgb"]
        if xyz.shape != (2048, 3) or rgb.shape != (2048, 3) or not np.isfinite(xyz).all():
            raise ValueError(f"Bad pointcloud at dataset row {index}")
        xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
        xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
        episode = int(np.asarray(sample["episode_index"]))
        episode_counts[episode] = episode_counts.get(episode, 0) + 1
        if index % 100 == 0:
            print(f"Checked {index + 1}/{len(dataset)}", flush=True)
    report = {
        "dataset_root": str(root),
        "rows": len(dataset),
        "points_per_row": 2048,
        "episode_row_counts": episode_counts,
        "world_xyz_min_m": xyz_min.tolist(),
        "world_xyz_max_m": xyz_max.tolist(),
        "all_rows_step_matched_and_finite": True,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
