#!/usr/bin/env python3
"""Compute OpenPI state/action stats for the audited UniVTAC clean split.

Stats are computed after the 7-joint delta transform. All modality ablations
share this same action/state contract, so no world-cloud decoding is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    assets = args.assets_dir.resolve()
    if assets.exists():
        parser.error(f"Assets directory exists: {assets}")
    model_config = Pi0Config(action_horizon=50)
    data_config = config.LeRobotUniVTACDataConfig(repo_id=str(root)).create(
        root / "unused_assets", model_config,
    )
    raw = data_loader.create_torch_dataset(data_config, 50, model_config)
    transform = transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ])
    accumulators = {key: normalize.RunningStats() for key in ("state", "actions")}
    for index in range(len(raw)):
        sample = transform(raw[index])
        for key, accumulator in accumulators.items():
            accumulator.update(np.asarray(sample[key]))
        if index % 100 == 0:
            print(f"Stats {index + 1}/{len(raw)}", flush=True)
    stats = {key: accumulator.get_statistics() for key, accumulator in accumulators.items()}
    normalize.save(assets, stats)
    reloaded = normalize.load(assets)
    if any(not np.allclose(stats[key].mean, reloaded[key].mean) for key in stats):
        raise ValueError("Normalization stats roundtrip failed")
    report = {
        "dataset_root": str(root),
        "assets_dir": str(assets),
        "training_rows": len(raw),
        "action_horizon": 50,
        "contract": "state=9D absolute joint; action=7D joint delta + 2D absolute gripper",
        "state_mean": stats["state"].mean.tolist(),
        "state_std": stats["state"].std.tolist(),
        "action_mean": stats["actions"].mean.tolist(),
        "action_std": stats["actions"].std.tolist(),
        "roundtrip_pass": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("training_rows", "action_horizon", "roundtrip_pass")}))


if __name__ == "__main__":
    main()
