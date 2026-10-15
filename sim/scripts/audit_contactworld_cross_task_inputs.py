#!/usr/bin/env python3
"""Exercise exact training transforms for one reviewed cross-task dataset.

The audit intentionally runs before the matrix launcher accepts a dataset.  It
checks that diagnostic TacFF videos remain masked and that each configuration
receives exactly the action/state/cloud/tactile arrays it claims to use.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from contactworld_cross_task_common import TASKS, get_task
from openpi.training import config as training_config
from openpi.training import data_loader

SUFFIXES = ("01_rgb", "02_fused_pc", "07_fused_pc_base_pooled", "10_fused_pc_base_map_matched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec, root = get_task(args.task), args.dataset.resolve()
    manifest = json.loads((root / "spatial/manifest.json").read_text())
    if manifest.get("format") != "contactworld-spatial-v3-bilateral" or manifest.get("task") != spec.key:
        raise ValueError("Wrong dataset sidecar contract")
    train = [row for row in manifest["episodes"] if row["split"] == "train"]
    if not train:
        raise ValueError("No train episodes in manifest")
    frame = min(2, int(train[0]["length"]) - 16)
    if frame < 0:
        raise ValueError("First train episode shorter than action horizon")
    rows = []
    for suffix in SUFFIXES:
        name = "pi0_cw3_%s_%s" % (spec.key, suffix)
        config = training_config.get_config(name)
        factory = dataclasses.replace(config.data, repo_id=str(root),
            sidecar=dataclasses.replace(config.data.sidecar, root=str(root)))
        settings = factory.create(config.assets_dirs, config.model)
        raw = data_loader.create_torch_dataset(settings, config.model.action_horizon, config.model)
        transformed = data_loader.transform_dataset(raw, settings, skip_norm_stats=True)
        item = transformed[frame]
        if np.asarray(item["state"]).shape[-1] < 18 or np.asarray(item["actions"]).shape != (16, config.model.action_dim):
            raise ValueError("%s: padded model state/action shape is wrong" % name)
        if not np.all(item["image_mask"]["base_0_rgb"]) or not np.all(item["image_mask"]["left_wrist_0_rgb"]) or np.any(item["image_mask"]["right_wrist_0_rgb"]):
            raise ValueError("%s: RGB image masking is wrong" % name)
        spatial = item.get("spatial")
        visual = bool(getattr(config.model, "use_visual", False))
        tactile = bool(getattr(config.model, "use_tactile", False))
        if visual:
            cloud = spatial["visual"]["xyz_m"]
            if cloud.shape != (4096, 3) or not np.isfinite(cloud).all():
                raise ValueError("%s: cloud input invalid" % name)
        if tactile:
            field = spatial["tactile"]
            if (field["xyz_m"].shape != (280, 3) or field["force"].shape != (280, 3) or
                    not np.array_equal(field["finger_id"], np.repeat(np.array([0, 1], np.int32), 140))):
                raise ValueError("%s: bilateral tactile input invalid" % name)
        rows.append({"config": name, "action_source_dim": spec.action_dim,
                     "model_action_dim": int(config.model.action_dim), "state_source_dim": 18,
                     "cloud_points": 4096 if visual else 0, "left_taxels": 140 if tactile else 0,
                     "right_taxels": 140 if tactile else 0, "third_image_masked": True,
                     "prompt": spec.prompt})
    output = args.output or root / "cross_task_model_input_audit.json"
    output.write_text(json.dumps({"status": "passed", "task": spec.key, "dataset": str(root),
                                  "frame": frame, "configs": rows}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
