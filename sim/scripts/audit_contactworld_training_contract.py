#!/usr/bin/env python3
"""Exercise train/val/test Zarr loading and report supported modality semantics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openpi.spatial_dataset.contactworld import ContactWorldDataset, TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"status": "running", "action_horizon": 16, "splits": {},
              "modality_contract": {
                  "front_rgb,wrist_rgb": "uint8 HWC, 256x256; released float [0,1] converted with round(255*x)",
                  "tactile_rgb_right": "uint8 HWC, 320x240; simulated GelSight rendering",
                  "state": "float32 18D: DOF positions[0:9], velocities[9:18]",
                  "pointcloud_base_xyzrgb": "float32 1024x6, first 3 meters in fixed robot base using measured front-camera source correction; RGB [0,1]",
                  "tactile_force_field_local": "signed float32 10x14x3 (normal and two local tangential components); physical Newton calibration unverified",
                  "actions": "float32 horizon x 6: original un-clipped relative task-space command; pad zeros only where action_valid is false",
                  "action_valid": "bool horizon; never crosses episode end",
                  "task_id,episode_index,frame_index": "bookkeeping only; exclude from policy observations",
              },
              "limitations": ["Released dataset has no per-frame depth or collection-time camera matrices",
                              "Historical collection frame time is unverified; horizon uses indices, not claimed seconds",
                              "Wrist RGB extrinsic is not calibrated by this dataset adapter",
                              "No episode success field; endpoint pose criterion is not an official success label"]}
    assigned = {}
    for split in ("train", "val", "test"):
        ds = ContactWorldDataset(args.data, args.geometry, split=split, action_horizon=16)
        assert len(ds) > 0
        assigned[split] = {(task, ep) for task, ep, _ in ds.episodes}
        task_counts = {task: sum(t == task for t, _, _ in ds.episodes) for task in TASKS}
        assert all(n > 0 for n in task_counts.values())
        sample = ds[0]
        assert sample["frame_index"] == 0
        assert sample["front_rgb"].shape == (256, 256, 3)
        assert sample["wrist_rgb"].shape == (256, 256, 3)
        assert sample["tactile_rgb_right"].shape == (320, 240, 3)
        assert sample["state"].shape == (18,)
        assert sample["pointcloud_base_xyzrgb"].shape == (1024, 6)
        assert sample["tactile_force_field_local"].shape == (10, 14, 3)
        assert sample["actions"].shape == (16, 6)
        last = ds[-1]
        assert last["frame_index"] == ds.episodes[-1][2] - 1
        assert int(last["action_valid"].sum()) == 1
        assert np.all(last["actions"][1:] == 0)
        batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
        assert batch["front_rgb"].shape == (2, 256, 256, 3)
        assert batch["actions"].shape == (2, 16, 6)
        assert batch["action_valid"].dtype == torch.bool
        report["splits"][split] = {"episodes": len(ds.episodes), "frames": len(ds),
                                   "episodes_by_task": task_counts,
                                   "sample_shapes": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")},
                                   "batch_shapes": {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")},
                                   "last_action_valid_count": int(last["action_valid"].sum())}
    assert not (assigned["train"] & assigned["val"] or assigned["train"] & assigned["test"] or assigned["val"] & assigned["test"])
    report["raw_data_checks"] = {}
    for task in TASKS:
        data = ds.groups[task]["data"]
        actions = np.asarray(data["action"][:], dtype=np.float32)
        force_sample = np.asarray(data["tactile_force_field_right"][::100], dtype=np.float32)
        if actions.shape[1] != 6 or force_sample.shape[1:] != (10, 14, 3):
            raise ValueError(f"Unexpected raw action or force shape: {task}")
        if not np.isfinite(actions).all() or not np.isfinite(force_sample).all():
            raise ValueError(f"Nonfinite raw action or sampled force: {task}")
        report["raw_data_checks"][task] = {
            "action_min_by_axis": actions.min(axis=0).tolist(),
            "action_max_by_axis": actions.max(axis=0).tolist(),
            "fraction_action_components_abs_gt_one": float(np.mean(np.abs(actions) > 1)),
            "force_sample_stride": 100,
            "force_component_min": force_sample.min(axis=(0, 1, 2)).tolist(),
            "force_component_max": force_sample.max(axis=(0, 1, 2)).tolist(),
            "published_fields": sorted(data.array_keys()),
        }
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
