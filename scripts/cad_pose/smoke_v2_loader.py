#!/usr/bin/env python3
"""Read-only V2 native DataLoader smoke and non-geometry comparison against v1."""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from openpi.cad_pose.v2_native import V2CandidateReader
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.training import config
from openpi.training import data_loader

TASKS = {
    "peg_all": "1018_peg_all",
    "press_the_button": "1028_press_button",
    "yellow_socket": "1030_yellow_socket",
    "eraser": "1102_eraser",
}


def equal(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"{path}: keys differ")
        for key in a:
            equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, str) or isinstance(b, str):
        if a != b:
            raise ValueError(f"{path}: string differs")
    else:
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        if hasattr(b, "detach"):
            b = b.detach().cpu().numpy()
        x, y = np.asarray(a), np.asarray(b)
        same = (
            np.array_equal(x, y, equal_nan=True)
            if x.dtype.kind in "fc" and y.dtype.kind in "fc"
            else np.array_equal(x, y)
        )
        if not same:
            raise ValueError(f"{path}: value differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    p.add_argument("--candidate-root", type=Path, default=Path("output/cad_pose_rgbd_v2/geometry_full_candidate"))
    a = p.parse_args()
    for task in a.tasks:
        cfg = config.get_config(f"pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/{task}")
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        geo = data.object_geometry
        pred = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
        old_geo = dataclasses.replace(
            geo, version="v1", require_pose_prediction=False, pose_prediction_schema="v1", pose_v2_cache_root=None
        )
        base = data_loader.create_torch_dataset(
            dataclasses.replace(data, object_geometry=old_geo), cfg.model.action_horizon, cfg.model
        )
        native = ObjectGeometryDerivedDataset(geo.dataset_root, version="pretrained_pose_v2")
        if task == "press_the_button" and len(native.episode_ids) != 88:
            raise ValueError(f"cleaned Button expected 88 episodes; got {len(native.episode_ids)}")
        if len(pred) != len(base) or len(native) != len(pred):
            raise ValueError(f"{task}: dataset length mismatch")
        candidate = V2CandidateReader(a.candidate_root, TASKS[task])
        for index in sorted({0, len(pred) // 2, len(pred) - 1}):
            new, old = dict(pred[index]), dict(base[index])
            spatial_new, spatial_old = dict(new.pop("spatial")), dict(old.pop("spatial"))
            geometry = spatial_new.pop("object_geometry")
            spatial_old.pop("object_geometry")
            equal(new, old)
            equal(spatial_new, spatial_old, "spatial")
            ep, frame = int(new["episode_index"]), int(new["frame_index"])
            expected = candidate.get(episode_index=ep, frame_index=frame)
            for role in ("manipulated", "target"):
                if not np.array_equal(geometry[f"{role}_xyz_m"], getattr(expected, f"{role}_xyz_m")):
                    raise ValueError(f"{task}: V2 predicted geometry mismatch for {role}")
                if not np.asarray(geometry[f"{role}_mask"], bool).all():
                    raise ValueError(f"{task}: finite-prediction mask mismatch")
        print(
            f"[PASS] {task}: {len(native.episode_ids)} episodes, {len(pred)} frames; sampled V2 join and unchanged fields",
            flush=True,
        )


if __name__ == "__main__":
    main()
