#!/usr/bin/env python3
"""Feed one real UniVTAC world point cloud through spatial Pi0 forward/backward.

This uses a small random model and one data-contract sample. It validates the
visual spatial interface only; GelSight, full dataset loading and rollout are
separate gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.model import Observation
from openpi.models.spatial_pi0_config import StructuredSpatialPi0Config
from openpi.training import config
from openpi.training import data_loader


def sample_cloud(data: np.lib.npyio.NpzFile, count_per_camera: int) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    for camera in ("head", "wrist"):
        xyz = data[f"{camera}_xyz_w_m"]
        rgb = data[f"{camera}_point_rgb"]
        if len(xyz) < count_per_camera:
            raise ValueError(f"{camera} has only {len(xyz)} pointcloud samples")
        indices = np.linspace(0, len(xyz) - 1, count_per_camera, dtype=np.int64)
        points.append(xyz[indices])
        colors.append(rgb[indices])
    return np.concatenate(points).astype(np.float32), np.concatenate(colors).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("sample_npz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.sample_npz) as sample:
        row = int(sample["row"])
        xyz, rgb = sample_cloud(sample, count_per_camera=1024)

    model_config = StructuredSpatialPi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=4, dtype="float32", use_visual=True, use_tactile=False,
        visual_points=2048,
    )
    data_config = config.LeRobotUniVTACDataConfig(
        repo_id=str(args.dataset_root.resolve()),
    ).create(args.dataset_root / "unused_assets", model_config)
    raw = data_loader.create_torch_dataset(data_config, 4, model_config)
    transformed = data_loader.transform_dataset(raw, data_config, skip_norm_stats=True)
    item = dict(transformed[row])
    actions = jnp.asarray(item.pop("actions"))[None]
    item["spatial"] = {"visual": {
        "xyz_m": xyz,
        "rgb": rgb,
        "rgb_valid": np.ones(len(xyz), dtype=bool),
        "point_mask": np.ones(len(xyz), dtype=bool),
    }}
    batched = jax.tree.map(lambda value: jnp.asarray(value)[None], item)
    observation = Observation.from_dict(batched)
    model = model_config.create(jax.random.key(0))

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.key(1), observation, actions, train=False))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    gradient_paths = jax.tree_util.tree_flatten_with_path(gradients)[0]
    spatial_grads = [np.asarray(value) for path, value in gradient_paths if "spatial" in str(path).lower()]
    if not np.isfinite(float(loss)) or not spatial_grads:
        raise ValueError("No finite loss or spatial gradients")
    if any(not np.isfinite(value).all() for value in spatial_grads):
        raise ValueError("Spatial gradient contains non-finite values")
    if not any(np.any(value != 0) for value in spatial_grads):
        raise ValueError("All spatial gradients are zero")

    report = {
        "sample_npz": str(args.sample_npz.resolve()),
        "source_row": row,
        "world_frame_point_count": len(xyz),
        "head_points": 1024,
        "wrist_points": 1024,
        "model": "StructuredSpatialPi0Config(dummy, dummy), visual only, random initialization",
        "loss": float(loss),
        "spatial_gradient_leaf_count": len(spatial_grads),
        "nonzero_spatial_gradient_leaves": sum(bool(np.any(value != 0)) for value in spatial_grads),
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
