#!/usr/bin/env python3
"""Forward/backward gates for UniVTAC GelSight RGB with and without world PC.

The two GelSight RGB views are letterboxed top and bottom in Pi0's third image
slot. This is a tactile-image baseline, not a calibrated force representation.
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
from openpi.models.pi0_config import Pi0Config
from openpi.models.spatial_pi0_config import StructuredSpatialPi0Config
from openpi.training import config
from openpi.training import data_loader


def check(root: Path, *, pointcloud: bool) -> dict:
    model_config = (
        StructuredSpatialPi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4,
            dtype="float32", use_visual=True, use_tactile=False, visual_points=2048,
        ) if pointcloud else Pi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4,
            dtype="float32",
        )
    )
    data_config = config.LeRobotUniVTACDataConfig(
        repo_id=str(root), tactile_rgb=True,
        pointcloud=config.UniVTACPointcloudDataConfig(dataset_root=str(root)) if pointcloud else None,
    ).create(root / "unused_assets", model_config)
    raw = data_loader.create_torch_dataset(data_config, 4, model_config)
    transformed = data_loader.transform_dataset(raw, data_config, skip_norm_stats=True)
    batch = jax.tree.map(lambda *values: jnp.stack(values), transformed[100], transformed[101])
    actions = batch.pop("actions")
    observation = Observation.from_dict(batch)
    tactile_image = np.asarray(observation.images["right_wrist_0_rgb"])
    if not np.asarray(observation.image_masks["right_wrist_0_rgb"]).all():
        raise ValueError("GelSight image is masked out")
    if np.all(tactile_image == 0):
        raise ValueError("GelSight image is empty")
    if pointcloud and (observation.spatial is None or observation.spatial.visual is None):
        raise ValueError("World pointcloud is absent")
    model = model_config.create(jax.random.key(0))

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.key(1), observation, actions, train=False))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    leaves = jax.tree_util.tree_flatten_with_path(gradients)[0]
    if not np.isfinite(float(loss)) or any(not np.isfinite(np.asarray(value)).all() for _, value in leaves):
        raise ValueError("Non-finite loss or gradient")
    spatial_nonzero = sum("spatial" in str(path).lower() and np.any(np.asarray(value) != 0)
                          for path, value in leaves)
    if pointcloud and not spatial_nonzero:
        raise ValueError("Pointcloud spatial branch has no gradient")
    return {
        "modalities": "RGB+world_PC+GelSight_RGB" if pointcloud else "RGB+GelSight_RGB",
        "sample_indices": [100, 101],
        "batch_size": 2,
        "loss": float(loss),
        "gradient_leaf_count": len(leaves),
        "nonzero_spatial_gradient_leaves": int(spatial_nonzero),
        "tactile_image_shape": list(tactile_image.shape),
        "tactile_image_mean_normalized": float(tactile_image.mean()),
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    results = [check(root, pointcloud=False), check(root, pointcloud=True)]
    report = {"dataset_root": str(root), "results": results, "pass": all(x["pass"] for x in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
