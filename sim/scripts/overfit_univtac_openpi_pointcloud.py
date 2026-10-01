#!/usr/bin/env python3
"""Fit two real RGB+world-cloud UniVTAC rows with a tiny spatial Pi0.

This checks the full HDF5 cloud join, transforms and optimization path. Fixed
diffusion noise and a random dummy model cannot establish pointcloud utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.model import Observation
from openpi.models.spatial_pi0_config import StructuredSpatialPi0Config
from openpi.training import config
from openpi.training import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    root = args.dataset_root.resolve()
    model_config = StructuredSpatialPi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=4, dtype="float32", use_visual=True, use_tactile=False,
        visual_points=2048,
    )
    data_config = config.LeRobotUniVTACDataConfig(
        repo_id=str(root),
        pointcloud=config.UniVTACPointcloudDataConfig(dataset_root=str(root)),
    ).create(root / "unused_assets", model_config)
    raw = data_loader.create_torch_dataset(data_config, 4, model_config)
    transformed = data_loader.transform_dataset(raw, data_config, skip_norm_stats=True)
    indices = (100, 101)
    batch = jax.tree.map(lambda *values: jnp.stack(values), *(transformed[i] for i in indices))
    actions = batch.pop("actions")
    observation = Observation.from_dict(batch)
    if observation.spatial is None or observation.spatial.visual is None:
        raise ValueError("Visual spatial input was lost in the OpenPI data pipeline")
    model = model_config.create(jax.random.key(0))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))
    fixed_rng = jax.random.key(1)

    @nnx.jit
    def step(model, opt_state, observation, actions):
        def loss_fn(m):
            return jnp.mean(m.compute_loss(fixed_rng, observation, actions, train=False))

        loss, gradients = nnx.value_and_grad(loss_fn)(model)
        params = nnx.state(model, nnx.Param)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return loss, opt_state

    def evaluate() -> float:
        return float(jnp.mean(model.compute_loss(fixed_rng, observation, actions, train=False)))

    losses = [{"step": 0, "loss": evaluate()}]
    print(losses[-1], flush=True)
    for index in range(1, args.steps + 1):
        _, opt_state = step(model, opt_state, observation, actions)
        if index % 10 == 0 or index == args.steps:
            entry = {"step": index, "loss": evaluate()}
            losses.append(entry)
            print(entry, flush=True)
    report = {
        "dataset_root": str(root),
        "sample_indices": list(indices),
        "model": "StructuredSpatialPi0Config(dummy, dummy), visual only, random initialization",
        "point_count_per_sample": 2048,
        "deterministic_diffusion_rng": True,
        "normalization": False,
        "steps": args.steps,
        "losses": losses,
        "final_to_initial_loss": losses[-1]["loss"] / losses[0]["loss"],
        "threshold_pass": losses[-1]["loss"] < 0.2 * losses[0]["loss"],
    }
    if not np.isfinite(report["final_to_initial_loss"]):
        raise ValueError("Non-finite training loss")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("final_to_initial_loss", "threshold_pass")}))


if __name__ == "__main__":
    main()
