#!/usr/bin/env python3
"""Try to fit two fixed UniVTAC RGB/joint samples with a tiny random Pi0.

Fixed diffusion RNG makes this a deterministic pipeline check. The dummy model
is not a pretrained π0 policy and this is not task-performance evidence.
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
from openpi.models.pi0_config import Pi0Config
from openpi.training import config
from openpi.training import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error("steps and learning-rate must be positive")

    model_config = Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=4, dtype="float32",
    )
    data_config = config.LeRobotUniVTACDataConfig(
        repo_id=str(args.dataset_root.resolve()),
    ).create(args.dataset_root / "unused_assets", model_config)
    raw = data_loader.create_torch_dataset(data_config, 4, model_config)
    transformed = data_loader.transform_dataset(raw, data_config, skip_norm_stats=True)
    indices = (100, 101)
    batched = jax.tree.map(lambda *values: jnp.stack(values), *(transformed[i] for i in indices))
    actions = batched.pop("actions")
    observation = Observation.from_dict(batched)

    model = model_config.create(jax.random.key(0))
    optimizer = optax.adam(args.learning_rate)
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
        value = jnp.mean(model.compute_loss(fixed_rng, observation, actions, train=False))
        return float(value)

    losses = [{"step": 0, "loss": evaluate()}]
    print(losses[-1], flush=True)
    for index in range(1, args.steps + 1):
        _, opt_state = step(model, opt_state, observation, actions)
        if index % 10 == 0 or index == args.steps:
            entry = {"step": index, "loss": evaluate()}
            losses.append(entry)
            print(entry, flush=True)

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_indices": list(indices),
        "model": "Pi0Config(dummy, dummy), random initialization",
        "deterministic_diffusion_rng": True,
        "normalization": False,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
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
