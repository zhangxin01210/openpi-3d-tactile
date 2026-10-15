#!/usr/bin/env python3
"""Check LeRobot -> OpenPI RGB/joint contract with a tiny random Pi0 model.

Uses Gemma dummy variants and skips normalization for a fast interface gate.
This does not validate pretrained weights, point-cloud/tactile conditioning,
action replay in the simulator, or a tiny-set overfit.
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
from openpi.training import config
from openpi.training import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_config = Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=4, dtype="float32",
    )
    data_config = config.LeRobotUniVTACDataConfig(
        repo_id=str(args.dataset_root.resolve()),
    ).create(args.dataset_root / "unused_assets", model_config)
    raw = data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    transformed = data_loader.transform_dataset(raw, data_config, skip_norm_stats=True)
    indices = (100, 101)
    samples = [transformed[index] for index in indices]
    first_raw = raw[indices[0]]
    state = np.asarray(first_raw["observation.state"])
    target = np.asarray(first_raw["action"][0])
    first_delta = np.asarray(samples[0]["actions"][0, :9])
    if not np.allclose(first_delta[:7], target[:7] - state[:7], atol=1e-6):
        raise ValueError("Arm action delta direction mismatch")
    if not np.allclose(first_delta[7:], target[7:], atol=1e-6):
        raise ValueError("Gripper target should remain absolute")
    from openpi import transforms
    restored = transforms.compose(data_config.data_transforms.outputs)({
        "state": np.asarray(samples[0]["state"]).copy(),
        "actions": np.asarray(samples[0]["actions"][0:1]).copy(),
    })["actions"][0]
    if not np.allclose(restored, target, atol=1e-6):
        raise ValueError("Action delta/absolute roundtrip mismatch")
    if any(sample["actions"].shape != (4, 32) for sample in samples):
        raise ValueError("Unexpected action horizon or padded action dimension")

    batched = jax.tree.map(lambda *values: jnp.stack(values), *samples)
    actions = batched.pop("actions")
    observation = Observation.from_dict(batched)
    model = model_config.create(jax.random.key(0))

    def loss_fn(m):
        return jnp.mean(m.compute_loss(jax.random.key(1), observation, actions, train=False))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    gradient_leaves = jax.tree.leaves(gradients)
    if not np.isfinite(float(loss)) or not gradient_leaves:
        raise ValueError("Pi0 forward/backward produced invalid loss or no gradients")
    if any(not np.isfinite(np.asarray(leaf)).all() for leaf in gradient_leaves):
        raise ValueError("Pi0 forward/backward produced non-finite gradients")

    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_frames": len(raw),
        "sample_indices": list(indices),
        "batch_size": 2,
        "action_horizon": 4,
        "model": "Pi0Config(dummy, dummy), random initialization, no normalization",
        "rgb_image_shape": list(observation.images["base_0_rgb"].shape),
        "joint_state_shape": list(observation.state.shape),
        "action_shape": list(actions.shape),
        "loss": float(loss),
        "gradient_leaf_count": len(gradient_leaves),
        "action_delta_roundtrip": True,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
