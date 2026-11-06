#!/usr/bin/env python3
# ruff: noqa: SLF001
"""Load one real joined XHand sample and evaluate a model loss; never trains."""
import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.training import config
from openpi.training import data_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--small-model", action="store_true", help="use dummy Gemma and small SigLIP; preserve real data and spatial encoder")
    args = parser.parse_args()
    cfg = config.get_config(args.config_name)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
    item = data_loader.transform_dataset(raw, data)[args.index]
    if args.small_model:
        original = pi0._siglip.Module
        pi0._siglip.Module = lambda num_classes=None, *, variant=None, **kwargs: original(
            num_classes=num_classes, variant="mu/14", **kwargs)
        model_cfg = dataclasses.replace(cfg.model, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32")
    else:
        model_cfg = cfg.model
    inputs = jax.tree.map(lambda x: jnp.asarray(np.asarray(x))[None, ...], item)
    obs = model_lib.Observation.from_dict(inputs)
    model = model_cfg.create(jax.random.key(0))
    # Random initialization is intentional: this is a data/model contract smoke,
    # not training, checkpoint acceptance, or a performance measurement.
    loss = model.compute_loss(jax.random.key(1), obs, inputs["actions"], train=False)
    loss.block_until_ready()
    if not bool(jnp.isfinite(loss).all()):
        raise ValueError("non-finite model loss")
    print(f"PASS {args.config_name} sample={args.index} loss_mean={float(jnp.mean(loss)):.6g} random_init=true")


if __name__ == "__main__":
    main()
