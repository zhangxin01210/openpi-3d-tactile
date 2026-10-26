#!/usr/bin/env python3
"""Instantiate a config and report frozen/trainable parameter groups.

This is an initialization audit only; it does not load a checkpoint or train.
Run it on the training machine for the four research configs, for example:

    PYTHONPATH=src python scripts/spatial/audit_finetune_strategy.py \
      --config pi0_xhand_spatial_dtcr_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import jax
import numpy as np
from flax import nnx

from openpi.training.config import get_config


def _module_group(path: tuple[Any, ...]) -> str:
    text = "/".join(str(part) for part in path)
    if text.startswith("PaliGemma/img"):
        return "SigLIP"
    if text.startswith("PaliGemma/llm_1"):
        return "Action Expert (Gemma expert)"
    if text.startswith("PaliGemma/llm"):
        return "PaliGemma/Gemma VLM"
    if "spatial_router" in text and "suffix_adapter" in text:
        return "Spatial Suffix Adapter"
    if "spatial_router" in text:
        return "Spatial Encoder / Router"
    if text.startswith(("action_in_proj", "action_out_proj", "state_proj", "action_time_mlp")):
        return "Action Expert projections"
    return "Other trainable model parameters"


def _count(state: nnx.State) -> tuple[int, int]:
    leaves = state.flat_state()
    count = 0
    elements = 0
    for value in leaves.values():
        array = value.value if hasattr(value, "value") else value
        count += 1
        elements += int(np.prod(array.shape))
    return count, elements


def audit(config_name: str) -> dict[str, Any]:
    config = get_config(config_name)
    model = config.model.create(jax.random.key(0))
    params = nnx.state(model, nnx.Param)
    frozen = params.filter(config.freeze_filter)
    trainable = params.filter(config.trainable_filter)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"leaves": 0, "parameters": 0, "trainable_parameters": 0})
    for path, value in params.flat_state().items():
        array = value.value if hasattr(value, "value") else value
        group = groups[_module_group(path)]
        group["leaves"] += 1
        group["parameters"] += int(np.prod(array.shape))
        if path in trainable.flat_state():
            group["trainable_parameters"] += int(np.prod(array.shape))
    total_leaves, total_parameters = _count(params)
    frozen_leaves, frozen_parameters = _count(frozen)
    trainable_leaves, trainable_parameters = _count(trainable)
    return {
        "config": config_name,
        "paligemma_variant": config.model.paligemma_variant,
        "action_expert_variant": config.model.action_expert_variant,
        "action_horizon": config.model.action_horizon,
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "weight_loader": type(config.weight_loader).__name__,
        "weight_path": getattr(config.weight_loader, "params_path", None),
        "total": {"leaves": total_leaves, "parameters": total_parameters},
        "frozen": {"leaves": frozen_leaves, "parameters": frozen_parameters},
        "trainable": {"leaves": trainable_leaves, "parameters": trainable_parameters},
        "groups": dict(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
