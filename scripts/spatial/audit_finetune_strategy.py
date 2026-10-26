#!/usr/bin/env python3
"""Instantiate a config and report frozen/trainable parameter groups.

This is an initialization audit only; it does not load a checkpoint or train.
Run it on the training machine for the four research configs, for example:

    PYTHONPATH=src python scripts/spatial/audit_finetune_strategy.py \
      --config pi0_xhand_spatial_dtcr_v1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from typing import Any

import jax
import numpy as np
from flax import nnx

from openpi.training.config import get_config


GROUP_NAMES = (
    "SigLIP",
    "PaliGemma Gemma Base",
    "PaliGemma Gemma LoRA",
    "Action Expert Gemma",
    "Action Expert Projections",
    "Spatial Encoder",
    "Object Relation Encoder",
    "Spatial Suffix Adapter",
    "Other",
)


def _is_action_expert_llm_path(path: tuple[Any, ...]) -> bool:
    """Use the real Gemma mixture-of-experts naming contract.

    ``Pi0`` stores both Linen experts below ``PaliGemma/llm``.  The second
    expert is not a sibling called ``llm_1``; ``gemma.Module._name`` appends
    ``_1`` to the inner modules (``attn_1``, ``mlp_1``, ``final_norm_1``, ...).
    """

    return any(str(part).endswith("_1") for part in path[2:])


def _module_group(path: tuple[Any, ...]) -> str:
    parts = tuple(str(part) for part in path)
    if parts[:2] == ("PaliGemma", "img"):
        return "SigLIP"
    if parts[:2] == ("PaliGemma", "llm"):
        if _is_action_expert_llm_path(path):
            return "Action Expert Gemma"
        if any("lora" in part.lower() for part in parts):
            return "PaliGemma Gemma LoRA"
        return "PaliGemma Gemma Base"
    if parts and parts[0] in {"action_in_proj", "action_out_proj", "state_proj", "action_time_mlp_in", "action_time_mlp_out"}:
        return "Action Expert Projections"
    if "spatial_router" in parts and "object_relation_encoder" in parts:
        return "Object Relation Encoder"
    if "spatial_router" in parts and "suffix_adapter" in parts:
        return "Spatial Suffix Adapter"
    if "spatial_router" in parts:
        return "Spatial Encoder"
    return "Other"


def _count(state: nnx.State) -> tuple[int, int]:
    leaves = state.flat_state()
    count = 0
    elements = 0
    for value in leaves.values():
        array = value.value if hasattr(value, "value") else value
        count += 1
        elements += int(np.prod(array.shape))
    return count, elements


def _make_model(config, *, abstract: bool):
    if abstract:
        return nnx.eval_shape(config.model.create, jax.random.key(0))
    return config.model.create(jax.random.key(0))


def audit(config_name: str, *, abstract: bool = False) -> dict[str, Any]:
    config = get_config(config_name)
    model = _make_model(config, abstract=abstract)
    params = nnx.state(model, nnx.Param)
    frozen = params.filter(config.freeze_filter)
    trainable = params.filter(config.trainable_filter)
    trainable_paths = set(trainable.flat_state())
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "leaves": 0,
            "parameters": 0,
            "trainable_leaves": 0,
            "trainable_parameters": 0,
            "frozen_leaves": 0,
            "frozen_parameters": 0,
        }
    )
    for path, value in params.flat_state().items():
        array = value.value if hasattr(value, "value") else value
        group = groups[_module_group(path)]
        group["leaves"] += 1
        parameter_count = int(np.prod(array.shape))
        group["parameters"] += parameter_count
        if path in trainable_paths:
            group["trainable_leaves"] += 1
            group["trainable_parameters"] += parameter_count
        else:
            group["frozen_leaves"] += 1
            group["frozen_parameters"] += parameter_count
    for name in GROUP_NAMES:
        groups.setdefault(name, {
            "leaves": 0,
            "parameters": 0,
            "trainable_leaves": 0,
            "trainable_parameters": 0,
            "frozen_leaves": 0,
            "frozen_parameters": 0,
        })
    for group in groups.values():
        # Keep the arithmetic explicit in the JSON output so an audit cannot
        # accidentally report a group whose frozen count is missing.
        group["frozen_parameters"] = group["parameters"] - group["trainable_parameters"]
        group["frozen_leaves"] = group["leaves"] - group["trainable_leaves"]
    total_leaves, total_parameters = _count(params)
    frozen_leaves, frozen_parameters = _count(frozen)
    trainable_leaves, trainable_parameters = _count(trainable)
    group_values = dict(groups)
    expected = group_values
    freeze_contract = {
        "gemma_base_frozen": expected["PaliGemma Gemma Base"]["parameters"] > 0
        and expected["PaliGemma Gemma Base"]["trainable_parameters"] == 0,
        "gemma_lora_trainable": expected["PaliGemma Gemma LoRA"]["parameters"] > 0
        and expected["PaliGemma Gemma LoRA"]["frozen_parameters"] == 0,
        "action_expert_full_trainable": expected["Action Expert Gemma"]["parameters"] > 0
        and expected["Action Expert Gemma"]["frozen_parameters"] == 0,
        "action_expert_projections_trainable": expected["Action Expert Projections"]["parameters"] > 0
        and expected["Action Expert Projections"]["frozen_parameters"] == 0,
        "siglip_trainable": expected["SigLIP"]["parameters"] > 0
        and expected["SigLIP"]["frozen_parameters"] == 0,
        "spatial_encoder_trainable": expected["Spatial Encoder"]["trainable_parameters"] > 0,
        "object_relation_encoder_trainable": expected["Object Relation Encoder"]["trainable_parameters"] > 0,
        "suffix_adapter_trainable": expected["Spatial Suffix Adapter"]["trainable_parameters"] > 0,
    }
    schedule = config.lr_schedule
    return {
        "config": config_name,
        "paligemma_variant": config.model.paligemma_variant,
        "action_expert_variant": config.model.action_expert_variant,
        "action_horizon": config.model.action_horizon,
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "weight_loader": type(config.weight_loader).__name__,
        "weight_path": getattr(config.weight_loader, "params_path", None),
        "abstract_model": abstract,
        "lr_schedule": dataclasses.asdict(schedule) if dataclasses.is_dataclass(schedule) else repr(schedule),
        "freeze_contract": freeze_contract,
        "total": {"leaves": total_leaves, "parameters": total_parameters},
        "frozen": {"leaves": frozen_leaves, "parameters": frozen_parameters},
        "trainable": {"leaves": trainable_leaves, "parameters": trainable_parameters},
        "groups": group_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--abstract",
        action="store_true",
        help="Use nnx.eval_shape for a memory-light parameter-tree audit (no weights are loaded).",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.config, abstract=args.abstract), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
