"""Tests for strict relational-checkpoint expansion."""

from __future__ import annotations

import copy
import re

from flax import nnx
import flax.traverse_util
import numpy as np
import pytest

from openpi.models.spatial_encoders.conditioning import SpatialConditioningConfig
from openpi.models.spatial_encoders.router import SpatialConditioningRouter
from openpi.models.spatial_encoders.structured_spatial_encoder import StructuredSpatialEncoderConfig
from openpi.training.spatial_weight_loaders import SpatialCheckpointWeightLoader
from openpi.training.spatial_weight_loaders import StrictObjectRelationExpansionCheckpointWeightLoader
from openpi.training.spatial_weight_loaders import _merge_spatial_checkpoint
from openpi.training.spatial_weight_loaders import audit_spatial_checkpoint


def _reference_params() -> dict:
    return {
        "PaliGemma": {"token_embedder": {"embedding": np.ones((2, 3), dtype=np.float32)}},
        "spatial_router": {
            "encoder": {
                "visual_patch_layers": {"0": {"kernel": np.ones((2, 2), dtype=np.float32)}},
                "tactile_taxel_layers": {"0": {"kernel": np.full((2, 2), 2.0, dtype=np.float32)}},
                "object_relation_encoder": {"ho_layers": {"0": {"kernel": np.full((2, 2), 3.0, dtype=np.float32)}}},
            },
            "suffix_adapter": {"linear": {"kernel": np.full((2, 2), 4.0, dtype=np.float32)}},
        },
    }


def _without_path(tree: dict, *path: str) -> dict:
    copied = copy.deepcopy(tree)
    current = copied
    for key in path[:-1]:
        current = current[key]
    del current[path[-1]]
    return copied


def test_strict_expansion_allows_only_real_relation_parameter_path() -> None:
    reference = _reference_params()
    loaded = _without_path(reference, "spatial_router", "encoder", "object_relation_encoder")
    loader = StrictObjectRelationExpansionCheckpointWeightLoader("unused")

    merged = _merge_spatial_checkpoint(
        loaded_params=loaded,
        reference_params=reference,
        allowed_missing_regexes=loader.allowed_missing_regexes,
    )
    assert np.array_equal(
        merged["spatial_router"]["encoder"]["object_relation_encoder"]["ho_layers"]["0"]["kernel"],
        reference["spatial_router"]["encoder"]["object_relation_encoder"]["ho_layers"]["0"]["kernel"],
    )
    audit = audit_spatial_checkpoint(
        loaded_params=loaded,
        reference_params=reference,
        allowed_missing_regexes=loader.allowed_missing_regexes,
    )
    assert audit.unexpected_missing == ()
    assert audit.allowed_missing == ("spatial_router/encoder/object_relation_encoder/ho_layers/0/kernel",)
    assert re.fullmatch(loader.allowed_missing_regexes[0], audit.allowed_missing[0])


def test_strict_expansion_regex_matches_actual_router_relation_parameters() -> None:
    encoder_config = StructuredSpatialEncoderConfig(enable_object_relations=True)
    router = SpatialConditioningRouter(
        encoder=encoder_config.create(rngs=nnx.Rngs(2)),
        encoder_token_dim=encoder_config.token_dim,
        conditioning=SpatialConditioningConfig(target="suffix"),
        prefix_dim=None,
        suffix_dim=32,
        rngs=nnx.Rngs(3),
    )
    flat_params = flax.traverse_util.flatten_dict(nnx.state(router).to_pure_dict())
    relation_paths = [path for path in flat_params if path[:2] == ("encoder", "object_relation_encoder")]
    assert relation_paths

    missing_regex = StrictObjectRelationExpansionCheckpointWeightLoader("unused").allowed_missing_regexes[0]
    for path in relation_paths:
        full_path = "/".join(("spatial_router", *(str(part) for part in path)))
        assert re.fullmatch(missing_regex, full_path)
    assert not re.fullmatch(missing_regex, "spatial_router/encoder/visual_patch_layers/0/kernel")


@pytest.mark.parametrize(
    ("missing_path", "description"),
    [
        (("spatial_router", "encoder", "visual_patch_layers"), "visual"),
        (("spatial_router", "encoder", "tactile_taxel_layers"), "tactile"),
        (("spatial_router", "suffix_adapter"), "router"),
    ],
)
def test_strict_expansion_rejects_missing_legacy_spatial_parameters(missing_path, description) -> None:
    reference = _reference_params()
    loaded = _without_path(reference, *missing_path)
    loader = StrictObjectRelationExpansionCheckpointWeightLoader("unused")

    with pytest.raises(ValueError, match=description):
        _merge_spatial_checkpoint(
            loaded_params=loaded,
            reference_params=reference,
            allowed_missing_regexes=loader.allowed_missing_regexes,
        )


def test_base_loader_allows_new_lora_and_spatial_but_not_base_or_action_expert() -> None:
    reference = {
        "PaliGemma": {
            "llm": {
                "dense": np.ones((2, 2), dtype=np.float32),
                "lora_a": np.full((2, 1), 3.0, dtype=np.float32),
                "layers": {"attn_1": {"w": np.full((2, 2), 4.0, dtype=np.float32)}},
            }
        },
        "action_in_proj": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "spatial_router": {"encoder": {"kernel": np.full((2, 2), 5.0, dtype=np.float32)}},
    }
    # A public Pi0 checkpoint has the dense base/action parameters but no new
    # LoRA or spatial leaves.  The loader must retain the initialized values
    # for exactly those declared missing modules.
    loaded = {
        "PaliGemma": {
            "llm": {
                "dense": reference["PaliGemma"]["llm"]["dense"],
                "layers": {"attn_1": {"w": reference["PaliGemma"]["llm"]["layers"]["attn_1"]["w"]}},
            }
        },
        "action_in_proj": reference["action_in_proj"],
    }
    loader = SpatialCheckpointWeightLoader("unused")
    merged = _merge_spatial_checkpoint(
        loaded_params=loaded,
        reference_params=reference,
        allowed_missing_regexes=loader.allowed_missing_regexes,
    )
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["lora_a"], reference["PaliGemma"]["llm"]["lora_a"])
    np.testing.assert_array_equal(merged["spatial_router"]["encoder"]["kernel"], reference["spatial_router"]["encoder"]["kernel"])

    loaded_without_action_expert = _without_path(loaded, "PaliGemma", "llm", "layers")
    with pytest.raises(ValueError, match="PaliGemma/llm/layers/attn_1/w"):
        _merge_spatial_checkpoint(
            loaded_params=loaded_without_action_expert,
            reference_params=reference,
            allowed_missing_regexes=loader.allowed_missing_regexes,
        )
