"""Compatibility tests for optional Level-1 relation tokens."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models.spatial_encoders.structured_spatial_encoder import StructuredSpatialEncoderConfig
from openpi.models.spatial_encoders.types import (
    ObjectGeometryInput,
    SpatialEncoderInput,
    TactileSpatialInput,
    VisualSpatialInput,
)


def _inputs(object_geometry: bool) -> SpatialEncoderInput:
    batch, visual_count, tactile_count, object_count = 1, 64, 60, 32
    keys = jax.random.split(jax.random.key(21), 4)
    visual = VisualSpatialInput(
        xyz_m=jax.random.normal(keys[0], (batch, visual_count, 3)),
        rgb=jnp.zeros((batch, visual_count, 3), dtype=jnp.uint8),
        rgb_valid=jnp.ones((batch, visual_count), dtype=jnp.bool_),
        point_mask=jnp.ones((batch, visual_count), dtype=jnp.bool_),
    )
    finger_id = jnp.tile(jnp.arange(5, dtype=jnp.int32), 12)[None, :]
    tactile = TactileSpatialInput(
        xyz_m=jax.random.normal(keys[1], (batch, tactile_count, 3)),
        force=jax.random.normal(keys[2], (batch, tactile_count, 3)),
        force_norm=jnp.ones((batch, tactile_count)),
        finger_id=finger_id,
        taxel_id=jnp.arange(tactile_count, dtype=jnp.int32)[None, :],
        point_mask=jnp.ones((batch, tactile_count), dtype=jnp.bool_),
    )
    geometry = None
    if object_geometry:
        xyz = jnp.arange(object_count, dtype=jnp.float32)[None, :, None].repeat(3, axis=2)
        mask = jnp.ones((batch, object_count), dtype=jnp.bool_)
        zero = jnp.zeros((batch, object_count), dtype=jnp.int8)
        age = jnp.zeros((batch, object_count), dtype=jnp.int16)
        confidence = jnp.ones((batch, object_count), dtype=jnp.float32)
        geometry = ObjectGeometryInput(
            manipulated_xyz_m=xyz,
            manipulated_mask=mask,
            manipulated_source=zero,
            manipulated_age=age,
            manipulated_confidence=confidence,
            target_xyz_m=xyz + 0.1,
            target_mask=mask,
            target_source=zero,
            target_age=age,
            target_confidence=confidence,
        )
    return SpatialEncoderInput(visual=visual, tactile=tactile, object_geometry=geometry)


def test_relations_append_exactly_fourteen_tokens() -> None:
    baseline = StructuredSpatialEncoderConfig(visual_num_centers=4, visual_knn=4, tactile_knn=4, tactile_local_tokens_per_finger=2)
    relational = dataclass_replace(baseline, enable_object_relations=True)
    base_output = baseline.create(rngs=nnx.Rngs(0))(_inputs(False))
    relation_output = relational.create(rngs=nnx.Rngs(0))(_inputs(True))
    assert base_output.tokens.shape == (1, 20, 128)
    assert relation_output.tokens.shape == (1, 34, 128)
    assert relation_output.aux["ho_attention_weights"].shape == (1, 5, 16)
    assert relation_output.aux["oo_attention_weights"].shape == (1, 2, 4, 64)


def test_invalid_object_points_are_masked() -> None:
    config = StructuredSpatialEncoderConfig(visual_num_centers=4, visual_knn=4, tactile_knn=4, tactile_local_tokens_per_finger=2, enable_object_relations=True)
    inputs = _inputs(True)
    invalid = ObjectGeometryInput(
        manipulated_xyz_m=inputs.object_geometry.manipulated_xyz_m,
        manipulated_mask=jnp.zeros_like(inputs.object_geometry.manipulated_mask),
        manipulated_source=inputs.object_geometry.manipulated_source,
        manipulated_age=inputs.object_geometry.manipulated_age,
        manipulated_confidence=inputs.object_geometry.manipulated_confidence,
        target_xyz_m=inputs.object_geometry.target_xyz_m,
        target_mask=jnp.zeros_like(inputs.object_geometry.target_mask),
        target_source=inputs.object_geometry.target_source,
        target_age=inputs.object_geometry.target_age,
        target_confidence=inputs.object_geometry.target_confidence,
    )
    output = config.create(rngs=nnx.Rngs(1))(SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=invalid))
    assert not bool(jnp.any(output.token_mask[:, -14:]))
    assert bool(jnp.all(jnp.isfinite(output.tokens)))


def dataclass_replace(value, **changes):
    from dataclasses import replace

    return replace(value, **changes)
