"""Compatibility tests for optional Level-1 relation tokens."""

from __future__ import annotations

from dataclasses import replace

from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.spatial_encoders.object_relation_encoder import _finite_geometry
from openpi.models.spatial_encoders.object_relation_encoder import _local_anchors
from openpi.models.spatial_encoders.types import HandSceneInput
from openpi.models.spatial_encoders.types import CausalTactileInput
from openpi.models.spatial_encoders.structured_spatial_encoder import StructuredSpatialEncoderConfig
from openpi.models.spatial_encoders.types import ObjectGeometryInput
from openpi.models.spatial_encoders.types import SpatialEncoderInput
from openpi.models.spatial_encoders.types import TactileSpatialInput
from openpi.models.spatial_encoders.types import VisualSpatialInput


def _inputs(*, with_object_geometry: bool, object_count: int = 32, tactile_count: int = 60) -> SpatialEncoderInput:
    batch, visual_count = 1, 64
    keys = jax.random.split(jax.random.key(21), 4)
    visual = VisualSpatialInput(
        xyz_m=jax.random.normal(keys[0], (batch, visual_count, 3)),
        rgb=jnp.zeros((batch, visual_count, 3), dtype=jnp.uint8),
        rgb_valid=jnp.ones((batch, visual_count), dtype=jnp.bool_),
        point_mask=jnp.ones((batch, visual_count), dtype=jnp.bool_),
    )
    assert tactile_count % 5 == 0
    finger_id = jnp.tile(jnp.arange(5, dtype=jnp.int32), tactile_count // 5)[None, :]
    tactile = TactileSpatialInput(
        xyz_m=jax.random.normal(keys[1], (batch, tactile_count, 3)),
        force=jax.random.normal(keys[2], (batch, tactile_count, 3)),
        force_norm=jnp.ones((batch, tactile_count)),
        finger_id=finger_id,
        taxel_id=jnp.arange(tactile_count, dtype=jnp.int32)[None, :],
        point_mask=jnp.ones((batch, tactile_count), dtype=jnp.bool_),
    )
    geometry = None
    if with_object_geometry:
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


def _hand_scene(*, tactile_count: int, contact_valid: bool = True, funnel_valid: bool = True) -> HandSceneInput:
    mask = jnp.zeros((1, 32, 32), dtype=jnp.bool_)
    mask = mask.at[:, 8:20, 10:22].set(True)
    return HandSceneInput(
        manipulated_relation_valid=jnp.asarray([True]),
        target_relation_valid=jnp.asarray([True]),
        manipulated_geometry_quality=jnp.asarray([0.8], dtype=jnp.float32),
        target_geometry_quality=jnp.asarray([0.6], dtype=jnp.float32),
        grasp_reference_force_norm=jnp.full((1, tactile_count), 0.2, dtype=jnp.float32),
        grasp_short_delta_force_norm=jnp.full((1, tactile_count), 0.1, dtype=jnp.float32),
        grasp_contact_valid=jnp.full((1, tactile_count), contact_valid, dtype=jnp.bool_),
        grasp_quality=jnp.asarray([0.9], dtype=jnp.float32),
        funnel_mask=mask,
        funnel_mask_valid=jnp.asarray([funnel_valid]),
        funnel_mask_quality=jnp.asarray([0.7], dtype=jnp.float32),
        funnel_mask_hand_offset_xy=jnp.asarray([[0.1, -0.2]], dtype=jnp.float32),
    )


def test_relations_append_exactly_fourteen_tokens() -> None:
    baseline = StructuredSpatialEncoderConfig(
        visual_num_centers=4, visual_knn=4, tactile_knn=4, tactile_local_tokens_per_finger=2
    )
    relational = dataclass_replace(baseline, enable_object_relations=True)
    base_output = baseline.create(rngs=nnx.Rngs(0))(_inputs(with_object_geometry=False))
    relation_output = relational.create(rngs=nnx.Rngs(0))(_inputs(with_object_geometry=True))
    assert base_output.tokens.shape == (1, 20, 128)
    assert relation_output.tokens.shape == (1, 34, 128)
    assert relation_output.aux["ho_attention_weights"].shape == (1, 5, 16)
    assert relation_output.aux["oo_attention_weights"].shape == (1, 2, 4, 64)


def test_relations_preserve_existing_token_values_with_copied_parameters() -> None:
    # Exercise the production token layout directly: 32 visual local + one
    # visual global + five fingers * (four local + one summary) = 58.
    baseline_config = StructuredSpatialEncoderConfig(enable_object_relations=False)
    relation_config = replace(baseline_config, enable_object_relations=True)
    baseline_model = baseline_config.create(rngs=nnx.Rngs(7))
    relation_model = relation_config.create(rngs=nnx.Rngs(13))

    # Do not rely on matching RNG seeds: explicitly transplant every legacy
    # parameter into the expanded model and verify the copied tree first.
    baseline_params = flax.traverse_util.flatten_dict(nnx.state(baseline_model).to_pure_dict())
    relation_state = nnx.state(relation_model)
    relation_params = flax.traverse_util.flatten_dict(relation_state.to_pure_dict())
    assert set(baseline_params) < set(relation_params)
    assert not hasattr(baseline_model, "object_relation_encoder")
    for path, value in baseline_params.items():
        relation_params[path] = value
    relation_state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(relation_params))
    nnx.update(relation_model, relation_state)

    copied_params = flax.traverse_util.flatten_dict(nnx.state(relation_model).to_pure_dict())
    for path, value in baseline_params.items():
        np.testing.assert_array_equal(copied_params[path], value)

    baseline_output = baseline_model(_inputs(with_object_geometry=False))
    relation_output = relation_model(_inputs(with_object_geometry=True))
    baseline_count = baseline_output.tokens.shape[1]
    assert baseline_count == 58
    assert relation_config.default_token_count == 72
    assert relation_output.tokens.shape[1] - baseline_count == 14
    assert relation_output.tokens.shape[-1] == 128
    np.testing.assert_allclose(relation_output.tokens[:, :baseline_count], baseline_output.tokens, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(relation_output.token_mask[:, :baseline_count], baseline_output.token_mask)
    np.testing.assert_allclose(
        relation_output.token_xyz_m[:, :baseline_count], baseline_output.token_xyz_m, rtol=1e-6, atol=1e-6
    )
    assert relation_output.tokens[:, baseline_count : baseline_count + 5].shape == (1, 5, 128)
    assert relation_output.tokens[:, baseline_count + 5 :].shape == (1, 9, 128)


def test_invalid_object_points_are_masked() -> None:
    config = StructuredSpatialEncoderConfig(
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
        enable_object_relations=True,
    )
    inputs = _inputs(with_object_geometry=True)
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
    output = config.create(rngs=nnx.Rngs(1))(
        SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=invalid)
    )
    assert not bool(jnp.any(output.token_mask[:, -14:]))
    assert bool(jnp.all(jnp.isfinite(output.tokens)))
    assert bool(jnp.all(jnp.isfinite(output.token_xyz_m)))


def _dtcr_hand_scene(tactile_count: int) -> HandSceneInput:
    base = _hand_scene(tactile_count=tactile_count)
    return HandSceneInput(
        manipulated_relation_valid=base.manipulated_relation_valid,
        target_relation_valid=base.target_relation_valid,
        manipulated_geometry_quality=base.manipulated_geometry_quality,
        target_geometry_quality=base.target_geometry_quality,
        grasp_reference_force_norm=base.grasp_reference_force_norm,
        grasp_short_delta_force_norm=base.grasp_short_delta_force_norm,
        grasp_contact_valid=base.grasp_contact_valid,
        grasp_quality=jnp.zeros_like(base.grasp_quality),
        funnel_mask=base.funnel_mask,
        funnel_mask_valid=base.funnel_mask_valid,
        funnel_mask_quality=base.funnel_mask_quality,
        funnel_mask_hand_offset_xy=base.funnel_mask_hand_offset_xy,
        causal_tactile=CausalTactileInput(
            ema_before_force_norm=jnp.full((1, tactile_count), 0.2, dtype=jnp.float32),
            ema_after_force_norm=jnp.full((1, tactile_count), 0.25, dtype=jnp.float32),
            current_minus_ema_force_norm=jnp.full((1, tactile_count), 0.4, dtype=jnp.float32),
            current_minus_previous_force_norm=jnp.full((1, tactile_count), 0.3, dtype=jnp.float32),
            reference_valid=jnp.ones((1, tactile_count), dtype=jnp.bool_),
            previous_valid=jnp.ones((1, tactile_count), dtype=jnp.bool_),
            sample_valid=jnp.ones((1, tactile_count), dtype=jnp.bool_),
            alpha=jnp.asarray([0.1], dtype=jnp.float32),
        ),
    )


def test_m2_and_dtcr_keep_ten_relation_tokens_and_masks() -> None:
    inputs = _inputs(with_object_geometry=True, object_count=32, tactile_count=60)
    scene = _dtcr_hand_scene(60)
    geometry = replace(inputs.object_geometry, hand_scene=scene)
    for mode in ("hand_scene_oo_v1", "dtcr_v1", "dtcr_independent_v1"):
        config = StructuredSpatialEncoderConfig(
            enable_object_relations=True,
            object_relation_mode=mode,
            visual_num_centers=4,
            visual_knn=4,
            tactile_knn=4,
            tactile_local_tokens_per_finger=2,
        )
        output = config.create(rngs=nnx.Rngs(17))(replace(inputs, object_geometry=geometry))
        assert output.tokens.shape[1] == 30  # 20 base tokens + 5 + 5
        assert output.token_mask.shape[1] == 30
        assert bool(jnp.all(jnp.isfinite(output.tokens)))
        assert output.aux["object_relation_token_count"].tolist() == [10]


def test_dtcr_hand_funnel_condition_has_gradient_and_changes_funnel_target() -> None:
    inputs = _inputs(with_object_geometry=True, object_count=32, tactile_count=60)
    geometry = replace(inputs.object_geometry, hand_scene=_dtcr_hand_scene(60))
    config = StructuredSpatialEncoderConfig(
        enable_object_relations=True,
        object_relation_mode="dtcr_v1",
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
    )
    model = config.create(rngs=nnx.Rngs(23))
    dtcr_inputs = replace(inputs, object_geometry=geometry)
    output = model(dtcr_inputs)
    altered_scene = replace(
        geometry.hand_scene,
        causal_tactile=replace(
            geometry.hand_scene.causal_tactile,
            current_minus_ema_force_norm=jnp.full((1, 60), 2.0, dtype=jnp.float32),
        ),
    )
    altered = model(replace(inputs, object_geometry=replace(geometry, hand_scene=altered_scene)))
    assert not bool(jnp.allclose(output.tokens[:, -5:], altered.tokens[:, -5:]))

    def loss_fn(module):
        return jnp.mean(jnp.square(module(dtcr_inputs).tokens[:, -5:]))

    _, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    relation_grads = [
        leaf.value
        for path, leaf in grads.flat_state().items()
        if "dtcr_hand" in "/".join(map(str, path)) or "dtcr_ft" in "/".join(map(str, path))
    ]
    assert relation_grads
    assert float(jnp.linalg.norm(jnp.concatenate([jnp.ravel(value) for value in relation_grads]))) > 0.0


def test_lifted_funnel_invalidates_hand_relation_but_keeps_object_relation() -> None:
    inputs = _inputs(with_object_geometry=True, object_count=32, tactile_count=60)
    scene = _dtcr_hand_scene(60)
    lifted_scene = replace(scene, manipulated_relation_valid=jnp.asarray([False]))
    config = StructuredSpatialEncoderConfig(
        enable_object_relations=True,
        object_relation_mode="dtcr_v1",
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
    )
    output = config.create(rngs=nnx.Rngs(29))(
        replace(inputs, object_geometry=replace(inputs.object_geometry, hand_scene=lifted_scene))
    )
    assert not bool(jnp.any(output.token_mask[:, -10:-5]))
    assert bool(jnp.all(output.token_mask[:, -5:]))


def test_local_anchors_are_spatial_and_permutation_invariant() -> None:
    points = jnp.array(
        [
            [-1.4, -0.2, 0.1],
            [-1.1, 0.3, 0.2],
            [-0.8, -0.4, 0.5],
            [-0.3, 0.8, -0.1],
            [0.0, -0.7, 0.4],
            [0.3, 0.2, -0.6],
            [0.7, -0.5, 0.3],
            [0.9, 0.6, -0.2],
            [1.2, -0.1, 0.8],
            [1.6, 0.5, 0.1],
            [1.9, -0.6, -0.4],
            [2.2, 0.2, 0.6],
        ],
        dtype=jnp.float32,
    )[None, ...]
    point_count = points.shape[1]
    source = (jnp.arange(point_count)[None, :] % 3).astype(jnp.float32)
    age = jnp.arange(point_count, dtype=jnp.float32)[None, :]
    confidence = jnp.linspace(0.2, 1.0, point_count, dtype=jnp.float32)[None, :]
    mask = jnp.ones((1, point_count), dtype=jnp.bool_)
    permutation = jnp.array([8, 2, 11, 5, 0, 10, 4, 7, 1, 9, 3, 6])

    anchors_a = _local_anchors(_finite_geometry(points, mask, source, age, confidence), count=8)
    anchors_b = _local_anchors(
        _finite_geometry(
            points[:, permutation],
            mask[:, permutation],
            source[:, permutation],
            age[:, permutation],
            confidence[:, permutation],
        ),
        count=8,
    )
    for actual, expected in zip(anchors_b, anchors_a, strict=True):
        if actual.dtype == jnp.bool_:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_relation_tokens_ignore_padding_values() -> None:
    config = StructuredSpatialEncoderConfig(
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
        enable_object_relations=True,
    )
    model = config.create(rngs=nnx.Rngs(5))
    inputs = _inputs(with_object_geometry=False)
    valid_count, point_count = 12, 32
    valid_xyz = jnp.arange(valid_count * 3, dtype=jnp.float32).reshape(1, valid_count, 3) / 10.0
    mask = jnp.arange(point_count)[None, :] < valid_count

    def geometry_with_padding(padding_value: float) -> ObjectGeometryInput:
        xyz = jnp.concatenate(
            [valid_xyz, jnp.full((1, point_count - valid_count, 3), padding_value, dtype=jnp.float32)], axis=1
        )
        source = jnp.concatenate(
            [jnp.ones((1, valid_count), dtype=jnp.int8), jnp.full((1, point_count - valid_count), 7, dtype=jnp.int8)],
            axis=1,
        )
        age = jnp.concatenate(
            [
                jnp.arange(valid_count, dtype=jnp.int16)[None, :],
                jnp.full((1, point_count - valid_count), 99, dtype=jnp.int16),
            ],
            axis=1,
        )
        confidence = jnp.concatenate(
            [
                jnp.linspace(0.2, 1.0, valid_count, dtype=jnp.float32)[None, :],
                jnp.full((1, point_count - valid_count), 0.01, dtype=jnp.float32),
            ],
            axis=1,
        )
        return ObjectGeometryInput(
            manipulated_xyz_m=xyz,
            manipulated_mask=mask,
            manipulated_source=source,
            manipulated_age=age,
            manipulated_confidence=confidence,
            target_xyz_m=xyz + 0.35,
            target_mask=mask,
            target_source=source,
            target_age=age,
            target_confidence=confidence,
        )

    output_a = model(
        SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=geometry_with_padding(0.0))
    )
    output_b = model(
        SpatialEncoderInput(
            visual=inputs.visual, tactile=inputs.tactile, object_geometry=geometry_with_padding(12345.0)
        )
    )
    np.testing.assert_array_equal(output_a.token_mask[:, -14:], output_b.token_mask[:, -14:])
    np.testing.assert_allclose(output_a.tokens[:, -14:], output_b.tokens[:, -14:], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output_a.token_xyz_m[:, -14:], output_b.token_xyz_m[:, -14:], rtol=1e-6, atol=1e-6)


def test_hand_scene_v0_keeps_58_base_tokens_and_appends_two_five_token_groups() -> None:
    base_config = StructuredSpatialEncoderConfig()
    config = replace(
        base_config,
        enable_object_relations=True,
        object_relation_mode="hand_scene_v0",
        hand_scene_enable_funnel_mask_encoder=True,
    )
    base_inputs = _inputs(with_object_geometry=True, object_count=512, tactile_count=600)
    geometry = replace(base_inputs.object_geometry, hand_scene=_hand_scene(tactile_count=600))
    output = config.create(rngs=nnx.Rngs(41))(
        SpatialEncoderInput(visual=base_inputs.visual, tactile=base_inputs.tactile, object_geometry=geometry)
    )
    assert base_config.default_token_count == 58
    assert config.default_token_count == 68
    assert output.tokens.shape == (1, 68, 128)
    assert output.token_xyz_m.shape == (1, 68, 3)
    assert bool(jnp.all(output.token_mask[:, 58:63]))
    assert bool(jnp.all(output.token_mask[:, 63:68]))
    assert output.aux["grasp_finger_gate"].shape == (1, 5)


def test_hand_scene_invalid_lift_empty_cloud_and_missing_mask_are_safe() -> None:
    config = StructuredSpatialEncoderConfig(
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
        enable_object_relations=True,
        object_relation_mode="hand_scene_v0",
        hand_scene_enable_funnel_mask_encoder=True,
    )
    inputs = _inputs(with_object_geometry=True, tactile_count=60)
    state = _hand_scene(tactile_count=60, contact_valid=False, funnel_valid=False)
    state = replace(
        state,
        manipulated_relation_valid=jnp.asarray([False]),
        manipulated_geometry_quality=jnp.asarray([0.0], dtype=jnp.float32),
    )
    lifted = config.create(rngs=nnx.Rngs(42))(
        SpatialEncoderInput(
            visual=inputs.visual,
            tactile=inputs.tactile,
            object_geometry=replace(inputs.object_geometry, hand_scene=state),
        )
    )
    assert not bool(jnp.any(lifted.token_mask[:, -10:-5]))
    assert bool(jnp.all(lifted.token_mask[:, -5:]))
    empty = replace(
        inputs.object_geometry,
        manipulated_mask=jnp.zeros_like(inputs.object_geometry.manipulated_mask),
        target_mask=jnp.zeros_like(inputs.object_geometry.target_mask),
        hand_scene=state,
    )
    output = config.create(rngs=nnx.Rngs(42))(
        SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=empty)
    )
    assert not bool(jnp.any(output.token_mask[:, -10:]))
    assert not bool(jnp.any(output.aux["funnel_mask_feature_valid"]))
    assert bool(jnp.all(jnp.isfinite(output.tokens)))
    assert bool(jnp.all(jnp.isfinite(output.token_xyz_m)))


def test_hand_scene_grasp_adapter_changes_only_hand_target_tokens_and_has_gradients() -> None:
    config = StructuredSpatialEncoderConfig(
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
        enable_object_relations=True,
        object_relation_mode="hand_scene_v0",
        hand_scene_enable_funnel_mask_encoder=True,
    )
    inputs = _inputs(with_object_geometry=True, tactile_count=60)
    with_grasp = replace(inputs.object_geometry, hand_scene=_hand_scene(tactile_count=60, contact_valid=True))
    without_grasp = replace(inputs.object_geometry, hand_scene=_hand_scene(tactile_count=60, contact_valid=False))
    model = config.create(rngs=nnx.Rngs(43))
    output_with = model(SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=with_grasp))
    output_without = model(SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=without_grasp))
    np.testing.assert_allclose(output_with.tokens[:, -10:-5], output_without.tokens[:, -10:-5], rtol=1e-6, atol=1e-6)
    assert not np.allclose(output_with.tokens[:, -5:], output_without.tokens[:, -5:])
    assert float(jnp.max(output_with.aux["grasp_finger_gate"])) > 0.0
    assert float(jnp.max(output_without.aux["grasp_finger_gate"])) == 0.0

    def loss_fn(encoder):
        return jnp.mean(jnp.square(encoder(
            SpatialEncoderInput(visual=inputs.visual, tactile=inputs.tactile, object_geometry=with_grasp)
        ).tokens))

    _, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    flat_grads = flax.traverse_util.flatten_dict(grads.to_pure_dict())
    grasp_grads = [
        jnp.ravel(value)
        for path, value in flat_grads.items()
        if "grasp_adapter" in path or "grasp_gate" in path or "grasp_taxel_layers" in path
    ]
    assert grasp_grads
    assert float(jnp.linalg.norm(jnp.concatenate(grasp_grads))) > 0.0
    mask_grads = [jnp.ravel(value) for path, value in flat_grads.items() if "funnel_mask_layers" in path]
    assert mask_grads
    assert float(jnp.linalg.norm(jnp.concatenate(mask_grads))) > 0.0


def dataclass_replace(value, **changes):
    from dataclasses import replace

    return replace(value, **changes)
