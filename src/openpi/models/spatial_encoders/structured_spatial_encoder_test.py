from __future__ import annotations

from dataclasses import replace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.spatial_encoders.structured_spatial_encoder import (
    StructuredSpatialEncoderConfig,
)
from openpi.models.spatial_encoders.types import (
    HandSceneInput,
    ObjectGeometryInput,
    SpatialEncoderInput,
    TactileSpatialInput,
    VisualSpatialInput,
)


def _object_centric_input() -> SpatialEncoderInput:
    base = _synthetic_input(batch_size=2, visual_points=4096, tactile_points=600)
    batch_size = 2
    object_points = 512
    xyz = jnp.linspace(-0.3, 0.3, object_points * 3, dtype=jnp.float32).reshape(1, object_points, 3)
    xyz = jnp.broadcast_to(xyz, (batch_size, object_points, 3))
    mask = jnp.ones((batch_size, object_points), dtype=jnp.bool_)
    source = jnp.zeros((batch_size, object_points), dtype=jnp.int8)
    age = jnp.zeros((batch_size, object_points), dtype=jnp.int16)
    confidence = jnp.ones((batch_size, object_points), dtype=jnp.float32)
    hand_scene = HandSceneInput(
        manipulated_relation_valid=jnp.ones((batch_size,), dtype=jnp.bool_),
        target_relation_valid=jnp.ones((batch_size,), dtype=jnp.bool_),
        manipulated_geometry_quality=jnp.ones((batch_size,), dtype=jnp.float32),
        target_geometry_quality=jnp.ones((batch_size,), dtype=jnp.float32),
        grasp_reference_force_norm=jnp.zeros((batch_size, 600), dtype=jnp.float32),
        grasp_short_delta_force_norm=jnp.zeros((batch_size, 600), dtype=jnp.float32),
        grasp_contact_valid=jnp.zeros((batch_size, 600), dtype=jnp.bool_),
        grasp_quality=jnp.zeros((batch_size,), dtype=jnp.float32),
        funnel_mask=jnp.zeros((batch_size, 32, 32), dtype=jnp.bool_),
        funnel_mask_valid=jnp.zeros((batch_size,), dtype=jnp.bool_),
        funnel_mask_quality=jnp.zeros((batch_size,), dtype=jnp.float32),
        funnel_mask_hand_offset_xy=jnp.zeros((batch_size, 2), dtype=jnp.float32),
    )
    geometry = ObjectGeometryInput(
        manipulated_xyz_m=xyz,
        manipulated_mask=mask,
        manipulated_source=source,
        manipulated_age=age,
        manipulated_confidence=confidence,
        target_xyz_m=xyz + 0.4,
        target_mask=mask,
        target_source=source,
        target_age=age,
        target_confidence=confidence,
        hand_scene=hand_scene,
    )
    return SpatialEncoderInput(visual=None, tactile=base.tactile, object_geometry=geometry)


def _synthetic_input(
    *,
    batch_size: int = 2,
    visual_points: int = 4096,
    tactile_points: int = 600,
    shuffled_tactile: bool = False,
    invalid_tail: bool = False,
) -> SpatialEncoderInput:
    key = jax.random.key(7)
    keys = jax.random.split(key, 5)

    visual_xyz = jax.random.normal(keys[0], (batch_size, visual_points, 3), dtype=jnp.float32)
    visual_rgb = jax.random.randint(keys[1], (batch_size, visual_points, 3), 0, 255, dtype=jnp.uint8)
    visual_rgb_valid = jnp.ones((batch_size, visual_points), dtype=jnp.bool_)
    visual_mask = jnp.ones((batch_size, visual_points), dtype=jnp.bool_)

    tactile_xyz = jax.random.normal(keys[2], (batch_size, tactile_points, 3), dtype=jnp.float32)
    tactile_force = jax.random.normal(keys[3], (batch_size, tactile_points, 3), dtype=jnp.float32)
    tactile_force_norm = jnp.linalg.norm(tactile_force, axis=-1)
    finger_id = jnp.tile(jnp.arange(5, dtype=jnp.int32), tactile_points // 5)
    finger_id = jnp.broadcast_to(finger_id[None, :], (batch_size, tactile_points))
    taxel_id = jnp.broadcast_to(jnp.arange(tactile_points, dtype=jnp.int32)[None, :], (batch_size, tactile_points))
    tactile_mask = jnp.ones((batch_size, tactile_points), dtype=jnp.bool_)

    if shuffled_tactile:
        permutation = jax.random.permutation(keys[4], tactile_points)
        tactile_xyz = tactile_xyz[:, permutation, :]
        tactile_force = tactile_force[:, permutation, :]
        tactile_force_norm = tactile_force_norm[:, permutation]
        finger_id = finger_id[:, permutation]
        taxel_id = taxel_id[:, permutation]
        tactile_mask = tactile_mask[:, permutation]

    if invalid_tail:
        visual_mask = visual_mask.at[:, -128:].set(False)
        visual_rgb_valid = visual_rgb_valid.at[:, -128:].set(False)
        visual_xyz = visual_xyz.at[:, -128:, :].set(1.0e6)
        tactile_mask = tactile_mask.at[:, -60:].set(False)
        tactile_xyz = tactile_xyz.at[:, -60:, :].set(-1.0e6)

    return SpatialEncoderInput(
        visual=VisualSpatialInput(
            xyz_m=visual_xyz,
            rgb=visual_rgb,
            rgb_valid=visual_rgb_valid,
            point_mask=visual_mask,
        ),
        tactile=TactileSpatialInput(
            xyz_m=tactile_xyz,
            force=tactile_force,
            force_norm=tactile_force_norm,
            finger_id=finger_id,
            taxel_id=taxel_id,
            point_mask=tactile_mask,
        ),
    )


def test_structured_spatial_encoder_shapes_and_finite_output():
    encoder = StructuredSpatialEncoderConfig().create(rngs=nnx.Rngs(0))
    output = encoder(_synthetic_input())

    assert output.tokens.shape == (2, 58, 128)
    assert output.token_mask.shape == (2, 58)
    assert output.token_xyz_m.shape == (2, 58, 3)
    assert bool(jnp.all(jnp.isfinite(output.tokens)))
    assert bool(jnp.all(output.token_mask))


def test_structured_spatial_encoder_masks_invalid_points():
    encoder = StructuredSpatialEncoderConfig().create(rngs=nnx.Rngs(1))
    clean = _synthetic_input(invalid_tail=True)
    perturbed = _synthetic_input(invalid_tail=True)

    perturbed = SpatialEncoderInput(
        visual=VisualSpatialInput(
            xyz_m=perturbed.visual.xyz_m.at[:, -128:, :].set(-9.0e8),
            rgb=perturbed.visual.rgb.at[:, -128:, :].set(255),
            rgb_valid=perturbed.visual.rgb_valid,
            point_mask=perturbed.visual.point_mask,
        ),
        tactile=TactileSpatialInput(
            xyz_m=perturbed.tactile.xyz_m.at[:, -60:, :].set(9.0e8),
            force=perturbed.tactile.force.at[:, -60:, :].set(9.0e8),
            force_norm=perturbed.tactile.force_norm.at[:, -60:].set(9.0e8),
            finger_id=perturbed.tactile.finger_id,
            taxel_id=perturbed.tactile.taxel_id,
            point_mask=perturbed.tactile.point_mask,
        ),
    )

    clean_output = encoder(clean)
    perturbed_output = encoder(perturbed)

    np.testing.assert_allclose(clean_output.tokens, perturbed_output.tokens, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(clean_output.token_mask, perturbed_output.token_mask)


def test_structured_spatial_encoder_all_invalid_modality_has_no_nan():
    inputs = _synthetic_input()
    inputs = SpatialEncoderInput(
        visual=VisualSpatialInput(
            xyz_m=inputs.visual.xyz_m,
            rgb=inputs.visual.rgb,
            rgb_valid=jnp.zeros_like(inputs.visual.rgb_valid),
            point_mask=jnp.zeros_like(inputs.visual.point_mask),
        ),
        tactile=TactileSpatialInput(
            xyz_m=inputs.tactile.xyz_m,
            force=inputs.tactile.force,
            force_norm=inputs.tactile.force_norm,
            finger_id=inputs.tactile.finger_id,
            taxel_id=inputs.tactile.taxel_id,
            point_mask=jnp.zeros_like(inputs.tactile.point_mask),
        ),
    )
    encoder = StructuredSpatialEncoderConfig().create(rngs=nnx.Rngs(2))
    output = encoder(inputs)

    assert bool(jnp.all(jnp.isfinite(output.tokens)))
    assert not bool(jnp.any(output.token_mask))


def _object_centric_encoder() -> StructuredSpatialEncoderConfig:
    return StructuredSpatialEncoderConfig(
        visual_geometry_mode="object_centric",
        visual_num_centers=32,
        visual_knn=32,
        tactile_knn=8,
        enable_object_relations=True,
        object_relation_mode="hand_scene_oo_v1",
    )


def test_object_centric_encoder_keeps_33_plus_25_plus_10_layout() -> None:
    encoder = _object_centric_encoder().create(rngs=nnx.Rngs(11))
    output = encoder(_object_centric_input())
    assert output.tokens.shape == (2, 68, 128)
    assert output.token_mask.shape == (2, 68)
    assert bool(jnp.all(output.token_mask[:, :33]))
    assert bool(jnp.all(jnp.isfinite(output.tokens)))
    assert output.aux["object_centric_valid_point_count_by_object"].shape == (2, 3)


def test_object_centric_encoder_masks_each_object_without_nan() -> None:
    inputs = _object_centric_input()
    geometry = inputs.object_geometry
    invalid_funnel = geometry.manipulated_mask.at[:, :].set(False)
    invalid_flask = geometry.target_mask.at[:, :].set(False)
    invalid_flask = invalid_flask.at[:, :32].set(True)
    invalid_geometry = replace(
        geometry,
        manipulated_mask=invalid_funnel,
        target_mask=invalid_flask,
        manipulated_xyz_m=geometry.manipulated_xyz_m.at[:, 0, :].set(jnp.nan),
    )
    tactile = replace(inputs.tactile, point_mask=inputs.tactile.point_mask.at[:, :60].set(False))
    output = _object_centric_encoder().create(rngs=nnx.Rngs(12))(
        replace(inputs, tactile=tactile, object_geometry=invalid_geometry)
    )
    # Hand centers are partly valid, the Funnel group is entirely invalid,
    # and the Flask group retains only its first 32 points.
    assert bool(jnp.any(output.token_mask[:, :8]))
    assert not bool(jnp.any(output.token_mask[:, 8:20]))
    assert bool(jnp.any(output.token_mask[:, 20:32]))
    assert bool(jnp.all(jnp.isfinite(output.tokens)))
    assert bool(jnp.all(jnp.isfinite(output.token_xyz_m)))


def test_object_centric_encoder_has_trainable_geometry_parameters() -> None:
    model = _object_centric_encoder().create(rngs=nnx.Rngs(13))
    inputs = _object_centric_input()

    def loss_fn(module):
        return jnp.mean(jnp.square(module(inputs).tokens[:, :33]))

    _, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    geometry_grads = [
        leaf.value
        for path, leaf in grads.flat_state().items()
        if "object_centric" in "/".join(map(str, path))
    ]
    assert geometry_grads
    assert float(jnp.linalg.norm(jnp.concatenate([jnp.ravel(value) for value in geometry_grads]))) > 0.0


def test_structured_spatial_encoder_uses_finger_id_not_tactile_order():
    encoder = StructuredSpatialEncoderConfig().create(rngs=nnx.Rngs(3))
    ordered = encoder(_synthetic_input())
    shuffled = encoder(_synthetic_input(shuffled_tactile=True))

    assert ordered.tokens.shape == shuffled.tokens.shape
    assert ordered.aux["tactile_valid_point_count_by_finger"].shape == (2, 5)
    np.testing.assert_allclose(ordered.tokens, shuffled.tokens, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ordered.token_xyz_m, shuffled.token_xyz_m, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(
        ordered.aux["tactile_valid_point_count_by_finger"],
        shuffled.aux["tactile_valid_point_count_by_finger"],
    )


def test_structured_spatial_encoder_gradient_is_finite_and_nonzero():
    encoder = StructuredSpatialEncoderConfig(
        visual_num_centers=4,
        visual_knn=4,
        tactile_knn=4,
        tactile_local_tokens_per_finger=2,
    ).create(rngs=nnx.Rngs(4))
    inputs = _synthetic_input(visual_points=64, tactile_points=60)

    def loss_fn(model):
        output = model(inputs)
        return jnp.mean(jnp.square(output.tokens))

    diff_state = nnx.DiffState(0, nnx.Param)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(encoder)
    leaves = jax.tree_util.tree_leaves(grads)
    flat = [jnp.ravel(leaf.value if hasattr(leaf, "value") else leaf) for leaf in leaves]
    grad_norm = jnp.linalg.norm(jnp.concatenate(flat))

    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(grad_norm))
    assert float(grad_norm) > 0.0
