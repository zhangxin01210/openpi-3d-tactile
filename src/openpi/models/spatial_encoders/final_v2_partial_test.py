"""Local contracts for the isolated V2 scene and tactile work in progress."""
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.spatial_encoders.structured_spatial_encoder import StructuredSpatialEncoderConfig
from openpi.models.spatial_encoders.structured_spatial_encoder import _masked_unique_fps_indices
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
from openpi.spatial.tactile import TactileConfig
from openpi.spatial.tactile_geometry import load_xhand_taxel_geometry
from openpi.spatial.tactile_regions import partition_taxel_geometry


def _regions():
    root = Path(__file__).resolve().parents[4] / "configs/ur7e_xhand/tactile"
    geometry = load_xhand_taxel_geometry(
        t16_transformed_path=root / "points_t16_transformed.json",
        t30_right_transformed_path=root / "points_t30_right_hand_transformed.json",
        tactile_config=TactileConfig(),
    )
    return partition_taxel_geometry(geometry.xyz_link_m)


def test_regions_use_canonical_real_geometry():
    regions = _regions()
    assert len(regions) == 5
    assert regions == _regions()
    for row in regions:
        assert np.bincount(row, minlength=4).tolist() == [30, 30, 30, 30]
    assert regions[0] != regions[1]  # thumb and non-thumb sensors differ


def test_unique_fps_with_coincident_and_masked_points():
    xyz = jnp.zeros((1, 5, 3))
    idx = np.asarray(_masked_unique_fps_indices(xyz, jnp.array([[True, False, True, True, False]]), 3))[0]
    assert set(idx.tolist()) == {0, 2, 3}


def test_scene_v2_shapes_and_empty_mask():
    inputs = _synthetic_input(batch_size=1, visual_points=4096, tactile_points=10)
    encoder = StructuredSpatialEncoderConfig(visual_hierarchy_v2=True, enable_tactile=False).create(rngs=nnx.Rngs(0))
    visual = inputs.visual
    tokens, mask, xyz, _ = encoder._encode_visual(visual)  # noqa: SLF001
    assert tokens.shape == (1, 33, 128)
    assert mask.shape == (1, 33)
    assert xyz.shape == (1, 33, 3)
    assert bool(jnp.all(mask))
    assert bool(jnp.all(jnp.isfinite(tokens)))
    empty = visual.replace(point_mask=jnp.zeros_like(visual.point_mask))
    tokens, mask, _, _ = encoder._encode_visual(empty)  # noqa: SLF001
    assert not bool(jnp.any(mask))
    assert bool(jnp.all(jnp.isfinite(tokens)))


def test_scene_v2_gradient_is_finite():
    visual = _synthetic_input(batch_size=1, visual_points=256, tactile_points=10).visual
    encoder = StructuredSpatialEncoderConfig(visual_hierarchy_v2=True, enable_tactile=False).create(rngs=nnx.Rngs(2))
    gradient = jax.grad(lambda xyz: jnp.sum(encoder._encode_visual(visual.replace(xyz_m=xyz))[0]))(visual.xyz_m)  # noqa: SLF001
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(gradient != 0))


def test_tactile_regions_no_learned_queries_and_mask():
    encoder = StructuredSpatialEncoderConfig(tactile_regions=_regions()).create(rngs=nnx.Rngs(1))
    assert not hasattr(encoder, "tactile_pool_queries")
    assert not hasattr(encoder, "tactile_key")
    tactile = _synthetic_input(batch_size=1, visual_points=32, tactile_points=600).tactile
    # Match the canonical finger/taxel ordering for this synthetic check.
    tactile = tactile.replace(finger_id=jnp.repeat(jnp.arange(5), 120)[None, :])
    tokens, mask, _, _ = encoder._encode_tactile(tactile)  # noqa: SLF001
    assert tokens.shape == (1, 25, 128)
    assert bool(jnp.all(mask))
    assert bool(jnp.all(jnp.isfinite(tokens)))
