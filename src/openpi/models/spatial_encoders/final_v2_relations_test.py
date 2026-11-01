from pathlib import Path
import runpy

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.spatial_encoders.final_v2_relations import FinalV2RelationEncoder
from openpi.models.spatial_encoders.final_v2_relations import keypoint_membership
from openpi.spatial.kinematics import RobotKinematics


def _membership():
    root = Path(__file__).resolve().parents[4]
    audit = runpy.run_path(str(root / "scripts/spatial/audit_xhand_keypoints.py"))
    kin = RobotKinematics(root / "configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    q = dict.fromkeys(kin.required_measured_joints(root="base_link"), 0.0)
    names, groups, *_ = audit["select_keypoints"](kin, kin.compute(q, root="base_link"))
    membership = keypoint_membership(names, groups)
    assert not any(row[0] for row in membership)
    with pytest.raises(ValueError, match="names/groups"):
        keypoint_membership(names[::-1], groups)
    return jnp.array(membership)


def test_relation_shape_masks_and_target_independence():
    membership = _membership()
    model = FinalV2RelationEncoder(rngs=nnx.Rngs(0))
    m = jax.random.normal(jax.random.key(1), (1, 512, 3))
    t = jax.random.normal(jax.random.key(2), (1, 512, 3))
    h = jax.random.normal(jax.random.key(3), (1, 18, 3))
    mask = jnp.ones((1, 512), dtype=bool)
    hmask = jnp.ones((1, 18), dtype=bool)
    ho, hm, oo, om = model(m, mask, t, mask, h, hmask, membership)
    assert ho.shape == (1, 5, 128)
    assert oo.shape == (1, 5, 128)
    assert bool(jnp.all(hm))
    assert bool(jnp.all(om))
    assert bool(jnp.all(jnp.isfinite(ho)))
    assert bool(jnp.all(jnp.isfinite(oo)))
    other_ho, _, _, _ = model(m, mask, t + 10, mask, h, hmask, membership)
    np.testing.assert_array_equal(ho, other_ho)
    empty_ho, empty_hm, empty_oo, empty_om = model(m, jnp.zeros_like(mask), t, mask, h, hmask, membership)
    assert not bool(jnp.any(empty_hm))
    assert not bool(jnp.any(empty_om))
    assert bool(jnp.all(empty_ho == 0))
    assert bool(jnp.all(empty_oo == 0))
    _, no_hand_mask, _, object_mask = model(m, mask, t, mask, h, jnp.zeros_like(hmask), membership)
    assert not bool(jnp.any(no_hand_mask))
    assert bool(jnp.all(object_mask))


def test_hand_geometry_gradients_are_finite():
    membership = _membership()
    model = FinalV2RelationEncoder(rngs=nnx.Rngs(7))
    m = jnp.zeros((1, 512, 3))
    h = jnp.zeros((1, 18, 3))
    mask = jnp.ones((1, 512), dtype=bool)
    hmask = jnp.ones((1, 18), dtype=bool)
    gradient = jax.grad(lambda hand: jnp.sum(model(m, mask, m, mask, hand, hmask, membership)[0]))(h)
    assert bool(jnp.all(jnp.isfinite(gradient)))
