# ruff: noqa: SLF001
"""Synthetic Pi0 integrations for the hand-object and tactile experiments."""
import dataclasses
import gc

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _object_centric_input
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
from openpi.policies import policy
from openpi.training import config

ARCHITECTURES = ("pi0_xhand_ho_v3", "pi0_xhand_tactile_t1b", "pi0_xhand_tactile_local_v1",
                 "pi0_xhand_pc_v2_tactile_local_v1")


def _input(model):
    spatial = _synthetic_input(batch_size=1, visual_points=256, tactile_points=600)
    spatial = spatial.replace(visual=spatial.visual if model.use_visual else None,
        tactile=spatial.tactile.replace(finger_id=jnp.repeat(jnp.arange(5), 120)[None, :]) if model.use_tactile else None)
    if model.encoder.enable_object_relations:
        geometry = jax.tree.map(lambda x: x[:1], _object_centric_input().object_geometry.replace(hand_scene=None))
        spatial = spatial.replace(object_geometry=geometry,
            hand_keypoints_xyz_m=jax.random.normal(jax.random.key(90), (1, 18, 3)),
            hand_keypoints_mask=jnp.ones((1, 18), dtype=bool))
    return spatial


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_loss_gradients_and_policy(architecture, monkeypatch):
    production = config.get_config(f"{architecture}/press_the_button")
    cfg = dataclasses.replace(production.model, paligemma_variant="dummy", action_expert_variant="dummy",
                              dtype="float32", visual_points=256, action_horizon=2, max_token_len=8)
    siglip = pi0._siglip.Module
    monkeypatch.setattr(pi0._siglip, "Module", lambda num_classes=None, *, variant=None, **kwargs:
                        siglip(num_classes=num_classes, variant="mu/14", **kwargs))
    model = cfg.create(jax.random.key(0))
    spatial = _input(cfg)
    observation = cfg.fake_obs(1).replace(spatial=spatial)
    actions = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), jnp.float32)
    def loss_fn(module):
        return jnp.mean(module.compute_loss(jax.random.key(5), observation, actions, train=False))
    loss, gradients = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    assert bool(jnp.isfinite(loss))
    flat = gradients.flat_state()
    assert all(bool(jnp.isfinite(value.value).all()) for value in flat.values())
    encoder_grads = [value.value for path, value in flat.items() if path[:2] == ("spatial_router", "encoder")]
    assert encoder_grads
    assert any(bool(jnp.any(g != 0)) for g in encoder_grads)
    serving = policy.Policy(model, sample_kwargs={"num_steps": 1})
    result = serving.infer(jax.tree.map(lambda x: np.asarray(x[0]), observation.to_dict()))
    assert result["actions"].shape == (cfg.action_horizon, cfg.action_dim)
    assert np.isfinite(result["actions"]).all()
    del serving, model, gradients
    jax.clear_caches()
    gc.collect()


def test_invalid_ho_and_tactile_are_masked():
    ho = config.get_config("pi0_xhand_ho_v3/press_the_button").model
    spatial = _input(ho)
    encoder = ho.encoder.create(rngs=nnx.Rngs(0))
    geometry = spatial.object_geometry.replace(manipulated_mask=jnp.zeros((1, 512), bool),
                                                target_xyz_m=spatial.object_geometry.target_xyz_m.at[0, 0].set(jnp.nan))
    out = encoder(spatial.replace(object_geometry=geometry))
    assert not bool(out.token_mask[0, :5].any())
    assert bool(out.token_mask[0, 5:].all())
    assert bool(jnp.isfinite(out.tokens).all())
    assert bool((out.tokens[0, :5] == 0).all())
    tactile = config.get_config("pi0_xhand_tactile_t1b/press_the_button").model
    spatial = _input(tactile)
    tactile_input = spatial.tactile.replace(point_mask=jnp.zeros((1, 600), bool),
        xyz_m=spatial.tactile.xyz_m.at[0, 0].set(jnp.nan))
    output = tactile.encoder.create(rngs=nnx.Rngs(0))(spatial.replace(tactile=tactile_input))
    assert not bool(output.token_mask.any())
    assert bool(jnp.isfinite(output.tokens).all())
    assert bool((output.tokens == 0).all())


def test_pc_local_production_point_count():
    model = config.get_config("pi0_xhand_pc_v2_tactile_local_v1/press_the_button").model
    spatial = _synthetic_input(batch_size=1, visual_points=4096, tactile_points=600)
    spatial = spatial.replace(tactile=spatial.tactile.replace(finger_id=jnp.repeat(jnp.arange(5), 120)[None, :]))
    output = model.encoder.create(rngs=nnx.Rngs(4))(spatial)
    assert output.tokens.shape == (1, 38, 128)
    assert bool(output.token_mask.all())
    assert bool(jnp.isfinite(output.tokens).all())
