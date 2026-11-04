"""Synthetic end-to-end Pi0 checks with small backbones and real spatial encoders."""

import dataclasses
import gc

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
from openpi.policies import policy
from openpi.training import config


@pytest.mark.parametrize("tactile", [False, True])
def test_synthetic_pi0_loss_gradients_and_policy(tactile, monkeypatch):
    name = ("pi0_xhand_pc_v2_tactile_v2/press_the_button" if tactile
            else "pi0_xhand_pc_v2/press_the_button")
    production = config.get_config(name)
    # Keep the actual encoder/router architecture; small backbones, shorter
    # horizon and 256 input points make the integration test practical on CPU.
    cfg = dataclasses.replace(production.model, paligemma_variant="dummy", action_expert_variant="dummy",
                              dtype="float32", visual_points=256, action_horizon=2, max_token_len=8)
    siglip_module = pi0._siglip.Module  # noqa: SLF001 - restrict test backbone memory
    monkeypatch.setattr(pi0._siglip, "Module",  # noqa: SLF001
                        lambda num_classes=None, *, variant=None, **kwargs:
                        siglip_module(num_classes=num_classes, variant="mu/14", **kwargs))
    model = cfg.create(jax.random.key(0))
    spatial = _synthetic_input(batch_size=1, visual_points=256, tactile_points=600)
    if tactile:
        spatial = spatial.replace(tactile=spatial.tactile.replace(
            finger_id=jnp.repeat(jnp.arange(5), 120)[None, :]
        ))
    else:
        spatial = spatial.replace(tactile=None)
    observation = cfg.fake_obs(1).replace(spatial=spatial)
    actions = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)

    def loss_fn(module):
        return jnp.mean(module.compute_loss(jax.random.key(5), observation, actions, train=False))

    loss, gradients = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model)
    flat = gradients.flat_state()
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(leaf.value).all()) for leaf in flat.values())
    spatial_grads = [leaf.value for path, leaf in flat.items() if path[:2] == ("spatial_router", "encoder")]
    assert spatial_grads
    assert any(bool(jnp.any(gradient != 0)) for gradient in spatial_grads)

    serving = policy.Policy(model, sample_kwargs={"num_steps": 1})
    unbatched = jax.tree.map(lambda value: np.asarray(value[0]), observation.to_dict())
    result = serving.infer(unbatched)
    assert result["actions"].shape == (cfg.action_horizon, cfg.action_dim)
    assert np.isfinite(result["actions"]).all()
    del serving, model, gradients
    jax.clear_caches()
    gc.collect()
