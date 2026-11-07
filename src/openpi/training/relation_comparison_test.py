# ruff: noqa: SLF001, PT018
"""Contracts for the optional V2 and generic object-set experiments."""
import dataclasses
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.spatial_encoders.final_v2_relations import FinalV2RelationEncoder
from openpi.models.spatial_encoders.final_v2_relations import RelationSetV1Encoder
from openpi.models.spatial_encoders.final_v2_relations import dual_geometry_to_object_set
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _object_centric_input
from openpi.models.spatial_encoders.types import ObjectSetInput
from openpi.training import config

ARCHES = {"pi0_xhand_pc_v2_tactile_mlp_ho_v2": "final_v2",
          "pi0_xhand_pc_v2_tactile_mlp_relation_set_v1": "relation_set_v1"}


@pytest.mark.parametrize("task", ["press_the_button", "yellow_socket"])
@pytest.mark.parametrize("arch", ARCHES)
def test_config_and_shapes(task, arch, monkeypatch):
    name = f"{arch}/{task}"
    cfg = config.get_config(name)
    base = config.get_config(f"pi0_xhand_pc_v2_tactile_mlp_ho_v3/{task}")
    assert dataclasses.replace(cfg.model.encoder, object_relation_mode="hand_object_v3") == base.model.encoder
    assert cfg.data == base.data and cfg.weight_loader == base.weight_loader
    assert cfg.freeze_filter == cfg.model.get_freeze_filter()
    assert cfg.model.encoder.object_relation_mode == ARCHES[arch]
    assert cfg.model.encoder.default_token_count == 48
    assert cfg.model.use_visual and cfg.model.use_tactile and cfg.model.encoder.enable_object_relations
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: cfg})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    assert config.cli().name == name
    enc = cfg.model.encoder.create(rngs=nnx.Rngs(0))
    assert isinstance(enc.object_relation_encoder, FinalV2RelationEncoder)
    assert isinstance(enc.object_relation_encoder, RelationSetV1Encoder) == (ARCHES[arch] == "relation_set_v1")
    assert jax.eval_shape(enc, cfg.model.inputs_spec()[0].spatial).tokens.shape == (1, 48, 128)


def test_object_counts_mask_and_nonfinite():
    model = RelationSetV1Encoder(rngs=nnx.Rngs(0))
    xyz = jax.random.normal(jax.random.key(1), (1, 8, 512, 3))
    xyz = xyz.at[0, 7, 0].set(jnp.nan)
    mask = jnp.ones((1, 8, 512), bool)
    membership = jnp.arange(5)[:, None] == jnp.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])[None]
    hand, hmask = jnp.zeros((1, 18, 3)), jnp.ones((1, 18), bool)
    run = jax.jit(lambda valid: model(ObjectSetInput(xyz, mask, valid), hand, hmask, membership))
    for count in (0, 1, 2, 3, 5):
        ho, hm, oo, om = run((jnp.arange(8) < count)[None])
        assert ho.shape == oo.shape == (1, 5, 128)
        assert bool(hm.all()) == (count > 0)
        assert bool(om.all()) == (count > 1)
        assert bool(jnp.isfinite(ho).all() & jnp.isfinite(oo).all())
        if count == 0:
            assert not bool(jnp.any(ho))
        if count < 2:
            assert not bool(jnp.any(oo))
    # A failed object contributes neither to KNN nor to object pairs.
    failed = ObjectSetInput(xyz, mask.at[:, 1:].set(False), jnp.ones((1, 8), bool))
    _, hm, _, om = model(failed, hand, hmask, membership)
    assert bool(hm.all()) and not bool(om.any())


def test_dual_adapter_and_online_set_validation():
    from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess
    geom = jax.tree.map(lambda x: x[:1], _object_centric_input().object_geometry.replace(hand_scene=None))
    objects = dual_geometry_to_object_set(geom)
    assert objects.object_xyz_m.shape == (1, 8, 512, 3)
    np.testing.assert_array_equal(objects.object_xyz_m[:, 0], geom.manipulated_xyz_m)
    np.testing.assert_array_equal(objects.object_xyz_m[:, 1], geom.target_xyz_m)
    assert not bool(objects.object_valid[:, 2:].any())
    online = XHandSpatialOnlinePreprocess(camera_roles=("front",), use_visual=False, use_tactile=False,
        require_object_geometry=True, require_hand_keypoints=True, allow_object_set=True)
    packet = {"object_xyz_m": np.asarray(objects.object_xyz_m[0]),
              "object_mask": np.asarray(objects.object_mask[0]), "object_valid": np.asarray(objects.object_valid[0])}
    checked = online._validate_object_set(packet)
    np.testing.assert_array_equal(checked["object_xyz_m"], packet["object_xyz_m"])
    hand = {"xyz_m": np.zeros((18, 3), np.float32), "point_mask": np.ones(18, bool)}
    forwarded = online({"spatial": {"object_set": packet, "hand_keypoints": hand}})
    np.testing.assert_array_equal(forwarded["spatial"]["object_set"]["object_xyz_m"], packet["object_xyz_m"])
    with pytest.raises(ValueError, match="not both"):
        online({"spatial": {"object_set": packet, "hand_keypoints": hand, "object_geometry": {}}})
    with pytest.raises(ValueError, match="shapes"):
        online._validate_object_set({**packet, "object_valid": np.ones(7)})


@pytest.mark.parametrize("arch", ARCHES)
def test_policy_factory_accepts_geometry_mode(arch, monkeypatch, tmp_path):
    from openpi.policies import policy_config
    cfg = config.get_config(f"{arch}/press_the_button")
    data = config.DataConfig(spatial=cfg.data.spatial, object_geometry=cfg.data.object_geometry)
    monkeypatch.setattr(type(cfg.data), "create", lambda self, assets, model: data)
    monkeypatch.setattr(type(cfg.model), "load", lambda self, params, **kwargs: object())
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(policy_config._policy, "Policy", lambda model, **kwargs: kwargs)
    pre = policy_config.create_trained_policy(cfg, tmp_path, norm_stats={})["transforms"][0]
    assert pre.require_object_geometry and pre.require_hand_keypoints
    assert pre.allow_object_set == (ARCHES[arch] == "relation_set_v1")


@pytest.mark.parametrize("arch", ARCHES)
def test_synthetic_model_loss_and_policy(arch, monkeypatch):
    import gc

    from openpi.models import pi0
    from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
    from openpi.policies import policy
    production = config.get_config(f"{arch}/press_the_button").model
    cfg = dataclasses.replace(production, paligemma_variant="dummy", action_expert_variant="dummy",
                              dtype="float32", visual_points=256, action_horizon=2, max_token_len=8)
    original = pi0._siglip.Module
    monkeypatch.setattr(pi0._siglip, "Module", lambda num_classes=None, *, variant=None, **kwargs:
                        original(num_classes=num_classes, variant="mu/14", **kwargs))
    model = cfg.create(jax.random.key(0))
    inp = _synthetic_input(batch_size=1, visual_points=256, tactile_points=600)
    inp = inp.replace(tactile=inp.tactile.replace(finger_id=jnp.repeat(jnp.arange(5), 120)[None]))
    geometry = jax.tree.map(lambda x: x[:1], _object_centric_input().object_geometry.replace(hand_scene=None))
    inp = inp.replace(object_geometry=geometry, hand_keypoints_xyz_m=jnp.ones((1, 18, 3)),
                      hand_keypoints_mask=jnp.ones((1, 18), bool))
    if ARCHES[arch] == "relation_set_v1":
        inp = inp.replace(object_set=dual_geometry_to_object_set(geometry), object_geometry=None)
    output = model.spatial_router.encoder(inp)
    assert output.tokens.shape == (1, 48, 128) and bool(jnp.isfinite(output.tokens).all())
    obs = cfg.fake_obs(1).replace(spatial=inp)
    loss = model.compute_loss(jax.random.key(5), obs, jnp.zeros((1, 2, cfg.action_dim)), train=False)
    assert bool(jnp.isfinite(loss).all())
    server = policy.Policy(model, sample_kwargs={"num_steps": 1})
    result = server.infer(jax.tree.map(lambda x: np.asarray(x[0]), obs.to_dict()))
    assert result["actions"].shape == (2, cfg.action_dim)
    assert np.isfinite(result["actions"]).all()
    del server, model
    jax.clear_caches()
    gc.collect()
