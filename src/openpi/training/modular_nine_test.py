# ruff: noqa: SLF001
"""Synthetic contracts for geometry-center HO and shared tactile MLP."""
import dataclasses
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.spatial_encoders.structured_spatial_encoder_test import _object_centric_input
from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
from openpi.spatial.taxel_contract import finger_geometry_centers
from openpi.spatial.taxel_contract import validate_taxel_order
from openpi.training import config

TASKS = ("peg_all", "press_the_button", "yellow_socket")
ARCHES = {"pi0_xhand_ho_v3_5p": 10, "pi0_xhand_pc_v2_tactile_mlp": 38,
          "pi0_xhand_pc_v2_tactile_mlp_ho_v3": 48}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("arch", ARCHES)
def test_config_cli_and_token_shapes(task, arch, monkeypatch):
    name = f"{arch}/{task}"
    c = config.get_config(name)
    base = config.get_config(f"pi0_xhand_pc_v2/{task}")
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: c})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    cfg = config.cli()
    assert cfg.name == name
    assert cfg.data.repo_id == base.data.repo_id
    assert cfg.data.assets == base.data.assets
    assert cfg.weight_loader == base.weight_loader
    assert cfg.freeze_filter == cfg.model.get_freeze_filter()
    assert cfg.model.encoder.default_token_count == ARCHES[arch]
    obs, _ = cfg.model.inputs_spec()
    encoder = cfg.model.encoder.create(rngs=nnx.Rngs(0))
    assert jax.eval_shape(encoder, obs.spatial).tokens.shape == (1, ARCHES[arch], 128)
    params = nnx.state(encoder, nnx.Param).flat_state()
    if cfg.model.encoder.tactile_encoder_type == "simple_mlp":
        assert not any(str(path[0]).startswith("tactile_") or path[0] == "finger_embedding" for path in params)
        assert encoder.simple_tactile_mlp.linear1.kernel.value.shape == (720, 512)


def test_taxel_centers_and_order():
    data = {"xyz_m": np.arange(1800, dtype=np.float32).reshape(600, 3),
            "point_mask": np.ones(600, bool), "finger_id": np.repeat(np.arange(5), 120),
            "taxel_id": np.tile(np.arange(1, 121), 5), "force": np.zeros((600, 3), np.float32)}
    centers = finger_geometry_centers(data)
    np.testing.assert_allclose(centers["xyz_m"], data["xyz_m"].reshape(5, 120, 3).mean(axis=1))
    data["force"][:] = 999
    np.testing.assert_array_equal(finger_geometry_centers(data)["xyz_m"], centers["xyz_m"])
    data["taxel_id"][[0, 1]] = data["taxel_id"][[1, 0]]
    with pytest.raises(ValueError, match="order"):
        validate_taxel_order(data)


def test_simple_mlp_mask_zero_force_and_gradients():
    c = config.get_config("pi0_xhand_pc_v2_tactile_mlp/peg_all").model.encoder
    enc = c.create(rngs=nnx.Rngs(2))
    x = _synthetic_input(batch_size=1, visual_points=256, tactile_points=600).tactile
    x = x.replace(finger_id=jnp.repeat(jnp.arange(5), 120)[None, :], force=jnp.zeros((1, 600, 3)))
    out, mask, _, _ = enc.simple_tactile_mlp(x, c.transforms)
    assert out.shape == (1, 5, 128)
    assert bool(mask.all())
    altered_norm = x.replace(force_norm=jnp.full((1, 600), jnp.nan))
    norm_output = enc.simple_tactile_mlp(altered_norm, c.transforms)[0]
    np.testing.assert_array_equal(out, norm_output)
    x = x.replace(point_mask=x.point_mask.at[:, :120].set(False), xyz_m=x.xyz_m.at[:, 0].set(jnp.nan))
    out, mask, _, _ = enc.simple_tactile_mlp(x, c.transforms)
    assert not bool(mask[0, 0])
    assert bool(mask[0, 1:].all())
    assert bool(jnp.isfinite(out).all())
    def loss(m): return jnp.mean(m.simple_tactile_mlp(x, c.transforms)[0] ** 2)
    value, grads = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.Param))(enc)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.isfinite(v.value).all()) for v in grads.flat_state().values())
    assert any(bool(jnp.any(v.value != 0)) for v in grads.flat_state().values())


def test_5p_shared_v3_parameters_and_effect():
    model = config.get_config("pi0_xhand_ho_v3_5p/press_the_button").model
    base = config.get_config("pi0_xhand_ho_v3/press_the_button").model
    enc = model.encoder.create(rngs=nnx.Rngs(3))
    old = base.encoder.create(rngs=nnx.Rngs(3))
    a, b = nnx.state(enc.object_relation_encoder, nnx.Param).flat_state(), nnx.state(old.object_relation_encoder, nnx.Param).flat_state()
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key].value, b[key].value)
    inp = _object_centric_input()
    inp = inp.replace(tactile=None, object_geometry=jax.tree.map(lambda x: x[:1], inp.object_geometry.replace(hand_scene=None)),
        hand_keypoints_xyz_m=jnp.ones((1, 5, 3)), hand_keypoints_mask=jnp.ones((1, 5), bool))
    output = enc(inp)
    assert output.tokens.shape == (1, 10, 128)
    assert bool(output.token_mask.all())
    slots = jnp.array([1, 5, 9, 12, 15])
    xyz18 = jnp.zeros((1, 18, 3)).at[:, slots].set(inp.hand_keypoints_xyz_m)
    mask18 = jnp.zeros((1, 18), dtype=bool).at[:, slots].set(inp.hand_keypoints_mask)
    original_output = old(inp.replace(hand_keypoints_xyz_m=xyz18, hand_keypoints_mask=mask18))
    np.testing.assert_array_equal(output.tokens, original_output.tokens)
    np.testing.assert_array_equal(output.token_mask, original_output.token_mask)
    changed = enc(inp.replace(hand_keypoints_xyz_m=inp.hand_keypoints_xyz_m.at[:, 0].add(1)))
    assert not np.allclose(output.tokens, changed.tokens)


@pytest.mark.parametrize("arch", ARCHES)
def test_synthetic_policy_loss_grad(arch, monkeypatch):
    import gc

    from openpi.models import pi0
    from openpi.policies import policy
    prod = config.get_config(f"{arch}/press_the_button").model
    cfg = dataclasses.replace(prod, paligemma_variant="dummy", action_expert_variant="dummy",
                              dtype="float32", visual_points=256, action_horizon=2, max_token_len=8)
    original = pi0._siglip.Module
    monkeypatch.setattr(pi0._siglip, "Module", lambda num_classes=None, *, variant=None, **kwargs:
                        original(num_classes=num_classes, variant="mu/14", **kwargs))
    model = cfg.create(jax.random.key(0))
    inp = _synthetic_input(batch_size=1, visual_points=256, tactile_points=600)
    inp = inp.replace(visual=inp.visual if cfg.use_visual else None,
        tactile=inp.tactile.replace(finger_id=jnp.repeat(jnp.arange(5), 120)[None]) if cfg.use_tactile else None)
    if cfg.encoder.enable_object_relations:
        geometry = jax.tree.map(lambda x: x[:1], _object_centric_input().object_geometry.replace(hand_scene=None))
        count = 5 if cfg.encoder.hand_object_hand_point_mode == "tactile_center_5" else 18
        inp = inp.replace(object_geometry=geometry, hand_keypoints_xyz_m=jnp.ones((1, count, 3)),
                          hand_keypoints_mask=jnp.ones((1, count), bool))
    obs = cfg.fake_obs(1).replace(spatial=inp)
    actions = jnp.zeros((1, 2, cfg.action_dim), jnp.float32)
    def loss(m): return jnp.mean(m.compute_loss(jax.random.key(5), obs, actions, train=False))
    value, gradients = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.Param))(model)
    assert bool(jnp.isfinite(value))
    flat = gradients.flat_state()
    assert all(bool(jnp.isfinite(v.value).all()) for v in flat.values())
    assert any(bool(jnp.any(v.value != 0)) for p, v in flat.items() if p[:2] == ("spatial_router", "encoder"))
    server = policy.Policy(model, sample_kwargs={"num_steps": 1})
    result = server.infer(jax.tree.map(lambda x: np.asarray(x[0]), obs.to_dict()))
    assert result["actions"].shape == (2, cfg.action_dim)
    assert np.isfinite(result["actions"]).all()
    del server, model, gradients
    jax.clear_caches()
    gc.collect()


def test_geometry_center_online_and_offline_agree():
    from types import SimpleNamespace

    from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess
    from openpi.spatial_dataset.adapter import SpatialAugmentedDataset
    online = XHandSpatialOnlinePreprocess(camera_roles=("front",), use_visual=False, use_tactile=False,
        require_object_geometry=True, require_hand_keypoints=True, hand_point_mode="tactile_center_5")
    state = np.zeros(1972, np.float32)
    tactile = online._preprocessor.preprocess_tactile(state)
    sample = SimpleNamespace(visual_xyz_m=np.zeros((4, 3), np.float32), visual_rgb=np.zeros((4, 3), np.uint8),
        visual_rgb_valid=np.ones(4, bool), tactile_xyz_m=tactile.xyz_m, tactile_force_base=tactile.force_base,
        tactile_force_norm=tactile.force_norm, finger_id=tactile.finger_id, taxel_id=tactile.taxel_id)
    class Sidecar:
        def get(self, *, episode_index, frame_index): return sample
    dataset = SpatialAugmentedDataset([{"observation.state": state, "episode_index": 0, "frame_index": 0}],
        Sidecar(), use_visual=False, use_tactile=False, hand_point_mode="tactile_center_5")
    offline = dataset[0]["spatial"]
    geometry = {}
    for role in ("manipulated", "target"):
        geometry[f"{role}_xyz_m"] = np.zeros((512, 3), np.float32)
        geometry[f"{role}_mask"] = np.ones(512, bool)
        geometry[f"{role}_source"] = np.zeros(512, np.int8)
        geometry[f"{role}_age"] = np.zeros(512, np.int16)
        geometry[f"{role}_confidence"] = np.ones(512, np.float32)
    actual = online({"observation.state": state, "object_geometry": geometry})["spatial"]
    assert "tactile" not in offline
    assert "tactile" not in actual
    np.testing.assert_array_equal(offline["hand_keypoints"]["xyz_m"], actual["hand_keypoints"]["xyz_m"])
    np.testing.assert_array_equal(offline["hand_keypoints"]["point_mask"], actual["hand_keypoints"]["point_mask"])
    no_force = state.copy()
    no_force[52:] = np.nan
    force_independent = online({"observation.state": no_force, "object_geometry": geometry})["spatial"]
    np.testing.assert_array_equal(actual["hand_keypoints"]["xyz_m"], force_independent["hand_keypoints"]["xyz_m"])


@pytest.mark.parametrize("arch", ARCHES)
def test_policy_factory_configuration(arch, monkeypatch, tmp_path):
    from openpi.policies import policy_config
    cfg = config.get_config(f"{arch}/press_the_button")
    data = config.DataConfig(spatial=cfg.data.spatial, object_geometry=cfg.data.object_geometry)
    monkeypatch.setattr(type(cfg.data), "create", lambda self, assets, model: data)
    monkeypatch.setattr(type(cfg.model), "load", lambda self, params, **kwargs: object())
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(policy_config._policy, "Policy", lambda model, **kwargs: kwargs)
    built = policy_config.create_trained_policy(cfg, tmp_path, norm_stats={})
    transform = built["transforms"][0]
    assert transform.hand_point_mode == cfg.model.encoder.hand_object_hand_point_mode
    assert transform.validate_tactile_order == (cfg.model.encoder.tactile_encoder_type == "simple_mlp")
    assert transform.require_object_geometry == cfg.model.encoder.enable_object_relations


def test_peg_target_flask_role_must_be_verified():
    from openpi.training.data_loader import validate_object_geometry_roles
    cfg = config.get_config("pi0_xhand_ho_v3_5p/peg_all")
    expected = cfg.data.object_geometry.expected_roles
    validate_object_geometry_roles({"source": {"roles": {"manipulated": "funnel", "target": "target_flask"}}}, expected)
    with pytest.raises(ValueError, match="roles differ"):
        validate_object_geometry_roles({"source": {"roles": {"manipulated": "funnel", "target": "support_flask"}}}, expected)
    with pytest.raises(ValueError, match="roles differ"):
        validate_object_geometry_roles({}, expected)
