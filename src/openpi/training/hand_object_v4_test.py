# ruff: noqa: SLF001
"""CPU synthetic contracts for the opt-in HO V4 architecture."""
import dataclasses
import gc
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models.spatial_encoders.hand_object_v4 import global_relations
from openpi.policies import policy
from openpi.training import config
from openpi.training.hand_tactile_model_test import _input


@pytest.mark.parametrize("task", ["press_the_button", "yellow_socket"])
@pytest.mark.parametrize("tactile", [False, True])
def test_config_and_tyro(task, tactile, monkeypatch):
    arch = "pi0_xhand_tactile_t1b_ho_v4" if tactile else "pi0_xhand_ho_v4"
    name = f"{arch}/{task}"
    selected = config.get_config(name)
    base = config.get_config(f"{'pi0_xhand_tactile_t1b' if tactile else 'pi0_xhand_ho_v3'}/{task}")
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: selected})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    cfg = config.cli()
    assert cfg.name == name
    assert cfg.checkpoint_dir == (config.pathlib.Path.cwd() / "checkpoints" / name / "seed42").resolve()
    assert cfg.data.repo_id == base.data.repo_id
    assert cfg.data.assets == base.data.assets
    assert cfg.data.object_geometry.anchor_asset_root == "assets/ho_v4_anchors"
    assert cfg.model.encoder.default_token_count == (15 if tactile else 10)
    assert cfg.model.encoder.object_relation_mode == "hand_object_v4"
    assert cfg.freeze_filter == cfg.model.get_freeze_filter()
    assert cfg.weight_loader == base.weight_loader
    assert (cfg.optimizer, cfg.lr_schedule, cfg.model.action_horizon) == (base.optimizer, base.lr_schedule, base.model.action_horizon)
    assert (cfg.batch_size, cfg.num_train_steps, cfg.seed, cfg.save_interval, cfg.keep_period, cfg.ema_decay) == (8, 20000, 42, 5000, 5000, None)
    if tactile:
        assert dataclasses.replace(cfg.model.encoder, enable_object_relations=False, object_relation_mode=base.model.encoder.object_relation_mode) == base.model.encoder
    obs, _ = cfg.model.inputs_spec()
    encoder = cfg.model.encoder.create(rngs=nnx.Rngs(0))
    assert jax.eval_shape(encoder, obs.spatial).tokens.shape == (1, 15 if tactile else 10, 128)
    if tactile:
        old_encoder = base.model.encoder.create(rngs=nnx.Rngs(0))
        old_params = nnx.state(old_encoder, nnx.Param).flat_state()
        new_params = nnx.state(encoder, nnx.Param).flat_state()
        for path, value in old_params.items():
            np.testing.assert_array_equal(value.value, new_params[path].value)


def _v4_input(model):
    inp = _input(model)
    indices = jnp.arange(16, dtype=jnp.int32)[None, :]
    return inp.replace(object_geometry=inp.object_geometry.replace(manipulated_anchor_indices=indices, target_anchor_indices=indices))


def test_encoder_mask_palm_and_both_branch_gradients():
    model = config.get_config("pi0_xhand_ho_v4/press_the_button").model
    encoder = model.encoder.create(rngs=nnx.Rngs(1))
    inputs = _v4_input(model)
    assert global_relations(inputs.hand_keypoints_xyz_m, inputs.object_geometry.target_xyz_m[:, :16]).shape == (1, 18, 16, 4)
    output = encoder(inputs)
    assert output.tokens.shape == (1, 10, 128)
    assert bool(output.token_mask.all())
    moved = inputs.replace(hand_keypoints_xyz_m=inputs.hand_keypoints_xyz_m.at[:, 0, :].add(0.3))
    assert not np.allclose(output.tokens, encoder(moved).tokens)
    invalid = inputs.object_geometry.replace(target_mask=jnp.zeros((1, 512), bool))
    masked = encoder(inputs.replace(object_geometry=invalid))
    assert not bool(masked.token_mask[0, 5:].any())
    assert bool(jnp.isfinite(masked.tokens).all())
    assert bool((masked.tokens[0, 5:] == 0).all())
    def loss(module):
        return jnp.mean(module(inputs).tokens ** 2)
    value, grads = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.Param))(encoder)
    assert bool(jnp.isfinite(value))
    flat = grads.flat_state()
    assert all(bool(jnp.isfinite(v.value).all()) for v in flat.values())
    for branch in ("local", "global_in"):
        assert any(bool(jnp.any(v.value != 0)) for p, v in flat.items() if p[:2] == ("object_relation_encoder", branch))


@pytest.mark.parametrize("tactile", [False, True])
def test_synthetic_pi0_loss_and_policy(tactile, monkeypatch):
    arch = "pi0_xhand_tactile_t1b_ho_v4" if tactile else "pi0_xhand_ho_v4"
    production = config.get_config(f"{arch}/press_the_button")
    cfg = dataclasses.replace(production.model, paligemma_variant="dummy", action_expert_variant="dummy",
                              dtype="float32", visual_points=256, action_horizon=2, max_token_len=8)
    siglip = pi0._siglip.Module
    monkeypatch.setattr(pi0._siglip, "Module", lambda num_classes=None, *, variant=None, **kwargs:
                        siglip(num_classes=num_classes, variant="mu/14", **kwargs))
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(1).replace(spatial=_v4_input(cfg))
    actions = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), jnp.float32)
    def loss(module):
        return jnp.mean(module.compute_loss(jax.random.key(5), obs, actions, train=False))
    value, gradients = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.Param))(model)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.isfinite(v.value).all()) for v in gradients.flat_state().values())
    for branch in ("local", "global_in"):
        assert any(bool(jnp.any(v.value != 0)) for p, v in gradients.flat_state().items()
                   if p[:3] == ("spatial_router", "encoder", "object_relation_encoder") and branch in p)
    serving = policy.Policy(model, sample_kwargs={"num_steps": 1})
    result = serving.infer(jax.tree.map(lambda value: np.asarray(value[0]), obs.to_dict()))
    assert result["actions"].shape == (cfg.action_horizon, cfg.action_dim)
    assert np.isfinite(result["actions"]).all()
    del serving, model, gradients
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("tactile", [False, True])
def test_policy_factory_with_provenance_checked_geometry(tactile, monkeypatch, tmp_path):
    from openpi import transforms
    from openpi.object_geometry import ho_v4_anchors
    from openpi.policies import policy_config

    class DummyModel(nnx.Module):
        def sample_actions(self, rng, observation, **kwargs):
            assert observation.spatial.object_geometry.manipulated_anchor_indices.shape == (1, 16)
            return jnp.zeros((observation.state.shape[0], 2, 32), dtype=jnp.float32)

    arch = "pi0_xhand_tactile_t1b_ho_v4" if tactile else "pi0_xhand_ho_v4"
    cfg = config.get_config(f"{arch}/press_the_button")
    source = _v4_input(cfg.model).object_geometry
    geom = {k: np.asarray(v[0]) for k, v in dataclasses.asdict(source).items() if v is not None and "indices" not in k}
    assets = {}
    for role in ("manipulated", "target"):
        points = geom[f"{role}_xyz_m"]
        idx = np.arange(16, dtype=np.int32)
        assets[role] = {"indices": idx, "surface_points_sha256": f"test-{role}",
                        "surface_distances_m": np.linalg.norm(points[:, None] - points[idx][None, :], axis=-1)}
    geom["surface_points_sha256"] = {role: item["surface_points_sha256"] for role, item in assets.items()}
    monkeypatch.setattr(ho_v4_anchors, "load_assets", lambda *args, **kwargs: assets)
    monkeypatch.setattr(type(cfg.model), "load", lambda self, params, **kwargs: DummyModel())
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    def to_model(_):
        return {"image": {"base_0_rgb": np.zeros((8, 8, 3), np.uint8)},
                "image_mask": {"base_0_rgb": np.ones((), dtype=np.bool_)}, "state": np.zeros(32, np.float32)}
    data = config.DataConfig(spatial=cfg.data.spatial, object_geometry=cfg.data.object_geometry,
                             data_transforms=transforms.Group(inputs=[to_model]))
    monkeypatch.setattr(type(cfg.data), "create", lambda self, assets_dirs, model: data)
    serving = policy_config.create_trained_policy(cfg, tmp_path, norm_stats={})
    observation = {"observation.state": np.zeros(1972, np.float32), "object_geometry": geom,
        "hand_keypoints": {"xyz_m": np.zeros((18, 3), np.float32), "point_mask": np.ones(18, bool)}}
    for camera in ("cam_front", "cam_left", "cam_right"):
        observation[f"observation.images.{camera}"] = np.zeros((8, 8, 3), np.uint8)
    result = serving.infer(observation)
    assert result["actions"].shape == (2, 32)
    assert np.isfinite(result["actions"]).all()
