# ruff: noqa: SLF001
"""Contracts for the eight hand-object and coordinate ablations."""
from pathlib import Path
import sys

from flax import nnx
import jax
import pytest

from openpi.training import config
from openpi.training import spatial_weight_loaders

ARCHITECTURES = {
    "pi0_xhand_ho_v3": 10,
    "pi0_xhand_tactile_t1b": 5,
    "pi0_xhand_tactile_local_v1": 5,
    "pi0_xhand_pc_v2_tactile_local_v1": 38,
}
TASKS = {"press_the_button": "1028_press_button", "yellow_socket": "1030_yellow_socket"}
ROOT = Path("/workspace/mnt/sqzhang26/openpi-3d-tactile")
NAMES = [f"{a}/{t}" for a in ARCHITECTURES for t in TASKS]


@pytest.mark.parametrize("name", NAMES)
def test_cli_configuration(name, monkeypatch):
    selected = config.get_config(name)
    architecture, task = name.split("/")
    base = config.get_config(f"pi0_xhand_pc_v2/{task}")
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: selected})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    cfg = config.cli()
    assert cfg.checkpoint_dir == (Path.cwd() / "checkpoints" / name / "seed42").resolve()
    assert cfg.data.repo_id == str(ROOT / "data" / TASKS[task])
    assert cfg.data.assets.assets_dir == str(ROOT / "assets/xhand")
    assert cfg.data.assets.asset_id == f"{TASKS[task]}_unified_v1"
    assert cfg.data.spatial.dataset_root == cfg.data.repo_id
    assert cfg.data.spatial.version == "v1"
    assert cfg.data.spatial.camera_roles == ("front",)
    assert cfg.data.spatial.front_calibration_profile == config.FRONT_CALIBRATION_PROFILE
    assert cfg.data.spatial.require_source_metadata_match
    assert cfg.data.spatial.tactile_coordinate_frame == ("finger_link2" if "local_v1" in architecture else "base")
    model = cfg.model
    assert model.encoder.default_token_count == ARCHITECTURES[architecture]
    assert model.paligemma_variant == "gemma_2b_lora"
    assert model.action_expert_variant == "gemma_300m"
    assert cfg.freeze_filter == model.get_freeze_filter()
    assert cfg.weight_loader == spatial_weight_loaders.SpatialCheckpointWeightLoader(
        "/workspace/mnt/sqzhang26/hf_weight/pi0_base/params")
    assert (cfg.batch_size, cfg.num_train_steps, cfg.seed, cfg.save_interval, cfg.keep_period, cfg.ema_decay) == (
        8, 20000, 42, 5000, 5000, None)
    assert (cfg.optimizer, cfg.lr_schedule, model.action_horizon) == (
        base.optimizer, base.lr_schedule, base.model.action_horizon)
    assert (cfg.data.object_geometry is not None) == (architecture == "pi0_xhand_ho_v3")
    obs, _ = model.inputs_spec()
    enc = model.encoder.create(rngs=nnx.Rngs(0))
    out = jax.eval_shape(enc, obs.spatial)
    assert out.tokens.shape == (1, ARCHITECTURES[architecture], 128)
    paths = nnx.state(enc, nnx.Param).flat_state()
    if not model.use_visual:
        assert not any(str(p[0]).startswith("visual_") for p in paths)
    if not model.use_tactile:
        assert not any(str(p[0]).startswith("tactile_") or p[0] == "finger_embedding" for p in paths)
    if architecture == "pi0_xhand_ho_v3":
        assert not any(str(part).startswith("oo_") for p in paths for part in p)


def test_tactile_control_encoder_is_identical():
    base = config.get_config("pi0_xhand_tactile_t1b/press_the_button").model.encoder
    local = config.get_config("pi0_xhand_tactile_local_v1/press_the_button").model.encoder
    previous = config.get_config("pi0_xhand_1028_press_button_st_t1b_v1").model.encoder
    assert base == local
    assert base == __import__("dataclasses").replace(previous, enable_visual=False)
    assert jax.tree.structure(nnx.state(base.create(rngs=nnx.Rngs(0)), nnx.Param)) == jax.tree.structure(
        nnx.state(local.create(rngs=nnx.Rngs(1)), nnx.Param))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_policy_factory_selects_preprocessing(architecture, monkeypatch, tmp_path):
    from openpi.policies import policy_config

    selected = config.get_config(f"{architecture}/press_the_button")
    data = config.DataConfig(spatial=selected.data.spatial, object_geometry=selected.data.object_geometry)
    monkeypatch.setattr(type(selected.data), "create", lambda self, assets, model: data)
    monkeypatch.setattr(type(selected.model), "load", lambda self, params: object())
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(policy_config._policy, "Policy", lambda model, **kwargs: kwargs)
    built = policy_config.create_trained_policy(selected, tmp_path, norm_stats={})
    transform = built["transforms"][0]
    assert transform.use_visual == selected.model.use_visual
    assert transform.use_tactile == selected.model.use_tactile
    assert transform.tactile_coordinate_frame == selected.data.spatial.tactile_coordinate_frame
    assert transform.require_object_geometry == selected.model.encoder.enable_object_relations
    assert transform.require_hand_keypoints == selected.model.encoder.enable_object_relations
    assert not transform.require_hand_scene
    assert built["metadata"]["rgb_only_baseline"] is False
