import hashlib
import json
from pathlib import Path
import sys

from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models.spatial_encoders.structured_spatial_encoder_test import _synthetic_input
from openpi.training import config
from openpi.training import data_loader
from openpi.training import spatial_weight_loaders

TASKS = {
    "peg_all": "1018_peg_all",
    "press_the_button": "1028_press_button",
    "yellow_socket": "1030_yellow_socket",
    "eraser": "1102_eraser",
}
CLOUD_ROOT = Path("/workspace/mnt/sqzhang26/openpi-3d-tactile")
NAMES = [f"pi0_xhand_pc_v2/{task}" for task in TASKS] + [
    f"pi0_xhand_pc_v2_tactile_v2/{task}" for task in ("press_the_button", "eraser")
]


@pytest.mark.parametrize("name", NAMES)
def test_registered_cli_and_training_contract(name, monkeypatch):
    selected = config.get_config(name)
    base = config.get_config("pi0_xhand_1028_press_button_st_v2")
    # Keep Tyro's real CLI path, but avoid rebuilding every unrelated config's
    # large parser for each of the six parametrized cases.
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: selected})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    cfg = config.cli()
    model = cfg.model
    task = name.split("/")[1]
    tactile = name.startswith("pi0_xhand_pc_v2_tactile_v2/")
    assert cfg.name == name
    assert cfg.checkpoint_dir == (Path.cwd() / "checkpoints" / name / "seed42").resolve()
    assert cfg.data.repo_id == str(CLOUD_ROOT / "data" / TASKS[task])
    assert cfg.data.assets == config.AssetsConfig(
        assets_dir=str(CLOUD_ROOT / "assets/xhand"), asset_id=f"{TASKS[task]}_unified_v1"
    )
    assert cfg.data.spatial.dataset_root == cfg.data.repo_id
    assert cfg.data.spatial.version == "v1"
    assert cfg.data.spatial.camera_roles == ("front",)
    assert cfg.data.spatial.front_calibration_profile == config.FRONT_CALIBRATION_PROFILE
    assert cfg.data.spatial.require_source_metadata_match
    assert cfg.data.object_geometry is None
    assert model.use_visual
    assert model.use_tactile == tactile
    assert model.encoder.enable_tactile == tactile
    assert model.encoder.visual_hierarchy_v2
    assert not model.encoder.enable_object_relations
    assert model.encoder.visual_num_centers == 32
    assert model.encoder.default_token_count == (58 if tactile else 33)
    assert model.conditioning.target == "suffix"
    assert model.paligemma_variant == "gemma_2b_lora"
    assert model.action_expert_variant == "gemma_300m"
    assert cfg.freeze_filter == model.get_freeze_filter()
    assert (cfg.batch_size, cfg.num_train_steps, cfg.seed, cfg.save_interval, cfg.keep_period, cfg.ema_decay) == (
        8, 20_000, 42, 5_000, 5_000, None
    )
    assert isinstance(cfg.weight_loader, spatial_weight_loaders.SpatialCheckpointWeightLoader)
    assert cfg.weight_loader.params_path == "/workspace/mnt/sqzhang26/hf_weight/pi0_base/params"
    assert cfg.optimizer == base.optimizer
    assert cfg.lr_schedule == base.lr_schedule
    assert model.action_horizon == base.model.action_horizon
    assert cfg.data.prompt_from_task
    assert not cfg.data.use_delta_arm_actions
    assert not cfg.data.delta_hand_actions
    if tactile:
        assert model == base.model


@pytest.mark.parametrize("name", ["pi0_xhand_pc_v2/eraser", "pi0_xhand_pc_v2_tactile_v2/eraser"])
def test_production_encoder_shapes_and_parameters(name):
    cfg = config.get_config(name)
    obs, _ = cfg.model.inputs_spec(batch_size=1)
    encoder = cfg.model.encoder.create(rngs=nnx.Rngs(0))
    out = jax.eval_shape(encoder, obs.spatial)
    assert out.tokens.shape == (1, 58 if cfg.model.use_tactile else 33, 128)
    assert out.token_mask.shape == out.tokens.shape[:2]
    assert obs.spatial.object_geometry is None
    assert obs.spatial.hand_keypoints_xyz_m is None
    params = nnx.state(encoder, nnx.Param).flat_state()
    assert not any("object_relation_encoder" in path for path in params)
    if not cfg.model.use_tactile:
        assert not any(any(str(part).startswith("tactile_") or part == "finger_embedding" for part in path)
                       for path in params)


@pytest.mark.parametrize("tactile", [False, True])
def test_production_encoder_forward_with_4096_points(tactile):
    name = ("pi0_xhand_pc_v2_tactile_v2/press_the_button" if tactile
            else "pi0_xhand_pc_v2/press_the_button")
    encoder = config.get_config(name).model.encoder.create(rngs=nnx.Rngs(0))
    inputs = _synthetic_input(batch_size=1, visual_points=4096, tactile_points=600)
    if tactile:
        inputs = inputs.replace(tactile=inputs.tactile.replace(
            finger_id=jax.numpy.repeat(jax.numpy.arange(5), 120)[None, :]
        ))
    else:
        inputs = inputs.replace(tactile=None)
    output = encoder(inputs)
    assert output.tokens.shape == (1, 58 if tactile else 33, 128)
    assert bool(output.token_mask.all())
    assert bool(jax.numpy.isfinite(output.tokens).all())


def _write_spatial_sidecar(root):
    sidecar = root / "spatial/v1"
    static = sidecar / "static"
    static.mkdir(parents=True)
    np.save(static / "finger_id.npy", np.repeat(np.arange(5, dtype=np.int8), 120))
    np.save(static / "taxel_id.npy", np.tile(np.arange(1, 121, dtype=np.int16), 5))
    episodes = []
    for ep in range(2):
        ep_dir = sidecar / f"episodes/e{ep}"
        shard = ep_dir / "shard_0"
        shard.mkdir(parents=True)
        arrays = {
            "frame_index": np.arange(2, dtype=np.int64),
            "timestamp_s": np.arange(2, dtype=np.float64) / 30,
            "visual_xyz_m": np.stack([np.full((4, 3), 10 * ep + frame, np.float32) for frame in range(2)]),
            "visual_rgb": np.zeros((2, 4, 3), np.uint8),
            "visual_rgb_valid": np.ones((2, 4), bool),
            "tactile_xyz_m": np.zeros((2, 600, 3), np.float32),
            "tactile_force_base": np.zeros((2, 600, 3), np.float32),
            "tactile_force_norm": np.zeros((2, 600), np.float32),
        }
        for field, value in arrays.items():
            np.save(shard / f"{field}.npy", value)
        (ep_dir / "manifest.json").write_text(json.dumps({
            "episode_index": ep, "selected_frame_count": 2,
            "shards": [{"name": "shard_0", "count": 2, "first_frame": 0, "last_frame": 1}],
        }))
        episodes.append({"episode_index": ep, "path": f"episodes/e{ep}"})
    provenance = {name: {"sha256": hashlib.sha256((root / "meta" / name).read_bytes()).hexdigest()}
                  for name in ("info.json", "episodes.jsonl")}
    (sidecar / "manifest.json").write_text(json.dumps({
        "status": "complete", "derived_schema_version": 1, "spatial_version": "v1",
        "source_dataset": {"source_meta_sha256": provenance},
        "spatial_contract": {"coordinate_frame": "base_link", "xyz_unit": "m", "force_unit": "dataset_native",
                             "visual_points_per_frame": 4, "tactile_points_per_frame": 600, "camera_roles": ["front"]},
        "storage": {"format": "npy_shards_v1", "static": {
            "finger_id": {"file": "static/finger_id.npy"}, "taxel_id": {"file": "static/taxel_id.npy"},
        }, "episodes": episodes},
        "runtime": {"total_selected_frames": 4},
    }))


def test_real_local_lerobot_spatial_join_and_stale_metadata(tmp_path):
    root = tmp_path / "1102_eraser"
    writer = data_loader.lerobot_dataset.LeRobotDataset.create(
        repo_id="local/eraser", root=root, fps=30, use_videos=False,
        features={"observation.state": {"dtype": "float32", "shape": (1972,), "names": None},
                  "action": {"dtype": "float32", "shape": (18,), "names": None}},
    )
    for ep in range(2):
        for frame in range(2):
            writer.add_frame({"observation.state": np.full(1972, ep * 10 + frame, np.float32),
                              "action": np.zeros(18, np.float32), "task": "erase"})
        writer.save_episode()
    _write_spatial_sidecar(root)
    spatial = config.SpatialDataConfig(dataset_root=str(root), camera_roles=("front",),
                                       require_source_metadata_match=True)
    data = config.DataConfig(repo_id=str(root), spatial=spatial, action_sequence_keys=("action",), prompt_from_task=True)
    dataset = data_loader.create_torch_dataset(data, 2, config.get_config("pi0_xhand_pc_v2/eraser").model)
    assert len(dataset) == 4
    for index in (0, 1, 2, 3):
        item = dataset[index]
        ep, frame = divmod(index, 2)
        assert int(item["episode_index"]) == ep
        assert int(item["frame_index"]) == frame
        assert item["prompt"] == "erase"
        assert item["spatial"]["visual"]["xyz_m"][0, 0] == ep * 10 + frame
        assert "tactile" not in item["spatial"]
    tactile_model = config.get_config("pi0_xhand_pc_v2_tactile_v2/eraser").model
    tactile_dataset = data_loader.create_torch_dataset(data, 2, tactile_model)
    assert tactile_dataset[3]["spatial"]["tactile"]["force"].shape == (600, 3)

    ep_manifest_path = root / "spatial/v1/episodes/e1/manifest.json"
    root_manifest_path = root / "spatial/v1/manifest.json"
    ep_original = ep_manifest_path.read_text()
    root_original = root_manifest_path.read_text()
    ep_manifest = json.loads(ep_original)
    root_manifest = json.loads(root_original)
    ep_manifest["selected_frame_count"] = 1
    ep_manifest["shards"][0].update(count=1, last_frame=0)
    root_manifest["runtime"]["total_selected_frames"] = 3
    ep_manifest_path.write_text(json.dumps(ep_manifest))
    root_manifest_path.write_text(json.dumps(root_manifest))
    with pytest.raises(ValueError, match="covers 1 frames of episode 1"):
        data_loader.create_torch_dataset(data, 2, config.get_config("pi0_xhand_pc_v2/eraser").model)
    ep_manifest_path.write_text(ep_original)
    root_manifest_path.write_text(root_original)

    episodes_path = root / "meta/episodes.jsonl"
    episodes_path.write_text(episodes_path.read_text().replace("\n", " \n"))
    with pytest.raises(ValueError, match="does not match current"):
        data_loader.create_torch_dataset(data, 2, config.get_config("pi0_xhand_pc_v2/eraser").model)


def test_missing_data_root_does_not_fall_back_to_hub(monkeypatch):
    assert data_loader.resolve_local_lerobot_root("data/1102_eraser").name == "1102_eraser"
    assert data_loader.resolve_local_lerobot_root("owner/remote") is None
    monkeypatch.setattr(data_loader, "resolve_local_lerobot_root", lambda _: Path("/does-not-exist/eraser"))
    with pytest.raises(FileNotFoundError, match="no Hub lookup"):
        data_loader.create_torch_dataset(config.DataConfig(repo_id="data/1102_eraser"), 2, object())
