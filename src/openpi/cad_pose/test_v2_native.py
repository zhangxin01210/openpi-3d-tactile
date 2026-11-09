"""Native V2 predictions, real local LeRobot joins, and provenance rejection."""

import dataclasses
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

import cad_pose_rgbd_single_v2 as model_v2
from openpi.cad_pose import v2_native
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.spatial.kinematics import RobotKinematics
from openpi.training import config
from openpi.training import data_loader
from openpi.training.pc_v2_config_test import _write_spatial_sidecar


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_sources(tmp_path, root, episodes):
    task = root.name
    cache, candidate = tmp_path / "cache", tmp_path / "candidate"
    cache.mkdir()
    candidate.mkdir()
    checkpoint = tmp_path / "pose.pt"
    model = model_v2.SingleCadRGBDPose()
    torch.save(
        {
            "schema": v2_native.SCHEMA,
            "model_state_dict": model.state_dict(),
            "width": 64,
            "height": 64,
            "depth_scale": 0.001,
            "script_sha256": digest(Path(model_v2.__file__)),
            "tasks": tuple(v2_native.TASKS),
        },
        checkpoint,
    )
    (cache / "cache_config.json").write_text(
        json.dumps({"schema": v2_native.SCHEMA, "width": 64, "height": 64, "script_sha256": "cache_script"})
    )
    (cache / "episode_splits.json").write_text(json.dumps({task: {"train": list(episodes), "val": [], "test": []}}))
    cad = np.random.default_rng(42).normal(0, 0.01, (2, 512, 3)).astype(np.float32)
    (cache / task).mkdir()
    np.save(cache / task / "cad.npy", cad)
    (candidate / task).mkdir()
    provenance = {
        "schema": "v2_pose_pred_full_candidate_v1",
        "tasks": list(v2_native.TASKS),
        "status": "candidate_unaccepted_not_native_object_geometry",
        "checkpoint_schema": v2_native.SCHEMA,
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_script_sha256": digest(Path(model_v2.__file__)),
        "rgbd_cache_config_sha256": digest(cache / "cache_config.json"),
        "rgbd_episode_splits_sha256": digest(cache / "episode_splits.json"),
        "cad_sha256_by_task": {task: digest(cache / task / "cad.npy")},
        "coordinate_frame": "base_link",
        "length_unit": "meter",
        "role_names": list(v2_native.ROLES),
        "role_to_object_by_task": {task: list(v2_native.TASKS[task])},
        "image_width": 64,
        "image_height": 64,
        "depth_scale": 0.001,
        "inference_inputs_only": ["rgb", "raw_depth", "CAD"],
    }
    (candidate / "provenance.json").write_text(json.dumps(provenance))
    (candidate / "summary.json").write_text(
        json.dumps(
            {
                "schema": "v2_pose_pred_full_candidate_v1",
                "status": "candidate_complete_unaccepted",
                "by_task": {task: {"episodes": len(episodes), "frames": len(episodes) * 2}},
                "all_split_episodes_exported": True,
                "source_checkpoint_sha256": digest(checkpoint),
                "source_cache_config_sha256": digest(cache / "cache_config.json"),
            }
        )
    )
    for ep in episodes:
        path = candidate / task / f"e{ep:05d}_candidate.npz"
        transform = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
        xyz = np.tile(cad[None], (2, 1, 1, 1))
        np.savez(
            path,
            frame_index=np.arange(2),
            candidate_xyz_m=xyz,
            pred_T_base_object=transform,
            cad_xyz_local_m=cad,
            role_names=np.asarray(v2_native.ROLES),
            object_names=np.asarray(v2_native.TASKS[task]),
            existing_gt_T_base_object=np.full((2, 2, 4, 4), np.nan),
            existing_gt_label_valid=np.zeros((2, 2), bool),
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema": provenance["schema"],
                    "task": task,
                    "episode": ep,
                    "frames": 2,
                    "checkpoint_sha256": digest(checkpoint),
                    "rgbd_cache_config_sha256": provenance["rgbd_cache_config_sha256"],
                    "cad_sha256": provenance["cad_sha256_by_task"][task],
                    "file_sha256": digest(path),
                }
            )
        )
    for ep in episodes:
        rgbd = cache / task / f"e{ep:05d}"
        rgbd.mkdir()
        np.save(rgbd / "rgb.npy", np.zeros((2, 64, 64, 3), np.uint8))
        np.save(rgbd / "depth.npy", np.zeros((2, 64, 64), np.uint16))
    return candidate, cache, checkpoint, cad


def write_old_geometry(root, episodes):
    path = root / "object_geometry/v1"
    path.mkdir(parents=True)
    records = []
    for ep in episodes:
        folder = path / f"s{ep}"
        folder.mkdir()
        np.save(folder / "frame_index.npy", np.arange(2))
        np.save(folder / "episode_index.npy", np.full(2, ep))
        for role in v2_native.ROLES:
            np.save(folder / f"{role}_xyz_m.npy", np.full((2, 512, 3), -99, np.float32))
            for field, dtype, value in (
                ("mask", bool, False),
                ("source", np.int8, 0),
                ("age", np.int16, 0),
                ("confidence", np.float32, 0),
            ):
                np.save(folder / f"{role}_{field}.npy", np.full((2, 512), value, dtype=dtype))
        records.extend({"episode_index": ep, "frame_index": f, "shard": f"s{ep}", "row": f} for f in range(2))
    from openpi.object_geometry.calibration import front_calibration_provenance

    repo = Path(__file__).resolve().parents[3]
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "object_geometry_contract": {"point_count": 512},
                "role_to_object": dict(zip(v2_native.ROLES, v2_native.TASKS[root.name], strict=True)),
                "calibration": front_calibration_provenance(repo, None),
                "records": records,
            }
        )
    )


def write_hand(root, episodes):
    repo = Path(__file__).resolve().parents[3]
    kin = RobotKinematics(repo / "configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    q = dict.fromkeys(kin.required_measured_joints(root="base_link"), 0.0)
    select = runpy.run_path(str(repo / "scripts/spatial/audit_xhand_keypoints.py"))["select_keypoints"]
    names, groups, *_ = select(kin, kin.compute(q, root="base_link"))
    folder = root / "hand_keypoints/v1"
    folder.mkdir(parents=True)
    entries = []
    for ep in episodes:
        np.savez(folder / f"e{ep}.npz", frame_index=np.arange(2), xyz_m=np.full((2, 18, 3), ep, np.float32))
        entries.append({"episode_index": ep, "frames": 2, "file": f"e{ep}.npz"})
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "keypoint_count": 18,
                "coordinate_frame": "base_link",
                "unit": "m",
                "spatial_version": "v1",
                "object_geometry_version": "v1",
                "names": names,
                "groups": groups,
                "urdf_sha256": digest(repo / "configs/ur7e_xhand/ur7e_xhand_verified.urdf"),
                "state_mapping_sha256": digest(repo / "configs/ur7e_xhand/state_mapping.csv"),
                "episodes": entries,
            }
        )
    )


def fake_raw_source_verification(root, cache, task, episode, cfg):
    result = {}
    for key, relative in (("parquet", "meta/info.json"), ("front_video", "meta/episodes.jsonl")):
        path = Path(root) / relative
        stat = path.stat()
        result[key] = {"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}
    return result


def test_real_lerobot_native_v2_join_and_provenance(tmp_path, monkeypatch):
    root = tmp_path / "1028_press_button"
    writer = data_loader.lerobot_dataset.LeRobotDataset.create(
        repo_id="local/button",
        root=root,
        fps=30,
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (1972,), "names": None},
            "action": {"dtype": "float32", "shape": (18,), "names": None},
            "observation.images.cam_front": {"dtype": "image", "shape": (8, 8, 3), "names": None},
        },
    )
    for ep in range(2):
        for frame in range(2):
            writer.add_frame(
                {
                    "observation.state": np.full(1972, ep * 10 + frame, np.float32),
                    "action": np.full(18, frame, np.float32),
                    "observation.images.cam_front": np.full((8, 8, 3), ep * 10 + frame, np.uint8),
                    "task": "press",
                }
            )
        writer.save_episode()
    _write_spatial_sidecar(root)
    write_old_geometry(root, range(2))
    write_hand(root, range(2))
    candidate, cache, checkpoint, cad = fixture_sources(tmp_path, root, range(2))
    monkeypatch.setattr(v2_native, "verify_raw_cache", fake_raw_source_verification)
    real_frame_ids = v2_native.raw_frame_ids

    def interrupted(dataset_root, episode):
        if episode == 1:
            raise RuntimeError("simulated interruption")
        return real_frame_ids(dataset_root, episode)

    monkeypatch.setattr(v2_native, "raw_frame_ids", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        v2_native.publish_task(
            candidate_root=candidate,
            cache_root=cache,
            checkpoint=checkpoint,
            dataset_root=root,
            task=root.name,
            device="cpu",
        )
    staged = root / "object_geometry/.pretrained_pose_v2.staging/shard_e00000/manipulated_xyz_m.npy"
    before_resume = digest(staged)
    monkeypatch.setattr(v2_native, "raw_frame_ids", real_frame_ids)
    path = v2_native.publish_task(
        candidate_root=candidate,
        cache_root=cache,
        checkpoint=checkpoint,
        dataset_root=root,
        task=root.name,
        device="cpu",
    )
    assert digest(path / "shard_e00000/manipulated_xyz_m.npy") == before_resume
    assert (
        v2_native.publish_task(
            candidate_root=candidate,
            cache_root=cache,
            checkpoint=checkpoint,
            dataset_root=root,
            task=root.name,
            device="cpu",
        )
        == path
    )
    native = ObjectGeometryDerivedDataset(root, version="pretrained_pose_v2")
    item = native.get(episode_index=1, frame_index=1)
    assert np.array_equal(item.manipulated_xyz_m, cad[0])
    assert np.array_equal(item.target_xyz_m, cad[1])
    assert item.manipulated_mask.all()
    assert item.target_mask.all()
    assert not item.manipulated_confidence.any()
    assert native.manifest["gt_used"] is False
    spatial = config.SpatialDataConfig(
        dataset_root=str(root), camera_roles=("front",), require_source_metadata_match=True
    )
    geo = config.ObjectGeometryDataConfig(
        dataset_root=str(root),
        version="pretrained_pose_v2",
        require_pose_prediction=True,
        pose_prediction_schema="v2",
        pose_checkpoint_path=str(checkpoint),
        pose_v2_cache_root=str(cache),
        expected_roles=tuple(zip(v2_native.ROLES, v2_native.TASKS[root.name], strict=True)),
        hand_keypoint_object_version="v1",
    )
    data = config.DataConfig(
        repo_id=str(root), spatial=spatial, object_geometry=geo, action_sequence_keys=("action",), prompt_from_task=True
    )
    model = config.get_config("pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/press_the_button").model
    predicted = data_loader.create_torch_dataset(data, 2, model)
    base_geo = dataclasses.replace(geo, version="v1", require_pose_prediction=False, pose_prediction_schema="v1")
    baseline = data_loader.create_torch_dataset(dataclasses.replace(data, object_geometry=base_geo), 2, model)
    for i in range(4):
        a, b = predicted[i], baseline[i]
        for key in ("observation.state", "action", "observation.images.cam_front", "episode_index", "frame_index"):
            assert np.array_equal(a[key], b[key])
        for key in ("visual", "tactile", "hand_keypoints"):
            for field in a["spatial"][key]:
                assert np.array_equal(a["spatial"][key][field], b["spatial"][key][field])
        obj = a["spatial"]["object_geometry"]
        assert np.array_equal(obj["manipulated_xyz_m"], cad[0])
        assert np.array_equal(obj["target_xyz_m"], cad[1])
        assert obj["manipulated_mask"].all()
        assert obj["target_mask"].all()
    from types import SimpleNamespace

    from etils import epath

    from openpi.training import checkpoints as policy_checkpoints

    class Manager:
        def save(self, step, items):
            items["assets"](epath.Path(tmp_path / "checkpoint_assets"))

    class Loader:
        def data_config(self):
            return SimpleNamespace(norm_stats=None, asset_id="button", object_geometry=geo)

    monkeypatch.setattr(policy_checkpoints, "_split_params", lambda state: (None, None))
    policy_checkpoints.save_state(Manager(), object(), Loader(), 0)
    saved_provenance = json.loads((tmp_path / "checkpoint_assets/button/pose_provenance.json").read_text())
    assert saved_provenance["pose_checkpoint_sha256"] == digest(checkpoint)
    assert "records" not in saved_provenance
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="provenance"):
        data_loader.create_torch_dataset(data, 2, model)
    checkpoint.write_bytes(original)
    original_cad = (cache / root.name / "cad.npy").read_bytes()
    (cache / root.name / "cad.npy").write_bytes(original_cad + b"changed")
    with pytest.raises(ValueError, match="provenance"):
        data_loader.create_torch_dataset(data, 2, model)
    (cache / root.name / "cad.npy").write_bytes(original_cad)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    broken = json.loads(json.dumps(manifest))
    broken["records"][0]["row"] = 1
    manifest_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="shard/row"):
        data_loader.create_torch_dataset(data, 2, model)
    manifest["preprocess"]["depth_scale"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        data_loader.create_torch_dataset(data, 2, model)


@pytest.mark.parametrize("task", ["peg_all", "press_the_button", "yellow_socket", "eraser"])
def test_config_only_changes_v2_geometry_source(task):
    old = config.get_config(f"pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain/{task}")
    new = config.get_config(f"pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/{task}")
    assert new.model == old.model
    assert new.freeze_filter == old.freeze_filter
    assert new.optimizer == old.optimizer
    assert new.lr_schedule == old.lr_schedule
    assert new.data.spatial == old.data.spatial
    assert new.data.assets == old.data.assets
    assert new.data.repo_id == old.data.repo_id
    assert new.data.object_geometry.version == "pretrained_pose_v2"
    assert new.data.object_geometry.pose_prediction_schema == "v2"
    assert old.data.object_geometry.version == "pretrained_pose_v1"


def test_cleaned_button_88_episode_publication(tmp_path, monkeypatch):
    from openpi.spatial_dataset.source import RawSpatialDataset

    root = tmp_path / "1028_press_button"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": ep, "length": 2}) + "\n" for ep in range(88))
    )
    episodes = tuple(range(88))
    candidate, cache, checkpoint, cad = fixture_sources(tmp_path, root, episodes)
    monkeypatch.setattr(RawSpatialDataset, "discover_episode_ids", lambda root: episodes)
    monkeypatch.setattr(v2_native, "raw_frame_ids", lambda root, ep: np.arange(2))
    monkeypatch.setattr(v2_native, "verify_raw_cache", fake_raw_source_verification)
    path = v2_native.publish_task(
        candidate_root=candidate,
        cache_root=cache,
        checkpoint=checkpoint,
        dataset_root=root,
        task=root.name,
        device="cpu",
    )
    native = ObjectGeometryDerivedDataset(root, version="pretrained_pose_v2")
    assert len(native.episode_ids) == 88
    assert len(native) == 176
    assert np.array_equal(native.get(episode_index=87, frame_index=1).target_xyz_m, cad[1])
    v2_native.validate_native_manifest(native.manifest, root, checkpoint, cache, reader=native)
    xyz_path = path / "shard_e00087/target_xyz_m.npy"
    xyz = np.load(xyz_path)
    xyz[1, 0, 0] += 1
    np.save(xyz_path, xyz)
    with pytest.raises(ValueError, match="modified native V2 geometry"):
        v2_native.validate_native_manifest(native.manifest, root, checkpoint, cache, reader=native)


@pytest.mark.parametrize("task", ["peg_all", "press_the_button", "yellow_socket", "eraser"])
def test_v2_configs_parse_with_real_tyro(task, monkeypatch):
    import sys

    name = f"pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/{task}"
    selected = config.get_config(name)
    monkeypatch.setattr(config, "_CONFIGS_DICT", {name: selected})
    monkeypatch.setattr(sys, "argv", ["train.py", name, "--exp-name", "seed42"])
    parsed = config.cli()
    assert parsed.name == name
    assert parsed.exp_name == "seed42"


def test_raw_rgbd_cache_verification_rejects_changed_pixels(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cv2

    from openpi.spatial_dataset import source

    root = tmp_path / "dataset"
    root.mkdir()
    parquet, video_file = root / "data.parquet", root / "front.mp4"
    parquet.write_bytes(b"parquet")
    video_file.write_bytes(b"video")
    rgb = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    depth = np.arange(12 * 12, dtype=np.uint16).reshape(12, 12)
    folder = tmp_path / "cache" / "task" / "e00000"
    folder.mkdir(parents=True)
    np.save(folder / "rgb.npy", np.stack([cv2.resize(rgb, (8, 8), interpolation=cv2.INTER_AREA)]))
    np.save(folder / "depth.npy", np.stack([cv2.resize(depth, (8, 8), interpolation=cv2.INTER_NEAREST)]))

    class Raw:
        def __init__(self, root, episode):
            self.root, self.parquet_path = Path(root), parquet

        def video_path(self, role):
            return video_file

        def iter_rows(self, *, depth_roles):
            yield {"frame_index": 0, "observation.depths.cam_front": depth}

        def iter_video_frames(self, role):
            yield SimpleNamespace(frame_index=0, rgb=rgb)

    monkeypatch.setattr(source, "RawSpatialDataset", Raw)
    report = v2_native.verify_raw_cache(root, tmp_path / "cache", "task", 0, {"height": 8, "width": 8})
    assert report["front_video"]["sha256"] == digest(video_file)
    v2_native.validate_raw_files(root, {"0": report})
    video_file.write_bytes(b"VIDEO")
    with pytest.raises(ValueError, match="raw front_video changed"):
        v2_native.validate_raw_files(root, {"0": report})
    video_file.write_bytes(b"video")
    cached = np.load(folder / "rgb.npy")
    cached[0, 0, 0, 0] ^= 1
    np.save(folder / "rgb.npy", cached)
    with pytest.raises(ValueError, match="current RGB-D differs"):
        v2_native.verify_raw_cache(root, tmp_path / "cache", "task", 0, {"height": 8, "width": 8})
