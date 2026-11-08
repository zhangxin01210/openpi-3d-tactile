# ruff: noqa: PT018, RUF012
"""End-to-end synthetic execution; no real-data claims or GPU required."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from openpi.cad_pose.pipeline import DEFAULT_TASKS
from openpi.cad_pose.pipeline import open_tasks
from openpi.cad_pose.pipeline import splits_for
from openpi.cad_pose.pipeline import train


def test_training_resume_and_holdout(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    tasks, _ = open_tasks(repo, repo / "data", DEFAULT_TASKS, synthetic=True)
    splits = splits_for(tasks)
    output = tmp_path / "training"
    first = train(
        tasks=tasks,
        splits=splits,
        output=output,
        device="cpu",
        epochs=1,
        batch_size=4,
        steps=2,
        val_batches=1,
        synthetic=True,
    )
    path = output / "checkpoints/epoch_0000.pt"
    assert path.is_file() and (output / "best.json").is_file()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 0 and checkpoint["synthetic"]
    log = json.loads((output / "training.jsonl").read_text().splitlines()[0])
    assert set(log["task_sample_counts"].values()) == {2}
    before = next(first.parameters()).detach().clone()
    resumed = train(
        tasks=tasks,
        splits=splits,
        output=output,
        device="cpu",
        epochs=2,
        batch_size=4,
        steps=2,
        val_batches=1,
        synthetic=True,
        resume=path,
    )
    assert (output / "checkpoints/epoch_0001.pt").is_file()
    assert not torch.equal(before, next(resumed.parameters()))
    for split in splits.values():
        assert not (set(split["train"]) & set(split["test"]))
        assert not (set(split["val"]) & set(split["test"]))
    with pytest.raises(ValueError, match="mismatch"):
        train(
            tasks=tasks,
            splits=splits,
            output=output,
            device="cpu",
            epochs=3,
            batch_size=8,
            steps=2,
            val_batches=1,
            synthetic=True,
            resume=path,
        )
    with pytest.raises(FileExistsError):
        train(tasks=tasks, splits=splits, output=output, epochs=1, steps=2, batch_size=4, synthetic=True)


def test_evaluate_visualize_real_predictions_on_synthetic_frames(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    tasks, _ = open_tasks(repo, repo / "data", DEFAULT_TASKS, synthetic=True)
    output = tmp_path / "train"
    train(
        tasks=tasks,
        splits=splits_for(tasks),
        output=output,
        epochs=1,
        steps=1,
        batch_size=4,
        val_batches=1,
        synthetic=True,
    )
    checkpoint = output / "checkpoints/epoch_0000.pt"
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    evaluate = tmp_path / "eval"
    subprocess.run(
        [
            sys.executable,
            "scripts/cad_pose/evaluate.py",
            "--synthetic",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(evaluate),
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((evaluate / "evaluation.json").read_text())
    assert report["synthetic"] and report["metrics"]["overall"]["translation_mm"]["count"] == 256
    for name in DEFAULT_TASKS:
        assert name in report["metrics"]
    visual = tmp_path / "visual"
    subprocess.run(
        [
            sys.executable,
            "scripts/cad_pose/visualize.py",
            "--synthetic",
            "--checkpoint",
            str(checkpoint),
            "--evaluation-dir",
            str(evaluate),
            "--output-dir",
            str(visual),
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (visual / "index.html").is_file()
    assert len(list(visual.rglob("*.png"))) >= 256
    assert "SYNTHETIC" in (visual / "index.html").read_text()
    assert all(
        np.isfinite(report["metrics"]["overall"][k]["mean"])
        for k in ("translation_mm", "rotation_deg", "add_mm", "adds_mm")
    )


def test_predicted_sidecar_contains_all_raw_frames_without_pose_labels(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openpi.cad_pose import sidecar
    from openpi.cad_pose.geometry import predicted_geometry_packet
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

    task = SimpleNamespace(
        root=tmp_path / "data" / "1028_press_button",
        dataset="1028_press_button",
        roles={"manipulated": "button", "target": "box"},
        episodes=(0,),
    )
    task.root.mkdir(parents=True)

    class Raw:
        def __init__(self, root, *, episode):
            pass

        def iter_rows(self, *, depth_roles):
            for f in range(3):
                yield {"frame_index": f, "observation.depths.cam_front": np.full((2, 2), f, np.uint16)}

        def iter_video_frames(self, camera):
            for f in range(3):
                yield SimpleNamespace(frame_index=f, rgb=np.zeros((2, 2, 3), np.uint8))

    class Runtime:
        cad = np.zeros((2, 512, 3), np.float32)

        def infer(self, rgb, depth):
            poses = np.tile(np.eye(4), (2, 1, 1))
            poses[:, 0, 3] = float(depth[0, 0])
            return poses, predicted_geometry_packet(self.cad, poses, [True, True])

    monkeypatch.setattr(sidecar, "front_calibration_provenance", lambda *args: {"profile": {"sha256": "test"}})
    path = sidecar.export_sidecar(
        task=task,
        runtime=Runtime(),
        repo=tmp_path,
        profile="x",
        checkpoint_sha256="abc",
        camera_hashes={},
        raw_factory=Raw,
        shard_size=2,
    )
    reader = ObjectGeometryDerivedDataset(task.root, version=path.name)
    assert len(reader) == 3
    assert reader.get(episode_index=0, frame_index=2).manipulated_xyz_m[0, 0] == 2
    assert reader.manifest["pose_checkpoint_sha256"] == "abc"
    with pytest.raises(FileExistsError):
        sidecar.export_sidecar(
            task=task,
            runtime=Runtime(),
            repo=tmp_path,
            profile="x",
            checkpoint_sha256="abc",
            camera_hashes={},
            raw_factory=Raw,
        )


@pytest.mark.parametrize("task", ["peg_all", "press_the_button", "yellow_socket", "eraser"])
def test_new_pi0_configs_have_prediction_only_sidecar(task):
    from openpi.training import config

    cfg = config.get_config(f"pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain/{task}")
    assert cfg.model.encoder.default_token_count == 48
    assert cfg.model.encoder.object_relation_mode == "final_v2"
    assert cfg.data.object_geometry.version == "pretrained_pose_v1"
    assert cfg.data.object_geometry.require_pose_prediction
    assert cfg.data.object_geometry.hand_keypoint_object_version == "v1"
    assert cfg.freeze_filter == cfg.model.get_freeze_filter()


def test_online_policy_never_uses_external_geometry():
    from openpi.cad_pose.policy import PoseConditionedPolicy

    class DummyPolicy:
        metadata = {"test": True}

        def infer(self, data, **kwargs):
            return data

    class Runtime:
        def infer(self, rgb, depth):
            assert rgb.shape == (2, 2, 3) and depth.shape == (2, 2)
            return np.zeros((2, 4, 4)), {"prediction": np.array([42])}

    bridge = PoseConditionedPolicy(DummyPolicy(), Runtime())
    obs = {
        "observation.images.cam_front": np.zeros((2, 2, 3), np.uint8),
        "observation.depths.cam_front": np.zeros((2, 2), np.uint16),
    }
    result = bridge.infer(obs)
    assert result["object_geometry"]["prediction"][0] == 42
    assert "object_geometry" not in obs
    with pytest.raises(ValueError, match="externally"):
        bridge.infer({**obs, "object_geometry": {}})
    with pytest.raises(ValueError, match="RGB and raw depth"):
        bridge.infer({})


def test_pose_hash_diagnostic_checks_all_frames_and_masks():
    from types import SimpleNamespace

    from scripts.cad_pose.preflight import compare_pose_geometry

    cad = np.zeros((512, 3), dtype=np.float32)
    rows = [{"valid": True, "T_base_link_object": np.eye(4).tolist()} for _ in range(3)]

    class Reader:
        mismatch = False
        bad_mask = False

        def get(self, *, episode_index, frame_index):
            points = cad.copy()
            if self.mismatch and frame_index == 1:
                points[2, 1] = 0.001
            mask = np.ones(512, bool)
            if self.bad_mask and frame_index == 1:
                mask[0] = False
            return SimpleNamespace(manipulated_xyz_m=points, manipulated_mask=mask)

    reader = Reader()
    result = compare_pose_geometry(rows, reader, 0, "manipulated", cad)
    assert result["status"] == "numerically_equal_sha_mismatch"
    assert result["compared_frames"] == 3
    reader.mismatch = True
    result = compare_pose_geometry(rows, reader, 0, "manipulated", cad)
    assert result["status"] == "geometry_mismatch"
    assert result["first_mismatches"][0]["frame"] == 1
    reader.mismatch = False
    reader.bad_mask = True
    assert compare_pose_geometry(rows, reader, 0, "manipulated", cad)["status"] == "geometry_mismatch"


def test_gt_review_renders_geometry_without_checkpoint(tmp_path):
    import cv2

    from scripts.cad_pose.visualize_gt import render_gt

    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    points = np.array([[0, 0, 1], [0.1, 0.1, 1]], dtype=np.float32)
    path = tmp_path / "review.png"
    render_gt(
        rgb,
        np.array([[50, 0, 40], [0, 50, 32], [0, 0, 1]]),
        np.eye(4),
        points,
        {"manipulated": points, "target": points},
        path,
        "candidate GT",
    )
    image = cv2.imread(str(path))
    assert image.shape == (64, 240, 3)
    assert np.any(image)
