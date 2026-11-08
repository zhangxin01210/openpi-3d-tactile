"""Synthetic checks; these do not validate real tracker pose labels."""
from pathlib import Path

import numpy as np
import pytest
import torch

from openpi.cad_pose.geometry import fit_corresponding_rigid_pose
from openpi.cad_pose.geometry import predicted_geometry_packet
from openpi.cad_pose.model import XHandCADPoseTransformerV1
from openpi.cad_pose.model import pose_loss
from openpi.cad_pose.model import rotation_6d_to_matrix
from scripts.cad_pose.preflight import audit_task


def test_rotation_is_finite_proper_and_has_gradients():
    x = torch.tensor([[0., 0, 0, 0, 0, 0], [1., 0, 0, 1, 0, 0],
                      [0.4, 0.3, 0.2, 0.1, 0.5, -0.7]], requires_grad=True)
    r = rotation_6d_to_matrix(x)
    torch.testing.assert_close(r.transpose(-1, -2) @ r, torch.eye(3).expand(3, -1, -1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(3), atol=1e-5, rtol=1e-5)
    r.sum().backward()
    assert bool(torch.isfinite(x.grad).all())


def test_forward_and_finite_training_step_and_cad_dependency():
    torch.manual_seed(42)
    net = XHandCADPoseTransformerV1()
    scene = torch.rand(1, 256, 6)
    mask = torch.ones(1, 256, dtype=torch.bool)
    cad = torch.rand(1, 2, 512, 3) * 0.1
    prediction = net(scene, mask, cad)
    assert prediction["translation"].shape == (1, 2, 3)
    assert prediction["rotation"].shape == (1, 2, 3, 3)
    changed = net(scene, mask, cad + 0.2)
    assert not torch.allclose(prediction["translation"], changed["translation"])
    target = torch.eye(4).expand(1, 2, 4, 4).clone()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    loss = pose_loss(prediction, target, cad, torch.tensor([[True, False]]))
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert bool(torch.any(net.cad_encoder[0].weight.grad != 0))
    optimizer.step()
    with pytest.raises(ValueError, match="no valid"):
        pose_loss(net(scene, mask, cad), target, cad, torch.zeros(1, 2, dtype=torch.bool))


def test_explicit_symmetry_does_not_mark_all_rotations_equivalent():
    torch.manual_seed(1)
    cad = torch.rand(1, 2, 512, 3)
    gt = torch.eye(4).expand(1, 2, 4, 4).clone()
    symmetry = torch.diag(torch.tensor([-1., -1., 1.]))
    rotation = symmetry.expand(1, 2, 3, 3).clone()
    prediction = {"translation": torch.zeros(1, 2, 3), "rotation": rotation}
    mask = torch.tensor([[True, False]])
    plain = pose_loss(prediction, gt, cad, mask)
    sym = symmetry.expand(1, 2, 1, 3, 3)
    symmetric = pose_loss(prediction, gt, cad, mask, symmetry_rotations=sym,
                          symmetry_mask=torch.ones(1, 2, 1, dtype=torch.bool))
    assert float(plain) > 0.1
    assert float(symmetric) < 1e-7


def test_correspondence_fit_rejects_mismatch_and_geometry_packet():
    rng = np.random.default_rng(4)
    local = rng.normal(size=(512, 3)).astype(np.float32) * .05
    angle = 0.5
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, [0.2, -0.3, 0.5]
    world = local @ rotation.T + transform[:3, 3]
    recovered, residual = fit_corresponding_rigid_pose(local, world, np.ones(512, bool), max_rmse_m=1e-5)
    np.testing.assert_allclose(recovered, transform, atol=1e-6)
    assert residual < 1e-6
    with pytest.raises(ValueError, match="residual"):
        fit_corresponding_rigid_pose(local, world[::-1], np.ones(512, bool), max_rmse_m=1e-5)
    packet = predicted_geometry_packet(np.stack((local, local)), np.stack((transform, transform)), [True, False])
    np.testing.assert_allclose(packet["manipulated_xyz_m"], world, atol=1e-6)
    assert packet["manipulated_mask"].all()
    assert not packet["target_mask"].any()
    assert not packet["target_xyz_m"].any()


def test_preflight_does_not_accept_missing_data_or_manifest_alone(tmp_path: Path):
    button = audit_task(tmp_path, tmp_path / "data", "1028_press_button", "missing-profile")
    peg = audit_task(tmp_path, tmp_path / "data", "1018_peg_all", "missing-profile")
    assert button["status"] == peg["status"] == "blocked"
    assert any("raw dataset metadata" in x for x in button["errors"])
    assert any("legacy Peg" in x for x in peg["errors"])


def test_episode_splits_and_balancing():
    from collections import Counter

    from openpi.cad_pose.split import balanced_tasks
    from openpi.cad_pose.split import split_episodes
    ids = list(range(100))
    split = split_episodes(ids)
    assert split == split_episodes(ids[::-1])
    assert [len(split[k]) for k in ("train", "val", "test")] == [80, 10, 10]
    assert len(set(split["train"]) | set(split["val"]) | set(split["test"])) == 100
    tasks = ("peg", "button", "socket", "eraser")
    assert Counter(balanced_tasks(tasks, 20)) == dict.fromkeys(tasks, 20)


def test_pose_metrics_and_visualization(tmp_path):
    from openpi.cad_pose.metrics import pose_errors
    from openpi.cad_pose.metrics import summarize
    from openpi.cad_pose.visualization import render_comparison
    points = np.random.default_rng(1).uniform(-0.1, 0.1, (512, 3))
    transform = np.eye(4)
    transform[2, 3] = 1
    changed = transform.copy()
    changed[0, 3] = .01
    result = pose_errors(changed, transform, points)
    assert result["translation_mm"] == pytest.approx(10.0)
    assert result["cad_error_mm"] == pytest.approx(10.0)
    assert summarize([1., 2., 3.])["median"] == 2
    path = tmp_path / "projection.png"
    intrinsics = np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]])
    metrics = render_comparison(rgb=np.zeros((64, 64, 3), np.uint8), intrinsic=intrinsics,
        t_base_color=np.eye(4), cad_points=points, predicted_pose=changed, target_pose=transform,
        label="synthetic test", output=path)
    assert path.is_file()
    assert metrics["translation_mm"] == pytest.approx(10.0)
    with pytest.raises(FileExistsError):
        render_comparison(rgb=np.zeros((64, 64, 3), np.uint8), intrinsic=intrinsics,
            t_base_color=np.eye(4), cad_points=points, predicted_pose=changed, target_pose=None,
            label="synthetic test", output=path)


def test_published_stream_uses_rgb_depth_and_masks_invalid_pose(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from openpi.cad_pose import data
    from scripts.preprocess.export_object_geometry_rgbtrack import sha256

    name = "1028_press_button"
    root = tmp_path / "data" / name
    prep = tmp_path / "output/preprocess" / name
    approved = prep / "geometry/approved"
    approved.mkdir(parents=True)
    (root / "object_geometry/v1").mkdir(parents=True)
    release_path = prep / "rgbtrack/release_v1.json"
    release_path.parent.mkdir(parents=True)
    release_path.write_text(json.dumps({"dataset": name, "objects": ["box", "button"],
        "episodes": [0], "run_tag": "v1"}))
    assets = {}
    for obj in ("box", "button"):
        mesh = approved / f"{obj}.glb"
        mesh.write_bytes(obj.encode())
        assets[obj] = {"sha256": sha256(mesh), "scale_m_per_unit": 1}
    approval_path = approved / "manifest.json"
    approval_path.write_text(json.dumps({"status": "human_accepted", "assets": assets}))
    run = prep / "rgbtrack/runs/v1/episodes/e00000"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"calibration": {"test": "calibration"}}))
    hashes = {}
    for obj in ("box", "button"):
        path = run / f"{obj}_poses.jsonl"
        rows = [{"frame_index": f, "object_name": obj, "valid": (obj == "button" or f == 1),
                 "T_base_link_object": np.eye(4).tolist()} for f in range(2)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes[obj] = sha256(path)
    episode_link = prep / "rgbtrack/episodes/e00000"
    episode_link.parent.mkdir(parents=True)
    episode_link.symlink_to(run, target_is_directory=True)
    manifest_path = root / "object_geometry/v1/manifest.json"
    manifest_path.write_text(json.dumps({"dataset": name, "source_release_sha256": sha256(release_path),
        "approved_geometry_sha256": sha256(approval_path), "role_to_object": {"manipulated": "button", "target": "box"},
        "surface_sampling_seed": 20261101, "approved_meshes_sha256": {k: v["sha256"] for k, v in assets.items()},
        "pose_files_sha256_by_episode": {"0": hashes}}))
    evidence = {"dataset": name, "status": "requires_visual_review", "checks": {
        "pose_hash_and_alignment": "passed", "geometry_manifest_sha256": sha256(manifest_path),
        "release_sha256": sha256(release_path), "profile_sha256": "profile-hash"}}
    monkeypatch.setattr(data, "load_surface_points", lambda path, seed: np.full((512, 3), seed - 20261100, np.float32))
    monkeypatch.setattr(data, "front_calibration_provenance", lambda *_: {"profile": {"sha256": "profile-hash"}})
    monkeypatch.setattr(data, "calibration_provenance_error", lambda *_: None)
    monkeypatch.setattr(data, "FrontRGBDPreprocessor", lambda *_: lambda rgb, depth: (
        np.full((256, 6), rgb[0, 0, 0] + depth[0, 0], np.float32), np.ones(256, bool)))
    class Raw:
        def __init__(self, root, *, episode): pass
        @staticmethod
        def discover_episode_ids(root): return [0]
        def iter_rows(self, *, depth_roles):
            for i in range(2):
                yield {"frame_index": i, "timestamp": float(i),
                       "observation.depths.cam_front": np.full((2, 2), i, np.uint16)}
        def iter_video_frames(self, camera):
            for i in range(2):
                yield SimpleNamespace(frame_index=i, rgb=np.full((2, 2, 3), 10 + i, np.uint8))
    monkeypatch.setattr(data, "RawSpatialDataset", Raw)
    task = data.PublishedPoseTask(tmp_path, tmp_path / "data", name, profile="profile", audit_entry=evidence)
    samples = list(task.iter_episode(0))
    assert len(samples) == 2
    np.testing.assert_array_equal(samples[0]["valid"], [True, False])
    np.testing.assert_array_equal(samples[1]["valid"], [True, True])
    assert samples[0]["scene"][0, 0] == 10
    assert samples[1]["scene"][0, 0] == 12
    assert samples[0]["cad"][0, 0, 0] == 2  # release order, not role order
    assert samples[0]["cad"][1, 0, 0] == 1
    (run / "button_poses.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="changed after audit"):
        list(task.iter_episode(0))


def test_runtime_uses_cached_model_cad_and_rejects_nonfinite():
    from openpi.cad_pose.runtime import PoseRuntime
    torch.manual_seed(7)
    model = XHandCADPoseTransformerV1()
    cad = np.random.default_rng(3).random((2, 512, 3), dtype=np.float32)
    calls = []
    def preprocess(rgb, depth):
        calls.append((rgb.shape, depth.shape))
        return np.zeros((256, 6), np.float32), np.ones(256, bool)
    runtime = PoseRuntime(model, preprocess, cad)
    poses, packet = runtime.infer(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.uint16))
    assert poses.shape == (2, 4, 4)
    assert packet["manipulated_xyz_m"].shape == (512, 3)
    assert packet["manipulated_mask"].all()
    assert len(calls) == 1
    with pytest.raises(ValueError, match="CAD"):
        PoseRuntime(model, preprocess, np.full((2, 512, 3), np.nan))


def test_checkpoint_runtime_rejects_camera_and_cad_changes(tmp_path, monkeypatch):
    import hashlib

    from openpi.cad_pose import runtime
    folder = tmp_path / "configs/ur7e_xhand"
    folder.mkdir(parents=True)
    paths = {"profile": folder / "profile.json", "camera": folder / "cameras_0903.json",
             "extrinsic": folder / "extrinsic_0909.yaml"}
    for name, path in paths.items():
        path.write_text(name)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    cad = np.zeros((2, 512, 3), np.float32)
    ckpt = tmp_path / "pose.pt"
    torch.save({"model_version": "xhand_cad_pose_transformer_v1",
                "ordered_cad_sha256_by_dataset": {"test": runtime.points_sha256(cad)}, "camera_hashes": hashes,
                "model_state_dict": XHandCADPoseTransformerV1().state_dict()}, ckpt)
    monkeypatch.setattr(runtime, "FrontRGBDPreprocessor", lambda *args: object())
    created = runtime.PoseRuntime.from_checkpoint(ckpt, tmp_path, str(paths["profile"]), cad, dataset="test")
    assert created.device.type == "cpu"
    changed = cad.copy()
    changed[0, 0, 0] = 1
    with pytest.raises(ValueError, match="provenance"):
        runtime.PoseRuntime.from_checkpoint(ckpt, tmp_path, str(paths["profile"]), changed, dataset="test")
    paths["camera"].write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        runtime.PoseRuntime.from_checkpoint(ckpt, tmp_path, str(paths["profile"]), cad, dataset="test")
