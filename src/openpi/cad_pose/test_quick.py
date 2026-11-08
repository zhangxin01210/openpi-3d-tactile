"""Quick checks are bounded, distinguish sampled agreement and fail closed."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from openpi.cad_pose import quick


def test_first_episode_index_stops_before_unrelated_episodes(tmp_path):
    path = tmp_path / "manifest.json"
    records = [
        {"episode_index": ep, "frame_index": frame, "shard": f"e{ep}", "row": frame}
        for ep in range(3)
        for frame in range(4)
    ]
    path.write_text(json.dumps({"status": "complete", "records": records}, indent=2))
    header, ep, found = quick.first_episode_index(path)
    assert header["status"] == "complete"
    assert ep == 0
    assert len(found) == 4
    assert all(r["episode_index"] == 0 for r in found)


def test_selected_pose_rows_does_not_decode_unselected_lines(tmp_path):
    path = tmp_path / "poses.jsonl"
    rows = [
        json.dumps({"frame_index": i, "object_name": "button", "valid": True}) if i in (0, 2, 4) else "not json"
        for i in range(5)
    ]
    path.write_text("\n".join(rows) + "\n")
    assert list(quick.selected_pose_rows(path, object_name="button", frames=[0, 2, 4])) == [0, 2, 4]


def test_quick_published_compares_three_frames_and_reports_unverified_hash(tmp_path, monkeypatch):
    dataset = "1028_press_button"
    root = tmp_path / "data" / dataset / "object_geometry/v1"
    shard = root / "shard_0"
    shard.mkdir(parents=True)
    prep = tmp_path / "output/preprocess" / dataset
    run = prep / "rgbtrack/runs/run/episodes/e00000"
    run.mkdir(parents=True)
    ep = prep / "rgbtrack/episodes/e00000"
    ep.parent.mkdir(parents=True)
    ep.symlink_to(run)
    records = [{"episode_index": 0, "frame_index": i, "shard": "shard_0", "row": i} for i in range(5)]
    roles = quick.ROLES[dataset]
    manifest = {
        "status": "complete",
        "object_geometry_contract": {"point_count": 512},
        "dataset": dataset,
        "role_to_object": roles,
        "coordinate_frame": "base_link",
        "length_unit": "m",
        "surface_sampling_seed": quick.SURFACE_SAMPLING_SEED,
        "approved_meshes_sha256": dict.fromkeys(roles.values(), "recorded"),
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (prep / "rgbtrack/release_v1.json").write_text(
        json.dumps({"dataset": dataset, "objects": list(roles.values()), "run_tag": "run"})
    )
    approval = prep / "geometry/approved"
    approval.mkdir(parents=True)
    (approval / "manifest.json").write_text(
        json.dumps(
            {
                "status": "human_accepted",
                "assets": {name: {"sha256": "recorded", "scale_m_per_unit": 1} for name in roles.values()},
            }
        )
    )
    (run / "manifest.json").write_text("{}")
    np.save(shard / "episode_index.npy", np.zeros(5, np.int32))
    np.save(shard / "frame_index.npy", np.arange(5))
    cad_points = np.random.default_rng(42).normal(0, 0.03, size=(512, 3)).astype(np.float32)
    for role, name in roles.items():
        (run / f"{name}_poses.jsonl").write_text(
            "".join(
                json.dumps(
                    {"frame_index": i, "object_name": name, "valid": True, "T_base_link_object": np.eye(4).tolist()}
                )
                + "\n"
                for i in range(9)
            )
        )
        np.save(shard / f"{role}_xyz_m.npy", np.tile(cad_points[None], (5, 1, 1)))
        np.save(shard / f"{role}_mask.npy", np.ones((5, 512), bool))
    monkeypatch.setattr(quick, "front_calibration_provenance", lambda *args: {"profile": {"sha256": "profile"}})
    monkeypatch.setattr(quick, "calibration_provenance_error", lambda *args: None)
    monkeypatch.setattr(quick, "load_surface_points", lambda *args: cad_points)
    report = quick.quick_audit(tmp_path, tmp_path / "data", dataset, "profile", do_smoke=False)
    assert report["status"] == "requires_visual_review"
    assert report["checks"]["frames"] == [0, 2, 4]
    assert report["checks"]["pose_hash_and_alignment"] == "sampled_pass"
    assert report["checks"]["adapter"] == "button_ordered_fit"
    assert "old PoseRecord not used" in report["checks"]["pose_source"]
    assert "not recomputed" in report["checks"]["source_hashes"]

    from types import SimpleNamespace

    from openpi.cad_pose import preprocess
    from openpi.spatial_dataset import source

    class FakeRaw:
        def __init__(self, root, *, episode):
            pass

        def iter_rows(self, *, selected, depth_roles):
            for frame in selected:
                yield {"frame_index": frame, "observation.depths.cam_front": np.zeros((2, 2), np.uint16)}

        def iter_video_frames(self, role, *, selected):
            for frame in selected:
                yield SimpleNamespace(frame_index=frame, rgb=np.zeros((2, 2, 3), np.uint8))

    class FakePreprocess:
        def __init__(self, *args):
            pass

        def __call__(self, rgb, depth):
            return np.zeros((256, 6), np.float32), np.ones(256, bool)

    monkeypatch.setattr(source, "RawSpatialDataset", FakeRaw)
    monkeypatch.setattr(preprocess, "FrontRGBDPreprocessor", FakePreprocess)
    smoke = quick.quick_audit(tmp_path, tmp_path / "data", dataset, "profile")
    assert smoke["status"] == "requires_visual_review"
    assert len(smoke["checks"]["smoke"]) == 3
    assert all(np.isfinite(item["loss"]) for item in smoke["checks"]["smoke"])

    approval_path = approval / "manifest.json"
    approved = json.loads(approval_path.read_text())
    approved["assets"]["box"]["sha256"] = "different"
    approval_path.write_text(json.dumps(approved))
    mismatch = quick.quick_audit(tmp_path, tmp_path / "data", dataset, "profile", do_smoke=False)
    assert mismatch["status"] == "blocked"
    assert len(mismatch["checks"]["sampled_residuals"]) == 6
    assert "source hashes differ" in mismatch["errors"][0]
    approved["assets"]["box"]["sha256"] = "recorded"
    approval_path.write_text(json.dumps(approved))

    points = np.tile(cad_points[None], (5, 1, 1))
    points[2, 5, 0] += 0.1
    np.save(shard / "target_xyz_m.npy", points)
    report = quick.quick_audit(tmp_path, tmp_path / "data", dataset, "profile", do_smoke=False)
    assert report["status"] == "blocked"
    assert "rigid-fit residual" in report["errors"][0]


def test_quick_cli_times_out_per_task_and_continues(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        "scripts/cad_pose/preflight.py",
        "--quick",
        "--timeout-seconds",
        "0.001",
        "--output",
        str(tmp_path / "report.json"),
    ]
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=20, check=False)
    report = json.loads((tmp_path / "report.json").read_text())
    assert result.returncode == 1
    assert "[1/4]" in result.stdout
    assert "[4/4]" in result.stdout
    assert all(x["status"] == "timeout" for x in report["datasets"])


def test_training_rejects_unseen_frame_geometry_mismatch():
    from types import SimpleNamespace

    import pytest

    from openpi.cad_pose.data import validate_sampled_pose_frame

    cad = np.zeros((2, 512, 3), dtype=np.float32)
    row = {"valid": True, "T_base_link_object": np.eye(4).tolist()}
    sample = SimpleNamespace(
        manipulated_mask=np.ones(512, bool),
        target_mask=np.ones(512, bool),
        manipulated_xyz_m=cad[0].copy(),
        target_xyz_m=cad[1].copy(),
    )
    validate_sampled_pose_frame(sample, [row, row], cad, 1, 7)
    sample.target_xyz_m[1, 0] = 0.01
    with pytest.raises(ValueError, match="episode=1 frame=7"):
        validate_sampled_pose_frame(sample, [row, row], cad, 1, 7)


def test_peg_historical_intrinsics_require_explicit_option_and_preserve_source():
    import pytest

    from openpi.cad_pose.legacy import legacy_calibration_check

    base = {
        "schema_version": 1,
        "camera_role": "front",
        "coordinate_frame": "base_link",
        "calibration_mode": "profile",
        "projection_contract": {},
        "applied_front_extrinsic_translation_base_m": [0, 0, 0],
        "applied_front_color_intrinsics": {"fx": 100},
        "profile": {"sha256": "new", "schema_version": 1},
    }
    historical = {
        **base,
        "applied_front_color_intrinsics": {"fx": 90},
        "profile": {"sha256": "old", "schema_version": 1},
    }
    with pytest.raises(ValueError, match="legacy calibration"):
        legacy_calibration_check(historical, base)
    result = legacy_calibration_check(historical, base, allow_legacy_intrinsics=True)
    assert result["source"]["profile"]["sha256"] == "old"
    assert result["mode"] == "legacy_intrinsics_explicit"
    with pytest.raises(ValueError, match="beyond"):
        legacy_calibration_check({**historical, "coordinate_frame": "camera"}, base, allow_legacy_intrinsics=True)


def test_button_training_reader_uses_clean_geometry_not_old_pose(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from openpi.cad_pose import legacy

    rng = np.random.default_rng(1)
    cad = rng.normal(0, 0.02, (2, 512, 3)).astype(np.float32)
    offset = np.array([0.12, -0.03, 0.8], np.float32)
    samples = [
        SimpleNamespace(
            manipulated_xyz_m=cad[0] + offset,
            target_xyz_m=cad[1] + offset,
            manipulated_mask=np.ones(512, bool),
            target_mask=np.ones(512, bool),
        )
        for _ in range(3)
    ]

    class Geometry:
        def get(self, *, episode_index, frame_index):
            return samples[frame_index]

        def frame_ids(self, episode):
            return tuple(range(len(samples)))

    class Raw:
        def __init__(self, root, *, episode):
            pass

        def iter_rows(self, *, depth_roles):
            for i in range(3):
                yield {
                    "frame_index": i,
                    "timestamp": i / 30,
                    "observation.depths.cam_front": np.zeros((2, 2), np.uint16),
                }

        def iter_video_frames(self, role):
            for i in range(3):
                yield SimpleNamespace(frame_index=i, rgb=np.zeros((2, 2, 3), np.uint8))

    monkeypatch.setattr(legacy, "RawSpatialDataset", Raw)
    task = legacy.ButtonGeometryPoseTask.__new__(legacy.ButtonGeometryPoseTask)
    task.dataset, task.root, task.episodes = "1028_press_button", tmp_path, (0,)
    task.ordered_cad, task.reader = cad, Geometry()
    task.preprocess = lambda rgb, depth: (np.zeros((4, 6), np.float32), np.ones(4, bool))
    result = list(task.iter_episode(0))
    assert len(result) == 3
    assert np.allclose(result[1]["target_transform"][:, :3, 3], offset, atol=1e-5)
    samples[1].target_xyz_m[4, 0] += 0.1
    with pytest.raises(ValueError, match="rigid-fit residual"):
        list(task.iter_episode(0))


def test_gt_cad_review_uses_fixed_correspondence_and_reports_errors():
    from types import SimpleNamespace

    from scripts.cad_pose.visualize_gt import fit_review_cad

    cad = np.random.default_rng(12).normal(0, 0.1, (2, 512, 3)).astype(np.float32)
    sample = SimpleNamespace(
        manipulated_xyz_m=cad[0] + 1,
        target_xyz_m=cad[1] + 1,
        manipulated_mask=np.ones(512, bool),
        target_mask=np.ones(512, bool),
    )
    fitted, errors = fit_review_cad(sample, cad)
    assert not errors
    assert np.allclose(fitted["target"], sample.target_xyz_m, atol=1e-5)
    sample.target_xyz_m[0, 0] += 1
    fitted, errors = fit_review_cad(sample, cad)
    assert "target" not in fitted
    assert errors


def test_peg_quick_reads_original_generation_assets_without_hashing(tmp_path, monkeypatch):
    from openpi.object_geometry import production

    dataset = "1018_peg_all"
    root = tmp_path / "data" / dataset / "object_geometry"
    current, old, shard = root / "v1", root / "v1_hand_scene_q2", root / "v1/shard_0"
    shard.mkdir(parents=True)
    old.mkdir(parents=True)
    config_path = tmp_path / "configs/object_geometry/datasets/1018_peg_all/config.json"
    config_path.parent.mkdir(parents=True)
    roles = quick.ROLES[dataset]
    assets = {
        name: {"mesh": f"{name}.glb", "sha256": f"{i + 1}" * 64, "sampling_seed": i + 1, "scale_m_per_unit": 1.0}
        for i, name in enumerate(roles.values())
    }
    config_path.write_text(json.dumps({"assets": assets}))
    asset_source = {
        name: {
            "source_sha256": value["sha256"],
            "sampler": "deterministic_surface_points.v1",
            "seed": value["sampling_seed"],
            "scale_m_per_unit": 1.0,
        }
        for name, value in assets.items()
    }
    expected = {
        "schema_version": 1,
        "camera_role": "front",
        "coordinate_frame": "base_link",
        "calibration_mode": "profile",
        "projection_contract": {},
        "applied_front_extrinsic_translation_base_m": [0, 0, 0],
        "applied_front_color_intrinsics": {"fx": 100},
        "profile": {"sha256": "new", "schema_version": 1},
    }
    historical = {
        **expected,
        "applied_front_color_intrinsics": {"fx": 90},
        "profile": {"sha256": "old", "schema_version": 1},
    }
    records = [{"episode_index": 0, "frame_index": i, "shard": "shard_0", "row": i} for i in range(3)]
    (old / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "producer": "mesh_6d_hand_scene_v1",
                "generation": {"assets": asset_source},
                "records": records,
            },
            indent=2,
        )
    )
    (current / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "object_geometry_contract": {"point_count": 512},
                "calibration": historical,
                "asset_provenance": None,
                "source": {
                    "mode": "geometry_only_drop_hand_scene_extras_keep_raw_q2",
                    "roles": roles,
                    "original_version": "v1_hand_scene_q2",
                    "original_manifest_sha256": "f" * 64,
                },
                "records": records,
            },
            indent=2,
        )
    )
    np.save(shard / "episode_index.npy", np.zeros(3, np.int32))
    np.save(shard / "frame_index.npy", np.arange(3))
    cad = np.random.default_rng(4).normal(0, 0.03, (2, 512, 3)).astype(np.float32)
    for i, role in enumerate(roles):
        np.save(shard / f"{role}_xyz_m.npy", np.tile(cad[i], (3, 1, 1)))
        np.save(shard / f"{role}_mask.npy", np.ones((3, 512), bool))

    def fake_mesh(path, *, scale_m_per_unit, seed, include_source_sha256=True):
        assert not include_source_sha256
        return cad[seed - 1], {}

    monkeypatch.setattr(quick, "front_calibration_provenance", lambda *args: expected)
    monkeypatch.setattr(production, "load_mesh_surface_points", fake_mesh)
    report = quick.quick_audit(
        tmp_path, tmp_path / "data", dataset, "profile", do_smoke=False, allow_legacy_intrinsics=True
    )
    assert report["status"] == "requires_visual_review"
    assert report["checks"]["legacy_source"]["asset_metadata"] == "historical generation.assets"
    assert report["checks"]["legacy_calibration"]["source"]["profile"]["sha256"] == "old"
    report = quick.quick_audit(tmp_path, tmp_path / "data", dataset, "profile", do_smoke=False)
    assert report["status"] == "blocked"
    assert "legacy calibration" in report["errors"][0]


def test_peg_training_refuses_mismatched_migration_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from openpi.cad_pose import legacy

    monkeypatch.setattr(legacy, "ObjectGeometryDerivedDataset", lambda root, version: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(legacy, "sha256", lambda path: "changed")
    entry = {
        "mode": "quick",
        "checks": {
            "adapter": "legacy_ordered_fit",
            "legacy_source": {"path": str(tmp_path / "old_manifest.json"), "recorded_sha256": "expected"},
        },
    }
    with pytest.raises(ValueError, match="manifest SHA differs"):
        legacy.LegacyGeometryPoseTask(tmp_path, tmp_path, "profile", entry)


def test_checkpoint_preserves_button_and_peg_adapter_choices():
    from types import SimpleNamespace

    from openpi.cad_pose.pipeline import checkpoint_open_options
    from openpi.cad_pose.pipeline import task_adapters

    tasks = {
        "1018_peg_all": SimpleNamespace(
            audit_mode="quick", audit_adapter="legacy_ordered_fit", peg_legacy_intrinsics=True
        ),
        "1028_press_button": SimpleNamespace(
            audit_mode="quick", audit_adapter="button_ordered_fit", peg_legacy_intrinsics=False
        ),
    }
    adapters = task_adapters(tasks)
    options = checkpoint_open_options({"data_adapters": adapters})
    assert options == {"quick": True, "peg_legacy_intrinsics": True}
    assert adapters["1028_press_button"]["adapter"] == "button_ordered_fit"
