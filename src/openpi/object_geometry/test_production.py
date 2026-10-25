from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openpi.object_geometry.production import build_episode_records
from openpi.object_geometry.production import CausalHandSceneState
from openpi.object_geometry.production import SpatialTactileProvider
from openpi.object_geometry.production import deterministic_surface_points
from openpi.object_geometry.production import read_pose_run
from openpi.object_geometry.production import transform_points_local_to_base
from openpi.object_geometry.production import write_episode_sidecar
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset


def _T(x: float, y: float, z: float) -> list[list[float]]:
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def test_surface_sampler_and_base_transform_are_deterministic() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    first = deterministic_surface_points(vertices, faces, count=512, seed=4)
    second = deterministic_surface_points(vertices, faces, count=512, seed=4)
    assert first.shape == (512, 3)
    np.testing.assert_array_equal(first, second)
    transformed = transform_points_local_to_base(first[:1], _T(0.1, -0.2, 0.3))
    np.testing.assert_allclose(transformed, first[:1] + [0.1, -0.2, 0.3])


def test_mesh_pose_sidecar_rejects_invalid_dynamic_pose_and_reads_mmap(tmp_path: Path) -> None:
    pose_root = tmp_path / "pose"
    pose_root.mkdir()
    manifest = {"episode": 23, "calibration": {"profile": {"sha256": "test"}}}
    (pose_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    funnel_rows = [
        {"frame_index": 0, "T_base_link_object": _T(1, 0, 0), "valid": True, "confidence": 0.0, "status": "observed"},
        {"frame_index": 1, "T_base_link_object": _T(1, 0, 0), "valid": False, "confidence": 0.0, "status": "invalid_no_observation"},
    ]
    flask_rows = [
        {"frame_index": 0, "T_base_link_object": _T(0, 0, 1), "valid": True, "confidence": 0.2, "status": "flask_upright_anchor_observed", "pose_source": "reviewed_initial_geometry"},
        {"frame_index": 1, "T_base_link_object": _T(0, 0, 1), "valid": True, "confidence": 0.08, "status": "flask_static_anchor_prior", "pose_source": "static_geometry_prior"},
    ]
    for name, rows in (("funnel", funnel_rows), ("target_flask", flask_rows)):
        (pose_root / f"{name}_poses.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8"
        )
    run = read_pose_run(pose_root, episode=23)
    triangle = deterministic_surface_points(
        np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        count=512,
    )
    records = build_episode_records(
        episode=23,
        pose_run=run,
        funnel_points=triangle,
        flask_points=triangle,
        flask_motion="static",
    )
    assert records[0]["manipulated_mask"].all()
    np.testing.assert_allclose(records[0]["manipulated_confidence"], 1.0)
    np.testing.assert_allclose(records[0]["manipulated_geometry_quality"], 1.0)
    assert not records[1]["manipulated_mask"].any()
    assert records[1]["target_source"].min() == 1
    assert not np.isnan(records[0]["grasp_reference_force_norm"]).any()
    result = write_episode_sidecar(
        dataset_root=tmp_path / "dataset",
        episode=23,
        records=records,
        manifest_context={"calibration": {"profile": {"sha256": "test"}}, "causal": {"future_frame_used": False}},
        requested_episodes=[23],
    )
    assert result["status"] == "written"
    manifest = json.loads((tmp_path / "dataset" / "object_geometry" / "v1_hand_scene" / "manifest.json").read_text())
    assert manifest["calibration"]["profile"]["sha256"] == "test"
    dataset = ObjectGeometryDerivedDataset(tmp_path / "dataset", version="v1_hand_scene")
    sample = dataset.get(episode_index=23, frame_index=1)
    assert sample.manipulated_mask.sum() == 0
    assert sample.target_mask.all()
    assert sample.hand_scene is not None
    assert not np.isnan(sample.target_xyz_m).any()
    with pytest.raises(ValueError, match="different provenance"):
        write_episode_sidecar(
            dataset_root=tmp_path / "dataset",
            episode=23,
            records=records,
            manifest_context={"calibration": {"profile": {"sha256": "changed"}}, "causal": {"future_frame_used": False}},
            requested_episodes=[23],
            version="v1_hand_scene",
        )


def test_spatial_tactile_state_is_causal_and_grasp_quality_stays_unknown() -> None:
    class FakeSpatial:
        def __init__(self) -> None:
            self.values = {
                (23, 0): np.ones((600,), dtype=np.float32),
                (23, 1): np.full((600,), 2.0, dtype=np.float32),
            }

        def get_force_norm(self, episode: int, frame: int) -> np.ndarray | None:
            return self.values.get((episode, frame))

    state = CausalHandSceneState(spatial_tactile=FakeSpatial(), contact_force_threshold=1.5)
    first = state.next(episode=23, frame=0)
    second = state.next(episode=23, frame=1)
    missing = state.next(episode=23, frame=3)
    assert first["grasp_contact_valid"].sum() == 0
    assert second["grasp_contact_valid"].sum() == 600
    np.testing.assert_allclose(second["grasp_reference_force_norm"], 1.0)
    np.testing.assert_allclose(second["grasp_short_delta_force_norm"], 1.0)
    assert float(second["grasp_quality"]) == 0.0
    assert float(missing["grasp_quality"]) == 0.0
    assert state.tactile_frames_valid == 2
    assert state.tactile_discontinuous_frames == 1


def test_spatial_sidecar_frame_join_uses_episode_and_frame_index(tmp_path: Path) -> None:
    root = tmp_path / "dataset" / "spatial" / "v1_front_candidate"
    episode_root = root / "episode_023"
    shard = episode_root / "shard_000"
    shard.mkdir(parents=True)
    np.save(root / "finger_id.npy", np.arange(600, dtype=np.int8))
    np.save(root / "taxel_id.npy", np.arange(600, dtype=np.int16))
    count = 2
    np.save(shard / "frame_index.npy", np.asarray([0, 2], dtype=np.int64))
    np.save(shard / "timestamp_s.npy", np.asarray([0.0, 0.02], dtype=np.float64))
    np.save(shard / "visual_xyz_m.npy", np.zeros((count, 1, 3), dtype=np.float32))
    np.save(shard / "visual_rgb.npy", np.zeros((count, 1, 3), dtype=np.uint8))
    np.save(shard / "visual_rgb_valid.npy", np.ones((count, 1), dtype=bool))
    np.save(shard / "tactile_xyz_m.npy", np.zeros((count, 600, 3), dtype=np.float32))
    np.save(shard / "tactile_force_base.npy", np.ones((count, 600, 3), dtype=np.float32))
    np.save(shard / "tactile_force_norm.npy", np.vstack([np.ones(600), np.full(600, 2.0)]).astype(np.float32))
    (episode_root / "manifest.json").write_text(json.dumps({
        "episode_index": 23,
        "selected_frame_count": count,
        "shards": [{"name": "shard_000", "count": count, "first_frame": 0, "last_frame": 2}],
    }), encoding="utf8")
    (root / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "derived_schema_version": 1,
        "spatial_version": "v1_front_candidate",
        "source_dataset": "synthetic",
        "spatial_contract": {
            "coordinate_frame": "base_link",
            "xyz_unit": "meters",
            "force_unit": "newton",
            "camera_roles": ["front"],
            "visual_points_per_frame": 1,
            "tactile_points_per_frame": 600,
        },
        "storage": {
            "format": "npy_shards_v1",
            "static": {
                "finger_id": {"file": "finger_id.npy"},
                "taxel_id": {"file": "taxel_id.npy"},
            },
            "episodes": [{"episode_index": 23, "path": "episode_023", "selected_frame_count": count}],
        },
        "runtime": {"total_selected_frames": count},
    }), encoding="utf8")
    provider = SpatialTactileProvider(tmp_path / "dataset", version="v1_front_candidate")
    np.testing.assert_allclose(provider.get_force_norm(23, 0), 1.0)
    np.testing.assert_allclose(provider.get_force_norm(23, 2), 2.0)
    assert provider.get_force_norm(23, 1) is None
