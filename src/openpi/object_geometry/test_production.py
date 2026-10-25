from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openpi.object_geometry.production import build_episode_records
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
        {"frame_index": 0, "T_base_link_object": _T(1, 0, 0), "valid": True, "confidence": 0.8, "status": "observed"},
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
