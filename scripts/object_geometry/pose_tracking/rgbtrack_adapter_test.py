"""Synthetic tests for the headless RGBTrack adapter.

These tests intentionally do not import the RGBTrack submodule or a dataset
reader.  Computer 1 has neither the CUDA model environment nor the e23 data;
the tests exercise the adapter's deterministic geometry and source-boundary
checks instead.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _module():
    name = "rgbtrack_adapter_test_module"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "scripts/object_geometry/pose_tracking/rgbtrack_adapter.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_projection_and_base_link_pose_composition() -> None:
    adapter = _module()
    T_base_color = np.eye(4)
    T_base_color[:3, 3] = [1.0, 2.0, 3.0]
    T_color_object = np.eye(4)
    T_color_object[:3, 3] = [0.1, 0.2, 1.0]
    np.testing.assert_allclose(
        adapter.pose_color_to_base(T_base_color, T_color_object)[:3, 3],
        [1.1, 2.2, 4.0],
    )
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1]])
    uv = adapter.project_points(np.array([[0.1, 0.2, 1.0], [0, 0, -1.0]]), K)
    np.testing.assert_allclose(uv[0], [60.0, 60.0])
    assert np.isnan(uv[1]).all()


def test_mask_resolution_alias_and_shape_validation(tmp_path: Path) -> None:
    adapter = _module()
    path = tmp_path / "target_flask" / "00007.png"
    path.parent.mkdir()
    image = np.zeros((4, 5), dtype=np.uint8)
    image[1:3, 2:4] = 255
    assert cv2.imwrite(str(path), image)
    mask = adapter.load_mask(tmp_path, "flask", 7, expected_shape=(4, 5))
    assert mask.dtype == bool
    assert int(mask.sum()) == 4
    with pytest.raises(ValueError, match="shape mismatch"):
        adapter.load_mask(tmp_path, "target_flask", 7, expected_shape=(5, 5))


def test_occlusion_is_invalid_and_causal(tmp_path: Path) -> None:
    adapter = _module()
    tracker = adapter.RGBTrackObject(
        repo_root=ROOT,
        object_name="funnel",
        mesh_path=tmp_path / "not_loaded.obj",
        mode="no_depth",
        debug_dir=tmp_path / "debug",
        min_mask_area_px=4,
    )
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    first = tracker.estimate(
        frame_index=0,
        rgb=rgb,
        depth_m=None,
        mask=mask,
        K=np.eye(3),
        T_base_color=np.eye(4),
    )
    assert not first.valid
    assert first.status == "invalid_no_observation"
    # A future frame cannot manufacture a pose after an invalid first frame.
    second = tracker.estimate(
        frame_index=1,
        rgb=rgb,
        depth_m=None,
        mask=mask,
        K=np.eye(3),
        T_base_color=np.eye(4),
    )
    assert not second.valid
    assert second.status == "invalid_no_observation"


def test_causal_state_rejects_future_and_predicts_short_horizon() -> None:
    adapter = _module()
    state = adapter._CausalState()  # noqa: SLF001 - source-boundary unit test.
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.0]
    state.observe(pose, 0)
    pose2 = pose.copy()
    pose2[0, 3] = 0.1
    state.observe(pose2, 1)
    predicted = state.predict(2)
    assert predicted is not None
    np.testing.assert_allclose(predicted[:3, 3], [0.2, 0.0, 1.0])
    assert state.predict(1) is None


def test_explicit_scale_manifest_has_no_inference(tmp_path: Path) -> None:
    adapter = _module()
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"schema_version": 1, "objects": {"funnel": 0.002}}), encoding="utf-8")
    assert adapter._load_scale_manifest(path, ("funnel",)) == {"funnel": 0.002}  # noqa: SLF001
    with pytest.raises(ValueError, match="no explicit scale"):
        adapter._load_scale_manifest(path, ("target_flask",))  # noqa: SLF001


def test_bbox_ply_is_written_in_base_link(tmp_path: Path) -> None:
    adapter = _module()
    vertices = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float64)
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 1.0]
    path = tmp_path / "bbox.ply"
    adapter.write_bbox_ply(path, vertices, pose)
    text = path.read_text(encoding="utf-8")
    assert "element vertex 8" in text
    assert "-0.900000000 -0.800000000 0.000000000" in text


def test_no_depth_and_depth_modes_call_distinct_upstream_paths(tmp_path: Path) -> None:
    adapter = _module()

    class FakeEstimator:
        track_good = True

        def __init__(self) -> None:
            self.calls: list[str] = []

        def register_without_depth(self, **kwargs):
            assert kwargs["iteration"] == 5
            assert "depth" not in kwargs
            self.calls.append("register_without_depth")
            pose = np.eye(4)
            pose[2, 3] = 1.0
            return pose

        def track_one_new_without_depth(self, **kwargs):
            assert "depth" not in kwargs
            self.calls.append("track_one_new_without_depth")
            pose = np.eye(4)
            pose[2, 3] = 1.0
            return pose

    tracker = adapter.RGBTrackObject(
        repo_root=ROOT,
        object_name="funnel",
        mesh_path=tmp_path / "mesh.ply",
        mode="no_depth",
        debug_dir=tmp_path / "debug",
        min_mask_area_px=1,
    )
    fake = FakeEstimator()
    tracker._estimator = fake  # noqa: SLF001 - isolate mode dispatch.
    mask = np.ones((3, 3), dtype=bool)
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    tracker.estimate(frame_index=0, rgb=rgb, depth_m=None, mask=mask, K=np.eye(3), T_base_color=np.eye(4))
    tracker.estimate(frame_index=1, rgb=rgb, depth_m=None, mask=mask, K=np.eye(3), T_base_color=np.eye(4))
    assert fake.calls == ["register_without_depth", "track_one_new_without_depth"]
