"""Synthetic tests for the sparse 3-D keypoint prototype."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from openpi.spatial.config import PinholeIntrinsics
from openpi.spatial.geometry import CameraCalibration

ROOT = Path(__file__).resolve().parents[2]


def _module():
    name = "pilot_sparse_3d_relations_test_module"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/object_geometry/pilot_sparse_3d_relations.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _camera():
    identity = np.eye(4, dtype=np.float64)
    intrinsics = PinholeIntrinsics(fx=10.0, fy=10.0, cx=2.0, cy=2.0)
    return CameraCalibration(
        role="front",
        depth_intrinsics=intrinsics,
        color_intrinsics=intrinsics,
        depth_scale_m_per_unit=0.001,
        T_color_depth=identity,
        T_base_color=identity,
    )


def test_projection_and_local_depth_observation_use_base_link() -> None:
    pilot = _module()
    depth = np.full((5, 5), 1000, dtype=np.uint16)
    projection = pilot.project_depth_to_base(depth, _camera())
    assert projection.xyz_base_m.shape == (25, 3)
    np.testing.assert_allclose(projection.xyz_base_m[12], [0.0, 0.0, 1.0], atol=1e-6)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    observation = pilot.sample_depth_observation(
        projection, mask, np.array([2.0, 2.0]), radius_px=1.0, min_samples=1
    )
    assert observation.position_m is not None
    np.testing.assert_allclose(observation.position_m, [0.0, 0.0, 1.0], atol=1e-6)
    assert observation.confidence > 0.0


def test_missing_depth_and_mask_outlier_are_invalid() -> None:
    pilot = _module()
    depth = np.zeros((5, 5), dtype=np.uint16)
    projection = pilot.project_depth_to_base(depth, _camera())
    observation = pilot.sample_depth_observation(
        projection, np.ones((5, 5), dtype=bool), np.array([2.0, 2.0]), min_samples=1
    )
    assert observation.position_m is None
    assert observation.reason == "missing_depth"


def test_mask_transport_handles_motion_and_explicit_initialization() -> None:
    pilot = _module()
    initialization = pilot.Initialization(
        initial_frame=0,
        keypoints={"shaft_end": np.array([2.0, 2.0]), "axis_reference": np.array([3.0, 2.0])},
    )
    tracker = pilot.ObjectTracker(initialization)
    mask0 = np.zeros((12, 12), dtype=bool)
    mask0[2:8, 2:5] = True
    first, anomalies0 = tracker.update(0, mask0)
    assert first["visible"]
    assert first["keypoints"]["shaft_end"]["source"] == "observed"
    mask1 = np.zeros((12, 12), dtype=bool)
    mask1[2:8, 5:8] = True
    second, anomalies1 = tracker.update(1, mask1)
    assert second["keypoints"]["shaft_end"]["uv"][0] > first["keypoints"]["shaft_end"]["uv"][0]
    assert not any("outside_current_mask" in item for item in anomalies0 + anomalies1)


def test_occlusion_prediction_is_conservative_and_reobservation_is_causal() -> None:
    pilot = _module()
    track = pilot.TemporalPointTrack(max_prediction_frames=2)
    observed = track.update(0, np.array([0.0, 0.0, 1.0]), 1.0)
    predicted = track.update(1, None)
    assert observed["valid"]
    assert observed["source"] == "observed"
    assert not predicted["valid"]
    assert predicted["source"] == "predicted"
    assert predicted["confidence"] < observed["confidence"]
    reobserved = track.update(2, np.array([0.1, 0.0, 1.0]), 1.0)
    assert reobserved["valid"]
    assert reobserved["reobserved"]
    assert reobserved["source"] == "observed"


def test_prediction_expires_without_future_frame_leakage() -> None:
    pilot = _module()
    track = pilot.TemporalPointTrack(max_prediction_frames=1)
    track.update(0, np.array([0.0, 0.0, 1.0]), 1.0)
    frame1 = track.update(1, None)
    assert frame1["source"] == "predicted"
    frame2 = track.update(2, None)
    assert frame2["source"] == "invalid"
    assert frame2["position_m"] is None
