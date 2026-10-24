"""Lightweight tests for the dependency-free object geometry core."""

from __future__ import annotations

import numpy as np
import pytest

from openpi.object_geometry import (
    HistoryObservation,
    PointCloud,
    build_causal_history,
    depth_to_base_points,
    mask_depth_to_cloud,
    transform_points,
)


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def test_empty_mask_and_depth_contract() -> None:
    depth = np.ones((2, 2), dtype=np.uint16)
    rays = np.zeros((2, 2, 3), dtype=np.float32)
    rays[..., 2] = 1.0
    cloud = mask_depth_to_cloud(
        depth,
        np.zeros((2, 2), dtype=bool),
        rays,
        _identity(),
        np.diag([1.0, 1.0, 1.0]),
        depth_scale_m_per_unit=0.1,
    )
    assert cloud.xyz.shape == (0, 3)
    assert cloud.source_pixel is not None and cloud.source_pixel.shape == (0,)

    points, source, depth_m = depth_to_base_points(
        depth,
        rays,
        depth_scale_m_per_unit=0.1,
    )
    assert points.shape == (4, 3)
    assert source.tolist() == [0, 1, 2, 3]
    assert np.allclose(depth_m, 0.1)


def test_coordinate_transform_and_shape_validation() -> None:
    transform = _identity()
    transform[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(transform_points([[0.0, 0.0, 0.0]], transform), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="shape \(N, 3\)"):
        PointCloud(np.zeros((2, 2), dtype=np.float32))


def test_history_excludes_future_frames() -> None:
    observations = [
        HistoryObservation(0, [[0.0, 0.0, 0.0]], _identity()),
        HistoryObservation(5, [[0.01, 0.0, 0.0]], _identity()),
        HistoryObservation(8, [[10.0, 0.0, 0.0]], _identity()),
    ]
    result = build_causal_history(
        observations,
        current_frame=5,
        window_frames=20,
        voxel_size_m=0.1,
        min_observations=1,
    )
    assert result.eligible_frames == (0, 5)
    assert np.max(result.points[:, 0]) < 1.0


def test_target_history_stays_in_base_frame() -> None:
    observations = [
        HistoryObservation(0, [[0.2, 0.3, 0.4]], _identity()),
        HistoryObservation(1, [[0.2, 0.3, 0.4]], _identity()),
    ]
    result = build_causal_history(
        observations,
        current_frame=1,
        window_frames=5,
        voxel_size_m=0.05,
        min_observations=2,
        require_motion=False,
    )
    assert np.allclose(result.points, [[0.2, 0.3, 0.4]])
    assert result.support_count.tolist() == [2]


def test_point_cloud_serialization_round_trip() -> None:
    cloud = PointCloud(
        xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        source_pixel=np.array([7], dtype=np.int32),
        source=np.array([1], dtype=np.int8),
        confidence=np.array([0.75], dtype=np.float32),
        age=np.array([2], dtype=np.int32),
    )
    restored = PointCloud.from_dict(cloud.to_dict())
    assert np.array_equal(restored.xyz, cloud.xyz)
    assert np.array_equal(restored.source_pixel, cloud.source_pixel)
    assert np.array_equal(restored.source, cloud.source)
    assert np.allclose(restored.confidence, cloud.confidence)
    assert np.array_equal(restored.age, cloud.age)
