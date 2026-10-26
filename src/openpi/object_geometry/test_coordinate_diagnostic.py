from __future__ import annotations

import numpy as np

from openpi.object_geometry.coordinate_diagnostic import frame_diagnostic
from openpi.object_geometry.coordinate_diagnostic import nearest_distance_m
from openpi.object_geometry.coordinate_diagnostic import point_summary


def test_point_summary_ignores_masked_nan_and_reports_scale() -> None:
    xyz = np.asarray([[0.0, 0.0, 0.0], [np.nan, 5.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)
    summary = point_summary(xyz, np.asarray([True, False, True]), max_abs_m=10.0)
    assert summary["valid_count"] == 2
    assert summary["nonfinite_count"] == 1
    assert summary["scale_warning"] is True


def test_nearest_distance_requires_finite_valid_points() -> None:
    distance = nearest_distance_m(
        np.asarray([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
        np.asarray([True, True]),
        np.asarray([[0.0, 0.0, 0.5]]),
        np.asarray([True]),
    )
    assert np.isclose(distance, 0.5)


def test_frame_diagnostic_reports_empty_object_without_fake_distance() -> None:
    empty = np.zeros((2, 3), dtype=np.float32)
    empty_mask = np.zeros((2,), dtype=bool)
    full = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    report = frame_diagnostic(
        hand_xyz=full,
        hand_mask=np.ones((2,), dtype=bool),
        funnel_xyz=empty,
        funnel_mask=empty_mask,
        flask_xyz=full,
        flask_mask=np.ones((2,), dtype=bool),
    )
    assert report["hand_funnel_nearest_distance_m"] is None
    assert report["funnel_flask_nearest_distance_m"] is None
    assert report["funnel"]["valid_count"] == 0
