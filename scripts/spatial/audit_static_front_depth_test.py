"""Focused checks for the raw-depth temporal and planar diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from audit_static_front_depth import fit_plane, parse_frames, temporal_metrics


def test_parse_frames_rejects_ambiguous_or_unbounded_selection() -> None:
    assert parse_frames("0:6:2") == [0, 2, 4]
    assert parse_frames("3,1,2") == [3, 1, 2]
    for value in ("0:2", "0,0,1", "-1,0,1", "0:301"):
        with pytest.raises(ValueError):
            parse_frames(value)


def test_temporal_metrics_ignore_holes_and_detect_drift() -> None:
    stack = np.array([
        [[1.000, np.nan], [2.000, 1.0]],
        [[1.001, np.nan], [2.000, 1.0]],
        [[1.002, np.nan], [2.000, 1.0]],
    ], np.float32)
    rows, mad, reliable = temporal_metrics(stack)
    assert [row["common_pixels"] for row in rows] == [3, 3, 3]
    assert rows[0]["temporal_abs_p90_mm"] > 0
    assert reliable.tolist() == [[True, False], [True, True]]
    assert mad[0, 0] == pytest.approx(1.0, abs=1e-4)


def test_plane_fit_flat_depth_with_one_outlier() -> None:
    depth = np.full((20, 20), 0.75, np.float32)
    depth[10, 10] = 0.90
    intrinsics = SimpleNamespace(fx=100, fy=100, cx=10, cy=10)
    result = fit_plane(depth, intrinsics, 0, 0)
    assert result["count"] >= 350
    assert result["abs_median_mm"] < 0.01
    assert result["abs_p90_mm"] < 0.01
