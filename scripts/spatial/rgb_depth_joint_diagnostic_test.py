"""Synthetic checks for raw depth-to-color registration and joint panels."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rgb_depth_joint_diagnostic import (
    align_depth_to_color, depth_discontinuities, make_color_cad_comparison, make_joint_sheet,
)


def camera(*, color_fx=2.0, color_tx=0.0):
    T_color_depth = np.eye(4)
    T_color_depth[0, 3] = color_tx
    return SimpleNamespace(
        depth_intrinsics=SimpleNamespace(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        color_intrinsics=SimpleNamespace(fx=color_fx, fy=2.0, cx=0.0, cy=0.0),
        T_color_depth=T_color_depth,
    )


def test_alignment_uses_depth_to_color_transform_and_zbuffer() -> None:
    depth = np.array([[1.0, 1.0, 1.0]], np.float32)
    nearest = align_depth_to_color(depth, camera(), (2, 8), splat="nearest")
    np.testing.assert_allclose(nearest[0, [0, 2, 4]], 1.0)
    assert not np.isfinite(nearest[0, 1])
    four = align_depth_to_color(depth, camera(), (2, 8), splat="four")
    assert np.isfinite(four).sum() > np.isfinite(nearest).sum()
    shifted = align_depth_to_color(depth, camera(color_tx=1.0), (2, 8))
    assert np.isfinite(shifted[0, 2]) and not np.isfinite(shifted[0, 0])

    collision = np.array([[1.0, 2.0]], np.float32)
    output = align_depth_to_color(collision, camera(color_fx=0.4), (1, 2))
    assert output[0, 0] == 1.0


def test_edges_exclude_holes_and_joint_panel_has_both_grids() -> None:
    depth = np.array([[1.0, 1.04, np.inf], [1.0, 1.0, 1.0]], np.float32)
    edges = depth_discontinuities(depth, 0.025)
    assert edges[0, 1]
    assert not edges[0, 2]
    rgb = np.zeros((8, 8, 3), np.uint8)
    raw = np.ones((8, 8), np.float32)
    aligned = np.ones((8, 8), np.float32)
    groups = np.zeros((8, 8), np.int16)
    sheet = make_joint_sheet(rgb, raw, aligned, aligned, groups, groups, (("base", ("base",)),))
    assert sheet.shape == (2 * (8 + 30), 3 * 8, 3)


def test_color_comparison_requires_one_pixel_grid() -> None:
    rgb = np.zeros((32, 32, 3), np.uint8)
    baseline = np.full((32, 32), -1, np.int16)
    corrected = baseline.copy()
    baseline[4:10, 4:10] = 0
    corrected[20:26, 20:26] = 0
    result = make_color_cad_comparison(rgb, baseline, corrected)
    assert result.shape == (32 + 30, 32, 3)
    assert result[36, 4, 2] > result[36, 4, 1]
    assert result[52, 20, 1] > result[52, 20, 2]
