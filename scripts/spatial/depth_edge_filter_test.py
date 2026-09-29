"""Synthetic raw-depth edge filtering checks."""

from __future__ import annotations

import numpy as np
import pytest

from depth_edge_filter import filter_depth_edges


def run(depth, *, radius=0, holes=False):
    return filter_depth_edges(
        depth, scale_m_per_unit=0.001, jump_mm=25, radius_px=radius,
        min_depth_m=0.1, max_depth_m=3.0, include_hole_boundaries=holes,
    )


def test_flat_depth_is_unchanged_and_step_removes_both_sides() -> None:
    flat = np.full((4, 6), 1000, np.uint16)
    filtered, removed, seeds = run(flat)
    np.testing.assert_array_equal(filtered, flat)
    assert not removed.any() and not seeds.any()

    depth = flat.copy()
    depth[:, 3:] = 1100
    filtered, removed, seeds = run(depth)
    assert removed[:, 2:4].all()
    assert not removed[:, :2].any() and not removed[:, 4:].any()
    assert np.array_equal(filtered[~removed], depth[~removed])
    assert filtered.dtype == depth.dtype


def test_radius_and_hole_boundary_are_explicit_options() -> None:
    depth = np.full((7, 7), 1000, np.uint16)
    depth[3, 3] = 0
    _, default_removed, _ = run(depth)
    assert not default_removed.any()
    _, hole_removed, _ = run(depth, holes=True)
    assert hole_removed.sum() == 4
    _, expanded, _ = run(depth, holes=True, radius=1)
    assert expanded.sum() > hole_removed.sum()
    assert not expanded[3, 3]


def test_bad_filter_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        run(np.zeros((2, 2, 2), np.uint16))
    with pytest.raises(ValueError):
        filter_depth_edges(np.ones((2, 2)), scale_m_per_unit=0.001,
                           jump_mm=0, radius_px=1, min_depth_m=0.1, max_depth_m=3.0)
