"""Checks for pixel-aligned depth-CAD comparison panels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("render_depth_cad_comparison.py")
SPEC = importlib.util.spec_from_file_location("render_depth_cad_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_comparison_uses_same_raw_depth_and_distinct_footprints() -> None:
    raw = np.full((30, 40, 3), 100, np.uint8)
    baseline = np.zeros((30, 40), np.uint8)
    corrected = np.zeros_like(baseline)
    baseline[5:20, 5:20] = 1
    corrected[5:20, 10:25] = 1
    residual = np.zeros_like(raw)
    stats = {"near_surface": {"signed_median_mm": -10.0, "count": 150}}
    image = renderer.make_comparison(
        (raw, baseline, residual), (raw.copy(), corrected, residual), stats, stats
    )
    assert image.shape == (132, 120, 3)
    footprint = image[36 + 30 + 36:, 80:120]
    np.testing.assert_array_equal(footprint[10, 7], [0, 0, 255])
    np.testing.assert_array_equal(footprint[10, 12], [0, 210, 255])
    np.testing.assert_array_equal(footprint[10, 22], [0, 255, 0])
    with pytest.raises(ValueError, match="raw-depth panels differ"):
        renderer.make_comparison((raw, baseline, residual), (raw + 1, corrected, residual), stats, stats)
