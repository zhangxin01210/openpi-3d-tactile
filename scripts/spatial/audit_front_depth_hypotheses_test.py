"""Fixed-pixel lag scoring and spatial residual bins."""

from __future__ import annotations

import numpy as np

from audit_front_depth_hypotheses import (
    centered_bias_probe, fixed_pixel_lag_scores, stratified_residual,
)


def test_lag_scores_use_identical_pixels_for_every_pose() -> None:
    candidates = {
        -1: (np.array([[0.03, 0.01]], np.float32), np.array([[True, True]])),
        0: (np.array([[0.01, 0.02]], np.float32), np.array([[True, True]])),
        1: (np.array([[0.02, 0.001]], np.float32), np.array([[False, True]])),
    }
    result, common = fixed_pixel_lag_scores(candidates, max_abs_m=0.05)
    assert common.tolist() == [[False, True]]
    assert result["common_pixels"] == 1
    assert result["best_offset"] == 1

    only_first, _ = fixed_pixel_lag_scores(
        candidates, max_abs_m=0.05, pixel_mask=np.array([[True, False]])
    )
    assert only_first["best_offset"] is None

    empty, mask = fixed_pixel_lag_scores({
        0: (np.zeros((1, 1)), np.array([[False]])),
        1: (np.zeros((1, 1)), np.array([[True]])),
    }, max_abs_m=0.05)
    assert not mask.any() and empty["best_offset"] is None


def test_bins_preserve_signed_residuals_and_counts() -> None:
    obs = np.full((3, 3), 0.8, np.float32)
    residual = np.full((3, 3), -0.012, np.float32)
    groups = np.zeros((3, 3), np.int16)
    bins = stratified_residual(obs, residual, np.ones((3, 3), bool), groups,
                               (("arm", ("link",)),), max_abs_m=0.05)
    arm = next(item for item in bins if item["kind"] == "link")
    depth = next(item for item in bins if item["kind"] == "depth_m" and item["count"])
    assert arm["count"] == 9 and depth["count"] == 9
    assert abs(arm["signed_median_mm"] + 12) < 0.001


def test_centered_bias_probe_does_not_change_input_residuals() -> None:
    candidates = {
        -1: (np.array([[-0.030, -0.025]], np.float32), np.ones((1, 2), bool)),
        0: (np.array([[-0.016, -0.016]], np.float32), np.ones((1, 2), bool)),
        1: (np.array([[-0.004, 0.001]], np.float32), np.ones((1, 2), bool)),
    }
    original = candidates[0][0].copy()
    result = centered_bias_probe(candidates, np.ones((1, 2), bool))
    assert abs(result["offset_mm"] - 16) < 0.001
    assert result["best_offset"] == 0
    np.testing.assert_array_equal(candidates[0][0], original)
