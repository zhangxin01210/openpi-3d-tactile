"""Conservative raw-depth invalidation for offline spatial diagnostics."""

from __future__ import annotations

import cv2
import numpy as np


def filter_depth_edges(
    depth_raw: np.ndarray,
    *,
    scale_m_per_unit: float,
    jump_mm: float,
    radius_px: int,
    min_depth_m: float,
    max_depth_m: float,
    include_hole_boundaries: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero valid pixels near depth discontinuities without changing retained values.

    A jump marks both sides because a single raw depth frame cannot establish
    which side contains mixed pixels. The returned masks are in the raw depth
    pixel grid: removed valid pixels and the pre-dilation seed edges.
    """
    depth = np.asarray(depth_raw)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("depth_raw must be a numeric [H,W] array")
    if (not np.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0
            or not np.isfinite(jump_mm) or jump_mm <= 0
            or radius_px < 0 or min_depth_m <= 0 or max_depth_m <= min_depth_m):
        raise ValueError("Invalid depth scale, jump threshold, radius, or valid depth range")

    meters = depth.astype(np.float32) * scale_m_per_unit
    valid = np.isfinite(meters) & (meters >= min_depth_m) & (meters <= max_depth_m)
    seeds = np.zeros(depth.shape, bool)
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    jump_m = jump_mm / 1000.0
    dx = horizontal & (np.abs(meters[:, 1:] - meters[:, :-1]) > jump_m)
    dy = vertical & (np.abs(meters[1:, :] - meters[:-1, :]) > jump_m)
    seeds[:, 1:] |= dx
    seeds[:, :-1] |= dx
    seeds[1:, :] |= dy
    seeds[:-1, :] |= dy
    if include_hole_boundaries:
        seeds[:, 1:] |= valid[:, 1:] & ~valid[:, :-1]
        seeds[:, :-1] |= valid[:, :-1] & ~valid[:, 1:]
        seeds[1:, :] |= valid[1:, :] & ~valid[:-1, :]
        seeds[:-1, :] |= valid[:-1, :] & ~valid[1:, :]

    if radius_px:
        width = radius_px * 2 + 1
        seeds_expanded = cv2.dilate(seeds.astype(np.uint8), np.ones((width, width), np.uint8)) != 0
    else:
        seeds_expanded = seeds
    removed = seeds_expanded & valid
    filtered = depth.copy()
    filtered[removed] = 0
    return filtered, removed, seeds
