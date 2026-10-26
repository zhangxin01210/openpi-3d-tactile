"""Small, non-labeling diagnostics for Hand/Object base-frame geometry."""

from __future__ import annotations

from typing import Any

import numpy as np


def point_summary(xyz: np.ndarray, mask: np.ndarray, *, max_abs_m: float = 10.0) -> dict[str, Any]:
    """Return finite-point statistics without treating distance as contact truth."""

    xyz = np.asarray(xyz, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or mask.shape != (xyz.shape[0],):
        raise ValueError(f"Expected xyz [N,3] and mask [N], got {xyz.shape} and {mask.shape}")
    finite = np.isfinite(xyz).all(axis=-1)
    valid = mask & finite
    points = xyz[valid]
    if points.size == 0:
        return {
            "valid_count": 0,
            "min_m": None,
            "max_m": None,
            "centroid_m": None,
            "max_abs_m": None,
            "nonfinite_count": int(np.count_nonzero(~finite)),
            "scale_warning": False,
        }
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    centroid = points.mean(axis=0)
    max_abs = float(np.max(np.abs(points)))
    return {
        "valid_count": int(points.shape[0]),
        "min_m": minimum.tolist(),
        "max_m": maximum.tolist(),
        "centroid_m": centroid.tolist(),
        "max_abs_m": max_abs,
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "scale_warning": bool(max_abs > max_abs_m),
    }


def nearest_distance_m(
    xyz_a: np.ndarray, mask_a: np.ndarray, xyz_b: np.ndarray, mask_b: np.ndarray
) -> float | None:
    """Return the nearest valid-point distance, or ``None`` for an empty group."""

    a = np.asarray(xyz_a, dtype=np.float64)[np.asarray(mask_a, dtype=bool)]
    b = np.asarray(xyz_b, dtype=np.float64)[np.asarray(mask_b, dtype=bool)]
    a = a[np.isfinite(a).all(axis=-1)]
    b = b[np.isfinite(b).all(axis=-1)]
    if a.size == 0 or b.size == 0:
        return None
    distances_sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return float(np.sqrt(np.min(distances_sq)))


def frame_diagnostic(
    *,
    hand_xyz: np.ndarray,
    hand_mask: np.ndarray,
    funnel_xyz: np.ndarray,
    funnel_mask: np.ndarray,
    flask_xyz: np.ndarray,
    flask_mask: np.ndarray,
    max_abs_m: float = 10.0,
) -> dict[str, Any]:
    """Summarize one frame in the shared base_link/meters convention."""

    summaries = {
        "hand": point_summary(hand_xyz, hand_mask, max_abs_m=max_abs_m),
        "funnel": point_summary(funnel_xyz, funnel_mask, max_abs_m=max_abs_m),
        "flask": point_summary(flask_xyz, flask_mask, max_abs_m=max_abs_m),
    }
    return {
        "hand": summaries["hand"],
        "funnel": summaries["funnel"],
        "flask": summaries["flask"],
        "hand_funnel_nearest_distance_m": nearest_distance_m(hand_xyz, hand_mask, funnel_xyz, funnel_mask),
        "funnel_flask_nearest_distance_m": nearest_distance_m(funnel_xyz, funnel_mask, flask_xyz, flask_mask),
        "coordinate_warning": any(summary["scale_warning"] for summary in summaries.values()),
    }
