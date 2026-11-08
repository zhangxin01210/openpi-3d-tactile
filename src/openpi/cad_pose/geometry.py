"""Pose/geometry contract: base_link metres, fixed exporter point ordering."""
from __future__ import annotations

import numpy as np


def fit_corresponding_rigid_pose(local_points: np.ndarray, observed_points: np.ndarray,
                                  mask: np.ndarray, *, max_rmse_m: float) -> tuple[np.ndarray, float]:
    """Recover only from known ordered correspondences; never performs ICP.

    A low fitting residual alone does not establish correspondence, calibration,
    or semantic correctness. The caller must verify all three independently.
    """
    source = np.asarray(local_points, dtype=np.float64)
    target = np.asarray(observed_points, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if source.shape != (512, 3) or target.shape != source.shape or mask.shape != (512,):
        raise ValueError("expected corresponding local/base arrays [512,3] and mask [512]")
    if not np.isfinite(max_rmse_m) or max_rmse_m <= 0:
        raise ValueError("max_rmse_m must be finite and positive")
    x, y = source[mask], target[mask]
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("at least three finite corresponding points are required")
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    if np.linalg.matrix_rank(x - x_mean, tol=1e-8) < 2:
        raise ValueError("correspondences are geometrically degenerate")
    u, _, vt = np.linalg.svd((x - x_mean).T @ (y - y_mean))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = y_mean - rotation @ x_mean
    rmse = float(np.sqrt(np.mean(np.sum((x @ rotation.T + translation - y) ** 2, axis=-1))))
    if not np.isfinite(rmse) or rmse > max_rmse_m:
        raise ValueError(f"ordered rigid-fit residual {rmse:.8g} m exceeds {max_rmse_m:.8g} m")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    return transform, rmse


def predicted_geometry_packet(local_points: np.ndarray, transforms: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Build the existing dual-object 512-point packet; no GT/tracker fallback.

    source=1 identifies inferred geometry; confidence=0 means no calibrated
    confidence is available. FinalV2RelationEncoder uses XYZ and mask only.
    """
    from scripts.preprocess.export_object_geometry_rgbtrack import cloud_from_pose

    points = np.asarray(local_points, dtype=np.float32)
    poses = np.asarray(transforms, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points.shape != (2, 512, 3) or poses.shape != (2, 4, 4) or valid.shape != (2,):
        raise ValueError("expected CAD [2,512,3], poses [2,4,4], valid [2]")
    if not np.isfinite(points).all():
        raise ValueError("CAD points must be finite")
    result = {}
    for i, role in enumerate(("manipulated", "target")):
        xyz = cloud_from_pose({"T_base_link_object": poses[i]}, points[i]) if valid[i] else np.zeros((512, 3), np.float32)
        result[f"{role}_xyz_m"] = xyz
        result[f"{role}_mask"] = np.full(512, valid[i], dtype=bool)
        result[f"{role}_source"] = np.full(512, 1, dtype=np.int8)
        result[f"{role}_age"] = np.zeros(512, dtype=np.int16)
        result[f"{role}_confidence"] = np.zeros(512, dtype=np.float32)
    return result
