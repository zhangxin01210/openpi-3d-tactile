"""Pose evaluation with explicit, object-specific rotational symmetries."""
from __future__ import annotations

import numpy as np


def pose_errors(predicted: np.ndarray, target: np.ndarray, points: np.ndarray,
                symmetry_rotations: np.ndarray | None = None) -> dict[str, float]:
    """Compute translation, symmetry-aware rotation and point correspondence errors.

    Symmetries are explicitly supplied local-frame rotations. Identity-only is
    the default; no object is automatically treated as rotationally symmetric.
    """
    p = np.asarray(predicted, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 4) or t.shape != (4, 4) or points.shape != (512, 3):
        raise ValueError("expected two 4x4 transforms and ordered [512,3] CAD")
    if not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isfinite(points).all():
        raise ValueError("pose and CAD inputs must be finite")
    for matrix in (p, t):
        r = matrix[:3, :3]
        if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(r), 1.0, atol=1e-4)):
            raise ValueError("pose is not a proper rigid transform")
    syms = np.eye(3)[None] if symmetry_rotations is None else np.asarray(symmetry_rotations, dtype=np.float64)
    if syms.ndim != 3 or syms.shape[1:] != (3, 3) or not len(syms):
        raise ValueError("symmetry rotations must have shape [S,3,3]")
    if any(not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4)
           for r in syms):
        raise ValueError("symmetry transformations must be proper rotations")
    pred_xyz = points @ p[:3, :3].T + p[:3, 3]
    angles, geometry = [], []
    for symmetry in syms:
        equivalent = t[:3, :3] @ symmetry
        cosine = (np.trace(p[:3, :3].T @ equivalent) - 1) / 2
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        xyz = points @ equivalent.T + t[:3, 3]
        geometry.append(float(np.linalg.norm(pred_xyz - xyz, axis=-1).mean()))
    return {"translation_mm": float(np.linalg.norm(p[:3, 3] - t[:3, 3]) * 1000),
            "rotation_deg": min(angles), "cad_error_mm": min(geometry) * 1000}


def summarize(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("metrics must contain at least one finite sample")
    return {"count": len(data), "mean": float(data.mean()), "median": float(np.median(data)),
            "p90": float(np.percentile(data, 90)), "p95": float(np.percentile(data, 95))}
