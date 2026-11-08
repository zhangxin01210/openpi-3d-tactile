"""RGB/CAD projection primitives for auditable pose predictions."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from openpi.cad_pose.metrics import pose_errors


def render_comparison(*, rgb: np.ndarray, intrinsic: np.ndarray, t_base_color: np.ndarray,
                      cad_points: np.ndarray, predicted_pose: np.ndarray, target_pose: np.ndarray | None,
                      label: str, output: Path, symmetry_rotations: np.ndarray | None = None) -> dict | None:
    """Write RGB, predicted CAD and GT/predicted CAD RGB projections to a new PNG.

    The caller must supply the actual predicted pose, the actual RGB frame and
    its calibration; no saved tracker overlay is used as a prediction.
    """
    rgb = np.asarray(rgb)
    k = np.asarray(intrinsic, dtype=np.float64)
    base_color = np.asarray(t_base_color, dtype=np.float64)
    cad = np.asarray(cad_points, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 [H,W,3]")
    if k.shape != (3, 3) or base_color.shape != (4, 4) or cad.shape != (512, 3):
        raise ValueError("invalid camera or 512-point CAD shape")
    if not np.isfinite(k).all() or not np.isfinite(base_color).all() or not np.isfinite(cad).all():
        raise ValueError("non-finite camera or CAD")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    color_base = np.linalg.inv(base_color)
    h, w = rgb.shape[:2]
    original, pred, compare = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for _ in range(3))

    def draw(image, pose, color):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 matrix")
        xyz_base = cad @ pose[:3, :3].T + pose[:3, 3]
        xyz = xyz_base @ color_base[:3, :3].T + color_base[:3, 3]
        good = np.isfinite(xyz).all(axis=-1) & (xyz[:, 2] > 1e-6)
        q = xyz[good] @ k.T
        pixels = np.rint(q[:, :2] / q[:, 2:3]).astype(np.int64)
        pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)]
        for u, v in pixels:
            cv2.circle(image, (int(u), int(v)), 1, color, -1, lineType=cv2.LINE_AA)

    draw(pred, predicted_pose, (255, 0, 255))
    draw(compare, predicted_pose, (255, 0, 255))
    metrics = None
    if target_pose is not None:
        metrics = pose_errors(predicted_pose, target_pose, cad, symmetry_rotations)
        draw(compare, target_pose, (0, 255, 0))
        line = f't={metrics["translation_mm"]:.1f}mm r={metrics["rotation_deg"]:.1f}deg CAD={metrics["cad_error_mm"]:.1f}mm'
    else:
        line = "GT unavailable/unverified"
    for image, title in ((original, "RGB"), (pred, "Prediction"), (compare, "GT green / prediction magenta")):
        cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.concatenate((original, pred, compare), axis=1)
    cv2.putText(panel, str(label), (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, line, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), panel):
        raise OSError(f"failed to write {output}")
    return metrics
