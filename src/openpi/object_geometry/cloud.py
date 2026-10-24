"""Dependency-free RGB-D and mask geometry operations.

The functions here deliberately accept calibration arrays instead of importing
OpenPI's camera classes. This keeps the object-geometry core usable in tests and
in downstream data tools without pulling in SAM2 or robot-specific code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openpi.object_geometry.types import PointCloud


def as_points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"points must have shape (N, 3) and finite values, got {array.shape}")
    return array


def transform_points(points: Any, transform: Any) -> np.ndarray:
    """Apply a homogeneous base-frame transform to row-major points."""

    xyz = as_points(points)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"transform must have finite shape (4, 4), got {matrix.shape}")
    return (xyz @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def depth_to_base_points(
    depth: Any,
    rays_base_per_meter: Any,
    *,
    depth_scale_m_per_unit: float,
    base_origin_m: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten valid depth pixels into base-frame points.

    Returns ``(xyz_base, source_pixel, depth_m)``. ``source_pixel`` contains
    flattened indices into the original depth image, so provenance survives
    later mask projection.
    """

    raw_depth = np.asarray(depth)
    rays = np.asarray(rays_base_per_meter, dtype=np.float32)
    if raw_depth.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {raw_depth.shape}")
    if rays.shape != (*raw_depth.shape, 3):
        raise ValueError(f"rays_base_per_meter must have shape {raw_depth.shape + (3,)}, got {rays.shape}")
    scale = float(depth_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale_m_per_unit must be positive and finite")
    origin = np.zeros(3, dtype=np.float32) if base_origin_m is None else np.asarray(base_origin_m, dtype=np.float32)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError(f"base_origin_m must have shape (3,), got {origin.shape}")
    depth_m = raw_depth.astype(np.float32).reshape(-1) * scale
    flat_rays = rays.reshape(-1, 3)
    valid = np.isfinite(depth_m) & (depth_m > 0) & np.isfinite(flat_rays).all(axis=1)
    source_pixel = np.flatnonzero(valid).astype(np.int32)
    xyz = flat_rays[valid] * depth_m[valid, None] + origin[None, :]
    return xyz.astype(np.float32), source_pixel, depth_m[valid].astype(np.float32)


def project_pinhole(points_camera: Any, intrinsics: Any) -> np.ndarray:
    """Project camera-frame points to pixel coordinates, returning NaN behind camera."""

    xyz = as_points(points_camera)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"intrinsics must have finite shape (3, 3), got {matrix.shape}")
    uvw = xyz @ matrix.T
    uv = np.full((len(xyz), 2), np.nan, dtype=np.float32)
    visible = np.isfinite(uvw).all(axis=1) & (uvw[:, 2] > 0)
    uv[visible] = (uvw[visible, :2] / uvw[visible, 2:3]).astype(np.float32)
    return uv


def mask_depth_to_cloud(
    depth: Any,
    mask: Any,
    rays_base_per_meter: Any,
    transform_base_color: Any,
    color_intrinsics: Any,
    *,
    depth_scale_m_per_unit: float,
    rgb: Any | None = None,
    workspace_bounds: tuple[float, float, float, float, float, float] | None = None,
    mask_confidence: Any | None = None,
) -> PointCloud:
    """Build candidate base-frame geometry from an RGB mask and depth.

    Depth pixels are projected into the color camera and selected when their
    projected pixel lies in ``mask``. This is candidate geometry: transparent
    objects can include background depth and are never asserted to be surfaces.
    """

    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {binary_mask.shape}")
    xyz_base, source_pixel, depth_m = depth_to_base_points(
        depth,
        rays_base_per_meter,
        depth_scale_m_per_unit=depth_scale_m_per_unit,
    )
    color_xyz = transform_points(xyz_base, np.linalg.inv(np.asarray(transform_base_color, dtype=np.float64)))
    uv = project_pinhole(color_xyz, color_intrinsics)
    pixels = np.rint(np.nan_to_num(uv, nan=-1e9)).astype(np.int64)
    h, w = binary_mask.shape
    valid = (
        np.isfinite(uv).all(axis=1)
        & (color_xyz[:, 2] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    inside = np.zeros(len(xyz_base), dtype=bool)
    inside[valid] = binary_mask[pixels[valid, 1], pixels[valid, 0]]
    selected = valid & inside
    if workspace_bounds is not None:
        if len(workspace_bounds) != 6:
            raise ValueError("workspace_bounds must be (xmin, xmax, ymin, ymax, zmin, zmax)")
        xmin, xmax, ymin, ymax, zmin, zmax = map(float, workspace_bounds)
        selected &= (
            (xyz_base[:, 0] >= xmin)
            & (xyz_base[:, 0] <= xmax)
            & (xyz_base[:, 1] >= ymin)
            & (xyz_base[:, 1] <= ymax)
            & (xyz_base[:, 2] >= zmin)
            & (xyz_base[:, 2] <= zmax)
        )

    colors = None
    if rgb is not None:
        color_image = np.asarray(rgb)
        if color_image.shape != (h, w, 3):
            raise ValueError(f"rgb must have shape {(h, w, 3)}, got {color_image.shape}")
        colors = color_image[pixels[selected, 1], pixels[selected, 0]].astype(np.uint8)

    confidence = None
    if mask_confidence is not None:
        confidence_image = np.asarray(mask_confidence, dtype=np.float32)
        if confidence_image.shape != binary_mask.shape:
            raise ValueError(f"mask_confidence must have shape {binary_mask.shape}, got {confidence_image.shape}")
        confidence = confidence_image[pixels[selected, 1], pixels[selected, 0]]

    return PointCloud(
        xyz=xyz_base[selected],
        rgb=colors,
        source_pixel=source_pixel[selected],
        confidence=confidence,
    )
