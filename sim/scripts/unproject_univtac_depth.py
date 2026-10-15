#!/usr/bin/env python3
"""Back-project Isaac Lab image-plane depth using an OpenGL camera pose.

Depth is in metres, the 3x3 intrinsic maps OpenCV pixel coordinates, and the
camera pose is (world position xyz, OpenGL quaternion wxyz).
"""

from __future__ import annotations

import numpy as np


def unproject_opengl(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    pose_w_opengl: np.ndarray,
    *,
    stride: int = 1,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world XYZ points and matching flat pixel indices."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    k = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(pose_w_opengl, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or pose.shape != (7,):
        raise ValueError(f"Expected depth HxW, intrinsic 3x3, pose 7; got {depth.shape}, {k.shape}, {pose.shape}")
    if stride < 1 or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError("Invalid stride, intrinsic or camera pose")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Focal lengths must be positive")

    height, width = depth.shape
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    values = depth[::stride, ::stride]
    valid = np.isfinite(values) & (values > 0)
    if max_depth_m is not None:
        valid &= values <= max_depth_m
    z = values[valid]
    u = cols[valid]
    v = rows[valid]
    # Image-plane depth uses +z forward and +y down in pixel coordinates.
    # OpenGL camera coordinates use -z forward and +y up.
    local = np.column_stack(((u - k[0, 2]) * z / k[0, 0],
                             -(v - k[1, 2]) * z / k[1, 1], -z))
    qw, qx, qy, qz = pose[3:]
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-8:
        raise ValueError("Camera quaternion has zero norm")
    qw, qx, qy, qz = np.array([qw, qx, qy, qz]) / norm
    rotation = np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ])
    world = local @ rotation.T + pose[:3]
    pixel_indices = (v * width + u).astype(np.int64)
    return world.astype(np.float32), pixel_indices


def write_binary_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a colored point cloud without a visualization dependency."""
    points = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(f"Expected matching Nx3 points and colors; got {points.shape}, {colors.shape}")
    vertices = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, axis]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
