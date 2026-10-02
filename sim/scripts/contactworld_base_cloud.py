"""Versioned ContactWorld RGB-D -> robot-base 4096-point cloud contract.

Pure NumPy/SciPy so the same implementation runs in the Python 3.8 Isaac Gym
runtime and during online policy evaluation. No object state or segmentation is
used. All points come from the corresponding camera depth pixel.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from contactworld_dense_capture import camera_xyz_from_matrices


ROI = ((-0.20, 0.65), (-0.35, 0.35), (-0.005, 0.35))
VOXEL_M = 0.005
TABLE_CUTOFF_M = 0.003
POINTS = 4096


def _part1by2(value):
    value = np.asarray(value, dtype=np.uint64).copy()
    value &= np.uint64(0x1FFFFF)
    value = (value | (value << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    value = (value | (value << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    value = (value | (value << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    value = (value | (value << np.uint64(4))) & np.uint64(0x10C30C30C30C3)
    value = (value | (value << np.uint64(2))) & np.uint64(0x1249249249249249)
    return value


def _stride(ix, iy, iz, count):
    if len(ix) < count:
        raise ValueError("Only %d voxel candidates for %d points" % (len(ix), count))
    code = _part1by2(ix) | (_part1by2(iy) << np.uint64(1)) | (
        _part1by2(iz) << np.uint64(2))
    order = np.argsort(code, kind="mergesort")
    return order[np.rint(np.linspace(0, len(order) - 1, count)).astype(np.int64)]


def base_cloud(depth_by_camera, rgb_by_camera, view_by_camera, projection_by_camera,
               base_pos_world, base_quat_world, *, cameras=("front", "wrist"),
               sampler="surface", points=POINTS):
    """Return (XYZ base, RGB, source camera ID, source pixel UV, counts)."""
    if cameras not in (("front",), ("front", "wrist")):
        raise ValueError("Only front or front+wrist clouds are approved")
    if sampler not in ("uniform", "surface"):
        raise ValueError(sampler)
    base_pos = np.asarray(base_pos_world, dtype=np.float64)
    base_quat = np.asarray(base_quat_world, dtype=np.float64)
    if base_pos.shape != (3,) or base_quat.shape != (4,):
        raise ValueError("Expected one robot base pose")
    inverse = Rotation.from_quat(base_quat).inv()
    xyz_parts, uv_parts, id_parts = [], [], []
    for camera_id, camera in enumerate(cameras):
        depth = np.asarray(depth_by_camera[camera], dtype=np.float32)
        rgb = np.asarray(rgb_by_camera[camera], dtype=np.uint8)
        if depth.shape != (256, 256) or rgb.shape != (256, 256, 3):
            raise ValueError("ContactWorld RGB-D shape mismatch: " + camera)
        world, uv = camera_xyz_from_matrices(
            depth, np.asarray(view_by_camera[camera]),
            np.asarray(projection_by_camera[camera]))
        xyz = inverse.apply(world - base_pos).astype(np.float32)
        inside = np.ones(len(xyz), dtype=bool)
        for axis, (low, high) in enumerate(ROI):
            inside &= (xyz[:, axis] >= low) & (xyz[:, axis] <= high)
        xyz_parts.append(xyz[inside])
        uv_parts.append(uv[inside])
        id_parts.append(np.full(int(inside.sum()), camera_id, dtype=np.uint8))
    xyz = np.concatenate(xyz_parts)
    uv = np.concatenate(uv_parts)
    camera_id = np.concatenate(id_parts)
    ix, iy, iz = (np.floor((xyz[:, i] - ROI[i][0]) / VOXEL_M).astype(np.int32)
                  for i in range(3))
    nx = int(np.ceil((ROI[0][1] - ROI[0][0]) / VOXEL_M)) + 1
    ny = int(np.ceil((ROI[1][1] - ROI[1][0]) / VOXEL_M)) + 1
    linear = ix.astype(np.int64) + nx * (iy.astype(np.int64) + ny * iz.astype(np.int64))
    _, first = np.unique(linear, return_index=True)
    first.sort()  # first observed RGB-D pixel per occupied voxel
    vx, vu, vc = xyz[first], uv[first], camera_id[first]
    ix, iy, iz = ix[first], iy[first], iz[first]
    if sampler == "uniform":
        chosen = _stride(ix, iy, iz, points)
    else:
        above = np.flatnonzero(vx[:, 2] > TABLE_CUTOFF_M)
        table = np.flatnonzero(vx[:, 2] <= TABLE_CUTOFF_M)
        if len(above) >= points:
            chosen = above[_stride(ix[above], iy[above], iz[above], points)]
        else:
            chosen = np.r_[above, table[_stride(
                ix[table], iy[table], iz[table], points - len(above))]]
            # Match the reviewed front/fused candidates exactly.
            seed = 2602 if cameras == ("front",) else 2603
            chosen = chosen[np.random.default_rng(seed).permutation(points)]
    xyz_out, uv_out, id_out = vx[chosen], vu[chosen], vc[chosen]
    rgb_out = np.empty((points, 3), dtype=np.uint8)
    for index, camera in enumerate(cameras):
        mask = id_out == index
        rgb_out[mask] = rgb_by_camera[camera][uv_out[mask, 1], uv_out[mask, 0]]
    if not (xyz_out.shape == (points, 3) and np.isfinite(xyz_out).all()):
        raise ValueError("Invalid sampled base cloud")
    counts = {"raw_depth_roi": int(len(xyz)), "voxel_candidates": int(len(vx)),
              "above_table_voxels": int((vx[:, 2] > TABLE_CUTOFF_M).sum()),
              "wrist_selected": int((id_out == 1).sum())}
    return xyz_out.astype(np.float32), rgb_out, id_out, uv_out.astype(np.int16), counts
