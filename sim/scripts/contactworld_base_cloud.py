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


def _stride(ix, iy, iz, count, *, allow_replacement=False):
    if len(ix) < count:
        if not allow_replacement or len(ix) == 0:
            raise ValueError("Only %d voxel candidates for %d points" % (len(ix), count))
    code = _part1by2(ix) | (_part1by2(iy) << np.uint64(1)) | (
        _part1by2(iz) << np.uint64(2))
    order = np.argsort(code, kind="mergesort")
    # The replacement path is only used to fill a fixed-size online input
    # when a new camera pose has fewer occupied voxels than the contract.  It
    # keeps the spatial ordering deterministic; the normal path remains
    # duplicate-free and unchanged.
    positions = np.rint(np.linspace(0, len(order) - 1, count)).astype(np.int64)
    return order[positions]


def base_cloud(depth_by_camera, rgb_by_camera, view_by_camera, projection_by_camera,
               base_pos_world, base_quat_world, *, cameras=("front", "wrist"),
               sampler="surface", points=POINTS, allow_sparse=False):
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
    raw_ix, raw_iy, raw_iz = (np.floor((xyz[:, i] - ROI[i][0]) / VOXEL_M).astype(np.int32)
                              for i in range(3))
    ix, iy, iz = raw_ix, raw_iy, raw_iz
    nx = int(np.ceil((ROI[0][1] - ROI[0][0]) / VOXEL_M)) + 1
    ny = int(np.ceil((ROI[1][1] - ROI[1][0]) / VOXEL_M)) + 1
    linear = ix.astype(np.int64) + nx * (iy.astype(np.int64) + ny * iz.astype(np.int64))
    _, first = np.unique(linear, return_index=True)
    first.sort()  # first observed RGB-D pixel per occupied voxel
    vx, vu, vc = xyz[first], uv[first], camera_id[first]
    ix, iy, iz = ix[first], iy[first], iz[first]
    sparse_fallback = False
    selected_xyz = selected_uv = selected_camera_id = None
    if sampler == "uniform":
        chosen = _stride(ix, iy, iz, points)
    else:
        above = np.flatnonzero(vx[:, 2] > TABLE_CUTOFF_M)
        table = np.flatnonzero(vx[:, 2] <= TABLE_CUTOFF_M)
        if len(above) >= points:
            chosen = above[_stride(ix[above], iy[above], iz[above], points)]
        elif len(table) >= points - len(above):
            chosen = np.r_[above, table[_stride(
                ix[table], iy[table], iz[table], points - len(above))]]
            # Match the reviewed front/fused candidates exactly.
            seed = 2602 if cameras == ("front",) else 2603
            chosen = chosen[np.random.default_rng(seed).permutation(points)]
        elif allow_sparse:
            # Some task geometries and random evaluation poses expose fewer
            # than 4096 occupied 5 mm voxels, although the raw depth ROI still
            # contains enough pixels.  Preserve the fixed 4096-point contract
            # by taking all available above/table voxel representatives and
            # filling only the deficit with deterministic raw ROI pixels.  The
            # fallback is deliberately restricted to the same ROI and is
            # reported in ``counts`` for downstream auditing.
            sparse_fallback = True
            need = points - len(above) - len(table)
            rep_mask = np.zeros(len(xyz), dtype=bool)
            rep_mask[first] = True
            raw_table = np.flatnonzero((xyz[:, 2] <= TABLE_CUTOFF_M) & ~rep_mask)
            if len(raw_table) == 0:
                raw_table = np.flatnonzero(xyz[:, 2] <= TABLE_CUTOFF_M)
            if len(raw_table) == 0:
                raise ValueError("No raw table points available for sparse cloud fallback")
            if len(raw_table) >= need:
                raw_pick = _stride(raw_ix[raw_table], raw_iy[raw_table],
                                   raw_iz[raw_table], need)
            else:
                raw_pick = _stride(raw_ix[raw_table], raw_iy[raw_table],
                                   raw_iz[raw_table], need,
                                   allow_replacement=True)
            raw_selected = raw_table[raw_pick]
            selected_xyz = np.concatenate((vx[above], vx[table], xyz[raw_selected]), axis=0)
            selected_uv = np.concatenate((vu[above], vu[table], uv[raw_selected]), axis=0)
            selected_camera_id = np.concatenate((vc[above], vc[table], camera_id[raw_selected]), axis=0)
            seed = 2604 if cameras == ("front",) else 2605
            order = np.random.default_rng(seed).permutation(points)
            selected_xyz = selected_xyz[order]
            selected_uv = selected_uv[order]
            selected_camera_id = selected_camera_id[order]
        else:
            raise ValueError("Only %d voxel candidates for %d points" % (len(vx), points))
    if selected_xyz is None:
        xyz_out, uv_out, id_out = vx[chosen], vu[chosen], vc[chosen]
    else:
        xyz_out, uv_out, id_out = selected_xyz, selected_uv, selected_camera_id
    rgb_out = np.empty((points, 3), dtype=np.uint8)
    for index, camera in enumerate(cameras):
        mask = id_out == index
        rgb_out[mask] = rgb_by_camera[camera][uv_out[mask, 1], uv_out[mask, 0]]
    if not (xyz_out.shape == (points, 3) and np.isfinite(xyz_out).all()):
        raise ValueError("Invalid sampled base cloud")
    counts = {"raw_depth_roi": int(len(xyz)), "voxel_candidates": int(len(vx)),
              "above_table_voxels": int((vx[:, 2] > TABLE_CUTOFF_M).sum()),
              "wrist_selected": int((id_out == 1).sum()),
              "sparse_fallback": bool(sparse_fallback)}
    return xyz_out.astype(np.float32), rgb_out, id_out, uv_out.astype(np.int16), counts
