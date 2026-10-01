"""On-demand world point clouds for UniVTAC LeRobot RGB/joint episodes.

The source HDF5 is retained as the calibration/depth source of truth. This
visual-only adapter does not reinterpret GelSight indentation as force.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


def _world_cloud(depth: np.ndarray, intrinsic: np.ndarray, pose: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Back-project image-plane depth with a world OpenGL camera pose."""
    depth = np.asarray(depth, dtype=np.float64).squeeze(-1)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if depth.ndim != 2 or intrinsic.shape != (3, 3) or pose.shape != (7,):
        raise ValueError("Unexpected UniVTAC depth/intrinsic/pose shapes")
    if not np.isfinite(intrinsic).all() or not np.isfinite(pose).all():
        raise ValueError("Invalid camera calibration")
    height, width = depth.shape
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0) & (z <= 3.0)
    z, u, v = z[valid], u[valid], v[valid]
    local = np.column_stack(((u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                             -(v - intrinsic[1, 2]) * z / intrinsic[1, 1], -z))
    qw, qx, qy, qz = pose[3:] / np.linalg.norm(pose[3:])
    rotation = np.asarray([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ])
    xyz = local @ rotation.T + pose[:3]
    return xyz.astype(np.float32), (v * width + u).astype(np.int64)


class UniVTACVisualDataset:
    """Join LeRobot rows to their original HDF5 world RGB-D observations."""

    def __init__(self, base: Any, dataset_root: str | Path, *, points_per_camera: int = 1024, stride: int = 4):
        self.base = base
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if points_per_camera < 1 or stride < 1:
            raise ValueError("points_per_camera and stride must be positive")
        self.points_per_camera = points_per_camera
        self.stride = stride
        manifest = json.loads((self.dataset_root / "source_manifest.json").read_text())
        self.source_root = Path(manifest["source"])
        self.seeds = tuple(int(episode["source_seed"]) for episode in manifest["episodes"])
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Duplicate source seeds in LeRobot manifest")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base[index])
        episode = int(np.asarray(item["episode_index"]))
        frame = int(np.asarray(item["frame_index"]))
        seed = self.seeds[episode]
        hdf5_path = self.source_root / "hdf5" / f"{seed}.hdf5"
        all_xyz = []
        all_rgb = []
        with h5py.File(hdf5_path, "r") as h5:
            if int(h5["step"][frame]) != int(np.asarray(item["source_step"])):
                raise ValueError(f"LeRobot/HDF5 step mismatch at episode {episode}, frame {frame}")
            for camera in ("head", "wrist"):
                prefix = f"observation/{camera}"
                xyz, pixels = _world_cloud(
                    h5[f"{prefix}/depth"][frame],
                    h5[f"{prefix}/intrinsic"][frame],
                    h5[f"{prefix}/pose_w_opengl"][frame],
                    self.stride,
                )
                image = cv2.imdecode(np.frombuffer(h5[f"{prefix}/rgb"][frame], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != h5[f"{prefix}/depth"].shape[1:3]:
                    raise ValueError(f"Bad {camera} RGB/depth at episode {episode}, frame {frame}")
                if len(xyz) < self.points_per_camera:
                    raise ValueError(f"Only {len(xyz)} {camera} points at episode {episode}, frame {frame}")
                selection = np.linspace(0, len(xyz) - 1, self.points_per_camera, dtype=np.int64)
                all_xyz.append(xyz[selection])
                all_rgb.append(image.reshape(-1, 3)[pixels[selection]])
        xyz = np.concatenate(all_xyz)
        rgb = np.concatenate(all_rgb)
        item["spatial"] = {"visual": {
            "xyz_m": xyz,
            "rgb": rgb,
            "rgb_valid": np.ones(len(xyz), dtype=bool),
            "point_mask": np.ones(len(xyz), dtype=bool),
        }}
        return item
