"""Join ContactWorld replayed LeRobot rows to approved 4096-point spatial data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ContactWorldV2LeRobotSpatial:
    def __init__(self, base, root: str | Path, *, cloud_mode: str,
                 force_mode: str, depth: bool = False, action_horizon: int = 16):
        if cloud_mode not in {"none", "front", "fused"}:
            raise ValueError(cloud_mode)
        if force_mode not in {"none", "local", "base3d", "local_at_base",
                              "left_base3d", "both_base3d", "both_local_grid"}:
            raise ValueError(force_mode)
        self.base = base
        self.root = Path(root)
        manifest = json.loads((self.root / "spatial/manifest.json").read_text())
        if manifest["format"] not in {"contactworld-spatial-v2", "contactworld-spatial-v3-bilateral"}:
            raise ValueError("Expected ContactWorld replayed spatial sidecar")
        self.bilateral = manifest["format"] == "contactworld-spatial-v3-bilateral"
        if force_mode in {"left_base3d", "both_base3d", "both_local_grid"} and not self.bilateral:
            raise ValueError("Left TacFF requires bilateral v3 dataset")
        self.episodes = {int(ep["episode_index"]): ep for ep in manifest["episodes"]}
        self.sample_indices = []
        offset = 0
        for episode in manifest["episodes"]:
            length = int(episode["length"])
            if episode["split"] == "train":
                self.sample_indices.extend(range(offset, offset + max(0, length - action_horizon + 1)))
            offset += length
        if offset != len(base) or not self.sample_indices:
            raise ValueError("ContactWorld v2 LeRobot episodes and spatial manifest differ")
        self.cloud_mode = cloud_mode
        self.force_mode = force_mode
        self.depth = depth
        self._arrays: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self):
        return len(self.sample_indices)

    def _row(self, episode: int, frame: int, key: str) -> np.ndarray:
        identity = episode, key
        if identity not in self._arrays:
            path = self.root / "spatial/episodes" / f"episode_{episode:06d}" / f"{key}.npy"
            self._arrays[identity] = np.load(path, mmap_mode="r")
        return np.asarray(self._arrays[identity][frame]).copy()

    def __getitem__(self, index):
        item = dict(self.base[self.sample_indices[index]])
        episode = int(np.asarray(item["episode_index"]).item())
        frame = int(np.asarray(item["frame_index"]).item())
        if episode not in self.episodes or not 0 <= frame < self.episodes[episode]["length"]:
            raise IndexError((episode, frame))
        if self.depth:
            if self.bilateral:
                raise ValueError("V3 bilateral force dataset has no unilateral tactile depth")
            item["observation.tactile_depth"] = self._row(
                episode, frame, "tactile_depth").astype(np.float32)
        visual = tactile = None
        if self.cloud_mode != "none":
            key = "pointcloud_front_xyz" if self.cloud_mode == "front" else "pointcloud_fused_xyz"
            xyz = self._row(episode, frame, key).astype(np.float32)
            if xyz.shape != (4096, 3) or not np.isfinite(xyz).all():
                raise ValueError("Invalid v2 4096-point cloud")
            visual = {"xyz_m": xyz, "rgb": np.zeros((4096, 3), np.uint8),
                      "rgb_valid": np.zeros(4096, bool), "point_mask": np.ones(4096, bool)}
        if self.force_mode != "none":
            if self.force_mode in {"both_base3d", "both_local_grid"}:
                left_xyz = self._row(episode, frame, "tactile_xyz_left_base").astype(np.float32)
                right_xyz = self._row(episode, frame, "tactile_xyz_right_base").astype(np.float32)
                if self.force_mode == "both_local_grid":
                    left_force = self._row(episode, frame, "force_grid_left_local").reshape(140, 3).astype(np.float32)
                    right_force = self._row(episode, frame, "force_grid_right_local").reshape(140, 3).astype(np.float32)
                else:
                    left_force = self._row(episode, frame, "tactile_force_left_base").astype(np.float32)
                    right_force = self._row(episode, frame, "tactile_force_right_base").astype(np.float32)
                xyz = np.concatenate((left_xyz, right_xyz))
                force = np.concatenate((left_force, right_force))
                finger_id = np.repeat(np.array([0, 1], np.int32), 140)
                taxel_id = np.tile(np.arange(140, dtype=np.int32), 2)
            elif self.force_mode == "left_base3d":
                xyz = self._row(episode, frame, "tactile_xyz_left_base").astype(np.float32)
                force = self._row(episode, frame, "tactile_force_left_base").astype(np.float32)
            elif self.force_mode in {"base3d", "local_at_base"}:
                xyz_key = "tactile_xyz_right_base" if self.bilateral else "tactile_xyz_base"
                xyz = self._row(episode, frame, xyz_key).astype(np.float32)
                key = ("tactile_force_right_base" if self.bilateral else "tactile_force_base") if self.force_mode == "base3d" else ("force_grid_right_local" if self.bilateral else "force_grid_local")
                force = self._row(episode, frame, key).reshape(140, 3).astype(np.float32)
                if self.force_mode == "local_at_base":
                    # Source channels are [normal, shear_x, shear_y], whose
                    # sensor-local Cartesian vector is [-shear_x, -normal, shear_y].
                    force = force[:, [1, 0, 2]] * np.array([-1, -1, 1], np.float32)
            else:
                key = "force_grid_right_local" if self.bilateral else "force_grid_local"
                force = self._row(episode, frame, key).reshape(140, 3).astype(np.float32)
                yy, xx = np.meshgrid(np.linspace(-1, 1, 10, dtype=np.float32),
                                     np.linspace(-1, 1, 14, dtype=np.float32), indexing="ij")
                xyz = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(140, 3)
            if self.force_mode not in {"both_base3d", "both_local_grid"}:
                finger_id = np.full(140, int(self.bilateral and self.force_mode != "left_base3d"), np.int32)
                taxel_id = np.arange(140, dtype=np.int32)
            count = 280 if self.force_mode in {"both_base3d", "both_local_grid"} else 140
            if (xyz.shape != (count, 3) or force.shape != (count, 3) or
                    not np.isfinite(xyz).all() or not np.isfinite(force).all()):
                raise ValueError("Invalid v2 right tactile field")
            tactile = {"xyz_m": xyz, "force": force,
                       "force_norm": np.linalg.norm(force, axis=-1),
                       "finger_id": finger_id,
                       "taxel_id": taxel_id,
                       "point_mask": np.ones(count, bool)}
        if visual is not None or tactile is not None:
            item["spatial"] = {"visual": visual, "tactile": tactile}
        return item
