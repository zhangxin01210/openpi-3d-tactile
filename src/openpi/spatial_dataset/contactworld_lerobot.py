"""Join the converted ContactWorld LeRobot episodes with their spatial arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ContactWorldLeRobotSpatial:
    def __init__(self, base, root: str | Path, *, visual: bool, force: bool,
                 depth: bool = False, cloud_noise_m: float = 0.0,
                 force_ee3d_proxy: bool = False, action_horizon: int = 16):
        self.base = base
        self.root = Path(root)
        manifest = json.loads((self.root / "spatial/manifest.json").read_text())
        if manifest["format"] != "contactworld-spatial-v1":
            raise ValueError("Wrong ContactWorld spatial format")
        self.episodes = {int(ep["episode_index"]): ep for ep in manifest["episodes"]}
        self.sample_indices = []
        offset = 0
        for ep in manifest["episodes"]:
            length = int(ep["length"])
            if ep.get("split", "train") == "train":
                self.sample_indices.extend(range(offset, offset + max(0, length - action_horizon + 1)))
            offset += length
        if offset != len(base) or not self.sample_indices:
            raise ValueError("ContactWorld episode table does not match LeRobot dataset")
        self.visual = visual
        self.force = force
        self.depth = depth
        self.cloud_noise_m = cloud_noise_m
        self._noise_rng = np.random.default_rng(314159)
        self.force_ee3d_proxy = force_ee3d_proxy
        self._arrays: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self):
        return len(self.sample_indices)

    def _row(self, episode: int, frame: int, name: str) -> np.ndarray:
        key = episode, name
        if key not in self._arrays:
            path = self.root / "spatial/episodes" / f"episode_{episode:06d}" / f"{name}.npy"
            self._arrays[key] = np.load(path, mmap_mode="r")
        array = self._arrays[key]
        if frame < 0 or frame >= self.episodes[episode]["length"]:
            raise IndexError((episode, frame))
        return np.asarray(array[frame]).copy()

    def __getitem__(self, index):
        sample = dict(self.base[self.sample_indices[index]])
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        if episode not in self.episodes:
            raise KeyError(f"Missing spatial episode {episode}")
        if self.depth:
            sample["observation.tactile_depth"] = self._row(episode, frame, "tactile_depth").astype(np.float32)
        visual = None
        tactile = None
        if self.visual:
            xyz = self._row(episode, frame, "pointcloud_xyz").astype(np.float32)
            if self.cloud_noise_m:
                xyz += self._noise_rng.normal(0.0, self.cloud_noise_m, xyz.shape).astype(np.float32)
            if xyz.shape != (1024, 3) or not np.isfinite(xyz).all():
                raise ValueError("Bad ContactWorld pointcloud")
            visual = {"xyz_m": xyz, "rgb": np.zeros((1024, 3), np.uint8),
                      "rgb_valid": np.zeros(1024, bool), "point_mask": np.ones(1024, bool)}
        if self.force:
            force = self._row(episode, frame, "force_grid").reshape(140, 3).astype(np.float32)
            if not np.isfinite(force).all():
                raise ValueError("Bad ContactWorld force field")
            yy, xx = np.meshgrid(np.linspace(-1, 1, 10, dtype=np.float32),
                                 np.linspace(-1, 1, 14, dtype=np.float32), indexing="ij")
            xyz = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(140, 3)
            if self.force_ee3d_proxy:
                # EE-centred plane proxy. This does not assert calibrated taxel poses.
                pose = self._row(episode, frame, "ee_pose").astype(np.float32)
                x, y, z, w = pose[3:]
                rotation = np.array([
                    [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                    [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                    [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
                ], dtype=np.float32)
                xyz = (xyz * 0.01) @ rotation.T + pose[:3]
                force = force @ rotation.T
            tactile = {"xyz_m": xyz, "force": force,
                       "force_norm": np.linalg.norm(force, axis=-1),
                       "finger_id": np.zeros(140, np.int32),
                       "taxel_id": np.arange(140, dtype=np.int32),
                       "point_mask": np.ones(140, bool)}
        if visual is not None or tactile is not None:
            sample["spatial"] = {"visual": visual, "tactile": tactile}
        return sample
