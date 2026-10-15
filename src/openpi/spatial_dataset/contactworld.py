"""Episode-safe ContactWorld Zarr rows for supervised policy training.

The released cloud uses a source-specific camera conversion. Its base-frame
correction comes from the measured legacy Gym front-camera geometry report.
Historical collection-time calibration was not published.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


TASKS = ("insertion_usb", "insertion_peg")


def episode_split(task: str, episode: int) -> str:
    """Stable 80/10/10 split by complete trajectory, independent of Python hash seed."""
    if task not in TASKS:
        raise ValueError(task)
    bucket = int(hashlib.sha256(f"{task}:{episode}".encode()).hexdigest()[:12], 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def _rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError(f"Expected finite HWC RGB, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if image.min() < -1e-5 or image.max() > 1 + 1e-5:
        raise ValueError("Released RGB must be in [0,1]")
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


class ContactWorldDataset:
    """Return RGB, joint state, corrected cloud, signed TacFF and action horizon.

    Uses on-demand Zarr reads. Each action horizon is padded at the trajectory
    boundary and accompanied by a validity mask; no observation of future frames
    or success label is included. Point clouds retain the released 1024 samples.
    """

    def __init__(self, root: str | Path, geometry_report: str | Path, *, split: str = "train",
                 action_horizon: int = 16, tasks: tuple[str, ...] = TASKS):
        if split not in {"train", "val", "test"} or action_horizon < 1:
            raise ValueError("Invalid split or action horizon")
        self.root = Path(root)
        self.split = split
        self.action_horizon = action_horizon
        report = json.loads(Path(geometry_report).read_text())
        if report.get("status") != "passed" or not all(report["tasks"][t]["passed"] for t in tasks):
            raise ValueError("Point-cloud geometry contract has not passed for requested tasks")
        self.rotation = np.asarray(report["A"], dtype=np.float32)
        self.translation = np.asarray(report["t"], dtype=np.float32)
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("Invalid point-cloud transform")
        self.groups = {task: zarr.open_group(str(self.root / task), mode="r") for task in tasks}
        self.samples: list[tuple[str, int, int, int, int]] = []
        self.episodes: list[tuple[str, int, int]] = []
        for task, group in self.groups.items():
            ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
            if not (np.diff(ends) > 0).all() or ends[-1] != len(group["data/action"]):
                raise ValueError(f"Corrupt episode boundaries: {task}")
            starts = np.r_[0, ends[:-1]]
            for episode, (start, end) in enumerate(zip(starts, ends)):
                if episode_split(task, episode) != split:
                    continue
                self.episodes.append((task, episode, int(end-start)))
                self.samples.extend((task, episode, row, int(end), row-int(start))
                                    for row in range(int(start), int(end)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        task, episode, row, end, local_frame = self.samples[index]
        data = self.groups[task]["data"]
        rgb_front = _rgb(data["front"][row])
        rgb_wrist = _rgb(data["wrist"][row])
        rgb_tactile = _rgb(data["tactile_rgb_right"][row])
        q = np.asarray(data["dof_pos"][row], dtype=np.float32)
        dq = np.asarray(data["dof_vel"][row], dtype=np.float32)
        state = np.concatenate((q, dq))
        cloud = np.asarray(data["pointcloud"][row], dtype=np.float32).copy()
        if cloud.shape != (1024, 6):
            raise ValueError(f"Invalid point-cloud shape {task}:{episode}:{row}: {cloud.shape}")
        cloud[:, :3] = cloud[:, :3] @ self.rotation.T + self.translation
        force = np.asarray(data["tactile_force_field_right"][row], dtype=np.float32)
        if force.shape != (10, 14, 3):
            raise ValueError(f"Invalid force-field shape {task}:{episode}:{row}: {force.shape}")
        stop = min(row + self.action_horizon, end)
        count = stop - row
        actions = np.zeros((self.action_horizon, 6), dtype=np.float32)
        actions[:count] = data["action"][row:stop]
        valid = np.arange(self.action_horizon) < count
        if not all(np.isfinite(array).all() for array in (state, cloud, force, actions)):
            raise ValueError(f"Nonfinite row {task}:{episode}:{row}")
        return {
            "front_rgb": rgb_front,
            "wrist_rgb": rgb_wrist,
            "tactile_rgb_right": rgb_tactile,
            "state": state,
            "pointcloud_base_xyzrgb": cloud,
            "tactile_force_field_local": force,
            "actions": actions,
            "action_valid": valid,
            "task_id": np.int64(TASKS.index(task)),
            "episode_index": np.int64(episode),
            "frame_index": np.int64(local_frame),
        }
