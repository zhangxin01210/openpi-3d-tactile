"""Capture both patched ContactWorld USB force fields during one replay.

This is a sensor audit sidecar, separate from the released right-only dataset
and from the unmodified v2 training export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contactworld_tactile_base import tactile_field_to_base


class BilateralTactileCapture:
    def __init__(self) -> None:
        self.data: dict[str, list[np.ndarray]] = {}

    def add(self, env, pre: dict, frame: int) -> None:
        base_pos = env.franka_base_pos[0].detach().cpu().numpy().copy()
        base_quat = env.franka_base_quat[0].detach().cpu().numpy().copy()
        values = {"frame": np.int32(frame),
                  "base_pose_world": np.r_[base_pos, base_quat].astype(np.float32),
                  "ee_pose": np.r_[pre["ee_pos"], pre["ee_quat"]].astype(np.float32),
                  "plug_pose": np.r_[pre["plug_pos"], pre["plug_quat"]].astype(np.float32),
                  "socket_pose": np.r_[pre["socket_pos_gt"], pre["socket_quat"]].astype(np.float32)}
        for side in ("left", "right"):
            key = "tactile_force_field_" + side
            field = np.asarray(pre[key], dtype=np.float32)
            pos = env.tactile_pos_world_dict[key][0].detach().cpu().numpy().copy()
            quat = env.tactile_quat_world_dict[key][0].detach().cpu().numpy().copy()
            xyz_base, force_base = tactile_field_to_base(field, pos, quat, base_pos, base_quat)
            values.update({"force_grid_" + side: field,
                           "taxel_xyz_base_" + side: xyz_base,
                           "force_base_" + side: force_base,
                           "taxel_quat_world_" + side: quat.astype(np.float32)})
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite bilateral capture array: " + key)
            self.data.setdefault(key, []).append(value)

    def save(self, output: Path) -> None:
        if not self.data:
            raise ValueError("No bilateral frames captured")
        path = Path(output) / "bilateral_tactile.npz"
        if path.exists():
            raise FileExistsError(path)
        stacked = {key: np.stack(value) for key, value in self.data.items()}
        np.savez_compressed(path, **stacked)
        (Path(output) / "bilateral_tactile.json").write_text(json.dumps({
            "status": "sensor_audit_only", "frames": len(stacked["frame"]),
            "sensors": ["left", "right"], "samples_per_sensor": 140,
            "channels": "source normal/shear-x/shear-y; base xyz derived using same-frame taxel poses",
            "force_units": "synthetic TacSL penalty-model units, not calibrated Newtons",
            "arrays": {key: list(value.shape) for key, value in stacked.items()},
        }, indent=2) + "\n")
