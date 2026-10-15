"""Capture same-frame replay data for the approved ContactWorld 4096-point variants.

The source simulator is unchanged. The robot base pose, camera matrices and
right-sensor taxel poses are read on every action-pre frame. Both cloud variants
use the same RGB-D and differ only in whether the wrist depth is fused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from contactworld_base_cloud import base_cloud, ROI, VOXEL_M, TABLE_CUTOFF_M, POINTS
from contactworld_tactile_base import tactile_field_to_base


class ContactWorldV2Capture:
    def __init__(self, episode):
        self.episode = int(episode)
        names = ("front", "wrist", "tactile_rgb", "tactile_depth",
                 "front_depth", "wrist_depth", "front_view", "front_projection",
                 "wrist_view", "wrist_projection", "base_pose_world",
                 "cloud_front_xyz", "cloud_front_uv", "cloud_fused_xyz",
                 "cloud_fused_uv", "cloud_fused_camera", "force_grid_local",
                 "tactile_xyz_base", "tactile_force_base", "tactile_quat_world",
                 "state", "ee_pose", "plug_pose", "socket_pose",
                 "action_executed", "action_recorded")
        self.data = {key: [] for key in names}
        self.rows = []

    def add(self, env, pre, demo, action, frame):
        images = env.get_camera_image_tensors_dict()
        front_depth = images["front_depth"][0].detach().cpu().numpy().astype(np.float32)
        wrist_depth = images["wrist_depth"][0].detach().cpu().numpy().astype(np.float32)
        rgb = {}
        for role in ("front", "wrist"):
            value = images[role][0].detach().cpu().numpy()
            rgb[role] = value if value.dtype == np.uint8 else np.rint(
                np.clip(value, 0, 1) * 255).astype(np.uint8)
            expected = np.rint(np.clip(pre[role], 0, 1) * 255).astype(np.uint8)
            if np.max(np.abs(rgb[role].astype(np.int16) - expected.astype(np.int16))) > 1:
                raise ValueError("Direct RGB differs from pre-action observation: " + role)
        views, projections = {}, {}
        for role in ("front", "wrist"):
            handle = env.camera_handles_list[0][role + "_depth"]
            # The reviewed pilot stored the fixed front matrix as float64 and
            # per-frame wrist matrices as float32. Preserve those exact inputs
            # so voxel boundary decisions match the approved preview.
            dtype = np.float64 if role == "front" else np.float32
            views[role] = np.asarray(env.gym.get_camera_view_matrix(
                env.sim, env.env_ptrs[0], handle), dtype=dtype)
            projections[role] = np.asarray(env.gym.get_camera_proj_matrix(
                env.sim, env.env_ptrs[0], handle), dtype=dtype)
        measured_base_pos = env.franka_base_pos[0].detach().cpu().numpy().astype(np.float32)
        measured_base_quat = env.franka_base_quat[0].detach().cpu().numpy().astype(np.float32)
        if (np.linalg.norm(measured_base_pos) > 1e-6 or
                np.linalg.norm(measured_base_quat - [0, 0, 0, 1]) > 1e-6):
            raise ValueError("Approved USB geometry assumes fixed world-aligned base")
        # Isaac Gym reports ~1e-8 m numerical jitter in the fixed base. Snap
        # that fixed coordinate frame to exact identity to reproduce the
        # reviewed 4096-point previews; retain the measured pose in the archive.
        base_pos = np.zeros(3, dtype=np.float32)
        base_quat = np.array([0, 0, 0, 1], dtype=np.float32)
        depths = {"front": front_depth, "wrist": wrist_depth}
        front, front_color, _, front_uv, front_count = base_cloud(
            depths, rgb, views, projections, base_pos, base_quat,
            cameras=("front",), sampler="surface", points=POINTS)
        fused, fused_color, fused_camera, fused_uv, fused_count = base_cloud(
            depths, rgb, views, projections, base_pos, base_quat,
            cameras=("front", "wrist"), sampler="surface", points=POINTS)
        force = np.asarray(pre["tactile_force_field_right"], dtype=np.float32)
        taxel_world = env.tactile_pos_world[0].detach().cpu().numpy().astype(np.float32)
        taxel_quat = env.tactile_quat_world[0].detach().cpu().numpy().astype(np.float32)
        tactile_xyz, tactile_force = tactile_field_to_base(
            force, taxel_world, taxel_quat, base_pos, base_quat)
        values = {
            "front": rgb["front"], "wrist": rgb["wrist"],
            "tactile_rgb": np.rint(np.clip(pre["tactile_rgb_right"], 0, 1) * 255).astype(np.uint8),
            "tactile_depth": np.asarray(pre["tactile_depth_right"], dtype=np.float16),
            "front_depth": front_depth, "wrist_depth": wrist_depth,
            "front_view": views["front"], "front_projection": projections["front"],
            "wrist_view": views["wrist"], "wrist_projection": projections["wrist"],
            "base_pose_world": np.r_[measured_base_pos, measured_base_quat],
            "cloud_front_xyz": front, "cloud_front_uv": front_uv,
            "cloud_fused_xyz": fused, "cloud_fused_uv": fused_uv,
            "cloud_fused_camera": fused_camera,
            "force_grid_local": force, "tactile_xyz_base": tactile_xyz,
            "tactile_force_base": tactile_force, "tactile_quat_world": taxel_quat,
            "state": np.r_[pre["dof_pos"], pre["dof_vel"]].astype(np.float32),
            "ee_pose": np.r_[pre["ee_pos"], pre["ee_quat"]].astype(np.float32),
            "plug_pose": np.r_[pre["plug_pos"], pre["plug_quat"]].astype(np.float32),
            "socket_pose": np.r_[pre["socket_pos_gt"], pre["socket_quat"]].astype(np.float32),
            "action_executed": np.asarray(action, dtype=np.float32).copy(),
            "action_recorded": np.asarray(demo["action"][frame], dtype=np.float32).copy(),
        }
        for key, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite capture array " + key)
            self.data[key].append(value)
        # RGB provenance is diagnostic only; model input remains XYZ.
        self.rows.append({"frame": int(frame), "front_roi_voxels": front_count["voxel_candidates"],
                          "fused_roi_voxels": fused_count["voxel_candidates"],
                          "front_above_table_voxels": front_count["above_table_voxels"],
                          "fused_above_table_voxels": fused_count["above_table_voxels"],
                          "fused_wrist_selected": fused_count["wrist_selected"],
                          "front_red_sampled": int(((front_color[:, 0] > 150) &
                                                    (front_color[:, 1] < 120) &
                                                    (front_color[:, 2] < 120)).sum()),
                          "fused_red_sampled": int(((fused_color[:, 0] > 150) &
                                                    (fused_color[:, 1] < 120) &
                                                    (fused_color[:, 2] < 120)).sum()),
                          "released_rgb_mae": float(np.abs(pre["front"] - demo["front"][frame]).mean()),
                          "plug_pos_error_mm": float(np.linalg.norm(
                              pre["plug_pos"] - demo["plug_pos"][frame]) * 1000),
                          "action_correction_l2": float(np.linalg.norm(
                              action - demo["action"][frame]))})

    def save(self, output, demo_path):
        output = Path(output)
        if not self.rows:
            raise ValueError("Empty v2 capture")
        if (output / "v2_capture.npz").exists():
            raise FileExistsError(output / "v2_capture.npz")
        np.savez_compressed(output / "v2_capture.npz",
                            **{key: np.stack(value) for key, value in self.data.items()})
        report = {"status": "captured_for_v2_audit_not_yet_training_dataset",
                  "source_demo": str(demo_path),
                  "source_demo_sha256": hashlib.sha256(Path(demo_path).read_bytes()).hexdigest(),
                  "source_episode": self.episode, "frames": len(self.rows),
                  "roi_base_xyz_m": ROI, "voxel_m": VOXEL_M,
                  "table_cutoff_m": TABLE_CUTOFF_M, "points": POINTS,
                  "cloud_modes": ["front_surface", "fused_surface"],
                  "force_frame": "right sensor actual taxel geometry + 3D source force in robot base; not calibrated N",
                  "actions": "actual pose-feedback actions paired with replay pre-action observations",
                  "arrays": {key: list(value[0].shape) for key, value in self.data.items()},
                  "rows": self.rows}
        (output / "v2_capture.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
