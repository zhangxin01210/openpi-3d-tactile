"""Capture calibrated front RGB-D and a fixed-workspace cloud during demo replay.

This is a pilot data product. The crop is fixed in the robot base frame and does
not read plug/socket ground truth; it must be reviewed before training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from contactworld_tactile_base import tactile_field_to_base


GEOMETRY = Path(__file__).resolve().parents[1] / "reports/2026-10-01/contactworld/runtime_geometry.json"
CROP = ((0.25, 0.62), (-0.20, 0.20), (0.01, 0.30))
POINTS = 1024


def camera_xyz(depth: np.ndarray, view: np.ndarray, projection: np.ndarray,
               correction: np.ndarray, translation: np.ndarray):
    height, width = depth.shape
    vv, uu = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 2.0)
    z = depth[valid]
    cam = np.stack((
        ((uu[valid] - width / 2) / width) * z * (2 / projection[0, 0]),
        ((vv[valid] - height / 2) / height) * z * (2 / projection[1, 1]),
        z, np.ones_like(z)), axis=-1)
    # Match the released source conversion, then correct its axis convention.
    raw = (cam @ np.linalg.inv(view).T)[:, :3]
    xyz = raw @ correction.T + translation
    pixels = np.stack((uu[valid], vv[valid]), axis=-1)
    return xyz.astype(np.float32), pixels.astype(np.int16)


def camera_xyz_from_matrices(depth: np.ndarray, view: np.ndarray, projection: np.ndarray):
    """Backproject any Isaac Gym camera with its current view/projection matrices."""
    height, width = depth.shape
    vv, uu = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 2.0)
    z = depth[valid]
    focal_x = projection[0, 0] * width / 2
    focal_y = projection[1, 1] * height / 2
    optical = np.stack((((uu[valid] + .5 - width / 2) / focal_x) * z,
                        ((vv[valid] + .5 - height / 2) / focal_y) * z, z), axis=-1)
    inverse = np.linalg.inv(view)
    camera_rotation = inverse[:3, :3].T
    camera_position = inverse[3, :3]
    xyz = optical @ np.diag([1., -1., -1.]) @ camera_rotation.T + camera_position
    pixels = np.stack((uu[valid], vv[valid]), axis=-1)
    return xyz.astype(np.float32), pixels.astype(np.int16)


def color_counts(rgb: np.ndarray) -> dict:
    rgb = rgb.astype(np.int16)
    return {
        "red_pixel_points": int(((rgb[:, 0] > 150) & (rgb[:, 1] < 120) & (rgb[:, 2] < 120)).sum()),
        "blue_pixel_points": int(((rgb[:, 2] > rgb[:, 0] + 20) &
                                  (rgb[:, 2] > rgb[:, 1] + 20)).sum()),
        "white_pixel_points": int((rgb.min(1) > 240).sum()),
    }


class DenseCapture:
    def __init__(self, episode: int):
        geometry = json.loads(GEOMETRY.read_text())
        if geometry["status"] != "passed":
            raise ValueError("Front camera geometry audit must pass")
        self.rotation = np.asarray(geometry["A"], dtype=np.float32)
        self.translation = np.asarray(geometry["t"], dtype=np.float32)
        self.reference = np.load(geometry["runtime_capture"])
        self.episode = episode
        self.rows = []
        self.data = {name: [] for name in (
            "front", "wrist", "front_depth", "cloud_xyz", "cloud_rgb", "cloud_uv",
            "source_cloud_xyz", "source_cloud_rgb", "source_cloud_uv",
            "wrist_depth", "wrist_view", "wrist_projection", "cloud_fused_xyz",
            "cloud_fused_rgb", "cloud_fused_uv", "cloud_fused_camera_id",
            "state", "force_grid", "ee_pose", "plug_pose", "socket_pose", "action_executed",
            "action_sim_clipped", "action_recorded", "tactile_xyz_base",
            "tactile_force_base", "tactile_quat_world", "base_pose_world",
            "tactile_rgb", "tactile_depth")}
        self.view = self.projection = None

    def add(self, env, pre: dict, demo: dict, action: np.ndarray, frame: int):
        images = env.get_camera_image_tensors_dict()
        rgb_native = images["front"][0].detach().cpu().numpy()
        depth = images["front_depth"][0].detach().cpu().numpy().astype(np.float32)
        wrist_native = images["wrist"][0].detach().cpu().numpy()
        wrist_depth = images["wrist_depth"][0].detach().cpu().numpy().astype(np.float32)
        if rgb_native.dtype == np.uint8:
            rgb_native = rgb_native.astype(np.float32) / 255.
        if depth.shape != (256, 256) or rgb_native.shape != (256, 256, 3):
            raise ValueError("Unexpected front camera dimensions")
        handle = env.camera_handles_list[0]["front_depth"]
        view = np.asarray(env.gym.get_camera_view_matrix(env.sim, env.env_ptrs[0], handle))
        projection = np.asarray(env.gym.get_camera_proj_matrix(env.sim, env.env_ptrs[0], handle))
        if not (np.allclose(view, self.reference["view"], atol=1e-5) and
                np.allclose(projection, self.reference["projection"], atol=1e-5)):
            raise ValueError("Front camera matrices differ from geometry audit")
        if self.view is None:
            self.view, self.projection = view, projection
        front_error = float(np.abs(rgb_native - pre["front"]).mean())
        if front_error > 1 / 255:
            raise ValueError(f"Direct camera RGB differs from action-pre observation: {front_error}")
        if wrist_native.dtype == np.uint8:
            wrist_native = wrist_native.astype(np.float32) / 255.
        wrist_error = float(np.abs(wrist_native - pre["wrist"]).mean())
        if wrist_error > 1 / 255:
            raise ValueError(f"Direct wrist RGB differs from action-pre observation: {wrist_error}")
        if not (np.linalg.norm(env.franka_base_pos[0].cpu().numpy()) < 1e-6 and
                np.linalg.norm(env.franka_base_quat[0].cpu().numpy() - [0, 0, 0, 1]) < 1e-6):
            raise ValueError("Geometry correction requires world-aligned robot base")
        xyz, pixels = camera_xyz(depth, view, projection, self.rotation, self.translation)
        direct_front_xyz, _ = camera_xyz_from_matrices(depth, view, projection)
        if np.max(np.abs(xyz - direct_front_xyz)) > 1e-6:
            raise ValueError("Front matrix backprojection differs from audited correction")
        wrist_handle = env.camera_handles_list[0]["wrist_depth"]
        wrist_view = np.asarray(env.gym.get_camera_view_matrix(env.sim, env.env_ptrs[0], wrist_handle))
        wrist_projection = np.asarray(env.gym.get_camera_proj_matrix(env.sim, env.env_ptrs[0], wrist_handle))
        wrist_xyz, wrist_pixels = camera_xyz_from_matrices(wrist_depth, wrist_view, wrist_projection)
        mask = np.ones(len(xyz), bool)
        for axis, (lo, hi) in enumerate(CROP):
            mask &= (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
        candidates = np.flatnonzero(mask)
        if len(candidates) < POINTS:
            raise ValueError(f"Only {len(candidates)} points in fixed crop at frame {frame}")
        rng = np.random.default_rng(20261002 + self.episode * 10000 + frame)
        chosen = rng.choice(candidates, POINTS, replace=False)
        selected_xyz, selected_uv = xyz[chosen], pixels[chosen]
        wrist_mask = np.ones(len(wrist_xyz), bool)
        for axis, (lo, hi) in enumerate(CROP):
            wrist_mask &= (wrist_xyz[:, axis] >= lo) & (wrist_xyz[:, axis] <= hi)
        wrist_candidates = np.flatnonzero(wrist_mask)
        if len(wrist_candidates) < POINTS // 2:
            raise ValueError(f"Only {len(wrist_candidates)} wrist points in fixed crop at frame {frame}")
        wrist_chosen = rng.choice(wrist_candidates, POINTS // 2, replace=False)
        fused_xyz = np.concatenate((selected_xyz[:POINTS // 2], wrist_xyz[wrist_chosen]))
        fused_uv = np.concatenate((selected_uv[:POINTS // 2], wrist_pixels[wrist_chosen]))
        fused_camera_id = np.r_[np.zeros(POINTS // 2, np.uint8),
                                np.ones(POINTS // 2, np.uint8)]
        shuffle = rng.permutation(POINTS)
        fused_xyz, fused_uv, fused_camera_id = (
            fused_xyz[shuffle], fused_uv[shuffle], fused_camera_id[shuffle])
        front_u8 = np.rint(np.clip(pre["front"], 0, 1) * 255).astype(np.uint8)
        wrist_u8 = np.rint(np.clip(pre["wrist"], 0, 1) * 255).astype(np.uint8)
        cloud_rgb = front_u8[selected_uv[:, 1], selected_uv[:, 0]]
        fused_rgb = np.empty((POINTS, 3), np.uint8)
        front_id = fused_camera_id == 0
        fused_rgb[front_id] = front_u8[fused_uv[front_id, 1], fused_uv[front_id, 0]]
        fused_rgb[~front_id] = wrist_u8[fused_uv[~front_id, 1], fused_uv[~front_id, 0]]
        old_raw = pre["pointcloud"]
        old_xyz = old_raw[:, :3] @ self.rotation.T + self.translation
        old_rgb = np.rint(np.clip(old_raw[:, 3:6], 0, 1) * 255).astype(np.uint8)
        inverse = np.linalg.inv(view)
        optical = (old_xyz - inverse[3, :3]) @ inverse[:3, :3].T @ np.diag([1., -1., -1.])
        focal = np.array([projection[0, 0] * 128, projection[1, 1] * 128])
        old_uv_float = optical[:, :2] / optical[:, 2:3] * focal + [128, 128] - 0.5
        old_uv = np.rint(old_uv_float).astype(np.int16)
        if not ((old_uv >= 0).all() and (old_uv < 256).all() and
                np.abs(front_u8[old_uv[:, 1], old_uv[:, 0]].astype(np.int16) -
                       old_rgb.astype(np.int16)).max() <= 1):
            raise ValueError("Original sparse cloud fails same-frame RGB projection")
        # The source 1024 points and the dense reconstruction should be samples
        # from the same camera frame. Check a few frames without slowing all rows.
        old_nearest_max = None
        if frame in (0, len(demo["action"]) // 2, len(demo["action"]) - 1):
            distance, _ = cKDTree(xyz).query(old_xyz)
            old_nearest_max = float(distance.max())
            if old_nearest_max > 2e-4:
                raise ValueError(f"Source cloud and dense depth disagree: {old_nearest_max} m")
        self.data["front"].append(front_u8)
        self.data["wrist"].append(wrist_u8)
        self.data["front_depth"].append(depth)
        self.data["wrist_depth"].append(wrist_depth)
        self.data["wrist_view"].append(wrist_view)
        self.data["wrist_projection"].append(wrist_projection)
        self.data["cloud_xyz"].append(selected_xyz)
        self.data["cloud_rgb"].append(cloud_rgb)
        self.data["cloud_uv"].append(selected_uv)
        self.data["cloud_fused_xyz"].append(fused_xyz)
        self.data["cloud_fused_rgb"].append(fused_rgb)
        self.data["cloud_fused_uv"].append(fused_uv)
        self.data["cloud_fused_camera_id"].append(fused_camera_id)
        self.data["source_cloud_xyz"].append(old_xyz.astype(np.float32))
        self.data["source_cloud_rgb"].append(old_rgb)
        self.data["source_cloud_uv"].append(old_uv)
        self.data["state"].append(np.r_[pre["dof_pos"], pre["dof_vel"]].astype(np.float32))
        self.data["force_grid"].append(pre["tactile_force_field_right"].astype(np.float32))
        if not (hasattr(env, "tactile_pos_world") and hasattr(env, "tactile_quat_world")):
            raise ValueError("Live TacFF lacks same-frame taxel world poses")
        tactile_xyz, tactile_force = tactile_field_to_base(
            pre["tactile_force_field_right"],
            env.tactile_pos_world[0].detach().cpu().numpy(),
            env.tactile_quat_world[0].detach().cpu().numpy(),
            env.franka_base_pos[0].detach().cpu().numpy(),
            env.franka_base_quat[0].detach().cpu().numpy())
        self.data["tactile_xyz_base"].append(tactile_xyz)
        self.data["tactile_force_base"].append(tactile_force)
        self.data["tactile_quat_world"].append(
            env.tactile_quat_world[0].detach().cpu().numpy().astype(np.float32).copy())
        self.data["base_pose_world"].append(np.r_[
            env.franka_base_pos[0].detach().cpu().numpy(),
            env.franka_base_quat[0].detach().cpu().numpy()].astype(np.float32))
        self.data["tactile_rgb"].append(np.rint(np.clip(
            pre["tactile_rgb_right"], 0, 1) * 255).astype(np.uint8))
        self.data["tactile_depth"].append(pre["tactile_depth_right"].astype(np.float16))
        self.data["ee_pose"].append(np.r_[pre["ee_pos"], pre["ee_quat"]].astype(np.float32))
        self.data["plug_pose"].append(np.r_[pre["plug_pos"], pre["plug_quat"]].astype(np.float32))
        self.data["socket_pose"].append(np.r_[pre["socket_pos_gt"], pre["socket_quat"]].astype(np.float32))
        self.data["action_executed"].append(action.astype(np.float32).copy())
        self.data["action_sim_clipped"].append(np.clip(action, -env.clip_actions,
                                                        env.clip_actions).astype(np.float32))
        self.data["action_recorded"].append(demo["action"][frame].astype(np.float32).copy())
        self.rows.append({"frame": frame, "valid_dense_points": len(xyz),
                          "fixed_crop_candidates": len(candidates),
                          "wrist_crop_candidates": len(wrist_candidates),
                          "action_pre_rgb_direct_mae": front_error,
                          "action_pre_wrist_direct_mae": wrist_error,
                          "released_rgb_mae": float(np.abs(pre["front"] - demo["front"][frame]).mean()),
                          "old_sparse_nearest_dense_max_m": old_nearest_max,
                          "plug_pos_error_mm": float(np.linalg.norm(pre["plug_pos"] -
                                                                   demo["plug_pos"][frame]) * 1000),
                          "action_correction_l2": float(np.linalg.norm(action - demo["action"][frame])),
                          "source_cloud_color_counts": color_counts(old_rgb),
                          "fused_cloud_color_counts": color_counts(fused_rgb),
                          **color_counts(cloud_rgb)})

    def save(self, output: Path, demo_path: Path):
        archive = output / "dense_capture.npz"
        np.savez_compressed(archive, **{key: np.stack(value) for key, value in self.data.items()},
                            camera_view=self.view, camera_projection=self.projection)
        report = {"status": "pilot_for_visual_review_not_approved_training_data",
                  "source_demo": str(demo_path),
                  "source_demo_sha256": hashlib.sha256(demo_path.read_bytes()).hexdigest(),
                  "geometry_report": str(GEOMETRY), "crop_base_m": CROP,
                  "selection": "front-only: uniform 1024 from fixed crop; fused: 512 front plus 512 wrist, shuffled; deterministic per episode/frame",
                  "action": "executed feedback action paired with pre-action observations",
                  "tactile_geometry": "same-frame simulated taxel poses and TacFF projected to robot base; source force units uncalibrated",
                  "frames": len(self.rows), "arrays": {key: list(value[0].shape)
                                                         for key, value in self.data.items()},
                  "rows": self.rows}
        (output / "dense_capture.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
