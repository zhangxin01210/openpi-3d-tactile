"""Online USB V3 observations using the reviewed bilateral LeRobot contract.

This module is imported by the old Isaac Gym Python 3.8 runtime. Keep its
sampling and coordinate conversion identical to the V3 capture/export path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from contactworld_base_cloud import POINTS, base_cloud
from contactworld_tactile_base import tactile_field_to_base

# The simulator uses a Python 3.8 Isaac Gym environment rather than the
# OpenPI virtualenv.  This pure-NumPy representation builder is shared with
# offline export instead of maintaining a second implementation.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from openpi.spatial_dataset.contactworld_contact_repr import build_frame as build_contact_repr


CONFIG_SUFFIXES = (
    "rgb", "fused_pc", "rgb_local_map", "fused_pc_local_map",
    "fused_pc_local_summary", "fused_pc_base_structured", "fused_pc_base_pooled",
    "fused_pc_base_map", "fused_pc_base_map_split", "fused_pc_base_map_matched",
    "fused_pc_base_structured_prefix", "fused_pc_base_structured_split_route",
    "fused_pc_local_map_3tok", "base_pooled_no_pc", "base_map_matched_no_pc",
    "contact_centered_point", "contact_centered_map", "base_pooled_relation",
    "base_map_matched_relation",
)
CONFIGS = tuple("pi0_cw3_usb_%02d_%s" % (index, suffix)
                for index, suffix in enumerate(CONFIG_SUFFIXES, 1))
LOCAL_FORCE_CONFIGS = {CONFIGS[index - 1] for index in (3, 4, 5, 13)}
BASE_FORCE_CONFIGS = {CONFIGS[index - 1] for index in (6, 7, 8, 9, 10, 11, 12)}
BASE_FORCE_CONFIGS |= {CONFIGS[index - 1] for index in (14, 15, 16, 17, 18, 19)}
CENTERED_CONTACT_CONFIGS = {CONFIGS[index - 1] for index in (16, 17)}
RELATION_CONFIGS = {CONFIGS[index - 1] for index in (18, 19)}
CLOUD_CONFIGS = set(CONFIGS) - {CONFIGS[0], CONFIGS[2], CONFIGS[13], CONFIGS[14]}


def obs_array(env, key):
    return env.obs_dict[key][0].detach().cpu().numpy().copy()


def _rgb(value):
    value = np.asarray(value)
    if value.dtype == np.uint8:
        return value.copy()
    return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)


def snapshot(env):
    """Capture all potential modalities on one pre-action frame.

    The fused cloud uses the capture's exact camera ordering, ROI, 5 mm voxel
    sampling and fixed robot-base identity. Force XYZ uses measured taxel/base
    geometry, as in ``contactworld_bilateral_capture.py``.
    """
    images = env.get_camera_image_tensors_dict()
    rgb = {role: _rgb(images[role][0].detach().cpu().numpy())
           for role in ("front", "wrist")}
    for role in ("front", "wrist"):
        observed = _rgb(obs_array(env, role))
        if rgb[role].shape != (256, 256, 3) or np.max(np.abs(
                rgb[role].astype(np.int16) - observed.astype(np.int16))) > 1:
            raise ValueError("Direct camera and policy RGB disagree: " + role)
    depth = {role: images[role + "_depth"][0].detach().cpu().numpy().astype(np.float32)
             for role in ("front", "wrist")}
    views, projections = {}, {}
    for role in ("front", "wrist"):
        handle = env.camera_handles_list[0][role + "_depth"]
        dtype = np.float64 if role == "front" else np.float32
        views[role] = np.asarray(env.gym.get_camera_view_matrix(
            env.sim, env.env_ptrs[0], handle), dtype=dtype)
        projections[role] = np.asarray(env.gym.get_camera_proj_matrix(
            env.sim, env.env_ptrs[0], handle), dtype=dtype)
    measured_base_pos = env.franka_base_pos[0].detach().cpu().numpy().astype(np.float32)
    measured_base_quat = env.franka_base_quat[0].detach().cpu().numpy().astype(np.float32)
    if (np.linalg.norm(measured_base_pos) > 1e-6 or
            np.linalg.norm(measured_base_quat - [0, 0, 0, 1]) > 1e-6):
        raise ValueError("V3 cloud assumes a fixed world-aligned robot base")
    cloud, _, source_camera, source_uv, counts = base_cloud(
        depth, rgb, views, projections,
        np.zeros(3, np.float32), np.array([0, 0, 0, 1], np.float32),
        cameras=("front", "wrist"), sampler="surface", points=POINTS)
    result = {
        "front": rgb["front"], "wrist": rgb["wrist"],
        "front_depth": depth["front"], "wrist_depth": depth["wrist"],
        "front_view": views["front"], "front_projection": projections["front"],
        "wrist_view": views["wrist"], "wrist_projection": projections["wrist"],
        "state": np.r_[obs_array(env, "dof_pos"), obs_array(env, "dof_vel")].astype(np.float32),
        "cloud_xyz": cloud, "cloud_camera": source_camera, "cloud_uv": source_uv,
        "cloud_counts": counts, "base_pose_world": np.r_[measured_base_pos, measured_base_quat],
    }
    xyz, forces, local = [], [], []
    for side in ("left", "right"):
        key = "tactile_force_field_" + side
        field = obs_array(env, key).astype(np.float32)
        position = env.tactile_pos_world_dict[key][0].detach().cpu().numpy().astype(np.float32)
        quaternion = env.tactile_quat_world_dict[key][0].detach().cpu().numpy().astype(np.float32)
        coordinates, vectors = tactile_field_to_base(
            field, position, quaternion, measured_base_pos, measured_base_quat)
        if field.shape != (10, 14, 3) or coordinates.shape != (140, 3):
            raise ValueError("Invalid bilateral TacFF shape")
        xyz.append(coordinates)
        forces.append(vectors)
        local.append(field)
    result["taxel_xyz_base"] = np.stack(xyz)
    result["force_base"] = np.stack(forces)
    result["force_local"] = np.stack(local)
    if result["state"].shape != (18,) or not all(np.isfinite(result[key]).all()
            for key in ("state", "cloud_xyz", "taxel_xyz_base", "force_base", "force_local")):
        raise ValueError("Nonfinite or incorrectly shaped V3 policy input")
    return result


def policy_input(snap, config):
    if config not in CONFIGS:
        raise ValueError(config)
    request = {
        "observation/front_rgb": snap["front"],
        "observation/wrist_rgb": snap["wrist"],
        "observation/state": snap["state"],
    }
    visual = tactile = None
    if config in CLOUD_CONFIGS:
        visual = {"xyz_m": snap["cloud_xyz"],
                  "rgb": np.zeros((POINTS, 3), np.uint8),
                  "rgb_valid": np.zeros(POINTS, bool),
                  "point_mask": np.ones(POINTS, bool)}
    if config in LOCAL_FORCE_CONFIGS or config in BASE_FORCE_CONFIGS:
        force = (snap["force_local"].reshape(280, 3) if config in LOCAL_FORCE_CONFIGS
                 else snap["force_base"].reshape(280, 3))
        tactile = {
            "xyz_m": snap["taxel_xyz_base"].reshape(280, 3),
            "force": force,
            "force_norm": np.linalg.norm(force, axis=-1),
            "finger_id": np.repeat(np.array([0, 1], np.int32), 140),
            "taxel_id": np.tile(np.arange(140, dtype=np.int32), 2),
            "point_mask": np.ones(280, bool),
        }
    if visual is not None or tactile is not None:
        spatial = {"visual": visual, "tactile": tactile}
        if config in CENTERED_CONTACT_CONFIGS or config in RELATION_CONFIGS:
            derived = build_contact_repr(snap["cloud_xyz"], snap["taxel_xyz_base"], snap["force_base"])
            if config in CENTERED_CONTACT_CONFIGS:
                spatial["contact_centered_tactile"] = {
                    "patch_xyz_local": derived["patch_xyz_local"],
                    "patch_force": derived["patch_force"],
                    "patch_mask": derived["patch_mask"],
                    "anchor_xyz_m": derived["anchor_xyz_base"],
                    "anchor_stats": derived["anchor_stats"],
                    "anchor_mask": derived["anchor_mask"],
                    "finger_summary": derived["finger_summary"],
                    "finger_center_xyz_m": derived["finger_center_base"],
                    "finger_valid": derived["finger_valid"],
                }
            if config in RELATION_CONFIGS:
                spatial["tactile_scene_relation"] = {
                    "query_xyz_m": derived["query_xyz_base"],
                    "query_features": derived["query_features"],
                    "query_valid": derived["query_valid"],
                    "candidate_xyz_m": derived["candidate_xyz_base"],
                    "candidate_mask": derived["candidate_mask"],
                }
        request["spatial"] = spatial
    return request


def tactile_summary(snap):
    """Diagnostics only; never substitutes for a model input."""
    out = {}
    for index, side in enumerate(("left", "right")):
        local = snap["force_local"][index].reshape(140, 3)
        base = snap["force_base"][index]
        xyz = snap["taxel_xyz_base"][index]
        strength = np.linalg.norm(base, axis=-1)
        active = strength > 1e-4
        weight = strength.sum()
        out[side] = {
            "local_signed_mean": local.mean(axis=0).tolist(),
            "local_abs_mean": np.abs(local).mean(axis=0).tolist(),
            "base_signed_mean": base.mean(axis=0).tolist(),
            "base_abs_mean": np.abs(base).mean(axis=0).tolist(),
            "active_fraction": float(active.mean()),
            "max_norm": float(strength.max()),
            "sum_norm": float(weight),
            "contact_centroid_base_m": ((xyz * strength[:, None]).sum(axis=0) /
                                        weight).tolist() if weight > 0 else None,
            "pad_centroid_base_m": xyz.mean(axis=0).tolist(),
        }
    return out
