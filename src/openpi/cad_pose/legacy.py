"""Strict adapter for legacy geometry with verified ordered CAD correspondence."""

import json
from pathlib import Path
import time

import numpy as np

from openpi.cad_pose.geometry import fit_corresponding_rigid_pose
from openpi.cad_pose.preprocess import FrontRGBDPreprocessor
from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.object_geometry.production import load_mesh_surface_points
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.spatial_dataset.source import RawSpatialDataset
from scripts.preprocess.export_object_geometry_rgbtrack import sha256

ROLES = {"manipulated": "funnel", "target": "target_flask"}


def inspect_legacy(repo, data_root, profile):
    repo, root = Path(repo), Path(data_root) / "1018_peg_all"
    result = {"dataset": "1018_peg_all", "roles": ROLES, "status": "blocked", "errors": [], "checks": {}}
    try:
        path = root / "object_geometry/v1/manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("source", {}).get("mode") != "geometry_only_drop_hand_scene_extras_keep_raw_q2":
            raise ValueError("not the reviewed legacy migration contract")
        if manifest.get("source", {}).get("roles") != ROLES:
            raise ValueError("legacy object roles mismatch")
        expected = front_calibration_provenance(repo, profile)
        error = calibration_provenance_error(manifest.get("calibration"), expected)
        if error:
            raise ValueError(f"legacy calibration: {error}")
        assets = manifest.get("asset_provenance")
        if not isinstance(assets, dict):
            raise ValueError("legacy manifest lacks asset sampling provenance")
        config = json.loads((repo / "configs/object_geometry/datasets/1018_peg_all/config.json").read_text())
        cad = []
        for role in ("manipulated", "target"):
            name = ROLES[role]
            cfg, provenance = config["assets"][name], assets[name]
            mesh = repo / cfg["mesh"]
            digest = sha256(mesh)
            if (
                digest != cfg["sha256"]
                or provenance.get("source_sha256") != digest
                or provenance.get("sampler") != "deterministic_surface_points.v1"
                or int(provenance.get("seed", -1)) != cfg["sampling_seed"]
                or float(provenance.get("scale_m_per_unit", -1)) != cfg["scale_m_per_unit"]
            ):
                raise ValueError(f"{name}: missing or mismatched CAD ordering/scale provenance")
            points, _ = load_mesh_surface_points(
                mesh, scale_m_per_unit=cfg["scale_m_per_unit"], seed=cfg["sampling_seed"]
            )
            cad.append(points)
        reader = ObjectGeometryDerivedDataset(root, version="v1")
        episodes = RawSpatialDataset.discover_episode_ids(root)
        if tuple(episodes) != reader.episode_ids or not episodes:
            raise ValueError("legacy episode mapping mismatch")
        max_rmse, checked = 0.0, 0
        for episode in episodes:
            frames = list(reader.frame_ids(episode))
            if frames != list(range(len(frames))):
                raise ValueError("legacy frame IDs not contiguous")
            if list(map(int, RawSpatialDataset(root, episode=episode).states()["frame_index"])) != frames:
                raise ValueError(f"episode {episode}: RGB-D and geometry frame mismatch")
            for frame in sorted({0, len(frames) // 2, len(frames) - 1}):
                sample = reader.get(episode_index=episode, frame_index=frame)
                for i, role in enumerate(("manipulated", "target")):
                    mask = np.asarray(getattr(sample, f"{role}_mask"), bool)
                    if not mask.any():
                        continue
                    _, rmse = fit_corresponding_rigid_pose(
                        cad[i], getattr(sample, f"{role}_xyz_m"), mask, max_rmse_m=0.0001
                    )
                    max_rmse, checked = max(max_rmse, rmse), checked + 1
        if checked < 2:
            raise ValueError("no sufficient valid keyframe correspondences")
        result["checks"] = {
            "pose_hash_and_alignment": "passed",
            "adapter": "legacy_ordered_fit",
            "geometry_manifest_sha256": sha256(path),
            "profile_sha256": expected["profile"]["sha256"],
            "max_correspondence_rmse_m": max_rmse,
            "checked_roles": checked,
            "geometric_accuracy": "not verified; visual keyframe review required",
        }
        result["status"] = "requires_visual_review"
        result["errors"].append("ordered rigid fit succeeded but legacy RGB-D keyframes require independent review")
    except (OSError, KeyError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        result["errors"].append(f"legacy audit failed: {type(exc).__name__}: {exc}")
    return result


class LegacyGeometryPoseTask:
    def __init__(self, repo, data_root, profile, audit_entry):
        self.repo, self.root = Path(repo), Path(data_root) / "1018_peg_all"
        self.dataset, self.roles = "1018_peg_all", ROLES
        self.reader = ObjectGeometryDerivedDataset(self.root, version="v1")
        if audit_entry.get("checks", {}).get("adapter") != "legacy_ordered_fit" or audit_entry["checks"][
            "geometry_manifest_sha256"
        ] != sha256(self.reader.root / "manifest.json"):
            raise ValueError("legacy provenance audit missing or stale")
        config = json.loads((self.repo / "configs/object_geometry/datasets/1018_peg_all/config.json").read_text())
        self.ordered_cad = np.stack(
            [
                load_mesh_surface_points(
                    self.repo / config["assets"][ROLES[r]]["mesh"],
                    scale_m_per_unit=config["assets"][ROLES[r]]["scale_m_per_unit"],
                    seed=config["assets"][ROLES[r]]["sampling_seed"],
                )[0]
                for r in ("manipulated", "target")
            ]
        )
        self.episodes = tuple(RawSpatialDataset.discover_episode_ids(self.root))
        self.preprocess = FrontRGBDPreprocessor(self.repo, profile)

    def iter_episode(self, episode, *, include_rgb=False):
        if episode not in self.episodes:
            raise ValueError("unknown episode")
        raw = RawSpatialDataset(self.root, episode=episode)
        for row, video in zip(raw.iter_rows(depth_roles=("front",)), raw.iter_video_frames("front"), strict=True):
            frame = int(row["frame_index"])
            if video.frame_index != frame:
                raise ValueError("RGB/depth frame mismatch")
            geometry = self.reader.get(episode_index=episode, frame_index=frame)
            valid = np.array([np.asarray(getattr(geometry, f"{r}_mask")).any() for r in ("manipulated", "target")])
            if not valid.any():
                continue
            target = np.zeros((2, 4, 4), np.float32)
            for i, role in enumerate(("manipulated", "target")):
                if valid[i]:
                    target[i], _ = fit_corresponding_rigid_pose(
                        self.ordered_cad[i],
                        getattr(geometry, f"{role}_xyz_m"),
                        getattr(geometry, f"{role}_mask"),
                        max_rmse_m=0.0001,
                    )
            started = time.perf_counter()
            scene, mask = self.preprocess(video.rgb, row["observation.depths.cam_front"])
            sample = {
                "dataset": self.dataset,
                "episode": episode,
                "frame": frame,
                "timestamp": float(row["timestamp"]),
                "scene": scene,
                "scene_mask": mask,
                "cad": self.ordered_cad,
                "target_transform": target,
                "valid": valid,
                "preprocessing_seconds": time.perf_counter() - started,
            }
            if include_rgb:
                sample["rgb"] = video.rgb
            yield sample
