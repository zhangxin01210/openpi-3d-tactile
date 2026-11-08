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


def legacy_calibration_check(actual, expected, *, allow_legacy_intrinsics=False):
    """Allow only the explicitly acknowledged historical RGB intrinsic difference."""
    err = calibration_provenance_error(actual, expected)
    if not err:
        return {"mode": "current_profile", "source": actual, "difference": None}
    if not allow_legacy_intrinsics or not isinstance(actual, dict):
        raise ValueError(f"legacy calibration: {err}")
    adjusted = dict(actual)
    if actual.get("applied_front_color_intrinsics") == expected.get("applied_front_color_intrinsics"):
        raise ValueError(f"legacy calibration: {err}")
    adjusted["applied_front_color_intrinsics"] = expected.get("applied_front_color_intrinsics")
    if not isinstance(actual.get("profile"), dict) or not isinstance(expected.get("profile"), dict):
        raise ValueError("legacy calibration: missing historical/current profile provenance")
    adjusted["profile"] = {**actual["profile"], "sha256": expected["profile"].get("sha256")}
    remaining = calibration_provenance_error(adjusted, expected)
    if remaining:
        raise ValueError(f"legacy calibration differs beyond approved intrinsic exception: {remaining}")
    return {
        "mode": "legacy_intrinsics_explicit",
        "source": actual,
        "current_profile": expected,
        "difference": "historical color intrinsics and profile SHA differ; visual review required",
    }


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
        if audit_entry.get("checks", {}).get("adapter") != "legacy_ordered_fit" or (
            audit_entry.get("mode") != "quick"
            and audit_entry["checks"]["geometry_manifest_sha256"] != sha256(self.reader.root / "manifest.json")
        ):
            raise ValueError("legacy provenance audit missing or stale")
        if audit_entry.get("checks", {}).get("legacy_source"):
            original = audit_entry["checks"]["legacy_source"]
            if sha256(Path(original["path"])) != original["recorded_sha256"]:
                raise ValueError("Peg original geometry manifest SHA differs from migration source")
        config = json.loads((self.repo / "configs/object_geometry/datasets/1018_peg_all/config.json").read_text())
        cad = []
        for role in ("manipulated", "target"):
            asset = config["assets"][ROLES[role]]
            points, metadata = load_mesh_surface_points(
                self.repo / asset["mesh"], scale_m_per_unit=asset["scale_m_per_unit"], seed=asset["sampling_seed"]
            )
            if metadata["source_sha256"] != asset["sha256"]:
                raise ValueError(f"Peg {role} mesh SHA differs from recorded sampling provenance")
            cad.append(points)
        self.ordered_cad = np.stack(cad)
        self.episodes = tuple(RawSpatialDataset.discover_episode_ids(self.root))
        if not self.episodes or self.episodes != self.reader.episode_ids:
            raise ValueError("Peg RGB-D and geometry episode IDs differ")
        self.preprocess = FrontRGBDPreprocessor(self.repo, profile)

    def iter_episode(self, episode, *, include_rgb=False):
        if episode not in self.episodes:
            raise ValueError("unknown episode")
        raw = RawSpatialDataset(self.root, episode=episode)
        count = 0
        for row, video in zip(raw.iter_rows(depth_roles=("front",)), raw.iter_video_frames("front"), strict=True):
            frame = int(row["frame_index"])
            if video.frame_index != frame or frame != count:
                raise ValueError("RGB/depth frame mismatch")
            count += 1
            geometry = self.reader.get(episode_index=episode, frame_index=frame)
            valid = np.array([np.asarray(getattr(geometry, f"{r}_mask")).any() for r in ("manipulated", "target")])
            if not valid.any():
                continue
            target = np.zeros((2, 4, 4), np.float32)
            for i, role in enumerate(("manipulated", "target")):
                if valid[i]:
                    if not np.asarray(getattr(geometry, f"{role}_mask"), dtype=bool).all():
                        raise ValueError(f"partial fixed-point mask episode={episode} frame={frame} role={role}")
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

        if count != len(self.reader.frame_ids(episode)):
            raise ValueError(f"RGB-D and geometry frame count differ in episode {episode}")


class ButtonGeometryPoseTask(LegacyGeometryPoseTask):
    """Fit poses to cleaned Geometry v1; never index stale pre-cleanup PoseRecord."""

    def __init__(self, repo, data_root, profile, audit_entry):
        from scripts.cad_pose.preflight import ROLES as ALL_ROLES
        from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
        from scripts.preprocess.export_object_geometry_rgbtrack import load_surface_points

        if audit_entry.get("checks", {}).get("adapter") != "button_ordered_fit":
            raise ValueError("button ordered correspondence audit required")
        self.repo = Path(repo)
        self.dataset = "1028_press_button"
        self.root = Path(data_root) / self.dataset
        self.roles = ALL_ROLES[self.dataset]
        self.reader = ObjectGeometryDerivedDataset(self.root, version="v1")
        prep = self.repo / "output/preprocess" / self.dataset
        release = json.loads((prep / "rgbtrack/release_v1.json").read_text())
        approval = json.loads((prep / "geometry/approved/manifest.json").read_text())
        if release.get("dataset") != self.dataset or self.reader.manifest.get("role_to_object") != self.roles:
            raise ValueError("button role provenance changed since audit")
        if approval.get("status") != "human_accepted" or set(release["objects"]) != set(self.roles.values()):
            raise ValueError("button CAD approval or role mismatch")
        cad = {}
        for i, name in enumerate(release["objects"]):
            asset = approval["assets"][name]
            if (
                not asset.get("sha256")
                or asset.get("sha256") != self.reader.manifest.get("approved_meshes_sha256", {}).get(name)
                or float(asset.get("scale_m_per_unit", -1)) != 1
            ):
                raise ValueError("button CAD provenance mismatch")
            cad[name] = load_surface_points(prep / "geometry/approved" / f"{name}.glb", SURFACE_SAMPLING_SEED + i)
        self.ordered_cad = np.stack([cad[self.roles[role]] for role in ("manipulated", "target")])
        self.episodes = tuple(RawSpatialDataset.discover_episode_ids(self.root))
        if self.episodes != self.reader.episode_ids:
            raise ValueError("button cleaned RGB-D and geometry episode IDs differ")
        self.preprocess = FrontRGBDPreprocessor(self.repo, profile)
