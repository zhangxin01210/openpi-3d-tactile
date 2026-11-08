"""Streaming current-frame RGB-D, explicitly ordered CAD, and pose supervision.

Only published, audited RGBTrack releases are accepted here. Legacy Peg and
any task without the same proven contract require a separate audited adapter;
this reader never invokes a tracker, ICP, or accesses GT geometry as input.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from openpi.cad_pose.preprocess import FrontRGBDPreprocessor
from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.spatial_dataset.source import RawSpatialDataset
from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
from scripts.preprocess.export_object_geometry_rgbtrack import cloud_from_pose
from scripts.preprocess.export_object_geometry_rgbtrack import load_surface_points
from scripts.preprocess.export_object_geometry_rgbtrack import sha256
from scripts.preprocess.export_object_geometry_rgbtrack import source_rows


class PublishedPoseTask:
    """Streaming reader for one audited task; frames remain inside their episode."""

    def __init__(self, repo: Path, data_root: Path, dataset: str, *, profile: str, audit_entry: dict):
        if (
            audit_entry.get("dataset") != dataset
            or audit_entry.get("checks", {}).get("pose_hash_and_alignment") != "passed"
            or audit_entry.get("status") not in {"requires_visual_review", "ready"}
        ):
            raise ValueError("task requires a passing current pose/CAD provenance audit")
        self.repo = Path(repo).resolve()
        self.root = Path(data_root).resolve() / dataset
        self.dataset = dataset
        self.prep = self.repo / "output/preprocess" / dataset
        release_path = self.prep / "rgbtrack/release_v1.json"
        manifest_path = self.root / "object_geometry/v1/manifest.json"
        approval_path = self.prep / "geometry/approved/manifest.json"
        self.release = json.loads(release_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        checks = audit_entry["checks"]
        self.resolved_pose_hashes = checks.get("resolved_pose_hashes", {})
        if checks.get("geometry_manifest_sha256") != sha256(manifest_path) or checks.get("release_sha256") != sha256(
            release_path
        ):
            raise ValueError("pose audit does not match current geometry manifest or release")
        if (
            self.release.get("dataset") != dataset
            or self.manifest.get("dataset") != dataset
            or self.manifest.get("source_release_sha256") != sha256(release_path)
            or self.manifest.get("approved_geometry_sha256") != sha256(approval_path)
            or approval.get("status") != "human_accepted"
        ):
            raise ValueError("release, approved CAD and geometry provenance differ")
        self.roles = self.manifest["role_to_object"]
        names = self.release["objects"]
        if set(self.roles) != {"manipulated", "target"} or set(names) != set(self.roles.values()) or len(names) != 2:
            raise ValueError("two explicit manipulated/target roles are required")
        if self.manifest.get("surface_sampling_seed") != SURFACE_SAMPLING_SEED:
            raise ValueError("surface sampling seed mismatch")
        self.cad = {}
        for i, name in enumerate(names):
            mesh = self.prep / "geometry/approved" / f"{name}.glb"
            entry = approval.get("assets", {}).get(name, {})
            digest = sha256(mesh)
            if (
                digest != entry.get("sha256")
                or digest != self.manifest["approved_meshes_sha256"].get(name)
                or float(entry.get("scale_m_per_unit", -1)) != 1
            ):
                raise ValueError(f"{name}: CAD hash/scale mismatch")
            self.cad[name] = load_surface_points(mesh, SURFACE_SAMPLING_SEED + i)
        self.ordered_cad = np.stack([self.cad[self.roles[role]] for role in ("manipulated", "target")])
        self.episodes = tuple(RawSpatialDataset.discover_episode_ids(self.root))
        if not self.episodes or set(self.episodes) != set(map(int, self.release["episodes"])):
            raise ValueError("release episodes differ from the raw dataset")
        self.calibration = front_calibration_provenance(self.repo, profile)
        if checks.get("profile_sha256") != self.calibration["profile"]["sha256"]:
            raise ValueError("pose audit does not match current calibration profile")
        self.preprocess = FrontRGBDPreprocessor(self.repo, profile)

    def iter_episode(self, episode: int, *, include_rgb: bool = False):
        if episode not in self.episodes:
            raise ValueError("episode is not in the audited task")
        root = self.prep / "rgbtrack/episodes" / f"e{episode:05d}"
        run = self.prep / "rgbtrack/runs" / self.release["run_tag"] / "episodes" / f"e{episode:05d}"
        if not root.is_symlink() or root.resolve() != run.resolve():
            raise ValueError("published pose episode symlink differs from release")
        run_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        cal_error = calibration_provenance_error(run_manifest.get("calibration"), self.calibration)
        if cal_error:
            raise ValueError(f"pose run calibration mismatch: {cal_error}")
        rows = []
        for role in ("manipulated", "target"):
            name = self.roles[role]
            path = root / f"{name}_poses.jsonl"
            if sha256(path) != self.resolved_pose_hashes.get(str(episode), {}).get(
                name, self.manifest.get("pose_files_sha256_by_episode", {}).get(str(episode), {}).get(name)
            ):
                raise ValueError(f"pose file changed after audit: {path}")
            rows.append(source_rows(path, object_name=name))
        if len(rows[0]) != len(rows[1]):
            raise ValueError("two pose streams have different frame counts")
        raw = RawSpatialDataset(self.root, episode=episode)
        count = 0
        for row, video in zip(raw.iter_rows(depth_roles=("front",)), raw.iter_video_frames("front"), strict=True):
            frame = int(row["frame_index"])
            if frame != count or video.frame_index != frame or frame >= len(rows[0]):
                raise ValueError(f"RGB/depth/pose frame mismatch in episode {episode}, frame {frame}")
            count += 1
            # Invalid frames are never used as zero-pose supervision.
            valid = np.array([rows[i][frame]["valid"] for i in range(2)], dtype=bool)
            if not valid.any():
                continue
            target = np.zeros((2, 4, 4), dtype=np.float32)
            for i, is_valid in enumerate(valid):
                if is_valid:
                    entry = rows[i][frame]
                    cloud_from_pose(entry, self.ordered_cad[i])  # validate finite SE(3)
                    target[i] = np.asarray(entry["T_base_link_object"], dtype=np.float32)
            started = time.perf_counter()
            scene, scene_mask = self.preprocess(video.rgb, row["observation.depths.cam_front"])
            preprocessing_seconds = time.perf_counter() - started
            sample = {
                "scene": scene,
                "scene_mask": scene_mask,
                "cad": self.ordered_cad,
                "target_transform": target,
                "valid": valid,
                "episode": episode,
                "frame": frame,
                "timestamp": float(row["timestamp"]),
                "dataset": self.dataset,
                "preprocessing_seconds": preprocessing_seconds,
            }
            if include_rgb:
                sample["rgb"] = video.rgb
            yield sample
        if count != len(rows[0]):
            raise ValueError(f"episode {episode}: raw frame count differs from pose files")
