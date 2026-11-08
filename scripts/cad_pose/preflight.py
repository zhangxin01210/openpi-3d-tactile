#!/usr/bin/env python3
"""Read-only four-task RGB-D / CAD / pose provenance audit; never runs tracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.spatial_dataset.source import RawSpatialDataset
from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
from scripts.preprocess.export_object_geometry_rgbtrack import cloud_from_pose
from scripts.preprocess.export_object_geometry_rgbtrack import load_surface_points
from scripts.preprocess.export_object_geometry_rgbtrack import sha256
from scripts.preprocess.export_object_geometry_rgbtrack import source_rows

ROLES = {
    "1018_peg_all": {"manipulated": "funnel", "target": "target_flask"},
    "1028_press_button": {"manipulated": "button", "target": "box"},
    "1030_yellow_socket": {"manipulated": "red", "target": "yellow"},
    "1102_eraser": {"manipulated": "eraser", "target": "whiteboard"},
}


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def audit_task(repo: Path, data_root: Path, dataset: str, profile: str) -> dict:
    """Fail closed if any task lacks provable pose/CAD/calibration/frame lineage."""
    root = data_root / dataset
    prep = repo / "output" / "preprocess" / dataset
    result = {"dataset": dataset, "roles": ROLES[dataset], "status": "blocked", "errors": [], "checks": {}}
    errors, checks = result["errors"], result["checks"]
    info, manifest, release, approval = (root / "meta/info.json", root / "object_geometry/v1/manifest.json",
        prep / "rgbtrack/release_v1.json", prep / "geometry/approved/manifest.json")
    for label, path in (("raw dataset metadata", info), ("object geometry manifest", manifest),
                        ("published pose release", release), ("approved CAD manifest", approval)):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    if manifest.is_file() and release.is_file() and approval.is_file():
        try:
            geometry = read_json(manifest)
            rel = read_json(release)
            approved = read_json(approval)
            names = rel["objects"]
            if (geometry.get("role_to_object") != ROLES[dataset] or len(names) != 2
                    or set(names) != set(ROLES[dataset].values())
                    or rel.get("dataset") != dataset or approved.get("status") != "human_accepted"
                    or geometry.get("source_release_sha256") != sha256(release)
                    or geometry.get("approved_geometry_sha256") != sha256(approval)
                    or geometry.get("surface_sampling_seed") != SURFACE_SAMPLING_SEED):
                raise ValueError("role, release, sampling or approval provenance mismatch")
            for name in names:
                mesh = prep / "geometry/approved" / f"{name}.glb"
                entry = approved.get("assets", {}).get(name, {})
                digest = sha256(mesh)
                if (digest != entry.get("sha256") or digest != geometry.get("approved_meshes_sha256", {}).get(name)
                        or float(entry.get("scale_m_per_unit", -1)) != 1):
                    raise ValueError(f"{name}: CAD hash or metric scale mismatch")
            checks["cad_hashes_and_release_order"] = {"status": "passed", "release_order": names,
                                                       "sampling_seed": SURFACE_SAMPLING_SEED}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"CAD/release audit failed: {type(exc).__name__}: {exc}")
    if release.is_file():
        try:
            rel = read_json(release)
            episode = rel["episodes"][0]
            for role, name in ROLES[dataset].items():
                pose = prep / "rgbtrack/episodes" / f"e{int(episode):05d}" / f"{name}_poses.jsonl"
                if not pose.is_file():
                    errors.append(f"missing {role} pose stream for episode {episode}: {pose}")
        except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
            errors.append(f"invalid published release: {exc}")
    if errors:
        if dataset == "1018_peg_all":
            errors.append("legacy Peg pose/calibration requires independent keyframe RGB-D projection review; "
                          "v1 migration preserves old geometry without recalibration")
        return result
    try:
        from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

        m, rel, approved = read_json(manifest), read_json(release), read_json(approval)
        if m.get("status") != "complete" or m.get("dataset") != dataset:
            raise ValueError("geometry is incomplete or belongs to another dataset")
        if m.get("object_geometry_contract", {}).get("point_count") != 512 or m.get("coordinate_frame") != "base_link" or m.get("length_unit") != "m":
            raise ValueError("geometry does not establish 512 metric base_link points")
        if m.get("role_to_object") != ROLES[dataset]:
            raise ValueError("geometry roles do not match the requested two objects")
        if rel.get("dataset") != dataset or set(rel.get("objects", [])) != set(ROLES[dataset].values()):
            raise ValueError("release objects/dataset mismatch")
        if m.get("source_release_sha256") != sha256(release) or m.get("approved_geometry_sha256") != sha256(approval):
            raise ValueError("release or approved CAD manifest hash mismatch")
        if approved.get("status") != "human_accepted":
            raise ValueError("CAD manifest has not been human accepted")
        if m.get("surface_sampling_seed") != SURFACE_SAMPLING_SEED:
            raise ValueError("512-point sampling seed mismatch")
        cad = {}
        for index, name in enumerate(rel["objects"]):
            mesh = prep / "geometry/approved" / f"{name}.glb"
            entry = approved.get("assets", {}).get(name, {})
            digest = sha256(mesh)
            if (digest != entry.get("sha256") or digest != m.get("approved_meshes_sha256", {}).get(name)
                    or float(entry.get("scale_m_per_unit", -1)) != 1.0):
                raise ValueError(f"{name}: CAD hash or metric scale mismatch")
            cad[name] = load_surface_points(mesh, SURFACE_SAMPLING_SEED + index)
        checks["cad_and_release"] = "passed"
        dataset_reader = ObjectGeometryDerivedDataset(root, version="v1")
        episodes = tuple(RawSpatialDataset.discover_episode_ids(root))
        if episodes != tuple(dataset_reader.episode_ids) or not episodes:
            raise ValueError("raw and geometry episode IDs differ or are empty")
        expected_calibration = front_calibration_provenance(repo, profile)
        rmses, valid_counts = [], dict.fromkeys(ROLES[dataset], 0)
        for episode in episodes:
            raw = RawSpatialDataset(root, episode=episode)
            if "front" not in raw.camera_roles or "front" not in raw.depth_roles:
                raise ValueError(f"episode {episode}: missing front RGB or depth")
            frames = list(dataset_reader.frame_ids(episode))
            if not frames or frames != list(range(len(frames))):
                raise ValueError(f"episode {episode}: geometry frame IDs are not contiguous")
            pose_dir = prep / "rgbtrack/episodes" / f"e{episode:05d}"
            expected_run = prep / "rgbtrack/runs" / rel["run_tag"] / "episodes" / f"e{episode:05d}"
            if not pose_dir.is_symlink() or pose_dir.resolve() != expected_run.resolve():
                raise ValueError(f"episode {episode}: published pose symlink missing/mismatched")
            run = read_json(pose_dir / "manifest.json")
            cal_error = calibration_provenance_error(run.get("calibration"), expected_calibration)
            if cal_error:
                raise ValueError(f"episode {episode}: {cal_error}")
            rows = {}
            for role, name in ROLES[dataset].items():
                path = pose_dir / f"{name}_poses.jsonl"
                if sha256(path) != m.get("pose_files_sha256_by_episode", {}).get(str(episode), {}).get(name):
                    raise ValueError(f"episode {episode} {name}: pose file hash mismatch")
                rows[role] = source_rows(path, object_name=name)
                if len(rows[role]) != len(frames):
                    raise ValueError(f"episode {episode} {name}: pose/geometry frame counts differ")
            states = raw.states()
            if list(map(int, states["frame_index"])) != frames:
                raise ValueError(f"episode {episode}: raw/pose/geometry frame mismatch")
            selected = sorted({0, len(frames)//2, len(frames)-1})
            for frame in selected:
                sample = dataset_reader.get(episode_index=episode, frame_index=frame)
                for role, name in ROLES[dataset].items():
                    row = rows[role][frame]
                    mask = np.asarray(getattr(sample, f"{role}_mask"), dtype=bool)
                    if bool(mask.any()) != row["valid"]:
                        raise ValueError(f"episode {episode} frame {frame} {role}: pose mask differs")
                    if row["valid"]:
                        actual = np.asarray(getattr(sample, f"{role}_xyz_m"))[mask]
                        expected = cloud_from_pose(row, cad[name])[mask]
                        rmse = float(np.sqrt(np.mean(np.sum((actual - expected) ** 2, axis=-1))))
                        if not np.isfinite(rmse) or rmse > 1e-4:
                            raise ValueError(f"episode {episode} frame {frame} {role}: ordered geometry RMSE {rmse}")
                        rmses.append(rmse)
                        valid_counts[role] += 1
        if min(valid_counts.values()) == 0:
            raise ValueError("no valid sampled frames for one or both roles")
        checks.update(pose_hash_and_alignment="passed", episodes=len(episodes),
                      valid_sampled_frames=valid_counts, max_correspondence_rmse_m=max(rmses),
                      geometry_manifest_sha256=sha256(manifest), release_sha256=sha256(release),
                      profile_sha256=expected_calibration["profile"]["sha256"])
        checks["geometric_accuracy"] = "not_verified: alignment with raw RGB-D requires visual review"
        result["status"] = "requires_visual_review"
        errors.append("Pose source and ordered cloud agree, but visual/geometric accuracy is not yet reviewed")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError, ImportError) as exc:
        errors.append(f"audit failed: {type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--profile", default="configs/ur7e_xhand/front_calibration_candidate.json")
    parser.add_argument("--output", type=Path, default=Path("output/cad_pose_transformer_v1/reports/preflight.json"))
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    data = (args.data_root or repo / "data").expanduser().resolve()
    report = {"status": "blocked", "datasets": [audit_task(repo, data, name, args.profile) for name in ROLES],
              "note": "Provenance checks cannot by themselves establish pose geometric accuracy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
