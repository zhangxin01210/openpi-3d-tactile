#!/usr/bin/env python3
"""Compare online perception with Reference Geometry under the training contract."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OBJECTS = {
    "manipulated": ("funnel", "assets/object_geometry/synthware_19_22/funnel_F181975.glb"),
    "target": ("target_flask", "assets/object_geometry/synthware_19_22/flask_F301950.glb"),
}
HAND_FIELDS = (
    "manipulated_relation_valid", "target_relation_valid", "manipulated_geometry_quality", "target_geometry_quality",
    "grasp_reference_force_norm", "grasp_short_delta_force_norm", "grasp_contact_valid", "grasp_quality",
    "funnel_mask", "funnel_mask_valid", "funnel_mask_quality", "funnel_mask_hand_offset_xy",
)
REQUIRED_FIELDS = tuple(
    f"{name}_{field}"
    for name in ("manipulated", "target")
    for field in ("xyz_m", "mask", "source", "age", "confidence")
) + tuple(f"hand_scene__{name}" for name in HAND_FIELDS)


def flattened(source) -> dict[str, np.ndarray]:
    prefix = "object_geometry__"
    result = {key.removeprefix(prefix): np.asarray(source[key]) for key in source.files if key.startswith(prefix)}
    result.pop("episode_index", None)
    result.pop("frame_index", None)
    return result


def fit_ordered_pose(local_points: np.ndarray, world_points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a rigid pose using corresponding points from the canonical sampler."""
    source = np.asarray(local_points, dtype=np.float64)
    target = np.asarray(world_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Pose fit needs matching [N,3] arrays, got {source.shape} and {target.shape}")
    if len(source) < 8 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Pose fit needs at least 8 finite ordered point pairs")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = (u @ correction @ vt).T
    translation = target_center - rotation @ source_center
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    reconstructed = source @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((reconstructed - target) ** 2, axis=1))))
    return transform, rmse


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a)[:3, :3] @ np.asarray(b)[:3, :3].T
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def symmetric_nn(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.spatial import cKDTree
        d1 = cKDTree(y).query(x, k=1, workers=1)[0]
        d2 = cKDTree(x).query(y, k=1, workers=1)[0]
    except ImportError:
        d1 = np.min(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1), axis=1)
        d2 = np.min(np.linalg.norm(y[:, None, :] - x[None, :, :], axis=-1), axis=1)
    distances = np.concatenate([d1, d2])
    return float(np.median(distances)), float(np.percentile(distances, 95))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--reference-npz-dir", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, default=ROOT / "configs/object_geometry/datasets/1018_peg_all/config.json")
    parser.add_argument("--episode", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    online = args.online_dir.expanduser().resolve()
    reference = args.reference_npz_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    online_rows = [json.loads(line) for line in (online / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    online_by_frame = {int(row["frame_index"]): row for row in online_rows}
    reference_paths = sorted(reference.glob("frame_*.npz"))
    reference_by_frame = {int(path.stem.removeprefix("frame_")): path for path in reference_paths}
    frames = sorted(set(online_by_frame) & set(reference_by_frame))
    if not frames:
        raise FileNotFoundError(f"No matching frames between {online} and {reference}")

    sys.path.insert(0, str(repo / "src"))
    from openpi.object_geometry.production import load_mesh_surface_points
    from openpi.policies.hand_scene_online import OnlineObjectGeometryBuilder, OnlinePose
    from openpi.spatial.calibration_profile import apply_front_calibration_profile
    from openpi.spatial.config import make_baseline_config
    from openpi.spatial.preprocess import SpatialPreprocessor
    sys.path.insert(0, str(repo / "scripts/deploy"))
    from run_policy_dry_run import _geometry_from_npz

    geometry_config_path = args.geometry_config.expanduser().resolve()
    geometry_config = json.loads(geometry_config_path.read_text(encoding="utf-8"))
    scale_manifest = json.loads((repo / "data/pose_eval/hand_scene_metric_mesh_scales.json").read_text(encoding="utf-8"))
    meshes: dict[str, np.ndarray] = {}
    samples: dict[str, dict[str, Any]] = {}
    for _field, (object_name, relative_mesh) in OBJECTS.items():
        asset = geometry_config["assets"][object_name]
        mesh_path = repo / relative_mesh
        if (repo / asset["mesh"]).resolve() != mesh_path.resolve():
            raise ValueError(f"{object_name} mesh differs from training geometry config")
        scale = float(asset["scale_m_per_unit"])
        if not np.isclose(scale, float(scale_manifest["objects"][object_name]), atol=0.0, rtol=0.0):
            raise ValueError(f"{object_name} scale differs between training config and scale manifest")
        points, metadata = load_mesh_surface_points(
            mesh_path,
            scale_m_per_unit=scale,
            count=512,
            seed=int(asset["sampling_seed"]),
        )
        meshes[object_name] = points
        samples[object_name] = {
            "seed": int(asset["sampling_seed"]), "scale_m_per_unit": scale,
            "point_count": 512, "sha256": metadata["source_sha256"],
        }

    spatial_config = apply_front_calibration_profile(
        make_baseline_config().with_camera_roles("front"),
        repo / "configs/ur7e_xhand/front_calibration_candidate.json",
    )
    tactile_preprocessor = SpatialPreprocessor.from_repo_root(repo_root=repo, config=spatial_config)
    if 0 not in reference_by_frame or 1 not in reference_by_frame:
        raise FileNotFoundError("Reference frames 0 and 1 are required for the frame-0 observation/static prior contract")
    with np.load(reference_by_frame[1], allow_pickle=False) as anchor_packet:
        target_anchor, target_anchor_fit_rmse = fit_ordered_pose(meshes["target_flask"], anchor_packet["object_geometry__target_xyz_m"])

    motion = geometry_config.get("motion_assumptions", {})
    prior_start = int(motion.get("static_prior_start_frame", 1))
    builder = OnlineObjectGeometryBuilder(
        funnel_surface_points_m=meshes["funnel"],
        flask_surface_points_m=meshes["target_flask"],
        static_target_anchor=target_anchor,
        static_target_anchor_provenance="reference_geometry",
        static_target_anchor_start_frame=prior_start,
    )
    builder.reset_episode()

    online_metadata_path = online / "run_metadata.json"
    online_metadata = json.loads(online_metadata_path.read_text(encoding="utf-8")) if online_metadata_path.is_file() else {}
    recorded_assets = online_metadata.get("training_geometry", {}).get("assets", {})
    sampling_matches = all(
        name in recorded_assets and int(recorded_assets[name].get("sampling_seed", -1)) == spec["seed"]
        for name, spec in samples.items()
    )
    policy_metadata = online_metadata.get("rgbtrack_training_policy", {})
    pose_rules_reused = bool(policy_metadata.get("same_hooks_as_training_pose_pipeline", False))

    schema_errors: list[str] = []
    reference_fields: list[str] | None = None
    online_fields: list[str] | None = None
    online_field_stats: dict[str, dict[str, Any]] = {}
    same_pose_stats: dict[str, dict[str, Any]] = {}
    same_pose_errors: list[str] = []
    fit_rmse: dict[str, list[float]] = {name: [] for name, _mesh in OBJECTS.values()}
    per_frame: list[dict[str, Any]] = []

    for frame in frames:
        online_packet_path = online / "frames" / f"frame_{frame:06d}.npz"
        if not online_packet_path.is_file():
            schema_errors.append(f"missing online frame packet {frame}")
            continue
        with np.load(online_packet_path, allow_pickle=False) as online_packet, np.load(reference_by_frame[frame], allow_pickle=False) as reference_packet:
            online_flat = flattened(online_packet)
            reference_flat = flattened(reference_packet)
            if reference_fields is None:
                reference_fields = sorted(reference_flat)
                online_fields = sorted(online_flat)
            absent = sorted(set(REQUIRED_FIELDS) - set(reference_flat))
            if absent:
                schema_errors.append(f"reference frame {frame} missing required features: {absent}")

            loaded = _geometry_from_npz(reference_packet)
            recovered = {key: np.asarray(value) for key, value in loaded.items() if key != "hand_scene"}
            for key, value in loaded["hand_scene"].items():
                if key == "causal_tactile":
                    continue
                recovered[f"hand_scene__{key}"] = np.asarray(value)
            if set(recovered) != set(reference_flat):
                schema_errors.append(f"policy loader changed feature inventory at frame {frame}")
            for key in set(recovered) & set(reference_flat):
                if not np.array_equal(recovered[key], reference_flat[key], equal_nan=True):
                    schema_errors.append(f"policy loader changed {key} at frame {frame}")

            for key in sorted(set(online_flat) & set(reference_flat)):
                actual, expected = online_flat[key], reference_flat[key]
                stats = online_field_stats.setdefault(key, {"frames": 0, "exact_frames": 0, "max_abs_error": 0.0})
                stats["frames"] += 1
                exact = bool(np.array_equal(actual, expected, equal_nan=True))
                stats["exact_frames"] += int(exact)
                if actual.shape == expected.shape and np.issubdtype(actual.dtype, np.number) and np.issubdtype(expected.dtype, np.number):
                    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                    stats["max_abs_error"] = max(stats["max_abs_error"], float(difference.max(initial=0.0)))

            ref_transforms: dict[str, np.ndarray | None] = {}
            ref_fit_error: dict[str, float | None] = {}
            for field, (object_name, _mesh) in OBJECTS.items():
                valid = np.asarray(reference_packet[f"object_geometry__{field}_mask"], dtype=bool).all()
                if valid:
                    transform, rmse = fit_ordered_pose(
                        meshes[object_name], reference_packet[f"object_geometry__{field}_xyz_m"]
                    )
                    ref_transforms[object_name] = transform
                    ref_fit_error[object_name] = rmse
                    fit_rmse[object_name].append(rmse)
                else:
                    ref_transforms[object_name] = None
                    ref_fit_error[object_name] = None
            if ref_transforms["target_flask"] is None:
                raise ValueError(f"Reference static flask invalid at frame {frame}")
            target_observed = ref_transforms["target_flask"] if frame == 0 else target_anchor
            funnel_valid = ref_transforms["funnel"] is not None
            funnel_pose = OnlinePose(
                T_base_link_object=np.eye(4) if not funnel_valid else ref_transforms["funnel"],
                valid=funnel_valid, confidence=1.0 if funnel_valid else 0.0,
                status="reference_observed" if funnel_valid else "invalid_reference",
                pose_source="rgbtrack_observed" if funnel_valid else "unknown",
                observed_pose_valid=funnel_valid,
            )
            target_pose = OnlinePose(
                T_base_link_object=target_observed, valid=True, confidence=1.0,
                status="reference_observed" if frame == 0 else "static_prior",
                pose_source="rgbtrack_observed" if frame == 0 else "static_geometry_prior",
                observed_pose_valid=frame == 0,
            )
            regenerated = builder.build_frame(
                frame_index=frame,
                timestamp_s=float(np.asarray(online_packet["timestamp_s"]).reshape(())),
                funnel_pose=funnel_pose,
                target_flask_pose=target_pose,
                tactile_force_norm=tactile_preprocessor.preprocess_tactile(
                    np.asarray(online_packet["robot_state"]).reshape(-1)
                ).force_norm,
                funnel_mask=None,
                target_flask_mask=None,
                funnel_mask_hand_offset_xy=None,
            ).as_dict()
            regenerated_flat = {
                key: np.asarray(value)
                for key, value in regenerated.items()
                if key not in {"episode_index", "frame_index", "hand_scene"}
            }
            regenerated_flat.update({
                f"hand_scene__{key}": np.asarray(value)
                for key, value in regenerated["hand_scene"].items()
                if key != "causal_tactile"
            })
            parity_frame = {"frame_index": frame, "fields": 0, "exact_fields": 0, "max_abs_error": 0.0}
            for key in REQUIRED_FIELDS:
                if key not in regenerated_flat or key not in reference_flat:
                    same_pose_errors.append(f"frame {frame}: missing same-pose field {key}")
                    continue
                actual, expected = regenerated_flat[key], reference_flat[key]
                if actual.shape != expected.shape:
                    same_pose_errors.append(f"frame {frame}: shape mismatch for {key}")
                    continue
                exact = bool(np.array_equal(actual, expected, equal_nan=True))
                stats = same_pose_stats.setdefault(key, {"frames": 0, "exact_frames": 0, "max_abs_error": 0.0})
                stats["frames"] += 1
                stats["exact_frames"] += int(exact)
                if np.issubdtype(actual.dtype, np.number) and np.issubdtype(expected.dtype, np.number):
                    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                    max_error = float(delta.max(initial=0.0))
                    stats["max_abs_error"] = max(stats["max_abs_error"], max_error)
                    parity_frame["max_abs_error"] = max(parity_frame["max_abs_error"], max_error)
                parity_frame["fields"] += 1
                parity_frame["exact_fields"] += int(exact)

            frame_report: dict[str, Any] = {"frame_index": frame, "same_pose_builder": parity_frame}
            row = online_by_frame[frame]
            for field, (object_name, _mesh) in OBJECTS.items():
                prefix = f"object_geometry__{field}"
                on_xyz = np.asarray(online_packet[prefix + "_xyz_m"], dtype=np.float64)
                on_mask = np.asarray(online_packet[prefix + "_mask"], dtype=bool)
                ref_xyz = np.asarray(reference_packet[prefix + "_xyz_m"], dtype=np.float64)
                ref_mask = np.asarray(reference_packet[prefix + "_mask"], dtype=bool)
                pose = row["objects"][object_name]
                pose_valid = bool(pose["valid"])
                ref_pose = ref_transforms[object_name]
                cloud_pose = None
                if sampling_matches and on_mask.all():
                    cloud_pose, _cloud_fit_rmse = fit_ordered_pose(meshes[object_name], on_xyz)
                metric = {
                    "online_valid": pose_valid,
                    "reference_valid_points": int(ref_mask.sum()),
                    "online_valid_points": int(on_mask.sum()),
                    "reference_pose_fit_rmse_m": ref_fit_error[object_name],
                    "tracker_position_delta_m": None,
                    "tracker_rotation_delta_deg": None,
                    "effective_geometry_position_delta_m": None,
                    "effective_geometry_rotation_delta_deg": None,
                    "pointcloud_nn_median_m": None,
                    "pointcloud_nn_p95_m": None,
                }
                if pose_valid and ref_pose is not None:
                    observed = np.asarray(pose["T_base_link_object"], dtype=np.float64)
                    metric["tracker_position_delta_m"] = float(np.linalg.norm(observed[:3, 3] - ref_pose[:3, 3]))
                    metric["tracker_rotation_delta_deg"] = rotation_delta_deg(observed, ref_pose)
                if cloud_pose is not None and ref_pose is not None:
                    metric["effective_geometry_position_delta_m"] = float(
                        np.linalg.norm(cloud_pose[:3, 3] - ref_pose[:3, 3])
                    )
                    metric["effective_geometry_rotation_delta_deg"] = rotation_delta_deg(cloud_pose, ref_pose)
                if sampling_matches and on_mask.all() and ref_mask.all():
                    median, p95 = symmetric_nn(on_xyz, ref_xyz)
                    metric["pointcloud_nn_median_m"] = median
                    metric["pointcloud_nn_p95_m"] = p95
                frame_report[object_name] = metric

            computed_tactile = tactile_preprocessor.preprocess_tactile(
                np.asarray(online_packet["robot_state"]).reshape(-1)
            ).force_norm
            ref_tactile = np.asarray(reference_packet["spatial__tactile_force_norm"], dtype=np.float32)
            tactile_delta = np.abs(computed_tactile.astype(np.float64) - ref_tactile.astype(np.float64))
            frame_report["spatial_tactile"] = {
                "exact": bool(np.array_equal(computed_tactile, ref_tactile)),
                "max_abs_error": float(tactile_delta.max(initial=0.0)),
            }
            per_frame.append(frame_report)

    object_status = {
        name: {
            "valid_frames": sum(bool(row["objects"][name]["valid"]) for row in online_rows),
            "invalid_frames": [int(row["frame_index"]) for row in online_rows if not row["objects"][name]["valid"]],
            "recovery_frames": [int(row["frame_index"]) for row in online_rows if row["objects"][name].get("recovery")],
            "mask_invalid_frames": [int(row["frame_index"]) for row in online_rows if not row["masks"][name]["valid"]],
        }
        for name in ("funnel", "target_flask")
    }
    tactile_exact = sum(bool(frame["spatial_tactile"]["exact"]) for frame in per_frame)
    all_parity_errors = [float(v["max_abs_error"]) for v in same_pose_stats.values()]
    same_pose_max_error = max(all_parity_errors, default=float("inf"))
    same_pose_pass = not same_pose_errors and same_pose_max_error <= 2e-5 and not schema_errors

    def summarize(obj: str) -> dict[str, Any]:
        records = [frame[obj] for frame in per_frame]
        summary: dict[str, Any] = {}
        for key in (
            "tracker_position_delta_m", "tracker_rotation_delta_deg",
            "effective_geometry_position_delta_m", "effective_geometry_rotation_delta_deg",
            "pointcloud_nn_median_m", "pointcloud_nn_p95_m", "reference_pose_fit_rmse_m",
        ):
            vals = [float(record[key]) for record in records if record[key] is not None]
            summary[key + "_mean"] = float(np.mean(vals)) if vals else None
            summary[key + "_p95"] = float(np.percentile(vals, 95)) if vals else None
            summary[key + "_frames"] = len(vals)
        return summary

    reference_fit_max = {name: max(values, default=0.0) for name, values in fit_rmse.items()}
    causal_fields_present = any(key.startswith("hand_scene__causal_tactile__") for key in (reference_fields or []))
    report = {
        "schema_version": 2,
        "episode": args.episode,
        "online_dir": str(online),
        "reference_npz_dir": str(reference),
        "training_geometry_config": str(geometry_config_path),
        "frames_compared": len(per_frame),
        "feature_consistency": {
            "reference_feature_pack_contract": "PASS" if not schema_errors else "FAIL",
            "reference_npz_fields": reference_fields,
            "required_reference_fields": list(REQUIRED_FIELDS),
            "reference_geometry_loaded_without_value_conversion": not schema_errors,
            "spatial_tactile_exact_frames": tactile_exact,
            "spatial_tactile_total_frames": len(per_frame),
            "online_reference_field_stats": online_field_stats,
            "online_only_fields": sorted(set(online_fields or []) - set(reference_fields or [])),
            "reference_only_fields": sorted(set(reference_fields or []) - set(online_field_stats)),
            "same_pose_geometry_builder_regeneration": "PASS" if same_pose_pass else "FAIL",
            "same_pose_frames": len(per_frame),
            "same_pose_feature_stats": same_pose_stats,
            "same_pose_max_abs_error": same_pose_max_error,
            "same_pose_errors": same_pose_errors,
            "reference_pose_fit_rmse_max_m": reference_fit_max,
            "canonical_mesh_sampling": samples,
            "causal_tactile_in_reference": causal_fields_present,
            "causal_tactile_note": "The exported Reference NPZ has no causal_tactile fields; those fields are an online extension and are not a same-feature parity claim.",
            "schema_errors": schema_errors,
        },
        "reperception_geometry_comparison": {
            "method": "Reference pose is recovered by rigid Kabsch fitting the exact ordered 512-point training sampler to the Reference NPZ cloud. No ICP or independent point resampling is used.",
            "online_training_pose_rules_reused": pose_rules_reused,
            "online_mesh_sampling_matches_training": sampling_matches,
            "reference_pose_is_inferred_from_ordered_cloud": True,
            "per_object": {
                "funnel": summarize("funnel"),
                "target_flask": summarize("target_flask"),
            },
            "objects": object_status,
            "interpretation": "Same-pose feature parity and re-perception pose deltas are reported separately. Raw RGBTrack target pose remains diagnostic when the training config uses a static target prior; effective geometry deltas measure the actual encoded cloud.",
        },
        "per_frame": per_frame,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "online_offline_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = ["frame_index", "spatial_tactile_exact", "spatial_tactile_max_abs_error", "same_pose_max_abs_error"]
    object_metrics = (
        "online_valid", "reference_valid_points", "online_valid_points", "reference_pose_fit_rmse_m",
        "tracker_position_delta_m", "tracker_rotation_delta_deg", "effective_geometry_position_delta_m",
        "effective_geometry_rotation_delta_deg", "pointcloud_nn_median_m", "pointcloud_nn_p95_m",
    )
    for name in ("funnel", "target_flask"):
        columns.extend([f"{name}_{metric}" for metric in object_metrics])
    with (output / "geometry_comparison_by_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame in per_frame:
            row = {
                "frame_index": frame["frame_index"],
                "spatial_tactile_exact": frame["spatial_tactile"]["exact"],
                "spatial_tactile_max_abs_error": frame["spatial_tactile"]["max_abs_error"],
                "same_pose_max_abs_error": frame["same_pose_builder"]["max_abs_error"],
            }
            for name in ("funnel", "target_flask"):
                row.update({f"{name}_{key}": value for key, value in frame[name].items()})
            writer.writerow(row)
    md = [
        f"# Episode {args.episode}: training-aligned geometry comparison", "",
        f"- Frames compared: {len(per_frame)}",
        f"- Reference feature-pack contract: {report['feature_consistency']['reference_feature_pack_contract']}",
        f"- Same-pose geometry builder regeneration: {report['feature_consistency']['same_pose_geometry_builder_regeneration']} (max error {same_pose_max_error:.3g})",
        f"- RGBTrack training pose rules reused: {pose_rules_reused}",
        f"- Canonical mesh sampling matches training: {sampling_matches}",
        f"- Replay state → SpatialPreprocessor tactile exact: {tactile_exact}/{len(per_frame)}", "",
        "## Re-perception deltas", "",
        f"- Funnel tracker translation / rotation mean: {summarize('funnel')['tracker_position_delta_m_mean']} m / {summarize('funnel')['tracker_rotation_delta_deg_mean']}°",
        f"- Target flask tracker translation / rotation mean: {summarize('target_flask')['tracker_position_delta_m_mean']} m / {summarize('target_flask')['tracker_rotation_delta_deg_mean']}°",
        f"- Target flask effective geometry translation mean: {summarize('target_flask')['effective_geometry_position_delta_m_mean']} m", "",
        "Reference transforms were recovered with the same ordered training mesh samples. Tracker pose deltas are kept separate from effective encoded geometry; for a static target, the raw RGBTrack pose is diagnostic and the builder prior determines the policy input.", "",
        "Per-frame results and feature-field statistics are in `online_offline_comparison.json` and `geometry_comparison_by_frame.csv`.",
    ]
    (output / "online_offline_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "frames_compared": len(per_frame),
        "same_pose_builder_regeneration": report["feature_consistency"]["same_pose_geometry_builder_regeneration"],
        "same_pose_max_abs_error": same_pose_max_error,
        "spatial_tactile_exact_frames": tactile_exact,
        "online_training_pose_rules_reused": pose_rules_reused,
        "online_mesh_sampling_matches_training": sampling_matches,
        "reperception_geometry_comparison": report["reperception_geometry_comparison"]["per_object"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
