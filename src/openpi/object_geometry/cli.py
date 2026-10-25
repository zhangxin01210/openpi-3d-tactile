"""Unified Object Geometry production CLI.

The command is an orchestration boundary around the already reviewed SAM2 and
RGBTrack scripts.  It never chooses a newest output directory.  ``batch``
records deterministic automatic QA and exports only the episodes that pass;
human review remains an independent, explicit status for offline auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.object_geometry.calibration import resolve_front_calibration_profile
from openpi.object_geometry.production import CausalHandSceneState
from openpi.object_geometry.production import SpatialTactileProvider
from openpi.object_geometry.production import as_transform
from openpi.object_geometry.production import build_episode_records
from openpi.object_geometry.production import load_mesh_surface_points
from openpi.object_geometry.production import merge_requested_episodes
from openpi.object_geometry.production import read_pose_run
from openpi.object_geometry.production import sha256_file
from openpi.object_geometry.production import validate_static_flask_prior
from openpi.object_geometry.production import write_episode_sidecar


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else match.group(0))

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_text(encoding="utf8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires the existing PyYAML dependency; JSON is supported without it") from exc
        config = yaml.safe_load(raw)
    else:
        config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("Object Geometry config must be a JSON/YAML object")
    config = _expand(config)
    # configs/object_geometry/datasets/<dataset>/config.json -> checkout root.
    repo_root = Path(os.environ.get("OPENPI_REPO_ROOT", path.parents[4])).expanduser().resolve()
    return config, path, repo_root


def _path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()


def _paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    values = config.get("paths", {})
    required = ("dataset_root", "scratch_root", "front_calibration", "dimensions_json")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Config paths missing {missing}")
    output = {name: _path(repo_root, values[name]) for name in required}
    for name in ("hoi_root", "mask_root_template", "frames_root_template", "spatial_root", "tactile_root", "hand_offset_root", "approved_pose_root"):
        if values.get(name):
            output[name] = _path(repo_root, values[name])
    return output


def _episodes(value: str | None, *, default: list[int]) -> list[int]:
    if value is None:
        return sorted(set(int(item) for item in default))
    output: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = map(int, token.split("-", 1))
            output.extend(range(start, stop + 1))
        else:
            output.append(int(token))
    if not output:
        raise ValueError("No episodes selected")
    return sorted(set(output))


def _asset_config(config: dict[str, Any], paths: dict[str, Path], name: str) -> tuple[Path, float, int]:
    assets = config.get("assets", {})
    if name not in assets:
        raise ValueError(f"Config has no asset {name}")
    asset = assets[name]
    mesh = _path(Path(config["_repo_root"]), asset["mesh"])
    scale = float(asset["scale_m_per_unit"])
    seed = int(asset.get("sampling_seed", 0))
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid explicit scale for {name}: {scale}")
    return mesh, scale, seed


def _format_template(value: Path, episode: int) -> Path:
    return Path(str(value).format(episode=int(episode), episode03=f"{int(episode):03d}"))


def _position_group(config: dict[str, Any], episode: int) -> dict[str, Any] | None:
    """Return the reviewed six-position group for an episode, if known."""

    for group in config.get("position_groups", []):
        if int(group.get("episode_start", 1)) <= int(episode) <= int(group.get("episode_end", 0)):
            return dict(group)
    return None


def _template_path(config: dict[str, Any], repo_root: Path, episode: int) -> Path | None:
    value = config.get("initialization_templates", {}).get("template_path")
    if not value:
        return None
    return _format_template(_path(repo_root, value), int(episode))


def _episode_path_overrides(config: dict[str, Any], paths: dict[str, Path], key: str, episode: int) -> Path | None:
    values = config.get("episode_overrides", {}).get(key, {})
    value = values.get(str(int(episode))) if isinstance(values, dict) else None
    if not value:
        return None
    return _format_template(_path(Path(config["_repo_root"]), value), episode)


def _mask_selection(config: dict[str, Any], episode: int, *, explicit: bool = False) -> str:
    if explicit:
        return "explicit_cli_override"
    if config.get("episode_overrides", {}).get("mask_root", {}).get(str(int(episode))):
        return "episode_config_override"
    return "configured_episode_template"


def _parse_episode_paths(values: list[str] | None) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected EPISODE=PATH, got {value!r}")
        episode_text, path_text = value.split("=", 1)
        episode = int(episode_text)
        if episode < 0 or not path_text:
            raise ValueError(f"Invalid episode path override: {value!r}")
        if episode in result:
            raise ValueError(f"Duplicate episode path override: {episode}")
        result[episode] = Path(path_text).expanduser().resolve()
    return result


def _mask_files(root: Path, object_name: str) -> list[Path]:
    directory = root / object_name
    return sorted(directory.glob("*.png")) if directory.is_dir() else []


def _numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem.replace("frame_", ""))
    except ValueError:
        return None


def _hash_mask_root(root: Path) -> dict[str, Any]:
    files = sorted(path for name in ("funnel", "target_flask") for path in _mask_files(root, name))
    digest = hashlib.sha256()
    file_hashes = {}
    for path in files:
        relative = str(path.relative_to(root))
        file_hash = sha256_file(path)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf8"))
        digest.update(file_hash.encode("ascii"))
    return {
        "root": str(root),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": [str(path.relative_to(root)) for path in files],
        "file_sha256": file_hashes,
    }


def _rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _raw_heuristic_summary(rows: list[Any]) -> dict[str, Any]:
    values = [row.raw_heuristic_score for row in rows if row.raw_heuristic_score is not None]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
        "semantics": "uncalibrated tracker heuristic; not pose accuracy or confidence",
    }


def _lerobot_frame_ids(dataset_root: Path, episode: int) -> tuple[set[int] | None, str | None]:
    try:
        from openpi.spatial_dataset.source import RawSpatialDataset

        states = RawSpatialDataset(dataset_root, episode=int(episode)).states()
        return {int(value) for value in np.asarray(states["frame_index"]).reshape(-1)}, None
    except Exception as exc:  # pragma: no cover - computer-4 data/dependency boundary
        return None, f"LeRobot frame index unavailable: {type(exc).__name__}: {exc}"


def _auto_pose_quality(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    episode: int,
    pose_run: Path,
    mask_root: Path,
    tactile_provider: SpatialTactileProvider | None,
) -> dict[str, Any]:
    run = read_pose_run(pose_run, episode=episode)
    funnel = run["funnel"]
    flask = run["target_flask"]
    threshold = config.get("qa_thresholds", {})
    expected_frames, frame_error = _lerobot_frame_ids(paths["dataset_root"], episode)
    pose_frames = {row.frame_index for row in funnel}
    mask_files = {name: _mask_files(mask_root, name) for name in ("funnel", "target_flask")}
    mask_frames = {name: {_numeric_stem(path) for path in files} - {None} for name, files in mask_files.items()}
    rgb_frame_root = _format_template(paths["frames_root_template"], episode) if paths.get("frames_root_template") else None
    rgb_frames = (
        {_numeric_stem(path) for path in rgb_frame_root.glob("*.jpg")} - {None}
        if rgb_frame_root is not None and rgb_frame_root.is_dir()
        else set()
    )
    frame_checks = {
        "pose_vs_le_robot": expected_frames is not None and pose_frames == expected_frames,
        "pose_vs_rgb": bool(rgb_frames) and pose_frames == rgb_frames,
        "pose_vs_funnel_mask": bool(mask_frames["funnel"]) and pose_frames == mask_frames["funnel"],
        "pose_vs_target_flask_mask": bool(mask_frames["target_flask"]) and pose_frames == mask_frames["target_flask"],
    }
    invalid_matrices = 0
    funnel_valid = []
    translation_jumps: list[float] = []
    rotation_jumps: list[float] = []
    failure_count = 0
    recovery_count = 0
    previous_valid = None
    for row in funnel:
        funnel_valid.append(bool(row.valid))
        try:
            matrix = as_transform(row.T_base_link_object, "T_base_link_object")
        except ValueError:
            invalid_matrices += 1
            continue
        if not row.valid:
            failure_count += 1
        if "recover" in row.status.lower():
            recovery_count += 1
        if previous_valid is not None and row.valid:
            translation_jumps.append(float(np.linalg.norm(matrix[:3, 3] - previous_valid[:3, 3])))
            rotation_jumps.append(_rotation_delta_deg(previous_valid, matrix))
        if row.valid:
            previous_valid = matrix
    failure_count += invalid_matrices
    flask_static_valid = True
    flask_translation_drift = 0.0
    flask_rotation_drift = 0.0
    try:
        validate_static_flask_prior(flask)
        anchor = flask[0].T_base_link_object
        for row in flask:
            flask_translation_drift = max(flask_translation_drift, float(np.linalg.norm(row.T_base_link_object[:3, 3] - anchor[:3, 3])))
            flask_rotation_drift = max(flask_rotation_drift, _rotation_delta_deg(anchor, row.T_base_link_object))
    except ValueError:
        flask_static_valid = False
    funnel_valid_fraction = float(np.mean(funnel_valid)) if funnel_valid else 0.0
    mask_provenance = _hash_mask_root(mask_root)
    pose_mask_value = run["manifest"].get("mask_root")
    pose_mask_root = Path(str(pose_mask_value)).expanduser().resolve() if pose_mask_value else None
    pose_mask_matches = pose_mask_root == mask_root.expanduser().resolve()
    expected_calibration = front_calibration_provenance(Path(config["_repo_root"]), paths["front_calibration"])
    calibration_error = calibration_provenance_error(run["manifest"].get("calibration"), expected_calibration)
    expected_profile_sha = config.get("reproducibility", {}).get("front_profile_sha256")
    scale_manifest_value = run["manifest"].get("scale_manifest")
    scale_manifest_path = Path(str(scale_manifest_value)).expanduser().resolve() if scale_manifest_value else None
    scale_manifest_sha = sha256_file(scale_manifest_path) if scale_manifest_path and scale_manifest_path.is_file() else None
    mesh_checks = {}
    dimensions_sha = sha256_file(paths["dimensions_json"])
    dimensions = json.loads(paths["dimensions_json"].read_text(encoding="utf8"))
    for name in ("funnel", "target_flask"):
        mesh, scale, _ = _asset_config(config, paths, name)
        actual_sha = sha256_file(mesh)
        mesh_checks[name] = {
            "sha256_matches_config": not config["assets"][name].get("sha256") or config["assets"][name]["sha256"] == actual_sha,
            "sha256_matches_dimensions": not dimensions.get("models", {}).get(name, {}).get("sha256") or dimensions["models"][name]["sha256"] == actual_sha,
            "scale_m_per_unit": scale,
            "dimensions_json_sha256": dimensions_sha,
            "dimensions_unit": dimensions.get("models", {}).get(name, {}).get("unit"),
        }
    tactile = {
        "source": "spatial_sidecar" if tactile_provider is not None else "missing",
        "frames_total": len(funnel),
        "frames_present": 0,
        "valid_fraction": 0.0,
        "missing_or_invalid": len(funnel),
        "discontinuous": 0,
    }
    if tactile_provider is not None:
        available_frames = tactile_provider.frame_indices(episode)
        valid_count = 0
        for frame in sorted(pose_frames):
            force = tactile_provider.get_force_norm(episode, frame)
            if force is not None:
                valid_count += 1
        available_pose_frames = sorted(pose_frames & available_frames)
        tactile["discontinuous"] = sum(
            int(right != left + 1)
            for left, right in zip(available_pose_frames, available_pose_frames[1:], strict=False)
        )
        tactile["frames_present"] = valid_count
        tactile["valid_fraction"] = float(valid_count / max(1, len(funnel)))
        tactile["missing_or_invalid"] = len(funnel) - valid_count
        tactile["sidecar_frames_outside_pose"] = len(available_frames - pose_frames)
    checks = {
        **frame_checks,
        "pose_matrices_valid": invalid_matrices == 0,
        "funnel_valid_fraction": funnel_valid_fraction >= float(threshold.get("funnel_min_valid_fraction", 0.5)),
        "translation_jumps": max(translation_jumps, default=0.0) <= float(threshold.get("max_translation_jump_m", 0.15)),
        "rotation_jumps": max(rotation_jumps, default=0.0) <= float(threshold.get("max_rotation_jump_deg", 45.0)),
        "flask_static_anchor": flask_static_valid,
        "flask_static_drift": flask_translation_drift <= float(threshold.get("max_flask_static_translation_drift_m", 1e-4)) and flask_rotation_drift <= float(threshold.get("max_flask_static_rotation_drift_deg", 0.1)),
        "calibration_provenance": calibration_error is None and (not expected_profile_sha or expected_profile_sha == expected_calibration.get("profile", {}).get("sha256")),
        "mesh_provenance": all(
            item["sha256_matches_config"]
            and item["sha256_matches_dimensions"]
            and item["dimensions_unit"] == "meter"
            and np.isfinite(item["scale_m_per_unit"])
            and item["scale_m_per_unit"] > 0
            for item in mesh_checks.values()
        ),
        "pose_mask_source_matches": pose_mask_matches,
        "scale_manifest_provenance": scale_manifest_sha is not None,
        "future_frames": not bool(run["manifest"].get("future_frames_used", False)),
        "tactile_sidecar_available": tactile_provider is not None,
    }
    if not bool(threshold.get("require_le_robot_frame_match", True)):
        checks["pose_vs_le_robot"] = True
    if bool(threshold.get("allow_missing_tactile", True)):
        checks["tactile_sidecar_available"] = True
    auto_qa_passed = all(bool(value) for value in checks.values())
    return {
        "schema_version": 1,
        "episode": int(episode),
        "status": "auto_qa_passed" if auto_qa_passed else "pending_review",
        "human_accepted": False,
        "auto_qa_passed": auto_qa_passed,
        "checks": checks,
        "metrics": {
            "frame_count": len(funnel),
            "expected_le_robot_frames": None if expected_frames is None else len(expected_frames),
            "funnel_valid_fraction": funnel_valid_fraction,
            "invalid_pose_matrices": invalid_matrices,
            "tracking_failures": failure_count,
            "recovery_count": recovery_count,
            "max_translation_jump_m": max(translation_jumps, default=0.0),
            "max_rotation_jump_deg": max(rotation_jumps, default=0.0),
            "raw_heuristic_score": _raw_heuristic_summary(funnel),
            "flask_static_valid_fraction": float(sum(row.valid for row in flask) / max(1, len(flask))),
            "flask_static_translation_drift_m": flask_translation_drift,
            "flask_static_rotation_drift_deg": flask_rotation_drift,
            "tactile": tactile,
            "mask": mask_provenance,
            "pose_mask_root": None if pose_mask_root is None else str(pose_mask_root),
            "scale_manifest": {
                "path": None if scale_manifest_path is None else str(scale_manifest_path),
                "sha256": scale_manifest_sha,
            },
            "mesh": mesh_checks,
            "calibration_error": calibration_error,
            "le_robot_frame_error": frame_error,
        },
        "pose_run": str(pose_run),
        "pose_manifest_sha256": sha256_file(pose_run / "manifest.json"),
        "mask_provenance": mask_provenance,
        "future_frame_used": False,
    }


def _write_batch_failure_html(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = html.escape(json.dumps(report, indent=2, ensure_ascii=False))
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Object Geometry batch QA</title>"
        f"<h1>Episode {report.get('episode')} — {report.get('status')}</h1><pre>{text}</pre>",
        encoding="utf8",
    )


def _existing_human_approval(review_root: Path) -> dict[str, Any] | None:
    """Return a preserved human review, if one already exists."""

    for name in ("approved_pose.json", "pose_review.json"):
        path = review_root / name
        if not path.is_file():
            continue
        review = json.loads(path.read_text(encoding="utf8"))
        if review.get("status") in {"human_accepted", "accepted"}:
            return review
    return None


def _write_auto_qa_review(review_root: Path, quality: dict[str, Any]) -> dict[str, Any] | None:
    """Write automatic QA separately without replacing human approval."""

    review_root.mkdir(parents=True, exist_ok=True)
    human_review = _existing_human_approval(review_root)
    auto_review = dict(quality)
    auto_review.update({"status": "auto_qa_passed", "human_accepted": False, "approved_by": "automatic_quality_gate"})
    if human_review is not None:
        auto_review["preserved_human_review"] = {
            "status": human_review.get("status"),
            "approved_by": human_review.get("approved_by"),
            "review_note": human_review.get("review_note"),
        }
        (review_root / "auto_qa.json").write_text(json.dumps(auto_review, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        return human_review
    (review_root / "approved_pose.json").write_text(json.dumps(auto_review, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return None


def _batch(
    config: dict[str, Any],
    paths: dict[str, Path],
    episodes: list[int],
    *,
    scratch: Path,
    resume: bool,
    auto_qa: bool,
    run_tracking: bool,
    pose_overrides: dict[int, Path],
    mask_overrides: dict[int, Path],
    output_version: str,
) -> int:
    repo_root = Path(config["_repo_root"])
    hoi_root = paths.get("hoi_root")
    tracker_script = repo_root / "scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh"
    output_root = repo_root / "output" / "pose_eval"
    batch_root = scratch / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    spatial_provider = None
    spatial_provider_error = None
    if paths.get("spatial_root") and paths["spatial_root"].is_dir():
        try:
            spatial_provider = SpatialTactileProvider(
                paths["spatial_root"], version=str(config.get("paths", {}).get("spatial_version", "v1_front_candidate"))
            )
        except Exception as exc:  # pragma: no cover - real dataset/dependency boundary
            spatial_provider_error = f"{type(exc).__name__}: {exc}"
    reports: list[dict[str, Any]] = []
    passed: list[int] = []
    for episode in episodes:
        report: dict[str, Any] = {"episode": int(episode), "status": "pending_review", "stage": "init", "human_accepted": False, "auto_qa_passed": False}
        report["position_group"] = _position_group(config, episode)
        if report["position_group"] is None:
            report.update(stage="init", failure_reason="episode_not_in_reviewed_position_group")
            _write_batch_failure_html(batch_root / f"e{episode:03d}" / "position_group_failure.html", report)
            reports.append(report)
            continue
        template_episode = int(report["position_group"]["template_episode"])
        template_path = _template_path(config, repo_root, template_episode)
        if template_path is not None:
            report["mask_template"] = str(template_path)
        mask_root = mask_overrides.get(episode) or _episode_path_overrides(config, paths, "mask_root", episode) or _format_template(paths["mask_root_template"], episode)
        report["mask_root"] = str(mask_root)
        report["mask_selection"] = _mask_selection(config, episode, explicit=episode in mask_overrides)
        mask_files = {name: _mask_files(mask_root, name) for name in ("funnel", "target_flask")}
        frame_root = _format_template(paths["frames_root_template"], episode)
        rgb_frame_ids = {_numeric_stem(path) for path in frame_root.glob("*.jpg")} - {None} if frame_root.is_dir() else set()
        mask_frame_ids = {name: {_numeric_stem(path) for path in files} - {None} for name, files in mask_files.items()}
        report["mask_provenance"] = _hash_mask_root(mask_root)
        mask_ready = bool(rgb_frame_ids) and all(frame_ids == rgb_frame_ids for frame_ids in mask_frame_ids.values())
        report["mask_check"] = {
            "rgb_frames": len(rgb_frame_ids),
            "funnel_masks": len(mask_frame_ids["funnel"]),
            "target_flask_masks": len(mask_frame_ids["target_flask"]),
            "frame_indices_match": mask_ready,
        }
        if not mask_ready:
            report.update(status="pending_review", stage="mask", failure_reason="mask_missing_or_frame_mismatch")
            _write_batch_failure_html(batch_root / f"e{episode:03d}" / "mask_failure.html", report)
            reports.append(report)
            continue
        sidecar_root = paths["dataset_root"] / "object_geometry" / output_version
        sidecar_manifest_path = sidecar_root / "manifest.json"
        existing_context = None
        if resume and sidecar_manifest_path.is_file():
            existing_manifest = json.loads(sidecar_manifest_path.read_text(encoding="utf8"))
            existing_context = existing_manifest.get("episode_provenance", {}).get(str(int(episode)))
        existing_records = (
            [row for row in existing_manifest.get("records", []) if int(row.get("episode_index", -1)) == int(episode)]
            if sidecar_manifest_path.is_file()
            else []
        )
        if resume and (existing_context is not None or existing_records) and (sidecar_root / f"shard_e{episode:03d}").is_dir():
            same_config = existing_context is not None and existing_context.get("config_sha256") == config["_config_sha256"]
            existing_mask = existing_context.get("pose_pipeline", {}).get("mask_provenance", {}) if existing_context else {}
            same_mask = (
                existing_context is not None
                and existing_mask.get("sha256") == report["mask_provenance"]["sha256"]
                and existing_mask.get("root") == report["mask_provenance"]["root"]
            )
            existing_pose = Path(existing_context.get("pose_pipeline", {}).get("pose_run", "")) if existing_context else Path()
            existing_manifest_file = existing_pose / "manifest.json"
            same_pose = bool(existing_context and existing_manifest_file.is_file() and sha256_file(existing_manifest_file) == existing_context.get("pose_pipeline", {}).get("pose_manifest_sha256"))
            if same_config and same_mask and same_pose:
                report.update(status="skipped_existing", stage="sidecar", auto_qa_passed=bool(existing_context.get("quality_gate", {}).get("auto_qa_passed", False)))
                reports.append(report)
                continue
            report.update(status="pending_review", stage="sidecar", failure_reason="existing_provenance_missing_or_mismatch_use_new_output_version")
            _write_batch_failure_html(batch_root / f"e{episode:03d}" / "version_mismatch.html", report)
            reports.append(report)
            continue
        pose_run = pose_overrides.get(episode) or _episode_path_overrides(config, paths, "pose_run", episode)
        run_tag = f"batch_{output_version}_e{episode:03d}"
        expected_run = output_root / f"e{episode}_crosscheck_static_target_{run_tag}"
        if pose_run is None and expected_run.is_dir():
            pose_run = expected_run
        if pose_run is None or not pose_run.is_dir():
            if not run_tracking:
                report.update(status="pending_review", stage="pose", failure_reason="pose_run_missing_and_tracking_disabled")
                _write_batch_failure_html(batch_root / f"e{episode:03d}" / "pose_failure.html", report)
                reports.append(report)
                continue
            if not tracker_script.is_file() or hoi_root is None or not hoi_root.is_dir():
                report.update(status="pending_review", stage="pose", failure_reason="tracking_script_or_HOI_root_missing")
                _write_batch_failure_html(batch_root / f"e{episode:03d}" / "pose_failure.html", report)
                reports.append(report)
                continue
            env = os.environ.copy()
            group = report["position_group"]
            env.update({
                "REPO": str(repo_root),
                "HOI": str(hoi_root),
                "MASK_ROOT": str(mask_root),
                "RUN_TAG": run_tag,
                # The existing tracker keeps the reviewed SAM2 masks as its
                # source of truth.  These variables make the selected
                # six-position template explicit to wrappers that prepare
                # masks, without changing RGBTrack itself.
                "OBJECT_GEOMETRY_POSITION_GROUP": str(group["name"]),
                "OBJECT_GEOMETRY_TEMPLATE_EPISODE": str(group["template_episode"]),
            })
            completed = subprocess.run(["bash", str(tracker_script), str(episode), "run"], cwd=repo_root, env=env, check=False)
            pose_run = expected_run
            if completed.returncode != 0 or not pose_run.is_dir():
                report.update(status="pending_review", stage="pose", failure_reason=f"rgbtrack_pipeline_failed_rc_{completed.returncode}")
                _write_batch_failure_html(batch_root / f"e{episode:03d}" / "pose_failure.html", report)
                reports.append(report)
                continue
        report["pose_run"] = str(pose_run)
        try:
            quality = _auto_pose_quality(config, paths, episode=episode, pose_run=pose_run, mask_root=mask_root, tactile_provider=spatial_provider)
        except Exception as exc:
            report.update(status="pending_review", stage="qa", failure_reason=f"qa_exception:{type(exc).__name__}: {exc}")
            _write_batch_failure_html(batch_root / f"e{episode:03d}" / "qa_failure.html", report)
            reports.append(report)
            continue
        report.update(quality)
        if spatial_provider_error is not None:
            report.setdefault("metrics", {})["tactile_provider_error"] = spatial_provider_error
            report["checks"]["tactile_sidecar_available"] = False
            report["status"] = "pending_review"
            report["auto_qa_passed"] = False
            quality["status"] = "pending_review"
            quality["auto_qa_passed"] = False
        review_root = scratch / "pose_reviews" / f"e{episode:03d}"
        review_root.mkdir(parents=True, exist_ok=True)
        if report["auto_qa_passed"]:
            passed.append(episode)
            human_review = _write_auto_qa_review(review_root, quality)
            if human_review is not None:
                report["status"] = "human_accepted"
                report["human_accepted"] = True
        else:
            _write_batch_failure_html(review_root / "auto_qa_failure.html", quality)
        reports.append(report)
    if auto_qa and passed:
        export_rc = _export(
            config,
            paths,
            passed,
            requested_episodes=episodes,
            scratch=scratch,
            resume=resume,
            output_version=output_version,
            mask_overrides=mask_overrides,
        )
    else:
        export_rc = 0
    try:
        requested_manifest = merge_requested_episodes(
            dataset_root=paths["dataset_root"],
            version=output_version,
            requested_episodes=episodes,
        )
    except Exception as exc:
        requested_manifest = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        export_rc = export_rc or 2
    summary = {
        "schema_version": 1,
        "status": "complete" if all(item.get("auto_qa_passed") or item.get("status") == "skipped_existing" for item in reports) else "partial",
        "output_version": output_version,
        "episodes_requested": episodes,
        "episodes_auto_qa_passed": passed,
        "episodes_pending_review": [item["episode"] for item in reports if item.get("status") == "pending_review"],
        "episodes_human_accepted": [item["episode"] for item in reports if item.get("human_accepted")],
        "episodes_rejected": [item["episode"] for item in reports if item.get("status") == "rejected"],
        "export_return_code": export_rc,
        "requested_manifest": requested_manifest,
        "reports": reports,
    }
    (batch_root / "batch_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if export_rc == 0 and not summary["episodes_pending_review"] else 2


def _mask_preview(config: dict[str, Any], paths: dict[str, Path], episodes: list[int], scratch: Path) -> int:
    representatives = [int(value) for value in config.get("representative_frames", [0, 120, 200, 300, 400, 440, 460])]
    reports = []
    for episode in episodes:
        mask_root = _episode_path_overrides(config, paths, "mask_root", episode) or _format_template(paths.get("mask_root_template", Path("")), episode)
        frame_root = _format_template(paths.get("frames_root_template", Path("")), episode)
        funnel = mask_root / "funnel"
        flask = mask_root / "target_flask"
        frames = list(frame_root.glob("*.jpg")) if frame_root.is_dir() else []
        funnel_masks = list(funnel.glob("*.png")) if funnel.is_dir() else []
        flask_masks = list(flask.glob("*.png")) if flask.is_dir() else []
        report = {
            "episode": episode,
            "status": "ready" if frames and len(funnel_masks) == len(frames) and len(flask_masks) == len(frames) else "missing_or_mismatch",
            "frames": len(frames),
            "funnel_masks": len(funnel_masks),
            "target_flask_masks": len(flask_masks),
            "representative_frames": representatives,
            "mask_root": str(mask_root),
            "manual_annotation": "Use the existing HOI annotate HTML and rerun the same mask preview/track command; no Python edit is required.",
        }
        reports.append(report)
        out = scratch / "mask_reviews" / f"e{episode:03d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "mask_preview_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        rows = []
        for frame in representatives:
            image = frame_root / f"{frame:05d}.jpg"
            fm = funnel / f"{frame:05d}.png"
            tm = flask / f"{frame:05d}.png"
            rows.append(
                f"<tr><td>{frame}</td><td>{html.escape(str(image))}</td><td>{html.escape(str(fm))}</td><td>{html.escape(str(tm))}</td></tr>"
            )
        page = "<!doctype html><meta charset='utf-8'><title>Mask review</title>" \
            + f"<h1>Episode {episode} mask review</h1><pre>{html.escape(json.dumps(report, indent=2))}</pre>" \
            + "<p>Open the existing HOI annotation HTML to edit prompts; this page records paths and coverage only.</p>" \
            + "<table><tr><th>frame</th><th>RGB</th><th>funnel mask</th><th>flask mask</th></tr>" \
            + "".join(rows) + "</table>"
        (out / "mask_review.html").write_text(page, encoding="utf8")
    (scratch / "mask_reviews" / "index.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"stage": "mask-preview", "reports": reports}, indent=2, ensure_ascii=False))
    return 0 if all(report["status"] == "ready" for report in reports) else 2


def _record_pose_review(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    episode: int,
    pose_run: Path,
    status: str,
    approved_by: str,
    note: str,
    scratch: Path,
) -> int:
    status = {"accepted": "human_accepted", "pending": "pending_review"}.get(status, status)
    run = read_pose_run(pose_run, episode=episode)
    profile = front_calibration_provenance(Path(config["_repo_root"]), paths["front_calibration"])
    expected_profile_sha = config.get("reproducibility", {}).get("front_profile_sha256")
    if expected_profile_sha and profile.get("profile", {}).get("sha256") != expected_profile_sha:
        raise ValueError("Configured front calibration SHA256 does not match the active profile")
    actual = run["manifest"].get("calibration")
    mismatch = calibration_provenance_error(actual, profile)
    if mismatch:
        raise ValueError(f"Pose run calibration does not match configured front profile: {mismatch}")
    if config.get("motion_assumptions", {}).get("target_flask") == "static":
        from openpi.object_geometry.production import validate_static_flask_prior

        validate_static_flask_prior(run["target_flask"])
    review = {
        "schema_version": 1,
        "episode": int(episode),
        "status": status,
        "approved_by": approved_by,
        "review_note": note,
        "pose_run": str(pose_run.expanduser().resolve()),
        "pose_manifest_sha256": sha256_file(pose_run / "manifest.json"),
        "calibration": profile,
        "representative_frames": config.get("representative_frames", [120, 200, 300, 400, 440, 460]),
        "future_frame_used": False,
    }
    out = scratch / "pose_reviews" / f"e{episode:03d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pose_review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    if status == "human_accepted":
        (out / "approved_pose.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(review, indent=2, ensure_ascii=False))
    return 0


def _load_approval(scratch: Path, episode: int) -> dict[str, Any]:
    path = scratch / "pose_reviews" / f"e{episode:03d}" / "approved_pose.json"
    if not path.is_file():
        raise FileNotFoundError(f"Episode {episode} has no accepted pose review: {path}")
    review = json.loads(path.read_text(encoding="utf8"))
    if review.get("status") not in {"human_accepted", "auto_qa_passed", "accepted"}:
        raise ValueError(f"Pose review is not accepted or auto-QA passed: {path}")
    if review.get("status") == "accepted":
        review["status"] = "human_accepted"
    pose_review_path = path.with_name("pose_review.json")
    if pose_review_path.is_file():
        pose_review = json.loads(pose_review_path.read_text(encoding="utf8"))
        if pose_review.get("status") in {"human_accepted", "accepted"}:
            # A legacy run may have an automatic approved_pose.json beside a
            # separately recorded human pose_review.json.  Human approval has
            # precedence without modifying either source file.
            if pose_review.get("status") == "accepted":
                pose_review["status"] = "human_accepted"
            review = pose_review
    auto_qa_path = path.with_name("auto_qa.json")
    if auto_qa_path.is_file():
        auto_qa = json.loads(auto_qa_path.read_text(encoding="utf8"))
        review["auto_qa_passed"] = bool(auto_qa.get("auto_qa_passed", False))
        review["checks"] = auto_qa.get("checks")
        review["metrics"] = auto_qa.get("metrics")
    pose_run = Path(review["pose_run"]).expanduser().resolve()
    if sha256_file(pose_run / "manifest.json") != review.get("pose_manifest_sha256"):
        raise ValueError(f"Pose run changed after review: {pose_run}")
    return review


def _export(
    config: dict[str, Any],
    paths: dict[str, Path],
    episodes: list[int],
    *,
    requested_episodes: list[int] | None = None,
    scratch: Path,
    resume: bool,
    output_version: str = "v1_hand_scene",
    mask_overrides: dict[int, Path] | None = None,
) -> int:
    repo_root = Path(config["_repo_root"])
    dimensions = json.loads(paths["dimensions_json"].read_text(encoding="utf8"))
    asset_points: dict[str, np.ndarray] = {}
    asset_manifest: dict[str, Any] = {}
    for name in ("funnel", "target_flask"):
        mesh, scale, seed = _asset_config(config, paths, name)
        model = dimensions.get("models", {}).get(name, {})
        expected_sha = model.get("sha256")
        actual_sha = sha256_file(mesh)
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"{name} mesh SHA256 differs from approved dimensions.json")
        config_sha = config["assets"][name].get("sha256")
        if config_sha and config_sha != actual_sha:
            raise ValueError(f"{name} mesh SHA256 differs from dataset config")
        points, metadata = load_mesh_surface_points(mesh, scale_m_per_unit=scale, seed=seed)
        cache = scratch / "mesh_samples" / f"{name}__{metadata['source_sha256'][:12]}__{seed}.npy"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.is_file():
            cached = np.load(cache, allow_pickle=False)
            if cached.shape != points.shape or not np.allclose(cached, points, atol=0.0, rtol=0.0):
                raise ValueError(f"Deterministic mesh sample cache differs: {cache}")
        else:
            np.save(cache, points)
        metadata["cache"] = str(cache)
        metadata["asset_version"] = config["assets"][name].get("version")
        asset_points[name] = points
        asset_manifest[name] = metadata
    calibration = front_calibration_provenance(repo_root, paths["front_calibration"])
    expected_profile_sha = config.get("reproducibility", {}).get("front_profile_sha256")
    actual_profile_sha = calibration.get("profile", {}).get("sha256")
    if expected_profile_sha and expected_profile_sha != actual_profile_sha:
        raise ValueError("Configured front calibration SHA256 does not match the active profile")
    results = []
    spatial_provider = None
    if paths.get("spatial_root"):
        spatial_provider = SpatialTactileProvider(
            paths["spatial_root"], version=str(config.get("paths", {}).get("spatial_version", "v1_front_candidate"))
        )
    for episode in episodes:
        review = _load_approval(scratch, episode)
        run = read_pose_run(review["pose_run"], episode=episode)
        group = _position_group(config, episode) or {}
        template_path = _template_path(config, repo_root, int(group.get("template_episode", episode)))
        scale_manifest_value = run["manifest"].get("scale_manifest")
        scale_manifest_path = Path(str(scale_manifest_value)).expanduser().resolve() if scale_manifest_value else None
        explicit_mask_root = (mask_overrides or {}).get(episode)
        mask_root = explicit_mask_root or _episode_path_overrides(config, paths, "mask_root", episode) or _format_template(paths["mask_root_template"], episode)
        state = CausalHandSceneState(
            tactile_root=paths.get("tactile_root"),
            spatial_tactile=spatial_provider,
            mask_root=mask_root if mask_root else None,
            hand_offset_root=_format_template(paths["hand_offset_root"], episode) if paths.get("hand_offset_root") else None,
            contact_force_threshold=float(config.get("state", {}).get("contact_force_threshold", 0.05)),
        )
        records = build_episode_records(
            episode=episode,
            pose_run=run,
            funnel_points=asset_points["funnel"],
            flask_points=asset_points["target_flask"],
            flask_motion=config.get("motion_assumptions", {}).get("target_flask", "static"),
            state=state,
        )
        context = {
            "config_path": str(config["_config_path"]),
            "config_sha256": config["_config_sha256"],
            "calibration": calibration,
            "assets": asset_manifest,
            "dimensions_json_sha256": sha256_file(paths["dimensions_json"]),
            "pose_pipeline": {
                "pose_run": review["pose_run"],
                "pose_manifest_sha256": review["pose_manifest_sha256"],
                "review_status": review["status"],
                "reviewed_by": review.get("approved_by"),
                "mask_provenance": {
                    **_hash_mask_root(mask_root),
                    "selection": _mask_selection(config, episode, explicit=explicit_mask_root is not None),
                    "review_source": {
                        "position_group": group.get("name"),
                        "template_episode": group.get("template_episode"),
                        "template_path": str(template_path) if template_path else None,
                    },
                },
                "raw_heuristic_score": {
                    "funnel": _raw_heuristic_summary(run["funnel"]),
                    "target_flask": _raw_heuristic_summary(run["target_flask"]),
                    "semantics": "uncalibrated tracker heuristic; not pose accuracy or confidence",
                },
                "scale_manifest": {
                    "path": None if scale_manifest_path is None else str(scale_manifest_path),
                    "sha256": sha256_file(scale_manifest_path) if scale_manifest_path and scale_manifest_path.is_file() else None,
                },
            },
            "quality_gate": {
                "status": review.get("status"),
                "auto_qa_passed": bool(review.get("auto_qa_passed", False)),
                "geometry_quality_semantics_version": "structural_validity_v2",
                "checks": review.get("checks"),
                "metrics": review.get("metrics"),
            },
            "causal": {
                "future_frame_used": False,
                "state_generator": "CausalHandSceneState.v1",
                "tactile_source": "spatial_sidecar" if spatial_provider is not None else (str(paths.get("tactile_root")) if paths.get("tactile_root") else None),
                "tactile_frames_total": state.tactile_frames_total,
                "tactile_frames_valid": state.tactile_frames_valid,
                "tactile_missing_frames": state.tactile_missing_frames,
                "tactile_discontinuous_frames": state.tactile_discontinuous_frames,
            },
        }
        results.append(
            write_episode_sidecar(
                dataset_root=paths["dataset_root"],
                episode=episode,
                records=records,
                manifest_context=context,
                requested_episodes=episodes if requested_episodes is None else requested_episodes,
                version=output_version,
                resume=resume,
            )
        )
    print(json.dumps({"stage": "export", "results": results}, indent=2, ensure_ascii=False))
    return 0


def _qa(config: dict[str, Any], paths: dict[str, Path], episodes: list[int], *, output_version: str = "v1_hand_scene") -> int:
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

    dataset = ObjectGeometryDerivedDataset(paths["dataset_root"], version=output_version)
    present = set()
    episode_metrics: dict[int, dict[str, Any]] = {}
    for episode, frame in dataset._records:  # noqa: SLF001 - QA owns the coverage boundary.
        episode = int(episode)
        frame = int(frame)
        present.add(episode)
        metrics = episode_metrics.setdefault(episode, {"frames": 0, "funnel_valid": 0, "target_valid": 0, "funnel_points_valid": 0, "target_points_valid": 0})
        metrics["frames"] += 1
        sample = dataset.get(episode_index=episode, frame_index=frame)
        for name in ("manipulated", "target"):
            xyz = getattr(sample, f"{name}_xyz_m")
            mask = getattr(sample, f"{name}_mask")
            if xyz.shape != (512, 3) or not np.isfinite(xyz).all():
                raise ValueError(f"Invalid {name} cloud at e{episode} f{frame}")
            metrics[f"{name}_points_valid"] += int(np.asarray(mask, dtype=bool).sum())
        if sample.hand_scene is not None:
            metrics["funnel_valid"] += int(bool(sample.hand_scene.manipulated_relation_valid))
            metrics["target_valid"] += int(bool(sample.hand_scene.target_relation_valid))
    missing = sorted(set(episodes) - present)
    expected_frame_checks: dict[int, Any] = {}
    for episode in episodes:
        expected, error = _lerobot_frame_ids(paths["dataset_root"], episode)
        actual = {int(frame) for ep, frame in dataset._records if int(ep) == int(episode)}
        expected_frame_checks[int(episode)] = {
            "actual_frames": len(actual),
            "lerobot_frames": None if expected is None else len(expected),
            "match": expected is not None and actual == expected,
            "error": error,
        }
    episode_provenance = dataset.manifest.get("episode_provenance", {})
    statuses = {int(episode): str(episode_provenance.get(str(int(episode)), {}).get("quality_gate", {}).get("status", "unknown")) for episode in present}
    auto_qa_passed = sorted(episode for episode, status in statuses.items() if status == "auto_qa_passed")
    human_accepted = sorted(episode for episode, status in statuses.items() if status in {"human_accepted", "accepted"})
    pending_review = sorted(episode for episode in present if statuses.get(episode) not in {"auto_qa_passed", "human_accepted", "accepted"})
    tracking_metrics = {episode: episode_provenance.get(str(episode), {}).get("quality_gate", {}).get("metrics", {}) for episode in present}
    total_frames = sum(metrics["frames"] for metrics in episode_metrics.values())
    tactile_total = sum(int(tracking_metrics[ep].get("tactile", {}).get("frames_total", 0)) for ep in tracking_metrics)
    tactile_valid = sum(int(tracking_metrics[ep].get("tactile", {}).get("frames_present", 0)) for ep in tracking_metrics)
    failed_frame_checks = sorted(ep for ep, item in expected_frame_checks.items() if not item["match"])
    expected_calibration = front_calibration_provenance(Path(config["_repo_root"]), paths["front_calibration"])
    expected_mesh_sha = {
        name: sha256_file(_asset_config(config, paths, name)[0])
        for name in ("funnel", "target_flask")
    }
    version_mismatches: dict[str, list[str]] = {}
    for episode in present:
        provenance = episode_provenance.get(str(episode), {})
        reasons: list[str] = []
        if provenance.get("config_sha256") != config["_config_sha256"]:
            reasons.append("config_sha256")
        calibration = provenance.get("calibration", {})
        if calibration.get("profile", {}).get("sha256") != expected_calibration.get("profile", {}).get("sha256"):
            reasons.append("front_calibration_sha256")
        for name, expected_sha in expected_mesh_sha.items():
            actual_sha = provenance.get("assets", {}).get(name, {}).get("source_sha256")
            if actual_sha != expected_sha:
                reasons.append(f"{name}_mesh_sha256")
        if reasons:
            version_mismatches[str(episode)] = reasons
    report_episodes = {
        str(episode): {
            **episode_metrics.get(episode, {"frames": 0}),
            "funnel_valid_fraction": episode_metrics.get(episode, {}).get("funnel_valid", 0) / max(1, episode_metrics.get(episode, {}).get("frames", 0)),
            "target_static_prior_valid_fraction": episode_metrics.get(episode, {}).get("target_valid", 0) / max(1, episode_metrics.get(episode, {}).get("frames", 0)),
            "funnel_point_valid_fraction": episode_metrics.get(episode, {}).get("funnel_points_valid", 0) / max(1, episode_metrics.get(episode, {}).get("frames", 0) * 512),
            "target_point_valid_fraction": episode_metrics.get(episode, {}).get("target_points_valid", 0) / max(1, episode_metrics.get(episode, {}).get("frames", 0) * 512),
            "frame_check": expected_frame_checks.get(episode),
            "status": statuses.get(episode, "missing"),
            "tracking_failures": tracking_metrics.get(episode, {}).get("tracking_failures", 0),
            "recovery_count": tracking_metrics.get(episode, {}).get("recovery_count", 0),
            "max_translation_jump_m": tracking_metrics.get(episode, {}).get("max_translation_jump_m"),
            "max_rotation_jump_deg": tracking_metrics.get(episode, {}).get("max_rotation_jump_deg"),
            "tactile": tracking_metrics.get(episode, {}).get("tactile"),
        }
        for episode in episodes
    }
    result = {
        "schema_version": 2,
        "version": output_version,
        "output_version": output_version,
        "records": len(dataset),
        "episodes_requested": episodes,
        "episodes_completed": sorted(present),
        "missing_episodes": missing,
        "episodes_auto_qa_passed": auto_qa_passed,
        "episodes_human_accepted": human_accepted,
        "episodes_pending_review": pending_review,
        "episodes_frame_mismatch": failed_frame_checks,
        "version_mismatches": version_mismatches,
        "frame_coverage": expected_frame_checks,
        "aggregate": {
            "total_frames": total_frames,
            "data_missing_episode_count": len(missing),
            "version_mismatch_episode_count": len(version_mismatches),
            "pending_review_episode_count": len(pending_review),
            "funnel_valid_fraction": sum(item["funnel_valid"] for item in episode_metrics.values()) / max(1, total_frames),
            "target_static_prior_valid_fraction": sum(item["target_valid"] for item in episode_metrics.values()) / max(1, total_frames),
            "funnel_point_valid_fraction": sum(item["funnel_points_valid"] for item in episode_metrics.values()) / max(1, total_frames * 512),
            "target_point_valid_fraction": sum(item["target_points_valid"] for item in episode_metrics.values()) / max(1, total_frames * 512),
            "tactile_valid_fraction": tactile_valid / max(1, tactile_total),
            "tracking_failures": sum(int(metrics.get("tracking_failures", 0)) for metrics in tracking_metrics.values()),
            "recovery_count": sum(int(metrics.get("recovery_count", 0)) for metrics in tracking_metrics.values()),
        },
        "episodes": report_episodes,
        "calibration": dataset.manifest.get("generation", {}).get("calibration"),
        "episode_provenance_count": len(dataset.manifest.get("episode_provenance", {})),
        "future_frame_used": dataset.manifest.get("generation", {}).get("causal", {}).get("future_frame_used", False),
        "status": "pass" if not missing and not failed_frame_checks and not pending_review and not version_mismatches else "incomplete",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" and not result["future_frame_used"] else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Accept both ``cli --config CONFIG export`` and the documented
    # ``cli export --config CONFIG`` spelling.
    parser.add_argument("--config", type=Path, default=None, help="Dataset JSON/YAML config")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "mask-preview", "annotate", "mesh-build", "mesh-calibrate", "mesh-approve", "pose-preview", "export", "batch", "qa"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="Dataset JSON/YAML config")
        if name in ("mask-preview", "annotate", "pose-preview", "export", "batch", "qa"):
            cmd.add_argument("--episodes")
        if name == "pose-preview":
            cmd.add_argument("--episode", type=int, required=True)
            cmd.add_argument("--pose-run", type=Path, required=True)
            cmd.add_argument("--review-status", choices=("pending", "accepted", "rejected"), default="pending")
            cmd.add_argument("--approved-by", default=os.environ.get("USER", "unknown"))
            cmd.add_argument("--review-note", default="")
        if name == "export":
            cmd.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
            cmd.add_argument("--output-version", default=None)
        if name == "batch":
            cmd.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
            cmd.add_argument("--auto-qa", action="store_true", help="Export only episodes passing automatic QA")
            cmd.add_argument("--run-tracking", action=argparse.BooleanOptionalAction, default=True)
            cmd.add_argument("--pose-run", action="append", default=[], metavar="EPISODE=PATH")
            cmd.add_argument("--mask-root", action="append", default=[], metavar="EPISODE=PATH")
            cmd.add_argument("--output-version", default=None)
        if name == "qa":
            cmd.add_argument("--output-version", default=None)
        if name == "mesh-approve":
            cmd.add_argument("--approved-by", default=os.environ.get("USER", "unknown"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.config is None:
        raise SystemExit("--config is required (place it before or after the subcommand)")
    config, config_path, repo_root = _load_config(args.config)
    config["_repo_root"] = str(repo_root)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = _sha256_bytes(config_path.read_bytes())
    paths = _paths(config, repo_root)
    output_version = getattr(args, "output_version", None) or str(config.get("output_version", "v1_hand_scene_q2"))
    scratch = paths["scratch_root"]
    scratch.mkdir(parents=True, exist_ok=True)
    default_episodes = [int(value) for value in config.get("episodes", {}).get("all", list(range(70)))]
    selected = _episodes(getattr(args, "episodes", None), default=default_episodes)
    if args.command == "init":
        for path in (scratch / "mask_reviews", scratch / "pose_reviews", scratch / "mesh_samples", scratch / "reports"):
            path.mkdir(parents=True, exist_ok=True)
        profile = resolve_front_calibration_profile(repo_root, paths["front_calibration"])
        report = {
            "schema_version": 1,
            "dataset": config.get("dataset_id"),
            "config": str(config_path),
            "calibration": front_calibration_provenance(repo_root, profile),
            "assets": {
                name: {"path": str(_asset_config(config, paths, name)[0]), "sha256": sha256_file(_asset_config(config, paths, name)[0])}
                for name in ("funnel", "target_flask")
            },
            "approved_assets_reused": True,
            "scratch_root": str(scratch),
        }
        (scratch / "init_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "mask-preview":
        return _mask_preview(config, paths, selected, scratch)
    if args.command == "annotate":
        # Historical SAM2 annotator remains the source of truth; invoke it when
        # the external checkout is present instead of cloning its reader.
        annotator = repo_root / "tools/hoi_level1_snapshot/scripts/level1_manual_six_positions_v4.py"
        hoi_root = paths.get("hoi_root")
        if annotator.is_file() and hoi_root and hoi_root.is_dir():
            command = [sys.executable, str(annotator), "annotate", "--hoi", str(hoi_root), "--episodes", ",".join(map(str, selected))]
            completed = subprocess.run(command, cwd=repo_root, check=False)
            if completed.returncode != 0:
                return int(completed.returncode)
            return 0
        # On computer 1 the HOI checkout is intentionally absent. Write a
        # reproducible handoff so computer 4 can run the same command.
        out = scratch / "annotation_requests.json"
        payload = {"episodes": selected, "source": str(annotator), "status": "manual_action_required", "hoi_root": str(hoi_root) if hoi_root else None}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
        print(json.dumps(payload, indent=2))
        return 0
    if args.command in ("mesh-build", "mesh-calibrate"):
        print(json.dumps({"stage": args.command, "status": "skipped_approved_assets", "message": "1018_peg_all reuses assets/object_geometry/synthware_19_22; no rebuild requested."}, ensure_ascii=False))
        return 0
    if args.command == "mesh-approve":
        dimensions = json.loads(paths["dimensions_json"].read_text(encoding="utf8"))
        report = {"schema_version": 1, "approved_by": args.approved_by, "assets": {}}
        for name in ("funnel", "target_flask"):
            mesh, scale, seed = _asset_config(config, paths, name)
            expected = dimensions.get("models", {}).get(name, {}).get("sha256")
            actual = sha256_file(mesh)
            if expected and expected != actual:
                raise ValueError(f"{name} does not match dimensions.json SHA256")
            config_sha = config["assets"][name].get("sha256")
            if config_sha and config_sha != actual:
                raise ValueError(f"{name} does not match dataset config SHA256")
            report["assets"][name] = {"mesh": str(mesh), "sha256": actual, "scale_m_per_unit": scale, "sampling_seed": seed}
        out = scratch / "mesh_approval.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "pose-preview":
        return _record_pose_review(config, paths, episode=args.episode, pose_run=args.pose_run, status=args.review_status, approved_by=args.approved_by, note=args.review_note, scratch=scratch)
    if args.command == "export":
        return _export(
            config,
            paths,
            selected,
            requested_episodes=selected,
            scratch=scratch,
            resume=args.resume,
            output_version=output_version,
        )
    if args.command == "batch":
        return _batch(
            config,
            paths,
            selected,
            scratch=scratch,
            resume=args.resume,
            auto_qa=args.auto_qa,
            run_tracking=args.run_tracking,
            pose_overrides=_parse_episode_paths(args.pose_run),
            mask_overrides=_parse_episode_paths(args.mask_root),
            output_version=output_version,
        )
    if args.command == "qa":
        return _qa(config, paths, selected, output_version=output_version)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
