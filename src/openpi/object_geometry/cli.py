"""Unified Object Geometry production CLI.

The command is an orchestration boundary around the already reviewed SAM2 and
RGBTrack scripts.  It never chooses a newest output directory: pose runs are
registered explicitly by ``pose-preview --review-status accepted`` and export
consumes those small approval records.
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
from openpi.object_geometry.production import build_episode_records
from openpi.object_geometry.production import load_mesh_surface_points
from openpi.object_geometry.production import read_pose_run
from openpi.object_geometry.production import sha256_file
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
    for name in ("hoi_root", "mask_root_template", "frames_root_template", "tactile_root", "hand_offset_root", "approved_pose_root"):
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


def _mask_preview(config: dict[str, Any], paths: dict[str, Path], episodes: list[int], scratch: Path) -> int:
    representatives = [int(value) for value in config.get("representative_frames", [0, 120, 200, 300, 400, 440, 460])]
    reports = []
    for episode in episodes:
        mask_root = _format_template(paths.get("mask_root_template", Path("")), episode)
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
    if status == "accepted":
        (out / "approved_pose.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(review, indent=2, ensure_ascii=False))
    return 0


def _load_approval(scratch: Path, episode: int) -> dict[str, Any]:
    path = scratch / "pose_reviews" / f"e{episode:03d}" / "approved_pose.json"
    if not path.is_file():
        raise FileNotFoundError(f"Episode {episode} has no accepted pose review: {path}")
    review = json.loads(path.read_text(encoding="utf8"))
    if review.get("status") != "accepted":
        raise ValueError(f"Pose review is not accepted: {path}")
    pose_run = Path(review["pose_run"]).expanduser().resolve()
    if sha256_file(pose_run / "manifest.json") != review.get("pose_manifest_sha256"):
        raise ValueError(f"Pose run changed after review: {pose_run}")
    return review


def _export(config: dict[str, Any], paths: dict[str, Path], episodes: list[int], *, scratch: Path, resume: bool) -> int:
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
    for episode in episodes:
        review = _load_approval(scratch, episode)
        run = read_pose_run(review["pose_run"], episode=episode)
        state = CausalHandSceneState(
            tactile_root=paths.get("tactile_root"),
            mask_root=_format_template(paths["mask_root_template"], episode) if paths.get("mask_root_template") else None,
            hand_offset_root=paths.get("hand_offset_root"),
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
            "pose_pipeline": {
                "pose_run": review["pose_run"],
                "pose_manifest_sha256": review["pose_manifest_sha256"],
                "review_status": review["status"],
                "reviewed_by": review.get("approved_by"),
            },
            "causal": {
                "future_frame_used": False,
                "state_generator": "CausalHandSceneState.v1",
                "tactile_root": str(paths.get("tactile_root")) if paths.get("tactile_root") else None,
            },
        }
        results.append(
            write_episode_sidecar(
                dataset_root=paths["dataset_root"],
                episode=episode,
                records=records,
                manifest_context=context,
                requested_episodes=episodes,
                resume=resume,
            )
        )
    print(json.dumps({"stage": "export", "results": results}, indent=2, ensure_ascii=False))
    return 0


def _qa(config: dict[str, Any], paths: dict[str, Path], episodes: list[int]) -> int:
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

    dataset = ObjectGeometryDerivedDataset(paths["dataset_root"], version="v1_hand_scene")
    missing = []
    present = set()
    for episode, frame in dataset._records:  # noqa: SLF001 - QA owns the coverage boundary.
        present.add(int(episode))
        sample = dataset.get(episode_index=episode, frame_index=frame)
        for name in ("manipulated", "target"):
            xyz = getattr(sample, f"{name}_xyz_m")
            if xyz.shape != (512, 3) or not np.isfinite(xyz).all():
                raise ValueError(f"Invalid {name} cloud at e{episode} f{frame}")
    missing = sorted(set(episodes) - present)
    result = {
        "version": "v1_hand_scene",
        "records": len(dataset),
        "present_episodes": sorted(present),
        "missing_episodes": missing,
        "calibration": dataset.manifest.get("generation", {}).get("calibration"),
        "episode_provenance_count": len(dataset.manifest.get("episode_provenance", {})),
        "future_frame_used": dataset.manifest.get("generation", {}).get("causal", {}).get("future_frame_used", False),
        "status": "pass" if not missing else "incomplete",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not missing and not result["future_frame_used"] else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Accept both ``cli --config CONFIG export`` and the documented
    # ``cli export --config CONFIG`` spelling.
    parser.add_argument("--config", type=Path, default=None, help="Dataset JSON/YAML config")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "mask-preview", "annotate", "mesh-build", "mesh-calibrate", "mesh-approve", "pose-preview", "export", "qa"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="Dataset JSON/YAML config")
        if name in ("mask-preview", "annotate", "pose-preview", "export", "qa"):
            cmd.add_argument("--episodes")
        if name == "pose-preview":
            cmd.add_argument("--episode", type=int, required=True)
            cmd.add_argument("--pose-run", type=Path, required=True)
            cmd.add_argument("--review-status", choices=("pending", "accepted", "rejected"), default="pending")
            cmd.add_argument("--approved-by", default=os.environ.get("USER", "unknown"))
            cmd.add_argument("--review-note", default="")
        if name == "export":
            cmd.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
        return _export(config, paths, selected, scratch=scratch, resume=args.resume)
    if args.command == "qa":
        return _qa(config, paths, selected)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
