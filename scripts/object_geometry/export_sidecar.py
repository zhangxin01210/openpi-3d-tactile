#!/usr/bin/env python3
"""Export reviewed Level-1 A/B clouds into an independent stride-1 sidecar.

This exporter refuses visualization-only stride-3 clouds and writes a manifest
with explicit incomplete episode records. It never modifies the spatial
sidecar and refuses geometry generated with a different front calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.object_geometry.calibration import resolve_front_calibration_profile

OBJECTS = {
    "manipulated": ("funnel_xyz_base_m", "B_funnel_supported_history_xyz_base_m", "B_funnel_support_count"),
    "target": ("target_flask_xyz_base_m", "B_flask_static_history_xyz_base_m", "B_flask_support_count"),
}
POINT_COUNT = 512
FRAME_RE = re.compile(r"^(\d+)\.npz$")


def _frame_ids(root: Path) -> list[int]:
    ids = []
    for path in sorted(root.glob("*.npz")):
        match = FRAME_RE.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    return ids


def _fixed(values: np.ndarray, source: np.ndarray, age: np.ndarray, confidence: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    source = np.asarray(source, dtype=np.int8).reshape(-1)
    age = np.asarray(age, dtype=np.int16).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if not (len(values) == len(source) == len(age) == len(confidence)):
        raise ValueError("object geometry metadata lengths do not match")
    finite = np.isfinite(values).all(axis=1) & np.isfinite(confidence)
    values, source, age, confidence = values[finite], source[finite], age[finite], confidence[finite]
    order = np.arange(len(values), dtype=np.int64)
    if len(values) > POINT_COUNT:
        order = np.linspace(0, len(values) - 1, POINT_COUNT, dtype=np.int64)
    values, source, age, confidence = values[order], source[order], age[order], confidence[order]
    count = len(values)
    xyz = np.zeros((POINT_COUNT, 3), dtype=np.float32)
    valid = np.zeros((POINT_COUNT,), dtype=bool)
    out_source = np.zeros((POINT_COUNT,), dtype=np.int8)
    out_age = np.zeros((POINT_COUNT,), dtype=np.int16)
    out_confidence = np.zeros((POINT_COUNT,), dtype=np.float32)
    xyz[:count], valid[:count], out_source[:count] = values, True, source
    out_age[:count], out_confidence[:count] = age, confidence
    return {"xyz_m": xyz, "mask": valid, "source": out_source, "age": out_age, "confidence": out_confidence}


def _cloud_directory(episode_root: Path, run_name: str) -> Path:
    if run_name == "default":
        return episode_root / "cloud_frames"
    return episode_root / f"cloud_frames_{run_name}"


def _memory_directory(episode_root: Path, run_name: str) -> Path:
    if run_name == "default":
        return episode_root / "abc_v1"
    return episode_root / run_name


def _sequence_report_path(episode_root: Path, cloud_run_name: str) -> Path:
    if cloud_run_name == "default":
        return episode_root / "sequence_report.json"
    return episode_root / f"sequence_report_{cloud_run_name}.json"


def _read_json(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {label}: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        return None, f"invalid {label}: {error}"
    if not isinstance(value, dict):
        return None, f"invalid {label}: expected JSON object"
    return value, None


def _validate_reports(
    *,
    episode_root: Path,
    cloud_root: Path,
    memory_root: Path,
    cloud_run_name: str,
    expected_calibration: dict[str, Any],
    require_stride: int,
) -> str | None:
    sequence_report_path = _sequence_report_path(episode_root, cloud_run_name)
    sequence_report, error = _read_json(sequence_report_path, "sequence report")
    if error:
        return error
    if int(sequence_report.get("stride", -1)) != require_stride:
        return f"sequence report stride is not {require_stride}"
    mismatch = calibration_provenance_error(sequence_report.get("calibration"), expected_calibration)
    if mismatch:
        return f"sequence report {mismatch}"

    memory_report, error = _read_json(memory_root.parent / "report.json", "memory report")
    if error:
        return error
    if memory_report.get("source_cloud_run") != cloud_root.name:
        return "memory report source cloud run differs"
    for label, actual in (
        ("memory report", memory_report.get("calibration")),
        ("memory source", memory_report.get("source_calibration")),
    ):
        mismatch = calibration_provenance_error(actual, expected_calibration)
        if mismatch:
            return f"{label} {mismatch}"
    source_report = memory_report.get("source_sequence_report")
    if not isinstance(source_report, str) or Path(source_report).name != sequence_report_path.name:
        return "memory report source sequence report differs"
    return None


def _episode_record(
    episode_root: Path,
    episode: int,
    *,
    require_stride: int,
    cloud_run_name: str,
    memory_run_name: str,
    expected_calibration: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    cloud_root = _cloud_directory(episode_root, cloud_run_name)
    memory_root = _memory_directory(episode_root, memory_run_name) / "frames"
    report_error = _validate_reports(
        episode_root=episode_root,
        cloud_root=cloud_root,
        memory_root=memory_root,
        cloud_run_name=cloud_run_name,
        expected_calibration=expected_calibration,
        require_stride=require_stride,
    )
    if report_error:
        return None, report_error
    cloud_ids, memory_ids = _frame_ids(cloud_root), _frame_ids(memory_root)
    expected = list(range(max(cloud_ids) + 1)) if cloud_ids else []
    if not cloud_ids:
        return None, "missing cloud_frames"
    if require_stride == 1 and cloud_ids != expected:
        return None, f"cloud_frames are not stride=1 ({len(cloud_ids)} files, last={cloud_ids[-1]})"
    if cloud_ids != memory_ids:
        return None, "cloud/memory frame IDs differ"
    records = []
    for frame in cloud_ids:
        with np.load(cloud_root / f"{frame:05d}.npz", allow_pickle=False) as current, np.load(
            memory_root / f"{frame:05d}.npz", allow_pickle=False
        ) as history:
            arrays: dict[str, np.ndarray] = {"episode_index": np.int64(episode), "frame_index": np.int64(frame)}
            for name, (current_key, history_key, support_key) in OBJECTS.items():
                current_xyz = np.asarray(current[current_key], dtype=np.float32)
                history_xyz = np.asarray(history[history_key], dtype=np.float32)
                support = np.asarray(history[support_key], dtype=np.float32)
                current_confidence = np.ones((len(current_xyz),), dtype=np.float32)
                history_confidence = np.clip(support / max(1.0, float(np.max(support, initial=1.0))), 0.0, 1.0)
                combined = _fixed(
                    np.concatenate([current_xyz, history_xyz], axis=0),
                    np.concatenate([np.zeros(len(current_xyz), np.int8), np.ones(len(history_xyz), np.int8)]),
                    np.concatenate([np.zeros(len(current_xyz), np.int16), np.ones(len(history_xyz), np.int16)]),
                    np.concatenate([current_confidence, history_confidence]),
                )
                for field, value in combined.items():
                    arrays[f"{name}_{field}"] = value
            records.append(arrays)
    return {"episode": episode, "records": records}, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoi-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="Dataset root containing object_geometry/<version>")
    parser.add_argument("--episodes", default="0-69")
    parser.add_argument("--require-stride", type=int, default=1)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--front-calibration-profile", type=Path, required=True)
    parser.add_argument("--cloud-run-name", default="front_candidate_stride1")
    parser.add_argument("--memory-run-name", default="abc_v1_front_candidate_stride1")
    parser.add_argument("--version", default="v1_front_candidate")
    parser.add_argument("--generation-commit", default="unknown")
    args = parser.parse_args()
    if args.require_stride != 1:
        parser.error("formal object geometry export only accepts --require-stride=1")
    if not args.version.replace("_", "").replace("-", "").isalnum():
        parser.error("--version must be alphanumeric, '_' or '-'")
    for flag, run_name in (("--cloud-run-name", args.cloud_run_name), ("--memory-run-name", args.memory_run_name)):
        if run_name != "default" and not run_name.replace("_", "").replace("-", "").isalnum():
            parser.error(f"{flag} must be alphanumeric, '_' or '-'")
    profile = resolve_front_calibration_profile(args.repo_root, args.front_calibration_profile)
    expected_calibration = front_calibration_provenance(args.repo_root, profile)
    episodes = []
    for token in args.episodes.split(","):
        if "-" in token:
            start, stop = map(int, token.split("-", 1))
            episodes.extend(range(start, stop + 1))
        else:
            episodes.append(int(token))
    root = args.output_root / "object_geometry" / args.version
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing sidecar: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    failures = []
    for episode in dict.fromkeys(episodes):
        result, error = _episode_record(
            args.hoi_output_root / f"level1_sequence_e{episode}",
            episode,
            require_stride=args.require_stride,
            cloud_run_name=args.cloud_run_name,
            memory_run_name=args.memory_run_name,
            expected_calibration=expected_calibration,
        )
        if error:
            failures.append({"episode_index": episode, "error": error})
            continue
        shard_name = f"shard_e{episode:03d}"
        shard = root / shard_name
        shard.mkdir(parents=True, exist_ok=True)
        fields = {"episode_index": [], "frame_index": []}
        fields.update({f"{name}_{field}": [] for name in OBJECTS for field in ("xyz_m", "mask", "source", "age", "confidence")})
        for row, record in enumerate(result["records"]):
            for field, value in record.items():
                fields[field].append(value)
            manifest_records.append({"episode_index": episode, "frame_index": int(record["frame_index"]), "shard": shard_name, "row": row})
        for field, values in fields.items():
            np.save(shard / f"{field}.npy", np.asarray(values))
    status = "complete" if not failures and manifest_records else "incomplete"
    manifest = {
        "status": status,
        "schema_version": 1,
        "generation_commit": args.generation_commit,
        "source_hoi_output_root": str(args.hoi_output_root),
        "source_runs": {
            "cloud_run_name": args.cloud_run_name,
            "memory_run_name": args.memory_run_name,
        },
        "calibration": expected_calibration,
        "object_geometry_contract": {
            "version": args.version, "point_count": POINT_COUNT, "units": "meters",
            "source_semantics": {"0": "current observation A", "1": "causal historical geometry B"},
            "confidence_semantics": "deterministic validity/support confidence; not learned probability",
            "history_age_semantics": "0=current; 1=historical marker because prototype B lacks per-point last-seen age",
        },
        "records": manifest_records,
        "failed_episodes": failures,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps({"status": status, "records": len(manifest_records), "failed_episodes": failures}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
