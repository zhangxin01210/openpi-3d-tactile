#!/usr/bin/env python3
"""Package online replay pose rows in the format consumed by the existing 3D QA viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-profile", type=Path, default=Path("configs/ur7e_xhand/front_calibration_candidate.json"))
    parser.add_argument("--scale-manifest", type=Path, default=Path("data/pose_eval/hand_scene_metric_mesh_scales.json"))
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    online = args.online_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty pose run: {out}")
    rows_path = online / "manifest.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)

    sys.path.insert(0, str(repo / "src"))
    from openpi.object_geometry.calibration import front_calibration_provenance

    profile = args.calibration_profile
    if not profile.is_absolute():
        profile = repo / profile
    scales_path = args.scale_manifest
    if not scales_path.is_absolute():
        scales_path = repo / scales_path
    run_metadata = json.loads((online / "run_metadata.json").read_text(encoding="utf-8"))
    static_anchor_record = run_metadata.get("static_target_anchor")
    static_anchor = None
    if isinstance(static_anchor_record, dict) and "T_base_link_object" in static_anchor_record:
        static_anchor = static_anchor_record["T_base_link_object"]
    static_prior_start = int(run_metadata.get("training_geometry", {}).get("static_prior_start_frame", 1))
    conversion_files = list((online / "rgbtrack_meshes").glob("*/mesh_conversion.json"))
    conversions = {}
    for path in conversion_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        conversions[value["object_name"]] = value
    if set(conversions) != {"funnel", "target_flask"}:
        raise ValueError(f"Expected conversion metadata for funnel and target_flask, got {sorted(conversions)}")

    output_rows = {name: [] for name in conversions}
    frame_ids = []
    for line_no, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        frame = json.loads(line)
        frame_id = int(frame["frame_index"])
        frame_ids.append(frame_id)
        for name in output_rows:
            source = frame["objects"][name]
            matrix = source["T_base_link_object"]
            valid = bool(source["valid"])
            confidence = float(source["confidence"])
            status = source["status"]
            pose_source = source["pose_source"]
            observed_pose_valid = bool(source["observed_pose_valid"])
            static_prior_applied = name == "target_flask" and static_anchor is not None and frame_id >= static_prior_start
            if static_prior_applied:
                matrix = static_anchor
                valid = True
                confidence = 1.0
                status = "flask_static_anchor_prior"
                pose_source = "static_geometry_prior"
                observed_pose_valid = False
            raw_translation_delta_m = None
            if static_prior_applied and bool(source["valid"]):
                raw_pose = np.asarray(source["T_base_link_object"], dtype=np.float64)
                raw_translation_delta_m = float(np.linalg.norm(raw_pose[:3, 3] - np.asarray(static_anchor, dtype=np.float64)[:3, 3]))
            output_rows[name].append({
                "object_name": name,
                "frame_index": frame_id,
                "T_base_link_object": matrix,
                "valid": valid,
                "confidence": confidence,
                "status": status,
                "pose_source": pose_source,
                "observed_pose_valid": observed_pose_valid,
                "static_prior_applied": static_prior_applied,
                "static_prior_provenance": static_anchor_record.get("provenance") if static_prior_applied else None,
                "rgbtrack_T_base_link_object": source["T_base_link_object"],
                "rgbtrack_valid": bool(source["valid"]),
                "rgbtrack_status": source["status"],
                "rgbtrack_translation_delta_from_static_prior_m": raw_translation_delta_m,
            })
    if frame_ids != list(range(len(frame_ids))):
        raise ValueError("Online pose manifest must contain continuous frame IDs from zero")

    out.mkdir(parents=True)
    for name, rows in output_rows.items():
        with (out / f"{name}_poses.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {
        "schema_version": 1,
        "episode": args.episode,
        "objects": ["funnel", "target_flask"],
        "mode": "causal_online_depth",
        "requested_init_frames": {"funnel": 0, "target_flask": 0},
        "dataset": str(online / "frames"),
        "mask_root": str(online / "masks"),
        "calibration": front_calibration_provenance(repo, profile),
        "rgbtrack_commit": run_metadata.get("rgbtrack_submodule_revision"),
        "scale_manifest": str(scales_path.resolve()),
        "weights_root": str(repo / "checkpoints/rgbtrack"),
        "mesh_conversions": conversions,
        "causal_online": True,
        "static_target_prior": {
            "applied_from_frame": static_prior_start if static_anchor is not None else None,
            "provenance": static_anchor_record.get("provenance") if static_anchor_record else None,
            "anchor": static_anchor,
        },
        "future_frames_used": False,
        "source_online_manifest": str(rows_path),
        "source_online_num_frames": len(frame_ids),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "frames": len(frame_ids), "pose_run": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
