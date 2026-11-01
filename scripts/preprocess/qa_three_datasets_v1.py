#!/usr/bin/env python3
"""Read-only exhaustive pretraining QA for the three UR7e/XHand datasets.

Runs against existing on-disk data, no RGB/depth decoding, no GPU, no writes to
input datasets. Checks all raw / spatial / object_geometry / hand_keypoints
frame identities and timestamps, full-shard value contracts and provenance.

Usage from repository root:
    PYTHONPATH=src .venv/bin/python scripts/preprocess/qa_three_datasets_v1.py

A PASS is structural/integrity acceptance, not proof of metric 6D accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pyarrow.parquet as pq

from openpi.spatial_dataset.derived import SpatialDerivedDataset
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.spatial_dataset.source import RawSpatialDataset

EXPECTED = {
    "1018_peg_all": (70, 32343, "funnel", "target_flask"),
    "1028_press_button": (100, 26485, "button", "box"),
    "1030_yellow_socket": (111, 34963, "red", "yellow"),
}
GEOMETRY_FIELDS = tuple(f"{role}_{suffix}" for role in ("manipulated", "target")
                        for suffix in ("xyz_m", "mask", "source", "age", "confidence"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(cond: bool, text: str) -> None:
    if not cond:
        raise AssertionError(text)


def finite_block(array, *, name: str, context: str, max_rows: int = 16) -> tuple[float, float]:
    """Scan all array values without allocating another whole-episode tensor."""
    lo, hi = float("inf"), float("-inf")
    for start in range(0, len(array), max_rows):
        block = np.asarray(array[start : start + max_rows])
        if block.size == 0:
            continue
        require(bool(np.isfinite(block).all()), f"{context}: nonfinite {name}, rows {start}..{start+len(block)-1}")
        lo = min(lo, float(np.min(block)))
        hi = max(hi, float(np.max(block)))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("output/preprocess/training_readiness/qa_three_datasets_v1"))
    ap.add_argument("--fast", action="store_true", help="Only inspect sampled numeric rows; index/timestamp checks remain exhaustive")
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "mode": "fast" if args.fast else "full_numeric_scan",
        "expected_total_episodes": 281, "expected_total_frames": 93791,
        "datasets": {}, "failures": [], "warnings": [],
        "notes": [
            "Structural/coordinate-contract checks do not establish RGBTrack metric pose accuracy.",
            "Old 1018 object_geometry/v1 is value-preserving migration from legacy q2, not recalibrated poses.",
            "Invalid object masks are retained; invalid frame counts are not evidence of pose accuracy.",
            "Force units remain dataset_native, not newtons.",
            "This is not an OpenPI training DataLoader/action-horizon smoke test.",
        ],
    }
    reference_spatial = None
    reference_ids = None
    reference_names = None
    reference_urdf_sha = None
    reference_mapping_sha = None
    grand_frames = 0
    grand_episodes = 0
    numeric_columns = ("visual_xyz_m", "tactile_xyz_m", "tactile_force_base", "tactile_force_norm")

    for name, (exp_eps, exp_frames, dynamic, static) in EXPECTED.items():
        root = repo / "data" / name
        entry = {"status": "running", "expected_episodes": exp_eps, "expected_frames": exp_frames,
                 "roles": {"manipulated": dynamic, "target": static}, "warnings": [],
                 "object_valid_frames": {"manipulated": 0, "target": 0},
                 "object_invalid_frames": {"manipulated": 0, "target": 0},
                 "checks": {"raw_spatial_object_hand_frame_join": False, "raw_spatial_hand_timestamp_join": False,
                            "full_spatial_numeric": not args.fast, "full_object_numeric": not args.fast},
                 "episodes": []}
        report["datasets"][name] = entry
        print(f"\n[DATASET] {name}", flush=True)
        try:
            raw_ids = tuple(RawSpatialDataset.discover_episode_ids(root))
            spatial = SpatialDerivedDataset(root, version="v1")
            objects = ObjectGeometryDerivedDataset(root, version="v1")
            hand_root = root / "hand_keypoints" / "v1"
            hand_manifest = json.loads((hand_root / "manifest.json").read_text(encoding="utf-8"))
            require(hand_manifest["status"] == "complete", f"{name}: hand manifest incomplete")
            hand_recs = hand_manifest["episodes"]
            hand_by_ep = {int(r["episode_index"]): r for r in hand_recs}
            require(len(hand_by_ep) == len(hand_recs), f"{name}: duplicate hand episode")

            require(len(raw_ids) == exp_eps, f"{name}: raw episodes={len(raw_ids)} expected={exp_eps}")
            require(tuple(spatial.episode_ids) == raw_ids, f"{name}: raw/spatial episode sets differ")
            require(tuple(objects.episode_ids) == raw_ids, f"{name}: raw/object episode sets differ")
            require(tuple(sorted(hand_by_ep)) == tuple(sorted(raw_ids)), f"{name}: raw/hand episode sets differ")
            require(len(spatial) == len(objects) == exp_frames, f"{name}: sidecar count mismatch")
            require(hand_manifest["total_frames"] == exp_frames, f"{name}: hand total_frames mismatch")
            require(int(hand_manifest["keypoint_count"]) == 18, f"{name}: hand count not 18")
            require(hand_manifest["coordinate_frame"] == "base_link" and hand_manifest["unit"] == "m", f"{name}: hand frame/unit")
            require(spatial.coordinate_frame == "base_link" and spatial.xyz_unit == "m", f"{name}: spatial frame/unit")
            require(spatial.force_unit == "dataset_native", f"{name}: spatial force unit changed")
            require(int(spatial.visual_points_per_frame) == 4096 and int(spatial.tactile_points_per_frame) == 600,
                    f"{name}: spatial point counts differ")
            require(int(objects.manifest["object_geometry_contract"]["point_count"]) == 512,
                    f"{name}: object points differ")
            require(int(len(spatial.finger_id)) == 600 and len(spatial.taxel_id) == 600,
                    f"{name}: finger/taxel topology wrong")
            require(tuple(spatial.camera_roles) == ("front",), f"{name}: expected front-only spatial")

            if reference_spatial is None:
                reference_spatial = spatial.manifest["preprocess_config"]
                reference_ids = (np.array(spatial.finger_id), np.array(spatial.taxel_id))
            else:
                require(spatial.manifest["preprocess_config"] == reference_spatial,
                        f"{name}: spatial preprocessing differs from 1018")
                require(np.array_equal(spatial.finger_id, reference_ids[0]) and
                        np.array_equal(spatial.taxel_id, reference_ids[1]),
                        f"{name}: tactile finger/taxel IDs differ")
            urdf_sha, mapping_sha = hand_manifest["urdf_sha256"], hand_manifest["state_mapping_sha256"]
            names = list(hand_manifest["names"])
            require(len(names) == 18 and len(set(names)) == 18, f"{name}: keypoint names malformed")
            if reference_names is None:
                reference_names, reference_urdf_sha, reference_mapping_sha = names, urdf_sha, mapping_sha
            else:
                require(names == reference_names and urdf_sha == reference_urdf_sha and
                        mapping_sha == reference_mapping_sha, f"{name}: FK anchor definition/URDF differs")

            # Must not have been changed after FK sidecar generation.
            require(sha256(repo / "configs/ur7e_xhand/ur7e_xhand_verified.urdf") == urdf_sha,
                    f"{name}: current URDF differs from generated keypoints")
            require(sha256(repo / "configs/ur7e_xhand/state_mapping.csv") == mapping_sha,
                    f"{name}: current state mapping differs from generated keypoints")

            # Validate object manifest record ordering, and detect duplicate keys.
            geo_by_ep = defaultdict(list)
            for r in objects.manifest["records"]:
                geo_by_ep[int(r["episode_index"])].append(r)
            require(sum(map(len, geo_by_ep.values())) == exp_frames, f"{name}: bad object record count")
            total_frames = 0
            for e, ep in enumerate(raw_ids):
                raw = RawSpatialDataset(root, episode=int(ep))
                # Lightweight raw frame indices and timestamps ONLY; avoid RGB/depth/1972-D state.
                tab = pq.read_table(raw.parquet_path, columns=["frame_index", "timestamp"])
                frames_raw = np.asarray(tab["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                times_raw = np.asarray(tab["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
                n = int(frames_raw.size)
                ctx = f"{name}/e{int(ep):05d}"
                require(n > 0 and np.array_equal(frames_raw, np.arange(n)), f"{ctx}: raw frames not 0..n-1")
                require(np.isfinite(times_raw).all(), f"{ctx}: bad raw timestamps")
                require(int(spatial.episode_length(ep)) == n, f"{ctx}: spatial length mismatch")
                require(tuple(objects.frame_ids(ep)) == tuple(range(n)), f"{ctx}: object frame ids mismatch")
                require(len(geo_by_ep[ep]) == n, f"{ctx}: missing object manifest records")

                hand_rec = hand_by_ep[ep]
                require(int(hand_rec["frames"]) == n, f"{ctx}: hand manifest count wrong")
                hand_path = hand_root / hand_rec["file"]
                require(hand_path.is_file(), f"{ctx}: hand NPZ absent")
                require(sha256(hand_path) == hand_rec["sha256"], f"{ctx}: hand NPZ hash mismatch")
                with np.load(hand_path, allow_pickle=False) as z:
                    k_frames = np.asarray(z["frame_index"], dtype=np.int64)
                    k_times = np.asarray(z["timestamp_s"], dtype=np.float64)
                    k_xyz = np.asarray(z["xyz_m"], dtype=np.float32)
                    require(k_xyz.shape == (n, 18, 3), f"{ctx}: keypoint shape {k_xyz.shape}")
                    require(np.array_equal(k_frames, frames_raw), f"{ctx}: hand frame indices differ")
                    require(np.array_equal(z["names"], np.asarray(reference_names)), f"{ctx}: hand names differ")
                    require(np.isfinite(k_xyz).all(), f"{ctx}: nonfinite keypoints")
                    require(np.allclose(k_times, times_raw, atol=1e-6, rtol=0), f"{ctx}: hand timestamp mismatch")
                    # Track obvious FK spikes without assuming a specific task motion.
                    if n > 1:
                        jumps = np.linalg.norm(np.diff(k_xyz, axis=0), axis=-1)
                        maximum = float(np.max(jumps))
                        if maximum > 0.30:
                            entry["warnings"].append(f"{ctx}: suspicious >30 cm consecutive-frame keypoint jump ({maximum:.3f}m)")

                all_sf, all_st = [], []
                for rec in spatial._episode_shards[ep]:  # noqa: SLF001 (read-only QA on frozen sidecar)
                    a = spatial._load_shard_arrays(rec)  # noqa: SLF001
                    sf = np.asarray(a["frame_index"], dtype=np.int64)
                    st = np.asarray(a["timestamp_s"], dtype=np.float64)
                    all_sf.append(sf)
                    all_st.append(st)
                    if not args.fast:
                        for field in numeric_columns:
                            finite_block(a[field], name=field, context=ctx)
                        force = a["tactile_force_base"]
                        fnorm = a["tactile_force_norm"]
                        for k in range(0, len(force), 16):
                            direct = np.linalg.norm(np.asarray(force[k:k+16], dtype=np.float64), axis=-1)
                            stored = np.asarray(fnorm[k:k+16], dtype=np.float64)
                            require(np.allclose(direct, stored, rtol=1e-4, atol=1e-3),
                                    f"{ctx}: tactile_force_norm disagrees with vector norm")
                    else:
                        for field in numeric_columns:
                            require(np.isfinite(np.asarray(a[field][:min(len(sf),2)])).all(),
                                    f"{ctx}: sample nonfinite {field}")
                frames_sp = np.concatenate(all_sf)
                times_sp = np.concatenate(all_st)
                require(np.array_equal(frames_sp, frames_raw), f"{ctx}: spatial frame ids differ")
                require(np.allclose(times_sp, times_raw, atol=1e-6, rtol=0),
                        f"{ctx}: spatial timestamp mismatch")

                per_ep_valid = {"manipulated": 0, "target": 0}
                per_ep_invalid = {"manipulated": 0, "target": 0}
                by_shard = defaultdict(list)
                for r in geo_by_ep[ep]:
                    by_shard[str(r["shard"])].append(int(r["row"]))
                object_found = []
                for shard_name, rows in by_shard.items():
                    arr = objects._load_shard(shard_name)  # noqa: SLF001
                    indices = np.asarray(rows, dtype=np.int64)
                    ids = np.asarray(arr["frame_index"][indices], dtype=np.int64)
                    eps = np.asarray(arr["episode_index"][indices], dtype=np.int64)
                    require(np.all(eps == ep), f"{ctx}: object shard episode mismatch")
                    object_found.extend(map(int, ids))
                    require(len(set(rows)) == len(rows), f"{ctx}: duplicate object shard row")
                    for role in ("manipulated", "target"):
                        xyz = arr[f"{role}_xyz_m"]
                        mask = arr[f"{role}_mask"]
                        confidence = arr[f"{role}_confidence"]
                        source = arr[f"{role}_source"]
                        age = arr[f"{role}_age"]
                        require(xyz.shape[1:] == (512,3) and mask.shape[1:] == (512,),
                                f"{ctx}: bad object geometry shape {role}")
                        require(np.issubdtype(mask.dtype, np.bool_), f"{ctx}: non-boolean {role} mask")
                        if not args.fast:
                            for field, array in (("xyz",xyz),("confidence",confidence),("age",age)):
                                finite_block(array, name=f"{role}_{field}", context=ctx)
                            require(np.isin(source, (0,1)).all(), f"{ctx}: invalid source tag {role}")
                            require((confidence[mask] >= 0).all(), f"{ctx}: negative confidence on valid {role}")
                        valid_frames = np.any(mask[indices], axis=1)
                        per_ep_valid[role] += int(np.count_nonzero(valid_frames))
                        per_ep_invalid[role] += int(len(valid_frames) - np.count_nonzero(valid_frames))
                require(sorted(object_found) == frames_raw.tolist(), f"{ctx}: object frame ids mismatch")
                for role in per_ep_valid:
                    entry["object_valid_frames"][role] += per_ep_valid[role]
                    entry["object_invalid_frames"][role] += per_ep_invalid[role]
                total_frames += n
                entry["episodes"].append({"episode": int(ep), "frames": n,
                                          "manipulated_valid_frames":per_ep_valid["manipulated"],
                                          "target_valid_frames":per_ep_valid["target"]})
                print(f"  [OK] e{int(ep):05d} frames={n} manipulated_valid={per_ep_valid['manipulated']} target_valid={per_ep_valid['target']}", flush=True)
            require(total_frames == exp_frames, f"{name}: raw total {total_frames} != expected {exp_frames}")
            entry["actual_episodes"], entry["actual_frames"] = len(raw_ids), total_frames
            entry["checks"]["raw_spatial_object_hand_frame_join"] = True
            entry["checks"]["raw_spatial_hand_timestamp_join"] = True
            entry["status"] = "PASS"
            grand_frames += total_frames
            grand_episodes += len(raw_ids)
            print(f"[PASS] {name}: {len(raw_ids)} episodes, {total_frames} frames", flush=True)
        except Exception as exc:
            entry["status"] = "FAIL"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["failures"].append(f"{name}: {entry['error']}")
            print(f"[FAIL] {name}: {entry['error']}", file=sys.stderr, flush=True)
            traceback.print_exc(limit=4)
        finally:
            report["warnings"].extend(entry["warnings"])

    report["actual_total_episodes"] = grand_episodes
    report["actual_total_frames"] = grand_frames
    report["status"] = "PASS" if (not report["failures"] and grand_episodes == 281 and grand_frames == 93791) else "FAIL"
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    rows = ["# Three-dataset v1 training-input QA", "", f"Overall: **{report['status']}**", "",
            f"Mode: `{report['mode']}`", "",
            "| Dataset | Episodes | Frames | Manipulated invalid frames | Target invalid frames | Status |",
            "|---|---:|---:|---:|---:|---|" ]
    for name, entry in report["datasets"].items():
        rows.append(f"| {name} | {entry.get('actual_episodes', '-')} | {entry.get('actual_frames', '-')} | "
                    f"{entry['object_invalid_frames']['manipulated']} | {entry['object_invalid_frames']['target']} | {entry['status']} |")
    rows += ["", f"Total verified: **{grand_episodes} episodes / {grand_frames} frames**", "",
             "PASS means structural/data integrity and per-frame indexing acceptance, not metric geometry precision."]
    if report["failures"]:
        rows += ["", "## Failures", *[f"- {x}" for x in report["failures"]]]
    if report["warnings"]:
        rows += ["", "## Warnings", *[f"- {x}" for x in report["warnings"][:60]]]
    (out / "report.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    print(f"\n[FINAL] {report['status']} episodes={grand_episodes}/281 frames={grand_frames}/93791 warnings={len(report['warnings'])}")
    print("[REPORT]", out / "report.md")
    print("[JSON]", out / "report.json")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
