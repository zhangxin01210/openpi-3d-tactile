#!/usr/bin/env python3
"""Convert legacy 1018 q2 geometry to universal two-object v1 without RGBTrack.

Default is a read-only audit; --pilot writes ONLY v1_legacy_probe for episode 23.
--all --acknowledge-legacy-pose writes v1; only after separate geometric review.
This is a byte-for-byte value-preserving geometry conversion, NOT recalibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

ROLE_FIELDS = tuple(f"{role}_{suffix}" for role in ("manipulated", "target")
                    for suffix in ("xyz_m", "mask", "source", "age", "confidence"))
ALL_FIELDS = ("episode_index", "frame_index", *ROLE_FIELDS)
EXPECTED_FRAMES = 32343
EXPECTED_EPISODES = 70
REFERENCE_DATASET = "1028_press_button"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def check_samples(dataset: ObjectGeometryDerivedDataset, by_ep: dict[int, list[int]]) -> dict:
    rows = []
    groups = [0, 23, 38, 49, 58, 65, 69]
    for ep in groups:
        ids = by_ep[ep]
        for frame in sorted(set([ids[0], ids[len(ids)//2], ids[-1]])):
            s = dataset.get(episode_index=ep, frame_index=frame)
            result = {"episode": ep, "frame": frame}
            for role in ("manipulated", "target"):
                xyz = np.asarray(getattr(s, f"{role}_xyz_m"))
                mask = np.asarray(getattr(s, f"{role}_mask"), dtype=bool)
                source = np.asarray(getattr(s, f"{role}_source"))
                assert xyz.shape == (512, 3), (role, xyz.shape)
                assert mask.shape == (512,), (role, mask.shape)
                assert np.isfinite(xyz).all(), (ep, frame, role)
                assert np.isin(source, (0,1)).all(), (ep,frame,role)
                result[f"{role}_valid"] = int(mask.sum())
                result[f"{role}_centroid_m"] = (xyz[mask].mean(axis=0).round(5).tolist()
                                                  if mask.any() else None)
            rows.append(result)
    return {"checked_frames":len(rows), "examples":rows}


def prepare(repo: Path):
    src_root = repo / "data/1018_peg_all"
    old = src_root / "object_geometry/v1_hand_scene_q2/manifest.json"
    ref = repo / f"data/{REFERENCE_DATASET}/object_geometry/v1/manifest.json"
    if not old.is_file() or not ref.is_file():
        raise FileNotFoundError(f"Required source/reference manifests: {old} {ref}")
    old_m, ref_m = read(old), read(ref)
    if old_m.get("status") != "complete" or ref_m.get("status") != "complete":
        raise ValueError("Both source and reference object_geometry manifests must be complete")
    src = ObjectGeometryDerivedDataset(src_root, version="v1_hand_scene_q2")
    if len(src) != EXPECTED_FRAMES or len(src.episode_ids) != EXPECTED_EPISODES:
        raise ValueError(f"Old geometry length/episodes {len(src)}/{len(src.episode_ids)} != 32343/70")
    if tuple(src.episode_ids) != tuple(range(EXPECTED_EPISODES)):
        raise ValueError(f"Old episode ids are unexpected: {src.episode_ids}")
    by_ep = {ep: list(src.frame_ids(ep)) for ep in src.episode_ids}
    if sum(map(len, by_ep.values())) != EXPECTED_FRAMES:
        raise AssertionError("Repeated frame keys or missing old frames")
    if any(ids != list(range(len(ids))) for ids in by_ep.values()):
        raise ValueError("Old frame indices are not 0..n-1. Stop and inspect before copying.")
    contract = ref_m.get("object_geometry_contract", {})
    if int(contract.get("point_count", -1)) != 512:
        raise ValueError("1028 reference universal contract does not specify 512 points")
    probe = src.get(episode_index=23, frame_index=0)
    for field in ROLE_FIELDS:
        if not hasattr(probe, field):
            raise ValueError(f"Source reader lacks {field}")
    report = {
        "status":"AUDIT_ONLY", "source_manifest_sha256": sha256(old),
        "reference_manifest_sha256": sha256(ref),
        "legacy_manifest_keys":sorted(old_m), "new_manifest_keys":sorted(ref_m),
        "legacy_geometry_contract":old_m.get("object_geometry_contract"),
        "universal_geometry_contract":contract,
        "legacy_calibration":old_m.get("calibration"),
        "new_reference_calibration":ref_m.get("calibration"),
        "legacy_hand_scene_contract_present":"hand_scene_contract" in old_m,
        "old_record_count":len(src), "old_episodes":len(src.episode_ids),
        "old_frame_counts":{str(ep):len(ids) for ep,ids in by_ep.items()},
        "warning": "Old q2 XYZ and pose conventions are preserved unchanged, NOT corrected for updated T_color_depth. Geometry reuse requires visual/RGB QA.",
        "sample_check": check_samples(src, by_ep),
    }
    return src_root, old_m, ref_m, src, by_ep, report, old, ref


def values_equal(a, b, field: str) -> bool:
    return np.array_equal(np.asarray(getattr(a,field)), np.asarray(getattr(b,field)))


def export(repo: Path, *, episodes: list[int], version: str, acknowledge: bool) -> Path:
    src_root, old_m, ref_m, source, by_ep, report, old_path, ref_path = prepare(repo)
    if version == "v1" and not acknowledge:
        raise ValueError("Formal v1 needs --acknowledge-legacy-pose AFTER visual review")
    if version == "v1" and episodes != list(range(EXPECTED_EPISODES)):
        raise ValueError("Formal v1 must cover all 70 episodes")
    output = src_root / "object_geometry" / version
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refuse overwrite existing output: {output}")
    parent = output.parent
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=parent))
    manifest = {
        "status": "writing",
        "derived_schema_version":ref_m.get("derived_schema_version", 1),
        "object_geometry_contract": {**ref_m["object_geometry_contract"], "version":version},
        "calibration":old_m.get("calibration"),
        "asset_provenance":old_m.get("asset_provenance"),
        "source": {
            "dataset_id":"1018_peg_all",
            "original_version":"v1_hand_scene_q2",
            "original_manifest_sha256":sha256(old_path),
            "universal_reference_dataset":REFERENCE_DATASET,
            "universal_reference_manifest_sha256":sha256(ref_path),
            "roles":{"manipulated":"funnel","target":"target_flask"},
            "pose_reuse_status":"legacy_pose_reused_unmodified_pending_metric_review",
            "mode":"geometry_only_drop_hand_scene_extras_keep_raw_q2",
        },
        "requested_episodes":list(episodes),
        "records":[],
    }
    try:
        for ep in episodes:
            ids = by_ep[ep]
            shard_name = f"shard_e{ep:03d}"
            folder = tmp / shard_name
            folder.mkdir()
            arrays = {key: [] for key in ALL_FIELDS}
            for frame in ids:
                s = source.get(episode_index=ep, frame_index=frame)
                arrays["episode_index"].append(ep)
                arrays["frame_index"].append(frame)
                for field in ROLE_FIELDS:
                    arrays[field].append(np.asarray(getattr(s, field)))
                manifest["records"].append({
                    "episode_index":ep, "frame_index":frame,
                    "shard":shard_name, "row":len(arrays["frame_index"])-1,
                })
            for key, value in arrays.items():
                np.save(folder / f"{key}.npy", np.stack(value))
            print(f"[COPIED] episode={ep:02d} frames={len(ids)}", flush=True)
        manifest["status"]="complete"  # structural completeness, NOT geometric accuracy
        # Verify the unpublished temporary directory through the *actual*
        # project reader. Some versions check contract.version against dirname.
        testing_version = tmp.name
        manifest["object_geometry_contract"]["version"] = testing_version
        (tmp / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        test_reader = ObjectGeometryDerivedDataset(src_root, version=testing_version)
        n = sum(len(by_ep[ep]) for ep in episodes)
        if len(test_reader) != n:
            raise AssertionError(f"Written records {len(test_reader)} != {n}")
        for ep in episodes:
            ids = by_ep[ep]
            for frame in sorted(set([ids[0],ids[len(ids)//2],ids[-1]])):
                before = source.get(episode_index=ep, frame_index=frame)
                after = test_reader.get(episode_index=ep, frame_index=frame)
                for field in ROLE_FIELDS:
                    if not values_equal(before,after,field):
                        raise AssertionError(f"Value changed at e{ep} f{frame} field={field}")
        # release mmap handles before move
        del test_reader
        manifest["object_geometry_contract"]["version"] = version
        (tmp / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        os.replace(tmp, output)
        # After publish, prove the formal pathname can be opened by the reader.
        reloaded = ObjectGeometryDerivedDataset(src_root, version=version)
        assert len(reloaded) == n
        print(f"[PUBLISHED] {output} episodes={len(episodes)} frames={n}")
        print("[NOTICE] Geometry values are identical to legacy q2; RGBTrack pose metric accuracy is NOT approved by this check")
        return output
    except Exception:
        print(f"[FAILED] Unpublished temporary directory preserved for inspection: {tmp}")
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,default=Path("."))
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--pilot",action="store_true",help="Write v1_legacy_probe for episode 23 only")
    mode.add_argument("--all",action="store_true",help="Write formal v1 for all 70 eps after manual review")
    ap.add_argument("--acknowledge-legacy-pose",action="store_true",help="Explicitly accept unchanged old RGBTrack pose lineage for data conversion (not physical accuracy)")
    args=ap.parse_args(); repo=args.repo.resolve()
    src_root, old_m, ref_m, source, by_ep, report, old, ref=prepare(repo)
    audit=repo / "output/preprocess/1018_peg_all/migration_audit"
    audit.mkdir(parents=True,exist_ok=True)
    result=audit / "object_geometry_conversion_preflight.json"
    result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("[LEGACY]",len(source.episode_ids),"episodes",len(source),"frames")
    print("[SOURCE]",old)
    print("[UNIVERSAL REFERENCE]",ref)
    print("[LEGACY OBJECT CONTRACT]",report["legacy_geometry_contract"])
    print("[NEW OBJECT CONTRACT]",report["universal_geometry_contract"])
    print("[EXAMPLE]",report["sample_check"]["examples"][3])
    print("[AUDIT]",result)
    if args.pilot:
        export(repo, episodes=[23],version="v1_legacy_probe",acknowledge=False)
    elif args.all:
        export(repo,episodes=list(range(EXPECTED_EPISODES)),version="v1",acknowledge=args.acknowledge_legacy_pose)
    else:
        print("[STOP] Audit only; --pilot to write one episode. No spatial data required.")

if __name__ == "__main__":
    main()
