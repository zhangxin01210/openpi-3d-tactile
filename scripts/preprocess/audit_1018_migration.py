#!/usr/bin/env python3
"""Read-only audit of 1018_peg_all vs completed 1028/1030 geometry data.

Reads manifests/configs, writes only audit reports. Does NOT run perception,
rewrite derived data, change symlinks, or assert RGBTrack pose equivalence.

Usage (from repo root):
  .venv/bin/python scripts/preprocess/audit_1018_migration.py
  .venv/bin/python scripts/preprocess/audit_1018_migration.py --repo "$PWD" \
      --out output/preprocess/1018_peg_all/migration_audit/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OLD = "1018_peg_all"
NEW = ("1028_press_button", "1030_yellow_socket")
EXPECTED = {OLD: (70, 32343), NEW[0]: (100, 26485), NEW[1]: (111, 34963)}
MISSING = object()


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("root is not a JSON object")
        return obj
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid manifest {path}: {exc}") from exc


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def at(d: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summarize_value(v: Any, depth: int = 0) -> Any:
    """Prevent per-frame records or large geometry fields leaking into report."""
    if depth > 5:
        return f"<{type(v).__name__}: nested>"
    if isinstance(v, dict):
        if len(v) > 60:
            return {"_dict_entries": len(v), "_sample_keys": list(v)[:10]}
        return {k: summarize_value(x, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        if len(v) > 30:
            return {"_list_length": len(v), "_sample_first": [summarize_value(x, depth + 1) for x in v[:2]]}
        return [summarize_value(x, depth + 1) for x in v]
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    return repr(v)


def flatten(obj: Any, prefix: str = "", limit: int = 1000) -> dict[str, Any]:
    """Leaf comparison for small configuration trees, never frames/records."""
    leaves: dict[str, Any] = {}
    def visit(v: Any, key: str, depth: int) -> None:
        if len(leaves) >= limit or depth > 12:
            return
        if isinstance(v, dict):
            for k, val in v.items():
                if k in ("records", "episodes", "shards", "source_meta_sha256") and depth > 0:
                    continue
                visit(val, f"{key}.{k}" if key else str(k), depth + 1)
        elif isinstance(v, list):
            if len(v) > 25:
                leaves[key] = f"<list:{len(v)}>"
            else:
                leaves[key] = v
        else:
            leaves[key] = v
    visit(obj or {}, prefix, 0)
    return leaves


def diff(a: Any, b: Any, max_lines: int = 80) -> dict[str, Any]:
    x, y = flatten(a), flatten(b)
    change = []
    for k in sorted(x.keys() | y.keys()):
        p, q = x.get(k, MISSING), y.get(k, MISSING)
        if p != q:
            change.append({"field": k, "old": "<absent>" if p is MISSING else p,
                           "new": "<absent>" if q is MISSING else q})
    return {"equal": not change, "changed_field_count": len(change),
            "differences": change[:max_lines], "truncated": len(change) > max_lines}


def episode_meta(path: Path) -> dict:
    if not path.is_file():
        return {"present": False}
    ids, frames = [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = obj.get("episode_index", obj.get("episode"))
            if idx is not None:
                ids.append(int(idx))
            if "length" in obj:
                frames += int(obj["length"])
    return {"present": True, "episodes": len(ids),
            "first_episode": min(ids) if ids else None,
            "last_episode": max(ids) if ids else None,
            "frames_from_lengths": frames if frames else None}


def entry(path: Path, root: Path, kind: str, version: str) -> dict:
    obj = load(path)
    relative = str(path.relative_to(root))
    result: dict[str, Any] = {"path": relative, "exists": obj is not None}
    if obj is None:
        return result
    result.update(status=obj.get("status"), version=version, sha256=sha256(path),
                  top_level_keys=sorted(obj.keys()))
    if kind == "spatial":
        result.update(
            schema_version=obj.get("derived_schema_version"),
            spatial_version=obj.get("spatial_version"),
            episode_count=at(obj, "source_dataset", "selected_episode_count"),
            frame_count=at(obj, "runtime", "total_selected_frames"),
            spatial_contract=summarize_value(obj.get("spatial_contract")),
            preprocess_config=summarize_value(obj.get("preprocess_config")),
            asset_provenance=summarize_value(obj.get("asset_provenance")),
            generated_with=summarize_value(obj.get("generator")),
        )
    elif kind == "object":
        records = obj.get("records")
        storage = at(obj, "storage", "episodes")
        result.update(
            schema_version=obj.get("derived_schema_version"),
            episode_count=(len({r.get("episode_index") for r in records}) if isinstance(records, list) and records else None),
            frame_count=len(records) if isinstance(records, list) and records else None,
            storage_episode_count=len(storage) if isinstance(storage, list) else None,
            object_geometry_contract=summarize_value(obj.get("object_geometry_contract")),
            hand_scene_contract=summarize_value(obj.get("hand_scene_contract")),
            calibration=summarize_value(obj.get("calibration")),
            asset_provenance=summarize_value(obj.get("asset_provenance")),
            source=summarize_value(obj.get("source")),
            manifest_keys=sorted(obj.keys()),
        )
        # Large format may store episode counts outside manifest.records.
        if result["episode_count"] is None:
            result["episode_count"] = at(obj, "source_dataset", "selected_episode_count")
        if result["frame_count"] is None:
            result["frame_count"] = at(obj, "runtime", "total_selected_frames")
    else:
        # Hand keypoint sidecar has its own manifest layout.
        result.update(
            episode_count=at(obj, "source_dataset", "selected_episode_count"),
            frame_count=at(obj, "runtime", "total_selected_frames"),
            schema=summarize_value(obj.get("hand_keypoint_contract", obj.get("keypoints_contract"))),
        )
    return result


def collect(repo: Path) -> dict:
    info: dict[str, Any] = {}
    for dataset in (OLD, *NEW):
        data_root = repo / "data" / dataset
        versions = {
            "spatial_old": "v1_front_candidate" if dataset == OLD else "v1",
            "object_old": "v1_hand_scene_q2" if dataset == OLD else "v1",
        }
        result = {"data_dir_exists": data_root.is_dir(),
                  "raw": episode_meta(data_root / "meta" / "episodes.jsonl"),
                  "expected_episodes": EXPECTED[dataset][0], "expected_frames": EXPECTED[dataset][1],
                  "spatial": entry(data_root / "spatial" / versions["spatial_old"] / "manifest.json", repo, "spatial", versions["spatial_old"]),
                  "object": entry(data_root / "object_geometry" / versions["object_old"] / "manifest.json", repo, "object", versions["object_old"]),
                  "hand": entry(data_root / "hand_keypoints/v1/manifest.json", repo, "hand", "v1"),
                  }
        if dataset == OLD:
            result["proposed_spatial_v1"] = entry(data_root / "spatial/v1/manifest.json", repo, "spatial", "v1")
            result["proposed_object_v1"] = entry(data_root / "object_geometry/v1/manifest.json", repo, "object", "v1")
            old_cfg = repo / "configs/object_geometry/datasets/1018_peg_all/config.json"
            cfg = load(old_cfg)
            result["old_geometry_config"] = {"path": str(old_cfg.relative_to(repo)),
                "exists": cfg is not None,
                "front_calibration": at(cfg, "paths", "front_calibration"),
                "spatial_version": at(cfg, "paths", "spatial_version"),
                "objects": summarize_value(cfg.get("assets", {})) if cfg else None,
                "pose_run_policy": at(cfg, "rgbtrack", "pose_run_policy"),
                "motion_assumptions": summarize_value(cfg.get("motion_assumptions")) if cfg else None}
        info[dataset] = result
    profile_path = repo / "configs/ur7e_xhand/front_calibration_candidate.json"
    profile = load(profile_path)
    return {"datasets": info, "legacy_front_calibration_profile": {
        "path": str(profile_path.relative_to(repo)), "exists": profile is not None,
        "sha256": sha256(profile_path), "values": summarize_value(profile)},
        "known_repository_assets": {
            rel: {"exists": (repo / rel).is_file(), "sha256": sha256(repo / rel)}
            for rel in (
                "configs/ur7e_xhand/cameras_0903.json",
                "configs/ur7e_xhand/front_calibration_candidate.json",
                "assets/object_geometry/synthware_19_22/funnel_F181975.glb",
                "assets/object_geometry/synthware_19_22/flask_F301950.glb",
            )
        },
        "scope": "Read-only audit; no raw datasets, poses, tracking, or derived modalities changed."}


def evaluate(info: dict) -> dict:
    datasets = info["datasets"]
    old = datasets[OLD]
    n1, n2 = (datasets[x] for x in NEW)
    ref1, ref2 = n1["spatial"], n2["spatial"]
    out: dict[str, Any] = {}
    if ref1["exists"] and ref2["exists"]:
        out["new_spatial_baseline_config"] = diff(ref1.get("preprocess_config"), ref2.get("preprocess_config"))
        out["new_spatial_contract"] = diff(ref1.get("spatial_contract"), ref2.get("spatial_contract"))
        out["new_calibration_assets"] = diff(ref1.get("asset_provenance"), ref2.get("asset_provenance"))
    else:
        out["new_spatial_baseline"] = "UNKNOWN: new spatial manifests missing"
    if old["spatial"]["exists"] and ref1["exists"]:
        out["old_vs_new_spatial_config"] = diff(old["spatial"].get("preprocess_config"), ref1.get("preprocess_config"))
        out["old_vs_new_spatial_contract"] = diff(old["spatial"].get("spatial_contract"), ref1.get("spatial_contract"))
        out["old_vs_new_calibration_assets"] = diff(old["spatial"].get("asset_provenance"), ref1.get("asset_provenance"))
    else:
        out["old_vs_new_spatial"] = "UNKNOWN: 1018 old or reference spatial manifest missing"
    if old["object"]["exists"] and n1["object"]["exists"]:
        out["old_vs_new_object_contract"] = diff(old["object"].get("object_geometry_contract"), n1["object"].get("object_geometry_contract"))
        out["old_vs_new_object_calibration"] = diff(old["object"].get("calibration"), n1["object"].get("calibration"))
    else:
        out["old_vs_new_object"] = "UNKNOWN: missing object manifest"
    warnings = []
    for name in (OLD, *NEW):
        d = datasets[name]
        raw = d["raw"]
        if not raw["present"]:
            warnings.append(f"{name}: raw meta/episodes.jsonl missing")
        elif raw["episodes"] != d["expected_episodes"]:
            warnings.append(f"{name}: raw episode count mismatch expected={d['expected_episodes']} got={raw['episodes']}")
        elif raw["frames_from_lengths"] is not None and raw["frames_from_lengths"] != d["expected_frames"]:
            warnings.append(f"{name}: raw frame count mismatch expected={d['expected_frames']} got={raw['frames_from_lengths']}")
        for k in ("spatial", "object"):
            x = d[k]
            if not x["exists"]:
                warnings.append(f"{name}: missing {k} manifest: {x['path']}")
            elif x["status"] != "complete":
                warnings.append(f"{name}: {k} manifest status={x['status']!r} (not complete)")
            elif x.get("frame_count") is not None and x["frame_count"] != d["expected_frames"]:
                warnings.append(f"{name}: {k} manifest frame_count={x['frame_count']} vs expected={d['expected_frames']}")
    out["warnings"] = warnings
    out["decisions"] = {
        "1018_spatial": "REBUILD only after checking the new camera contract and verifying a few e23 frames; never overwrite v1_front_candidate",
        "1018_object_v1": "NEW UNIVERSAL EXPORT after verifying legacy RGBTrack pose/base_link frame and approved mesh provenance; old q2 remains intact",
        "1018_rgbtrack_rerun": "UNDECIDED; NOT AUTHORIZED by this manifest audit (pose provenance, frame alignment and visual check first)",
        "1018_hand_keypoints_v1": "EXPORT via already tested 18-point FK script (independent of RGBTrack)",
        "model_code": "OUT OF SCOPE",
    }
    return out


def md_report(repo: Path, info: dict, results: dict) -> str:
    def info_row(name: str, key: str) -> str:
        d = info["datasets"][name][key]
        if not d.get("exists"):
            return "MISSING"
        return f"{d.get('status', 'unknown')}, {d.get('episode_count', '?')} eps, {d.get('frame_count', '?')} frames"
    s = ["# 1018 → unified v1 migration audit", "", f"Repository: `{repo}`", "", "## Manifests", "",
         "| Dataset | Spatial | Object Geometry | Hand Keypoints |", "|---|---|---|---|" ]
    for n in (OLD, *NEW):
        s.append(f"| {n} | {info_row(n,'spatial')} | {info_row(n,'object')} | {info_row(n,'hand')} |")
    s += ["", "## Comparison summaries", ""]
    for k,v in results.items():
        if isinstance(v, dict) and "equal" in v:
            s.append(f"- **{k}**: {'MATCH' if v['equal'] else 'DIFFERENT'} ({v['changed_field_count']} differing leaf fields)")
            if not v["equal"]:
                for item in v["differences"][:16]:
                    s.append(f"  - `{item['field']}`: `{str(item['old'])[:180]}` → `{str(item['new'])[:180]}`")
                if len(v["differences"]) > 16:
                    s.append("  - More differences in JSON report")
        elif k in ("old_vs_new_spatial", "new_spatial_baseline", "old_vs_new_object"):
            s.append(f"- **{k}**: {v}")
    s += ["", "## Warnings", ""]
    s += [f"- {x}" for x in results["warnings"]] or ["- None (this does not establish geometric correctness)"]
    s += ["", "## Decisions / gates", ""]
    for k,v in results["decisions"].items():
        s.append(f"- **{k}**: {v}")
    s += ["", "## Important limitations", "",
          "- Equality of two manifest fields does NOT prove the camera remained mechanically fixed across collection dates.",
          "- A complete sidecar or valid RGBTrack pose does NOT prove the transform is metric-accurate.",
          "- Older dataset may have different `asset_provenance` layout; missing fields are UNKNOWN, not evidence of equality.",
          "- Do not copy/rename old q2 object geometry into universal v1, or replace stored run outputs.", ""]
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path, default=Path("output/preprocess/1018_peg_all/migration_audit/report.json"))
    args = ap.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / "src/openpi/spatial").is_dir():
        ap.error(f"Not an openpi-3d-tactile repository: {repo}")
    info = collect(repo)
    findings = evaluate(info)
    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"audit_version": 1, "repo": str(repo), **info, "comparison": findings}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(md_report(repo, info, findings), encoding="utf-8")
    for name in (OLD, *NEW):
        d = info["datasets"][name]
        print(f"[DATASET] {name}")
        for kind in ("spatial", "object", "hand"):
            z = d[kind]
            print(f"  {kind:8} {z['path']}: {z.get('status', 'MISSING')} frames={z.get('frame_count', '?')}")
    for label in ("new_spatial_baseline_config", "new_spatial_contract", "old_vs_new_spatial_config", "old_vs_new_spatial_contract", "old_vs_new_calibration_assets", "old_vs_new_object_contract"):
        val = findings.get(label)
        if val:
            print(f"[COMPARE] {label}: {'MATCH' if val['equal'] else 'DIFFERENT'} fields={val['changed_field_count']}")
            for d in val["differences"][:6]:
                print(f"    {d['field']}: {str(d['old'])[:100]} -> {str(d['new'])[:100]}")
    print(f"[WARNINGS] {len(findings['warnings'])}")
    for warning in findings["warnings"][:8]:
        print("  -", warning)
    print("[AUDIT JSON]", out)
    print("[AUDIT MD]", md)
    print("[STOP] Manifest audit only: pose re-use decision requires camera/pose provenance + visual validation.")


if __name__ == "__main__":
    main()
