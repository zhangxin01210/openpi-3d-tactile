#!/usr/bin/env python3
"""Export approved RGBTrack v1 + metric GLBs to the EXISTING object_geometry/v1 reader contract.

CPU only. No tracking, FK or tactile computation. Invalid poses never create valid geometry.
Result: data/<dataset>/object_geometry/<version>/{manifest.json,shard_*/...npy}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import numpy as np

POINT_COUNT = 512
FIELDS = ("xyz_m", "mask", "source", "age", "confidence")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_surface_points(path: Path, seed: int) -> np.ndarray:
    """Same GLB scene flattening/units as rgbtrack_adapter.convert_glb_to_rgbtrack_mesh."""
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or len(vertices) == 0:
        raise ValueError(f"Invalid vertices in {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"Mesh must contain triangle faces: {path}")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Mesh surface area is zero: {path}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=POINT_COUNT, replace=True, p=areas / total)
    r = rng.random((POINT_COUNT, 2))
    s = np.sqrt(r[:, 0:1])
    tri = triangles[chosen]
    xyz = (1.0 - s) * tri[:, 0] + s * (1.0 - r[:, 1:2]) * tri[:, 1] + s * r[:, 1:2] * tri[:, 2]
    return np.asarray(xyz, dtype=np.float32)


def source_rows(path: Path, *, object_name: str) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                raise ValueError(f"Blank pose row in {path}, row={idx}")
            row = json.loads(line)
            if row.get("frame_index") != idx or row.get("object_name") != object_name:
                raise ValueError(f"Frame/object mismatch in {path}, row={idx}")
            if type(row.get("valid")) is not bool:
                raise ValueError(f"Missing boolean valid in {path}, row={idx}")
            records.append(row)
    if not records:
        raise ValueError(f"Empty pose stream: {path}")
    return records


def cloud_from_pose(row: dict, points_object: np.ndarray) -> np.ndarray:
    mat = np.asarray(row.get("T_base_link_object"), dtype=np.float64)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise ValueError("Valid pose must have finite 4x4 T_base_link_object")
    if not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous matrix last row")
    rot = mat[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=2e-2) or np.linalg.det(rot) < 0.9:
        raise ValueError("Pose rotation matrix is not a proper rotation")
    xyz = points_object @ rot.T + mat[:3, 3]
    return xyz.astype(np.float32)


def convert_chunk(
    index_start: int,
    episode: int,
    rows: dict[str, list[dict]],
    points: dict[str, np.ndarray],
    roles: dict[str, str],
    last_valid: dict[str, int | None],
    shard_root: Path,
    record_index: list[dict],
) -> None:
    frame_count = len(next(iter(rows.values())))
    n = frame_count
    arrays: dict[str, np.ndarray] = {
        "episode_index": np.full(n, episode, dtype=np.int64),
        "frame_index": np.arange(index_start, index_start + n, dtype=np.int64),
    }
    for role, name in roles.items():
        xyz = np.zeros((n, POINT_COUNT, 3), dtype=np.float32)
        valid = np.zeros((n, POINT_COUNT), dtype=np.bool_)
        # Legacy v1 source: 0 = current pose geometry, 1 = history/inferred. This exporter never infers.
        source = np.zeros((n, POINT_COUNT), dtype=np.int8)
        age = np.zeros((n, POINT_COUNT), dtype=np.int16)
        confidence = np.zeros((n, POINT_COUNT), dtype=np.float32)
        for j, item in enumerate(rows[name]):
            frame = index_start + j
            if item["valid"]:
                xyz[j] = cloud_from_pose(item, points[name])
                valid[j, :] = True
                last_valid[role] = frame
                c = float(item.get("confidence", 0))
                if not np.isfinite(c) or not 0 <= c <= 1:
                    raise ValueError(f"e{episode} {name} f{frame}: invalid confidence {c}")
                confidence[j, :] = c
            else:
                # Never use the predicted pose from invalid records: zeros AND mask=false.
                age[j, :] = -1 if last_valid[role] is None else min(32767, frame - last_valid[role])
        for field, a in zip(FIELDS, (xyz, valid, source, age, confidence)):
            arrays[f"{role}_{field}"] = a
    shard_root.mkdir(parents=True, exist_ok=False)
    for field, a in arrays.items():
        np.save(shard_root / f"{field}.npy", a, allow_pickle=False)
    for i in range(n):
        record_index.append({"episode_index": episode, "frame_index": index_start + i,
                             "shard": shard_root.name, "row": i})


def run(args: argparse.Namespace) -> None:
    if not NAME.fullmatch(args.dataset) or not NAME.fullmatch(args.version):
        raise ValueError("Invalid --dataset or --version name")
    repo = args.repo.expanduser().resolve()
    dataset = repo / "data" / args.dataset
    prep = repo / "output" / "preprocess" / args.dataset
    release_path = prep / "rgbtrack" / "release_v1.json"
    release = read_json(release_path)
    if release.get("dataset") != args.dataset:
        raise ValueError("Published RGBTrack release belongs to a different dataset")
    policy = release.get("policy", {})
    dynamic = policy.get("dynamic_object")
    static = policy.get("static_object")
    objects = release.get("objects")
    if not isinstance(objects, list) or len(objects) != 2 or set(objects) != {dynamic, static}:
        raise ValueError("Release must name exactly one dynamic and one static object")
    roles = {"manipulated": dynamic, "target": static}
    ids_all = release.get("episodes")
    if not isinstance(ids_all, list) or not ids_all or len(ids_all) != len(set(ids_all)):
        raise ValueError("Bad episode list in release manifest")
    ids = sorted(int(x) for x in ids_all)
    if args.episodes == "all":
        selected = ids
    else:
        selected = sorted(set(int(x) for x in args.episodes.split(",")))
        if not selected or not set(selected).issubset(ids):
            raise ValueError("Episode selection is not in published release")
    geo_manifest_path = prep / "geometry" / "approved" / "manifest.json"
    geo_manifest = read_json(geo_manifest_path)
    if geo_manifest.get("status") != "human_accepted" or set(geo_manifest.get("objects", [])) != set(objects):
        raise ValueError("Geometry manifest is not approved or object names mismatch")
    mesh_hashes = {}
    for name in objects:
        path = prep / "geometry" / "approved" / f"{name}.glb"
        actual = sha256(path)
        entry = geo_manifest.get("assets", {}).get(name, {})
        if actual != entry.get("sha256") or float(entry.get("scale_m_per_unit", -1)) != 1.0:
            raise ValueError(f"{name}: approved metric GLB hash/scale mismatch")
        mesh_hashes[name] = actual
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    output = dataset / "object_geometry" / args.version
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    # 512 sampled points per object, deterministic and identical for all frames/episodes.
    points = {name: load_surface_points(prep / "geometry" / "approved" / f"{name}.glb", 20261101 + i)
              for i, name in enumerate(objects)}
    if args.dry_run:
        print(f"[PLAN] {args.dataset} {len(selected)} episodes roles={roles} output={output}")
        print(f"[MESH] SHA256 {mesh_hashes}")
        return
    staging = output.with_name(output.name + f".building_{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Refuse to overwrite staging directory: {staging}")
    staging.mkdir(parents=True)
    all_records: list[dict] = []
    episode_counts = {}
    pose_files = {}
    begin = time.perf_counter()
    # No manifest with status=complete until the entire selected set succeeds.
    for ord_idx, ep in enumerate(selected, 1):
        canonical = prep / "rgbtrack" / "episodes" / f"e{ep:05d}"
        target = canonical.parent.parent / "runs" / release["run_tag"] / "episodes" / f"e{ep:05d}"
        if not canonical.is_symlink() or canonical.resolve() != target.resolve() or not canonical.is_dir():
            raise RuntimeError(f"e{ep}: official RGBTrack symlink does not match release")
        rows = {name: source_rows(canonical / f"{name}_poses.jsonl", object_name=name)
                for name in objects}
        nframes = len(rows[dynamic])
        if len(rows[static]) != nframes:
            raise ValueError(f"e{ep}: pose stream frame mismatch")
        # Do not fabricate frames absent from either pose stream.
        qc = read_json(canonical / "quality_summary.json")
        for name in objects:
            q = qc.get("objects", {}).get(name, {})
            if int(q.get("frames", -1)) != nframes or int(q.get("valid_frames", -1)) != sum(r["valid"] for r in rows[name]):
                raise ValueError(f"e{ep} {name}: pose/QC count mismatch")
        last_valid: dict[str, int | None] = {"manipulated": None, "target": None}
        for start in range(0, nframes, args.shard_size):
            cut = {name: records[start:start+args.shard_size] for name, records in rows.items()}
            shard_name = f"shard_e{ep:05d}_{start//args.shard_size:04d}"
            convert_chunk(start, ep, cut, points, roles, last_valid, staging / shard_name, all_records)
        pose_files[str(ep)] = {name: sha256(canonical / f"{name}_poses.jsonl") for name in objects}
        episode_counts[str(ep)] = {"frames": nframes, "valid": {name: sum(r["valid"] for r in rows[name]) for name in objects}}
        print(f"[{ord_idx}/{len(selected)}] e{ep:05d} frames={nframes} "+
              " ".join(f"{name}={episode_counts[str(ep)]['valid'][name]}/{nframes}" for name in objects), flush=True)
    manifest = {
        "status": "complete",
        "object_geometry_contract": {"version": args.version, "point_count": POINT_COUNT},
        "schema_version": 1,
        "producer": "scripts/preprocess/export_object_geometry_rgbtrack.py",
        "dataset": args.dataset,
        "source_release": str(release_path.relative_to(repo)),
        "source_release_sha256": sha256(release_path),
        "approved_geometry_sha256": sha256(geo_manifest_path),
        "approved_meshes_sha256": mesh_hashes,
        "role_to_object": roles,
        "pose_files_sha256_by_episode": pose_files,
        "episode_summary": episode_counts,
        "number_of_records": len(all_records),
        "source_semantics": "source=0 for directly tracked/frozen poses; source=1 reserved for inferred history; this exporter never infers",
        "validity_semantics": "invalid pose: xyz=0 mask=false confidence=0; age=time since last valid (-1 before first)",
        "coordinate_frame": "base_link",
        "length_unit": "m",
        "surface_sampling_seed": 20261101,
        "shard_size": args.shard_size,
        "records": all_records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output appeared while building: {output}")
    staging.rename(output)
    print(f"[DONE] {args.dataset} episodes={len(selected)} frames={len(all_records)} -> {output}")
    print(f"[TIME] {time.perf_counter()-begin:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", default="v1")
    p.add_argument("--episodes", default="all", help="all or comma-separated episode ids")
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.shard_size < 1:
        p.error("--shard-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
