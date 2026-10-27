#!/usr/bin/env python3
"""Validate metric CAD provenance and the explicit RGBTrack scale manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    metadata_path = root / "assets/object_geometry/synthware_19_22/dimensions.json"
    manifest_path = root / "data/pose_eval/hand_scene_metric_mesh_scales.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("units") != "metres_per_mesh_unit":
        raise ValueError(f"Unsupported metric scale manifest: {manifest_path}")
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Asset extent verification needs trimesh; run with the project Python") from exc

    results = {}
    for object_name in ("funnel", "target_flask"):
        record = metadata["models"][object_name]
        scale_record = manifest["provenance"]["funnel_glb" if object_name == "funnel" else "target_flask_glb"]
        mesh_path = (root / "assets/object_geometry/synthware_19_22" / record["file"]).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        digest = sha256(mesh_path)
        if digest != record["sha256"] or digest != scale_record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {object_name}: {digest}")
        if record.get("unit") != "meter":
            raise ValueError(f"CAD metadata does not explicitly declare meters for {object_name}")
        scale = float(manifest["objects"][object_name])
        if scale != float(scale_record["scale_m_per_unit"]):
            raise ValueError(f"Scale provenance mismatch for {object_name}")
        scene = trimesh.load(str(mesh_path), force="scene", process=False)
        mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
        extents_mm_from_mesh = (mesh.extents.astype("float64") * scale * 1000.0)
        extents_mm_recorded = list(map(float, record["mesh_dimensions_mm"]))
        error_mm = max(abs(a - b) for a, b in zip(extents_mm_from_mesh, extents_mm_recorded, strict=True))
        if error_mm > 0.2:
            raise ValueError(
                f"{object_name} bounds do not corroborate the declared scale: max error={error_mm:.4f} mm"
            )
        results[object_name] = {
            "path": str(mesh_path.relative_to(root)),
            "sha256": digest,
            "scale_m_per_unit": scale,
            "extents_m": (mesh.extents.astype("float64") * scale).tolist(),
            "metadata_extents_mm": extents_mm_recorded,
            "max_extent_error_mm": error_mm,
        }
    print(json.dumps({"ok": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
