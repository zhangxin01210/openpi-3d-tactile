#!/usr/bin/env python3
"""Smoke-test the actual TWO-object RGBTrack deployment constructors, without frames or robot commands.

Validates training CAD identity/scales, shared scorer/refiner, compiled pose
grid, and FoundationPose initialization with the *converted metric meshes*.
This does not claim any pose accuracy or SAM2/RGB-D parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "object_geometry" / "pose_tracking"))

import rgbtrack_adapter as adapter  # noqa: E402
from scripts.deploy.run_online_perception import _load_training_geometry_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--scale-manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    weights = args.weights_root.expanduser().resolve()
    manifest = args.scale_manifest.expanduser().resolve()
    objects = ("funnel", "target_flask")
    mesh_paths = {
        "funnel": root / "assets/object_geometry/synthware_19_22/funnel_F181975.glb",
        "target_flask": root / "assets/object_geometry/synthware_19_22/flask_F301950.glb",
    }
    scales = adapter._load_scale_manifest(manifest, objects)
    spec = _load_training_geometry_spec(
        root / "configs/object_geometry/datasets/1018_peg_all/config.json",
        repo_root=root,
        meshes=mesh_paths,
        scale_manifest=scales,
    )
    calibration = adapter.load_front_calibration(
        root, root / "configs/ur7e_xhand/front_calibration_candidate.json"
    )
    report = {"source": "training_geometry_config", "objects": {}, "shared_models": False}
    with tempfile.TemporaryDirectory(prefix="rgbtrack_objects_smoke_") as tmp:
        temp = Path(tmp)
        trackers = {}
        for name in objects:
            conversion = adapter.convert_glb_to_rgbtrack_mesh(
                mesh_paths[name], temp / "meshes", name, scales[name]
            )
            tracker = adapter.RGBTrackObject(
                repo_root=root,
                object_name=name,
                mesh_path=conversion["output_mesh"],
                mode="depth",
                debug_dir=temp / "debug",
                weights_root=weights,
            )
            tracker._ensure_estimator()
            est = tracker._estimator
            if est is None:
                raise RuntimeError(f"{name}: FoundationPose not initialized")
            verts = tracker.mesh_vertices
            if verts is None or not np.isfinite(verts).all():
                raise ValueError(f"{name}: invalid mesh vertices")
            extents = (verts.max(axis=0) - verts.min(axis=0)).tolist()
            if not all(0.005 < value < 0.5 for value in extents):
                raise ValueError(f"{name}: implausible metric mesh extents (m): {extents}")
            if not hasattr(est.scorer, "predict") or not hasattr(est.refiner, "predict"):
                raise RuntimeError(f"{name}: missing prediction interface")
            if not hasattr(est.scorer, "model") or not hasattr(est.refiner, "model"):
                raise RuntimeError(f"{name}: scorer/refiner missing upstream .model attribute")
            trackers[name] = tracker
            report["objects"][name] = {
                "source_sha256": spec["assets"][name]["sha256"],
                "sample_seed": spec["assets"][name]["sampling_seed"],
                "scale_m_per_unit": scales[name],
                "mesh_extents_m": extents,
                "candidate_rotations": int(len(est.rot_grid)) if hasattr(est, "rot_grid") else None,
                "mycpp": tracker.mycpp_provenance,
            }
        a, b = (trackers[name]._estimator for name in objects)
        report["shared_models"] = (
            a.scorer is b.scorer and a.refiner is b.refiner
        )
        if not report["shared_models"]:
            raise RuntimeError("Two-object tracking did not share scorer/refiner as expected")
        report["calibration"] = {
            "K_color": calibration.K_color.tolist(),
            "T_base_color": calibration.T_base_color.tolist(),
        }
    report["ok"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
