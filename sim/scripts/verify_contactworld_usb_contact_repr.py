#!/usr/bin/env python3
"""Validate the derived USB contact representation before model training.

The verifier checks every episode's row count, dtype, finite values, and
deterministic frame reconstruction.  It deliberately uses only the reviewed
LeRobot V3 spatial sidecar, never simulator ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from openpi.spatial_dataset.contactworld_contact_repr import (  # noqa: E402
    build_frame,
)


SHAPES = {
    "patch_xyz_local": (2, 2, 16, 2),
    "patch_force": (2, 2, 16, 3),
    "patch_mask": (2, 2, 16),
    "anchor_xyz_base": (2, 2, 3),
    "anchor_uv": (2, 2, 2),
    "anchor_stats": (2, 2, 10),
    "anchor_mask": (2, 2),
    "finger_summary": (2, 13),
    "finger_center_base": (2, 3),
    "finger_valid": (2,),
    "query_xyz_base": (2, 3),
    "query_features": (2, 13),
    "query_valid": (2,),
    "candidate_xyz_base": (2, 48, 3),
    "candidate_mask": (2, 48),
    "candidate_distance": (2, 48),
    "candidate_shell": (2, 48),
}


def check_array(name: str, value: np.ndarray, length: int) -> None:
    if value.shape != (length, *SHAPES[name]):
        raise ValueError(f"{name}: got {value.shape}, expected {(length, *SHAPES[name])}")
    if name.endswith("mask") or name in {"finger_valid", "query_valid"}:
        if value.dtype != np.bool_:
            raise ValueError(f"{name}: expected bool, got {value.dtype}")
    elif name == "candidate_shell":
        if value.dtype != np.int8:
            raise ValueError(f"{name}: expected int8, got {value.dtype}")
        if np.any((value < -1) | (value > 3)):
            raise ValueError(f"{name}: invalid shell id")
    elif not np.isfinite(value).all():
        raise ValueError(f"{name}: non-finite value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/contactworld_usb_v3_bilateral")
    parser.add_argument("--repr-root", type=Path, default=None)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples_per_episode < 1:
        parser.error("--samples-per-episode must be positive")
    dataset = args.dataset.resolve()
    repr_root = (args.repr_root or dataset / "spatial/contact_repr_v2").resolve()
    source_manifest = json.loads((dataset / "spatial/manifest.json").read_text())
    manifest = json.loads((repr_root / "manifest.json").read_text())
    if source_manifest.get("format") != "contactworld-spatial-v3-bilateral":
        raise ValueError("Expected bilateral USB V3 source sidecar")
    if manifest.get("format") != "contactworld-usb-contact-representation-v2":
        raise ValueError("Unsupported contact representation format")
    source_rows = source_manifest["episodes"]
    derived_rows = manifest["episodes"]
    if [row["episode_index"] for row in source_rows] != [row["episode_index"] for row in derived_rows]:
        raise ValueError("Contact representation episodes differ from source sidecar")

    reconstructed = 0
    candidate_counts = []
    for source_row, derived_row in zip(source_rows, derived_rows, strict=True):
        episode = int(source_row["episode_index"])
        length = int(source_row["length"])
        if int(derived_row["length"]) != length:
            raise ValueError(f"episode {episode}: mismatched length")
        path = repr_root / "episodes" / f"episode_{episode:06d}.npz"
        with np.load(path) as archive:
            if set(archive.files) != set(SHAPES):
                raise ValueError(f"episode {episode}: unexpected keys {sorted(archive.files)}")
            arrays = {key: archive[key] for key in archive.files}
        for name, value in arrays.items():
            check_array(name, value, length)
        candidate_counts.append(arrays["candidate_mask"].sum(axis=(0, 2)).astype(int).tolist())

        source = dataset / "spatial/episodes" / f"episode_{episode:06d}"
        cloud = np.load(source / "pointcloud_fused_xyz.npy", mmap_mode="r")
        xyz = np.stack((np.load(source / "tactile_xyz_left_base.npy", mmap_mode="r"),
                        np.load(source / "tactile_xyz_right_base.npy", mmap_mode="r")), axis=1)
        force = np.stack((np.load(source / "tactile_force_left_base.npy", mmap_mode="r"),
                          np.load(source / "tactile_force_right_base.npy", mmap_mode="r")), axis=1)
        frame_ids = np.unique(np.rint(np.linspace(0, length - 1, args.samples_per_episode)).astype(int))
        for frame in frame_ids:
            expected = build_frame(cloud[frame], xyz[frame], force[frame])
            for key, value in expected.items():
                if not np.array_equal(value, arrays[key][frame]):
                    raise ValueError(f"episode {episode} frame {frame}: stale or altered {key}")
            reconstructed += 1

    result = {
        "status": "passed",
        "dataset": str(dataset),
        "contact_repr_root": str(repr_root),
        "episodes": len(source_rows),
        "reconstructed_frames": reconstructed,
        "candidate_count_sum_left_right": np.asarray(candidate_counts, dtype=int).sum(axis=0).tolist(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
