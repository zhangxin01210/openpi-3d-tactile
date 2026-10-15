#!/usr/bin/env python3
"""Build the USB V3 contact-centred tactile and local relation sidecar.

No simulator object pose, collision record, object ID, or segmentation is used.
The derived arrays are solely a deterministic function of the reviewed fused
point cloud plus bilateral taxel XYZ/force fields at the same pre-action frame.
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

from openpi.spatial_dataset import contactworld_contact_repr as contact_repr  # noqa: E402


FORMAT = "contactworld-usb-contact-representation-v2"


def _episodes(root: Path, requested: list[int] | None, limit: int | None) -> list[dict]:
    manifest = json.loads((root / "spatial/manifest.json").read_text())
    if manifest.get("format") != "contactworld-spatial-v3-bilateral":
        raise ValueError("Expected the reviewed USB V3 bilateral sidecar")
    rows = manifest["episodes"]
    if requested:
        wanted = set(requested)
        rows = [row for row in rows if int(row["episode_index"]) in wanted]
        if len(rows) != len(wanted):
            raise ValueError("Requested episode absent from V3 manifest")
    return rows[:limit] if limit is not None else rows


def _array(folder: Path, key: str) -> np.ndarray:
    return np.load(folder / f"{key}.npy", mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/contactworld_usb_v3_bilateral")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = (args.output or dataset / "spatial/contact_repr_v2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _episodes(dataset, args.episodes, args.limit)
    manifest_rows = []
    for ordinal, row in enumerate(rows, 1):
        episode = int(row["episode_index"])
        length = int(row["length"])
        src = dataset / "spatial/episodes" / f"episode_{episode:06d}"
        dest = output / "episodes" / f"episode_{episode:06d}.npz"
        if dest.exists() and not args.overwrite:
            with np.load(dest) as stored:
                if int(stored["finger_valid"].shape[0]) != length:
                    raise ValueError("Existing contact sidecar has wrong length: " + str(dest))
        else:
            cloud = _array(src, "pointcloud_fused_xyz")
            xyz = np.stack((_array(src, "tactile_xyz_left_base"),
                            _array(src, "tactile_xyz_right_base")), axis=1)
            force = np.stack((_array(src, "tactile_force_left_base"),
                              _array(src, "tactile_force_right_base")), axis=1)
            derived = contact_repr.build_sequence(cloud, xyz, force)
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **derived)
        with np.load(dest) as stored:
            candidate_counts = stored["candidate_mask"].sum(axis=-1)
            valid_fraction = stored["finger_valid"].mean(axis=0)
        manifest_rows.append({
            "episode_index": episode,
            "length": length,
            "file": str(dest.relative_to(output)),
            "left_candidate_count_mean": float(candidate_counts[:, 0].mean()),
            "right_candidate_count_mean": float(candidate_counts[:, 1].mean()),
            "left_contact_valid_fraction": float(valid_fraction[0]),
            "right_contact_valid_fraction": float(valid_fraction[1]),
        })
        print(f"{ordinal}/{len(rows)} episode={episode} frames={length}", flush=True)

    manifest = {
        "format": FORMAT,
        "source_format": "contactworld-spatial-v3-bilateral",
        "source_dataset": str(dataset),
        "taxels_per_finger": contact_repr.TAXELS,
        "anchors_per_finger": contact_repr.ANCHORS_PER_FINGER,
        "patch_points_per_anchor": contact_repr.PATCH_POINTS_PER_ANCHOR,
        "relation_candidates_per_finger": contact_repr.RELATION_CANDIDATES,
        "relation_filter": {
            "name": "tactile-facing local half-space",
            "radius_m": float(contact_repr.RELATION_RADIUS_M),
            "shells": [list(shell) for shell in contact_repr.RELATION_SHELLS],
            "backside_tolerance_m": float(contact_repr.BACKSIDE_TOLERANCE_M),
            "self_surface_margin_m": float(contact_repr.SELF_SURFACE_MARGIN_M),
            "limitation": "Not full robot mesh segmentation; audit visualizes all retained candidates.",
        },
        "episodes": manifest_rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output": str(output), "episodes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
