#!/usr/bin/env python3
"""Print reproducible XHand geometry region IDs and optionally validate a hand manifest.

This is a geometry audit only; it is not a training preflight or smoke test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.models.spatial_encoders.final_v2_relations import keypoint_membership
from openpi.spatial.tactile import TactileConfig
from openpi.spatial.tactile_geometry import load_xhand_taxel_geometry
from openpi.spatial.tactile_regions import partition_taxel_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    geometry_root = root / "configs/ur7e_xhand/tactile"
    geometry = load_xhand_taxel_geometry(
        t16_transformed_path=geometry_root / "points_t16_transformed.json",
        t30_right_transformed_path=geometry_root / "points_t30_right_hand_transformed.json",
        tactile_config=TactileConfig(),
    )
    regions = partition_taxel_geometry(geometry.xyz_link_m)
    report = {"finger_order": geometry.finger_names, "regions": {}}
    for finger, labels in zip(geometry.finger_names, regions, strict=True):
        report["regions"][finger] = {
            str(region): [taxel_id + 1 for taxel_id, label in enumerate(labels) if label == region]
            for region in range(4)
        }
    if args.hand_manifest:
        manifest = json.loads(args.hand_manifest.read_text(encoding="utf-8"))
        membership = keypoint_membership(manifest["names"], manifest["groups"])
        report["hand_keypoints"] = {
            finger: [name for name, include in zip(manifest["names"], row, strict=True) if include]
            for finger, row in zip(geometry.finger_names, membership, strict=True)
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
