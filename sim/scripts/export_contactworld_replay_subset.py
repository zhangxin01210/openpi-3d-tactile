#!/usr/bin/env python3
"""Export fixed, complete episodes for isolated USB/Peg replay validation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


EPISODES = {"insertion_usb": (1, 4, 67, 101, 132), "insertion_peg": (1, 33, 101)}
FIELDS = ("action", "dof_pos", "dof_vel", "ee_pos", "ee_quat", "plug_pos",
          "plug_quat", "socket_pos_gt", "socket_quat", "front", "wrist",
          "tactile_force_field_right", "tactile_rgb_right", "tactile_depth_right", "step_idx")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for task, episodes in EPISODES.items():
        root = zarr.open_group(str(args.data / task), mode="r")
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        starts = np.r_[0, ends[:-1]]
        for ep in episodes:
            a, b = int(starts[ep]), int(ends[ep])
            values = {key: np.asarray(root[f"data/{key}"][a:b]) for key in FIELDS}
            assert all(len(v) == b-a and np.isfinite(v).all() for v in values.values())
            assert values["action"].shape[1] == 6
            name = f"{task}_episode_{ep:03d}.npz"
            path = args.output / name
            np.savez_compressed(path, **values)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            receipts.append({"task": task, "episode": ep, "global_rows": [a, b],
                             "frames": b-a, "socket_initial_xy_m": values["socket_pos_gt"][0, :2].tolist(),
                             "file": str(path), "sha256": checksum,
                             "columns": {key: list(value.shape) for key, value in values.items()}})
            print(task, ep, b-a, flush=True)
    (args.output / "manifest.json").write_text(json.dumps({"selection": "Fixed episode IDs chosen before online replay; offline endpoint pose criterion passed for all eight; not an official success label", "fields": FIELDS, "episodes": receipts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
