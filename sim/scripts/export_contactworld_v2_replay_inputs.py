#!/usr/bin/env python3
"""Export the already selected positive USB source episodes for v2 replay.

Uses the frozen v1 LeRobot manifest for source IDs and train/val/test splits.
Writes only to a new directory and checks every source interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from export_contactworld_replay_subset import FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("data/contactworld_usb_positive_all"))
    parser.add_argument("--source", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/data/contactworld"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_v2_full_20261002/input"))
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((args.v1 / "spatial/manifest.json").read_text())
    if previous["format"] != "contactworld-spatial-v1":
        raise ValueError("Expected frozen v1 selection manifest")
    selected = previous["episodes"][:args.max_episodes]
    group = zarr.open_group(str(args.source / "insertion_usb"), mode="r")
    ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    data = group["data"]
    receipts = []
    for original in selected:
        source_episode = int(original["source_episode"])
        start, end = int(starts[source_episode]), int(ends[source_episode])
        if int(original["source_start"]) != start or int(original["length"]) != end - start:
            raise ValueError("Original selection interval drift: %d" % source_episode)
        values = {key: np.asarray(data[key][start:end]) for key in FIELDS}
        if not all(len(value) == end-start and np.isfinite(value).all()
                   for value in values.values()):
            raise ValueError("Invalid released source episode %d" % source_episode)
        path = args.output / ("insertion_usb_episode_%03d.npz" % source_episode)
        np.savez_compressed(path, **values)
        receipts.append({"task": "insertion_usb", "episode": source_episode,
                         "output_episode_v1": int(original["episode_index"]),
                         "split": original["split"], "global_rows": [start, end],
                         "frames": end-start,
                         "socket_initial_xy_m": values["socket_pos_gt"][0, :2].tolist(),
                         "file": str(path),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print("%d/%d source=%d frames=%d" % (
            len(receipts), len(selected), source_episode, end-start), flush=True)
    manifest = {"selection": "identical positive USB source IDs and episode splits to frozen v1 dataset",
                "v1_manifest": str(args.v1 / "spatial/manifest.json"),
                "source": str(args.source), "fields": FIELDS, "episodes": receipts}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
