"""Check every exported ContactWorld episode before transferring the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, help="Optional source Zarr for action/state/force alignment checks")
    args = parser.parse_args()
    if shutil.which("ffprobe") is None:
        parser.error("ffprobe is required to verify exported MP4 files; install FFmpeg or add ffprobe to PATH")
    root = args.root
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "spatial/manifest.json").read_text())
    episodes = manifest["episodes"]
    if args.source:
        import zarr

        source = zarr.open_group(str(args.source / "insertion_usb"), mode="r")["data"]
    else:
        source = None
    if info["total_episodes"] != len(episodes):
        raise ValueError("Episode count mismatch")
    total = 0
    digest = hashlib.sha256()
    for ep in episodes:
        idx = ep["episode_index"]
        n = ep["length"]
        chunk = idx // 1000
        parquet = root / f"data/chunk-{chunk:03d}/episode_{idx:06d}.parquet"
        table = pq.read_table(parquet, columns=["episode_index", "frame_index", "action", "observation.state"])
        if table.num_rows != n or set(table["episode_index"].to_pylist()) != {idx}:
            raise ValueError(f"LeRobot rows mismatch episode {idx}")
        if table["frame_index"].to_pylist() != list(range(n)):
            raise ValueError(f"Frame order mismatch episode {idx}")
        if not np.isfinite(np.asarray(table["action"].to_pylist(), dtype=np.float32)).all():
            raise ValueError(f"Nonfinite actions episode {idx}")
        if source is not None:
            start = int(ep["source_start"])
            stop = start + n
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            expected_states = np.concatenate((source["dof_pos"][start:stop],
                                              source["dof_vel"][start:stop]), axis=-1)
            if not np.array_equal(actions, source["action"][start:stop]) or not np.array_equal(states, expected_states):
                raise ValueError(f"Source action/state mismatch episode {idx}")
            force_path = root / f"spatial/episodes/episode_{idx:06d}/force_grid.npy"
            if not np.array_equal(np.load(force_path), source["tactile_force_field_right"][start:stop]):
                raise ValueError(f"Source force mismatch episode {idx}")
        for key in ("pointcloud_xyz", "force_grid", "tactile_depth", "ee_pose"):
            path = root / f"spatial/episodes/episode_{idx:06d}/{key}.npy"
            array = np.load(path, mmap_mode="r")
            if len(array) != n or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key} episode {idx}")
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(str(path.stat().st_size).encode())
        for key in ("front", "wrist", "tactile"):
            video = root / f"videos/chunk-{chunk:03d}/observation.images.{key}/episode_{idx:06d}.mp4"
            result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                     "-show_entries", "stream=codec_name,pix_fmt,nb_frames",
                                     "-of", "json", str(video)], capture_output=True, text=True, check=True)
            stream = json.loads(result.stdout)["streams"][0]
            if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p" or int(stream["nb_frames"]) != n:
                raise ValueError(f"Video mismatch: {video}, {stream}")
            digest.update(video.relative_to(root).as_posix().encode())
            digest.update(str(video.stat().st_size).encode())
        total += n
    if total != info["total_frames"]:
        raise ValueError("Frame count mismatch")
    print(json.dumps({"status": "passed", "episodes": len(episodes), "frames": total,
                      "inventory_sha256": digest.hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
