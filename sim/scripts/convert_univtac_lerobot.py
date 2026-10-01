#!/usr/bin/env python3
"""Convert audited UniVTAC HDF5 episodes to a small LeRobot v2 RGB base set.

World RGB-D and tactile press depth remain in the source HDF5. This conversion
only establishes LeRobot episode, RGB, state and next-state target alignment;
it does not claim a spatial/force sidecar or a complete OpenPI training gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def decode_source_rgb(encoded: bytes) -> np.ndarray:
    # UniVTAC stores source RGB values via cv2.imencode without a channel swap.
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode UniVTAC JPEG")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="UniVTAC episode root containing hdf5/*.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/univtac_insert_hole_depth")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    files = sorted((source / "hdf5").glob("*.hdf5"), key=lambda p: int(p.stem))
    if not files:
        parser.error(f"No HDF5 files under {source / 'hdf5'}")
    if output.exists():
        parser.error(f"Output exists: {output}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": None},
        "action": {"dtype": "float32", "shape": (9,), "names": None},
        "source_step": {"dtype": "int64", "shape": (1,), "names": None},
    }
    streams = {
        "observation.images.head": "observation/head/rgb",
        "observation.images.wrist": "observation/wrist/rgb",
        "observation.images.left_tactile": "tactile/left_tactile/rgb",
        "observation.images.right_tactile": "tactile/right_tactile/rgb",
    }
    for key, path in streams.items():
        shape = (270, 480, 3) if "observation/" in path else (240, 320, 3)
        features[key] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=60, root=output, robot_type="franka",
        features=features, use_videos=True, image_writer_threads=4,
    )
    manifest = {"source": str(source), "fps": 60, "action_semantics": "next saved joint observation", "episodes": []}
    try:
        for file in files:
            with h5py.File(file, "r") as h5:
                joint = np.asarray(h5["embodiment/joint"], dtype=np.float32)
                steps = np.asarray(h5["step"], dtype=np.int64)
                if joint.ndim != 2 or joint.shape[1] != 9 or len(joint) < 2:
                    raise ValueError(f"Bad joint trajectory: {file}")
                if not np.all(np.diff(steps) == 2):
                    raise ValueError(f"Unexpected time steps: {file}")
                for row in range(len(joint) - 1):
                    frame = {
                        "observation.state": joint[row],
                        "action": joint[row + 1],
                        "source_step": np.asarray([steps[row]], dtype=np.int64),
                        "task": "insert the peg into the hole",
                    }
                    for key, path in streams.items():
                        frame[key] = decode_source_rgb(h5[path][row])
                    dataset.add_frame(frame)
                dataset.save_episode()
                manifest["episodes"].append({
                    "source_seed": int(file.stem), "source_frames": len(joint),
                    "converted_frames": int(len(joint) - 1),
                    "first_source_step": int(steps[0]), "last_source_step": int(steps[-2]),
                })
                print(f"Converted seed {file.stem}: {len(joint) - 1} frames", flush=True)
    finally:
        dataset.stop_image_writer()
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {sum(e['converted_frames'] for e in manifest['episodes'])} frames to {output}")


if __name__ == "__main__":
    main()
