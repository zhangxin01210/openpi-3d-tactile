"""Export audited ContactWorld USB demonstrations to LeRobot v2.1 plus spatial arrays.

The file contains positive train/val/test episodes. The loader only trains on
train episodes, with full action horizons. Source IDs/splits are in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path

import numpy as np
import zarr

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import lerobot.common.datasets.lerobot_dataset as lerobot_module

# The tested LeRobot writer defaults to AV1. Use H.264/yuv420p for the
# ContactWorld review and training videos, without changing the installed package.
lerobot_module.encode_video_frames = functools.partial(
    lerobot_module.encode_video_frames, vcodec="h264", pix_fmt="yuv420p", crf=18,
)


def as_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {frame.shape}")
    if not np.isfinite(frame).all():
        raise ValueError("Nonfinite RGB")
    return np.rint(np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None, help="Only for conversion smoke tests")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    audit = list(csv.DictReader(args.audit.open()))
    chosen = [(int(row["episode"]), row["split"]) for row in audit if row["task"] == "insertion_usb"
              and row["source_endpoint_pass"] == "True"]
    if args.max_episodes is not None:
        chosen = chosen[:args.max_episodes]
    if not chosen:
        raise ValueError("No positive USB episodes")

    geometry = json.loads(args.geometry.read_text())
    if geometry.get("status") != "passed" or not geometry["tasks"]["insertion_usb"]["passed"]:
        raise ValueError("ContactWorld geometry report is not passed")
    rotation = np.asarray(geometry["A"], dtype=np.float32)
    translation = np.asarray(geometry["t"], dtype=np.float32)
    group = zarr.open_group(str(args.source / "insertion_usb"), mode="r")
    data = group["data"]
    ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]

    dataset = LeRobotDataset.create(
        repo_id="local/contactworld_usb_positive_train", root=args.output,
        robot_type="contactworld_franka_gripper", fps=10,
        features={
            "action": {"dtype": "float32", "shape": (6,), "names": [f"control_{i}" for i in range(6)]},
            "observation.state": {"dtype": "float32", "shape": (18,),
                                  "names": [f"joint_{i}.pos" for i in range(9)] +
                                           [f"joint_{i}.vel" for i in range(9)]},
            "observation.images.front": {"dtype": "video", "shape": (256, 256, 3),
                                          "names": ["height", "width", "channel"]},
            "observation.images.wrist": {"dtype": "video", "shape": (256, 256, 3),
                                          "names": ["height", "width", "channel"]},
            "observation.images.tactile": {"dtype": "video", "shape": (320, 240, 3),
                                            "names": ["height", "width", "channel"]},
        },
        image_writer_threads=4,
    )
    spatial_root = args.output / "spatial" / "episodes"
    episodes = []
    for output_episode, (source_episode, split) in enumerate(chosen):
        start, end = int(starts[source_episode]), int(ends[source_episode])
        length = end - start
        ep_dir = spatial_root / f"episode_{output_episode:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        cloud = np.asarray(data["pointcloud"][start:end], dtype=np.float32)
        cloud_xyz = cloud[..., :3] @ rotation.T + translation
        force = np.asarray(data["tactile_force_field_right"][start:end], dtype=np.float32)
        depth = np.asarray(data["tactile_depth_right"][start:end], dtype=np.float32)
        pose = np.concatenate((np.asarray(data["ee_pos"][start:end]),
                               np.asarray(data["ee_quat"][start:end])), axis=-1)
        if cloud_xyz.shape != (length, 1024, 3) or force.shape != (length, 10, 14, 3):
            raise ValueError(f"Unexpected spatial shape at source episode {source_episode}")
        if not all(np.isfinite(x).all() for x in (cloud_xyz, force, depth, pose)):
            raise ValueError(f"Nonfinite spatial data at source episode {source_episode}")
        np.save(ep_dir / "pointcloud_xyz.npy", cloud_xyz.astype(np.float32))
        np.save(ep_dir / "force_grid.npy", force.astype(np.float32))
        np.save(ep_dir / "tactile_depth.npy", depth.astype(np.float16))
        np.save(ep_dir / "ee_pose.npy", pose.astype(np.float32))
        for row in range(start, end):
            state = np.r_[data["dof_pos"][row], data["dof_vel"][row]].astype(np.float32)
            action = np.asarray(data["action"][row], dtype=np.float32)
            dataset.add_frame({
                "observation.images.front": as_rgb(data["front"][row]),
                "observation.images.wrist": as_rgb(data["wrist"][row]),
                "observation.images.tactile": as_rgb(data["tactile_rgb_right"][row]),
                "observation.state": state, "action": action,
                "task": "Insert the USB plug into the socket.",
            })
        dataset.save_episode()
        episodes.append({"episode_index": output_episode, "source_episode": source_episode,
                         "split": split, "source_start": start, "length": length})
        print(f"{output_episode + 1}/{len(chosen)} source={source_episode} split={split} frames={length}", flush=True)

    manifest = {
        "format": "contactworld-spatial-v1", "task": "insertion_usb",
        "split": "positive_train_val_test", "source": str(args.source),
        "audit": str(args.audit), "geometry": str(args.geometry),
        "fps": 10, "fps_note": "Index-time convention for LeRobot; original collection FPS unverified",
        "pointcloud_frame": "base, corrected from release using geometry report",
        "force_frame": "right tactile sensor local, source signed units (not calibrated newtons)",
        "episodes": episodes,
    }
    (args.output / "spatial" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    info_path = args.output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.images.front", "observation.images.wrist", "observation.images.tactile"):
        info["features"][key]["info"]["video.codec"] = "h264"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Wrote {len(episodes)} episodes, {sum(e['length'] for e in episodes)} frames to {args.output}")


if __name__ == "__main__":
    main()
