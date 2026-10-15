#!/usr/bin/env python3
"""Export reviewed bilateral replays into a task-isolated LeRobot V3 dataset."""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import lerobot.common.datasets.lerobot_dataset as lerobot_module

from contactworld_cross_task_common import TASKS, get_task

lerobot_module.encode_video_frames = functools.partial(
    lerobot_module.encode_video_frames, vcodec="h264", pix_fmt="yuv420p", crf=18)

SPATIAL = {"pointcloud_front_xyz": (4096, 3), "pointcloud_fused_xyz": (4096, 3),
           "ee_pose": (7,)}
BILATERAL = {"force_grid_left_local": (10, 14, 3), "force_grid_right_local": (10, 14, 3),
             "tactile_xyz_left_base": (140, 3), "tactile_xyz_right_base": (140, 3),
             "tactile_force_left_base": (140, 3), "tactile_force_right_base": (140, 3)}
GEOMETRY = ("front_depth", "wrist_depth", "front_view", "front_projection", "wrist_view",
            "wrist_projection", "base_pose_world", "cloud_front_uv", "cloud_fused_uv", "cloud_fused_camera")
SOURCE = {"force_grid_left_local": "force_grid_left", "force_grid_right_local": "force_grid_right",
          "tactile_xyz_left_base": "taxel_xyz_base_left", "tactile_xyz_right_base": "taxel_xyz_base_right",
          "tactile_force_left_base": "force_base_left", "tactile_force_right_base": "force_base_right"}


def preview(field: np.ndarray) -> np.ndarray:
    encoded = np.rint(np.clip(field / .003, -1, 1) * 127 + 128).astype(np.uint8)
    return cv2.resize(encoded, (256, 256), interpolation=cv2.INTER_NEAREST)


def split(index: int, total: int) -> str:
    # Entire source episodes are separated; no temporal leakage across splits.
    if index < max(1, round(total * .8)):
        return "train"
    if index < max(1, round(total * .9)):
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    spec = get_task(args.task)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite " + str(args.output))
    input_manifest = json.loads((args.input / "manifest.json").read_text())
    if input_manifest.get("task") != spec.key:
        raise ValueError("Wrong source manifest task")
    candidates = input_manifest["episodes"][:args.max_episodes]
    selected, excluded = [], []
    for source in candidates:
        name = "%s_episode_%03d" % (spec.source_group, source["source_episode"])
        folder = args.replay / name
        report_path = folder / "replay.json"
        if not report_path.exists() or not (folder / "v2_capture.npz").is_file() or not (folder / "bilateral_tactile.npz").is_file():
            raise ValueError("Missing bilateral replay capture: " + str(folder))
        report = json.loads(report_path.read_text())
        if report.get("demo_sha256") != source["sha256"] or report.get("frames") != source["frames"]:
            raise ValueError("Replay receipt does not match fixed input: " + name)
        # Source success is retained for auditing, but a demo is not dropped solely
        # because reconstruction is imperfect; export remains an exact replay record.
        selected.append(source)
    if not selected:
        raise ValueError("No complete replay episodes")
    images = {f"observation.images.{role}": {"dtype": "video", "shape": (256, 256, 3),
              "names": ["height", "width", "channel"]} for role in
              ("front", "wrist", "tacff_left_preview", "tacff_right_preview")}
    dataset = LeRobotDataset.create(repo_id="local/" + spec.output_dataset, root=args.output,
        robot_type="contactworld_franka_gripper", fps=10,
        features={"action": {"dtype": "float32", "shape": (spec.action_dim,),
                              "names": ["control_%d" % i for i in range(spec.action_dim)]},
                  "observation.state": {"dtype": "float32", "shape": (18,),
                  "names": ["joint_%d.pos" % i for i in range(9)] + ["joint_%d.vel" % i for i in range(9)]},
                  **images}, image_writer_threads=4)
    receipts = []
    for output_index, source in enumerate(selected):
        source_episode = int(source["source_episode"])
        name = "%s_episode_%03d" % (spec.source_group, source_episode)
        folder = args.replay / name
        out = args.output / ("spatial/episodes/episode_%06d" % output_index)
        out.mkdir(parents=True, exist_ok=False)
        capture_path, bilateral_path = folder / "v2_capture.npz", folder / "bilateral_tactile.npz"
        with contextlib.ExitStack() as stack:
            data = stack.enter_context(np.load(capture_path))
            bilateral = stack.enter_context(np.load(bilateral_path))
            length = int(source["frames"])
            for key, shape in SPATIAL.items():
                source_key = {"pointcloud_front_xyz": "cloud_front_xyz", "pointcloud_fused_xyz": "cloud_fused_xyz"}.get(key, key)
                value = np.asarray(data[source_key])
                if value.shape != (length,) + shape or not np.isfinite(value).all():
                    raise ValueError("Invalid %s in %s" % (key, name))
                np.save(out / (key + ".npy"), value)
            for key, shape in BILATERAL.items():
                value = np.asarray(bilateral[SOURCE[key]])
                if value.shape != (length,) + shape or not np.isfinite(value).all():
                    raise ValueError("Invalid %s in %s" % (key, name))
                np.save(out / (key + ".npy"), value)
            for source_key, right_key in (("force_grid_right", "force_grid_local"),
                                          ("taxel_xyz_base_right", "tactile_xyz_base"),
                                          ("force_base_right", "tactile_force_base"),
                                          ("taxel_quat_world_right", "tactile_quat_world")):
                if source_key in bilateral and right_key in data and not np.allclose(bilateral[source_key], data[right_key], atol=1e-6, rtol=0):
                    raise ValueError("Right tactile changed between capture streams: " + name)
            geometry = {key: np.asarray(data[key]) for key in GEOMETRY}
            geometry.update({"tactile_quat_%s_world" % side: np.asarray(bilateral["taxel_quat_world_" + side])
                             for side in ("left", "right")})
            np.savez_compressed(out / "sensor_geometry.npz", **geometry)
            expected = {"front": (256, 256, 3), "wrist": (256, 256, 3), "state": (18,),
                        "action_executed": (spec.action_dim,)}
            for key, shape in expected.items():
                if np.asarray(data[key]).shape != (length,) + shape:
                    raise ValueError("Invalid %s shape in %s: %s" % (key, name, np.asarray(data[key]).shape))
            for frame in range(length):
                dataset.add_frame({"observation.images.front": data["front"][frame],
                    "observation.images.wrist": data["wrist"][frame],
                    "observation.images.tacff_left_preview": preview(bilateral["force_grid_left"][frame]),
                    "observation.images.tacff_right_preview": preview(bilateral["force_grid_right"][frame]),
                    "observation.state": np.asarray(data["state"][frame], dtype=np.float32),
                    "action": np.asarray(data["action_executed"][frame], dtype=np.float32), "task": spec.prompt})
            dataset.save_episode()
            correction = np.linalg.norm(np.asarray(data["action_executed"]) - np.asarray(data["action_recorded"]), axis=-1)
            replay = json.loads((folder / "replay.json").read_text())
            receipts.append({"episode_index": output_index, "source_episode": source_episode,
                "split": split(output_index, len(selected)), "length": length,
                "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                "bilateral_sha256": hashlib.sha256(bilateral_path.read_bytes()).hexdigest(),
                "replay_final_source_success": replay.get("final_source_success"),
                "max_action_correction_l2": float(correction.max())})
        print("%d/%d source=%d frames=%d" % (output_index + 1, len(selected), source_episode, length), flush=True)
    manifest = {"format": "contactworld-spatial-v3-bilateral", "task": spec.key,
        "source_group": spec.source_group, "action_dim": spec.action_dim, "prompt": spec.prompt,
        "selection_manifest": str(args.input / "manifest.json"), "capture_root": str(args.replay), "fps": 10,
        "coordinate_frame": "Franka base; point cloud and both tactile taxel positions/forces use the same frame",
        "pointcloud": "4096 front or fused RGB-D points, base ROI/voxel/surface-balanced sampling inherited from USB V3",
        "tactile": "both isolated-simulator TacFF pads, 140 taxels each, local force grids plus base XYZ and 3D force; not calibrated Newtons",
        "tactile_preview": "diagnostic local normal/shear-X/shear-Y encoding; not a camera image and not a model image input",
        "actions": "actual pose-feedback action paired with replay pre-action observation; all source action dimensions are retained",
        "episodes": receipts, "excluded_replays": excluded}
    (args.output / "spatial/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    info_path = args.output / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in images:
        info["features"][key]["info"]["video.codec"] = "h264"
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print("Wrote %d episodes / %d frames to %s" % (len(receipts), sum(row["length"] for row in receipts), args.output))


if __name__ == "__main__":
    main()
