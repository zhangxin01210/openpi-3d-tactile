#!/usr/bin/env python3
"""Fail-fast integrity verifier for a cross-task bilateral LeRobot export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pyarrow.parquet as pq

from contactworld_cross_task_common import TASKS, get_task

SPATIAL = {"pointcloud_front_xyz": (4096, 3), "pointcloud_fused_xyz": (4096, 3), "ee_pose": (7,)}
BILATERAL = {"force_grid_left_local": (10, 14, 3), "force_grid_right_local": (10, 14, 3),
             "tactile_xyz_left_base": (140, 3), "tactile_xyz_right_base": (140, 3),
             "tactile_force_left_base": (140, 3), "tactile_force_right_base": (140, 3)}


def expected_preview(field: np.ndarray) -> np.ndarray:
    value = np.rint(np.clip(field / .003, -1, 1) * 127 + 128).astype(np.uint8)
    return cv2.resize(value, (256, 256), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    spec, root = get_task(args.task), args.root
    if shutil.which("ffprobe") is None:
        parser.error("ffprobe is required for LeRobot video validation")
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "spatial/manifest.json").read_text())
    if manifest.get("format") != "contactworld-spatial-v3-bilateral" or manifest.get("task") != spec.key:
        raise ValueError("Wrong sidecar task or format")
    if manifest.get("action_dim") != spec.action_dim or len(manifest["episodes"]) != info["total_episodes"]:
        raise ValueError("Manifest action or episode count mismatch")
    expected_roles = {"front", "wrist", "tacff_left_preview", "tacff_right_preview"}
    roles = {key.removeprefix("observation.images.") for key, value in info["features"].items()
             if key.startswith("observation.images.") and value["dtype"] == "video"}
    if roles != expected_roles:
        raise ValueError("Video roles mismatch: %s" % roles)
    frames_total = 0
    for item in manifest["episodes"]:
        ep, length = int(item["episode_index"]), int(item["length"])
        table = pq.read_table(root / ("data/chunk-%03d/episode_%06d.parquet" % (ep // 1000, ep)),
                              columns=["episode_index", "frame_index", "action", "observation.state"])
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if (table.num_rows != length or action.shape != (length, spec.action_dim) or state.shape != (length, 18) or
                table["frame_index"].to_pylist() != list(range(length))):
            raise ValueError("Parquet state/action/frame alignment failed: episode %d" % ep)
        spatial = root / ("spatial/episodes/episode_%06d" % ep)
        arrays = {}
        for key, shape in (SPATIAL | BILATERAL).items():
            array = np.load(spatial / (key + ".npy"), mmap_mode="r")
            if array.shape != (length,) + shape or not np.isfinite(array).all():
                raise ValueError("Invalid %s episode %d" % (key, ep))
            arrays[key] = array
        if np.array_equal(arrays["tactile_force_left_base"], arrays["tactile_force_right_base"]):
            raise ValueError("Left/right tactile force was duplicated: episode %d" % ep)
        # A gripping pair can be closer than 5 mm.  Detect genuine duplicated
        # pad geometry (zero separation) without rejecting a valid close grip.
        pad_distance = np.linalg.norm(
            arrays["tactile_xyz_left_base"].mean(1)
            - arrays["tactile_xyz_right_base"].mean(1),
            axis=-1,
        )
        if not np.all(pad_distance > .001):
            raise ValueError(
                "Left/right tactile geometry overlaps: episode %d; "
                "minimum pad-centre separation=%.6f m" % (ep, pad_distance.min())
            )
        with np.load(spatial / "sensor_geometry.npz") as geometry:
            for key in ("front_depth", "wrist_depth", "front_view", "front_projection", "wrist_view", "wrist_projection",
                        "base_pose_world", "cloud_front_uv", "cloud_fused_uv", "cloud_fused_camera",
                        "tactile_quat_left_world", "tactile_quat_right_world"):
                if key not in geometry or len(geometry[key]) != length or not np.isfinite(geometry[key]).all():
                    raise ValueError("Missing/nonfinite geometry %s episode %d" % (key, ep))
        for role in sorted(expected_roles):
            video = root / ("videos/chunk-%03d/observation.images.%s/episode_%06d.mp4" % (ep // 1000, role, ep))
            output = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt,nb_frames", "-of", "json", str(video)],
                capture_output=True, text=True, check=True)
            stream = json.loads(output.stdout)["streams"][0]
            if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p" or int(stream["nb_frames"]) != length:
                raise ValueError("Video encoding/frame mismatch: " + str(video))
            if role.startswith("tacff_"):
                side = role.split("_")[1]
                reader = cv2.VideoCapture(str(video))
                for frame in (0, length // 2, length - 1):
                    reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, bgr = reader.read()
                    if not ok or np.abs(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int16) -
                                         expected_preview(arrays["force_grid_%s_local" % side][frame]).astype(np.int16)).mean() > 5:
                        raise ValueError("TacFF preview mismatch: " + str(video))
                reader.release()
        frames_total += length
    if frames_total != info["total_frames"]:
        raise ValueError("Total frame count mismatch")
    print(json.dumps({"status": "passed", "task": spec.key, "episodes": len(manifest["episodes"]),
                      "frames": frames_total, "action_dim": spec.action_dim}, indent=2))


if __name__ == "__main__":
    main()
