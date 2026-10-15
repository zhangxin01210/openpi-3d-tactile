#!/usr/bin/env python3
"""Verify ContactWorld replay-based LeRobot v2/v3 and spatial sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq
import cv2
from scipy.spatial.transform import Rotation


SPATIAL_SHAPES = {
    "pointcloud_front_xyz": (4096, 3),
    "pointcloud_fused_xyz": (4096, 3),
    "force_grid_local": (10, 14, 3),
    "tactile_xyz_base": (140, 3),
    "tactile_force_base": (140, 3),
    "tactile_depth": (320, 240),
    "ee_pose": (7,),
}
BILATERAL_SHAPES = {
    "force_grid_left_local": (10, 14, 3),
    "force_grid_right_local": (10, 14, 3),
    "tactile_xyz_left_base": (140, 3),
    "tactile_xyz_right_base": (140, 3),
    "tactile_force_left_base": (140, 3),
    "tactile_force_right_base": (140, 3),
}


def expected_force_preview(field: np.ndarray) -> np.ndarray:
    encoded = np.rint(np.clip(field / 0.003, -1, 1) * 127 + 128).astype(np.uint8)
    return cv2.resize(encoded, (256, 256), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--capture-root", type=Path,
                        help="Optional stronger alignment audit against local replay archives")
    args = parser.parse_args()
    if shutil.which("ffprobe") is None:
        parser.error("ffprobe command is required")
    root = args.root
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "spatial/manifest.json").read_text())
    bilateral = manifest["format"] == "contactworld-spatial-v3-bilateral"
    if manifest["format"] not in ("contactworld-spatial-v2", "contactworld-spatial-v3-bilateral"):
        raise ValueError("Wrong spatial version")
    expected_roles = ({"front", "wrist", "tacff_left_preview", "tacff_right_preview"}
                      if bilateral else {"front", "wrist", "tactile"})
    video_roles = {key.removeprefix("observation.images.") for key, value in info["features"].items()
                   if key.startswith("observation.images.") and value["dtype"] == "video"}
    if video_roles != expected_roles:
        raise ValueError(f"Video roles differ: {video_roles} != {expected_roles}")
    episodes = manifest["episodes"]
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode count mismatch")
    total = 0
    splits = {"train": 0, "val": 0, "test": 0}
    for item in episodes:
        episode = int(item["episode_index"])
        length = int(item["length"])
        splits[item["split"]] += 1
        chunk = episode // 1000
        table = pq.read_table(root / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
                              columns=["episode_index", "frame_index", "action", "observation.state"])
        if (table.num_rows != length or set(table["episode_index"].to_pylist()) != {episode} or
                table["frame_index"].to_pylist() != list(range(length))):
            raise ValueError("Parquet row/episode mismatch: %d" % episode)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if action.shape != (length, 6) or state.shape != (length, 18):
            raise ValueError("Parquet action/state shape mismatch: %d" % episode)
        spatial = root / f"spatial/episodes/episode_{episode:06d}"
        if bilateral and any((spatial / (key + ".npy")).exists() for key in (
                "force_grid_local", "tactile_xyz_base", "tactile_force_base", "tactile_depth")):
            raise ValueError("Unpaired right-only tactile arrays in V3 episode %d" % episode)
        arrays = {}
        shapes = (SPATIAL_SHAPES if not bilateral else {
            key: shape for key, shape in SPATIAL_SHAPES.items()
            if key not in {"force_grid_local", "tactile_xyz_base", "tactile_force_base", "tactile_depth"}
        } | BILATERAL_SHAPES)
        for key, shape in shapes.items():
            value = np.load(spatial / (key + ".npy"), mmap_mode="r")
            if value.shape != (length,) + shape or not np.isfinite(value).all():
                raise ValueError("Invalid %s episode %d" % (key, episode))
            arrays[key] = value
        with np.load(spatial / "sensor_geometry.npz") as geometry:
            geometry_keys = (
                "front_depth", "wrist_depth", "front_view", "front_projection",
                "wrist_view", "wrist_projection", "base_pose_world",
                "cloud_front_uv", "cloud_fused_uv", "cloud_fused_camera",
            ) + (("tactile_quat_left_world", "tactile_quat_right_world") if bilateral else
                 ("tactile_quat_world",))
            for key in geometry_keys:
                if key not in geometry or len(geometry[key]) != length or not np.isfinite(geometry[key]).all():
                    raise ValueError("Invalid sensor geometry %s episode %d" % (key, episode))
            for frame in (0, length // 2, length - 1):
                base_quat = geometry["base_pose_world"][frame, 3:]
                for side in (("left", "right") if bilateral else ("right",)):
                    force_key = f"tactile_force_{side}_base" if bilateral else "tactile_force_base"
                    grid_key = f"force_grid_{side}_local" if bilateral else "force_grid_local"
                    quat_key = f"tactile_quat_{side}_world" if bilateral else "tactile_quat_world"
                    force_world = Rotation.from_quat(base_quat).apply(arrays[force_key][frame])
                    force_local = Rotation.from_quat(geometry[quat_key][frame]).inv().apply(force_world)
                    reconstructed = np.stack((-force_local[:, 1], -force_local[:, 0],
                                              force_local[:, 2]), axis=-1)
                    if not np.allclose(reconstructed, arrays[grid_key][frame].reshape(140, 3),
                                       atol=1e-6, rtol=0):
                        raise ValueError("3D force axis/sign round trip failed: %s %d frame %d" %
                                         (side, episode, frame))
        sides = ("left", "right") if bilateral else ("right",)
        for side in sides:
            force_key = f"tactile_force_{side}_base" if bilateral else "tactile_force_base"
            grid_key = f"force_grid_{side}_local" if bilateral else "force_grid_local"
            if not np.allclose(np.linalg.norm(arrays[force_key], axis=-1),
                               np.linalg.norm(arrays[grid_key].reshape(length, 140, 3), axis=-1),
                               atol=1e-6):
                raise ValueError("Tactile force magnitude changed: %s %d" % (side, episode))
        if bilateral:
            if (not np.any(np.linalg.norm(arrays["tactile_force_left_base"], axis=-1) > 1e-6) or
                    not np.any(np.linalg.norm(arrays["tactile_force_right_base"], axis=-1) > 1e-6)):
                raise ValueError("A tactile pad has no signal: %d" % episode)
            if np.array_equal(arrays["tactile_force_left_base"], arrays["tactile_force_right_base"]):
                raise ValueError("Left and right tactile force arrays are identical: %d" % episode)
            pad_distance = np.linalg.norm(
                arrays["tactile_xyz_left_base"].mean(axis=1) -
                arrays["tactile_xyz_right_base"].mean(axis=1), axis=-1)
            if not np.all(pad_distance > 0.005):
                raise ValueError("Left and right pad geometry overlaps: %d" % episode)
        if args.capture_root:
            capture = args.capture_root / ("insertion_usb_episode_%03d/v2_capture.npz" %
                                            item["source_episode"])
            with np.load(capture) as source:
                if (not np.array_equal(action, source["action_executed"]) or
                        not np.array_equal(state, source["state"])):
                    raise ValueError("Replay action/state alignment differs: %d" % episode)
                source_pairs = (("pointcloud_front_xyz", "cloud_front_xyz"),
                                        ("pointcloud_fused_xyz", "cloud_fused_xyz"),
                                        ) + (() if bilateral else (
                                        ("force_grid_local", "force_grid_local"),
                                        ("tactile_xyz_base", "tactile_xyz_base"),
                                        ("tactile_force_base", "tactile_force_base")))
                for key, source_key in source_pairs:
                    if not np.array_equal(arrays[key], source[source_key]):
                        raise ValueError("Replay spatial alignment differs: %s %d" % (key, episode))
                if bilateral:
                    bilateral_path = args.capture_root / ("insertion_usb_episode_%03d/bilateral_tactile.npz" %
                                                          item["source_episode"])
                    if hashlib.sha256(bilateral_path.read_bytes()).hexdigest() != item["bilateral_sha256"]:
                        raise ValueError("Bilateral archive changed: %d" % episode)
                    with np.load(bilateral_path) as both:
                        for side in ("left", "right"):
                            for key, source_key in ((f"force_grid_{side}_local", f"force_grid_{side}"),
                                                    (f"tactile_xyz_{side}_base", f"taxel_xyz_base_{side}"),
                                                    (f"tactile_force_{side}_base", f"force_base_{side}")):
                                if not np.array_equal(arrays[key], both[source_key]):
                                    raise ValueError("Bilateral spatial alignment differs: %s %d" % (key, episode))
                with np.load(spatial / "sensor_geometry.npz") as geometry:
                    for key in geometry.files:
                        if key in ("tactile_quat_left_world", "tactile_quat_right_world"):
                            with np.load(args.capture_root / ("insertion_usb_episode_%03d/bilateral_tactile.npz" %
                                                              item["source_episode"])) as both:
                                side = "left" if key == "tactile_quat_left_world" else "right"
                                expected = both[f"taxel_quat_world_{side}"]
                        else:
                            expected = source[key]
                        if not np.array_equal(geometry[key], expected):
                            raise ValueError("Sensor calibration/depth differs: %s %d" % (key, episode))
        roles = ("front", "wrist", "tacff_left_preview", "tacff_right_preview") if bilateral else (
            "front", "wrist", "tactile")
        for role in roles:
            video = root / f"videos/chunk-{chunk:03d}/observation.images.{role}/episode_{episode:06d}.mp4"
            result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,pix_fmt,nb_frames", "-of", "json", str(video)],
                capture_output=True, text=True, check=True)
            stream = json.loads(result.stdout)["streams"][0]
            if (stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p" or
                    int(stream["nb_frames"]) != length):
                raise ValueError("Video codec/frame mismatch: " + str(video))
            if bilateral and role.startswith("tacff_"):
                side = role.split("_")[1]
                capture = cv2.VideoCapture(str(video))
                if not capture.isOpened():
                    raise ValueError("Cannot decode " + str(video))
                for frame in (0, length // 2, length - 1):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError("Preview frame missing: " + str(video))
                    expected = expected_force_preview(arrays[f"force_grid_{side}_local"][frame])
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if np.abs(rgb.astype(np.int16) - expected.astype(np.int16)).mean() > 5:
                        raise ValueError("Preview does not match %s force grid: %s" % (side, video))
                capture.release()
            elif args.capture_root and role in ("front", "wrist"):
                capture = cv2.VideoCapture(str(video))
                with np.load(args.capture_root / ("insertion_usb_episode_%03d/v2_capture.npz" %
                                                  item["source_episode"])) as source:
                    for frame in (0, length // 2, length - 1):
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                        ok, bgr = capture.read()
                        if not ok:
                            raise ValueError("RGB video frame missing: " + str(video))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if np.abs(rgb.astype(np.int16) - source[role][frame].astype(np.int16)).mean() > 5:
                            raise ValueError("RGB video differs from source: " + str(video))
                capture.release()
        total += length
    if total != info["total_frames"]:
        raise ValueError("Total frame mismatch")
    print(json.dumps({"status": "passed", "bilateral_tactile": bilateral,
                      "episodes": len(episodes),
                      "frames": total, "episode_splits": splits,
                      "source_capture_checked": bool(args.capture_root)}, indent=2))


if __name__ == "__main__":
    main()
