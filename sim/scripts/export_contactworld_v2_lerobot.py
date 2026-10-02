#!/usr/bin/env python3
"""Export reviewed ContactWorld replay rows to isolated LeRobot v2.1 + spatial v2/v3.

Uses same-frame replay RGB/tactile/state and the actual executed feedback
action. This intentionally does not mutate the frozen v1 training dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import lerobot.common.datasets.lerobot_dataset as lerobot_module


lerobot_module.encode_video_frames = functools.partial(
    lerobot_module.encode_video_frames, vcodec="h264", pix_fmt="yuv420p", crf=18)

SPATIAL = {
    "pointcloud_front_xyz": (4096, 3),
    "pointcloud_fused_xyz": (4096, 3),
    "force_grid_local": (10, 14, 3),
    "tactile_xyz_base": (140, 3),
    "tactile_force_base": (140, 3),
    "tactile_depth": (320, 240),
    "ee_pose": (7,),
}
GEOMETRY = ("front_depth", "wrist_depth", "front_view", "front_projection",
            "wrist_view", "wrist_projection", "base_pose_world",
            "tactile_quat_world", "cloud_front_uv", "cloud_fused_uv",
            "cloud_fused_camera")
BILATERAL_SPATIAL = {
    "force_grid_left_local": (10, 14, 3),
    "force_grid_right_local": (10, 14, 3),
    "tactile_xyz_left_base": (140, 3),
    "tactile_xyz_right_base": (140, 3),
    "tactile_force_left_base": (140, 3),
    "tactile_force_right_base": (140, 3),
}
BILATERAL_SOURCE = {
    "force_grid_left_local": "force_grid_left",
    "force_grid_right_local": "force_grid_right",
    "tactile_xyz_left_base": "taxel_xyz_base_left",
    "tactile_xyz_right_base": "taxel_xyz_base_right",
    "tactile_force_left_base": "force_base_left",
    "tactile_force_right_base": "force_base_right",
}


def force_preview(field: np.ndarray) -> np.ndarray:
    """Diagnostic RGB encoding of [normal, shear X, shear Y], never a camera image."""
    if field.shape != (10, 14, 3):
        raise ValueError("Expected one 10x14x3 pad field")
    encoded = np.rint(np.clip(field / 0.003, -1, 1) * 127 + 128).astype(np.uint8)
    return cv2.resize(encoded, (256, 256), interpolation=cv2.INTER_NEAREST)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_v2_full_20261002/input"))
    p.add_argument("--replay", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_v2_full_20261002/replay"))
    p.add_argument("--output", type=Path, default=Path("data/contactworld_usb_v2"))
    p.add_argument("--max-episodes", type=int, help="Bounded export smoke only")
    p.add_argument("--bilateral-tactile", action="store_true",
                   help="Export a distinct v3 dataset with patched left and original right TacFF")
    args = p.parse_args()
    if args.bilateral_tactile and args.output == Path("data/contactworld_usb_v2"):
        p.error("Bilateral export requires a distinct --output, e.g. data/contactworld_usb_v3_bilateral")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite " + str(args.output))
    manifest = json.loads((args.input / "manifest.json").read_text())
    candidates = manifest["episodes"][:args.max_episodes]
    if not candidates:
        raise ValueError("No source episodes")
    selected = []
    excluded = []
    for source in candidates:
        name = "insertion_usb_episode_%03d" % source["episode"]
        folder = args.replay / name
        report = json.loads((folder / "replay.json").read_text())
        if (report["demo_sha256"] != source["sha256"] or
                report["frames"] != source["frames"] or
                not report["v2_capture"] or not (folder / "v2_capture.npz").is_file()):
            raise ValueError("Replay incomplete or mismatched: " + name)
        if args.bilateral_tactile and (not report.get("bilateral_tactile_capture") or
                                       not (folder / "bilateral_tactile.npz").is_file()):
            raise ValueError("Bilateral capture incomplete: " + name)
        if not report["final_source_success"]:
            excluded.append({"source_episode": source["episode"],
                             "split": source["split"],
                             "reason": "replay terminal source success false",
                             "final_plug_error_mm": report["final_plug_error_mm"]})
        else:
            selected.append(source)
    if not selected:
        raise ValueError("No successful replay demonstrations")
    print("Replayed positives: %d/%d accepted; excluded source IDs: %s" % (
        len(selected), len(candidates), [x["source_episode"] for x in excluded]), flush=True)
    image_features = {
        "observation.images.front": {"dtype": "video", "shape": (256, 256, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "video", "shape": (256, 256, 3),
                                     "names": ["height", "width", "channel"]},
    }
    if args.bilateral_tactile:
        for side in ("left", "right"):
            image_features[f"observation.images.tacff_{side}_preview"] = {
                "dtype": "video", "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            }
    else:
        image_features["observation.images.tactile"] = {
            "dtype": "video", "shape": (320, 240, 3),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=("local/contactworld_usb_v3_bilateral" if args.bilateral_tactile else
                 "local/contactworld_usb_v2"), root=args.output,
        robot_type="contactworld_franka_gripper", fps=10,
        features={
            "action": {"dtype": "float32", "shape": (6,),
                       "names": [f"control_{i}" for i in range(6)]},
            "observation.state": {"dtype": "float32", "shape": (18,),
                "names": [f"joint_{i}.pos" for i in range(9)] +
                         [f"joint_{i}.vel" for i in range(9)]},
            **image_features,
        }, image_writer_threads=4)
    receipts = []
    for output_episode, source in enumerate(selected):
        episode = int(source["episode"])
        name = "insertion_usb_episode_%03d" % episode
        folder = args.replay / name
        report = json.loads((folder / "v2_capture.json").read_text())
        capture_path = folder / "v2_capture.npz"
        episode_path = args.output / "spatial/episodes" / (
            "episode_%06d" % output_episode)
        episode_path.mkdir(parents=True, exist_ok=False)
        with contextlib.ExitStack() as stack:
            data = stack.enter_context(np.load(capture_path))
            bilateral = (stack.enter_context(np.load(folder / "bilateral_tactile.npz"))
                         if args.bilateral_tactile else None)
            length = int(source["frames"])
            spatial_fields = (SPATIAL if bilateral is None else {
                key: shape for key, shape in SPATIAL.items()
                if key not in {"force_grid_local", "tactile_xyz_base",
                               "tactile_force_base", "tactile_depth"}
            })
            for out_key, shape in spatial_fields.items():
                source_key = {
                    "pointcloud_front_xyz": "cloud_front_xyz",
                    "pointcloud_fused_xyz": "cloud_fused_xyz",
                }.get(out_key, out_key)
                value = np.asarray(data[source_key])
                if value.shape != (length,) + shape or not np.isfinite(value).all():
                    raise ValueError("Invalid %s in %s: %s" % (out_key, name, value.shape))
                np.save(episode_path / (out_key + ".npy"), value)
            if bilateral is not None:
                for out_key, shape in BILATERAL_SPATIAL.items():
                    value = np.asarray(bilateral[BILATERAL_SOURCE[out_key]])
                    if value.shape != (length,) + shape or not np.isfinite(value).all():
                        raise ValueError("Invalid bilateral %s in %s" % (out_key, name))
                    np.save(episode_path / (out_key + ".npy"), value)
                for source_key, right_key in (("force_grid_right", "force_grid_local"),
                                              ("taxel_xyz_base_right", "tactile_xyz_base"),
                                              ("force_base_right", "tactile_force_base"),
                                              ("taxel_quat_world_right", "tactile_quat_world")):
                    # V2 rounds the fixed base's ~1e-8 m pose jitter to exact
                    # identity for cloud parity; the bilateral audit preserves
                    # the measured base pose. The right field itself is exact.
                    equal = (np.array_equal(bilateral[source_key], data[right_key])
                             if source_key in ("force_grid_right", "taxel_quat_world_right")
                             else np.allclose(bilateral[source_key], data[right_key],
                                              rtol=0, atol=1e-7))
                    if not equal:
                        raise ValueError("Bilateral right does not match original right: " + source_key)
            # Keep the metric source of the point clouds and per-frame camera
            # calibration in the transfer-ready dataset for independent audits.
            geometry = {key: np.asarray(data[key]) for key in GEOMETRY
                        if bilateral is None or key != "tactile_quat_world"}
            if bilateral is not None:
                geometry.update({f"tactile_quat_{side}_world": np.asarray(
                    bilateral[f"taxel_quat_world_{side}"]) for side in ("left", "right")})
            np.savez_compressed(episode_path / "sensor_geometry.npz", **geometry)
            for key, shape in (("front", (256, 256, 3)),
                               ("wrist", (256, 256, 3)),
                               ("tactile_rgb", (320, 240, 3)),
                               ("state", (18,)), ("action_executed", (6,))):
                if data[key].shape != (length,) + shape:
                    raise ValueError("Invalid %s shape in %s" % (key, name))
            for frame in range(length):
                images = ({
                    f"observation.images.tacff_{side}_preview": force_preview(
                        bilateral[f"force_grid_{side}"][frame])
                    for side in ("left", "right")
                } if bilateral is not None else {
                    "observation.images.tactile": data["tactile_rgb"][frame]
                })
                dataset.add_frame({
                    "observation.images.front": data["front"][frame],
                    "observation.images.wrist": data["wrist"][frame],
                    **images,
                    "observation.state": np.asarray(data["state"][frame], dtype=np.float32),
                    "action": np.asarray(data["action_executed"][frame], dtype=np.float32),
                    "task": "Insert the USB plug into the socket.",
                })
            dataset.save_episode()
            action_delta = np.linalg.norm(data["action_executed"] -
                                          data["action_recorded"], axis=-1)
            receipts.append({"episode_index": output_episode,
                             "source_episode": episode, "split": source["split"],
                             "length": length,
                             "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                             **({"bilateral_sha256": hashlib.sha256(
                                 (folder / "bilateral_tactile.npz").read_bytes()).hexdigest()}
                                if bilateral is not None else {}),
                             "max_action_correction_l2": float(action_delta.max()),
                             "max_plug_pos_error_mm": max(x["plug_pos_error_mm"] for x in report["rows"])})
        print("%d/%d source=%d split=%s frames=%d" % (
            output_episode+1, len(selected), episode, source["split"], length), flush=True)
    spatial_manifest = {
        "format": ("contactworld-spatial-v3-bilateral" if args.bilateral_tactile else
                   "contactworld-spatial-v2"), "task": "insertion_usb",
        "selection_manifest": str(args.input / "manifest.json"),
        "capture_root": str(args.replay), "fps": 10,
        "fps_note": "Index-time convention; historical collection FPS unverified",
        "coordinate_frame": "Franka base, fixed world-aligned in this USB scene",
        "pointcloud": "4096 actual front or front+wrist RGB-D points, wide fixed ROI, 5mm voxel, surface-balanced Morton table fill",
        "sensor_geometry": "per-episode sensor_geometry.npz preserves same-frame metric depths, camera matrices, base pose and point/pixel provenance",
        "tactile": ("both simulator tactile pads, 140 taxels each, in Franka base; source force scale not calibrated Newtons"
                    if args.bilateral_tactile else
                    "right simulator tactile sample XYZ and vector force in Franka base; source force scale not calibrated Newtons"),
        "tactile_scope": ("left patched in isolated simulator; original right verified unchanged"
                           if args.bilateral_tactile else
                           "right finger only; no left force field is observed or reconstructed"),
        "tactile_preview": ("paired tacff_left_preview/tacff_right_preview videos encode local normal/shear X/shear Y with zero=128 and fixed +/-0.003 scale; diagnostic only, not model input or raw tactile RGB"
                            if args.bilateral_tactile else "single released right tactile RGB camera video"),
        "tactile_model": "synthetic pad/plug penetration normal and relative-velocity shear; socket axial force is not directly sensed",
        "tactile_base_note": ("bilateral vectors use measured base pose; v2 right/cloud round fixed-base numerical jitter to identity (<1e-7 m difference)"
                              if args.bilateral_tactile else "fixed base rounded to identity for reviewed cloud parity"),
        "actions": "actual pose-feedback actions paired with same replay pre-action observations",
        "selection_rule": "source endpoint-pass and replay terminal env._check_success true; failed replays retained in capture_root",
        "source_candidates": len(candidates), "excluded_replays": excluded,
        "episodes": receipts,
    }
    (args.output / "spatial/manifest.json").write_text(json.dumps(spatial_manifest, indent=2) + "\n")
    info_path = args.output / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in image_features:
        info["features"][key]["info"]["video.codec"] = "h264"
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print("Wrote %d episodes, %d frames to %s" % (
        len(receipts), sum(x["length"] for x in receipts), args.output), flush=True)


if __name__ == "__main__":
    main()
