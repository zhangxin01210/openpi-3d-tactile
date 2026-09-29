#!/usr/bin/env python3
"""Read-only RealSense RGB-D capture with per-stream frame metadata."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


METADATA_NAMES = (
    "frame_counter", "frame_timestamp", "sensor_timestamp",
    "time_of_arrival", "backend_timestamp", "actual_exposure",
)


def intrinsics_dict(intrinsics) -> dict:
    return {
        "width": intrinsics.width, "height": intrinsics.height,
        "fx": intrinsics.fx, "fy": intrinsics.fy,
        "cx": intrinsics.ppx, "cy": intrinsics.ppy,
        "model": str(intrinsics.model), "coeffs": list(intrinsics.coeffs),
    }


def frame_info(frame, rs) -> dict:
    metadata = {}
    for name in METADATA_NAMES:
        field = getattr(rs.frame_metadata_value, name, None)
        if field is not None and frame.supports_frame_metadata(field):
            metadata[name] = int(frame.get_frame_metadata(field))
    return {
        "frame_number": int(frame.get_frame_number()),
        "timestamp_ms": float(frame.get_timestamp()),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
        "metadata": metadata,
    }


def comparable_delta_ms(color: dict, depth: dict) -> float | None:
    if color["timestamp_domain"] != depth["timestamp_domain"]:
        return None
    return color["timestamp_ms"] - depth["timestamp_ms"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="347622074420")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if (args.width <= 0 or args.height <= 0 or args.fps <= 0
            or args.frames <= 0 or args.warmup_frames < 0):
        raise ValueError("Invalid stream dimensions, FPS, or frame count")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("Run on a machine with pyrealsense2, numpy and opencv-python") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        calibration = {
            "serial": args.serial, "stream": {"width": args.width, "height": args.height,
                                                "fps": args.fps, "color_format": "rgb8",
                                                "depth_format": "z16"},
            "color_intrinsics": intrinsics_dict(color_profile.get_intrinsics()),
            "depth_intrinsics": intrinsics_dict(depth_profile.get_intrinsics()),
            "depth_scale_m_per_unit": float(profile.get_device().first_depth_sensor().get_depth_scale()),
            "T_color_depth": {"rotation_row_major_3x3": np.asarray(extrinsics.rotation).reshape(3, 3).tolist(),
                              "translation_m": list(extrinsics.translation)},
        }
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames()
        output = args.out.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        records = []
        for index in range(args.frames):
            before_ns = time.monotonic_ns()
            frameset = pipeline.wait_for_frames()
            after_ns = time.monotonic_ns()
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError(f"Missing color/depth frame at capture {index}")
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            if color.shape != (args.height, args.width, 3) or depth.shape != (args.height, args.width):
                raise ValueError(f"Unexpected RGB-D shapes: {color.shape}, {depth.shape}")
            color_name = f"frame_{index:04d}_rgb.png"
            depth_name = f"frame_{index:04d}_depth_z16.png"
            if not cv2.imwrite(str(output / color_name), cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write {color_name}")
            if not cv2.imwrite(str(output / depth_name), depth):
                raise RuntimeError(f"Could not write {depth_name}")
            color_meta = frame_info(color_frame, rs)
            depth_meta = frame_info(depth_frame, rs)
            records.append({
                "capture_index": index,
                "host_time_utc": datetime.now(timezone.utc).isoformat(),
                "wait_start_monotonic_ns": before_ns,
                "wait_end_monotonic_ns": after_ns,
                "color_file": color_name, "depth_file": depth_name,
                "color": color_meta, "depth": depth_meta,
                "color_minus_depth_timestamp_ms": comparable_delta_ms(color_meta, depth_meta),
            })
        (output / "frames.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        with (output / "timing.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=(
                "capture_index", "color_frame_number", "depth_frame_number",
                "color_timestamp_ms", "depth_timestamp_ms", "timestamp_domain",
                "color_minus_depth_timestamp_ms", "wait_start_monotonic_ns", "wait_end_monotonic_ns",
            ))
            writer.writeheader()
            for item in records:
                writer.writerow({
                    "capture_index": item["capture_index"],
                    "color_frame_number": item["color"]["frame_number"],
                    "depth_frame_number": item["depth"]["frame_number"],
                    "color_timestamp_ms": item["color"]["timestamp_ms"],
                    "depth_timestamp_ms": item["depth"]["timestamp_ms"],
                    "timestamp_domain": item["color"]["timestamp_domain"]
                    if item["color"]["timestamp_domain"] == item["depth"]["timestamp_domain"] else "different",
                    "color_minus_depth_timestamp_ms": item["color_minus_depth_timestamp_ms"],
                    "wait_start_monotonic_ns": item["wait_start_monotonic_ns"],
                    "wait_end_monotonic_ns": item["wait_end_monotonic_ns"],
                })
        print(f"Wrote {len(records)} raw RGB-D frame pairs and metadata to {output}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
