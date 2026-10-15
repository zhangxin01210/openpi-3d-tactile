#!/usr/bin/env python3
"""Export synchronized UniVTAC camera and tactile streams from one HDF5 episode."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np


def decode(frame: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a JPEG frame")
    # UniVTAC fed RGB tensors directly to cv2.imencode. imdecode returns the
    # same channel values, so convert them before OpenCV's BGR VideoWriter.
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def panel(image: np.ndarray, title: str, width: int = 480, height: int = 270) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


def press_panel(values: np.ndarray, title: str, scale: float) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(np.clip(values / scale * 255, 0, 255)), cv2.COLORMAP_TURBO)
    return panel(colored, f"{title} max={float(np.max(values)):.4g}")


def depth_panel(values: np.ndarray, title: str, scale_m: float) -> np.ndarray:
    depth = np.asarray(values).squeeze(-1)
    valid = np.isfinite(depth) & (depth > 0)
    intensity = np.uint8(np.clip((scale_m - np.where(valid, depth, scale_m)) / scale_m * 255, 0, 255))
    colored = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return panel(colored, f"{title} valid={float(valid.mean()):.0%}")


def export(episode: Path, output: Path, fps: float, stride: int, max_frames: int | None,
           include_camera_depth: bool = False, encoder: str = "av1_nvenc") -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(episode, "r") as root:
        count = len(root["step"])
        indices = list(range(0, count, stride))
        if max_frames is not None:
            indices = indices[:max_frames]
        if not indices:
            raise ValueError(f"No frames to export from {episode}")
        if include_camera_depth and any(
            f"observation/{name}/depth" not in root for name in ("head", "wrist")
        ):
            raise ValueError("This episode has no head/wrist world-camera depth")
        # A shared scale makes left/right press depth visually comparable.
        scale = max(
            float(root["tactile/left_tactile/press_depth"][:].max()),
            float(root["tactile/right_tactile/press_depth"][:].max()),
            1e-6,
        )
        output_height = 1080 if include_camera_depth else 810
        encoder_args = {
            "av1_nvenc": ["-c:v", "av1_nvenc", "-preset", "p4", "-cq", "30"],
            "libsvtav1": ["-c:v", "libsvtav1", "-preset", "10", "-crf", "35"],
            "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"],
        }[encoder]
        temporary = output.with_name(output.stem + ".part.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "bgr24", "-video_size", f"960x{output_height}",
            "-framerate", str(fps / stride), "-i", "pipe:0", "-an", *encoder_args,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(temporary),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in indices:
                parts = []
                for camera in ("head", "wrist"):
                    parts.append(panel(decode(root[f"observation/{camera}/rgb"][i]), f"{camera} RGB"))
                if include_camera_depth:
                    for camera, scale_m in (("head", 2.0), ("wrist", 0.5)):
                        parts.append(depth_panel(root[f"observation/{camera}/depth"][i],
                                                 f"{camera} world depth (m)", scale_m))
                for side in ("left", "right"):
                    key = f"tactile/{side}_tactile"
                    parts.append(panel(decode(root[f"{key}/rgb"][i]), f"{side} tactile RGB"))
                for side in ("left", "right"):
                    values = root[f"tactile/{side}_tactile/press_depth"][i]
                    parts.append(press_panel(values, f"{side} press depth", scale))
                frame = np.vstack([np.hstack(parts[j:j + 2]) for j in range(0, len(parts), 2)])
                cv2.putText(frame, f"episode={episode.stem} row={i} sim_step={int(root['step'][i])}",
                            (8, output_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode("utf-8", errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed for {episode}: {error}")
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames",
                 "-of", "json", str(temporary)],
                check=True, capture_output=True, text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            expected_codec = "h264" if encoder == "libx264" else "av1"
            if (stream["codec_name"] != expected_codec or stream["pix_fmt"] != "yuv420p"
                    or int(stream["nb_read_frames"]) != len(indices)
                    or (stream["width"], stream["height"]) != (960, output_height)):
                raise RuntimeError(f"Unexpected video stream for {episode}: {stream}")
            temporary.replace(output)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            temporary.unlink(missing_ok=True)
            raise
        finally:
            process.stderr.close()
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
    return {"source": str(episode), "output": str(output), "source_frames": count,
            "video_frames": len(indices), "fps": fps / stride,
            "duration_s": len(indices) / (fps / stride), "codec": expected_codec,
            "camera_depth_included": include_camera_depth, "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=60.0, help="Recorded frame rate before stride")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--include-camera-depth", action="store_true")
    parser.add_argument("--encoder", choices=("av1_nvenc", "libsvtav1", "libx264"),
                        default="av1_nvenc", help="AV1 GPU matches the playable reference video")
    args = parser.parse_args()
    if args.fps <= 0 or args.stride < 1:
        parser.error("fps must be positive and stride at least one")
    result = export(args.episode, args.output, args.fps, args.stride, args.max_frames,
                    args.include_camera_depth, args.encoder)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
