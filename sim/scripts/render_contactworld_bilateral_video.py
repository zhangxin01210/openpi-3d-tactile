"""Synchronize one USB replay's RGB views with both TacFF pad maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
from matplotlib import colormaps
import numpy as np


def force_image(field: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip((field / scale + 1) / 2, 0, 1)
    color = (colormaps["RdBu_r"](normalized)[..., :3] * 255).astype(np.uint8)
    return cv2.resize(color, (256, 256), interpolation=cv2.INTER_NEAREST)


def labeled(image: np.ndarray, label: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (255, 26), (24, 24, 24), thickness=-1)
    cv2.putText(image, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v2_capture", type=Path)
    parser.add_argument("bilateral_capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.v2_capture) as visual, np.load(args.bilateral_capture) as tactile:
        n = len(visual["front"])
        if n != len(tactile["force_grid_left"]):
            raise ValueError("Visual and tactile frame counts differ")
        scales = {}
        for channel in (0, 2):
            values = np.concatenate((tactile["force_grid_left"][..., channel].ravel(),
                                     tactile["force_grid_right"][..., channel].ravel()))
            scales[channel] = max(float(np.quantile(np.abs(values), 0.995)), 1e-6)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", "768x512", "-r", "10",
                   "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(args.output)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for t in range(n):
            left = tactile["force_grid_left"][t]
            right = tactile["force_grid_right"][t]
            panels = [
                labeled(visual["front"][t], f"front RGB  frame {t:03d}"),
                labeled(force_image(left[..., 0], scales[0]), "left normal"),
                labeled(force_image(right[..., 0], scales[0]), "right normal"),
                labeled(visual["wrist"][t], "wrist RGB"),
                labeled(force_image(left[..., 2], scales[2]), "left shear Y"),
                labeled(force_image(right[..., 2], scales[2]), "right shear Y"),
            ]
            frame = np.concatenate((np.concatenate(panels[:3], axis=1),
                                    np.concatenate(panels[3:], axis=1)), axis=0)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to render bilateral review video")
        print(args.output, "frames", n, "normal_scale", scales[0],
              "shear_y_scale", scales[2])


if __name__ == "__main__":
    main()
