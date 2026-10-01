#!/usr/bin/env python3
"""Run the real UniVTAC insert_hole scene and inspect all requested sensors."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from unproject_univtac_depth import unproject_opengl, write_binary_ply

RUNTIME_SOURCE = Path(__file__).resolve().parents[3] / "openpi-sim-runtime" / "third_party" / "UniVTAC-full"
sys.path.insert(0, str(RUNTIME_SOURCE))

from envs.utils.env_parser import build_task_env_cfg, load_task_config
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "univtac_insert_hole_depth.yml")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--until", choices=("construct", "reset"), default="reset")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1

config, config_path = load_task_config(args.config)
launcher = AppLauncher(args)
app = launcher.app

import torch
from PIL import Image


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main() -> None:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"pass": False, "stage": "launch", "config": str(config_path), "seed": args.seed}
    task = None
    try:
        task_module, cfg, timing, _ = build_task_env_cfg(
            "insert_hole", config, args.config.stem, "collect", save_dir=out / "scratch"
        )
        cfg.reset_time_limit = 600.0
        cfg.save_frequency = 0
        cfg.video_frequency = 0
        report["physx_overrides"] = dict(config.physx_overrides)
        task = task_module.Task(cfg, mode="collect")
        report["stage"] = "constructed"
        report["camera_names"] = list(task._camera_manager.cameras)
        report["tactile_names"] = list(task._tactile_manager.tactiles)
        assert set(report["camera_names"]) == {"head", "wrist"}
        assert len(report["tactile_names"]) == 2
        if args.until == "construct":
            report["pass"] = True
            return

        task.reset(seed=args.seed)
        report["stage"] = "reset"
        for _ in range(args.steps):
            task._step(is_save=False)
        task._update_render()
        obs = task._get_observations()
        report["stage"] = "observations"
        report["steps_after_reset"] = args.steps
        report["camera"] = {}
        for name in ("head", "wrist"):
            camera = obs["observation"][name]
            rgb = _array(camera["rgb"])
            depth = _array(camera["depth"])
            intrinsic = _array(camera["intrinsic"])
            pose = _array(camera["pose_w_opengl"])
            finite = np.isfinite(depth) & (depth > 0)
            assert rgb.ndim == 3 and rgb.shape[-1] == 3, (name, rgb.shape)
            assert depth.shape == (*rgb.shape[:2], 1), (name, depth.shape)
            assert intrinsic.shape == (3, 3), (name, intrinsic.shape)
            assert pose.shape == (7,), (name, pose.shape)
            assert float(rgb.mean()) > 1.0, name
            assert float(finite.mean()) > 0.01, name
            assert np.isfinite(intrinsic).all() and np.isfinite(pose).all(), name
            Image.fromarray(rgb).save(out / f"{name}_rgb.png")
            np.save(out / f"{name}_depth.npy", depth)
            points, pixel_indices = unproject_opengl(depth, intrinsic, pose, stride=4, max_depth_m=3.0)
            assert len(points) > 100, (name, len(points))
            write_binary_ply(out / f"{name}_cloud_stride4.ply", points, rgb.reshape(-1, 3)[pixel_indices])
            report["camera"][name] = {
                "rgb_shape": list(rgb.shape), "rgb_mean": float(rgb.mean()),
                "depth_shape": list(depth.shape), "valid_depth_fraction": float(finite.mean()),
                "valid_depth_min_m": float(depth[finite].min()),
                "valid_depth_max_m": float(depth[finite].max()),
                "intrinsic": intrinsic.tolist(), "pose_w_opengl": pose.tolist(),
                "pointcloud_count_stride4": int(len(points)),
            }

        report["tactile"] = {}
        for name, sensor in obs["tactile"].items():
            rgb = _array(sensor["rgb"])
            press = _array(sensor["press_depth"])
            assert rgb.ndim == 3 and rgb.shape[-1] == 3, (name, rgb.shape)
            assert press.shape == rgb.shape[:2], (name, press.shape)
            assert np.isfinite(press).all(), name
            Image.fromarray(rgb).save(out / f"{name}_rgb.png")
            np.save(out / f"{name}_press_depth_mm.npy", press)
            report["tactile"][name] = {
                "rgb_shape": list(rgb.shape), "rgb_mean": float(rgb.mean()),
                "press_depth_shape": list(press.shape),
                "press_depth_max_mm": float(press.max()),
            }
        report["pass"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if task is not None:
            task.close()
        app.close()
        if not report["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
