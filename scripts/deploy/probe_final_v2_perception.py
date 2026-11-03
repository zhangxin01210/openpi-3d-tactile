"""Exercise the real final-v2 task perception worker on raw LeRobot frames.

Use the perception environment, with PYTHONPATH pointing at the chosen checkout.
Baseline calibration is explicit here, matching the task config with no profile
environment override. This does not establish calibration parity with training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "scripts/object_geometry/pose_tracking")]
    report = {
        "task": args.task,
        "robot_command_sent": False,
        "calibration": "baseline_unprofiled",
        "training_calibration_parity": "UNKNOWN: exported training manifests have no calibration provenance",
        "real_sam2": True,
        "real_rgbtrack": True,
    }
    start = time.perf_counter()
    try:
        from rgbtrack_adapter import _load_front_camera_for_depth_registration
        from rgbtrack_adapter import _rgb_registered_depth

        from openpi.spatial_dataset.source import RawSpatialDataset
        from scripts.deploy.export_training_episode_packets import export_episode
        from scripts.deploy.run_online_perception import _run_task_geometry
        from scripts.deploy.run_online_perception import _task_geometry_inputs

        camera = _load_front_camera_for_depth_registration(root, None)
        raw = output / "raw"
        export_episode(
            dataset_root=root / "data" / args.task,
            episode=0,
            max_frames=args.frames,
            output_dir=raw,
            camera=camera,
            register_depth=_rgb_registered_depth,
            reader_class=RawSpatialDataset,
        )
        report["raw_export"] = "PASS"
        spec = _task_geometry_inputs(root / "output/preprocess" / args.task / "geometry/approved/manifest.json", root)
        report["asset_validation"] = "PASS"
        dest = output / "perception"
        dest.mkdir()
        (dest / "frames").mkdir()
        worker_args = SimpleNamespace(
            sam2_backend="video_memory",
            register_missing_color_depth=True,
            sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
            sam2_source=Path.home() / ".cache/openpi_deploy/sam2",
            sam2_checkpoint=root / "checkpoints/sam2/sam2.1_hiera_small.pt",
            front_calibration_profile=None,
            weights_root=root / "checkpoints/rgbtrack",
            min_mask_area_px=32,
            max_mask_fraction=0.8,
            min_sam2_score=0.05,
            device="cuda",
            follow=False,
            max_frames=args.frames,
        )
        _run_task_geometry(worker_args, spec, root, raw, dest)
        report["status"] = "PASS"
        report["frames_written"] = len(list((dest / "frames").glob("frame_*.npz")))
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    report["wall_s"] = time.perf_counter() - start
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
