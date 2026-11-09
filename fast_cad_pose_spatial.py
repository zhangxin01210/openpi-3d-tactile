#!/usr/bin/env python3
"""Run the existing CAD-pose trainer with Spatial v1 scene input.

No project source files are edited. Training-only fast path; visualization and
anything requesting include_rgb=True retain the original RGB-D path.

Usage (from repository root):
    .venv/bin/python -u fast_cad_pose_spatial.py --check
    .venv/bin/python -u fast_cad_pose_spatial.py --train --repo . ...

The --check command compares four actual frames against the old RGB-D pipeline
and fails closed if scene or labels do not match closely.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

TASK_NAMES = ("1018_peg_all", "1028_press_button", "1030_yellow_socket", "1102_eraser")
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def spatial_reader(task):
    obj = getattr(task, "_fast_pose_spatial_reader", None)
    if obj is None:
        from openpi.spatial_dataset.derived import SpatialDerivedDataset
        obj = SpatialDerivedDataset(task.root, version="v1", max_cached_shards=8)
        task._fast_pose_spatial_reader = obj
    return obj


def spatial_scene(task, episode: int, frame: int):
    s = spatial_reader(task).get(episode_index=int(episode), frame_index=int(frame))
    if int(s.episode_index) != int(episode) or int(s.frame_index) != int(frame):
        raise ValueError(f"Spatial v1 frame alignment mismatch: {task.dataset} e{episode} f{frame}")
    xyz = np.asarray(s.visual_xyz_m, dtype=np.float32)
    rgb = np.asarray(s.visual_rgb)
    mask = np.asarray(s.visual_rgb_valid, dtype=np.bool_)
    if xyz.shape != (4096, 3) or rgb.shape != (4096, 3) or mask.shape != (4096,):
        raise ValueError(f"Unexpected Spatial scene shape: {xyz.shape}, {rgb.shape}, {mask.shape}")
    if rgb.dtype != np.uint8 or not np.isfinite(xyz).all() or not mask.any():
        raise ValueError(f"Invalid spatial scene: {task.dataset} e{episode} f{frame}")
    scene = np.empty((4096, 6), dtype=np.float32)
    scene[:, :3] = xyz
    scene[:, 3:] = rgb.astype(np.float32) / 255.0
    return scene, mask, float(s.timestamp_s)


def legacy_spatial_episode(task, episode: int):
    from openpi.cad_pose.legacy import fit_corresponding_rigid_pose

    if episode not in task.episodes:
        raise ValueError(f"unknown episode: {task.dataset} e{episode}")
    frames = [int(f) for f in task.reader.frame_ids(episode)]
    if not frames or frames != list(range(len(frames))):
        raise ValueError(f"non-contiguous geometry frames: {task.dataset} e{episode}")
    for frame in frames:
        geometry = task.reader.get(episode_index=episode, frame_index=frame)
        valid = np.array(
            [np.asarray(getattr(geometry, f"{r}_mask")).any() for r in ("manipulated", "target")],
            dtype=bool,
        )
        if not valid.any():
            continue
        target = np.zeros((2, 4, 4), dtype=np.float32)
        for i, role in enumerate(("manipulated", "target")):
            if valid[i]:
                part_mask = np.asarray(getattr(geometry, f"{role}_mask"), dtype=bool)
                if not part_mask.all():
                    raise ValueError(f"partial fixed-point mask: {task.dataset} e{episode} f{frame} {role}")
                target[i], _ = fit_corresponding_rigid_pose(
                    task.ordered_cad[i],
                    getattr(geometry, f"{role}_xyz_m"),
                    getattr(geometry, f"{role}_mask"),
                    max_rmse_m=0.0001,
                )
        t0 = time.perf_counter()
        scene, mask, timestamp = spatial_scene(task, episode, frame)
        yield {
            "dataset": task.dataset, "episode": episode, "frame": frame,
            "timestamp": timestamp, "scene": scene, "scene_mask": mask,
            "cad": task.ordered_cad, "target_transform": target, "valid": valid,
            "preprocessing_seconds": time.perf_counter() - t0,
        }


def published_spatial_episode(task, episode: int):
    from openpi.cad_pose.data import (
        calibration_provenance_error,
        cloud_from_pose,
        source_rows,
        validate_sampled_pose_frame,
        sha256,
    )

    if episode not in task.episodes:
        raise ValueError(f"unknown episode: {task.dataset} e{episode}")
    root = task.prep / "rgbtrack/episodes" / f"e{episode:05d}"
    run = task.prep / "rgbtrack/runs" / task.release["run_tag"] / "episodes" / f"e{episode:05d}"
    if not root.is_symlink() or root.resolve() != run.resolve():
        raise ValueError(f"published pose symlink differs from release: {task.dataset} e{episode}")
    run_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cal_error = calibration_provenance_error(run_manifest.get("calibration"), task.calibration)
    if cal_error:
        raise ValueError(f"pose run calibration mismatch: {cal_error}")
    rows = []
    for role in ("manipulated", "target"):
        name = task.roles[role]
        path = root / f"{name}_poses.jsonl"
        if not task.quick and sha256(path) != task.resolved_pose_hashes.get(str(episode), {}).get(
            name, task.manifest.get("pose_files_sha256_by_episode", {}).get(str(episode), {}).get(name)
        ):
            raise ValueError(f"pose file changed after audit: {path}")
        rows.append(source_rows(path, object_name=name))
    if len(rows[0]) != len(rows[1]):
        raise ValueError(f"two pose streams have different frame counts: {task.dataset} e{episode}")
    # This training path is used with the approved quick-audit task adapter.
    # Recheck the saved ordered geometry for each consumed pose label.
    if not getattr(task, "quick", False) or not hasattr(task, "geometry_reader"):
        raise ValueError("Fast PublishedPoseTask requires quick-mode geometry validation")
    frames = [int(f) for f in task.geometry_reader.frame_ids(episode)]
    if not frames or frames != list(range(len(frames))) or len(rows[0]) != len(frames):
        raise ValueError(f"pose/geometry frame count mismatch: {task.dataset} e{episode}")
    for frame in frames:
        geometry = task.geometry_reader.get(episode_index=episode, frame_index=frame)
        validate_sampled_pose_frame(
            geometry, [rows[i][frame] for i in range(2)], task.ordered_cad, episode, frame
        )
        valid = np.array([rows[i][frame]["valid"] for i in range(2)], dtype=bool)
        if not valid.any():
            continue
        target = np.zeros((2, 4, 4), dtype=np.float32)
        for i, is_valid in enumerate(valid):
            if is_valid:
                entry = rows[i][frame]
                cloud_from_pose(entry, task.ordered_cad[i])
                target[i] = np.asarray(entry["T_base_link_object"], dtype=np.float32)
        t0 = time.perf_counter()
        scene, mask, timestamp = spatial_scene(task, episode, frame)
        yield {
            "scene": scene, "scene_mask": mask,
            "cad": task.ordered_cad, "target_transform": target,
            "valid": valid, "episode": episode, "frame": frame,
            "timestamp": timestamp, "dataset": task.dataset,
            "preprocessing_seconds": time.perf_counter() - t0,
        }


def install():
    from openpi.cad_pose.data import PublishedPoseTask
    from openpi.cad_pose.legacy import LegacyGeometryPoseTask

    original_published = PublishedPoseTask.iter_episode
    original_legacy = LegacyGeometryPoseTask.iter_episode

    def published(self, episode, *, include_rgb=False):
        if include_rgb or not getattr(self, "quick", False):
            yield from original_published(self, episode, include_rgb=include_rgb)
        else:
            yield from published_spatial_episode(self, episode)

    def legacy(self, episode, *, include_rgb=False):
        if include_rgb:
            yield from original_legacy(self, episode, include_rgb=True)
        else:
            yield from legacy_spatial_episode(self, episode)

    PublishedPoseTask.iter_episode = published
    LegacyGeometryPoseTask.iter_episode = legacy
    return original_published, original_legacy


def check(original_published, original_legacy):
    from openpi.cad_pose.pipeline import open_tasks
    print("[CHECK] Running current quick audit and comparing one real frame per task", flush=True)
    tasks, reports = open_tasks(
        _ROOT, _ROOT / "data", TASK_NAMES,
        quick=True, peg_legacy_intrinsics=True,
    )
    missing = set(TASK_NAMES) - set(tasks)
    if missing:
        for name in sorted(missing):
            print(f"[BLOCKED] {name}: {reports.get(name, {}).get('errors', [])}", flush=True)
        raise SystemExit(1)
    for name, task in tasks.items():
        episode = int(task.episodes[0])
        fast_sample = next(task.iter_episode(episode))
        original_fn = original_published if name in TASK_NAMES[2:] else original_legacy
        old_sample = next(original_fn(task, episode, include_rgb=True))
        same_frame = (fast_sample["episode"], fast_sample["frame"]) == (
            old_sample["episode"], old_sample["frame"]
        )
        same_valid = np.array_equal(fast_sample["valid"], old_sample["valid"])
        same_mask = np.array_equal(fast_sample["scene_mask"], old_sample["scene_mask"])
        same_target = np.allclose(
            fast_sample["target_transform"], old_sample["target_transform"], rtol=0, atol=1e-5
        )
        xyz_max = float(np.max(np.abs(fast_sample["scene"][:, :3] - old_sample["scene"][:, :3])))
        rgb_max = float(np.max(np.abs(fast_sample["scene"][:, 3:] - old_sample["scene"][:, 3:])))
        passed = same_frame and same_valid and same_mask and same_target and xyz_max < 1e-5 and rgb_max < 1e-6
        print(
            f"[{name}] {'PASS' if passed else 'DIFFERENT'}"
            f" e{episode} f{fast_sample['frame']}"
            f" xyz_max={xyz_max:.6g}m rgb_max={rgb_max:.6g}"
            f" mask={'same' if same_mask else 'DIFF'}"
            f" pose={'same' if same_target else 'DIFF'}", flush=True
        )
        if not passed:
            raise SystemExit(
                "[STOP] Spatial v1 differs from live RGB-D preprocessing or supervision. "
                "Do not use fast training until the mismatch is understood."
            )
    print("[OK] Four-task first-frame parity passed. Fast launcher is ready.", flush=True)


def write_provenance(argv):
    """Write source hashes beside, not inside, the trainer's initially empty output dir."""
    if "--output-dir" not in argv:
        return
    idx = argv.index("--output-dir")
    if idx + 1 >= len(argv):
        return
    output = Path(argv[idx + 1])
    for p in (output, output.parent):
        if p == output and p.exists() and any(p.iterdir()):
            raise FileExistsError(f"refusing nonempty training output: {p}")
    manifests = {}
    for name in TASK_NAMES:
        path = _ROOT / "data" / name / "spatial/v1/manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Spatial v1 manifest: {path}")
        manifests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {
        "input_source": "spatial/v1 (four tasks)",
        "training_runtime": "fast_cad_pose_spatial.py",
        "spatial_manifest_sha256": manifests,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_args": argv,
        "note": "Four-task first-frame parity must be checked separately using --check.",
    }
    dest = Path(str(output) + ".spatial_sources.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"[FAST] Spatial v1 source manifest: {dest}", flush=True)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        original_published, original_legacy = install()
        check(original_published, original_legacy)
    elif len(sys.argv) > 1 and sys.argv[1] == "--train":
        argv = sys.argv[2:]
        write_provenance(argv)
        install()
        print("[FAST] Four-task Spatial v1 training data path enabled", flush=True)
        script = str(_ROOT / "scripts/cad_pose/train.py")
        sys.argv = [script, *argv]
        runpy.run_path(script, run_name="__main__")
    else:
        print("Usage: python fast_cad_pose_spatial.py --check | --train [original train.py args...]", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
