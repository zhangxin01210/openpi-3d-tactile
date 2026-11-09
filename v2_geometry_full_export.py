#!/usr/bin/env python3
"""Resume-safe four-task V2 RGB-D + single-CAD -> two-role geometry export.

READ-ONLY sources: V2 cached RGB-D, approved CAD, frozen V2 checkpoint.
WRITES ONLY to --output-dir (default: output/cad_pose_rgbd_v2/geometry_full_candidate).
The result is a **candidate**, not an accepted native ObjectGeometryDerivedDataset
sidecar and NOT approved for pi0 training/deployment without integration QA.

Usage from openpi-3d-tactile repository root:
  PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u v2_geometry_full_export.py --episodes-per-task 1
  PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u v2_geometry_full_export.py
  PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u v2_geometry_full_export.py --verify-only

No GT transforms or GT validity masks enter the model forward pass. Labels are
saved separately to support diagnostic comparisons only. A finite predicted
pose is NOT a detected/visible-object confidence estimate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the already cloud-tested checkpoint/model and pilot infer implementation.
import cad_pose_rgbd_single_v2 as v2
import v2_geometry_bridge_pilot as pilot

TASKS = tuple(v2.TASK_NAMES)
ROLES = ("manipulated", "target")
EXPECTED_CAD_SHAPE = (2, 512, 3)
EXPORT_SCHEMA = "v2_pose_pred_full_candidate_v1"


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp_{os.getpid()}")
    if tmp.exists():
        raise RuntimeError(f"leftover temporary path: {tmp}")
    try:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def check_paths(cache: Path, output: Path) -> None:
    if not cache.is_dir():
        raise FileNotFoundError(f"cache not found: {cache}")
    # Never allow writing into the source cache or original data trees.
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("output directory must be separate from RGB-D cache")
    if any(part in ("object_geometry", "checkpoints") for part in output.parts):
        raise ValueError("candidate export must not target native geometry/checkpoint directories")
    root = Path.cwd().resolve()
    allowed = root / "output"
    if not output.is_relative_to(allowed):
        raise ValueError(f"output must be inside {allowed} to prevent modifying original datasets")


def list_sources(cache: Path, tasks: tuple[str, ...]) -> tuple[dict, dict, dict]:
    cfgpath = cache / "cache_config.json"
    splitpath = cache / "episode_splits.json"
    if not (cfgpath.is_file() and splitpath.is_file()):
        raise FileNotFoundError("cache_config.json or episode_splits.json missing")
    config = json.loads(cfgpath.read_text(encoding="utf-8"))
    splits = json.loads(splitpath.read_text(encoding="utf-8"))
    if config.get("schema") != v2.SCHEMA:
        raise ValueError("RGB-D cache schema does not match V2")
    if not isinstance(config.get("width"), int) or not isinstance(config.get("height"), int):
        raise ValueError("missing cache image dimensions")
    all_eps = {}
    for task in tasks:
        if task not in splits:
            raise ValueError(f"task missing from splits: {task}")
        sets = [list(map(int, splits[task][part])) for part in ("train", "val", "test")]
        joined = [ep for group in sets for ep in group]
        if len(joined) != len(set(joined)):
            raise ValueError(f"train/val/test Episode overlap in {task}")
        if not joined:
            raise ValueError(f"task has zero episodes: {task}")
        all_eps[task] = sorted(joined)
        cadpath = cache / task / "cad.npy"
        if not cadpath.is_file():
            raise FileNotFoundError(cadpath)
        cad = np.load(cadpath, mmap_mode="r", allow_pickle=False)
        if cad.shape != EXPECTED_CAD_SHAPE or cad.dtype != np.float32:
            raise ValueError(f"{task}: CAD shape/dtype mismatch {cad.shape}/{cad.dtype}")
        if not np.isfinite(cad).all():
            raise ValueError(f"{task}: CAD has non-finite points")
        for ep in joined:
            meta = cache / task / f"e{ep:05d}" / "meta.json"
            if not meta.is_file():
                raise FileNotFoundError(f"missing expected Episode cache {meta}")
    return config, splits, all_eps


def build_provenance(cache: Path, ckpt_path: Path, ckpt: dict, tasks: tuple[str, ...], splits: dict) -> dict:
    return {
        "schema": EXPORT_SCHEMA,
        "status": "candidate_unaccepted_not_native_object_geometry",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": pilot.sha256(ckpt_path),
        "checkpoint_schema": ckpt["schema"],
        "checkpoint_script_sha256": ckpt.get("script_sha256"),
        "rgbd_cache_dir": str(cache),
        "rgbd_cache_config_sha256": pilot.sha256(cache / "cache_config.json"),
        "rgbd_episode_splits_sha256": pilot.sha256(cache / "episode_splits.json"),
        "image_width": int(ckpt["width"]),
        "image_height": int(ckpt["height"]),
        "depth_scale": float(ckpt["depth_scale"]),
        "tasks": list(tasks),
        "role_names": list(ROLES),
        "role_to_object_by_task": {task: list(pilot.OBJECTS[task]) for task in tasks},
        "cad_sha256_by_task": {task: pilot.sha256(cache / task / "cad.npy") for task in tasks},
        "coordinate_frame": "base_link",
        "length_unit": "meter",
        "pose_layout": "[frame, role, 4,4]",
        "geometry_layout": "[frame, role, 512,3]",
        "frame_index": "same Episode/Frame indices as RGB-D cache; contiguous 0..N-1",
        "inference_inputs_only": ["rgb", "raw_depth", "CAD"],
        "gt_transforms_and_masks_saved_for_diagnostics_only": True,
        "predicted_visibility_or_confidence": False,
        "do_not_use_as_pi0_sidecar_until_native_schema_and_inference_parity_are_verified": True,
    }


def compare_provenance(existing: dict, expected: dict) -> None:
    changes = [k for k, value in expected.items() if existing.get(k) != value]
    if changes:
        raise RuntimeError("existing output uses different sources/schema: " + ", ".join(changes))


def check_episode_source(cache: Path, task: str, ep: int, cfg: dict) -> tuple[int, int]:
    path = cache / task / f"e{ep:05d}"
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    n = int(meta["n"])
    if meta.get("schema") != v2.SCHEMA or meta.get("task") != task or int(meta.get("episode", -1)) != ep:
        raise RuntimeError(f"unexpected Episode metadata in {path}")
    if (int(meta["width"]), int(meta["height"])) != (cfg["width"], cfg["height"]):
        raise RuntimeError(f"RGB-D image dimensions differ at {path}")
    if n < 1:
        raise RuntimeError(f"empty Episode at {path}")
    # Header-only memmap reads: this does not load full frame video/data.
    rgb = np.load(path / "rgb.npy", mmap_mode="r", allow_pickle=False)
    depth = np.load(path / "depth.npy", mmap_mode="r", allow_pickle=False)
    gt = np.load(path / "transform.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(path / "valid.npy", mmap_mode="r", allow_pickle=False)
    if rgb.shape != (n, cfg["height"], cfg["width"], 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"RGB shape/dtype mismatch {path}")
    if depth.shape != (n, cfg["height"], cfg["width"]) or depth.dtype != np.uint16:
        raise RuntimeError(f"Depth shape/dtype mismatch {path}")
    if gt.shape != (n, 2, 4, 4) or valid.shape != (n, 2) or valid.dtype != np.bool_:
        raise RuntimeError(f"GT/valid frame count mismatch {path}")
    return n, int(valid.sum())


def check_predictions(pred_t: np.ndarray, pred_xyz: np.ndarray, cad: np.ndarray, task: str, ep: int) -> None:
    n = len(pred_t)
    if pred_t.shape != (n, 2, 4, 4) or pred_xyz.shape != (n, 2, 512, 3):
        raise RuntimeError(f"{task}/e{ep}: bad output array shape")
    if pred_t.dtype != np.float32 or pred_xyz.dtype != np.float32:
        raise RuntimeError(f"{task}/e{ep}: arrays must be float32")
    if not np.isfinite(pred_t).all() or not np.isfinite(pred_xyz).all():
        raise RuntimeError(f"{task}/e{ep}: nonfinite model output")
    if not np.allclose(pred_t[:, :, 3], [0, 0, 0, 1], atol=1e-6):
        raise RuntimeError(f"{task}/e{ep}: invalid homogeneous transform bottom row")
    r = pred_t[:, :, :3, :3]
    orth = np.matmul(np.swapaxes(r, -1, -2), r)
    if np.max(np.abs(orth - np.eye(3))) > 3e-3 or np.max(np.abs(np.linalg.det(r) - 1)) > 3e-3:
        raise RuntimeError(f"{task}/e{ep}: non-rigid 6D Pose output")
    # Check three frames; full array validation above already covers finite shape.
    for f in sorted({0, n // 2, n - 1}):
        reconstructed = np.einsum("rij,rkj->rki", r[f], cad) + pred_t[f, :, None, :3, 3]
        if np.max(np.abs(reconstructed - pred_xyz[f])) > 1e-5:
            raise RuntimeError(f"{task}/e{ep} f{f}: 512-point geometry not from predicted SE(3)")


def check_saved(path: Path, meta_path: Path, provenance: dict, cache: Path,
                task: str, ep: int, expected_n: int, verify_contents: bool) -> dict:
    if not (path.is_file() and meta_path.is_file()):
        raise RuntimeError(f"partially exported episode: {path} / {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("schema") != EXPORT_SCHEMA or meta.get("task") != task or
            meta.get("episode") != ep or meta.get("frames") != expected_n or
            meta.get("checkpoint_sha256") != provenance["checkpoint_sha256"] or
            meta.get("rgbd_cache_config_sha256") != provenance["rgbd_cache_config_sha256"] or
            meta.get("cad_sha256") != provenance["cad_sha256_by_task"][task]):
        raise RuntimeError(f"existing result provenance/length mismatch: {path}")
    if pilot.sha256(path) != meta.get("file_sha256"):
        raise RuntimeError(f"existing result hash mismatch: {path}")
    if verify_contents:
        with np.load(path, allow_pickle=False) as a:
            expected = {"frame_index", "pred_T_base_object", "candidate_xyz_m",
                        "cad_xyz_local_m", "existing_gt_T_base_object", "existing_gt_label_valid",
                        "role_names", "object_names"}
            if set(a.files) != expected:
                raise RuntimeError(f"{path}: wrong set of array fields")
            idx = a["frame_index"]
            if not np.array_equal(idx, np.arange(expected_n, dtype=np.int64)):
                raise RuntimeError(f"frame_index mismatch: {path}")
            cad = np.load(cache / task / "cad.npy", allow_pickle=False)
            if not np.array_equal(a["cad_xyz_local_m"], cad):
                raise RuntimeError(f"CAD source has changed: {path}")
            if not np.array_equal(a["role_names"], np.asarray(ROLES)):
                raise RuntimeError(f"role ordering mismatch: {path}")
            check_predictions(a["pred_T_base_object"], a["candidate_xyz_m"], cad, task, ep)
            if a["existing_gt_T_base_object"].shape != (expected_n, 2, 4, 4):
                raise RuntimeError(f"GT shape mismatch: {path}")
            if a["existing_gt_label_valid"].shape != (expected_n, 2):
                raise RuntimeError(f"GT validity mismatch: {path}")
    return meta


def write_episode(output: Path, cache: Path, task: str, ep: int, n: int,
                  data: dict, provenance: dict, analysis: dict) -> dict:
    folder = output / task
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"e{ep:05d}_candidate.npz"
    meta_path = folder / f"e{ep:05d}_candidate.json"
    if dest.exists() or meta_path.exists():
        raise FileExistsError(f"output already exists without valid resume: {dest}")
    tmp = folder / f".{dest.name}.tmp_{os.getpid()}"
    if tmp.exists():
        raise RuntimeError(f"leftover partial export {tmp}")
    try:
        with tmp.open("wb") as f:
            np.savez(f,
                frame_index=np.arange(n, dtype=np.int64),
                pred_T_base_object=np.asarray(data["pred_T"], dtype=np.float32),
                candidate_xyz_m=np.asarray(data["pred_xyz"], dtype=np.float32),
                cad_xyz_local_m=np.asarray(data["cad"], dtype=np.float32),
                existing_gt_T_base_object=np.asarray(data["gt_T"], dtype=np.float32),
                existing_gt_label_valid=np.asarray(data["gt_valid"], dtype=np.bool_),
                role_names=np.asarray(ROLES),
                object_names=np.asarray(pilot.OBJECTS[task]),
            )
        with np.load(tmp, allow_pickle=False) as reader:
            if not np.array_equal(reader["frame_index"], np.arange(n, dtype=np.int64)):
                raise RuntimeError("atomic export frame index mismatch")
        file_sha = pilot.sha256(tmp)
        meta = {
            "schema": EXPORT_SCHEMA,
            "status": "candidate_unaccepted_not_native_object_geometry",
            "task": task, "episode": ep, "frames": n,
            "valid_gt_labels_diagnostic_only": int(data["gt_valid"].sum()),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "rgbd_cache_config_sha256": provenance["rgbd_cache_config_sha256"],
            "cad_sha256": provenance["cad_sha256_by_task"][task],
            "file_sha256": file_sha,
            "bytes": tmp.stat().st_size,
            "seconds": round(float(data["seconds"]), 3),
            "analysis_against_noisy_gt": analysis,
            "visibility": "unknown_not_estimated",
        }
        # Write metadata before data publication. A crash leaves only a hidden temp
        # file and a .json; next run stops explicitly instead of overwriting.
        write_json_atomic(meta_path, meta)
        os.replace(tmp, dest)
        return meta
    finally:
        tmp.unlink(missing_ok=True)


def load_model(checkpoint_json: Path, device: torch.device) -> tuple[dict, Path, torch.nn.Module]:
    pointer = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    ckpt_path = Path(pointer["checkpoint"]).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != v2.SCHEMA or not {"width", "height", "depth_scale", "model_state_dict"} <= ckpt.keys():
        raise RuntimeError("invalid V2 single-CAD checkpoint contract")
    model = v2.SingleCadRGBDPose()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt, ckpt_path, model.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=Path("output/cad_pose_rgbd_v2/cache"))
    p.add_argument("--checkpoint-json", type=Path, default=Path("output/cad_pose_rgbd_v2/train_20261109_162628/best.json"))
    p.add_argument("--output-dir", type=Path, default=Path("output/cad_pose_rgbd_v2/geometry_full_candidate"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--episodes-per-task", type=int, default=0,
                   help="0=ALL Episode IDs from train/val/test split; 1=fast four-task pilot; auto resume")
    p.add_argument("--verify-only", action="store_true", help="verify every exported episode without GPU")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.batch_size > 256 or args.episodes_per_task < 0:
        p.error("invalid batch size / episodes-per-task")
    cache = args.cache_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    check_paths(cache, output)
    cfg, splits, episodes = list_sources(cache, TASKS)
    ckpt_path = Path(json.loads(args.checkpoint_json.read_text())["checkpoint"]).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    # Read only checkpoint header for --verify-only; no GPU is initialized.
    if args.verify_only:
        saved = output / "provenance.json"
        if not saved.is_file():
            raise FileNotFoundError(saved)
        prov = json.loads(saved.read_text(encoding="utf-8"))
        current = dict(prov)
        if prov.get("checkpoint_sha256") != pilot.sha256(ckpt_path):
            raise RuntimeError("checkpoint has changed since export")
        if prov.get("rgbd_cache_config_sha256") != pilot.sha256(cache / "cache_config.json"):
            raise RuntimeError("RGB-D cache config has changed")
        if prov.get("rgbd_episode_splits_sha256") != pilot.sha256(cache / "episode_splits.json"):
            raise RuntimeError("Episode split mapping has changed")
        for task in TASKS:
            if prov.get("cad_sha256_by_task", {}).get(task) != pilot.sha256(cache / task / "cad.npy"):
                raise RuntimeError(f"{task}: CAD changed since export")
        model = None
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        ckpt, real_path, model = load_model(args.checkpoint_json, device)
        if (ckpt["width"], ckpt["height"]) != (cfg["width"], cfg["height"]):
            raise RuntimeError("checkpoint and cached RGB-D dimensions differ")
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        prov = build_provenance(cache, ckpt_path, ckpt, TASKS, splits)
        output.mkdir(parents=True, exist_ok=True)
        saved = output / "provenance.json"
        if saved.exists():
            compare_provenance(json.loads(saved.read_text(encoding="utf-8")), prov)
        else:
            write_json_atomic(saved, prov)

    selected = {task: ids[:args.episodes_per_task] if args.episodes_per_task else ids
                for task, ids in episodes.items()}
    total = sum(len(v) for v in selected.values())
    print(f"[SOURCE] episodes={sum(map(len, episodes.values()))} cached split union; selected={total}; "
          f"ckpt={ckpt_path.name}; dir={output}", flush=True)
    n_done, n_skip, n_frames, labels = 0, 0, 0, 0
    stats_by_task = {t: {"episodes": 0, "frames": 0, "gt_valid_labels_diagnostic_only": 0} for t in TASKS}
    t0 = time.monotonic()
    for task in TASKS:
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for ix, ep in enumerate(selected[task], 1):
            n, cache_labels = check_episode_source(cache, task, ep, cfg)
            path = task_dir / f"e{ep:05d}_candidate.npz"
            meta_path = task_dir / f"e{ep:05d}_candidate.json"
            if path.exists() or meta_path.exists():
                meta = check_saved(path, meta_path, prov, cache, task, ep, n,
                                   verify_contents=args.verify_only)
                n_skip += 1
                status = "SKIP" if not args.verify_only else "VERIFY"
            else:
                if args.verify_only:
                    raise FileNotFoundError(f"missing export {path}")
                data = pilot.infer_episode(model, ckpt, cache, task, ep, device, args.batch_size)
                if len(data["pred_T"]) != n or data["gt_valid"].shape != (n, 2):
                    raise RuntimeError(f"{task}/e{ep}: prediction/cache frame alignment mismatch")
                if int(data["gt_valid"].sum()) != cache_labels:
                    raise RuntimeError(f"{task}/e{ep}: GT validity changed after cache creation")
                check_predictions(data["pred_T"], data["pred_xyz"], data["cad"], task, ep)
                meta = write_episode(output, cache, task, ep, n, data, prov,
                                     pilot.analyze(data, task, ep))
                n_done += 1
                status = "OK"
            stats_by_task[task]["episodes"] += 1
            stats_by_task[task]["frames"] += n
            stats_by_task[task]["gt_valid_labels_diagnostic_only"] += cache_labels
            n_frames += n
            labels += cache_labels
            if not args.quiet or ix == len(selected[task]):
                print(f"[{status}] {task} {ix}/{len(selected[task])} e{ep:05d} "
                      f"frames={n} labels={cache_labels}", flush=True)
        print(f"[TASK] {task}: {stats_by_task[task]['episodes']} episodes, "
              f"{stats_by_task[task]['frames']} frames", flush=True)

    is_full = all(len(selected[t]) == len(episodes[t]) for t in TASKS)
    report = {
        "schema": EXPORT_SCHEMA,
        "status": "candidate_complete_unaccepted" if is_full else "candidate_partial_unaccepted",
        "source_checkpoint_sha256": prov["checkpoint_sha256"],
        "source_cache_config_sha256": prov["rgbd_cache_config_sha256"],
        "episodes": n_done + n_skip,
        "frames": n_frames,
        "gt_valid_labels_diagnostic_only": labels,
        "newly_exported_episodes": n_done,
        "resumed_existing_episodes": n_skip,
        "by_task": stats_by_task,
        "all_split_episodes_exported": is_full,
        "seconds": round(time.monotonic() - t0, 3),
        "warning": "Candidate not a native ObjectGeometry sidecar. No visibility/confidence; not approved for pi0 training/deployment",
    }
    write_json_atomic(output / "summary.json", report)
    if is_full:
        print(f"[FINAL] FULL CANDIDATE: {n_done+n_skip} episodes, {n_frames} frames; "
              "all train/val/test splits included, original data untouched.", flush=True)
    else:
        print(f"[FINAL] PARTIAL PILOT: {n_done+n_skip} episodes, {n_frames} frames; "
              "re-run without --episodes-per-task to resume all.", flush=True)
    print(f"[SUMMARY] {output/'summary.json'}", flush=True)
    print("[SAFE] V1 geometry, pi0 configs/checkpoints, and raw RGB-D unchanged.", flush=True)


if __name__ == "__main__":
    main()
