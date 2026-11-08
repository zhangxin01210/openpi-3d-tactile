"""Epoch-safe training and evaluation for the standalone CAD pose network."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from openpi.cad_pose.data import PublishedPoseTask
from openpi.cad_pose.metrics import pose_errors
from openpi.cad_pose.metrics import summarize
from openpi.cad_pose.model import XHandCADPoseTransformerV1
from openpi.cad_pose.model import pose_loss
from openpi.cad_pose.runtime import points_sha256
from openpi.cad_pose.split import split_episodes
from scripts.cad_pose.preflight import ROLES
from scripts.cad_pose.preflight import audit_task

DEFAULT_TASKS = tuple(ROLES)
PROFILE = "configs/ur7e_xhand/front_calibration_candidate.json"


class SyntheticTask:
    """Explicitly synthetic samples for executable pipeline smoke tests only."""

    def __init__(self, name):
        self.dataset = name
        self.roles = ROLES[name]
        self.episodes = tuple(range(20))
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        self.ordered_cad = np.random.default_rng(seed).uniform(-0.05, 0.05, (2, 512, 3)).astype(np.float32)

    def iter_episode(self, episode, *, include_rgb=False):
        for frame in range(16):
            rng = np.random.default_rng(episode * 100 + frame)
            target = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
            target[:, :3, 3] = [[0.1, 0, 0.7], [0.15, 0, 0.8]]
            sample = {
                "scene": rng.random((256, 6), dtype=np.float32),
                "scene_mask": np.ones(256, bool),
                "cad": self.ordered_cad,
                "target_transform": target,
                "valid": np.ones(2, bool),
                "episode": episode,
                "frame": frame,
                "timestamp": frame / 30,
                "dataset": self.dataset,
            }
            if include_rgb:
                sample["rgb"] = np.zeros((128, 128, 3), np.uint8)
            yield sample


def open_tasks(
    repo, data_root, names, profile=PROFILE, *, synthetic=False, quick=False, timeout=120, peg_legacy_intrinsics=False
):
    tasks, reports = {}, {}
    for index, name in enumerate(names, 1):
        if name not in ROLES or name in tasks:
            raise ValueError(f"unknown or repeated dataset {name}")
        if synthetic:
            tasks[name], reports[name] = SyntheticTask(name), {"status": "synthetic"}
        else:
            if quick:
                from openpi.cad_pose.quick import run_quick_task

                print(f"[{index}/{len(names)}] {name}: quick audit", flush=True)
                entry = run_quick_task(
                    Path(repo),
                    Path(data_root),
                    name,
                    profile,
                    timeout=timeout,
                    allow_legacy_intrinsics=peg_legacy_intrinsics,
                )
                print(f"[{index}/{len(names)}] {name}: {entry['status']} {entry.get('errors', [])}", flush=True)
            else:
                entry = audit_task(Path(repo), Path(data_root), name, profile)
            reports[name] = entry
            if entry["checks"].get("pose_hash_and_alignment") != ("sampled_pass" if quick else "passed"):
                continue
            try:
                if entry["checks"].get("adapter") == "legacy_ordered_fit":
                    from openpi.cad_pose.legacy import LegacyGeometryPoseTask

                    tasks[name] = LegacyGeometryPoseTask(Path(repo), Path(data_root), profile, entry)
                elif entry["checks"].get("adapter") == "button_ordered_fit":
                    from openpi.cad_pose.legacy import ButtonGeometryPoseTask

                    tasks[name] = ButtonGeometryPoseTask(Path(repo), Path(data_root), profile, entry)
                else:
                    tasks[name] = PublishedPoseTask(
                        Path(repo), Path(data_root), name, profile=profile, audit_entry=entry
                    )
                tasks[name].audit_adapter = entry["checks"].get("adapter", "published_full")
                tasks[name].audit_mode = "quick" if quick else "full"
                tasks[name].peg_legacy_intrinsics = peg_legacy_intrinsics and name == "1018_peg_all"
            except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
                reports[name] = {**entry, "status": "blocked", "errors": [*entry["errors"], str(exc)]}
    return tasks, reports


def task_adapters(tasks):
    return {
        name: {
            "adapter": getattr(t, "audit_adapter", "synthetic"),
            "mode": getattr(t, "audit_mode", "synthetic"),
            "peg_legacy_intrinsics": getattr(t, "peg_legacy_intrinsics", False),
        }
        for name, t in tasks.items()
    }


def checkpoint_open_options(ckpt):
    """Recreate the same label adapter and explicit calibration exception."""
    adapters = ckpt.get("data_adapters", {})
    if not adapters:
        return {"quick": False, "peg_legacy_intrinsics": False}
    quick = any(
        x.get("mode") == "quick"
        or ("mode" not in x and x.get("adapter") in {"button_ordered_fit", "published_sampled"})
        for x in adapters.values()
    )
    return {
        "quick": quick,
        "peg_legacy_intrinsics": adapters.get("1018_peg_all", {}).get("peg_legacy_intrinsics", False),
    }


def splits_for(tasks, seed=42):
    return {name: split_episodes(list(task.episodes), seed) for name, task in tasks.items()}


def batchify(items, device):
    return {
        k: torch.as_tensor(np.stack([x[k] for x in items]), device=device)
        for k in ("scene", "scene_mask", "cad", "target_transform", "valid")
    }


def symmetry_tensors(items, tasks, symmetry, device):
    identity = np.eye(3, dtype=np.float32)[None]
    rows = []
    for sample in items:
        task = tasks[sample["dataset"]]
        row = []
        for role in ("manipulated", "target"):
            key = sample["dataset"] + "/" + task.roles[role]
            rotations = np.asarray(symmetry.get(key, identity), dtype=np.float32)
            if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or len(rotations) < 1:
                raise ValueError(f"{key}: symmetries must be [S,3,3]")
            if not all(
                np.allclose(r.T @ r, np.eye(3), atol=1e-4) and np.isclose(np.linalg.det(r), 1, atol=1e-4)
                for r in rotations
            ):
                raise ValueError(f"{key}: invalid symmetry rotation")
            if not any(np.allclose(r, np.eye(3), atol=1e-4) for r in rotations):
                raise ValueError(f"{key}: symmetries must include identity")
            row.append(rotations)
        rows.append(row)
    maximum = max(len(r) for row in rows for r in row)
    values = np.tile(np.eye(3, dtype=np.float32), (len(items), 2, maximum, 1, 1))
    mask = np.zeros((len(items), 2, maximum), bool)
    for i, row in enumerate(rows):
        for j, rotations in enumerate(row):
            values[i, j, : len(rotations)] = rotations
            mask[i, j, : len(rotations)] = True
    return torch.as_tensor(values, device=device), torch.as_tensor(mask, device=device)


def epoch_samples(task, episodes, seed):
    rng = np.random.default_rng(seed)
    while True:
        produced = False
        for episode in rng.permutation(episodes):
            for sample in task.iter_episode(int(episode)):
                produced = True
                yield sample
        if not produced:
            raise ValueError(f"{task.dataset}: no valid training samples")


def balanced_batches(tasks, splits, *, batch_size, steps, seed):
    names = tuple(tasks)
    if batch_size < 1 or steps < 1 or not names or batch_size * steps % len(names):
        raise ValueError("batch size, steps, and task count must be positive")
    streams = {name: epoch_samples(tasks[name], splits[name]["train"], seed + i) for i, name in enumerate(names)}
    order = itertools.cycle(names)
    for _ in range(steps):
        yield [next(streams[name]) for name in (next(order) for _ in range(batch_size))]


def evaluate_loss(model, tasks, splits, device, *, batches=100, batch_size=8, symmetry=None):
    model.eval()
    task_means = []
    with torch.inference_mode():
        for name, task in tasks.items():
            pending, losses = [], []
            for episode in splits[name]["val"]:
                for sample in task.iter_episode(episode):
                    pending.append(sample)
                    if len(pending) < batch_size:
                        continue
                    b = batchify(pending, device)
                    sym, sym_mask = symmetry_tensors(pending, tasks, symmetry or {}, device)
                    loss = pose_loss(
                        model(b["scene"], b["scene_mask"], b["cad"]),
                        b["target_transform"],
                        b["cad"],
                        b["valid"],
                        symmetry_rotations=sym,
                        symmetry_mask=sym_mask,
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite validation loss")
                    losses.append(float(loss))
                    pending.clear()
                    if batches and len(losses) >= batches:
                        break
                if batches and len(losses) >= batches:
                    break
            if pending:
                b = batchify(pending, device)
                sym, sym_mask = symmetry_tensors(pending, tasks, symmetry or {}, device)
                losses.append(
                    float(
                        pose_loss(
                            model(b["scene"], b["scene_mask"], b["cad"]),
                            b["target_transform"],
                            b["cad"],
                            b["valid"],
                            symmetry_rotations=sym,
                            symmetry_mask=sym_mask,
                        )
                    )
                )
            if not losses or not np.isfinite(losses).all():
                raise ValueError(f"{name}: validation produced no finite labels")
            task_means.append(np.mean(losses))
    return float(np.mean(task_means))


def camera_hashes(repo, profile):
    root = Path(repo)
    profile = Path(profile)
    if not profile.is_absolute():
        profile = root / profile
    paths = {
        "profile": profile,
        "camera": root / "configs/ur7e_xhand/cameras_0903.json",
        "extrinsic": root / "configs/ur7e_xhand/extrinsic_0909.yaml",
    }
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}


def train(
    *,
    tasks,
    splits,
    output,
    device="cpu",
    epochs=2,
    batch_size=8,
    steps=1000,
    lr=3e-4,
    seed=42,
    resume=None,
    repo=None,
    profile=PROFILE,
    val_batches=100,
    patience=10,
    synthetic=False,
    symmetry=None,
):
    output = Path(output)
    symmetry = symmetry or {}
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    torch.manual_seed(seed)
    model = XHandCADPoseTransformerV1().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    cad_hashes = {name: points_sha256(t.ordered_cad) for name, t in tasks.items()}
    provenance = {
        name: (
            {
                "manifest": hashlib.sha256((t.root / "object_geometry/v1/manifest.json").read_bytes()).hexdigest(),
                "raw_metadata": hashlib.sha256((t.root / "meta/info.json").read_bytes()).hexdigest(),
            }
            if not synthetic
            else {"synthetic": "true"}
        )
        for name, t in tasks.items()
    }
    cameras = {} if synthetic else camera_hashes(repo, profile)
    split_json = {k: {s: list(e) for s, e in v.items()} for k, v in splits.items()}
    first, best, bad = 0, float("inf"), 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            saved["model_version"] != "xhand_cad_pose_transformer_v1"
            or saved["synthetic"] != synthetic
            or saved["ordered_cad_sha256_by_dataset"] != cad_hashes
            or saved["camera_hashes"] != cameras
            or saved["splits"] != split_json
            or saved.get("provenance") != provenance
            or ("data_adapters" in saved and saved["data_adapters"] != task_adapters(tasks))
            or saved["seed"] != seed
            or saved["batch_size"] != batch_size
            or saved["steps"] != steps
            or saved.get("symmetry", {}) != symmetry
        ):
            raise ValueError("resume checkpoint model/data/CAD/calibration/split mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        opt.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        first, best, bad = saved["epoch"] + 1, saved["best_validation_loss"], saved["bad_epochs"]
    if first >= epochs:
        raise ValueError("epochs must exceed the completed checkpoint epoch count")
    log_path = output / "training.jsonl"
    for epoch in range(first, epochs):
        torch.manual_seed(seed + epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start, total = time.perf_counter(), 0.0
        task_counts = dict.fromkeys(tasks, 0)
        for items in balanced_batches(tasks, splits, batch_size=batch_size, steps=steps, seed=seed + epoch):
            for item in items:
                task_counts[item["dataset"]] += 1
            b = batchify(items, device)
            # Mild per-frame brightness perturbation; geometry and GT remain unchanged.
            brightness = torch.empty((len(items), 1, 1), device=device).uniform_(0.9, 1.1)
            b["scene"][..., 3:] = (b["scene"][..., 3:] * brightness).clamp(0, 1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                prediction = model(b["scene"], b["scene_mask"], b["cad"])
                sym, sym_mask = symmetry_tensors(items, tasks, symmetry, device)
                loss = pose_loss(
                    prediction,
                    b["target_transform"],
                    b["cad"],
                    b["valid"],
                    symmetry_rotations=sym,
                    symmetry_mask=sym_mask,
                )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("non-finite training gradient")
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
        seconds = time.perf_counter() - start
        val = evaluate_loss(model, tasks, splits, device, batches=val_batches, batch_size=batch_size, symmetry=symmetry)
        improved = val < best
        best, bad = (val, 0) if improved else (best, bad + 1)
        record = {
            "epoch": epoch,
            "train_loss": total / steps,
            "validation_loss": val,
            "train_seconds": seconds,
            "epoch_seconds": time.perf_counter() - start,
            "samples_per_second": batch_size * steps / seconds,
            "task_sample_counts": task_counts,
            "peak_gpu_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
            "synthetic": synthetic,
            "symmetry": list(symmetry),
        }
        saved = {
            "model_version": "xhand_cad_pose_transformer_v1",
            "data_adapters": task_adapters(tasks),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best,
            "bad_epochs": bad,
            "splits": split_json,
            "ordered_cad_sha256_by_dataset": cad_hashes,
            "camera_hashes": cameras,
            "synthetic": synthetic,
            "provenance": provenance,
            "seed": seed,
            "batch_size": batch_size,
            "steps": steps,
            "symmetry": symmetry,
        }
        path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        with path.open("xb") as f:
            torch.save(saved, f)
        if improved:
            (output / "best.json").write_text(
                json.dumps({"checkpoint": str(path.resolve()), "epoch": epoch, "validation_loss": val}) + "\n"
            )
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if bad >= patience:
            break
    return model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_version") != "xhand_cad_pose_transformer_v1":
        raise ValueError("unsupported checkpoint")
    model = XHandCADPoseTransformerV1().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval(), ckpt


def verify_checkpoint_sources(ckpt, tasks, repo, profile, *, synthetic):
    if bool(ckpt.get("synthetic")) != synthetic or set(ckpt["splits"]) != set(tasks):
        raise ValueError("checkpoint task set or synthetic mode differs")
    if "data_adapters" in ckpt and ckpt["data_adapters"] != task_adapters(tasks):
        raise ValueError("pose data adapter or legacy calibration option differs from checkpoint")
    if not synthetic and ckpt["camera_hashes"] != camera_hashes(repo, profile):
        raise ValueError("camera calibration differs from checkpoint")
    for name, task in tasks.items():
        if ckpt["ordered_cad_sha256_by_dataset"][name] != points_sha256(task.ordered_cad):
            raise ValueError(f"{name}: CAD differs from checkpoint")
        expected = (
            {"synthetic": "true"}
            if synthetic
            else {
                "manifest": hashlib.sha256((task.root / "object_geometry/v1/manifest.json").read_bytes()).hexdigest(),
                "raw_metadata": hashlib.sha256((task.root / "meta/info.json").read_bytes()).hexdigest(),
            }
        )
        if ckpt.get("provenance", {}).get(name) != expected:
            raise ValueError(f"{name}: dataset provenance differs from checkpoint")
        groups = ckpt["splits"][name]
        ids = [episode for split in ("train", "val", "test") for episode in groups[split]]
        if len(ids) != len(set(ids)) or set(ids) != set(task.episodes):
            raise ValueError(f"{name}: split is stale or contains episode leakage")


def predict(model, sample, device):
    b = batchify([sample], device)
    with torch.inference_mode():
        out = model(b["scene"], b["scene_mask"], b["cad"])
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[:, :3, :3] = out["rotation"][0].float().cpu().numpy()
    poses[:, :3, 3] = out["translation"][0].float().cpu().numpy()
    if not np.isfinite(poses).all():
        raise ValueError("pose prediction is non-finite")
    return poses


def report_metrics(rows):
    groups = {}
    for row in rows:
        for key in ("overall", row["dataset"], row["dataset"] + "/" + row["object"]):
            groups.setdefault(key, []).append(row)
    return {
        key: {
            metric: summarize([r[metric] for r in values])
            for metric in ("translation_mm", "rotation_deg", "add_mm", "adds_mm")
        }
        for key, values in groups.items()
    }


def make_metric_rows(sample, predicted, roles, symmetry=None):
    from scipy.spatial import cKDTree

    rows = []
    for i, role in enumerate(("manipulated", "target")):
        if not sample["valid"][i]:
            continue
        target, cad = sample["target_transform"][i], sample["cad"][i]
        key = sample["dataset"] + "/" + roles[role]
        result = pose_errors(predicted[i], target, cad, (symmetry or {}).get(key))
        pred_points = cad @ predicted[i, :3, :3].T + predicted[i, :3, 3]
        gt_points = cad @ target[:3, :3].T + target[:3, 3]
        adds = float(cKDTree(gt_points).query(pred_points)[0].mean() * 1000)
        rows.append(
            {
                "dataset": sample["dataset"],
                "episode": int(sample["episode"]),
                "frame": int(sample["frame"]),
                "role": role,
                "object": roles[role],
                "translation_mm": result["translation_mm"],
                "rotation_deg": result["rotation_deg"],
                "add_mm": result["cad_error_mm"],
                "adds_mm": adds,
            }
        )
    return rows
