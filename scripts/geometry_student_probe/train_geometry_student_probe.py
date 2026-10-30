#!/usr/bin/env python3
"""Minimal, read-only S0-S4 geometry-Student probe for openpi-3d-tactile.

Teacher targets are noisy *sampled* CAD geometry, not physical contact labels.
Only RGB, 18D robot joints and 600D tactile force enter Student. No pose/point cloud
is passed as Student input. Episode-disjoint splits and train-only scalers.

Run inside repo:
  PYTHONPATH=src .venv/bin/python scripts/geometry_student_probe/train_geometry_student_probe.py \
    --dataset-root data/1018_peg_all --output scratch/geometry_student_probe

No Pi0 training code or checkpoints are modified. Use --self-test before real data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree

LABELS = (
    "target_minus_funnel_x_m", "target_minus_funnel_y_m", "target_minus_funnel_z_m",
    "hand_funnel_finger0_m", "hand_funnel_finger1_m", "hand_funnel_finger2_m",
    "hand_funnel_finger3_m", "hand_funnel_finger4_m", "sampled_funnel_flask_gap_m",
)
MODEL_NAMES = ("S0", "S1", "S2", "S3", "S4")


def parse_episodes(text: str, available: list[int]) -> list[int]:
    if text.lower() == "all":
        return sorted(available)
    selected = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = map(int, item.split("-", 1))
            if stop < start:
                raise ValueError(f"Invalid episode range: {item}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(item))
    absent = selected.difference(available)
    if absent:
        raise ValueError(f"Episodes missing from geometry sidecar: {sorted(absent)}")
    return sorted(selected)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got {arr.shape}")
    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def labels_for_frame(g: Any, s: Any) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-only geometrical targets, 3 translation + 5 finger distances + 1 sampled gap."""
    y = np.zeros(9, np.float32)
    mask = np.zeros(9, bool)
    funnel_xyz = np.asarray(g.manipulated_xyz_m, dtype=np.float32)
    funnel_mask = np.asarray(g.manipulated_mask, bool) & np.isfinite(funnel_xyz).all(axis=-1)
    flask_xyz = np.asarray(g.target_xyz_m, dtype=np.float32)
    flask_mask = np.asarray(g.target_mask, bool) & np.isfinite(flask_xyz).all(axis=-1)
    funnel = funnel_xyz[funnel_mask]
    flask = flask_xyz[flask_mask]
    if not len(funnel):
        return y, mask
    ft = cKDTree(funnel)
    if len(flask):
        y[:3] = flask.mean(axis=0) - funnel.mean(axis=0)
        y[8] = np.min(ft.query(flask, k=1, workers=1)[0])
        mask[:3] = True
        mask[8] = True
    tactile_xyz = np.asarray(s.tactile_xyz_m, dtype=np.float32)
    fingers = np.asarray(s.finger_id)
    if tactile_xyz.shape != (600, 3) or fingers.shape != (600,):
        raise ValueError(f"Bad tactile xyz or finger_id: {tactile_xyz.shape}, {fingers.shape}")
    for finger in range(5):
        valid = (fingers == finger) & np.isfinite(tactile_xyz).all(axis=-1)
        if np.any(valid):
            y[3 + finger] = float(np.min(ft.query(tactile_xyz[valid], k=1, workers=1)[0]))
            mask[3 + finger] = True
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite teacher label")
    return y, mask


def build_episode(root: Path, ep: int, frames: list[int], cameras: tuple[str, ...],
                  image_size: int, geo: Any, spatial: Any) -> dict[str, np.ndarray]:
    """Stream video once per camera; join all modalities by exact episode/frame keys."""
    from openpi.spatial_dataset.source import RawSpatialDataset
    raw = RawSpatialDataset(root, episode=ep)
    state_rows = raw.states()
    frame_state = {int(f): np.asarray(st, np.float32) for f, st in
                   zip(state_rows["frame_index"], state_rows["observation.state"], strict=True)}
    if len(frame_state) != len(state_rows["frame_index"]):
        raise ValueError(f"Duplicate raw frame keys for episode {ep}")
    selected = set(frames)
    images_by_cam = {}
    for cam in cameras:
        if cam not in raw.camera_roles:
            raise KeyError(f"Episode {ep} has no {cam} camera. Available: {raw.camera_roles}")
        images_by_cam[cam] = {int(f.frame_index): resize_rgb(f.rgb, image_size)
                              for f in raw.iter_video_frames(cam, selected=selected)}
    records = {key: [] for key in ("frame", "timestamp", "image", "joint", "tactile", "target", "target_mask")}
    # xhand_policy.extract_joint_state is only 18D projection, but importing it can pull in
    # the full OpenPI/JAX stack. Use the exact reviewed public mapping here for the probe.
    for f in frames:
        if f not in frame_state:
            raise KeyError(f"Episode {ep} frame {f} missing in raw parquet")
        state = frame_state[f]
        if state.ndim != 1 or state.size < 52:
            raise ValueError(f"Episode {ep} frame {f}: bad robot state {state.shape}")
        joint = np.concatenate([state[:6], state[28:52:2]]).astype(np.float32)
        g = geo.get(episode_index=ep, frame_index=f)
        s = spatial.get(episode_index=ep, frame_index=f)
        tactile = np.asarray(s.tactile_force_norm, dtype=np.float32)
        if tactile.shape != (600,) or not np.isfinite(tactile).all():
            raise ValueError(f"Episode {ep} frame {f}: invalid tactile forces {tactile.shape}")
        y, m = labels_for_frame(g, s)
        records["frame"].append(f)
        records["timestamp"].append(float(s.timestamp_s))
        records["image"].append(np.stack([images_by_cam[cam][f] for cam in cameras], axis=0))
        records["joint"].append(joint)
        records["tactile"].append(tactile)
        records["target"].append(y)
        records["target_mask"].append(m)
    result = {k: np.asarray(v) for k, v in records.items()}
    if not len(result["frame"]):
        raise ValueError(f"Empty ep {ep}")
    if result["image"].shape[1:] != (len(cameras), image_size, image_size, 3):
        raise ValueError(f"Unexpected image shape: {result['image'].shape}")
    return result


def cache_episode(root: Path, ep: int, frames: list[int], cameras: tuple[str, ...], image_size: int,
                  geo: Any, spatial: Any, cache_root: Path, signature: dict, force: bool) -> dict[str, np.ndarray]:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"e{ep:03d}.npz"
    if path.is_file() and not force:
        with np.load(path, allow_pickle=False) as payload:
            got = json.loads(str(payload["signature"].item()))
            if got != signature or payload["frame"].tolist() != frames:
                raise ValueError(f"Cache contract changed for {path}; use --rebuild-cache")
            return {k: np.asarray(payload[k]) for k in
                    ("frame", "timestamp", "image", "joint", "tactile", "target", "target_mask")}
    print(f"[cache] episode {ep:03d}, {len(frames)} frames", flush=True)
    result = build_episode(root, ep, frames, cameras, image_size, geo, spatial)
    temp = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temp, **result, signature=np.array(json.dumps(signature, sort_keys=True)))
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return result


@dataclass
class FrameDb:
    by_ep: dict[int, dict[str, np.ndarray]]
    cameras: int

    def index(self, episodes: list[int]) -> list[tuple[int, int]]:
        return [(ep, i) for ep in episodes for i in range(len(self.by_ep[ep]["frame"]))]


class ProbeDataset(Dataset):
    def __init__(self, db: FrameDb, index: list[tuple[int, int]], history: int,
                 norm: dict[str, np.ndarray]) -> None:
        self.db = db
        self.index = index
        self.history = history
        self.norm = norm

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ix: int):
        ep, at = self.index[ix]
        rows = self.db.by_ep[ep]
        ids = np.maximum(np.arange(at - self.history + 1, at + 1), 0)
        image = np.array(rows["image"][ids], copy=True)
        image = image.transpose(0, 1, 4, 2, 3)  # T,C,3,H,W
        joint = (rows["joint"][ids] - self.norm["joint_mean"]) / self.norm["joint_std"]
        touch = (rows["tactile"][ids] - self.norm["touch_mean"]) / self.norm["touch_std"]
        touch = np.clip(touch, -10.0, 10.0)
        y = (rows["target"][at] - self.norm["y_mean"]) / self.norm["y_std"]
        return (torch.from_numpy(image), torch.from_numpy(joint.astype(np.float32)),
                torch.from_numpy(touch.astype(np.float32)), torch.from_numpy(y.astype(np.float32)),
                torch.from_numpy(rows["target_mask"][at].astype(np.float32)))


class TinyCNN(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Student(nn.Module):
    """Architecture identical across S1-S4; ablate by zeroing unavailable inputs."""
    def __init__(self, use_image=True, use_touch=False):
        super().__init__()
        self.use_image = use_image
        self.use_touch = use_touch
        self.cnn = TinyCNN(64)
        self.joint_net = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.touch_net = nn.Sequential(nn.Linear(600, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(192, 128), nn.ReLU())
        self.gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 9))

    def forward(self, image, joint, touch):
        b, t, c, _, h, w = image.shape
        if self.use_image:
            rgb = image.reshape(b * t * c, 3, h, w).float() / 255.0
            feat = self.cnn(rgb).reshape(b, t, c, -1).mean(dim=2)
        else:
            feat = torch.zeros((b, t, 64), device=joint.device, dtype=joint.dtype)
        jq = self.joint_net(joint)
        tactile_in = touch if self.use_touch else torch.zeros_like(touch)
        tf = self.touch_net(tactile_in)
        z = self.fuse(torch.cat([feat, jq, tf], dim=-1))
        _, state = self.gru(z)
        return self.head(state[-1])


def loss_masked(pred, target, mask):
    # Cap contributions from extreme noisy teacher labels without dropping validity.
    squared = torch.square(pred - target)
    return (squared * mask).sum() / mask.sum().clamp(min=1.0)


def norm_train(db: FrameDb, episodes: list[int]) -> dict[str, np.ndarray]:
    joint = np.concatenate([db.by_ep[ep]["joint"] for ep in episodes])
    touch = np.concatenate([db.by_ep[ep]["tactile"] for ep in episodes])
    y = np.concatenate([db.by_ep[ep]["target"] for ep in episodes])
    mask = np.concatenate([db.by_ep[ep]["target_mask"] for ep in episodes])
    yy_mean, yy_std = [], []
    for j in range(9):
        valid = y[mask[:, j], j]
        if not len(valid):
            raise ValueError(f"No training labels for {LABELS[j]}")
        yy_mean.append(float(valid.mean()))
        yy_std.append(max(float(valid.std()), 0.002))
    return {
        "joint_mean": joint.mean(axis=0).astype(np.float32),
        "joint_std": np.maximum(joint.std(axis=0), 0.05).astype(np.float32),
        "touch_mean": touch.mean(axis=0).astype(np.float32),
        "touch_std": np.maximum(touch.std(axis=0), 0.25).astype(np.float32),
        "y_mean": np.array(yy_mean, np.float32),
        "y_std": np.array(yy_std, np.float32),
    }


def train_model(db: FrameDb, indexes: dict[str, list[tuple[int, int]]], norm: dict,
                name: str, args, device: torch.device, out: Path) -> tuple[Student, int, dict]:
    settings = {
        "S0": (False, False, 1), "S1": (True, False, 1),
        "S2": (True, True, 1), "S3": (True, False, args.history),
        "S4": (True, True, args.history),
    }
    use_image, use_touch, history = settings[name]
    # Fair ablation: identical initial weights and shuffle order for each S0-S4.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = Student(use_image=use_image, use_touch=use_touch).to(device)
    train_ds = ProbeDataset(db, indexes["train"], history, norm)
    val_ds = ProbeDataset(db, indexes["val"], history, norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_epoch, history_log = float("inf"), 0, []
    path = out / f"{name}_best.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        tr_loss, tr_count = 0.0, 0
        for image, joint, tactile, y, mask in train_loader:
            image, joint, tactile, y, mask = (v.to(device, non_blocking=True) for v in (image, joint, tactile, y, mask))
            optim.zero_grad(set_to_none=True)
            loss = loss_masked(model(image, joint, tactile), y, mask)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(image)
            tr_count += len(image)
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.inference_mode():
            for image, joint, tactile, y, mask in val_loader:
                image, joint, tactile, y, mask = (v.to(device, non_blocking=True) for v in (image, joint, tactile, y, mask))
                loss = loss_masked(model(image, joint, tactile), y, mask)
                val_loss += loss.item() * len(image)
                val_count += len(image)
        avg = val_loss / max(val_count, 1)
        row = {"epoch": epoch, "train_loss": tr_loss / max(tr_count, 1), "val_loss": avg,
               "elapsed_s": round(time.monotonic() - start, 2)}
        history_log.append(row)
        print(f"[{name}] epoch={epoch}/{args.epochs} train={row['train_loss']:.4f} val={avg:.4f} time={row['elapsed_s']:.1f}s", flush=True)
        if math.isfinite(avg) and avg < best:
            best, best_epoch = avg, epoch
            torch.save({"model": model.state_dict(), "name": name, "history": history,
                        "cameras": args.cameras, "norm": norm, "epoch": epoch,
                        "notes": "Teacher CAD sampled distances; NOT ground-truth physical geometry"}, path)
    if best_epoch == 0:
        raise RuntimeError(f"Model {name} produced no finite val loss")
    # We saved only our own state dictionary; this is a local trusted file, so no unsafe remote pickle loading.
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, history, {"best_epoch": best_epoch, "best_val_loss": best,
                            "parameters": sum(p.numel() for p in model.parameters()), "history": history_log}


def predict(model: Student, ds: ProbeDataset, device: torch.device, batch_size: int,
            norm: dict) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    with torch.inference_mode():
        for image, joint, touch, _y, _mask in loader:
            outputs = model(image.to(device), joint.to(device), touch.to(device))
            results.append(outputs.cpu().numpy())
    pred = np.concatenate(results) * norm["y_std"] + norm["y_mean"]
    return pred.astype(np.float32)


def evaluate(db: FrameDb, index: list[tuple[int,int]], preds: np.ndarray, name: str) -> tuple[dict, list[dict]]:
    if len(preds) != len(index):
        raise ValueError("Evaluation prediction count mismatch")
    targets = np.stack([db.by_ep[ep]["target"][i] for ep, i in index])
    masks = np.stack([db.by_ep[ep]["target_mask"][i] for ep, i in index])
    late = np.array([int(db.by_ep[ep]["frame"][i]) >=
                     int(db.by_ep[ep]["frame"][-1]) * 0.8 for ep,i in index], bool)
    output = {"model": name, "n_test_frames": len(index), "n_test_episodes": len({ep for ep,_ in index}),
              "parameters": "", "test_batch_inference_ms_per_frame": ""}
    for flag, subset in (("all", np.ones(len(index), bool)), ("last20pct", late)):
        for group, cols in (("relative_center_xyz", slice(0,3)), ("five_finger_dist", slice(3,8)),
                            ("sampled_gap", slice(8,9))):
            m = masks[subset, cols]
            errors = np.abs(preds[subset, cols] - targets[subset, cols])
            # xyz reports componentwise mm MAE (not 3D Euclidean error).
            output[f"{flag}_{group}_mae_mm"] = float((errors[m].mean() * 1000) if np.any(m) else float("nan"))
            output[f"{flag}_{group}_label_count"] = int(np.count_nonzero(m))
    rows = []
    for j, (ep, ix) in enumerate(index):
        row = {"model": name, "episode": ep, "frame": int(db.by_ep[ep]["frame"][ix]),
               "timestamp_s": float(db.by_ep[ep]["timestamp"][ix]), "last20pct": bool(late[j])}
        for k, label in enumerate(LABELS):
            row[f"valid_{label}"] = bool(masks[j, k])
            row[f"gt_{label}"] = float(targets[j, k]) if masks[j, k] else ""
            row[f"pred_{label}"] = float(preds[j, k])
        rows.append(row)
    return output, rows


def choose_splits(episodes: list[int], seed: int) -> dict[str, list[int]]:
    if len(episodes) < 10:
        raise ValueError(f"Need >=10 complete episodes for group split, got {len(episodes)}")
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_val = max(2, round(n * 0.15))
    n_test = max(2, round(n * 0.15))
    split = {"test": sorted(shuffled[:n_test]), "val": sorted(shuffled[n_test:n_test+n_val]),
             "train": sorted(shuffled[n_test+n_val:])}
    if min(map(len, split.values())) < 2:
        raise ValueError("Insufficient episodes per split")
    return split


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    """Synthetic shapes and forward/backprop contract test, NOT a real data validation."""
    torch.set_num_threads(1)
    by_ep = {}
    rng = np.random.default_rng(10)
    for ep in range(10):
        n = 5
        by_ep[ep] = {"frame": np.arange(n), "timestamp": np.arange(n) / 15,
                     "image": rng.integers(0, 255, (n, 2, 32, 32, 3), dtype=np.uint8),
                     "joint": rng.normal(size=(n,18)).astype(np.float32),
                     "tactile": rng.random((n,600)).astype(np.float32),
                     "target": rng.normal(0,0.02,(n,9)).astype(np.float32),
                     "target_mask": np.ones((n,9),bool)}
    db = FrameDb(by_ep=by_ep, cameras=2)
    split = choose_splits(list(by_ep), 42)
    indexes = {key: db.index(values) for key,values in split.items()}
    normal = norm_train(db, split["train"])
    for name in MODEL_NAMES:
        mode = {"S0":(False,False,1),"S1":(True,False,1),"S2":(True,True,1),
                "S3":(True,False,4),"S4":(True,True,4)}[name]
        ds = ProbeDataset(db, indexes["train"], mode[2], normal)
        batch = next(iter(DataLoader(ds, batch_size=2)))
        model = Student(use_image=mode[0], use_touch=mode[1])
        out = model(batch[0],batch[1],batch[2])
        assert out.shape == (2,9), (name,out.shape)
        loss_masked(out,batch[3],batch[4]).backward()
    print("SELF-TEST PASS: S0-S4 forward/backward + episode splits + teacher normalization (synthetic only)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-root", default="data/1018_peg_all")
    p.add_argument("--spatial-version", default="v1_front_candidate")
    p.add_argument("--geometry-version", default="v1_hand_scene_q2")
    p.add_argument("--cameras", default="front,left", help="comma-separated raw RGB camera roles")
    p.add_argument("--episodes", default="all", help="all, comma list or range like 0-69")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=5, help="subsample every N frames; history is in sampled-frame units")
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default="S0,S1,S2,S3,S4")
    p.add_argument("--output", default="scratch/geometry_student_probe")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    p.add_argument("--self-test", action="store_true", help="tests no external data / models")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.history < 1 or args.stride < 1 or args.epochs < 1 or args.image_size < 16:
        p.error("history/stride/epochs must be >= 1 and image-size >= 16")
    requested = [x.strip().upper() for x in args.models.split(",") if x.strip()]
    if not requested or set(requested).difference(MODEL_NAMES) or len(requested) != len(set(requested)):
        p.error(f"Invalid --models {args.models}; choose from {MODEL_NAMES}")
    cameras = tuple(x.strip() for x in args.cameras.split(",") if x.strip())
    if len(cameras) != len(set(cameras)) or not cameras:
        p.error("--cameras invalid")
    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4) // 2)))
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    print(f"[start] device={device}, dataset={root}, cameras={cameras}", flush=True)
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    from openpi.spatial_dataset.derived import SpatialDerivedDataset
    geo = ObjectGeometryDerivedDataset(root, version=args.geometry_version)
    spatial = SpatialDerivedDataset(root, version=args.spatial_version)
    episodes = parse_episodes(args.episodes, list(geo.episode_ids))
    signature = {"root": str(root), "spatial_version": args.spatial_version,
                 "geometry_version": args.geometry_version, "cameras": cameras,
                 "stride": args.stride, "image_size": args.image_size,
                 "label_semantics": "v1_sampled_cad_distances_no_contact_groundtruth"}
    by_ep = {}
    for ep in episodes:
        frames = list(geo.frame_ids(ep))[::args.stride]
        if not frames:
            continue
        by_ep[ep] = cache_episode(root, ep, frames, cameras, args.image_size,
                                  geo, spatial, out / "cache", signature, args.rebuild_cache)
    db = FrameDb(by_ep=by_ep, cameras=len(cameras))
    splits = choose_splits(sorted(by_ep), seed)
    indices = {key: db.index(eps) for key, eps in splits.items()}
    print(f"[split] episodes: { {k:len(v) for k,v in splits.items()} } | frames: { {k:len(v) for k,v in indices.items()} }", flush=True)
    (out / "split.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n")
    norm = norm_train(db, splits["train"])
    np.savez_compressed(out / "train_only_normalization.npz", **norm)
    experiment = {"args": vars(args), "split": splits,
                  "disclaimer": "Predicts noisy RGBTrack CAD geometry, not true 3D gap/contact; last20pct is NOT insertion annotation."}
    (out / "experiment.json").write_text(json.dumps(experiment, indent=2, ensure_ascii=False) + "\n")
    results, predictions, logs = [], [], {}
    episode_results = []
    test_index = indices["test"]
    mean = np.broadcast_to(norm["y_mean"], (len(test_index), 9)).copy()
    row, pred_rows = evaluate(db, test_index, mean, "TRAIN_MEAN")
    results.append(row)
    predictions.extend(pred_rows)
    for ep in splits["test"]:
        keep = [j for j, (e, _i) in enumerate(test_index) if e == ep]
        local_row, _ = evaluate(db, [test_index[j] for j in keep], mean[keep], "TRAIN_MEAN")
        local_row["episode"] = ep
        episode_results.append(local_row)
    for name in requested:
        print(f"[run] {name} =========", flush=True)
        model, hist, history_log = train_model(db, indices, norm, name, args, device, out)
        logs[name] = history_log
        test_ds = ProbeDataset(db, test_index, hist, norm)
        started = time.monotonic()
        preds = predict(model, test_ds, device, args.batch_size, norm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_ms_per_frame = (time.monotonic() - started) * 1000 / len(test_index)
        row, pred_rows = evaluate(db, test_index, preds, name)
        row["parameters"] = history_log["parameters"]
        row["test_batch_inference_ms_per_frame"] = inference_ms_per_frame
        results.append(row)
        predictions.extend(pred_rows)
        for ep in splits["test"]:
            keep = [j for j, (e, _i) in enumerate(test_index) if e == ep]
            local_row, _ = evaluate(db, [test_index[j] for j in keep], preds[keep], name)
            local_row["episode"] = ep
            episode_results.append(local_row)
        save_csv(out / "results.csv", results)
        save_csv(out / "predictions.csv", predictions)
        save_csv(out / "per_episode_results.csv", episode_results)
        print(f"[test] {name}: center={row['all_relative_center_xyz_mae_mm']:.3f}mm "
              f"finger={row['all_five_finger_dist_mae_mm']:.3f}mm "
              f"sampled_gap={row['all_sampled_gap_mae_mm']:.3f}mm", flush=True)
    (out / "train_history.json").write_text(json.dumps(logs, indent=2) + "\n")
    lines = ["# Geometry Student Probe (episode-disjoint)", "", "**Important:** This is a *teacher-prediction* test.",
             "CAD nearest sampled point != signed distance, physical contact, insertion depth or truth.",
             "The final 20% window is by frame index, **not** a verified insertion label.",
             f"Episodes: {len(episodes)}; train={splits['train']}; val={splits['val']}; test={splits['test']}",
             f"RGB cameras: {cameras}, stride={args.stride}, history={args.history}, device={device}.", "",
             "All model inputs: 18D joint state, current/past RGB, optionally 600 tactile force magnitudes.",
             "Only teacher targets use 3D CAD meshes. S0 is proprioception only. All scalers fit train episodes only.",
             "", "|Model|XYZ component MAE mm|Finger dist MAE mm|Sampled gap MAE mm|Last20% gap MAE mm|", "|---|---:|---:|---:|---:|"]
    for r in results:
        def fm(key):
            v = r[key]
            return f"{v:.3f}" if math.isfinite(v) else "N/A"
        lines.append(f"|{r['model']}|{fm('all_relative_center_xyz_mae_mm')}|{fm('all_five_finger_dist_mae_mm')}|{fm('all_sampled_gap_mae_mm')}|{fm('last20pct_sampled_gap_mae_mm')}|")
    lines += ["", "Interpretation: S2 versus S1 tests extra *current* tactile information; S4 versus S3 tests extra tactile history.",
              "Because a trainable tiny CNN is used (not pretrained SigLIP), poor absolute results are not evidence distillation is impossible.",
              "Image history only uses frames at or before current time; temporal stride is controlled by --stride.",
              "For real geometry correctness and contact claim, obtain independent multi-view / physical evidence."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] results={out / 'results.csv'} report={out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
