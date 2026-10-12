#!/usr/bin/env python3
"""Offline ContactWorld USB sensor/action audit with raw-image case gallery.

Uses the existing bilateral LeRobot export and saved simulator replay. No
simulator, downloads, checkpoints, or policy training are required. Each stage
is resumable; all fit statistics use train episodes only, tuning uses val, and
the reported estimates use untouched test episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr

from contactworld_neighbor_features import Features, DEFAULT_WEIGHTS, decode


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/contactworld_usb_v3_bilateral"
DEFAULT_EVAL = Path("/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003")
DEFAULT_OUTPUT = Path("/home/sai/zx/openpi-sim-runtime/runs/contactworld_sensor_discovery_20261004")
VERSION = "cw-usb-sensor-discovery-v3"
PHASE = ("train", "val", "test")
AUDIT_NOTICE = '<section id="method-audit-20261008" class="intro"><strong>2026-10-08 方法复核：本页案例尚未匹配完整状态与动作历史。</strong><p>前四张 Fz 图是在“已接触、相对高度 10–25 mm”内按力值极大／极小选取的独立帧，仅供查看信号。它们不是相邻状态对照；XY 误差小也不能证明已经可以直接下插。前两例轴角误差为 1.12°／6.12°，夹持位置和前三步动作均不同。</p><p>当前触觉 T[t] 在动作 a[t] 执行前采集，已经包含此前动作引起的运动和接触。力与下一动作相关，不能证明力唯一确定了该动作，更不能把示范动作当成唯一正确动作。“视觉相似、动作相反”也是按特征与动作筛选出的候选，尚非触觉有效性的受控证据。</p><p>四步位移探针只输入 a[t]，未控制 a[t+1:t+4]，因此其收益混合了物理响应和后续操作规律。现有统计保留为描述性结果；先完成全位姿、速度、夹持与动作历史匹配及滞后检验，再用新案例解释机制。</p></section>'
MODE_BLOCKS = {
    "S": ("state",),
    "R": ("rgb",),
    "T_force": ("map",),
    "S_R": ("state", "rgb"),
    "S_Fz": ("state", "fz"),
    "S_Fxy": ("state", "fxy"),
    "S_Tsummary": ("state", "summary"),
    "S_Tmap": ("state", "map"),
    "S_R_Tmap": ("state", "rgb", "map"),
    "S_R_Tmap_XYZ": ("state", "rgb", "map", "taxel_xyz"),
    "S_R_Tmap_pastA": ("state", "rgb", "map", "past_action"),
    "S_R_Tmap_history": ("state", "rgb", "map", "past_action", "past_map"),
}


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def load_npz(path: Path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_entries(manifest, smoke):
    entries = [e for e in manifest["episodes"] if e["split"] in PHASE]
    if smoke:
        quotas = {"train": 6, "val": 2, "test": 2}
        chosen, used = [], Counter()
        for e in entries:
            if used[e["split"]] < quotas[e["split"]]:
                chosen.append(e)
                used[e["split"]] += 1
        entries = chosen
    counts = Counter(e["split"] for e in entries)
    if any(counts[s] == 0 for s in PHASE):
        raise ValueError(f"Missing a required trajectory split: {counts}")
    return entries


def force_summary(force_base, xyz_base, local):
    """Signed/absolute force, spread, active area, center and four pad regions."""
    f = np.asarray(force_base, np.float32)
    xyz = np.asarray(xyz_base, np.float32)
    active = np.linalg.norm(f, axis=-1) > 1e-4
    weights = np.linalg.norm(f, axis=-1)
    out = []
    for finger in range(2):
        x, p, w, a = f[:, finger], xyz[:, finger], weights[:, finger], active[:, finger]
        pad_center = p.mean(axis=1)
        total = w.sum(axis=1, keepdims=True)
        center = np.where(total > 1e-9, (p * w[..., None]).sum(axis=1) / np.maximum(total, 1e-9), pad_center)
        region = np.asarray(local[:, finger], np.float32).reshape(len(local), 10, 14, 3)
        quadrants = [region[:, rr, cc].mean(axis=(1, 2)) for rr in (slice(0, 5), slice(5, 10))
                     for cc in (slice(0, 7), slice(7, 14))]
        out.append(np.concatenate([x.mean(axis=1), np.abs(x).mean(axis=1), x.std(axis=1),
                                   np.max(np.abs(x), axis=1), a.mean(axis=1)[:, None],
                                   (center-pad_center)*1000, *quadrants], axis=1))
    return np.concatenate(out, axis=1).astype(np.float32)


def image_paths(dataset, episode):
    root = dataset / "videos/chunk-000"
    return {name: root / f"observation.images.{name}/episode_{episode:06d}.mp4"
            for name in ("front", "wrist", "tacff_left_preview", "tacff_right_preview")}


def episode_file(output, entry):
    return output / "episodes" / f"{entry['split']}_source_{entry['source_episode']:03d}.npz"


def prepare_episode(args, entry, manifest, extractor):
    source, ep, length = entry["source_episode"], entry["episode_index"], entry["length"]
    count = length - 4
    if count < 8:
        raise ValueError(f"Episode {source} too short for a four-step target")
    folder = Path(manifest["capture_root"]) / f"insertion_usb_episode_{source:03d}"
    with np.load(folder / "v2_capture.npz") as z:
        plug, socket, ee = (z[k][:length].copy() for k in ("plug_pose", "socket_pose", "ee_pose"))
        state, action = z["state"][:length].copy(), z["action_executed"][:length].copy()
    replay = json.loads((folder / "replay.json").read_text())["rows"]
    if len(replay) != length or len(plug) != length or state.shape != (length, 18):
        raise ValueError(f"Replay/capture length mismatch, source {source}")
    spatial = args.dataset / f"spatial/episodes/episode_{ep:06d}"
    take = lambda name: np.load(spatial / f"{name}.npy", mmap_mode="r")[:count].copy()
    local = np.stack([take("force_grid_left_local"), take("force_grid_right_local")], axis=1)
    base = np.stack([take("tactile_force_left_base"), take("tactile_force_right_base")], axis=1)
    xyz = np.stack([take("tactile_xyz_left_base"), take("tactile_xyz_right_base")], axis=1)
    if base.shape != (count, 2, 140, 3) or local.shape != (count, 2, 10, 14, 3):
        raise ValueError(f"Unexpected bilateral force shape, source {source}")
    videos = image_paths(args.dataset, ep)
    front, wrist = decode(videos["front"], count), decode(videos["wrist"], count)
    rgb = np.concatenate([extractor.rgb(front), extractor.rgb(wrist)], axis=1)
    # A fixed central crop supplies a second visual scale without target-GT cropping.
    crop = np.concatenate([extractor.rgb(front[:, 32:224, 32:224]),
                           extractor.rgb(wrist[:, 32:224, 32:224])], axis=1)
    rotation = Rotation.from_quat(socket[:, 3:])
    relative = rotation.inv().apply(plug[:, :3]-socket[:, :3])*1000
    plug_axis = Rotation.from_quat(plug[:, 3:]).apply([0, 0, 1])
    socket_axis = rotation.apply([0, 0, 1])
    axis = np.degrees(np.arccos(np.clip(np.abs(np.sum(plug_axis*socket_axis, axis=1)), 0, 1)))
    # Frame f is pre-action; the replay row f-1 is the preceding physics step.
    socket_force = np.r_[0, [np.linalg.norm(r["plug_socket_force_post"]) for r in replay[:-1]]]
    act = action[:count].astype(np.float32)
    act_mm_deg = np.concatenate([rotation[:count].inv().apply(act[:, :3] * .01)*1000,
                                 act[:, 3:] * (.05*180/np.pi)], axis=1).astype(np.float32)
    previous = np.concatenate([np.zeros((1, 6), np.float32), act_mm_deg[:-1]], axis=0)
    signed_log = np.sign(base)*np.log1p(np.abs(base)/.001)
    arr = dict(source=np.full(count, source, np.int32), episode=np.full(count, ep, np.int32),
               frame=np.arange(count, dtype=np.int32), split=np.array(entry["split"]),
               state=state[:count].astype(np.float32), rgb=rgb, crop=crop,
               summary=force_summary(base, xyz, local),
               map=signed_log.reshape(count, -1).astype(np.float32),
               fz=signed_log[..., 2].reshape(count, -1).astype(np.float32),
               fxy=signed_log[..., :2].reshape(count, -1).astype(np.float32),
               taxel_xyz=xyz.reshape(count, -1).astype(np.float32),
               force_base=base.astype(np.float32), force_local=local.astype(np.float32),
               rel=relative[:count].astype(np.float32), axis=axis[:count].astype(np.float32),
               socket_force=socket_force[:count].astype(np.float32),
               contact=(socket_force[:count]>.1).astype(np.int8),
               action=act_mm_deg, past_action=previous,
               future_delta=(relative[4:]-relative[:count]).astype(np.float32),
               plug_motion=rotation[:count].inv().apply(
                   np.r_[np.zeros((1, 3)), np.diff(plug[:count, :3], axis=0)]*1000).astype(np.float32),
               slip=np.r_[0, np.linalg.norm(np.diff(plug[:count, :3]-ee[:count, :3], axis=0)*1000, axis=1)].astype(np.float32))
    for key, value in arr.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key} in source {source}")
    save_npz(episode_file(args.output, entry), **arr)


def stage_prepare(args):
    manifest_path = args.dataset / "spatial/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = select_entries(manifest, args.smoke)
    identity = dict(version=VERSION, manifest_sha256=sha256(manifest_path),
                    vision_weights_sha256=sha256(args.vision_weights), smoke=args.smoke,
                    episode_ids=[[e["split"], e["source_episode"]] for e in entries])
    check = args.output / "identity.json"
    if check.exists() and json.loads(check.read_text()) != identity:
        raise ValueError("Dataset or parameters changed; choose a new output directory")
    atomic_json(check, identity)
    extractor = None
    for i, entry in enumerate(entries, 1):
        file = episode_file(args.output, entry)
        if not file.exists():
            if extractor is None:
                extractor = Features(args.vision_weights, args.device, args.batch)
            prepare_episode(args, entry, manifest, extractor)
        print(f"prepare {i}/{len(entries)} {entry['split']} source={entry['source_episode']} {'cached' if extractor is None else ''}", flush=True)
    atomic_json(args.output / "episodes.json", entries)
    return entries


def join_split(args, entries, split):
    chunks = [load_npz(episode_file(args.output, e)) for e in entries if e["split"] == split]
    keys = ("state", "rgb", "crop", "summary", "map", "fz", "fxy", "taxel_xyz",
            "action", "past_action", "rel", "axis", "socket_force", "contact",
            "future_delta", "plug_motion", "slip", "force_base", "source", "episode", "frame")
    out = {k: np.concatenate([piece[k] for piece in chunks]) for k in keys}
    previous_maps = []
    for piece in chunks:
        x = piece["map"]
        previous_maps.append(np.concatenate([x[:1], x[:-1]], axis=0))
    out["past_map"] = np.concatenate(previous_maps)
    return out


def fit_pca(x, width, seed):
    """Train-only randomized PCA with a deterministic fixed sketch."""
    x = np.asarray(x, np.float32)
    center = x.mean(0)
    centered = x - center
    width = min(width, len(x)-1, x.shape[1])
    sketch_width = min(width+12, len(x)-1, x.shape[1])
    rng = np.random.default_rng(seed)
    sketch = centered @ rng.standard_normal((x.shape[1], sketch_width), dtype=np.float32)
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    _, _, vt = np.linalg.svd(basis.T @ centered, full_matrices=False)
    return center.astype(np.float32), vt[:width].T.astype(np.float32)


def feature_blocks(split_data, train_data):
    reduced = {s: {} for s in PHASE}
    pca_info = {}
    # Bound the frozen two-view visual feature dimension before the ridge fits;
    # the projection is learned from train demonstrations only.
    train_rgb = np.concatenate([train_data["rgb"], train_data["crop"]], axis=1)
    rgb_mean, rgb_projection = fit_pca(train_rgb, 128, 4000)
    pca_info["rgb"] = {"original_dim": int(len(rgb_mean)), "reduced_dim": int(rgb_projection.shape[1])}
    for split in PHASE:
        visual = np.concatenate([split_data[split]["rgb"], split_data[split]["crop"]], axis=1)
        reduced[split]["rgb"] = (visual-rgb_mean) @ rgb_projection
    for name, width, seed in (("map", 96, 4001), ("fz", 32, 4002),
                              ("fxy", 64, 4003), ("taxel_xyz", 32, 4004)):
        mean, projection = fit_pca(train_data[name], width, seed)
        pca_info[name] = {"original_dim": int(len(mean)), "reduced_dim": int(projection.shape[1])}
        for split in PHASE:
            reduced[split][name] = (split_data[split][name]-mean) @ projection
        if name == "map":
            # Reuse exactly the same train-only map subspace for previous frames.
            for split in PHASE:
                reduced[split]["past_map"] = (split_data[split]["past_map"]-mean) @ projection
    for split in PHASE:
        d = split_data[split]
        reduced[split].update(state=d["state"], summary=d["summary"], past_action=d["past_action"],
                              action=d["action"])
    return reduced, pca_info


def design(reduced, blocks, extra_action=False):
    fields = list(blocks) + (["action"] if extra_action else [])
    out = {}
    for name in fields:
        train = reduced["train"][name]
        mean = train.mean(0)
        std = train.std(0)
        std[std < 1e-6] = 1
        size = max(1, train.shape[1]) ** .5
        for split in PHASE:
            reduced[split].setdefault("_scaled", {})[name] = np.clip(
                (reduced[split][name]-mean)/std, -8, 8)/size
    for split in PHASE:
        out[split] = np.concatenate([reduced[split]["_scaled"][name] for name in fields], axis=1).astype(np.float32)
    return out


def targets(d, task):
    if task == "action":
        return d["action"]
    if task == "xy":
        return d["rel"][:, :2]
    if task == "contact":
        return d["contact"][:, None].astype(np.float32)
    if task == "future":
        return d["future_delta"]
    raise ValueError(task)


def fit_ridge(x, y, alpha):
    # Centering and scaling target makes alpha selection comparable across axes.
    mean, std = y.mean(0), y.std(0)
    std[std < 1e-6] = 1
    yy = (y-mean)/std
    xtx = x.T @ x
    xtx.flat[::xtx.shape[0]+1] += alpha
    weights = np.linalg.solve(xtx.astype(np.float64), (x.T @ yy).astype(np.float64)).astype(np.float32)
    return weights, mean, std


def tune_ridge(x, y, train_mask, val_mask, task):
    yy = targets(y["train"], task)[train_mask]
    vv = targets(y["val"], task)[val_mask]
    candidate = None
    for alpha in (.1, 1., 10., 100.):
        weights, mean, std = fit_ridge(x["train"][train_mask], yy, alpha)
        pred = x["val"][val_mask] @ weights * std + mean
        loss = float(np.mean(((pred-vv)/std)**2))
        if candidate is None or loss < candidate[0]:
            candidate = (loss, alpha, weights, mean, std)
    _, alpha, weights, mean, std = candidate
    return {"alpha": alpha, "validation_scaled_mse": candidate[0]}, {
        split: (x[split] @ weights * std + mean).astype(np.float32) for split in PHASE}


def auc(y, score):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(score)
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))


def measure(d, prediction, task, mask):
    if not np.any(mask):
        return {}
    actual = targets(d, task)[mask]
    pred = prediction[mask]
    result = {"count": int(mask.sum())}
    if task == "action":
        delta = pred-actual
        result.update(xy_rmse_mm=float(np.sqrt(np.mean(np.sum(delta[:, :2]**2, axis=1)))),
                      xyz_rmse_mm=float(np.sqrt(np.mean(np.sum(delta[:, :3]**2, axis=1)))),
                      rotation_rmse_deg=float(np.sqrt(np.mean(np.sum(delta[:, 3:]**2, axis=1)))))
        large = np.linalg.norm(actual[:, :2], axis=1) > .5
        result["xy_direction_accuracy"] = float(np.mean(np.sum(actual[large, :2]*pred[large, :2], axis=1)>0)) if large.any() else None
    elif task == "xy":
        delta = pred-actual
        result.update(xy_rmse_mm=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                      x_sign_accuracy=float(np.mean(np.sign(pred[:, 0]) == np.sign(actual[:, 0]))),
                      y_sign_accuracy=float(np.mean(np.sign(pred[:, 1]) == np.sign(actual[:, 1]))))
    elif task == "contact":
        result.update(auroc=auc(actual[:, 0], pred[:, 0]), balanced_accuracy=None)
        y = actual[:, 0] > .5
        if y.any() and (~y).any():
            result["balanced_accuracy"] = float((np.mean(pred[y, 0]>.5)+np.mean(pred[~y, 0]<=.5))/2)
    elif task == "future":
        delta = pred-actual
        result.update(xy_rmse_mm=float(np.sqrt(np.mean(np.sum(delta[:, :2]**2, axis=1)))),
                      z_rmse_mm=float(np.sqrt(np.mean(delta[:, 2]**2))))
    return result


def summarize_predictions(split_data, predictions, tuning):
    rows = []
    for (mode, task, condition), by_split in predictions.items():
        d = split_data["test"]
        for group, mask in (("all", np.ones(len(d["source"]), bool)),
                            ("pre", d["contact"] == 0), ("contact", d["contact"] == 1)):
            result = measure(d, by_split["test"], task, mask)
            if result:
                rows.append({"mode": mode, "task": task, "training_condition": condition,
                             "test_phase": group, "alpha": tuning[(mode, task, condition)]["alpha"], **result})
    return rows


def stage_fit(args, entries):
    output = args.output / "fit"
    marker = output / "complete.json"
    if marker.exists():
        print("fit cached", flush=True)
        return
    split_data = {s: join_split(args, entries, s) for s in PHASE}
    reduced, pca_info = feature_blocks(split_data, split_data["train"])
    predictions, tuning = {}, {}
    for mode, blocks in MODE_BLOCKS.items():
        x = design(reduced, blocks)
        for task in ("action", "xy", "contact"):
            key = (mode, task, "all")
            info, pred = tune_ridge(x, split_data,
                                    np.ones(len(x["train"]), bool), np.ones(len(x["val"]), bool), task)
            tuning[key], predictions[key] = info, pred
        print(f"fit {mode} 3 tasks", flush=True)
    for mode in ("S", "S_R", "S_Tmap", "S_R_Tmap", "S_R_Tmap_history"):
        x = design(reduced, MODE_BLOCKS[mode])
        key = (mode, "action", "contact")
        train_mask = split_data["train"]["contact"] == 1
        val_mask = split_data["val"]["contact"] == 1
        if train_mask.sum() < 20 or val_mask.sum() < 5:
            print(f"skip {mode} contact-specific fit: too few frames", flush=True)
            continue
        info, pred = tune_ridge(x, split_data, train_mask, val_mask, "action")
        tuning[key], predictions[key] = info, pred
    # Given the already chosen action, ask whether the sensors predict its
    # physical four-step outcome. This is observational, not an action oracle.
    for mode in ("S", "S_R", "S_Tmap", "S_R_Tmap"):
        x = design(reduced, MODE_BLOCKS[mode], extra_action=True)
        key = (mode, "future", "all")
        info, pred = tune_ridge(x, split_data,
                                np.ones(len(x["train"]), bool), np.ones(len(x["val"]), bool), "future")
        tuning[key], predictions[key] = info, pred
    rows = summarize_predictions(split_data, predictions, tuning)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)
    compact = {"pca": pca_info, "metrics": rows, "tuning": {"/".join(k): v for k, v in tuning.items()},
               "splits": {s: {"episodes": len({int(v) for v in split_data[s]["source"]}),
                              "frames": len(split_data[s]["source"]),
                              "contact_frames": int(split_data[s]["contact"].sum())} for s in PHASE}}
    atomic_json(output / "results.json", compact)
    save_npz(output / "heldout_predictions.npz", source=split_data["test"]["source"],
             frame=split_data["test"]["frame"], contact=split_data["test"]["contact"],
             target_action=split_data["test"]["action"], rel=split_data["test"]["rel"],
             **{f"{mode}_{task}_{condition}": pred["test"] for (mode, task, condition), pred in predictions.items()})
    atomic_json(marker, {"version": VERSION, "modes": len(MODE_BLOCKS),
                         "tasks": len(predictions), "metrics": len(rows)})
    print(f"fit complete: {len(predictions)} comparisons", flush=True)


def source_bootstrap(values, sources, seed=20261004, repeats=500):
    sources = np.asarray(sources)
    unique = np.unique(sources)
    if len(unique) < 2:
        return [None, None]
    per_source = np.array([np.mean(np.asarray(values)[sources == s]) for s in unique])
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(unique), size=(repeats, len(unique)))
    return np.quantile(per_source[samples].mean(axis=1), [.025, .975]).tolist()


def signed_force_stats(data):
    base = data["force_base"]
    mean = base.mean(axis=2)
    return np.concatenate([mean[:, 0], mean[:, 1], mean[:, 0]-mean[:, 1]], axis=1)


def correlations(data):
    signal = signed_force_stats(data)
    movement = data["plug_motion"]
    values = np.column_stack([data["rel"][:, :3], movement, data["slip"], data["action"][:, :3]])
    sx = ["L Fx", "L Fy", "L Fz", "R Fx", "R Fy", "R Fz", "L-R Fx", "L-R Fy", "L-R Fz"]
    sy = ["plug X", "plug Y", "plug Z", "prior dX", "prior dY", "prior dZ", "prior slip", "next action X", "next action Y", "next action Z"]
    result = []
    for group, mask in (("pre", data["contact"] == 0), ("contact", data["contact"] == 1),
                        ("contact/near-mouth/slow", (data["contact"] == 1) &
                         (data["rel"][:, 2] >= 10) & (data["rel"][:, 2] <= 25) &
                         (np.abs(movement[:, 2]) <= 2) & (data["axis"] <= 7))):
        support = len(np.unique(data["source"][mask]))
        if int(mask.sum()) < 30 or support < 3:
            continue
        for i, name in enumerate(sx):
            for j, target in enumerate(sy):
                x, y = signal[mask, i], values[mask, j]
                rho = float(spearmanr(x, y).statistic) if np.std(x) > 1e-10 and np.std(y) > 1e-10 else None
                result.append(dict(group=group, signal=name, target=target, spearman=rho,
                                   frames=int(mask.sum()), episodes=support))
    return result


def frame_at(path, frame):
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()


def local_force_preview(local_grid, base_scale):
    """False-color magnitude from saved local 3-component TacFF, fixed scale."""
    magnitude = np.linalg.norm(np.asarray(local_grid, np.float32), axis=-1)
    scaled = np.clip(magnitude / max(float(np.max(base_scale)), 1e-9), 0, 1)
    return plt.get_cmap("magma")(scaled)[..., :3]


def image_card(output, name, front, wrist, previews, base, scale, caption, preview_labels=("Left TacFF preview", "Right TacFF preview")):
    fig = plt.figure(figsize=(15.5, 8.3), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1.65, 1, 1])
    for column, (image, label) in enumerate(zip((front, wrist, *previews),
                                                 ("Front RGB", "Wrist RGB", *preview_labels))):
        ax = fig.add_subplot(grid[0, column])
        if image is not None:
            ax.imshow(image)
        else:
            ax.text(.5, .5, "not recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.add_subplot(grid[1:, 3]).axis("off")
    for finger, side in enumerate(("Left", "Right")):
        for channel, axis in enumerate("XYZ"):
            panel = fig.add_subplot(grid[1+finger, channel])
            component = base[finger, :, channel].reshape(10, 14)
            panel.imshow(component, cmap="RdBu_r", vmin=-scale[channel], vmax=scale[channel],
                              interpolation="nearest", aspect="equal")
            panel.set_title(f"{side} base F{axis} · mean {component.mean():+.5f}", fontsize=10)
            panel.set_xticks([]); panel.set_yticks([])
    fig.suptitle(caption, fontsize=12)
    target = output / "cases" / f"{name}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=125)
    plt.close(fig)
    return target


def choose_cases(args, entries, by_source, heldout, prediction):
    selected, seen = [], set()
    def add(title, source, frame, why, policy=None):
        token = (source, frame, policy)
        if token not in seen and len(selected) < args.max_case_images:
            seen.add(token)
            selected.append(dict(title=title, source=int(source), frame=int(frame), why=why, policy=policy))
    d = heldout
    sources, frames = d["source"], d["frame"]
    contact = d["contact"].astype(bool)
    rel, action = d["rel"], d["action"]
    mask = contact & (rel[:, 2] >= 10) & (rel[:, 2] <= 25)
    f = signed_force_stats(d)
    for finger, column in (("left", 2), ("right", 5)):
        for polarity, direction in (("positive", 1), ("negative", -1)):
            candidates = np.flatnonzero(mask)
            if len(candidates):
                i = candidates[np.argmax(direction*f[candidates, column])]
                add(f"{finger} Fz {polarity}", sources[i], frames[i],
                    "Force-extreme illustration within contact Z=10–25 mm. Full pose, velocity, grasp and past actions are NOT matched; not a neighboring-state comparison.")
    for label, dim, direction in (("plug left", 0, -1), ("plug right", 0, 1),
                                  ("plug back", 1, -1), ("plug front", 1, 1)):
        candidates = np.flatnonzero(mask)
        if len(candidates):
            i = candidates[np.argmax(direction*rel[candidates, dim])]
            add(label, sources[i], frames[i], "Signed plug offset is measured in the socket frame.")
    for label, conditions in (("no socket contact", d["contact"] == 0),
                              ("near target height", contact & (np.abs(rel[:, 2]) <= 5) &
                               (np.linalg.norm(rel[:, :2], axis=1) <= 2)),
                              ("contact with large lateral error", contact & (np.linalg.norm(rel[:, :2], axis=1) > 6))):
        candidates = np.flatnonzero(conditions)
        if len(candidates):
            i = candidates[len(candidates)//2]
            add(label, sources[i], frames[i], "Compare the raw fields and both cameras with nearby states.")
    # Pick cases where adding tactile changes held-out lateral action prediction.
    a = prediction.get("S_R_action_all")
    b = prediction.get("S_R_Tmap_action_all")
    if a is not None and b is not None:
        gain = np.sum((a[:, :2]-action[:, :2])**2, axis=1)-np.sum((b[:, :2]-action[:, :2])**2, axis=1)
        candidates = np.flatnonzero(contact)
        if len(candidates):
            for label, order in (("tactile helps this action", np.argsort(-gain[candidates])),
                                 ("tactile hurts this action", np.argsort(gain[candidates]))):
                for index in order[:15]:
                    i = candidates[index]
                    if (int(sources[i]), int(frames[i]), None) not in seen:
                        add(label, sources[i], frames[i],
                            f"Paired held-out action residual difference: {gain[i]:+.2f} squared mm; one case is not a model ranking.")
                        break
    # Find very similar RGB observations with conflicting lateral actions.
    visual = np.concatenate([d["rgb"], d["crop"]], axis=1)
    candidate = np.flatnonzero(contact & (np.linalg.norm(action[:, :2], axis=1) > .7))
    if len(candidate) > 20:
        from scipy.spatial import cKDTree
        normed = visual[candidate]
        normed = (normed-normed.mean(0))/(normed.std(0)+1e-3)
        tree = cKDTree(normed)
        distances, neighbors = tree.query(normed, k=min(25, len(candidate)))
        pairs = []
        for row, neigh in enumerate(neighbors):
            i = candidate[row]
            for rank, local_j in enumerate(neigh[1:], 1):
                j = candidate[local_j]
                if sources[i] == sources[j]:
                    continue
                cosine = np.dot(action[i, :2], action[j, :2]) / (np.linalg.norm(action[i, :2])*np.linalg.norm(action[j, :2])+1e-9)
                if cosine < -.5:
                    pairs.append((distances[row, rank], i, j))
                    break
        if pairs:
            _, i, j = min(pairs)
            add("similar RGB, opposite lateral actions A", sources[i], frames[i], "Nearest frozen RGB features across different held-out episodes; actions point opposite ways.")
            add("similar RGB, opposite lateral actions B", sources[j], frames[j], "Compare this paired case to the previous card; distance is visual-feature distance.")
    # Two paired initial states: one where adding P/T helps and one where it hurts.
    # This is illustrative; the entire 40-seed evaluation remains in its own report.
    if not args.smoke and args.eval_root.exists():
        rgb = args.eval_root / "pi0_cw3_usb_01_rgb"
        fusion = args.eval_root / "pi0_cw3_usb_04_fused_pc_local_map"
        contrasts = {"P/T succeeds, RGB fails": None, "RGB succeeds, P/T fails": None}
        for seed in range(10000, 10040):
            paths = [root / f"seed_{seed}" for root in (rgb, fusion)]
            if not all((folder / "trajectory.json").exists() and (folder / "inputs.npz").exists() for folder in paths):
                continue
            outcome = [bool(json.loads((folder / "trajectory.json").read_text())["result"]["source_final"])
                       for folder in paths]
            label = "P/T succeeds, RGB fails" if outcome == [False, True] else (
                "RGB succeeds, P/T fails" if outcome == [True, False] else None)
            if label is not None and contrasts[label] is None:
                contrasts[label] = (seed, paths)
        for label, pair in contrasts.items():
            if pair is None:
                continue
            seed, folders = pair
            for folder in folders:
                trajectory = json.loads((folder / "trajectory.json").read_text())
                rows = trajectory["rows"]
                onset = next((i+1 for i, r in enumerate(rows)
                              if np.linalg.norm(r["plug_socket_contact_force_xyz"]) > .1), 0)
                file = folder / "inputs.npz"
                with np.load(file) as z:
                    steps = z["steps"]
                frame = int(np.argmin(np.abs(steps-onset)))
                add("paired policy outcome contrast", -1, frame,
                    f"seed {seed}: {label}; model {folder.parent.name}; final success "
                    f"{trajectory['result']['source_final']}. This image is near first socket contact.",
                    policy=str(file))
    return selected


def stage_cases(args, entries):
    path = args.output / "cases.json"
    if path.exists():
        return json.loads(path.read_text())
    test = join_split(args, entries, "test")
    z = load_npz(args.output / "fit/heldout_predictions.npz")
    chosen = choose_cases(args, entries, None, test, z)
    # Color limits use train episodes only; no per-frame auto contrast.
    limits = []
    for e in entries:
        if e["split"] == "train":
            with np.load(episode_file(args.output, e)) as d:
                limits.append(np.abs(d["force_base"]).reshape(-1, 3)[::12])
    scale = np.quantile(np.concatenate(limits), .995, axis=0).astype(float)
    scale = np.maximum(scale, 1e-6)
    lookup = {e["source_episode"]: e for e in entries}
    for i, case in enumerate(chosen, 1):
        if case["policy"]:
            file = Path(case["policy"])
            with np.load(file) as z:
                idx = case["frame"]
                front, wrist, base = z["front"][idx], z["wrist"][idx], z["force_base"][idx]
                local_grid = z["force_local"][idx]
                step = int(z["steps"][idx])
            case["step"] = step
            case["video"] = str(file.parent / "front_wrist.mp4")
            previews = tuple(local_force_preview(local_grid[finger], scale) for finger in range(2))
            preview_labels = ("Left local |F| (false color)", "Right local |F| (false color)")
            trajectory = json.loads((file.parent / "trajectory.json").read_text())
            current = trajectory["result"]["initial"] if step == 0 else trajectory["rows"][step-1]
            case["xy_mm"] = round(current["plug_socket_xy_error_mm"], 2)
            case["z_mm"] = round(current["plug_socket_z_mm"], 2)
            case["axis_deg"] = round(current["unsigned_axis_error_deg"], 2)
            case["socket_contact"] = bool(np.linalg.norm(current["plug_socket_contact_force_xyz"]) > .1)
            case["action_xy_mm"] = None
        else:
            entry = lookup[case["source"]]
            with np.load(episode_file(args.output, entry)) as z:
                idx = case["frame"]
                base = z["force_base"][idx]
                case["xy_mm"] = round(float(np.linalg.norm(z["rel"][idx, :2])), 2)
                case["z_mm"] = round(float(z["rel"][idx, 2]), 2)
                case["axis_deg"] = round(float(z["axis"][idx]), 2)
                case["socket_contact"] = bool(z["contact"][idx])
                case["action_xy_mm"] = np.round(z["action"][idx, :2], 2).tolist()
                case["prior_dz_mm"] = round(float(z["plug_motion"][idx, 2]), 2)
            videos = image_paths(args.dataset, entry["episode_index"])
            front, wrist = frame_at(videos["front"], idx), frame_at(videos["wrist"], idx)
            previews = (frame_at(videos["tacff_left_preview"], idx),
                        frame_at(videos["tacff_right_preview"], idx))
            preview_labels = ("Left TacFF preview", "Right TacFF preview")
            case["video"] = str(videos["front"])
            case["wrist_video"] = str(videos["wrist"])
        title = f"{case['title']} · source {case['source']} frame {case['frame']}"
        image = image_card(args.output, f"case_{i:02d}", front, wrist, previews, base, scale, title, preview_labels)
        case["image"] = str(image)
        print(f"case {i}/{len(chosen)} {case['title']}", flush=True)
    atomic_json(path, {"force_color_scale_train_p995": scale.tolist(), "cases": chosen})
    return {"force_color_scale_train_p995": scale.tolist(), "cases": chosen}


def plot_contact_heatmaps(output, data):
    force = data["force_base"].mean(axis=2)
    x, y = data["rel"][:, 0], data["rel"][:, 1]
    selected = (data["contact"] == 1) & (data["rel"][:, 2] >= 10) & (data["rel"][:, 2] <= 25)
    edges = np.linspace(-15, 15, 11)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained", sharex=True, sharey=True)
    for finger in range(2):
        for channel in range(3):
            ax = axes[finger, channel]
            matrix = np.full((10, 10), np.nan)
            for iy in range(10):
                for ix in range(10):
                    mask = selected & (x >= edges[ix]) & (x < edges[ix+1]) & (y >= edges[iy]) & (y < edges[iy+1])
                    if mask.sum() >= 5 and len(np.unique(data["source"][mask])) >= 2:
                        matrix[iy, ix] = force[mask, finger, channel].mean()
            scale = np.nanquantile(np.abs(matrix), .95) if np.isfinite(matrix).any() else 1e-6
            scale = max(float(scale), 1e-6)
            ax.imshow(matrix, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale,
                      extent=[-15, 15, -15, 15], interpolation="nearest")
            ax.set_title(f"{'Left' if finger == 0 else 'Right'} mean base F{'XYZ'[channel]} · {np.isfinite(matrix).sum()} supported cells")
            ax.axhline(0, color="#555", lw=.7); ax.axvline(0, color="#555", lw=.7)
            ax.set_xlabel("Plug X relative to socket (mm)"); ax.set_ylabel("Plug Y relative to socket (mm)")
    fig.suptitle("Held-out contact at Z 10–25 mm; cells need ≥5 frames and ≥2 episodes. White means insufficient data.")
    path = output / "figures/force_by_offset.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145); plt.close(fig)
    return path


def plot_fz_sign(output, data):
    force = data["force_base"].mean(axis=2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    categories = (("before socket contact", data["contact"] == 0),
                  ("socket contact, moving down", (data["contact"] == 1) & (data["plug_motion"][:, 2] < -.2)),
                  ("socket contact, nearly still", (data["contact"] == 1) & (np.abs(data["plug_motion"][:, 2]) <= .2)))
    for finger, ax in enumerate(axes):
        values = force[:, finger, 2]
        limit = np.quantile(np.abs(values), .99)
        for label, mask in categories:
            if mask.sum() >= 5:
                ax.hist(values[mask], bins=55, range=(-limit, limit), density=True,
                        histtype="step", lw=1.8, label=f"{label} (n={mask.sum()})")
        ax.axvline(0, color="black", lw=.7)
        ax.set(title=f"{'Left' if finger == 0 else 'Right'} signed mean base Fz",
               xlabel="Synthetic pad force (signed)", ylabel="Density")
        ax.legend(fontsize=8)
    path = output / "figures/fz_by_motion.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145); plt.close(fig)
    return path


def plot_local_base_sign(output, data):
    """Show whether base-frame Fz sign tracks local channel 3 or prior motion."""
    base = data["force_base"].mean(axis=2)
    local = data["force_local"].mean(axis=(2, 3))
    contact = data["contact"] == 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for finger in range(2):
        mask = contact & (data["rel"][:, 2] >= 5) & (data["rel"][:, 2] <= 30)
        x = base[mask, finger, 2]
        y = local[mask, finger, 2]
        move = data["plug_motion"][mask, 2]
        side = "Left" if finger == 0 else "Right"
        axes[finger, 0].scatter(x, y, s=9, alpha=.38, c=move, cmap="coolwarm", vmin=-2, vmax=2)
        axes[finger, 0].set(title=f"{side}: base Fz versus sensor-local channel 3",
                            xlabel="Mean base Fz", ylabel="Mean local component 3")
        axes[finger, 1].scatter(move, x, s=9, alpha=.38, c=data["rel"][mask, 2],
                                cmap="viridis", vmin=5, vmax=30)
        axes[finger, 1].set(title=f"{side}: base Fz versus prior plug motion",
                            xlabel="Prior plug ΔZ in socket frame (mm/frame)", ylabel="Mean base Fz")
        for ax in axes[finger]:
            ax.axhline(0, color="#777", lw=.7)
            ax.axvline(0, color="#777", lw=.7)
            ax.grid(alpha=.12)
    fig.suptitle("Held-out socket-contact observations at Z=5–30 mm; each dot is a frame, correlated within episodes")
    path = output / "figures/local_base_fz_sign.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_examples(output, entries, args):
    chosen = [e for e in entries if e["split"] == "test"][:3]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 3.4*len(chosen)), layout="constrained")
    axes = np.atleast_2d(axes)
    for row, entry in enumerate(chosen):
        d = load_npz(episode_file(args.output, entry)); t = d["frame"]
        load = d["socket_force"]; force = d["force_base"].mean(axis=2)
        hit = np.flatnonzero(load > .1)
        contact = hit[0] if len(hit) else None
        ax = axes[row, 0]; ax.plot(t, load, color="black", label="plug/socket load")
        ax.set_ylabel(f"source {entry['source_episode']}\nPair force")
        ax = axes[row, 1]; ax.plot(t, force[:, 0, 2], label="left Fz"); ax.plot(t, force[:, 1, 2], label="right Fz")
        ax.legend(fontsize=8); ax.set_ylabel("Signed pad Fz")
        ax = axes[row, 2]; ax.plot(t, np.linalg.norm(d["rel"][:, :2], axis=1), label="XY error mm")
        ax.plot(t, d["rel"][:, 2], label="Z mm"); ax.legend(fontsize=8)
        for ax in axes[row]:
            if contact is not None:ax.axvline(contact, color="#e78c39", ls="--", lw=1)
            ax.set_xlabel("Pre-action frame")
    fig.suptitle("Same-frame force and geometry; dashed line is first recorded plug/socket contact")
    path = output / "figures/contact_timelines.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def plot_probe(output, rows):
    modes = ["S", "S_R", "S_Fz", "S_Fxy", "S_Tsummary", "S_Tmap", "S_R_Tmap", "S_R_Tmap_history"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, task, ylabel in ((axes[0], "action", "Lateral action RMSE (mm)"),
                              (axes[1], "xy", "Plug XY position RMSE (mm)")):
        values = []
        for mode in modes:
            match = [r for r in rows if r["mode"] == mode and r["task"] == task and
                     r["training_condition"] == "all" and r["test_phase"] == "contact"]
            values.append(match[0].get("xy_rmse_mm", np.nan) if match else np.nan)
        ax.bar(range(len(modes)), values, color=["#798a9a", "#6ba8d3", "#d69367", "#d69367",
                                                  "#d69367", "#d69367", "#667fcb", "#617dbb"])
        ax.set(xticks=range(len(modes)), xticklabels=modes, ylabel=ylabel, title=f"Held-out test episodes: socket-contact frames ({task})")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=.2)
    path = output / "figures/heldout_probe.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145); plt.close(fig)
    return path


def plot_correlations(output, rows):
    selected = [r for r in rows if r["group"] == "contact"]
    xs = list(dict.fromkeys(r["signal"] for r in selected)); ys = list(dict.fromkeys(r["target"] for r in selected))
    table = np.full((len(xs), len(ys)), np.nan)
    for r in selected:
        if r["spearman"] is not None:
            table[xs.index(r["signal"]), ys.index(r["target"])] = r["spearman"]
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    im = ax.imshow(table, cmap="coolwarm", vmin=-.6, vmax=.6, aspect="auto")
    ax.set_xticks(range(len(ys)), ys, rotation=45, ha="right")
    ax.set_yticks(range(len(xs)), xs)
    ax.set_title("Held-out contact frames: descriptive Spearman correlation (sign matters)")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    path = output / "figures/force_correlations.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=145); plt.close(fig)
    return path


def pair_interval(predictions, data, mode1, mode2, phase="contact"):
    mask = data["contact"] == 1 if phase == "contact" else data["contact"] == 0
    a = predictions[f"{mode1}_action_all"]
    b = predictions[f"{mode2}_action_all"]
    target = data["action"]
    sources = np.unique(data["source"][mask])
    if len(sources) < 2:
        return None
    a2 = np.sum((a[:, :2]-target[:, :2])**2, axis=1)
    b2 = np.sum((b[:, :2]-target[:, :2])**2, axis=1)
    pairs = []
    for source in sources:
        include = mask & (data["source"] == source)
        pairs.append((a2[include].mean(), b2[include].mean()))
    pairs = np.asarray(pairs)
    rng = np.random.default_rng(20261004)
    samples = rng.integers(len(sources), size=(1000, len(sources)))
    draw = pairs[samples].mean(axis=1)
    delta = np.sqrt(draw[:, 1])-np.sqrt(draw[:, 0])
    return dict(delta_rmse_mm=float(np.sqrt(pairs[:, 1].mean())-np.sqrt(pairs[:, 0].mean())),
                episode_bootstrap95=np.quantile(delta, [.025, .975]).tolist(),
                heldout_episodes=len(sources), comparison=f"{mode2} minus {mode1}")


def fmt_metric(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"


def report_table(rows):
    selected = [r for r in rows if r["task"] == "action" and r["training_condition"] == "all"]
    by = {(r["mode"], r["test_phase"]): r for r in selected}
    tr = []
    for mode in MODE_BLOCKS:
        pre, contact = by.get((mode, "pre"), {}), by.get((mode, "contact"), {})
        tr.append(f"<tr><td>{html.escape(mode)}</td><td>{fmt_metric(pre.get('xy_rmse_mm'))}</td>"
                  f"<td>{fmt_metric(contact.get('xy_rmse_mm'))}</td>"
                  f"<td>{fmt_metric(contact.get('xy_direction_accuracy'))}</td></tr>")
    return "\n".join(tr)


def physical_table(rows):
    by = {(r["mode"], r["task"], r["test_phase"]): r for r in rows
          if r["training_condition"] == "all"}
    tr = []
    for mode in MODE_BLOCKS:
        xy = by.get((mode, "xy", "contact"), {})
        contact = by.get((mode, "contact", "all"), {})
        tr.append(f"<tr><td>{html.escape(mode)}</td><td>{fmt_metric(xy.get('xy_rmse_mm'))}</td>"
                  f"<td>{fmt_metric(contact.get('auroc'))}</td></tr>")
    return "\n".join(tr)


def stage_report(args, entries):
    fit = json.loads((args.output / "fit/results.json").read_text())
    predictions = load_npz(args.output / "fit/heldout_predictions.npz")
    gallery = stage_cases(args, entries)
    heldout_parts = [load_npz(episode_file(args.output, e)) for e in entries if e["split"] in ("val", "test")]
    keys = ("source", "rel", "axis", "contact", "socket_force", "force_base", "force_local", "plug_motion", "slip", "action")
    heldout = {k: np.concatenate([d[k] for d in heldout_parts]) for k in keys}
    test = join_split(args, entries, "test")
    corr = correlations(heldout)
    with (args.output / "correlations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group", "signal", "target", "spearman", "frames", "episodes"))
        writer.writeheader(); writer.writerows(corr)
    intervals = {}
    for other in ("S", "S_Tmap", "S_R_Tmap"):
        if other != "S_R":
            for phase in ("pre", "contact"):
                intervals[f"{other}_{phase}"] = pair_interval(predictions, test, "S_R", other, phase)
    atomic_json(args.output / "paired_intervals.json", intervals)
    figs = [plot_contact_heatmaps(args.output, heldout), plot_fz_sign(args.output, heldout),
            plot_local_base_sign(args.output, heldout), plot_examples(args.output, entries, args), plot_probe(args.output, fit["metrics"]),
            plot_correlations(args.output, corr)]
    # The gallery leads the report; quantitative plots are accessible immediately after it.
    cards = []
    for i, case in enumerate(gallery["cases"], 1):
        image = html.escape(os.path.relpath(case["image"], args.output))
        video = html.escape(os.path.relpath(case["video"], args.output))
        extra = ""
        if case.get("wrist_video"):
            extra = f' · <a href="{html.escape(os.path.relpath(case["wrist_video"], args.output))}">wrist MP4</a>'
        action = case.get("action_xy_mm")
        details = (f"XY {case['xy_mm']} mm · Z {case['z_mm']} mm · axis {case['axis_deg']}° · "
                   f"socket contact {case['socket_contact']}" + (f" · next action XY {action} mm" if action is not None else ""))
        cards.append(f'<details class="case" {"open" if i <= 3 else ""}><summary>{i:02d} · {html.escape(case["title"])}</summary>'
                     f'<p>{html.escape(case["why"])}</p><p>{html.escape(details)} · <a href="{video}">original RGB MP4</a>{extra}</p>'
                     f'<a href="{image}"><img loading="lazy" src="{image}" alt="raw RGB and tactile force maps"></a></details>')
    figure_cards = "".join(f'<figure><img loading="lazy" src="{html.escape(os.path.relpath(path, args.output))}"><figcaption>{html.escape(path.stem.replace("_", " "))}</figcaption></figure>' for path in figs)
    delta = intervals.get("S_R_Tmap_contact")
    comparison = (f"On the 10 held-out test demonstrations, contact-frame lateral action RMSE for S+R+Tmap minus S+R is "
                  f"{delta['delta_rmse_mm']:+.2f} mm; episode bootstrap interval "
                  f"[{delta['episode_bootstrap95'][0]:+.2f}, {delta['episode_bootstrap95'][1]:+.2f}] mm."
                  if delta else "Too few held-out episodes for a paired interval.")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ContactWorld USB: sensors, contact and action</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101923;color:#e9f1fa;font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:29px}}h2{{margin-top:34px}}a{{color:#98d6ff}}
.intro{{padding:15px 19px;background:#1b3046;border-left:4px solid #8bc8ff}}.case{{border:1px solid #40546b;border-radius:8px;margin:12px 0;padding:12px 15px}}
.case summary{{font-weight:650;cursor:pointer;font-size:18px}}.case img{{width:100%;height:auto;display:block;background:white}}
figure{{background:#1b2938;margin:14px 0;padding:8px}}figure img{{width:100%;height:auto}}figcaption{{color:#bdd1e2}}
table{{border-collapse:collapse;width:100%;max-width:780px}}td,th{{padding:8px 10px;border-bottom:1px solid #394b5d;text-align:left}}
.small{{font-size:14px;color:#bdcfe1}}details>summary{{cursor:pointer}}@media(max-width:700px){{main{{padding:10px}}}}
</style></head><body><main>
<h1>ContactWorld USB：从原始画面理解触觉与动作</h1>
{AUDIT_NOTICE}
<p class="intro">先看下面的原始前视、腕视、左右指触觉预览及六张 base 坐标力图；每张图旁保留对应动作与插座真值。后面再看分布和留出示范上的预测检验。颜色在所有案例中固定：训练示范绝对力 P99.5 = {gallery['force_color_scale_train_p995']}，红正、蓝负、白零。TacFF 为合成单位。</p>
<p>数据：{fit['splits']['train']['episodes']} 条训练示范、{fit['splits']['val']['episodes']} 条验证示范、{fit['splits']['test']['episodes']} 条未参与训练和选参的测试示范。测试帧接触标注使用上一动作后的 U 盘—插座合力；左右指力来自当前动作前的观测。左右方向用插座坐标定义。模型输入不含插座真值、接触标签或未来动作。</p>
<h2>原始数据案例</h2><p class="small">左/右指图片上排为仿真触觉预览，下两排是同帧 base Fx/Fy/Fz 原始力场。点击图片可放大；每张案例可打开原视频。策略回放未存独立触觉预览视频，卡片中改用保存的原始局部三分量力场制作幅值伪彩图。案例由预设状态和留出预测残差自动挑选，单例只用于理解，结论以整条示范的统计为准。</p>
{''.join(cards)}
<h2>触觉分量与物理状态</h2><p>白色空格代表该偏移/深度区域的数据不足。接触状态由插座合力定义，指尖力即使未碰插座也可能非零。时间曲线将接触时刻、左右 Fz 和实际位置放在同一帧索引下；纵轴仍是不同量纲。</p>
{figure_cards}
<h2>动作可预测性：按完整示范留出</h2><p>{html.escape(comparison)} 负值表示加入该模态后的测试误差降低。视觉使用冻结 ResNet18 的全图与固定中心裁剪特征，并仅用训练示范拟合 128 维 PCA；这项探针不能代替 π0 编码器。完整力场使用训练集拟合的 PCA，局部力位置不会在测试集重新拟合。</p>
<table><thead><tr><th>输入</th><th>接触前动作 XY RMSE mm</th><th>接触后动作 XY RMSE mm</th><th>接触后 XY 方向一致率</th></tr></thead><tbody>{report_table(fit['metrics'])}</tbody></table>
<h3>还能辨认什么物理状态</h3><p>这两个读数使用仿真真值作验证标签，不把它们送给输入特征。XY 是接触后 U 盘相对插座的位置误差；接触 AUROC 比较所有帧。分数下降才表明留出示范上有实际可用的信息。</p>
<table><thead><tr><th>输入</th><th>接触后 XY 位置 RMSE mm</th><th>插座接触 AUROC</th></tr></thead><tbody>{physical_table(fit['metrics'])}</tbody></table>
<p>“触觉单独”只含双指力值；S 是关节状态；S_R 是关节状态加双 RGB；S_R_Tmap_XYZ 额外加入了 taxel 坐标，用于检验位姿信息是否被误算作力的贡献。历史配置还可拆分前一动作与前一触觉。未来四步运动预测只额外输入当前动作，后续三步动作未受控，因此混合了物理响应与后续操作规律，不能解释为同一动作下的纯物理效果。</p>
<details><summary>读数边界与下载</summary><p>相关系数是描述，不是因果效应。力场符号可能随手指姿态和相对滑动改变。完整报告同时保留接触前/后的每项分数、条件相关的覆盖帧数和独立示范数。测试仅有 10 条独立示范，区间宽度应按示范数理解，不把相邻帧当作独立实验。</p>
<p><a href="fit/results.json">全部探针结果</a> · <a href="fit/test_metrics.csv">指标 CSV</a> · <a href="correlations.csv">力与状态/动作相关 CSV</a> · <a href="paired_intervals.json">配对区间</a> · <a href="cases.json">案例来源和数值</a></p></details>
</main></body></html>'''
    path = args.output / "index.html"
    path.write_text(page)
    atomic_json(args.output / "report_summary.json", dict(version=VERSION, paired_intervals=intervals,
               correlations=len(corr), cases=len(gallery["cases"]), figures=[str(x) for x in figs]))
    print(f"REPORT {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vision-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu", choices=("cpu", "auto", "cuda"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--stage", choices=("all", "prepare", "fit", "report"), default="all")
    parser.add_argument("--smoke", action="store_true", help="Use 6 train, 2 val and 2 test episodes; choose a separate output")
    parser.add_argument("--max-case-images", type=int, default=20)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.eval_root = args.eval_root.resolve()
    args.vision_weights = args.vision_weights.expanduser().resolve()
    if not args.vision_weights.is_file():
        parser.error(f"Local ResNet18 weights missing: {args.vision_weights}")
    manifest = json.loads((args.dataset / "spatial/manifest.json").read_text())
    entries = select_entries(manifest, args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.stage in ("all", "prepare"):
        stage_prepare(args)
    if args.stage in ("all", "fit"):
        if not (args.output / "episodes.json").exists():
            parser.error("prepare stage has not finished")
        stage_fit(args, entries)
    if args.stage in ("all", "report"):
        if not (args.output / "fit/complete.json").exists():
            parser.error("fit stage has not finished")
        stage_report(args, entries)
    print(f"DONE elapsed_s={time.monotonic()-started:.1f} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
