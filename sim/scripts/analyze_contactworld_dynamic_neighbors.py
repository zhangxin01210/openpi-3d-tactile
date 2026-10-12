#!/usr/bin/env python3
"""Retrieve training frames anew at every recorded policy observation.

This is an offline behavioral diagnostic. Geometry retrieval sees simulator
ground-truth poses; pixel retrieval uses low-resolution two-camera LAB pixels.
Neither is a claim that pi0 explicitly performs nearest-neighbor lookup.
"""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from analyze_contactworld_training_neighbors import block_distances, geometry, score


def image_feature(rgb, *, front):
    small = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    # The wrist camera is a close-up. Give the front camera twice the weight,
    # and discount static borders/table edges in both views. This remains a
    # transparent pixel-distance proxy, not a learned visual representation.
    weight = np.ones((16, 16, 1), np.float32) * (2 if front else 1)
    weight[:, :3] *= .2
    weight[:, 13:] *= .2
    weight[14:] *= .2
    return (lab * weight).reshape(-1)


def video_features(path, expected, *, front):
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(expected):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Missing frame {len(frames)}: {path}")
        frames.append(image_feature(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), front=front))
    cap.release()
    return np.stack(frames)


def build_library(dataset, manifest, cache):
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k].copy() for k in z.files}
    data = defaultdict(list)
    for e in manifest["episodes"]:
        if e["split"] != "train":
            continue
        capture = Path(manifest["capture_root"]) / f"insertion_usb_episode_{e['source_episode']:03d}" / "v2_capture.npz"
        with np.load(capture) as z:
            g = geometry(z["plug_pose"], z["socket_pose"], z["ee_pose"])
            actions = z["action_executed"]
        n = min(int(e["length"]) - 16 + 1, len(actions) - 3)
        assert n > 0
        ep = f"episode_{e['episode_index']:06d}.mp4"
        front = video_features(dataset / "videos/chunk-000/observation.images.front" / ep, n, front=True)
        wrist = video_features(dataset / "videos/chunk-000/observation.images.wrist" / ep, n, front=False)
        for key in g:
            data[key].append(g[key][:n])
        data["image"].append(np.concatenate([front, wrist], axis=1))
        data["action4"].append(np.stack([actions[i:i+4, :3] for i in range(n)]))
        data["source"].append(np.full(n, e["source_episode"], np.int32))
        data["frame"].append(np.arange(n, dtype=np.int32))
    library = {k: np.concatenate(v) for k, v in data.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **library)
    return library


def distinct_order(distances, source, limit=10, allowed=None):
    order = np.argsort(distances)
    selected = []
    seen = set()
    for i in order:
        if allowed is not None and source[i] not in allowed:
            continue
        if source[i] in seen:
            continue
        selected.append(int(i))
        seen.add(int(source[i]))
        if len(selected) == limit:
            break
    assert len(selected) == limit
    return selected


def cosine(a, b):
    mag = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    active = mag > .01
    return float(np.mean(np.sum(a[active] * b[active], axis=1) / mag[active])) if active.any() else None


def metrics(actual, candidate):
    return {"cos3": cosine(actual, candidate), "cosxy": cosine(actual[:, :2], candidate[:, :2]),
            "rms_mm": float(10 * np.sqrt(np.mean(np.sum((actual-candidate)**2, axis=1)))),
            "candidate_rms_mm": float(10 * np.sqrt(np.mean(np.sum(candidate**2, axis=1))))}


def summarize(details):
    rng = np.random.default_rng(20261003)
    keys = sorted(set((r["retrieval"], r["phase"], r["setting"]) for r in details))
    metrics_list = ("cos3", "cosxy", "rms_mm", "candidate_rms_mm")
    rows = []
    for retrieval, phase, setting in keys:
        group = [r for r in details if (r["retrieval"], r["phase"], r["setting"]) == (retrieval, phase, setting)]
        by_trajectory = defaultdict(list)
        for r in group:
            by_trajectory[(r["seed"], r["model"])].append(r)
        for (seed, model), values in by_trajectory.items():
            record = {"seed": seed, "model": model, "retrieval": retrieval, "phase": phase,
                      "setting": setting, "observations": len(values)}
            for key in metrics_list:
                v = [r[key] for r in values if r[key] is not None]
                record[key] = float(np.mean(v)) if v else None
            record["top1_switches"] = sum(a["top1_source"] != b["top1_source"] for a, b in zip(values, values[1:]))
            rows.append(record)
    baseline = {(r["seed"], r["model"], r["retrieval"], r["phase"]): r for r in rows if r["setting"] == "nearest1"}
    summary = {}
    for retrieval, phase, setting in keys:
        subset = [r for r in rows if (r["retrieval"], r["phase"], r["setting"]) == (retrieval, phase, setting)]
        entry = {"trajectories": len(subset), "observations": sum(r["observations"] for r in subset)}
        for metric in metrics_list:
            byseed = []
            deltas = []
            for seed in sorted({r["seed"] for r in subset}):
                seed_rows = [r for r in subset if r["seed"] == seed and r[metric] is not None]
                if seed_rows:
                    byseed.append(np.mean([r[metric] for r in seed_rows]))
                pairs = [(r[metric], baseline[(seed, r["model"], retrieval, phase)][metric]) for r in seed_rows
                         if baseline[(seed, r["model"], retrieval, phase)][metric] is not None]
                if pairs:
                    deltas.append(np.mean([a-b for a, b in pairs]))
            if byseed:
                entry[metric] = float(np.mean(byseed))
                entry[metric + "_seed_count"] = len(byseed)
            if deltas:
                d = np.array(deltas)
                entry[metric + "_minus_nearest1"] = float(d.mean())
                entry[metric + "_seed_fraction_improved"] = float(np.mean(d > 0 if metric.startswith("cos") else d < 0))
                if len(d) >= 10:
                    b = rng.integers(len(d), size=(5000, len(d)))
                    entry[metric + "_delta_bootstrap95"] = np.quantile(d[b].mean(axis=1), [.025, .975]).tolist()
        entry["top1_switches_per_trajectory"] = float(np.mean([r["top1_switches"] for r in subset]))
        summary[f"{retrieval}/{phase}/{setting}"] = entry
    return rows, summary


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("root", type=Path)
    pa.add_argument("--dataset", type=Path, default=Path("data/contactworld_usb_v3_bilateral"))
    args = pa.parse_args()
    out = args.root / "training_neighbors/ensemble_audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset / "spatial/manifest.json").read_text())
    library = build_library(args.dataset, manifest, out / "train_frame_library_roi.npz")
    geom = {k: library[k] for k in ("target", "relative", "grasp", "rotation", "grasp_rotation")}
    image = library["image"]
    image_norm = np.sum(image**2, axis=1)
    source_frames = {int(source): np.flatnonzero(library["source"] == source)
                     for source in np.unique(library["source"])}
    models = sorted(args.root.glob("pi0_cw3_usb_*/summary.json"))
    assert len(models) == 13
    details = []
    for model_id, summary_path in enumerate(models, 1):
        for result in json.loads(summary_path.read_text())["results"]:
            seed = result["seed"]
            trial = summary_path.parent / f"seed_{seed}"
            t = json.loads((trial / "trajectory.json").read_text())
            rows = t["rows"]
            action = np.asarray([r["action_sim_clipped"][:3] for r in rows])
            forces = np.linalg.norm([r["plug_socket_contact_force_xyz"] for r in rows], axis=1)
            hit = np.flatnonzero(forces > .1)
            contact = int(hit[0]) + 1 if len(hit) else None
            with np.load(trial / "inputs.npz") as z:
                steps = z["steps"].copy()
                fronts = z["front"].copy()
                wrists = z["wrist"].copy()
            assert len(steps) == len(fronts) == len(wrists)
            # At step s, row s-1 is the post-action state and thus the
            # physical observation at which the next action was chosen.
            states = [t["result"]["initial_physical_state"]] + rows[:-1]
            poses = geometry(np.asarray([r["plug_pos"] + r["plug_quat"] for r in states]),
                             np.asarray([r["socket_pos_gt"] + r["socket_quat"] for r in states]),
                             np.asarray([r["ee_pos"] + r["ee_quat"] for r in states]))
            observations = np.stack([np.concatenate((image_feature(f, front=True), image_feature(w, front=False)))
                                     for f, w in zip(fronts, wrists)])
            assert len(observations) == len(steps)
            # Batched low-resolution pixel distances, independent of geometry.
            visual_sq = np.sum(observations**2, axis=1, keepdims=True) + image_norm - 2 * observations @ image.T
            visual = np.sqrt(np.maximum(visual_sq, 0) / image.shape[1])
            fixed_sources = {}
            for oi, step in enumerate(steps):
                step = int(step)
                if step + 4 > len(action):
                    continue
                q = {k: poses[k][step:step+1] for k in geom}
                gd = score(block_distances(q, geom))
                distances = {"geometry": gd, "pixel": visual[oi]}
                if contact is not None and step < contact < step + 4:
                    # A four-action window crossing first contact belongs to
                    # neither clean phase.
                    continue
                phase = "pre" if contact is None or step < contact else "post"
                actual = action[step:step+4]
                for retrieval, d in distances.items():
                    selected = distinct_order(d, library["source"])
                    sources = [int(library["source"][i]) for i in selected]
                    if oi == 0:
                        fixed_sources[retrieval] = (set(sources[:1]), set(sources[:10]))
                    fixed_one = distinct_order(d, library["source"], limit=1,
                                               allowed=fixed_sources[retrieval][0])
                    fixed = distinct_order(d, library["source"], allowed=fixed_sources[retrieval][1])
                    rng = np.random.default_rng(20261003 + seed*1000 + step)
                    random_sources = rng.choice([s for s in source_frames if s not in sources],
                                                size=10, replace=False)
                    random_idx = np.array([rng.choice(source_frames[int(s)]) for s in random_sources])
                    configs = [("nearest1", selected[:1], np.ones(1)),
                               ("fixed_source1", fixed_one, np.ones(1)),
                               ("fixed_source10", fixed, np.ones(10)/10),
                               ("random10", random_idx, np.ones(10)/10)]
                    for k in (3, 5, 10):
                        ix = selected[:k]
                        configs.append((f"uniform{k}", ix, np.ones(k)/k))
                        # Unitless relative weighting adapts to this frame's
                        # distance scale, with a lower bound to avoid collapse.
                        local_scale = max(float(d[ix[-1]] - d[ix[0]]), float(d[ix[0]]) * .05, 1e-6)
                        w = np.exp(-(d[ix] - d[ix[0]]) / local_scale)
                        configs.append((f"distance{k}", ix, w/w.sum()))
                    for setting, ix, w in configs:
                        prediction = np.einsum("i,ijk->jk", w, library["action4"][ix])
                        detail = {"seed": seed, "model": model_id, "model_name": summary_path.parent.name,
                                  "step": step, "phase": phase, "contact_step": contact,
                                  "retrieval": retrieval, "setting": setting,
                                  "top1_source": sources[0], "top1_frame": int(library["frame"][selected[0]]),
                                  "top1_distance": float(d[selected[0]]),
                                  "sources": ";".join(str(int(library["source"][i])) for i in ix),
                                  "frames": ";".join(str(int(library["frame"][i])) for i in ix),
                                  "effective_k": float(1 / np.sum(w**2)), **metrics(actual, prediction)}
                        details.append(detail)
        print(f"completed model {model_id}: {summary_path.parent.name}", flush=True)
    with gzip.open(out / "dynamic_observations.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    per_trajectory, summary = summarize(details)
    with (out / "dynamic_per_trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_trajectory[0]))
        writer.writeheader()
        writer.writerows(per_trajectory)
    report = {"train_frames": len(image), "models": len(models), "retrieval": ["current ground-truth geometry", "current front+wrist LAB 16x16 task-ROI-weighted pixels"],
              "candidate_rule": "Nearest valid training frame, unique source episodes within top k; each frame contributes its following four executed translation actions.",
              "sampling": "Every saved policy observation (every four simulation steps); actual four executed actions compared to a weighted average of retrieved four-action windows.",
              "fixed_source_control": "Initial top 1/10 source episodes held fixed, but frame within each source reselected at every step.",
              "distance_weight": "softmax(-(d-d_min)/max(d_k-d_min,0.05*d_min,1e-6))",
              "statistical_unit": "40 initial seeds, 13 models averaged per seed; repeated observations are not independent.",
              "summary": summary}
    (out / "dynamic_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(details), "train_frames": len(image)}, indent=2))


if __name__ == "__main__":
    main()
