#!/usr/bin/env python3
"""Test whether a geometry-neighbor action/path ensemble explains USB policy rollouts.

Retrieval uses only the initial physical state and the train split. Contact phase
alignment uses the observed first contact and is therefore a retrospective
diagnostic, not a deployable policy or a causal estimate of policy retrieval.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_contactworld_contact_phase import phase_actions, resample, similarity
from analyze_contactworld_training_neighbors import block_distances, geometry, score
from check_contactworld_visual_aliasing import features, first_frame


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def path_phase(g, contact, phase):
    if contact is None:
        return None
    if phase == "pre":
        if contact < 4:
            return None
        sl = slice(0, contact + 1)
        n = 25
    else:
        if len(g["ee_relative"]) - contact < 9:
            return None
        sl = slice(contact, min(contact + 21, len(g["ee_relative"])))
        n = 21
    return {key: resample(g[key][sl], n) for key in ("ee_relative", "relative")}


def action_metrics(query, candidate):
    if query is None or candidate is None:
        return None
    cos3, count3 = similarity(query, candidate)
    cosxy, countxy = similarity(query[:, :2], candidate[:, :2])
    rms = float(np.sqrt(np.mean(np.sum((query - candidate) ** 2, axis=1))) * 10)
    qrms = float(np.sqrt(np.mean(np.sum(query ** 2, axis=1))) * 10)
    crms = float(np.sqrt(np.mean(np.sum(candidate ** 2, axis=1))) * 10)
    return {"cos3": cos3, "cosxy": cosxy, "active3": count3, "activexy": countxy,
            "rms_mm": rms, "test_action_rms_mm": qrms, "candidate_action_rms_mm": crms}


def path_metrics(query, candidate):
    if query is None or candidate is None:
        return None
    out = {}
    for key in ("ee_relative", "relative"):
        q, c = query[key], candidate[key]
        name = "ee" if key == "ee_relative" else "plug"
        out[f"{name}_absolute_rmse_mm"] = float(np.sqrt(np.mean(np.sum((q - c) ** 2, axis=1))) * 1000)
        out[f"{name}_shape_rmse_mm"] = float(np.sqrt(np.mean(np.sum(((q-q[0])-(c-c[0])) ** 2, axis=1))) * 1000)
    return out


def pooled(demos, indices, weights, phase, kind):
    values = [demos[i][phase][kind] for i in indices]
    if any(v is None for v in values):
        return None
    if kind == "action":
        return np.einsum("i,ijk->jk", weights, np.stack(values))
    return {key: np.einsum("i,ijk->jk", weights, np.stack([v[key] for v in values]))
            for key in ("ee_relative", "relative")}


def summarize_rows(rows, seeds):
    # Average models within a seed before bootstrap; 520 rollouts are not 520
    # independent initial-state samples.
    rng = np.random.default_rng(20261003)
    boot = rng.integers(len(seeds), size=(5000, len(seeds)))
    setups = list(dict.fromkeys((r["label"] for r in rows)))
    baseline = {(r["model"], r["seed"], r["phase"]): r for r in rows if r["label"] == "nearest1"}
    metrics = ("cos3", "cosxy", "rms_mm", "ee_absolute_rmse_mm", "ee_shape_rmse_mm",
               "plug_absolute_rmse_mm", "plug_shape_rmse_mm", "candidate_action_rms_mm")
    result = {}
    for label in setups:
        result[label] = {}
        for phase in ("pre", "post"):
            phase_rows = [r for r in rows if r["label"] == label and r["phase"] == phase]
            entry = {"trajectories": len(phase_rows)}
            for metric in metrics:
                values = []
                paired = []
                for seed in seeds:
                    group = [r for r in phase_rows if r["seed"] == seed and r[metric] is not None]
                    if group:
                        values.append(np.mean([r[metric] for r in group]))
                    pairs = [(r[metric], baseline[(r["model"], seed, phase)][metric]) for r in group
                             if baseline[(r["model"], seed, phase)][metric] is not None]
                    if pairs:
                        paired.append(np.mean([x-y for x, y in pairs]))
                if values:
                    v = np.asarray(values)
                    entry[metric] = {"mean_by_seed": float(v.mean()), "seeds": len(v)}
                    if len(v) == len(seeds):
                        entry[metric]["bootstrap95"] = np.quantile(v[boot].mean(axis=1), [.025, .975]).tolist()
                if paired:
                    d = np.asarray(paired)
                    entry[metric + "_minus_nearest1"] = {"mean_by_seed": float(d.mean()),
                                                           "seeds": len(d), "seed_fraction_improved": float(np.mean(d > 0 if metric.startswith("cos") else d < 0))}
                    if len(d) == len(seeds):
                        entry[metric + "_minus_nearest1"]["bootstrap95"] = np.quantile(d[boot].mean(axis=1), [.025, .975]).tolist()
            result[label][phase] = entry
    return result


def nearest_switches(perturb_root, dataset, entries, initial):
    if not perturb_root.exists():
        return [], {}
    front, wrist = [], []
    for e in entries:
        p = f"episode_{e['episode_index']:06d}.mp4"
        front.append(features(first_frame(dataset / "videos/chunk-000/observation.images.front" / p)))
        wrist.append(features(first_frame(dataset / "videos/chunk-000/observation.images.wrist" / p)))
    front, wrist = np.stack(front), np.stack(wrist)
    model_paths = sorted(perturb_root.glob("pi0_cw3_usb_*"))
    # The two policies share each physical initial state. Use the RGB run for
    # the image proxy; independently verify the multimodal run's first images.
    rgb = next((p for p in model_paths if p.name == "pi0_cw3_usb_01_rgb"), None)
    if rgb is None:
        return [], {}
    rows = []
    for seed_dir in sorted(rgb.glob("seed_*")):
        baseline = None
        for condition in ("zero", "x_plus3", "x_minus3", "y_plus3", "y_minus3"):
            folder = seed_dir / condition
            if not (folder / "summary.json").exists():
                continue
            info = json.loads((folder / "summary.json").read_text())
            state = info["initial_perturbed_state"]
            q = geometry(np.r_[state["plug_pos"], state["plug_quat"]],
                         np.r_[state["socket_pos_gt"], state["socket_quat"]],
                         np.r_[state["ee_pos"], state["ee_quat"]])
            gd = score(block_distances(q, initial))
            with np.load(folder / "inputs.npz") as z:
                f, w = features(z["front"][0]), features(z["wrist"][0])
                for other in model_paths:
                    check = other / seed_dir.name / condition / "inputs.npz"
                    if other == rgb or not check.exists():
                        continue
                    with np.load(check) as zz:
                        assert np.array_equal(z["front"][0], zz["front"][0])
                        assert np.array_equal(z["wrist"][0], zz["wrist"][0])
            fd = np.sqrt((np.mean((front - f) ** 2, axis=1) + np.mean((wrist - w) ** 2, axis=1)) / 2)
            go, fo = np.argsort(gd), np.argsort(fd)
            record = {"seed": int(seed_dir.name.removeprefix("seed_")), "condition": condition,
                      "actual_offset_xyz_mm": info["offset_actual_xyz_mm"],
                      "geometry_top10": [int(entries[i]["source_episode"]) for i in go[:10]],
                      "geometry_distance_top10": gd[go[:10]].tolist(),
                      "pixel_top10": [int(entries[i]["source_episode"]) for i in fo[:10]],
                      "pixel_distance_top10": fd[fo[:10]].tolist()}
            if condition == "zero":
                baseline = record
            else:
                assert baseline is not None
                for method in ("geometry", "pixel"):
                    current, zero = record[method + "_top10"], baseline[method + "_top10"]
                    record[method + "_top1_changed"] = current[0] != zero[0]
                    record[method + "_top3_overlap"] = len(set(current[:3]) & set(zero[:3]))
                    record[method + "_top5_overlap"] = len(set(current[:5]) & set(zero[:5]))
                    record[method + "_top10_overlap"] = len(set(current) & set(zero))
            rows.append(record)
    shifted = [r for r in rows if r["condition"] != "zero"]
    summary = {"unique_initial_states": len(rows), "shifted_states": len(shifted),
               "geometry_top1_changes": sum(r["geometry_top1_changed"] for r in shifted),
               "pixel_top1_changes": sum(r["pixel_top1_changed"] for r in shifted),
               "mean_top5_overlap_geometry": float(np.mean([r["geometry_top5_overlap"] for r in shifted])),
               "mean_top5_overlap_pixel": float(np.mean([r["pixel_top5_overlap"] for r in shifted]))}
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--dataset", type=Path, default=Path("data/contactworld_usb_v3_bilateral"))
    p.add_argument("--perturb", type=Path, default=Path("/home/sai/zx/openpi-sim-runtime/runs/contactworld_target_perturb_20261003"))
    args = p.parse_args()
    out = args.root / "training_neighbors/ensemble_audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset / "spatial/manifest.json").read_text())
    entries = [e for e in manifest["episodes"] if e["split"] == "train"]
    demos = []
    for e in entries:
        folder = Path(manifest["capture_root"]) / f"insertion_usb_episode_{e['source_episode']:03d}"
        with np.load(folder / "v2_capture.npz") as z:
            actions = z["action_executed"].copy()
            g = geometry(z["plug_pose"], z["socket_pose"], z["ee_pose"])
        forces = np.linalg.norm([r["plug_socket_force_post"] for r in json.loads((folder / "replay.json").read_text())["rows"]], axis=1)
        hits = np.flatnonzero(forces > .1)
        contact = int(hits[0]) + 1 if len(hits) else None
        d = {"source": e["source_episode"], "contact": contact, "g": g}
        for phase in ("pre", "post"):
            d[phase] = {"action": phase_actions(actions, contact, phase) if contact is not None else None,
                        "path": path_phase(g, contact, phase)}
        demos.append(d)
    initial = {k: np.stack([d["g"][k][0] for d in demos]) for k in demos[0]["g"]}
    paths = sorted(args.root.glob("pi0_cw3_usb_*/summary.json"))
    assert len(paths) == 13, f"Expected 13 models, found {len(paths)}"
    seeds = [int(r["seed"]) for r in json.loads(paths[0].read_text())["results"]]
    assert len(seeds) == 40
    neighbors, results = [], []
    seed_order = {}
    for seed in seeds:
        t = json.loads((paths[0].parent / f"seed_{seed}/trajectory.json").read_text())
        st = t["result"]["initial_physical_state"]
        q = geometry(np.r_[st["plug_pos"], st["plug_quat"]], np.r_[st["socket_pos_gt"], st["socket_quat"]],
                     np.r_[st["ee_pos"], st["ee_quat"]])
        distances = score(block_distances(q, initial))
        order = np.argsort(distances)
        seed_order[seed] = (order, distances)
        for rank, i in enumerate(order[:10], 1):
            neighbors.append({"seed": seed, "rank": rank, "source_episode": demos[i]["source"],
                              "distance": float(distances[i])})
    for model, path in enumerate(paths, 1):
        for seed in seeds:
            t = json.loads((path.parent / f"seed_{seed}/trajectory.json").read_text())
            rows = t["rows"]
            init = t["result"]["initial_physical_state"]
            state_rows = [init] + rows
            g = geometry(np.array([r["plug_pos"] + r["plug_quat"] for r in state_rows]),
                         np.array([r["socket_pos_gt"] + r["socket_quat"] for r in state_rows]),
                         np.array([r["ee_pos"] + r["ee_quat"] for r in state_rows]))
            force = np.linalg.norm([r["plug_socket_contact_force_xyz"] for r in rows], axis=1)
            hit = np.flatnonzero(force > .1)
            contact = int(hit[0]) + 1 if len(hit) else None
            actions = np.asarray([r["action_sim_clipped"] for r in rows])
            order, distances = seed_order[seed]
            random_order = np.random.default_rng(20261003 + seed).choice(order[10:], size=10, replace=False)
            configs = [("nearest1", order[:1], np.ones(1))]
            for k in (3, 5, 10):
                idx = order[:k]
                configs.append((f"uniform{k}", idx, np.ones(k) / k))
                weights = np.exp(-(distances[idx] - distances[idx[0]]))
                configs.append((f"distance{k}", idx, weights / weights.sum()))
            configs.append(("random10", random_order, np.ones(10) / 10))
            for phase in ("pre", "post"):
                qa = phase_actions(actions, contact, phase) if contact is not None else None
                qp = path_phase(g, contact, phase)
                for label, idx, weights in configs:
                    candidate_action = pooled(demos, idx, weights, phase, "action")
                    candidate_path = pooled(demos, idx, weights, phase, "path")
                    am = action_metrics(qa, candidate_action)
                    pm = path_metrics(qp, candidate_path)
                    results.append({"model": model, "model_name": path.parent.name, "seed": seed,
                                    "ever": int(t["result"]["source_ever"]), "test_contact": contact,
                                    "phase": phase, "label": label, "k": len(idx),
                                    "effective_k": float(1 / np.sum(weights ** 2)),
                                    "sources": ";".join(str(demos[i]["source"]) for i in idx),
                                    **(am or {k: None for k in ("cos3", "cosxy", "active3", "activexy", "rms_mm", "test_action_rms_mm", "candidate_action_rms_mm")}),
                                    **(pm or {f"{part}_{kind}_rmse_mm": None for part in ("ee", "plug") for kind in ("absolute", "shape")})})
    write_csv(out / "neighbor_top10.csv", neighbors)
    write_csv(out / "ensemble_metrics.csv", results)
    switches, switch_summary = nearest_switches(args.perturb, args.dataset, entries, initial)
    (out / "perturb_neighbor_switches.json").write_text(json.dumps(switches, indent=2) + "\n")
    summary = {"train_demos": len(demos), "test_seeds": len(seeds), "models": len(paths),
               "units": "Means and bootstrap intervals over 40 seeds, models averaged within seed.",
               "retrieval": "Initial full geometry distance; train split only; top k fixed per seed before test actions.",
               "distance_weighting": "softmax(-(D-Dmin)/1.0), where D is the dimensionless initial geometry distance",
               "phase_alignment": "24 pre-contact action points, 20 post-contact action points; first force norm > 0.1 N. Path: 25 pre/21 post samples.",
               "configurations": summarize_rows(results, seeds), "perturb_nearest": switch_summary}
    (out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(results), "train_demos": len(demos), "perturb": switch_summary}, indent=2))


if __name__ == "__main__":
    main()
