#!/usr/bin/env python3
"""Check whether the ±3 mm trials switch nearest training frames as they move."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_contactworld_dynamic_neighbors import build_library, distinct_order, image_feature
from analyze_contactworld_training_neighbors import block_distances, geometry, score


def trial_queries(folder):
    trajectory = json.loads((folder / "trajectory.json").read_text())
    rows = trajectory["rows"]
    info = json.loads((folder / "summary.json").read_text())
    states = [info["initial_perturbed_state"]] + rows[:-1]
    g = geometry(np.asarray([r["plug_pos"] + r["plug_quat"] for r in states]),
                 np.asarray([r["socket_pos_gt"] + r["socket_quat"] for r in states]),
                 np.asarray([r["ee_pos"] + r["ee_quat"] for r in states]))
    with np.load(folder / "inputs.npz") as z:
        steps = z["steps"].copy()
        images = np.stack([np.concatenate((image_feature(f, front=True), image_feature(w, front=False)))
                           for f, w in zip(z["front"], z["wrist"])])
    actions = np.asarray([r["action_sim_clipped"][:3] for r in rows])
    return g, steps, images, actions


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("perturb", type=Path)
    pa.add_argument("--dataset", type=Path, default=Path("data/contactworld_usb_v3_bilateral"))
    pa.add_argument("--library", type=Path, required=True)
    args = pa.parse_args()
    manifest = json.loads((args.dataset / "spatial/manifest.json").read_text())
    lib = build_library(args.dataset, manifest, args.library)
    geom = {k: lib[k] for k in ("target", "relative", "grasp", "rotation", "grasp_rotation")}
    image = lib["image"]
    image_norm = np.sum(image**2, axis=1)
    records = []
    for model_dir in sorted(args.perturb.glob("pi0_cw3_usb_*")):
        for seed_dir in sorted(model_dir.glob("seed_*")):
            baseline = None
            for condition in ("zero", "x_plus3", "x_minus3", "y_plus3", "y_minus3"):
                folder = seed_dir / condition
                if not (folder / "trajectory.json").exists():
                    continue
                g, steps, obs, actions = trial_queries(folder)
                pixel_sq = np.sum(obs**2, axis=1, keepdims=True) + image_norm - 2*obs @ image.T
                pixel = np.sqrt(np.maximum(pixel_sq, 0) / image.shape[1])
                per_step = []
                for n, step in enumerate(steps):
                    step = int(step)
                    q = {k: g[k][step:step+1] for k in geom}
                    d = {"geometry": score(block_distances(q, geom)), "pixel": pixel[n]}
                    r = {"model": model_dir.name, "seed": int(seed_dir.name[5:]), "condition": condition,
                         "step": step}
                    for method, distances in d.items():
                        order = distinct_order(distances, lib["source"])
                        r[method + "_top10"] = [int(lib["source"][i]) for i in order]
                        r[method + "_frames10"] = [int(lib["frame"][i]) for i in order]
                        r[method + "_distance1"] = float(distances[order[0]])
                    per_step.append(r)
                if condition == "zero":
                    baseline = (per_step, actions)
                    continue
                assert baseline is not None
                base_rows, base_actions = baseline
                assert [r["step"] for r in base_rows] == [r["step"] for r in per_step]
                for r, b in zip(per_step, base_rows):
                    step = r["step"]
                    a = actions[step:step+4]
                    aa = base_actions[step:step+4]
                    r["action4_rms_delta_mm"] = float(10*np.sqrt(np.mean(np.sum((a-aa)**2, axis=1))))
                    for method in ("geometry", "pixel"):
                        current = r[method + "_top10"]
                        zero = b[method + "_top10"]
                        r[method + "_top1_changed"] = current[0] != zero[0]
                        r[method + "_top5_overlap"] = len(set(current[:5]) & set(zero[:5]))
                        r[method + "_nearest_frame_changed"] = (current[0], r[method + "_frames10"][0]) != (zero[0], b[method + "_frames10"][0])
                    records.append(r)
    compact = []
    for r in records:
        compact.append({k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()})
    out = args.perturb / "analysis"
    with (out / "dynamic_neighbor_switches.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(compact[0]))
        w.writeheader()
        w.writerows(compact)
    summary = []
    for model_dir in sorted(args.perturb.glob("pi0_cw3_usb_*")):
        for seed in sorted({r["seed"] for r in records}):
            for condition in ("x_plus3", "x_minus3", "y_plus3", "y_minus3"):
                rr = [r for r in records if r["model"] == model_dir.name and r["seed"] == seed and r["condition"] == condition]
                if not rr:
                    continue
                entry = {"model": model_dir.name, "seed": seed, "condition": condition,
                         "first_action_delta_over_0.1mm_step": next((r["step"] for r in rr if r["action4_rms_delta_mm"] > .1), None)}
                for method in ("geometry", "pixel"):
                    entry[method + "_initial_top1_changed"] = bool(rr[0][method + "_top1_changed"])
                    entry[method + "_first_top1_switch_step"] = next((r["step"] for r in rr if r[method + "_top1_changed"]), None)
                    entry[method + "_first_frame_switch_step"] = next((r["step"] for r in rr if r[method + "_nearest_frame_changed"]), None)
                    entry[method + "_fraction_top1_changed"] = float(np.mean([r[method + "_top1_changed"] for r in rr]))
                    entry[method + "_mean_top5_overlap"] = float(np.mean([r[method + "_top5_overlap"] for r in rr]))
                summary.append(entry)
    (out / "dynamic_neighbor_switches.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"snapshots": len(records), "pairs": len(summary),
                      "initial_geometry_switches": sum(r["geometry_initial_top1_changed"] for r in summary),
                      "later_geometry_switches": sum(r["geometry_first_top1_switch_step"] is not None and not r["geometry_initial_top1_changed"] for r in summary),
                      "initial_pixel_switches": sum(r["pixel_initial_top1_changed"] for r in summary),
                      "later_pixel_switches": sum(r["pixel_first_top1_switch_step"] is not None and not r["pixel_initial_top1_changed"] for r in summary)}, indent=2))


if __name__ == "__main__":
    main()
