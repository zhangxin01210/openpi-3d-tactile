#!/usr/bin/env python3
"""Summarize paired USB rollouts and select contrasting cases for visual diagnosis."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path

import numpy as np

CONFIGS = (
    "pi0_cw_usb_01_rgb", "pi0_cw_usb_02_rgb_pc", "pi0_cw_usb_03_rgb_ff",
    "pi0_cw_usb_04_rgb_pc_ff", "pi0_cw_usb_05_ff_summary",
    "pi0_cw_usb_06_ff_ee3d_proxy", "pi0_cw_usb_07_tacrgb",
    "pi0_cw_usb_08_tacdepth", "pi0_cw_usb_09_prefix",
    "pi0_cw_usb_10_split", "pi0_cw_usb_11_pcnoise",
)


def interval(success, total):
    if not total:
        return (0.0, 0.0)
    p = success / total
    z = 1.96
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return (float(center - half), float(center + half))


def details(root, config, seed):
    record = json.loads((root / config / f"seed_{seed}" / "trajectory.json").read_text())
    result, rows = record["result"], record["rows"]
    error = np.array([r["keypoint_error_mm"] for r in rows], np.float64)
    contact = np.array([np.linalg.norm(r["plug_socket_contact_force_xyz"]) for r in rows])
    grip = np.array([r["ee_plug_distance_mm"] for r in rows])
    action = np.array([r["action_raw"] for r in rows])
    active = np.array([r["tacff_active_fraction"] for r in rows])
    first_contact = np.flatnonzero(contact > 0.1)
    contact_step = int(first_contact[0]) if len(first_contact) else None
    return {
        "seed": seed, "config": config, "source_ever": result["source_ever"],
        "source_final": result["source_final"], "first_success_step": result["first_success_step"],
        "max_consecutive_source_steps": result["max_consecutive_source_steps"],
        "min_keypoint_error_mm": float(error.min()), "final_keypoint_error_mm": float(error[-1]),
        "first_socket_contact_step": contact_step, "socket_contact_steps": int((contact > 0.1).sum()),
        "max_socket_contact_force": float(contact.max()),
        "max_ee_plug_distance_change_mm": float(np.max(np.abs(grip - result["initial"]["ee_plug_distance_mm"]))),
        "mean_tacff_active_fraction": float(active.mean()),
        "action_clip_fraction": float((np.abs(action) > 1.0).mean()),
        "error_at_first_contact_mm": float(error[contact_step]) if contact_step is not None else None,
        "post_contact_best_error_mm": float(error[contact_step:].min()) if contact_step is not None else None,
        "post_contact_error_reduction_mm": (float(error[contact_step] - error[contact_step:].min())
                                            if contact_step is not None else None),
        "status": ("source_at_end" if result["source_final"] else
                   "transient_source" if result["source_ever"] else
                   "near_goal_no_source" if error.min() < 30 else "no_close_approach"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summaries = {config: json.loads((args.root / config / "summary.json").read_text()) for config in CONFIGS}
    seeds = [{r["seed"] for r in summary["results"]} for summary in summaries.values()]
    paired = sorted(set.intersection(*seeds))
    if not paired:
        raise ValueError("No paired seeds")
    requested = {s["trials_requested"] for s in summaries.values()}
    if len(requested) != 1 or (not args.allow_partial and len(paired) != requested.pop()):
        raise ValueError("Evaluation is incomplete; pass --allow-partial for an interim report")
    for seed in paired:
        hashes = {next(r for r in s["results"] if r["seed"] == seed)["initial_state_sha256"]
                  for s in summaries.values()}
        if len(hashes) != 1:
            raise ValueError(f"Unpaired initial physical state for seed {seed}")
    records = [details(args.root, config, seed) for seed in paired for config in CONFIGS]
    with (args.root / "paired_diagnostics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_config = {config: [r for r in records if r["config"] == config] for config in CONFIGS}
    metrics = {}
    for config, values in by_config.items():
        total = len(values)
        ever = sum(r["source_ever"] for r in values)
        final = sum(r["source_final"] for r in values)
        metrics[config] = {"n": total, "ever": ever, "final": final,
                           "ever_wilson95": interval(ever, total),
                           "final_wilson95": interval(final, total),
                           "statuses": {label: sum(r["status"] == label for r in values) for label in
                                        ("source_at_end", "transient_source", "near_goal_no_source", "no_close_approach")}}
    rgb = {r["seed"]: r for r in by_config[CONFIGS[0]]}
    paired_difference = {}
    for config in CONFIGS[1:]:
        current = {r["seed"]: r for r in by_config[config]}
        for metric in ("source_ever", "source_final"):
            better = [s for s in paired if current[s][metric] and not rgb[s][metric]]
            worse = [s for s in paired if rgb[s][metric] and not current[s][metric]]
            differences = np.array([int(current[s][metric]) - int(rgb[s][metric]) for s in paired])
            draw = np.random.default_rng(0).integers(len(paired), size=(5000, len(paired)))
            bootstrap = differences[draw].mean(axis=1)
            paired_difference[f"{config}:{metric}"] = {
                "improved_seeds": better, "regressed_seeds": worse,
                "rate_difference": (len(better) - len(worse)) / len(paired),
                "paired_bootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            }
    focused_pairs = ((CONFIGS[0], CONFIGS[1]), (CONFIGS[0], CONFIGS[2]),
                     (CONFIGS[1], CONFIGS[3]), (CONFIGS[3], CONFIGS[4]),
                     (CONFIGS[3], CONFIGS[5]), (CONFIGS[1], CONFIGS[6]),
                     (CONFIGS[1], CONFIGS[7]), (CONFIGS[3], CONFIGS[8]),
                     (CONFIGS[3], CONFIGS[9]), (CONFIGS[3], CONFIGS[10]))
    focused_comparisons = {}
    for base, added in focused_pairs:
        left = {r["seed"]: r for r in by_config[base]}
        right = {r["seed"]: r for r in by_config[added]}
        for metric in ("source_ever", "source_final"):
            gains = [seed for seed in paired if right[seed][metric] and not left[seed][metric]]
            losses = [seed for seed in paired if left[seed][metric] and not right[seed][metric]]
            differences = np.array([int(right[seed][metric]) - int(left[seed][metric]) for seed in paired])
            samples = differences[np.random.default_rng(0).integers(len(paired), size=(5000, len(paired)))].mean(1)
            focused_comparisons[f"{base}->{added}:{metric}"] = {
                "gains": gains, "losses": losses, "rate_difference": float(differences.mean()),
                "paired_bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
            }
    report = {"paired_seeds": paired, "metrics": metrics, "vs_rgb": paired_difference,
              "focused_comparisons": focused_comparisons,
              "note": "Source success is proximity under the simulator threshold, not a separate physical insertion test. "
                      "Contact and grip fields are diagnostic associations, not causal failure labels."}
    initial_by_seed = {
        r["seed"]: r["initial"] for r in summaries[CONFIGS[0]]["results"] if r["seed"] in paired
    }
    for field in ("socket_y_m", "initial_keypoint_error_mm"):
        values = np.array([
            initial_by_seed[seed]["socket_xy_m"][1] if field == "socket_y_m"
            else initial_by_seed[seed]["keypoint_error_mm"] for seed in paired
        ])
        cuts = np.quantile(values, [1 / 3, 2 / 3])
        report.setdefault("exploratory_strata", {})[field] = {
            "cutpoints": cuts.tolist(),
            "bins": [
                {"n": len(subset), "seed_count": len(subset),
                 "final_rate": {config: sum(r["source_final"] for r in by_config[config]
                                            if r["seed"] in subset) / len(subset)
                                for config in CONFIGS}}
                for subset in [
                    {seed for seed, value in zip(paired, values, strict=True)
                     if np.searchsorted(cuts, value, side="right") == band}
                    for band in range(3)
                ] if subset
            ],
        }
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    # Put disagreements first, then hard shared failures and easy shared successes.
    ranked = sorted(paired, key=lambda seed: (
        -len({next(r for r in by_config[c] if r["seed"] == seed)["source_final"] for c in CONFIGS}),
        -sum(next(r for r in by_config[c] if r["seed"] == seed)["source_final"] for c in CONFIGS), seed))
    cards = []
    for seed in ranked[:15]:
        videos = []
        for config in CONFIGS:
            row = next(r for r in by_config[config] if r["seed"] == seed)
            videos.append(f'<div><h3>{escape(config)} · {escape(row["status"])} · '
                          f'{row["min_keypoint_error_mm"]:.1f} mm</h3><video controls preload="none" '
                          f'src="{config}/seed_{seed}/front_wrist.mp4"></video></div>')
        cards.append(f'<section><h2>Seed {seed}</h2><button onclick="playAll(this)">Play together</button> '
                     f'<button onclick="pauseAll(this)">Pause all</button><div class="grid">'
                     + "".join(videos) + "</div></section>")
    (args.root / "case_index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Paired USB cases</title>"
        "<style>body{font:15px sans-serif;max-width:1500px;margin:2rem auto;background:#16191d;color:#eee}"
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}"
        "video{width:100%}section{border-top:1px solid #555}h3{font-size:15px}</style></head><body>"
        "<h1>Paired USB cases</h1><p>Model disagreements first. All videos use the same initial state per seed; "
        "sensor and trajectory archives are linked from each model index.</p>" + "".join(cards) +
        "<script>function playAll(b){const v=b.closest('section').querySelectorAll('video');"
        "v.forEach(x=>{x.currentTime=0;x.play()})}function pauseAll(b){"
        "b.closest('section').querySelectorAll('video').forEach(x=>x.pause())}</script></body></html>",
        encoding="utf-8")
    print(f"paired={len(paired)} report={args.root / 'analysis.json'} cases={args.root / 'case_index.html'}")


if __name__ == "__main__":
    main()
