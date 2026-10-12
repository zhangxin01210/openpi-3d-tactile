#!/usr/bin/env python3
"""Paired V3 success and event diagnostics; highlight model disagreements."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path

import numpy as np

from contactworld_v3_eval_inputs import CONFIGS


PAIRS = (
    (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (6, 7),
    (4, 8), (8, 9), (8, 10), (9, 10), (6, 11), (6, 12), (4, 13),
)


def interval(success, total):
    z = 1.96
    p = success / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [float(center - half), float(center + half)]


def case_metrics(root, config, seed, contact_threshold):
    record = json.loads((root / config / ("seed_%d" % seed) / "trajectory.json").read_text())
    result, rows = record["result"], record["rows"]
    error = np.asarray([row["keypoint_error_mm"] for row in rows])
    root_error = np.asarray([row["plug_socket_xyz_error_mm"] for row in rows])
    contact = np.asarray([np.linalg.norm(row["plug_socket_contact_force_xyz"]) for row in rows])
    grip = np.asarray([row["ee_plug_distance_mm"] for row in rows])
    action = np.asarray([row["action_raw"] for row in rows])
    active_left = np.asarray([row["tacff_left_active_fraction"] for row in rows])
    active_right = np.asarray([row["tacff_right_active_fraction"] for row in rows])
    contact_steps = np.flatnonzero(contact > contact_threshold)
    first_contact = int(contact_steps[0]) if len(contact_steps) else None
    return {
        "seed": seed, "config": config,
        "source_ever": bool(result["source_ever"]),
        "source_final": bool(result["source_final"]),
        "first_success_step": result["first_success_step"],
        "longest_success_streak": result["max_consecutive_source_steps"],
        "initial_error_mm": float(result["initial"]["keypoint_error_mm"]),
        "min_error_mm": float(error.min()), "final_error_mm": float(error[-1]),
        "min_root_error_mm": float(root_error.min()),
        "first_socket_contact_step": first_contact,
        "socket_contact_steps": int(len(contact_steps)),
        "max_socket_contact_force": float(contact.max()),
        "error_at_first_contact_mm": float(error[first_contact]) if first_contact is not None else None,
        "post_contact_best_error_mm": float(error[first_contact:].min()) if first_contact is not None else None,
        "post_contact_error_reduction_mm": float(error[first_contact] - error[first_contact:].min())
        if first_contact is not None else None,
        "max_grip_slip_proxy_mm": float(np.max(np.abs(grip - result["initial"]["ee_plug_distance_mm"]))),
        "mean_left_active_fraction": float(active_left.mean()),
        "mean_right_active_fraction": float(active_right.mean()),
        "mean_pad_activity_difference": float(np.mean(active_left - active_right)),
        "action_clip_fraction": float((np.abs(action) > 1.0).mean()),
        "event_label": ("success_at_end" if result["source_final"] else
                        "transient_success" if result["source_ever"] else
                        "contact_without_success" if first_contact is not None else
                        "near_goal_without_contact" if error.min() < 30 else
                        "no_close_approach"),
    }


def error_sparkline(rows, contact_step, ceiling):
    """Self-contained SVG; opens from a local file without a JS/CDN fetch."""
    points = []
    for index, row in enumerate(rows):
        x = 5 + index * 290 / max(1, len(rows) - 1)
        y = 5 + (1 - min(float(row["keypoint_error_mm"]), ceiling) / ceiling) * 65
        points.append("%.1f,%.1f" % (x, y))
    marker = ""
    if contact_step is not None:
        x = 5 + contact_step * 290 / max(1, len(rows) - 1)
        marker = '<line x1="%.1f" x2="%.1f" y1="0" y2="75" stroke="#ed9240"/>' % (x, x)
    return ('<svg viewBox="0 0 300 75" style="width:100%;height:75px;background:#23282f" '
            'aria-label="keypoint error over time, orange mark is first socket contact">'
            '<line x1="0" x2="300" y1="70" y2="70" stroke="#777"/>' + marker +
            '<polyline fill="none" stroke="#7bbde8" stroke-width="2" points="%s"/></svg>' %
            " ".join(points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--contact-threshold", type=float, default=0.1,
                        help="Exploratory plug/socket pair-force event threshold")
    args = parser.parse_args()
    summaries = {}
    for config in CONFIGS:
        path = args.root / config / "summary.json"
        if path.is_file():
            summaries[config] = json.loads(path.read_text())
        elif not args.allow_partial:
            parser.error("Missing model summary: " + str(path))
    if not summaries:
        parser.error("No V3 model summaries found")
    seeds = sorted(set.intersection(*({row["seed"] for row in summary["results"]}
                                      for summary in summaries.values())))
    if not seeds:
        parser.error("No paired seeds yet")
    if not args.allow_partial and (len(summaries) != len(CONFIGS) or
                                   len(seeds) != next(iter(summaries.values()))["trials_requested"]):
        parser.error("Matrix incomplete; pass --allow-partial for interim analysis")
    for seed in seeds:
        hashes = {next(row for row in summary["results"] if row["seed"] == seed)["initial_state_sha256"]
                  for summary in summaries.values()}
        if len(hashes) != 1:
            raise ValueError("Unpaired physical initial state for seed %d" % seed)
    configs = [name for name in CONFIGS if name in summaries]
    records = [case_metrics(args.root, config, seed, args.contact_threshold)
               for seed in seeds for config in configs]
    by_config = {config: {row["seed"]: row for row in records if row["config"] == config}
                 for config in configs}
    with (args.root / "paired_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    rates = {}
    for config in configs:
        values = list(by_config[config].values())
        rates[config] = {
            "n": len(values),
            "source_ever": sum(row["source_ever"] for row in values),
            "source_final": sum(row["source_final"] for row in values),
            "ever_wilson95": interval(sum(row["source_ever"] for row in values), len(values)),
            "final_wilson95": interval(sum(row["source_final"] for row in values), len(values)),
            "event_counts": {label: sum(row["event_label"] == label for row in values)
                             for label in ("success_at_end", "transient_success",
                                           "contact_without_success", "near_goal_without_contact",
                                           "no_close_approach")},
        }
    comparisons = {}
    for left_number, right_number in PAIRS:
        left, right = CONFIGS[left_number - 1], CONFIGS[right_number - 1]
        if left not in by_config or right not in by_config:
            continue
        for metric in ("source_ever", "source_final"):
            improved = [seed for seed in seeds if by_config[right][seed][metric]
                        and not by_config[left][seed][metric]]
            regressed = [seed for seed in seeds if by_config[left][seed][metric]
                         and not by_config[right][seed][metric]]
            delta = np.asarray([int(by_config[right][seed][metric]) -
                                int(by_config[left][seed][metric]) for seed in seeds])
            samples = delta[np.random.default_rng(0).integers(len(seeds), size=(5000, len(seeds)))].mean(1)
            comparisons["%02d->%02d:%s" % (left_number, right_number, metric)] = {
                "improved_seeds": improved, "regressed_seeds": regressed,
                "rate_difference": float(delta.mean()),
                "paired_bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
            }
    report = {
        "paired_seeds": seeds, "models": configs, "rates": rates,
        "hypothesis_pairs": comparisons,
        "contact_threshold": args.contact_threshold,
        "interpretation": (
            "Source success is the simulator keypoint criterion. Event labels, pair-force thresholds "
            "and grip-slip proxy are exploratory associations, not verified physical failure causes. "
            "Forty seeds are an early diagnostic sample, not a definitive rank order."
        ),
    }
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    ranked = sorted(seeds, key=lambda seed: (
        -len({by_config[name][seed]["source_final"] for name in configs}),
        -sum(by_config[name][seed]["source_final"] for name in configs), seed))
    cards = []
    for seed in ranked[:20]:
        videos = []
        traces = {config: json.loads((args.root / config / ("seed_%d" % seed) /
                                      "trajectory.json").read_text())["rows"] for config in configs}
        ceiling = max(50., max(row["keypoint_error_mm"] for rows in traces.values() for row in rows))
        for config in configs:
            row = by_config[config][seed]
            curve = error_sparkline(traces[config], row["first_socket_contact_step"], ceiling)
            improvement = row["post_contact_error_reduction_mm"]
            diagnosis = ("contact %s · post-contact gain %s · grip change %.1f mm · L/R active %.2f/%.2f" % (
                row["first_socket_contact_step"],
                "%.1f mm" % improvement if improvement is not None else "n/a",
                row["max_grip_slip_proxy_mm"], row["mean_left_active_fraction"],
                row["mean_right_active_fraction"]))
            videos.append('<div><h3>%s · %s · %.1f mm</h3><video controls preload="none" '
                          'src="%s/seed_%d/front_wrist.mp4"></video>%s<p>%s</p><p>'
                          '<a href="%s/seed_%d/trajectory.json">trace</a> · '
                          '<a href="%s/seed_%d/inputs.npz">inputs</a></p></div>' % (
                              escape(config), escape(row["event_label"]), row["min_error_mm"],
                              config, seed, curve, escape(diagnosis), config, seed, config, seed))
        cards.append('<section><h2>Seed %d</h2><button onclick="playAll(this)">Play together</button> '
                     '<button onclick="pauseAll(this)">Pause all</button><div class="grid">%s</div></section>' %
                     (seed, "".join(videos)))
    (args.root / "case_index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>V3 paired USB cases</title>"
        "<style>body{font:15px sans-serif;max-width:1500px;margin:2rem auto;background:#16191d;color:#eee}"
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}"
        "video{width:100%}section{border-top:1px solid #555}h3{font-size:15px}a{color:#9bd}</style>"
        "</head><body><h1>Paired V3 USB cases</h1><p>Model disagreements first. Use the trace and "
        "inputs links to inspect action, contact and sensing around each event.</p>" + "".join(cards) +
        "<script>function playAll(b){b.closest('section').querySelectorAll('video')"
        ".forEach(x=>{x.currentTime=0;x.play()})}"
        "function pauseAll(b){b.closest('section').querySelectorAll('video').forEach(x=>x.pause())}</script>"
        "</body></html>", encoding="utf-8")
    print("paired=%d models=%d report=%s cases=%s" % (
        len(seeds), len(configs), args.root / "analysis.json", args.root / "case_index.html"))


if __name__ == "__main__":
    main()
