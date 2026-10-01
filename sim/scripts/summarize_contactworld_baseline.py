#!/usr/bin/env python3
"""Summarize fixed-seed q/dq/time pilot without treating pose proxies as physical truth."""
import argparse
import json
from pathlib import Path

import numpy as np


def wilson(successes, count, z=1.96):
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    spread = z*np.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [float(max(0, center-spread)), float(min(1, center+spread))]


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--usb", type=Path, required=True)
    parser.add_argument("--peg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    train = json.loads(args.train.read_text())
    report = {"scope": "Fixed seed pilot; source keypoint proximity and strict held pose proxy, not independent physical insertion success",
              "audit": {task: {k: entry[k] for k in
                       ("episodes", "source_endpoint_pass", "source_ever_pass", "strict_pose_proxy",
                        "retraction_candidates", "initial_socket_x_m", "initial_socket_y_m",
                        "initial_plug_to_socket_xy_m")}
                        for task, entry in audit["tasks"].items()},
              "train": {task: {"best_epoch": entry["best_epoch"],
                               "splits": entry["splits"]}
                        for task, entry in train["tasks"].items()},
              "online": {}}
    for task, path in (("insertion_usb", args.usb), ("insertion_peg", args.peg)):
        data = json.loads(path.read_text())
        results = data["results"]
        if len(results) != data["trials_requested"]:
            raise ValueError(f"Incomplete evaluation: {task}: {len(results)}")
        source_ever = sum(r["source_ever"] for r in results)
        source_final = sum(r["final"]["source_success"] for r in results)
        held_strict = sum(r["held_strict_pose"] for r in results)
        minimum = []
        for result in results:
            trial = json.loads(Path(result["trajectory"]).read_text())
            errors = [result["initial"]["keypoint_error_mm"]]
            errors.extend(row["keypoint_error_mm"] for row in trial["rows"])
            minimum.append(min(errors))
        report["online"][task] = {
            "trials": len(results), "seed_base": data["seed_base"],
            "max_steps": data["max_steps"], "policy_sha256": data["policy_sha256"],
            "source_ever_count": source_ever,
            "source_ever_wilson_95": wilson(source_ever, len(results)),
            "source_final_count": source_final,
            "held_strict_pose_count": held_strict,
            "initial_keypoint_error_mm": describe([r["initial"]["keypoint_error_mm"] for r in results]),
            "minimum_keypoint_error_mm": describe(minimum),
            "final_keypoint_error_mm": describe([r["final"]["keypoint_error_mm"] for r in results]),
            "initial_socket_x_m": describe([r["initial"]["socket_xy_m"][0] for r in results]),
            "initial_socket_y_m": describe([r["initial"]["socket_xy_m"][1] for r in results]),
            "seeds": [r["seed"] for r in results],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["online"], indent=2))


if __name__ == "__main__":
    main()
