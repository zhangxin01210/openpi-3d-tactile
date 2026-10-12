#!/usr/bin/env python3
"""Resumable replay/capture batch for a precommitted cross-task input manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from contactworld_cross_task_common import TASKS, get_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-approved", action="store_true")
    args = parser.parse_args()
    spec = get_task(args.task)
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest.get("task") != spec.key or manifest.get("action_dim") != spec.action_dim:
        raise ValueError("Input manifest contract does not match requested task")
    episodes = manifest["episodes"][:args.limit]
    if not episodes:
        raise ValueError("No input episodes selected")
    if len(episodes) > 1 and not args.review_approved:
        parser.error("Bulk replay requires one reviewed episode: rerun with --review-approved")
    args.output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    results = []
    for ordinal, item in enumerate(episodes, 1):
        demo = Path(item["file"])
        if hashlib.sha256(demo.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Input NPZ checksum changed: " + str(demo))
        name = "%s_episode_%03d" % (spec.source_group, int(item["source_episode"]))
        folder, log = args.output / name, args.output / (name + ".log")
        replay_file = folder / "replay.json"
        v2, bilateral = folder / "v2_capture.npz", folder / "bilateral_tactile.npz"
        if replay_file.exists() and v2.is_file() and bilateral.is_file():
            replay, code, elapsed = json.loads(replay_file.read_text()), 0, 0.0
        else:
            command = ["bash", str(project / "sim/scripts/run_contactworld_sm120.sh"),
                       str(project / "sim/scripts/probe_contactworld_cross_task.py"),
                       "--task", spec.key, "--source", str(args.source), "--output", str(folder),
                       "--demo", str(demo), "--pose-feedback", "--seed", "0"]
            began = time.monotonic()
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=900, check=False)
            code, elapsed = result.returncode, time.monotonic() - began
            replay = json.loads(replay_file.read_text()) if replay_file.exists() else {}
        row = {"ordinal": ordinal, "source_episode": item["source_episode"], "frames": item["frames"],
               "returncode": code, "elapsed_s": elapsed, "replay": str(folder), "log": str(log),
               "executed_all_actions": replay.get("frames") == item["frames"],
               "v2_capture_exists": v2.is_file(), "bilateral_capture_exists": bilateral.is_file(),
               "source_success": replay.get("final_source_success")}
        results.append(row)
        (args.output / "batch_result.json").write_text(json.dumps({"status": "running", "task": spec.key,
            "input_manifest": str(args.input / "manifest.json"), "results": results}, indent=2) + "\n")
        print("%d/%d source=%s exit=%d %.1fs" % (ordinal, len(episodes), item["source_episode"], code, elapsed), flush=True)
    failures = [row for row in results if row["returncode"] or not row["executed_all_actions"] or
                not row["v2_capture_exists"] or not row["bilateral_capture_exists"]]
    (args.output / "batch_result.json").write_text(json.dumps({"status": "passed" if not failures else "needs_review",
        "task": spec.key, "input_manifest": str(args.input / "manifest.json"), "results": results,
        "failed_source_episodes": [row["source_episode"] for row in failures]}, indent=2) + "\n")
    if failures:
        raise SystemExit("Replay did not complete for: " + ", ".join(map(str, [row["source_episode"] for row in failures])))


if __name__ == "__main__":
    main()
