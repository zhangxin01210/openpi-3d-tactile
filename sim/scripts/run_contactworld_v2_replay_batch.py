#!/usr/bin/env python3
"""Resumable, isolated replay/capture of the frozen positive USB selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_v2_full_20261002/input"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_v2_full_20261002/replay"))
    parser.add_argument("--source", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-sm120"))
    parser.add_argument("--limit", type=int, help="Stop after this many source episodes for a smoke run")
    parser.add_argument("--dual-tactile", action="store_true",
                        help="Require patched bilateral sensor and capture both pads")
    parser.add_argument("--review-approved", action="store_true",
                        help="Confirm that the single-episode bilateral review was accepted")
    args = parser.parse_args()
    if args.dual_tactile and (args.limit is None or args.limit > 1) and not args.review_approved:
        parser.error("Bilateral bulk replay requires the one-episode review first (--review-approved)")
    manifest = json.loads((args.input / "manifest.json").read_text())
    selected = manifest["episodes"][:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    results = []
    report_path = args.output / "batch_result.json"
    for ordinal, item in enumerate(selected, 1):
        episode = int(item["episode"])
        demo = Path(item["file"])
        if hashlib.sha256(demo.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source demo checksum changed: " + str(demo))
        name = "insertion_usb_episode_%03d" % episode
        out = args.output / name
        log = args.output / (name + ".log")
        capture = out / "v2_capture.npz"
        bilateral_capture = out / "bilateral_tactile.npz"
        replay_path = out / "replay.json"
        if replay_path.exists():
            previous = json.loads(replay_path.read_text())
            if (not capture.is_file() or previous.get("demo_sha256") != item["sha256"] or
                    previous.get("frames") != item["frames"] or
                    previous.get("pose_feedback") is not True or
                    previous.get("v2_capture") is not True or
                    (args.dual_tactile and (not bilateral_capture.is_file() or
                     previous.get("bilateral_tactile_capture") is not True)) or
                    previous.get("white_gel") is not True or
                    previous.get("white_mounts") is not True):
                raise ValueError("Existing replay is incompatible: " + str(out))
            returncode = 0
            elapsed = 0.0
        else:
            command = ["bash", str(project / "sim/scripts/run_contactworld_sm120.sh"),
                       str(project / "sim/scripts/probe_contactworld_usb.py"),
                       "--task", "usb", "--source", str(args.source),
                       "--output", str(out), "--demo", str(demo),
                       "--seed", "0", "--pose-feedback", "--capture-v2",
                       "--white-gel", "--white-mounts"]
            if args.dual_tactile:
                command.append("--dual-tactile")
            start = time.monotonic()
            try:
                with log.open("w") as stream:
                    result = subprocess.run(command, cwd=project, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=600,
                                            check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
            elapsed = time.monotonic() - start
        replay = json.loads(replay_path.read_text()) if replay_path.exists() else {}
        row = {"ordinal": ordinal, "episode": episode, "output_episode_v1": item["output_episode_v1"],
               "split": item["split"], "frames_expected": item["frames"],
               "returncode": returncode, "elapsed_s": elapsed,
               "executed_all_actions": replay.get("frames") == item["frames"],
               "final_source_success": replay.get("final_source_success"),
               "final_plug_error_mm": replay.get("final_plug_error_mm"),
               "v2_capture_exists": capture.is_file(), "log": str(log)}
        if args.dual_tactile:
            row["bilateral_capture_exists"] = bilateral_capture.is_file()
        results.append(row)
        report_path.write_text(json.dumps({"selection_manifest": str(args.input / "manifest.json"),
            "status": "running", "selected": len(selected), "completed": len(results),
            "results": results}, indent=2) + "\n")
        print("%d/%d episode=%d exit=%d success=%s time=%.1fs" % (
            ordinal, len(selected), episode, returncode,
            replay.get("final_source_success"), elapsed), flush=True)
    failed = [row for row in results if row["returncode"] or
              not row["executed_all_actions"] or not row["v2_capture_exists"] or
              (args.dual_tactile and not row["bilateral_capture_exists"]) or
              not row["final_source_success"]]
    report_path.write_text(json.dumps({"selection_manifest": str(args.input / "manifest.json"),
        "status": "passed" if not failed else "needs_review", "selected": len(selected),
        "completed": len(results), "failed_episodes": [x["episode"] for x in failed],
        "results": results}, indent=2) + "\n")
    if failed:
        raise SystemExit("V2 replay/capture needs review for source episodes: " +
                         ", ".join(str(x["episode"]) for x in failed))


if __name__ == "__main__":
    main()
