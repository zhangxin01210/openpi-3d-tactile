#!/usr/bin/env python3
"""Run a preregistered eight-episode ContactWorld replay audit sequentially."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--white-gel", action="store_true", help="Match released pale gel appearance (visual only)")
    p.add_argument("--white-mounts", action="store_true", help="Match released pale finger mounts (visual only)")
    p.add_argument("--usb-only", action="store_true", help="Run only the five fixed USB demonstrations")
    p.add_argument("--capture-dense", action="store_true", help="Collect a pre-action RGB-D/cloud pilot")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    manifest = json.loads((args.input / "manifest.json").read_text())
    results = []
    report = args.output / "batch_result.json"
    for item in manifest["episodes"]:
        task, episode = item["task"], item["episode"]
        if args.usb_only and task != "insertion_usb":
            continue
        demo = Path(item["file"])
        if hashlib.sha256(demo.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Exported demo checksum differs from manifest: {demo}")
        name = f"{task}_episode_{episode:03d}"
        path = args.output / name
        log = args.output / (name + ".log")
        if (path / "replay.json").exists():
            row = json.loads((path / "replay.json").read_text())
            if (row.get("frames") != item["frames"] or
                row.get("pose_feedback") is not True or
                row.get("white_gel") != args.white_gel or
                row.get("white_mounts") != args.white_mounts or
                row.get("demo_sha256") != item["sha256"] or
                Path(row.get("demo", "")) != demo or
                row.get("dense_capture", False) != args.capture_dense or
                (args.capture_dense and not (path / "dense_capture.npz").is_file())):
                raise ValueError(f"Existing replay does not match requested configuration: {path}")
            exit_code = 0
        else:
            cmd = ["bash", str(project / "sim/scripts/run_contactworld_sm120.sh"),
                   str(project / "sim/scripts/probe_contactworld_usb.py"),
                   "--task", "usb" if task == "insertion_usb" else "peg",
                   "--source", str(args.source), "--output", str(path),
                   "--demo", str(demo), "--seed", str(args.seed), "--pose-feedback"]
            if args.capture_dense:
                cmd.append("--capture-dense")
            if args.white_gel:
                cmd.append("--white-gel")
            if args.white_mounts:
                cmd.append("--white-mounts")
            started = time.monotonic()
            try:
                with log.open("w") as stream:
                    completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                               timeout=600 if args.capture_dense else 300, check=False)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
            row = json.loads((path / "replay.json").read_text()) if (path / "replay.json").exists() else {}
            row["process_elapsed_s"] = time.monotonic() - started
        result = {"task": task, "episode": episode, "frames": item["frames"],
                  "socket_initial_xy_m": item["socket_initial_xy_m"],
                  "exit_code": exit_code, "executed_all_actions": row.get("frames") == item["frames"],
                  "final_source_success": row.get("final_source_success"),
                  "final_plug_error_mm": row.get("final_plug_error_mm"),
                  "max_pre_action_plug_error_mm": max((x["plug_pre_error_mm"] for x in row.get("rows", [])), default=None),
                  "pointcloud_rgb_depth_agreement": row.get("source_pointcloud_alignment_passed"),
                  "dense_capture": row.get("dense_capture"),
                  "initial_errors": row.get("initial_errors"), "log": str(log),
                  "output": str(path), "process_elapsed_s": row.get("process_elapsed_s")}
        results.append(result)
        report.write_text(json.dumps({"scope": "fixed released demos; recorded EE targets available to replay; not a policy evaluation",
                                      "visual_white_gel": args.white_gel, "visual_white_mounts": args.white_mounts,
                                      "seed": args.seed,
                                      "results": results}, indent=2) + "\n")
        print(name, "exit", exit_code, "success", result["final_source_success"],
              "plug_error_mm", result["final_plug_error_mm"], flush=True)
    if any(x["exit_code"] or not x["executed_all_actions"] for x in results):
        raise SystemExit("One or more scheduled episode executions failed; inspect batch_result.json")


if __name__ == "__main__":
    main()
