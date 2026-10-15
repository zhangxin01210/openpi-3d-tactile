#!/usr/bin/env python3
"""Preflight and launch the four informative V3 configs for selected tasks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from contactworld_cross_task_common import TASKS, get_task

SUFFIXES = ("01_rgb", "02_fused_pc", "07_fused_pc_base_pooled", "10_fused_pc_base_map_matched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["peg", "barbed_flat", "valve"], choices=TASKS)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--exclude", nargs="*", default=[], help="Task/config names to skip")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error("Invalid label")
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        parser.error("--gpus must be distinct numeric GPU IDs")
    repo = Path(__file__).resolve().parents[2]
    weight = Path(os.environ.get("OPENPI_CONTACTWORLD_BASE_WEIGHTS", repo / "checkpoints/pi0_base/params"))
    if not weight.is_absolute():
        weight = repo / weight
    if not (weight / "_METADATA").exists():
        parser.error("Missing original pi0 base weights; set OPENPI_CONTACTWORLD_BASE_WEIGHTS: " + str(weight))
    selected = []
    for task in args.tasks:
        spec = get_task(task)
        root = repo / "data" / spec.output_dataset
        for required in (root / "meta/info.json", root / "spatial/manifest.json", root / "assets/norm/norm_stats.json"):
            if not required.exists():
                parser.error("Missing reviewed training input: " + str(required))
        subprocess.run([sys.executable, "sim/scripts/verify_contactworld_cross_task_lerobot.py", "--task", task, str(root)], cwd=repo, check=True)
        subprocess.run([sys.executable, "sim/scripts/audit_contactworld_cross_task_inputs.py", "--task", task, str(root)],
                       cwd=repo, env=dict(os.environ, JAX_PLATFORM_NAME="cpu"), check=True)
        for suffix in SUFFIXES:
            name = "pi0_cw3_%s_%s" % (task, suffix)
            if task not in args.exclude and suffix not in args.exclude and name not in args.exclude:
                selected.append(name)
    if not selected:
        parser.error("No configs remain after --exclude")
    if args.check_only:
        print("Preflight passed: %d configurations; original base=%s" % (len(selected), weight))
        print("\n".join(selected))
        return
    run_root = repo / "runs/contactworld_cross_task_train" / args.label
    if run_root.exists():
        parser.error("Run directory exists: " + str(run_root))
    for name in selected:
        target = repo / "checkpoints" / name / args.label
        if target.exists():
            parser.error("Checkpoint directory exists: " + str(target))
    run_root.mkdir(parents=True)
    pending, active, failed = list(selected), {}, []
    while pending or active:
        for gpu in gpus:
            if not pending or gpu in active:
                continue
            name = pending.pop(0)
            log = (run_root / (name + ".log")).open("w")
            command = [sys.executable, "scripts/train.py", name, "--exp-name", args.label]
            if not args.wandb:
                command.append("--no-wandb-enabled")
            process = subprocess.Popen(command, cwd=repo,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, XLA_PYTHON_CLIENT_MEM_FRACTION="0.88"),
                stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = process, log, name
            print("GPU %s: %s pid=%d" % (gpu, name, process.pid), flush=True)
        time.sleep(5)
        for gpu, (process, log, name) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code:
                failed.append(name)
            print("GPU %s: %s exit=%d" % (gpu, name, code), flush=True)
            del active[gpu]
    if failed:
        raise SystemExit("Failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
