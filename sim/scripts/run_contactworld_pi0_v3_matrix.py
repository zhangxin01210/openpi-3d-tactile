#!/usr/bin/env python3
"""Run/resume 13 paired bilateral USB evaluations with isolated simulators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from contactworld_v3_eval_inputs import CONFIGS


REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe")
CHECKPOINTS = Path("/home/sai/zsq/FactileLDM/1002_ckpt")
OUTPUT = Path("/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003")


def checkpoint_bytes(path):
    if not (path / "_CHECKPOINT_METADATA").is_file() or not (path / "params/_METADATA").is_file():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--limit-this-run", type=int, help="New seeds per model in this invocation")
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-incomplete", action="store_true",
                        help="Run ready checkpoints; a later identical command picks up missing ones")
    args = parser.parse_args()
    if args.trials < 1 or args.max_steps < 1 or not 1 <= args.replan_every <= 16:
        parser.error("Invalid trial/step/replan count")
    if args.limit_this_run is not None and args.limit_this_run < 1:
        parser.error("--limit-this-run must be positive")
    if len(set(args.configs)) != len(args.configs):
        parser.error("Repeated configuration in --configs")
    patched = args.source / ("thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/"
                             "tacsl_task_usb.py")
    bilateral_line = "force_keys = ['tactile_force_field_left', 'tactile_force_field_right']"
    if not patched.is_file() or bilateral_line not in patched.read_text():
        parser.error("--source must point at the isolated patched bilateral simulator")
    paths = {config: args.checkpoints / config / "cw3_round1_20k/19999" for config in args.configs}
    sizes = {config: checkpoint_bytes(path) for config, path in paths.items()}
    complete = [size for size in sizes.values() if size > 4_000_000_000]
    reference_size = sorted(complete)[len(complete) // 2] if complete else 0
    threshold = int(reference_size * 0.90) if reference_size else 4_000_000_000
    pending = []
    for config, path in paths.items():
        status = "ready" if sizes[config] >= threshold else "incomplete"
        print("%s %-12s %.2f GB %s" % (config, status, sizes[config] / 1e9, path), flush=True)
        if status != "ready":
            pending.append(config)
    if pending and not args.skip_incomplete and not args.check_only:
        parser.error("Incomplete checkpoints: %s; wait, or pass --skip-incomplete" % ", ".join(pending))
    if args.check_only:
        print("CHECK_ONLY ready=%d pending=%d" % (len(paths) - len(pending), len(pending)))
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS="cuda",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.55")
    for config in args.configs:
        if config in pending:
            print("SKIP incomplete", config, flush=True)
            continue
        output = args.output / config
        summary_path = output / "summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            expected = {
                "policy_config": config, "checkpoint": str(paths[config].resolve()),
                "checkpoint_metadata_sha256": hashlib.sha256(
                    (paths[config] / "_CHECKPOINT_METADATA").read_bytes()).hexdigest(),
                "seed_base": args.seed_base, "trials_requested": args.trials,
                "max_steps": args.max_steps, "replan_every": args.replan_every,
                "source": str(args.source.resolve()),
            }
            for key, value in expected.items():
                if previous.get(key) != value:
                    raise ValueError("Existing evaluation protocol/checkpoint differs on %s in %s" % (
                        key, summary_path))
            completed = len(previous["results"])
            if completed >= args.trials:
                print("SKIP complete %s %d/%d" % (config, completed, args.trials), flush=True)
                continue
        else:
            completed = 0
        stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
        output.mkdir(parents=True, exist_ok=True)
        # Linux AF_UNIX paths are capped at 107 bytes; checkpoint/eval paths
        # are intentionally descriptive and exceed that limit.
        socket = Path("/tmp") / ("cw3-%s.sock" % hashlib.sha256(
            str(output.resolve()).encode()).hexdigest()[:20])
        if socket.exists():
            raise FileExistsError("Active or stale socket, inspect before removing: " + str(socket))
        server_cmd = [sys.executable, "-u", "sim/scripts/serve_contactworld_rgb_policy.py",
                      "--config", config, "--checkpoint", str(paths[config]),
                      "--socket", str(socket), "--num-steps", "10"]
        eval_cmd = ["bash", "sim/scripts/run_contactworld_sm120.sh",
                    "sim/scripts/eval_contactworld_pi0_v3.py", "--source", str(args.source),
                    "--output", str(output), "--socket", str(socket),
                    "--checkpoint", str(paths[config]), "--config", config,
                    "--trials", str(args.trials), "--seed-base", str(args.seed_base),
                    "--max-steps", str(args.max_steps), "--replan-every", str(args.replan_every),
                    "--limit-this-run", "1"]
        print("START %s %d->%d/%d" % (config, completed, stop, args.trials), flush=True)
        with (output / "policy.log").open("a") as policy_log, (output / "eval.log").open("a") as eval_log:
            server = subprocess.Popen(server_cmd, cwd=REPO, env=env,
                                      stdout=policy_log, stderr=subprocess.STDOUT)
            try:
                for _ in range(240):
                    if socket.exists():
                        break
                    if server.poll() is not None:
                        raise RuntimeError("Policy server failed; see " + str(output / "policy.log"))
                    time.sleep(0.5)
                else:
                    raise TimeoutError("Policy server not ready; see " + str(output / "policy.log"))
                while completed < stop:
                    with subprocess.Popen(eval_cmd, cwd=REPO, env=env,
                                          stdout=eval_log, stderr=subprocess.STDOUT) as sim:
                        status = sim.wait()
                    if status != 0:
                        raise RuntimeError("Simulator failed with code %d; see %s" % (
                            status, output / "eval.log"))
                    updated = len(json.loads(summary_path.read_text())["results"])
                    if updated != completed + 1:
                        raise RuntimeError("Simulator did not commit exactly one result")
                    completed = updated
                    print("%s %d/%d" % (config, completed, args.trials), flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                socket.unlink(missing_ok=True)
        print("DONE %s %s" % (config, output / "index.html"), flush=True)


if __name__ == "__main__":
    main()
