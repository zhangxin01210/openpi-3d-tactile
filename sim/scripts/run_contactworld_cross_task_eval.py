#!/usr/bin/env python3
"""Run/resume the 12 cross-task ContactWorld policy evaluations.

The default matrix is 3 tasks × (RGB, fused point cloud, model 07, model 10),
40 paired random resets per checkpoint.  It reuses one policy server per
checkpoint and writes a self-contained result directory for every pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from contactworld_cross_task_common import TASKS, get_task
from contactworld_cross_task_eval_inputs import SUFFIXES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe")
DEFAULT_CHECKPOINTS = Path("/home/sai/zsq/FactileLDM/1014_ckpt")


def checkpoint_path(root: Path, task: str, suffix: str) -> Path:
    label = "cw3_peg_round1" if task == "peg" else "cw3_cross_round1"
    return root / f"pi0_cw3_{task}_{suffix}" / label / "19999"


def complete_checkpoint(path: Path) -> bool:
    return (path / "_CHECKPOINT_METADATA").is_file() and (path / "params/_METADATA").is_file()


def run_preflight(tasks: list[str], source: Path) -> None:
    for task in tasks:
        spec = get_task(task)
        root = ROOT / "data" / spec.output_dataset
        required = (root / "meta/info.json", root / "spatial/manifest.json",
                    root / "assets/norm/norm_stats.json")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("Missing %s input(s):\n%s" % (task, "\n".join(missing)))
        subprocess.run([
            sys.executable, "sim/scripts/verify_contactworld_cross_task_lerobot.py",
            "--task", task, str(root)
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, "sim/scripts/audit_contactworld_cross_task_inputs.py",
            "--task", task, str(root)
        ], cwd=ROOT, env=dict(os.environ, JAX_PLATFORM_NAME="cpu"), check=True)
    patched = source / "thirdparty/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl"
    for task in tasks:
        if not (patched / ("tacsl_task_" + get_task(task).hydra_config + ".py")).is_file():
            raise SystemExit("Missing task implementation in --source: " + task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--configs", nargs="+", choices=SUFFIXES, default=list(SUFFIXES),
                        help="Configuration suffixes, e.g. 01_rgb 02_fused_pc")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ("runs/contactworld_cross_task_eval_%s" %
                                        time.strftime("%Y%m%d")))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--limit-this-run", type=int,
                        help="New trials per task/config in this invocation")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-incomplete", action="store_true",
                        help="Skip missing checkpoints instead of stopping")
    parser.add_argument("--wandb", action="store_true", help="Reserved for protocol compatibility")
    args = parser.parse_args()
    if args.trials < 1 or args.max_steps < 1 or not 1 <= args.replan_every <= 16:
        parser.error("Invalid trial/step/replan count")
    if args.limit_this_run is not None and args.limit_this_run < 1:
        parser.error("--limit-this-run must be positive")
    if not args.source.is_dir():
        parser.error("Missing patched simulator source: " + str(args.source))
    if not args.checkpoints.is_dir():
        parser.error("Missing checkpoint root: " + str(args.checkpoints))
    if not args.gpu.isdigit():
        parser.error("--gpu must be a numeric GPU index")

    tasks = list(dict.fromkeys(args.tasks)); suffixes = list(dict.fromkeys(args.configs))
    run_preflight(tasks, args.source)
    pending = []
    for task in tasks:
        for suffix in suffixes:
            config = f"pi0_cw3_{task}_{suffix}"
            checkpoint = checkpoint_path(args.checkpoints, task, suffix)
            ready = complete_checkpoint(checkpoint)
            print("%s %-12s %s" % (config, "ready" if ready else "incomplete", checkpoint), flush=True)
            if not ready:
                pending.append((task, suffix, checkpoint))
    if pending and not args.skip_incomplete and not args.check_only:
        raise SystemExit("Incomplete checkpoints: " + ", ".join(f"{t}/{s}" for t, s, _ in pending))
    if args.check_only:
        print("CHECK_ONLY tasks=%d configs=%d ready=%d incomplete=%d" % (
            len(tasks), len(suffixes), len(tasks) * len(suffixes) - len(pending), len(pending)))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS="cuda",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.55")
    for task in tasks:
        for suffix in suffixes:
            checkpoint = checkpoint_path(args.checkpoints, task, suffix)
            if not complete_checkpoint(checkpoint):
                print("SKIP incomplete", task, suffix, flush=True)
                continue
            config = f"pi0_cw3_{task}_{suffix}"
            output = args.output / task / suffix
            summary_path = output / "summary.json"
            if summary_path.exists():
                saved = json.loads(summary_path.read_text())
                if int(saved.get("trials_requested", 0)) > args.trials:
                    raise ValueError("Existing evaluation uses more trials than requested: " + str(summary_path))
                completed = len(saved.get("results", []))
                if completed >= args.trials:
                    print("SKIP complete %s %d/%d" % (config, completed, args.trials), flush=True)
                    continue
            else:
                completed = 0
            stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
            output.mkdir(parents=True, exist_ok=True)
            socket = Path("/tmp") / ("cw3-cross-%s.sock" % hashlib.sha256(
                str(output.resolve()).encode()).hexdigest()[:20])
            if socket.exists():
                raise FileExistsError("Active or stale socket, inspect before removing: " + str(socket))
            server_cmd = [sys.executable, "-u", "sim/scripts/serve_contactworld_rgb_policy.py",
                          "--config", config, "--checkpoint", str(checkpoint),
                          "--socket", str(socket), "--num-steps", "10"]
            eval_cmd = ["bash", "sim/scripts/run_contactworld_sm120.sh",
                        "sim/scripts/eval_contactworld_cross_task.py", "--task", task,
                        "--source", str(args.source), "--output", str(output),
                        "--socket", str(socket), "--checkpoint", str(checkpoint),
                        "--config", config, "--trials", str(args.trials),
                        "--seed-base", str(args.seed_base), "--max-steps", str(args.max_steps),
                        "--replan-every", str(args.replan_every), "--limit-this-run", "1"]
            print("START %s %d->%d/%d" % (config, completed, stop, args.trials), flush=True)
            with (output / "policy.log").open("a") as policy_log, (output / "eval.log").open("a") as eval_log:
                server = subprocess.Popen(server_cmd, cwd=ROOT, env=env,
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
                        sim = subprocess.Popen(eval_cmd, cwd=ROOT, env=env,
                                               stdout=eval_log, stderr=subprocess.STDOUT)
                        status = sim.wait()
                        if status != 0:
                            raise RuntimeError("Simulator failed with code %d; see %s" %
                                               (status, output / "eval.log"))
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
                        server.kill(); server.wait()
                    socket.unlink(missing_ok=True)
            print("DONE %s %s" % (config, output / "index.html"), flush=True)


if __name__ == "__main__":
    main()

