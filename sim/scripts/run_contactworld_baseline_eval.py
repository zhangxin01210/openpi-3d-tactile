#!/usr/bin/env python3
"""Validate, load and evaluate exported ContactWorld Pi0.5 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TTT = Path("/home/sai/zsq/TactileTTT")
DEFAULT_SOURCE = Path("/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe")
TASK_DIRS = {
    "peg": "peg_baseline20k_3000",
    "usb": "usb_baseline20k_3000",
    "barbed_flat": "barbed_flat_baseline20k_2000",
    "valve": "valve_baseline20k_2000",
}


def checkpoint_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def checkpoint_problems(path: Path) -> list[str]:
    required = (
        "_CHECKPOINT_METADATA", "EXPORT_INFO.json", "params/_METADATA",
        "params/_sharding", "params/manifest.ocdbt",
        "params/ocdbt.process_0/manifest.ocdbt",
    )
    problems = ["missing " + item for item in required if not (path / item).is_file()]
    if not any((path / "assets").glob("*/norm_stats.json")):
        problems.append("missing assets/*/norm_stats.json")
    size = checkpoint_size(path) if path.exists() else 0
    if size < 8 * 1024**3:
        problems.append(f"only {size / 1024**3:.2f} GiB (expected a full ~12 GiB export)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_DIRS), default=tuple(TASK_DIRS))
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints/1023")
    parser.add_argument("--ttt-root", type=Path, default=DEFAULT_TTT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/contactworld_baseline_eval_1023")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=16)
    parser.add_argument("--limit-this-run", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    selected = []
    invalid = []
    for task in dict.fromkeys(args.tasks):
        checkpoint = args.checkpoints / TASK_DIRS[task]
        problems = checkpoint_problems(checkpoint)
        if problems:
            invalid.append((task, checkpoint, problems))
            print("INCOMPLETE", task, checkpoint, "; ".join(problems), flush=True)
        else:
            export = json.loads((checkpoint / "EXPORT_INFO.json").read_text())
            if export.get("task") != task or export.get("model") != "pi05_baseline":
                raise SystemExit(f"EXPORT_INFO mismatch: {checkpoint}")
            selected.append((task, checkpoint))
            print("READY", task, checkpoint, flush=True)
    if args.check_only:
        print(f"CHECK_ONLY ready={len(selected)} incomplete={len(invalid)}", flush=True)
        raise SystemExit(2 if invalid else 0)
    if invalid:
        raise SystemExit("Refusing partial campaign; finish importing every selected checkpoint first")
    if not selected:
        raise SystemExit("No checkpoints selected")

    ttt_python = args.ttt_root / ".venv/bin/python"
    if not ttt_python.is_file():
        raise SystemExit("Missing TactileTTT Python: " + str(ttt_python))
    args.output.mkdir(parents=True, exist_ok=True)
    policy_env = os.environ.copy()
    policy_env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.45",
        PYTHONPATH=str(args.ttt_root / "src"),
    )
    sim_env = os.environ.copy()
    sim_env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for task, checkpoint in selected:
        config = "pi05_cw_baseline_" + task
        output = args.output / task
        summary_path = output / "summary.json"
        completed = len(json.loads(summary_path.read_text()).get("results", [])) if summary_path.exists() else 0
        if completed >= args.trials:
            print("SKIP complete", task, flush=True)
            continue
        stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
        output.mkdir(parents=True, exist_ok=True)
        socket = Path("/tmp") / ("cw-base-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:20] + ".sock")
        if socket.exists():
            raise FileExistsError("Active or stale socket: " + str(socket))
        server_cmd = [str(ttt_python), "-u", "sim/scripts/serve_contactworld_baseline_policy.py",
                      "--config", config, "--checkpoint", str(checkpoint), "--socket", str(socket)]
        eval_cmd = ["bash", "sim/scripts/run_contactworld_sm120.sh", "sim/scripts/eval_contactworld_ttt.py",
                    "--task", task, "--source", str(args.source), "--output", str(output),
                    "--socket", str(socket), "--checkpoint", str(checkpoint), "--config", config,
                    "--trials", str(args.trials), "--seed-base", str(args.seed_base),
                    "--max-steps", str(args.max_steps), "--replan-every", str(args.replan_every),
                    "--limit-this-run", "1", "--ttt-runtime", "baseline"]
        print("START", config, completed, "->", stop, flush=True)
        with (output / "policy.log").open("a") as policy_log, (output / "eval.log").open("a") as eval_log:
            server = subprocess.Popen(server_cmd, cwd=ROOT, env=policy_env,
                                      stdout=policy_log, stderr=subprocess.STDOUT)
            try:
                for _ in range(600):
                    if socket.exists():
                        break
                    if server.poll() is not None:
                        raise RuntimeError("Policy server failed; see " + str(output / "policy.log"))
                    time.sleep(0.5)
                else:
                    raise TimeoutError("Policy server did not become ready")
                while completed < stop:
                    sim = subprocess.run(eval_cmd, cwd=ROOT, env=sim_env,
                                         stdout=eval_log, stderr=subprocess.STDOUT)
                    if sim.returncode:
                        raise RuntimeError("Simulator failed; see " + str(output / "eval.log"))
                    updated = len(json.loads(summary_path.read_text())["results"])
                    if updated != completed + 1:
                        raise RuntimeError("Simulator did not commit exactly one result")
                    completed = updated
                    print(config, completed, "/", args.trials, flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
                socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
