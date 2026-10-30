#!/usr/bin/env python3
"""Load TactileTTT checkpoints and run/resume paired ContactWorld rollouts."""

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


def complete_checkpoint(path: Path) -> bool:
    return all((path / item).is_file() for item in (
        "_CHECKPOINT_METADATA", "params/_METADATA", "params/manifest.ocdbt", "EXPORT_INFO.json"
    )) and any((path / "assets").glob("*/norm_stats.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("peg", "usb", "barbed_flat", "valve"),
                        default=("peg", "usb", "barbed_flat", "valve"))
    parser.add_argument("--modes", nargs="+", choices=("v0", "v1"), default=("v0", "v1"))
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints/1021")
    parser.add_argument("--ttt-root", type=Path, default=DEFAULT_TTT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/contactworld_ttt_eval_20261022")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=16)
    parser.add_argument("--limit-this-run", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--disable-ttt", action="store_true",
                        help="Ablate TTT updates and residual contribution while keeping tactile inputs.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.checkpoints / "MANIFEST.json").read_text())
    entries = {(item["task"], item["ttt_version"]): item for item in manifest["models"]}
    selected = []
    for task in dict.fromkeys(args.tasks):
        for mode in dict.fromkeys(args.modes):
            item = entries[(task, mode)]
            checkpoint = args.checkpoints / item["dir"]
            if not complete_checkpoint(checkpoint):
                raise SystemExit("Incomplete checkpoint: " + str(checkpoint))
            selected.append((item, checkpoint))
            print("READY", item["config_name"], checkpoint, flush=True)
    if not (args.ttt_root / "src/openpi/policies/contactworld_ttt_policy.py").is_file():
        raise SystemExit("Missing TactileTTT source: " + str(args.ttt_root))
    ttt_python = args.ttt_root / ".venv/bin/python"
    if not ttt_python.is_file():
        raise SystemExit("Missing TactileTTT Python: " + str(ttt_python))
    if args.check_only:
        print("CHECK_ONLY_PASS models=%d" % len(selected), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    policy_env = os.environ.copy()
    policy_env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.55",
        PYTHONPATH=str(args.ttt_root / "src"),
    )
    sim_env = os.environ.copy()
    sim_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for item, checkpoint in selected:
        task, mode, config = item["task"], item["ttt_version"], item["config_name"]
        output = args.output / task / mode
        summary_path = output / "summary.json"
        completed = len(json.loads(summary_path.read_text()).get("results", [])) if summary_path.exists() else 0
        if completed >= args.trials:
            print("SKIP complete", task, mode, flush=True)
            continue
        stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
        output.mkdir(parents=True, exist_ok=True)
        socket = Path("/tmp") / ("cw-ttt-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:20] + ".sock")
        if socket.exists():
            raise FileExistsError("Active or stale socket: " + str(socket))
        server_cmd = [str(ttt_python), "-u", "sim/scripts/serve_contactworld_ttt_policy.py",
                      "--config", config, "--checkpoint", str(checkpoint), "--socket", str(socket)]
        if args.disable_ttt:
            server_cmd.append("--disable-ttt")
        eval_cmd = ["bash", "sim/scripts/run_contactworld_sm120.sh", "sim/scripts/eval_contactworld_ttt.py",
                    "--task", task, "--source", str(args.source), "--output", str(output),
                    "--socket", str(socket), "--checkpoint", str(checkpoint), "--config", config,
                    "--trials", str(args.trials), "--seed-base", str(args.seed_base),
                    "--max-steps", str(args.max_steps), "--replan-every", str(args.replan_every),
                    "--limit-this-run", "1", "--ttt-runtime", "noop" if args.disable_ttt else "online"]
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
