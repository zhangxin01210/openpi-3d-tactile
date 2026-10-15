#!/usr/bin/env python3
"""Run resumable paired ContactWorld USB evaluations, one checkpoint at a time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-sm120")
DEFAULT_CHECKPOINTS = Path("/home/sai/zsq/FactileLDM/1002_ckpt")
DEFAULT_OUTPUT = Path("/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_11models_40trials_20261002")
CONFIGS = (
    "pi0_cw_usb_01_rgb", "pi0_cw_usb_02_rgb_pc", "pi0_cw_usb_03_rgb_ff",
    "pi0_cw_usb_04_rgb_pc_ff", "pi0_cw_usb_05_ff_summary",
    "pi0_cw_usb_06_ff_ee3d_proxy", "pi0_cw_usb_07_tacrgb",
    "pi0_cw_usb_08_tacdepth", "pi0_cw_usb_09_prefix",
    "pi0_cw_usb_10_split", "pi0_cw_usb_11_pcnoise",
)
SPARSE_CLOUD_CONFIGS = set(CONFIGS) - {"pi0_cw_usb_01_rgb", "pi0_cw_usb_03_rgb_ff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--limit-this-run", type=int)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fresh-env-per-trial", action="store_true")
    parser.add_argument("--isolate-simulator-per-trial", action="store_true",
                        help="Launch a new Isaac Gym process for every seed")
    parser.add_argument("--accept-sparse-cloud", action="store_true",
                        help="Acknowledge that point-cloud models use the sparse, unfiltered released cloud")
    args = parser.parse_args()
    if args.trials < 1 or (args.limit_this_run is not None and args.limit_this_run < 1):
        parser.error("Trial counts must be positive")
    if args.isolate_simulator_per_trial and not args.fresh_env_per_trial:
        parser.error("--isolate-simulator-per-trial requires --fresh-env-per-trial")
    if SPARSE_CLOUD_CONFIGS.intersection(args.configs) and not args.accept_sparse_cloud:
        parser.error("Point-cloud models use an unfiltered 1024-point cloud dominated by table pixels. "
                     "Review sim/reports/2026-10-02/contactworld_cloud_sampling_audit.md; "
                     "pass --accept-sparse-cloud only for an exploratory evaluation.")
    for config in args.configs:
        checkpoint = args.checkpoints / config / "cw_usb_4way/19999"
        if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
            parser.error(f"Incomplete checkpoint: {checkpoint}")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS="cuda",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.55")
    for config in args.configs:
        checkpoint = args.checkpoints / config / "cw_usb_4way/19999"
        if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")
        output = args.output / config
        output.mkdir(parents=True, exist_ok=True)
        socket = output / "policy.sock"
        if socket.exists():
            raise FileExistsError(f"Stale or active socket: {socket}")
        server_cmd = [sys.executable, "-u",
                      "sim/scripts/serve_contactworld_rgb_policy.py", "--config", config,
                      "--checkpoint", str(checkpoint), "--socket", str(socket), "--num-steps", "10"]
        sim_cmd = ["bash", "sim/scripts/run_contactworld_sm120.sh",
                   "sim/scripts/eval_contactworld_pi0_matrix.py", "--source", str(SOURCE),
                   "--output", str(output), "--socket", str(socket), "--checkpoint", str(checkpoint),
                   "--config", config, "--trials", str(args.trials), "--seed-base", str(args.seed_base),
                   "--max-steps", str(args.max_steps), "--replan-every", str(args.replan_every)]
        if args.limit_this_run:
            sim_cmd += ["--limit-this-run", str(args.limit_this_run)]
        if args.fresh_env_per_trial:
            sim_cmd += ["--fresh-env-per-trial"]
        print(f"START {config} output={output}", flush=True)
        with (output / "policy.log").open("a") as policy_log, (output / "eval.log").open("a") as eval_log:
            server = subprocess.Popen(server_cmd, cwd=REPO, env=env, stdout=policy_log,
                                      stderr=subprocess.STDOUT)
            try:
                for _ in range(240):
                    if socket.exists():
                        break
                    if server.poll() is not None:
                        raise RuntimeError(f"Policy server exited {server.returncode}; see {output / 'policy.log'}")
                    time.sleep(0.5)
                else:
                    raise TimeoutError(f"Policy server not ready; see {output / 'policy.log'}")
                if args.isolate_simulator_per_trial:
                    summary_path = output / "summary.json"
                    completed = len(json.loads(summary_path.read_text())["results"]) if summary_path.exists() else 0
                    stop = min(args.trials, completed + args.limit_this_run) if args.limit_this_run else args.trials
                    while completed < stop:
                        command = list(sim_cmd)
                        if "--limit-this-run" in command:
                            start = command.index("--limit-this-run")
                            del command[start:start + 2]
                        command += ["--limit-this-run", "1"]
                        with subprocess.Popen(command, cwd=REPO, env=env, stdout=eval_log,
                                              stderr=subprocess.STDOUT) as sim:
                            status = sim.wait()
                        if status != 0:
                            raise RuntimeError(f"Simulator exited {status}; see {output / 'eval.log'}")
                        completed = len(json.loads(summary_path.read_text())["results"])
                        print(f"{config} {completed}/{args.trials}", flush=True)
                else:
                    with subprocess.Popen(sim_cmd, cwd=REPO, env=env, stdout=eval_log,
                                          stderr=subprocess.STDOUT) as sim:
                        status = sim.wait()
                    if status != 0:
                        raise RuntimeError(f"Simulator exited {status}; see {output / 'eval.log'}")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                socket.unlink(missing_ok=True)
        print(f"DONE {config} index={output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
