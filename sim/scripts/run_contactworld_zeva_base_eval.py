#!/usr/bin/env python3
"""Validate and evaluate the frozen Stage-1 backbones shipped in export 1024."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = Path("/home/sai/csy/1024")
DEFAULT_RUNTIME_ROOT = Path("/home/sai/csy/1024")
DEFAULT_POLICY_PYTHON = Path("/home/sai/zsq/Eligibility_Tracking/envs/sim5090/bin/python")
DEFAULT_SOURCE = Path("/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-bilateral-probe")
TASKS = {
    "peg": (6, "contactworld_peg_v3_bilateral"),
    "usb": (6, "contactworld_usb_v3_bilateral"),
    "barbed_flat": (7, "contactworld_barbed_flat_v3_bilateral"),
    "valve": (7, "contactworld_valve_v3_bilateral"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def task_problems(export: Path, runtime_root: Path, task: str) -> list[str]:
    task_dir = export / "tasks" / task
    action_dim, task_name = TASKS[task]
    config_path = task_dir / "policy/config.json"
    norm = task_dir / "policy/norm_stats.json"
    latest = task_dir / "policy/checkpoints/latest.json"
    tokenizer = runtime_root / "assets/paligemma_tokenizer.model"
    required = (config_path, norm, latest, tokenizer)
    problems = ["missing " + str(path) for path in required if not path.is_file()]
    if problems:
        return problems
    try:
        config = json.loads(config_path.read_text())
        selected = json.loads(latest.read_text())["checkpoint"]
        if Path(selected).name != selected:
            raise ValueError("latest.json is not a sibling directory name")
        manifest = json.loads((latest.parent / selected / "manifest.json").read_text())
        mode = config.get("mode")
        if mode == "baseline":
            backbone = latest.parent / selected / "backbone.safetensors"
            expected_backbone = "backbone.safetensors"
            expected_memory = None
        elif mode == "zeva":
            backbone = task_dir / "policy/base_backbone.safetensors"
            expected_backbone = "../../base_backbone.safetensors"
            expected_memory = "memory.safetensors"
        else:
            raise ValueError(f"unsupported source mode {mode!r}")
        if not backbone.is_file():
            problems.append("missing " + str(backbone))
        expected = {
            "mode": mode, "dataset_family": "contactworld", "pi05": True,
            "camera_contract": "contactworld_bilateral_v1", "horizon": 16,
            "real_action_dim": action_dim, "real_state_dim": 18,
        }
        for key, value in expected.items():
            if config.get(key) != value or manifest.get("config", {}).get(key) != value:
                problems.append(f"{key} mismatch (expected {value!r})")
        if task_name not in config.get("data_root", ""):
            problems.append("data_root does not identify the selected task")
        if manifest.get("backbone") != expected_backbone:
            problems.append("manifest does not select the expected baseline backbone")
        if manifest.get("memory") != expected_memory:
            problems.append("manifest memory field disagrees with source mode")
        if sha256(norm) != manifest.get("norm_sha256"):
            problems.append("norm stats SHA-256 mismatch")
        expected_tokenizer = manifest.get("artifact_hashes", {}).get("tokenizer")
        if not expected_tokenizer or sha256(tokenizer) != expected_tokenizer:
            problems.append("tokenizer SHA-256 mismatch")
        if backbone.stat().st_size < 7_000_000_000:
            problems.append("Stage-1 backbone is unexpectedly small")
    except Exception as error:
        problems.append(f"invalid export metadata: {type(error).__name__}: {error}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=tuple(TASKS))
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/contactworld_zeva_base_eval_1024")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--limit-this-run", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.replan_every != 4:
        parser.error("The Zeva comparison requires replan_every=4")
    export = args.export.resolve()
    runtime_root = args.runtime_root.resolve()

    selected, invalid = [], []
    for task in dict.fromkeys(args.tasks):
        problems = task_problems(export, runtime_root, task)
        if problems:
            invalid.append((task, problems))
            print("INCOMPLETE", task, "; ".join(problems), flush=True)
        else:
            selected.append((task, export / "tasks" / task))
            config = json.loads((export / "tasks" / task / "policy/config.json").read_text())
            latest = export / "tasks" / task / "policy/checkpoints/latest.json"
            selected_checkpoint = json.loads(latest.read_text())["checkpoint"]
            backbone = (
                latest.parent / selected_checkpoint / "backbone.safetensors"
                if config["mode"] == "baseline" else
                export / "tasks" / task / "policy/base_backbone.safetensors"
            )
            print("READY", task, backbone, flush=True)
    if args.check_only:
        print(f"CHECK_ONLY ready={len(selected)} incomplete={len(invalid)}", flush=True)
        raise SystemExit(2 if invalid else 0)
    if invalid:
        raise SystemExit("Refusing partial campaign because selected exports failed validation")

    overlay = ROOT / ".venv-zeva/lib/python3.11/site-packages"
    if not args.policy_python.is_file() or not overlay.is_dir():
        raise SystemExit("Prepared Blackwell-compatible Zeva Python runtime is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    policy_env = os.environ.copy()
    pythonpath = [overlay, runtime_root / "code", runtime_root / "code/cosmos-framework"]
    if policy_env.get("PYTHONPATH"):
        pythonpath.append(Path(policy_env["PYTHONPATH"]))
    policy_env.update(
        CUDA_VISIBLE_DEVICES=args.gpu, COSMOS_TRAINING="0", COSMOS_INTERNAL="0",
        COSMOS_DEVICE="cuda", PYTHONPATH=os.pathsep.join(map(str, pythonpath)),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    sim_env = os.environ.copy()
    sim_env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for task, task_dir in selected:
        output = args.output / task
        summary_path = output / "summary.json"
        completed = len(json.loads(summary_path.read_text()).get("results", [])) \
            if summary_path.exists() else 0
        if completed >= args.trials:
            print("SKIP complete", task, flush=True)
            continue
        stop = min(args.trials, completed + args.limit_this_run) \
            if args.limit_this_run else args.trials
        output.mkdir(parents=True, exist_ok=True)
        socket = Path("/tmp") / (
            "cw-zeva-base-" + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
            + ".sock")
        if socket.exists():
            raise FileExistsError("Active or stale socket: " + str(socket))
        server_cmd = [
            str(args.policy_python), "-u", "sim/scripts/serve_contactworld_zeva_base_policy.py",
            "--root", str(export), "--runtime-root", str(runtime_root),
            "--task", task, "--socket", str(socket),
        ]
        eval_cmd = [
            "bash", "sim/scripts/run_contactworld_sm120.sh",
            "sim/scripts/eval_contactworld_ttt.py", "--task", task,
            "--source", str(args.source), "--output", str(output),
            "--socket", str(socket), "--checkpoint", str(task_dir),
            "--config", "pi05_zeva_base_" + task, "--trials", str(args.trials),
            "--seed-base", str(args.seed_base), "--max-steps", str(args.max_steps),
            "--replan-every", "4", "--limit-this-run", "1",
            "--ttt-runtime", "zeva_base",
        ]
        print("START", task, completed, "->", stop, flush=True)
        with (output / "policy.log").open("a") as policy_log, \
                (output / "eval.log").open("a") as eval_log:
            server = subprocess.Popen(
                server_cmd, cwd=ROOT, env=policy_env,
                stdout=policy_log, stderr=subprocess.STDOUT)
            try:
                for _ in range(1200):
                    if socket.exists():
                        break
                    if server.poll() is not None:
                        raise RuntimeError("Policy server failed; see " + str(output / "policy.log"))
                    time.sleep(0.5)
                else:
                    raise TimeoutError("Zeva-base policy server did not become ready")
                while completed < stop:
                    sim = subprocess.run(
                        eval_cmd, cwd=ROOT, env=sim_env,
                        stdout=eval_log, stderr=subprocess.STDOUT)
                    if sim.returncode:
                        raise RuntimeError("Simulator failed; see " + str(output / "eval.log"))
                    updated = len(json.loads(summary_path.read_text())["results"])
                    if updated != completed + 1:
                        raise RuntimeError("Simulator did not commit exactly one result")
                    completed = updated
                    print(task, completed, "/", args.trials, flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
