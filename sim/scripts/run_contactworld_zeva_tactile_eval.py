#!/usr/bin/env python3
"""Validate and evaluate the four 1025 ContactWorld force-tactile Zeva policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path("/home/sai/csy/1024")
DEFAULT_INCREMENT = Path("/home/sai/csy/1025")
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


def task_problems(base: Path, increment: Path, task: str) -> tuple[list[str], Path | None]:
    shared = base / "tasks" / task
    tactile = increment / "tasks" / task
    action_dim, task_name = TASKS[task]
    required = (
        shared / "policy/base_backbone.safetensors",
        shared / "policy/norm_stats.json",
        shared / "cte/cte_step_001000.pt",
        shared / "cte_features/manifest.json",
        shared / "cte_features/features_000000.npz",
        base / "assets/paligemma_tokenizer.model",
        base / "assets/Wan2.2_VAE.pth",
        tactile / "policy/config.json",
        tactile / "policy/checkpoints/latest.json",
        increment / "code/pi0_zeva/model.py",
        increment / "code/cosmos-framework/cosmos_framework/model/zeva/contactworld_tactile_encoder.py",
    )
    problems = ["missing " + str(path) for path in required if not path.is_file()]
    if problems:
        return problems, None
    step_dir = None
    try:
        latest_value = json.loads(required[8].read_text())["checkpoint"]
        if Path(latest_value).name != latest_value:
            raise ValueError("latest.json is not a sibling directory name")
        step_dir = required[8].parent / latest_value
        manifest_path = step_dir / "manifest.json"
        memory_path = step_dir / "memory.safetensors"
        if not manifest_path.is_file() or not memory_path.is_file():
            raise FileNotFoundError("selected manifest or memory.safetensors is missing")
        manifest = json.loads(manifest_path.read_text())
        config = json.loads(required[7].read_text())
        expected = {
            "mode": "zeva_tactile", "dataset_family": "contactworld", "pi05": True,
            "camera_contract": "contactworld_bilateral_v1", "horizon": 16,
            "real_action_dim": action_dim, "real_state_dim": 18,
            "pim_top_k": 4, "zeva_architecture": "pi05_contactworld_v1",
            "tactile_architecture": "contactworld_force_v1", "tactile_memory_steps": 30,
        }
        for key, value in expected.items():
            if config.get(key) != value or manifest.get("config", {}).get(key) != value:
                problems.append(f"{key} mismatch (expected {value!r})")
        if config != manifest.get("config"):
            problems.append("policy config differs from selected checkpoint manifest config")
        if task_name not in config.get("data_root", ""):
            problems.append("data_root does not identify the selected task")
        if manifest.get("format_version") != 1 or manifest.get("memory") != "memory.safetensors":
            problems.append("unsupported tactile Zeva checkpoint manifest")
        if manifest.get("backbone") != "../../base_backbone.safetensors":
            problems.append("manifest does not declare the shared pinned backbone")
        hashes = manifest.get("artifact_hashes", {})
        checks = (
            (shared / "policy/norm_stats.json", manifest.get("norm_sha256"), "norm stats"),
            (base / "assets/paligemma_tokenizer.model", hashes.get("tokenizer"), "tokenizer"),
            (shared / "cte_features/manifest.json", hashes.get("cte_manifest"), "CTE manifest"),
        )
        for path, expected_hash, label in checks:
            if not expected_hash or sha256(path) != expected_hash:
                problems.append(label + " SHA-256 mismatch")
        with safe_open(memory_path, framework="pt", device="cpu") as weights:
            keys = list(weights.keys())
            if len(keys) != 108:
                problems.append(f"memory has {len(keys)} tensors, expected 108")
            for key, field in (
                ("tactile_encoder_projector.force_mean", "mean"),
                ("tactile_encoder_projector.force_std", "std"),
            ):
                if key not in keys or not np.allclose(
                        weights.get_tensor(key).numpy(), config["tactile_force_stats"][field],
                        rtol=0, atol=1e-9):
                    problems.append(f"checkpoint {field} force statistics mismatch")
            if "tactile_effect_gate" not in keys:
                problems.append("checkpoint is missing tactile_effect_gate")
        if required[0].stat().st_size < 7_000_000_000:
            problems.append("shared backbone is unexpectedly small")
        if memory_path.stat().st_size < 15_000_000:
            problems.append("tactile Zeva memory is unexpectedly small")
    except Exception as error:
        problems.append(f"invalid export metadata: {type(error).__name__}: {error}")
        step_dir = None
    return problems, step_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=tuple(TASKS))
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--increment", type=Path, default=DEFAULT_INCREMENT)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/contactworld_zeva_tactile_eval_1025")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--limit-this-run", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.replan_every != 4:
        parser.error("Zeva tactile export requires replan_every=4")
    base, increment = args.base.resolve(), args.increment.resolve()

    selected = []
    invalid = []
    for task in dict.fromkeys(args.tasks):
        problems, step_dir = task_problems(base, increment, task)
        if problems:
            invalid.append((task, problems))
            print("INCOMPLETE", task, "; ".join(problems), flush=True)
        else:
            selected.append((task, increment / "tasks" / task))
            print("READY", task, step_dir, flush=True)
    if args.check_only:
        print(f"CHECK_ONLY ready={len(selected)} incomplete={len(invalid)}", flush=True)
        raise SystemExit(2 if invalid else 0)
    if invalid:
        raise SystemExit("Refusing partial campaign because selected exports failed validation")
    if not selected:
        raise SystemExit("No policies selected")

    python = args.policy_python
    overlay = ROOT / ".venv-zeva/lib/python3.11/site-packages"
    if not python.is_file() or not overlay.is_dir():
        raise SystemExit("Prepared Blackwell-compatible Zeva Python runtime is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    policy_env = os.environ.copy()
    pythonpath = [overlay, base / "code", base / "code/cosmos-framework"]
    if policy_env.get("PYTHONPATH"):
        pythonpath.append(Path(policy_env["PYTHONPATH"]))
    policy_env.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        COSMOS_TRAINING="0",
        COSMOS_INTERNAL="0",
        COSMOS_DEVICE="cuda",
        PYTHONPATH=os.pathsep.join(map(str, pythonpath)),
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
            "cw-zeva-tactile-" +
            hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16] + ".sock")
        if socket.exists():
            raise FileExistsError("Active or stale socket: " + str(socket))
        server_cmd = [
            str(python), "-u", "sim/scripts/serve_contactworld_zeva_tactile_policy.py",
            "--base-root", str(base), "--increment-root", str(increment),
            "--task", task, "--socket", str(socket),
        ]
        eval_cmd = [
            "bash", "sim/scripts/run_contactworld_sm120.sh",
            "sim/scripts/eval_contactworld_ttt.py",
            "--task", task, "--source", str(args.source), "--output", str(output),
            "--socket", str(socket), "--checkpoint", str(task_dir),
            "--config", "pi05_zeva_tactile_" + task, "--trials", str(args.trials),
            "--seed-base", str(args.seed_base), "--max-steps", str(args.max_steps),
            "--replan-every", "4", "--limit-this-run", "1",
            "--ttt-runtime", "zeva_tactile",
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
                    raise TimeoutError("Tactile Zeva policy server did not become ready")
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
