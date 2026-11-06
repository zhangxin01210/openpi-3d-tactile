#!/usr/bin/env python3
"""Independent fail-closed, single-GPU-per-job XHand tactile-state queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from queue_final_v2 import atomic_json
from queue_final_v2 import checkpoint_state
from queue_final_v2 import gpu_processes
from queue_final_v2 import now

# The controller itself never allocates GPU memory. Worker processes opt in.
os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "output/tactile_state_ten_runner"
GPUS = (0, 1, 2, 3)
TASKS = ("peg_all", "press_the_button", "yellow_socket", "eraser")
NAMES = [f"{prefix}_xhand_rgb_baseline/eraser" for prefix in ("pi0", "pi05")] + [
    f"{prefix}_xhand_rgb_tactile_state/{task}" for task in TASKS for prefix in ("pi0", "pi05")]


def source_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def fingerprint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(path):
    if path.exists() and path.is_dir() and not any(p.name.isdigit() for p in path.iterdir()):
        if any(path.iterdir()):
            return "invalid", None
        return "fresh", None
    return checkpoint_state(path, 19999)


def gpu_available():
    busy = gpu_processes(GPUS)  # Raises on missing/ambiguous GPU process info.
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    usage = {}
    for line in raw.splitlines():
        index, memory, util = (int(x.strip()) for x in line.split(","))
        usage[index] = (memory, util)
    if any(gpu not in usage for gpu in GPUS):
        raise RuntimeError("GPU memory/utilization unavailable")
    return [gpu for gpu in GPUS if gpu not in busy and usage[gpu][0] < 100 and usage[gpu][1] == 0]


def inspect(name):
    import numpy as np

    from openpi.shared import normalize
    from openpi.training import config
    cfg = config.get_config(name)
    root = Path(cfg.data.repo_id)
    norm = Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id / "norm_stats.json"
    weight = Path(cfg.weight_loader.params_path)
    ckpt = ROOT / "checkpoints" / name / "seed42"
    record = {"config": name, "dataset": str(root), "backbone": "pi05" if cfg.model.pi05 else "pi0",
        "raw_tactile_state": cfg.data.raw_tactile_state, "state_dim": 1818 if cfg.data.raw_tactile_state else 18,
        "action_dim": cfg.model.action_dim, "action_horizon": cfg.model.action_horizon,
        "norm_stats": str(norm), "weight": str(weight), "max_token_len": cfg.model.max_token_len,
        "token_length_status": "unmeasured", "truncated": None, "state_values_encoded": None,
        "train_step": "not_run", "weight_compatibility": "not_tested", "errors": []}
    for label, fn in (("dataset_meta", lambda: fingerprint(root / "meta/info.json")),
                      ("norm_sha256", lambda: fingerprint(norm))):
        try:
            record[label] = fn()
        except Exception as exc:
            record["errors"].append(f"{label}: {exc}")
    if not weight.is_dir():
        record["errors"].append(f"base weight directory missing: {weight}")
    else:
        try:
            record["weight_metadata_sha256"] = fingerprint(weight / "_METADATA")
        except Exception as exc:
            record["errors"].append(f"base checkpoint metadata: {exc}")
    mode, step = classify(ckpt)
    record.update(checkpoint=str(ckpt), checkpoint_status=mode, checkpoint_step=step)
    if mode == "invalid":
        record["errors"].append("invalid or nonempty checkpoint directory; refusing overwrite")
    if norm.is_file():
        try:
            stats = normalize.load(norm.parent)
            for key, size in (("state", record["state_dim"]), ("actions", 18)):
                item = stats[key]
                if np.asarray(item.mean).shape != (size,) or np.asarray(item.std).shape != (size,):
                    raise ValueError(f"{key} expected {size} normalization values")
                if cfg.model.pi05 and (item.q01 is None or item.q99 is None):
                    raise ValueError(f"{key} missing quantiles")
                if any(not np.isfinite(v).all() for v in (item.mean, item.std, item.q01, item.q99) if v is not None):
                    raise ValueError(f"{key} contains non-finite stats")
        except Exception as exc:
            record["errors"].append(f"normalization: {exc}")
    return record


def write_report(report):
    STATE.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE / "preflight_report.json", report)
    lines = ["# Raw Tactile State 10-task preflight", "", f"Revision: `{report['revision']}`", "",
             "| Config | Dataset | State | Action / Horizon | Prompt max | Smoke | Status |", "|---|---|---:|---|---:|---|---|"]
    lines.extend(f"| {item['config']} | {item['dataset']} | {item['state_dim']} | {item['action_dim']} / {item['action_horizon']} | {item['max_token_len']} | {item['train_step']} | {item['status']} |" for item in report["jobs"])
    for item in report["jobs"]:
        lines += ["", f"## {item['config']}", "", f"- Norm: `{item['norm_stats']}`", f"- Weight: `{item['weight']}`",
                  f"- Checkpoint: `{item['checkpoint']}` ({item['checkpoint_status']})",
                  f"- Prompt: {item['token_length_status']}, truncated={item['truncated']}, values={item['state_values_encoded']}"]
        if item.get("smoke"):
            lines.append(f"- Real smoke: `{json.dumps(item['smoke'], ensure_ascii=False)}`")
        lines += [f"- ERROR: {error}" for error in item["errors"]]
    regression = report["legacy_regression"]
    lines += ["", f"Legacy regression: **{regression['status']}**", "",
              f"```\n{regression['output']}\n```", "",
              f"Queue readiness: **{report['status']}**", ""]
    (STATE / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def preflight():
    STATE.mkdir(parents=True, exist_ok=True)
    jobs = [inspect(name) for name in NAMES]
    regression_command = [sys.executable, "-m", "pytest", "-q",
        "src/openpi/training/final_v2_config_test.py", "src/openpi/training/pc_v2_config_test.py",
        "src/openpi/training/hand_tactile_config_test.py", "src/openpi/training/object_relation_config_test.py",
        "src/openpi/transforms_test.py", "src/openpi/models/tokenizer_test.py"]
    try:
        result = subprocess.run(regression_command, cwd=ROOT,
            env=dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES=""),
            text=True, capture_output=True, check=False, timeout=600)
        regression = {"status": "passed" if result.returncode == 0 else "failed",
                      "command": regression_command, "output": (result.stdout + result.stderr)[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        regression = {"status": "failed", "command": regression_command, "output": str(exc)}
    if regression["status"] != "passed":
        for job in jobs:
            job["errors"].append("legacy regression tests did not pass")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False).returncode != 0
    if dirty:
        for job in jobs:
            job["errors"].append("tracked source has uncommitted modifications; preflight must run on a committed revision")
    try:
        free = gpu_available()
        gpu_error = None if free else "No idle GPU in 0-3; real optimizer smoke not run"
    except Exception as exc:
        free, gpu_error = [], f"GPU state unavailable; no smoke started: {exc}"
    for job in jobs:
        if not job["errors"] and job["backbone"] == "pi05":
            try:
                audit_path = STATE / "audit" / (job["config"] + ".json")
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.unlink(missing_ok=True)
                env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
                audit = subprocess.run([sys.executable, str(ROOT / "scripts/experiments/smoke_tactile_state_ten.py"),
                    job["config"], "--audit-only", "--output", str(audit_path)],
                    cwd=ROOT, env=env, text=True, capture_output=True, check=False)
                if audit.returncode:
                    raise RuntimeError(audit.stderr[-2000:])
                evidence = json.loads(audit_path.read_text())
                lengths = [x["prompt_length_before_padding"] for x in evidence["samples"]]
                maximum = max(lengths)
                recommended = ((maximum + max(128, (maximum + 19) // 20) + 127) // 128) * 128
                job.update(token_length_status=f"measured {min(lengths)}..{maximum}", truncated=False,
                           state_values_encoded=job["state_dim"], recommended_max_token_len=recommended,
                           tokenizer_audit=evidence)
                if job["raw_tactile_state"] and job["max_token_len"] < recommended:
                    raise ValueError(f"prompt limit {job['max_token_len']} lacks 5%/128-token measured margin; "
                                     f"recommended {recommended}")
            except Exception as exc:
                job["errors"].append(f"real tokenizer audit: {exc}")
        if not job["errors"] and job["backbone"] == "pi05" and job["raw_tactile_state"]:
            # Three 224px/14 images, action horizon and padded language prompt.
            estimated_context = 3 * 256 + job["max_token_len"] + job["action_horizon"]
            job["estimated_context_tokens"] = estimated_context
            if estimated_context > 8192:
                job["errors"].append(f"conservative 8192-token total context budget exceeded: {estimated_context}; "
                    "use measured recommended_max_token_len via OPENPI_PI05_TACTILE_MAX_TOKEN_LEN after review")
        if not job["errors"] and free:
            # Recheck immediately before each smoke. No existing task is interrupted.
            try:
                free = gpu_available()
                if not free:
                    raise RuntimeError("GPU became occupied; smoke not started")
                gpu = free[0]
                output = STATE / "smoke" / (job["config"] + ".json")
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    output.unlink()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda", WANDB_MODE="offline",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false")
                command = [sys.executable, str(ROOT / "scripts/experiments/smoke_tactile_state_ten.py"),
                           job["config"], "--output", str(output)]
                result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"real smoke exit={result.returncode}: {result.stderr[-3000:]}")
                smoke = json.loads(output.read_text())
                if smoke.get("status") != "passed" or smoke.get("optimizer_steps") != 1:
                    raise RuntimeError("smoke did not prove a real optimizer step")
                job.update(smoke=smoke, train_step="passed", weight_compatibility="passed")
                if job["backbone"] == "pi05":
                    lengths = [sample["prompt_length_before_padding"] for sample in smoke["samples"]]
                    job.update(token_length_status=f"measured {min(lengths)}..{max(lengths)}",
                        truncated=False, state_values_encoded=job["state_dim"])
                    if job["raw_tactile_state"] and smoke["total_context_tokens"] > 8192:
                        raise RuntimeError(f"padded model context {smoke['total_context_tokens']} exceeds 8192; "
                                           "pretrained context limit has not been verified")
            except Exception as exc:
                job["errors"].append(f"real smoke: {exc}")
        elif gpu_error:
            job["errors"].append(gpu_error)
        if job["train_step"] != "passed":
            job["errors"].append("required real model/optimizer step not passed")
        job["status"] = "blocked" if job["errors"] else "passed"
        print(f"{job['config']}: {job['status']}", flush=True)
    report = {"revision": source_revision(), "status": "passed" if all(j["status"] == "passed" for j in jobs) else "blocked",
              "gpus": list(GPUS), "jobs": jobs, "created": now(),
              "legacy_regression": regression,
              "pi05_tactile_max_token_len_env": os.environ.get("OPENPI_PI05_TACTILE_MAX_TOKEN_LEN"),
              "note": "π0.5 8192 prompt length is provisional until representative real-data/context/memory preflight passes"}
    write_report(report)
    print(f"Report: {STATE / 'preflight_report.md'}")
    return 0 if report["status"] == "passed" else 1


def run():
    path = STATE / "preflight_report.json"
    if not path.is_file():
        raise RuntimeError("preflight report missing; run preflight")
    report = json.loads(path.read_text())
    if (report.get("status") != "passed" or report.get("revision") != source_revision() or
            report.get("pi05_tactile_max_token_len_env") != os.environ.get("OPENPI_PI05_TACTILE_MAX_TOKEN_LEN")):
        raise RuntimeError("preflight blocked or stale; run preflight again")
    if report.get("legacy_regression", {}).get("status") != "passed":
        raise RuntimeError("preflight report lacks passing legacy regression tests")
    jobs = report.get("jobs", [])
    if (len(jobs) != len(NAMES) or {job.get("config") for job in jobs} != set(NAMES) or
            any(job.get("status") != "passed" or job.get("train_step") != "passed" or
                job.get("smoke", {}).get("optimizer_steps") != 1 for job in jobs)):
        raise RuntimeError("preflight report lacks passing real smoke evidence for all ten configurations")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("tracked source changed after preflight; refusing to start")
    # Check all inputs and checkpoints again, and refuse changed assets.
    for previous in report["jobs"]:
        current = inspect(previous["config"])
        if current["errors"] or any(current.get(k) != previous.get(k) for k in ("dataset_meta", "norm_sha256", "weight_metadata_sha256")):
            raise RuntimeError(f"preflight inputs changed: {previous['config']}")
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = STATE / "stop.request"
        stop.unlink(missing_ok=True)
        running, jobs = {}, {}
        pending = list(NAMES)
        stop_started = None
        state = {"status": "running", "revision": source_revision(), "started": now(), "jobs": jobs}
        def save(): atomic_json(STATE / "state.json", state)
        def request_stop(*_): stop.touch()
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        save()
        while pending or running:
            for gpu, item in list(running.items()):
                code = item["process"].poll()
                if code is not None:
                    item["stream"].close()
                    mode, step = classify(item["checkpoint"])
                    jobs[item["name"]].update(status="completed" if code == 0 and mode == "complete" else "failed",
                        end_time=now(), exit_code=code, checkpoint_step=step)
                    del running[gpu]
                    save()
            if stop.exists():
                if stop_started is None:
                    stop_started = time.monotonic()
                for item in running.values():
                    if item["process"].poll() is None:
                        try:
                            sig = signal.SIGKILL if time.monotonic() - stop_started > 30 else signal.SIGTERM
                            os.killpg(item["process"].pid, sig)
                        except ProcessLookupError:
                            pass
                pending.clear()
                state["status"] = "stopping"
            if not stop.exists() and pending:
                try:
                    free = gpu_available()
                except Exception as exc:
                    state.update(status="gpu_state_unavailable", reason=str(exc))
                    save()
                    free = []
                for gpu in free:
                    if gpu in running or not pending:
                        continue
                    name = pending.pop(0)
                    checkpoint = ROOT / "checkpoints" / name / "seed42"
                    mode, step = classify(checkpoint)
                    if mode == "complete":
                        jobs[name] = {"status": "completed", "checkpoint_step": step, "skipped": True}
                        continue
                    if mode == "invalid":
                        jobs[name] = {"status": "failed", "reason": "invalid checkpoint"}
                        continue
                    log = STATE / "logs" / f"{name}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    stream = log.open("a", encoding="utf-8")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda", WANDB_MODE="offline",
                               XLA_PYTHON_CLIENT_PREALLOCATE="false")
                    cmd = [sys.executable, str(ROOT / "scripts/train.py"), name, "--exp-name", "seed42"]
                    if mode == "resume" or checkpoint.exists():
                        cmd.append("--resume")
                    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    running[gpu] = {"process": process, "stream": stream, "name": name, "checkpoint": checkpoint}
                    jobs[name] = {"config": name, "gpu": gpu, "pid": process.pid, "start_time": now(),
                                  "end_time": None, "checkpoint_step": step, "exit_code": None,
                                  "log_path": str(log), "status": "running"}
                    save()
            if pending or running:
                time.sleep(5)
        state.update(status="stopped" if stop.exists() else ("completed" if all(
            j.get("status") == "completed" for j in jobs.values()) else "failed"), ended=now())
        save()
        return 0 if state["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "status", "stop"))
    args = parser.parse_args()
    if args.action == "preflight":
        return preflight()
    if args.action == "run":
        return run()
    if args.action == "status":
        path = STATE / "state.json"
        print(path.read_text() if path.exists() else "Queue has not been started")
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "stop.request").touch()
    print("Stop requested; only this queue's own children will be signalled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
