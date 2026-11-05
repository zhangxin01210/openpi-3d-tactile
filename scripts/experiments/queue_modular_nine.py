#!/usr/bin/env python3
"""Nine modular XHand experiments: one GPU per run, opportunistic GPU 0..3 queue.

Shares checkpoint validation, resume conventions, and training invocation with
queue_final_v2.py, but DOES NOT require all GPUs idle before starting. Never
interrupts unrelated GPU processes. Run from an environment with OpenPI ready.

    nohup .venv/bin/python scripts/experiments/queue_modular_nine.py run > output/modular_nine_runner/nohup.log 2>&1 &
    .venv/bin/python scripts/experiments/queue_modular_nine.py status
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import queue_final_v2 as previous

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "output/modular_nine_runner"
EXPERIMENT = "seed42"
GPUS = (0, 1, 2, 3)
ARCHES = (
    "pi0_xhand_pc_v2_tactile_mlp_ho_v3",
    "pi0_xhand_pc_v2_tactile_mlp",
    "pi0_xhand_ho_v3_5p",
)
TASKS = ("peg_all", "press_the_button", "yellow_socket")
JOBS = tuple(f"{arch}/{task}" for arch in ARCHES for task in TASKS)


class OpportunisticQueue(previous.Queue):
    """Reuse the tested checkpoint and child-process handling, change GPU admission."""

    def worker_command(self, name, resume, gpu):
        args = [
            sys.executable,
            str(ROOT / "scripts/train.py"),
            name,
            "--exp-name", self.exp_name,
            "--checkpoint-base-dir", str(self.checkpoint_base),
            "--log-interval", "100",
        ]
        if resume:
            args.append("--resume")
        return args

    def launch(self, name, gpu, mode, step):
        # Config names contain '/', unlike legacy V2 names.
        (self.log_dir / f"{name}.log").parent.mkdir(parents=True, exist_ok=True)
        return super().launch(name, gpu, mode, step)

    def available(self, gpu):
        # Fail closed: if the GPU state cannot be inspected, don't schedule.
        processes = previous.gpu_processes([gpu])
        utilization = previous._gpu_utilization()[gpu]
        return (
            not processes
            and utilization["memory_mib"] <= 2048
            and utilization["utilization_percent"] <= 10
        )

    def _run(self):
        self.state.update(status="running", started=previous.now(), gpu_pool=list(self.gpus))
        pending = []
        for name in JOBS:
            mode, step = self.classify(name)
            path = str(self.log_dir / f"{name}.log")
            if mode == "complete":
                self.state["jobs"][name] = {
                    "status": "completed", "checkpoint_step": step,
                    "log": path, "skipped": True,
                }
            elif mode in {"invalid", "failed"}:
                self.state["jobs"][name] = {
                    "status": "failed", "checkpoint_step": step, "log": path,
                    "reason": f"Existing checkpoint classification: {mode}",
                }
            else:
                self.state["jobs"][name] = {"status": "queued", "checkpoint_step": step, "log": path}
                pending.append((name, mode, step))
        self.save()
        if any(item["status"] == "failed" for item in self.state["jobs"].values()):
            self.emit("Existing failed/invalid checkpoint; refusing to start queue")
            self.state.update(status="failed", ended=previous.now())
            self.save()
            return 1

        # Two consecutive checks reduce races with just-exited GPU workers.
        stable = {gpu: 0 for gpu in self.gpus}
        last_wait_log, last_error_log = 0.0, 0.0
        failure_seen = False
        while pending or self.running:
            self.reap()
            if any(job.get("status") == "failed" for job in self.state["jobs"].values()):
                failure_seen = True
            if not self.stopping and not failure_seen:
                for gpu in self.gpus:
                    if not pending or len(self.running) >= self.max_parallel:
                        break
                    if gpu in self.running:
                        stable[gpu] = 0
                        continue
                    try:
                        idle = self.available(gpu)
                    except (RuntimeError, OSError, KeyError, ValueError) as exc:
                        idle = False
                        if time.monotonic() - last_error_log > 60:
                            self.emit(f"GPU query failed (no task launched): {exc}")
                            last_error_log = time.monotonic()
                    stable[gpu] = stable[gpu] + 1 if idle else 0
                    if stable[gpu] < 2:
                        continue
                    # Query one more time immediately before launching.
                    try:
                        if not self.available(gpu):
                            stable[gpu] = 0
                            continue
                    except (RuntimeError, OSError, KeyError, ValueError):
                        stable[gpu] = 0
                        continue
                    name, mode, step = pending.pop(0)
                    self.launch(name, gpu, mode, step)
                    stable[gpu] = 0
            if self.stopping:
                if not self.running:
                    break
                self.kill_stuck()
            if pending and not self.stopping and not failure_seen and time.monotonic() - last_wait_log >= 120:
                self.emit(f"WAIT queued={len(pending)} running={len(self.running)} GPUs={self.gpus}")
                last_wait_log = time.monotonic()
            if pending or self.running:
                # Failure stops new launches; running jobs finish naturally.
                if failure_seen and not self.running:
                    break
                time.sleep(self.poll_seconds)
        self.reap()
        if self.stopping:
            status = "stopped"
            code = 130
        elif failure_seen or any(self.state["jobs"][name]["status"] != "completed" for name in JOBS):
            status = "failed"
            code = 1
        else:
            status = "completed"
            code = 0
        self.state.update(status=status, ended=previous.now())
        self.save()
        self.emit(f"QUEUE {status}")
        return code


def stop_runner(state_dir):
    info = json.loads((state_dir / "runner.json").read_text())
    pid = int(info["pid"])
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    if previous.proc_start(pid) != info.get("proc_start") or "queue_modular_nine.py" not in command or " run " not in f" {command} ":
        raise RuntimeError("Stale or unrelated runner PID: no signal sent")
    os.kill(pid, signal.SIGTERM)
    print(f"Requested graceful stop of modular queue PID {pid}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "stop"))
    parser.add_argument("--state-dir", type=Path, default=STATE)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    if args.action == "status":
        print((state_dir / "state.json").read_text())
        return 0
    if args.action == "stop":
        stop_runner(state_dir)
        return 0
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    # The old runner's worker performs a CUDA assertion. Preserve the
    # important property here: never silently fall back to CPU training.
    os.environ["JAX_PLATFORMS"] = "cuda"
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This modular queue is already running") from exc
        queue = OpportunisticQueue(
            state_dir=state_dir,
            checkpoint_base=ROOT / "checkpoints",
            exp_name=EXPERIMENT,
            gpus=GPUS,
            max_parallel=4,
            threads=2,
            phases=(JOBS,),
            final_step=19999,
            poll_seconds=args.poll_seconds,
        )
        previous.atomic_json(
            state_dir / "runner.json",
            {"pid": os.getpid(), "proc_start": previous.proc_start(os.getpid()),
             "started": previous.now()},
        )
        signal.signal(signal.SIGTERM, queue.request_stop)
        signal.signal(signal.SIGINT, queue.request_stop)
        queue.emit(f"RUN tasks={len(JOBS)} gpus={GPUS} max_parallel=4")
        return queue.run()


if __name__ == "__main__":
    sys.exit(main())
