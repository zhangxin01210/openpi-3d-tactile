#!/usr/bin/env python3
"""Run the ten V2 experiments as a single-GPU dynamic queue.

Use `nohup ... run ... &` for a detached runner; `status` and `stop` address
its state directory. Benchmark runs always use an isolated checkpoint root.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PHASE_A = [
    f"{prefix}_xhand_{dataset}_{mode}_v2"
    for dataset in ("1028_press_button", "1030_yellow_socket")
    for prefix, mode in (("pi0", "rgb"), ("pi05", "rgb"), ("pi0", "st"), ("pi0", "full"))
]
PHASE_B = [f"pi0_xhand_1018_peg_all_{mode}_v2" for mode in ("st", "full")]
DEFAULT_STATE = ROOT / "output/final_v2_runner/main"
LOG_PATTERN = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d+)\s+\[[A-Z]\]\s+Step\s+(\d+):")


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def valid_checkpoint(path):
    """Conservatively recognize finalized Orbax composite checkpoints."""
    try:
        meta = json.loads((path / "_CHECKPOINT_METADATA").read_text())
        if not meta.get("commit_timestamp_nsecs"):
            return False
        for item in ("params", "train_state"):
            directory = path / item
            metadata = json.loads((directory / "_METADATA").read_text())
            if not metadata.get("tree_metadata"):
                return False
            if not (directory / "manifest.ocdbt").is_file():
                return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def checkpoint_state(directory, final_step):
    """Return (complete|resume|fresh|invalid, latest step or None)."""
    if not directory.exists():
        return "fresh", None
    if not directory.is_dir():
        return "invalid", None
    numeric = [p for p in directory.iterdir() if p.name.isdigit()]
    valid = sorted(int(p.name) for p in numeric if p.is_dir() and valid_checkpoint(p))
    invalid = [p.name for p in numeric if not p.is_dir() or not valid_checkpoint(p)]
    if invalid:
        return "invalid", None
    if not valid:
        return "fresh", None
    latest = valid[-1]
    return ("complete" if latest >= final_step else "resume"), latest


def gpu_processes(gpus):
    """Return compute processes on requested physical GPU indices; fail closed."""

    def query(args):
        try:
            return subprocess.run(["nvidia-smi", *args], text=True, check=True, capture_output=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Cannot inspect GPUs with nvidia-smi: {exc}") from exc

    rows = query(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    mapping = {}
    for row in rows.splitlines():
        fields = [x.strip() for x in row.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
            raise RuntimeError(f"Invalid nvidia-smi GPU row: {row!r}")
        mapping[int(fields[0])] = fields[1]
    missing = set(gpus) - set(mapping)
    if missing:
        raise RuntimeError(f"Requested GPU indices do not exist: {sorted(missing)}")
    wanted = {mapping[index]: index for index in gpus}
    busy = {}
    processes = query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    for row in processes.splitlines():
        fields = [x.strip() for x in row.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            raise RuntimeError(f"Cannot determine GPU process ownership: {row!r}")
        if fields[0] in wanted:
            busy.setdefault(wanted[fields[0]], []).append(int(fields[1]))
        elif fields[0] not in mapping.values():
            raise RuntimeError(f"Unknown or MIG GPU UUID in process list: {fields[0]}")
    return busy


def proc_start(pid):
    try:
        # comm can contain spaces and parentheses; fields following the last ')'
        # start at field 3, so index 19 is /proc stat field 22 (starttime).
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def send_stop(state_dir):
    path = state_dir / "runner.json"
    info = json.loads(path.read_text())
    pid = int(info["pid"])
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    if proc_start(pid) != info.get("proc_start") or "queue_final_v2.py" not in cmd or " run " not in f" {cmd} ":
        raise RuntimeError("Runner PID is stale or does not identify this queue; no signal sent")
    os.kill(pid, signal.SIGTERM)
    print(f"Stop requested for runner {pid}; only its own child process groups will be signalled.")


def _cpu_counters():
    values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(values), values[3] + values[4]


def _gpu_utilization():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    return {
        int(a.strip()): {"utilization_percent": float(b.strip()), "memory_mib": float(c.strip())}
        for a, b, c in (row.split(",") for row in result.stdout.splitlines())
    }


def stable_rates(state, warmup, log_interval=100):
    """Compute per-model and aggregate stable throughput from trainer step logs."""
    models, windows = {}, []
    for name, item in state["jobs"].items():
        path = Path(item["log"])
        if item.get("status") != "completed" or not path.is_file() or not item.get("started"):
            continue
        started = dt.datetime.fromisoformat(item["started"])
        midnight = started.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        points, day_offset, last_clock = [], 0, None
        for line in path.read_text(errors="replace").splitlines():
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            hour, minute, second, fraction, step = match.groups()
            clock = int(hour) * 3600 + int(minute) * 60 + int(second) + float("0." + fraction)
            if last_clock is None and clock < (started.timestamp() - midnight) - 43200:
                day_offset += 86400
            if last_clock is not None and clock < last_clock - 43200:
                day_offset += 86400
            last_clock = clock
            step = int(step)
            if step >= warmup:
                points.append((step, midnight + day_offset + clock))
        if len(points) < 2:
            continue
        first, last = points[0], points[-1]
        if last[0] <= first[0] or last[1] <= first[1]:
            continue
        work = last[0] - first[0]
        models[name] = {
            "steps_per_second": work / (last[1] - first[1]),
            "from_step": first[0],
            "to_step": last[0],
            "stable_seconds": last[1] - first[1],
        }
        windows.append((first[1], last[1], work, name))
    if not windows:
        return {"models": models, "total_steps_per_second": None, "note": "Insufficient stable step logs"}
    samples_path = Path(state["state_dir"]) / "utilization.jsonl"
    if samples_path.exists():
        samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
        for left, right, _, name in windows:
            matching = [
                (sample, gpu)
                for sample in samples
                if left <= dt.datetime.fromisoformat(sample["time"]).timestamp() <= right
                for gpu, job in sample["running"].items()
                if job == name
            ]
            cpus = [s["cpu_percent"] for s, _ in matching if s["cpu_percent"] is not None]
            gpus = [s["gpus"][gpu]["utilization_percent"] for s, gpu in matching if gpu in s["gpus"]]
            models[name].update(
                cpu_percent_mean=sum(cpus) / len(cpus) if cpus else None,
                gpu_percent_mean=sum(gpus) / len(gpus) if gpus else None,
                utilization_samples=len(matching),
            )
    windows.sort()
    merged, start, end = 0.0, windows[0][0], windows[0][1]
    for left, right, _, _ in windows[1:]:
        if left > end:
            merged += end - start
            start, end = left, right
        else:
            end = max(end, right)
    merged += end - start
    return {
        "models": models,
        "total_steps_per_second": sum(w[2] for w in windows) / merged,
        "aggregate_stable_seconds": merged,
        "log_interval": log_interval,
    }


class Queue:
    def __init__(
        self,
        *,
        state_dir,
        checkpoint_base,
        exp_name,
        gpus,
        max_parallel,
        threads,
        phases=(PHASE_A, PHASE_B),
        final_step=19999,
        command_factory=None,
        gpu_check=gpu_processes,
        skip_preflight=False,
        benchmark=False,
        benchmark_steps=400,
        warmup_steps=100,
        retry_failed=False,
        poll_seconds=1.0,
    ):
        self.state_dir, self.checkpoint_base = Path(state_dir), Path(checkpoint_base)
        self.exp_name, self.gpus, self.max_parallel, self.threads = exp_name, list(gpus), max_parallel, threads
        self.phases, self.final_step = phases, final_step
        self.command_factory = command_factory or self.worker_command
        self.gpu_check, self.skip_preflight, self.benchmark = gpu_check, skip_preflight, benchmark
        self.benchmark_steps, self.warmup_steps, self.retry_failed = benchmark_steps, warmup_steps, retry_failed
        self.poll_seconds = poll_seconds
        self.running, self.stopping = {}, False
        self.stop_started = None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.state_dir / "models"
        self.log_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "state.json"
        self.runner_log = self.state_dir / "runner.log"
        self.state = {
            "status": "starting",
            "jobs": {},
            "phase": None,
            "benchmark": benchmark,
            "state_dir": str(self.state_dir),
            "checkpoint_base": str(self.checkpoint_base),
            "exp_name": exp_name,
        }
        if self.state_file.exists():
            previous = json.loads(self.state_file.read_text())
            if previous.get("checkpoint_base") != str(self.checkpoint_base) or previous.get("exp_name") != exp_name:
                raise ValueError("State directory belongs to a different checkpoint base or experiment")
            self.state["jobs"] = previous.get("jobs", {})

    def emit(self, message):
        line = f"{now()} {message}"
        print(line, flush=True)
        with self.runner_log.open("a", encoding="utf-8") as stream:
            print(line, file=stream, flush=True)

    def save(self):
        atomic_json(self.state_file, self.state)

    def worker_command(self, name, resume, gpu):
        args = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--config",
            name,
            "--exp-name",
            self.exp_name,
            "--checkpoint-base-dir",
            str(self.checkpoint_base),
        ]
        if resume:
            args.append("--resume")
        if self.skip_preflight:
            args.append("--skip-preflight")
        if self.benchmark:
            args += ["--benchmark-steps", str(self.benchmark_steps)]
        return args

    def classify(self, name):
        directory = self.checkpoint_base / name / self.exp_name
        mode, step = checkpoint_state(directory, self.final_step)
        previous = self.state["jobs"].get(name, {})
        if mode == "invalid":
            return "invalid", step
        if mode == "complete" and previous.get("status") in {"failed", "interrupted"}:
            return ("resume" if self.retry_failed else "failed"), step
        return mode, step

    def request_stop(self, *_):
        self.stopping = True
        self.stop_started = self.stop_started or time.monotonic()
        self.emit("STOP requested; terminating only this runner's training process groups")
        for entry in self.running.values():
            process = entry["process"]
            if process.poll() is None:
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def launch(self, name, gpu, mode, step):
        log_path = self.log_dir / f"{name}.log"
        stream = log_path.open("a", encoding="utf-8", buffering=1)
        print(f"\n[{now()}] launch gpu={gpu} mode={mode} checkpoint_step={step}", file=stream, flush=True)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            OMP_NUM_THREADS=str(self.threads),
            MKL_NUM_THREADS=str(self.threads),
            OPENBLAS_NUM_THREADS=str(self.threads),
            PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])),
        )
        env.setdefault("WANDB_MODE", "offline")
        # Existing empty experiment directories need --resume to avoid the
        # trainer's FileExistsError; it initializes afresh when there are no steps.
        resume = mode == "resume" or (self.checkpoint_base / name / self.exp_name).exists()
        process = subprocess.Popen(
            self.command_factory(name, resume, gpu),
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.running[gpu] = {"process": process, "stream": stream, "name": name}
        self.state["jobs"][name] = {
            "status": "running",
            "gpu": gpu,
            "pid": process.pid,
            "started": now(),
            "ended": None,
            "exit_code": None,
            "checkpoint_step": step,
            "log": str(log_path),
        }
        self.emit(f"START {name} gpu={gpu} pid={process.pid} mode={mode} checkpoint={step}")
        self.save()

    def reap(self):
        for gpu, entry in list(self.running.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["stream"].close()
            name = entry["name"]
            mode, step = checkpoint_state(self.checkpoint_base / name / self.exp_name, self.final_step)
            status = "completed" if code == 0 and mode == "complete" else ("interrupted" if self.stopping else "failed")
            self.state["jobs"][name].update(status=status, ended=now(), exit_code=code, checkpoint_step=step)
            self.emit(f"END {name} gpu={gpu} exit={code} status={status} checkpoint={step}")
            del self.running[gpu]
            self.save()

    def sample_utilization(self, previous_cpu):
        try:
            total, idle = _cpu_counters()
            cpu = (
                None
                if previous_cpu is None or total == previous_cpu[0]
                else 100 * (1 - (idle - previous_cpu[1]) / (total - previous_cpu[0]))
            )
            sample = {
                "time": now(),
                "cpu_percent": cpu,
                "gpus": _gpu_utilization(),
                "running": {str(gpu): entry["name"] for gpu, entry in self.running.items()},
            }
            with (self.state_dir / "utilization.jsonl").open("a", encoding="utf-8") as out:
                print(json.dumps(sample), file=out)
            return total, idle
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.emit(f"UTILIZATION ERROR {exc}")
            return previous_cpu

    def run(self):
        try:
            return self._run()
        except BaseException as exc:
            self.emit(f"RUNNER ERROR {exc}")
            self.request_stop()
            while self.running:
                self.reap()
                self.kill_stuck()
                time.sleep(min(self.poll_seconds, 0.1))
            self.state.update(status="failed", ended=now(), reason=str(exc))
            self.save()
            raise

    def kill_stuck(self):
        if self.stop_started is None or time.monotonic() - self.stop_started < 10:
            return
        for entry in self.running.values():
            process = entry["process"]
            try:
                if process.poll() is None and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _run(self):
        occupied = self.gpu_check(self.gpus)
        if occupied:
            raise RuntimeError(f"Requested GPUs have running compute processes: {occupied}")
        previous_cpu, next_sample = None, 0.0
        self.state.update(status="running", started=now())
        self.save()
        for phase_index, phase in enumerate(self.phases, 1):
            self.state["phase"] = phase_index
            pending = []
            for name in phase:
                mode, step = self.classify(name)
                if mode == "complete":
                    self.state["jobs"][name] = {
                        **self.state["jobs"].get(name, {}),
                        "status": "completed",
                        "checkpoint_step": step,
                        "log": str(self.log_dir / f"{name}.log"),
                        "skipped": True,
                    }
                    self.emit(f"SKIP {name} complete checkpoint={step}")
                elif mode in {"invalid", "failed"}:
                    self.state["jobs"][name] = {
                        "status": "failed",
                        "checkpoint_step": step,
                        "log": str(self.log_dir / f"{name}.log"),
                        "reason": f"{mode} checkpoint or previous failed final run",
                    }
                    self.emit(f"FAIL {name}: {mode}; inspect checkpoint and runner state")
                else:
                    pending.append((name, mode, step))
                    self.state["jobs"][name] = {
                        "status": "queued",
                        "checkpoint_step": step,
                        "log": str(self.log_dir / f"{name}.log"),
                    }
            self.save()
            while (pending or self.running) and not (self.stopping and not self.running):
                self.reap()
                self.kill_stuck()
                while pending and not self.stopping and len(self.running) < self.max_parallel:
                    free = next((gpu for gpu in self.gpus if gpu not in self.running), None)
                    if free is None:
                        break
                    occupied = self.gpu_check([free])
                    if occupied:
                        self.emit(f"GPU CONFLICT {occupied}; stopping queue")
                        self.stopping = True
                        break
                    name, mode, step = pending.pop(0)
                    self.launch(name, free, mode, step)
                if self.benchmark and time.monotonic() >= next_sample:
                    previous_cpu = self.sample_utilization(previous_cpu)
                    next_sample = time.monotonic() + 5.0
                if pending or self.running:
                    time.sleep(self.poll_seconds)
            self.reap()
            if self.stopping:
                self.state.update(status="stopped", ended=now())
                self.save()
                return 130
            if any(self.state["jobs"][name]["status"] != "completed" for name in phase):
                self.emit(f"PHASE {phase_index} failed; later phases will not start")
                self.state.update(status="failed", ended=now())
                self.save()
                return 1
        self.state.update(status="completed", ended=now())
        self.save()
        if self.benchmark:
            report = stable_rates(self.state, self.warmup_steps)
            atomic_json(self.state_dir / "throughput.json", report)
            self.emit(f"BENCHMARK REPORT {self.state_dir / 'throughput.json'}")
        return 0


def worker(args):
    if not args.skip_preflight:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/experiments/preflight_final_v2.py"), "--config", args.config],
            cwd=ROOT,
            check=True,
        )
    import jax

    if jax.device_count() != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError(f"Expected exactly one visible GPU, got {jax.devices()}")
    print(f"[WORKER] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} jax.device_count=1", flush=True)
    sys.argv = [
        str(ROOT / "scripts/train.py"),
        args.config,
        "--exp-name",
        args.exp_name,
        "--checkpoint-base-dir",
        args.checkpoint_base_dir,
        "--log-interval",
        "100",
    ]
    if args.resume:
        sys.argv.append("--resume")
    if args.benchmark_steps is not None:
        sys.argv += ["--num-train-steps", str(args.benchmark_steps)]
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--max-parallel", type=int, choices=range(1, 9), default=4)
    run.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--state-dir", type=Path)
    run.add_argument("--exp-name", default="final_v2")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--benchmark", action="store_true")
    run.add_argument("--benchmark-steps", type=int, default=400)
    run.add_argument("--warmup-steps", type=int, default=100)
    for action in ("status", "stop"):
        command = sub.add_parser(action)
        command.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    child = sub.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--config", choices=PHASE_A + PHASE_B, required=True)
    child.add_argument("--exp-name", required=True)
    child.add_argument("--checkpoint-base-dir", required=True)
    child.add_argument("--resume", action="store_true")
    child.add_argument("--skip-preflight", action="store_true")
    child.add_argument("--benchmark-steps", type=int)
    return p


def main():
    args = parser().parse_args()
    if args.action == "worker":
        worker(args)
        return 0
    if args.action == "status":
        print((args.state_dir / "state.json").read_text())
        return 0
    if args.action == "stop":
        send_stop(args.state_dir)
        return 0
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
        if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
            raise ValueError("--gpus must be distinct nonnegative GPU indices")
        if args.max_parallel > len(gpus) or args.threads < 1:
            raise ValueError("max-parallel must not exceed GPU count; threads must be positive")
        if args.exp_name in {"", ".", ".."} or Path(args.exp_name).name != args.exp_name:
            raise ValueError("--exp-name must be a single directory name")
        if args.benchmark and (args.benchmark_steps < args.warmup_steps + 201 or args.warmup_steps < 100):
            raise ValueError("benchmark needs warmup >=100 and at least two later 100-step log points")
        if args.benchmark:
            run_id = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            state_dir = (args.state_dir or ROOT / "output/final_v2_benchmarks" / run_id).resolve()
            checkpoint_base = state_dir / "checkpoints"
            if state_dir == (ROOT / "checkpoints").resolve() or (ROOT / "checkpoints").resolve() in state_dir.parents:
                raise ValueError("Benchmark state must be outside the formal checkpoint directory")
            if state_dir.exists() and any(state_dir.iterdir()):
                raise ValueError("Benchmark state directory must be new or empty")
        else:
            state_dir = (args.state_dir or DEFAULT_STATE).resolve()
            checkpoint_base = (ROOT / "checkpoints").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as locks:
            lock_dir = ROOT / "output/final_v2_runner/locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            experiment_key = hashlib.sha256(f"{checkpoint_base}/{args.exp_name}".encode()).hexdigest()
            paths = [state_dir / "runner.lock", lock_dir / f"experiment-{experiment_key}.lock"]
            paths += [lock_dir / f"gpu-{gpu}.lock" for gpu in sorted(gpus)]
            for path in paths:
                lock = locks.enter_context(path.open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"Another runner holds {path}") from exc
            queue = Queue(
                state_dir=state_dir,
                checkpoint_base=checkpoint_base,
                exp_name=args.exp_name,
                gpus=gpus,
                max_parallel=args.max_parallel,
                threads=args.threads,
                skip_preflight=args.skip_preflight,
                benchmark=args.benchmark,
                benchmark_steps=args.benchmark_steps,
                warmup_steps=args.warmup_steps,
                final_step=args.benchmark_steps - 1 if args.benchmark else 19999,
                retry_failed=args.retry_failed,
            )
            atomic_json(
                state_dir / "runner.json", {"pid": os.getpid(), "proc_start": proc_start(os.getpid()), "started": now()}
            )
            signal.signal(signal.SIGTERM, queue.request_stop)
            signal.signal(signal.SIGINT, queue.request_stop)
            queue.emit(f"RUN state={state_dir} gpus={gpus} max_parallel={args.max_parallel} threads={args.threads}")
            return queue.run()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
