#!/usr/bin/env python3
"""Parallel, GPU-aware queue for the EXISTING RGBTrack batch CLI.

No tracking algorithms, output formats or accepted pose files are changed.
Usage:
  python scripts/preprocess/rgbtrack_parallel.py \
    --job 1028_press_button:button:box:press_button_v1 --workers 7
  python scripts/preprocess/rgbtrack_parallel.py \
    --job 1028_press_button:button:box:press_button_v1 \
    --job 1030_yellow_socket:red:yellow:yellow_socket_v1 --workers 7

Each GPU runs ONE episode at a time, then draws the next one from a shared queue.
Existing verified episodes are skipped by the official rgbtrack_task.py batch command.
Do NOT run another scheduler or serial batch for the same (dataset, run tag).
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
from typing import Optional


IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class Job:
    dataset: str
    dynamic: str
    static: str
    tag: str


def parse_job(value: str) -> Job:
    parts = value.split(":")
    if len(parts) != 4 or any(not IDENT.fullmatch(x) for x in parts):
        raise ValueError("--job format must be DATASET:DYNAMIC:STATIC:RUN_TAG (letters/digits/_/- only)")
    task = Job(*parts)
    if task.dynamic == task.static:
        raise ValueError(f"{value}: static and dynamic objects must differ")
    return task


def gpu_stats() -> list[tuple[int, int, int]]:
    """Return (physical index, GPU utilization %, GPU memory in use MiB)."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    )
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"Could not read nvidia-smi row: {line!r}")
        rows.append(tuple(int(x) for x in fields))
    return rows


def choose_gpus(spec: str, *, workers: int, util_limit: int, mem_limit: int) -> list[int]:
    if spec != "auto":
        result = [int(x) for x in spec.split(",")]
        if len(result) != len(set(result)) or any(x < 0 for x in result):
            raise ValueError("--gpus must be unique nonnegative GPU indices")
        return result[:workers]
    available = []
    observed = gpu_stats()
    for idx, util, used in observed:
        status = "IDLE" if util <= util_limit and used <= mem_limit else "BUSY"
        print(f"[GPU] {idx}: util={util}% used={used} MiB {status}", flush=True)
        if status == "IDLE":
            available.append(idx)
    return available[:workers]


def read_episode_ids(repo: Path, job: Job) -> list[int]:
    manifest = repo / "output/preprocess" / job.dataset / "sam2/manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    objects = set(raw.get("objects", []))
    if objects != {job.dynamic, job.static}:
        raise ValueError(f"{job.dataset}: SAM2 objects={sorted(objects)} do not match "
                         f"--job static/dynamic={job.static},{job.dynamic}")
    ids = [int(x) for x in raw["episode_ids"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"{job.dataset}: invalid episode IDs")
    return sorted(ids)


def round_robin_tasks(repo: Path, jobs: list[Job]) -> list[tuple[Job, int]]:
    pending = [deque(read_episode_ids(repo, x)) for x in jobs]
    result: list[tuple[Job, int]] = []
    while any(pending):
        for job, episodes in zip(jobs, pending):
            if episodes:
                result.append((job, episodes.popleft()))
    return result


def prepare_scale_manifest(repo: Path, job: Job) -> None:
    """Avoid a first-run race from several processes creating the same scale file."""
    p = repo / "output/preprocess" / job.dataset / "rgbtrack/scale_manifest.json"
    expected = {"schema_version": 1, "objects": {job.dynamic: 1.0, job.static: 1.0}}
    if p.exists():
        actual = json.loads(p.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Scale manifest differs from approved metre meshes: {p}")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def other_tracker_processes(jobs: list[Job]) -> list[str]:
    """Best effort guard against a legacy serial run for one of these datasets."""
    procroot = Path("/proc")
    if not procroot.is_dir():
        return []
    names = ("rgbtrack_task.py", "rgbtrack_task_runner.py", "rgbtrack_adapter.py",
             "rgbtrack_parallel.py")
    conflicts = []
    mine = os.getpid()
    for entry in procroot.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == mine:
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\x00")
        except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
            continue
        # Require an actual script argv element, not a shell string mentioning it.
        if not any(Path(a).name in names for a in args if a):
            continue
        line = " ".join(a for a in args if a)
        if any(j.dataset in line for j in jobs):
            conflicts.append(f"pid={entry.name} {line[:350]}")
    return conflicts


def quarantine_partial_outputs(repo: Path, jobs: list[Job], timestamp: str) -> int:
    """Move only unfinished per-episode directories to backup, never delete.

    A missing task_input_provenance.json indicates the official batch command
    had not committed completion. Directories with provenance are left for the
    original batch CLI to validate (including changed-source failures).
    Only call after proving no other RGBTrack worker is active for these tasks.
    """
    moved = 0
    for job in jobs:
        episodes = read_episode_ids(repo, job)
        base = repo / "output/preprocess" / job.dataset / "rgbtrack"
        source = base / "runs" / job.tag / "episodes"
        backup = base / ("interrupted_scheduler_" + timestamp) / job.tag
        for ep in episodes:
            current = source / f"e{ep:05d}"
            if not current.exists() and not current.is_symlink():
                continue
            if current.is_symlink() or not current.is_dir():
                raise ValueError(f"Unexpected episode output path: {current}")
            if (current / "task_input_provenance.json").is_file():
                continue
            backup.mkdir(parents=True, exist_ok=True)
            dest = backup / current.name
            if dest.exists():
                raise FileExistsError(f"Backup collision: {dest}")
            shutil.move(str(current), str(dest))
            moved += 1
            print(f"[BACKUP PARTIAL] {current} -> {dest}", flush=True)
    return moved


def build_child_command(repo: Path, job: Job, ep: int, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(repo / "scripts/preprocess/rgbtrack_task.py"),
        "--dataset", job.dataset, "batch", "--episodes", str(ep),
        "--dynamic-object", job.dynamic, "--static-object", job.static,
        "--dynamic-init-up-axis", "+z",
        "--dynamic-init-max-tilt-deg", str(args.init_max_tilt),
        "--guard-max-step-cm", str(args.step_cm),
        "--guard-max-turn-deg", str(args.turn_deg),
        "--guard-recovery-max-turn-deg", str(args.recovery_deg),
        "--run-tag", job.tag,
    ]


def make_child_env(repo: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["REPO"] = str(repo)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    env["MAX_JOBS"] = "2"
    return env


def run_parallel(repo: Path, jobs: list[Job], tasks: list[tuple[Job, int]],
                 gpus: list[int], args: argparse.Namespace, logdir: Path) -> int:
    q = deque(tasks)
    mu = threading.Lock()
    stop = threading.Event()
    results = {"ok": 0, "failed": 0}
    failures: list[str] = []

    def on_signal(signum, _frame):
        if not stop.is_set():
            print(f"\n[STOP REQUESTED] signal={signum}; finish active episodes, launch no new ones", flush=True)
            stop.set()
        else:
            print("[STOP REQUESTED] Current episodes still running. To abort, terminate their process groups explicitly.", flush=True)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

    def worker(gpu: int):
        while not stop.is_set():
            with mu:
                if not q or stop.is_set():
                    break
                job, ep = q.popleft()
            logfile = logdir / f"{job.dataset}_{job.tag}_e{ep:05d}_gpu{gpu}.log"
            cmd = build_child_command(repo, job, ep, args)
            print(f"[START GPU{gpu}] {job.dataset} e{ep:05d} -> {logfile}", flush=True)
            try:
                with logfile.open("w", encoding="utf-8") as stream:
                    stream.write("$ " + " ".join(cmd) + "\n")
                    stream.flush()
                    status = subprocess.run(cmd, cwd=repo, env=make_child_env(repo, gpu),
                                            stdout=stream, stderr=subprocess.STDOUT).returncode
            except Exception as e:
                status = -1
                with logfile.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n[LAUNCH ERROR] {type(e).__name__}: {e}\n")
            with mu:
                if status == 0:
                    results["ok"] += 1
                    print(f"[DONE GPU{gpu}] {job.dataset} e{ep:05d} ({results['ok']}/{len(tasks)})", flush=True)
                else:
                    results["failed"] += 1
                    failures.append(f"{job.dataset} e{ep:05d} GPU{gpu} return={status} log={logfile}")
                    print(f"[FAIL GPU{gpu}] {job.dataset} e{ep:05d}; see {logfile}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    summary = {"jobs": [j.__dict__ for j in jobs], "gpus": gpus,
               "episodes_queued": len(tasks), "commands_ok": results["ok"],
               "failed": results["failed"], "remaining": len(q),
               "stop_requested": stop.is_set(), "failures": failures}
    (logdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[SUMMARY] " + json.dumps({k: v for k, v in summary.items() if k not in ("jobs", "failures")}), flush=True)
    for f in failures[:30]:
        print("[FAIL]", f, flush=True)
    return 0 if not failures and not stop.is_set() else 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path(os.environ.get("REPO", "/workspace/mnt/sqzhang26/openpi-3d-tactile")))
    p.add_argument("--job", action="append", required=True,
                   help="DATASET:DYNAMIC_OBJECT:STATIC_OBJECT:RUN_TAG; repeat for multiple datasets")
    p.add_argument("--gpus", default="auto", help="auto or comma-separated physical GPU indices")
    p.add_argument("--workers", type=int, default=7, help="Maximum simultaneous GPUs (default: 7)")
    p.add_argument("--idle-util", type=int, default=15, help="Max utilization percentage for idle GPU")
    p.add_argument("--idle-mem-mib", type=int, default=1536, help="Max used MiB for idle GPU")
    p.add_argument("--step-cm", type=float, default=2.0)
    p.add_argument("--turn-deg", type=float, default=25.0)
    p.add_argument("--recovery-deg", type=float, default=60.0)
    p.add_argument("--init-max-tilt", type=float, default=25.0)
    p.add_argument("--dry-run", action="store_true", help="Only verify inputs, print jobs and GPUs; do not run RGBTrack")
    a = p.parse_args(argv)
    if a.workers < 1 or not 0 < a.idle_util <= 100 or a.idle_mem_mib < 0:
        p.error("Invalid --workers / --idle-util / --idle-mem-mib")
    if min(a.step_cm, a.turn_deg, a.recovery_deg, a.init_max_tilt) <= 0:
        p.error("Motion thresholds must be positive")
    repo = a.repo.expanduser().resolve()
    script = repo / "scripts/preprocess/rgbtrack_task.py"
    if not script.is_file():
        p.error(f"Missing official RGBTrack script: {script}")
    jobs = [parse_job(x) for x in a.job]
    if len({(x.dataset, x.tag) for x in jobs}) != len(jobs):
        p.error("Duplicate (dataset, run-tag) among --job entries")

    # Reuse official 'check' validation; NO GPU work in this phase.
    for j in jobs:
        print(f"[CHECK] {j.dataset} {j.dynamic}/{j.static} tag={j.tag}", flush=True)
        subprocess.run([sys.executable, str(script), "--dataset", j.dataset, "check", "--episodes", "all"],
                       cwd=repo, env=make_child_env(repo, 0), check=True)
    tasks = round_robin_tasks(repo, jobs)
    gpus = choose_gpus(a.gpus, workers=a.workers,
                       util_limit=a.idle_util, mem_limit=a.idle_mem_mib)
    if not gpus:
        print("[NO IDLE GPU] No workers launched. Recheck nvidia-smi or use --gpus explicitly.", flush=True)
        return 2
    print(f"[PLAN] datasets={len(jobs)} episodes={len(tasks)} GPUs={gpus} (one episode per GPU at a time)", flush=True)
    if a.dry_run:
        for j in jobs:
            print(f"[JOB] {j.dataset}: {len(read_episode_ids(repo, j))} episode(s), tag={j.tag}")
        print("[DRY RUN] no inference or output changes", flush=True)
        return 0

    # Ensure no legacy serial run or second scheduler writes to the same episode.
    conflicts = other_tracker_processes(jobs)
    if conflicts:
        print("[BLOCKED] Existing tracking processes for same dataset detected:", flush=True)
        for line in conflicts[:20]:
            print("  " + line, flush=True)
        print("Stop those processes only after their current episode finishes; do not run concurrent writers.", flush=True)
        return 3

    # Locks prevent TWO parallel schedulers for the same dataset and run-tag.
    locks = []
    try:
        for j in sorted(jobs, key=lambda x: (x.dataset, x.tag)):
            root = repo / "output/preprocess" / j.dataset / "rgbtrack/runs" / j.tag
            root.mkdir(parents=True, exist_ok=True)
            lockpath = root / ".parallel_scheduler.lock"
            lock = lockpath.open("a+")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError(f"Another parallel scheduler holds {lockpath}")
            locks.append(lock)
        for j in jobs:
            prepare_scale_manifest(repo, j)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine_partial_outputs(repo, jobs, stamp + f"_pid{os.getpid()}")
        logdir = repo / "output/preprocess/rgbtrack_parallel_logs" / (stamp + f"_pid{os.getpid()}")
        logdir.mkdir(parents=True, exist_ok=False)
        print(f"[LOGS] {logdir}", flush=True)
        return run_parallel(repo, jobs, tasks, gpus, a, logdir)
    finally:
        for lock in locks:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
