import json
import subprocess
import sys

import pytest
import queue_final_v2 as runner


def checkpoint(base, name, step, exp="final_v2"):
    path = base / name / exp / str(step)
    path.mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
    for item in ("params", "train_state"):
        directory = path / item
        directory.mkdir()
        (directory / "_METADATA").write_text('{"tree_metadata": {"x": {}}}')
        (directory / "manifest.ocdbt").write_text("manifest")


DUMMY = r"""
import json, os, pathlib, sys, time
name, root, event_path, duration, resume, fail = sys.argv[1:]
def log(kind):
    event = json.dumps({"name": name, "event": kind, "time": time.monotonic(),
                        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "resume": resume,
                        "threads": os.environ["OMP_NUM_THREADS"], "wandb": os.environ["WANDB_MODE"]}) + "\n"
    fd = os.open(event_path, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
    os.write(fd, event.encode()); os.close(fd)
log("start")
time.sleep(float(duration))
if fail == "yes":
    log("end"); sys.exit(3)
p = pathlib.Path(root) / name / "final_v2" / "19"
p.mkdir(parents=True, exist_ok=True)
(p / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
for item in ("params", "train_state"):
    d = p / item; d.mkdir(exist_ok=True)
    (d / "_METADATA").write_text('{"tree_metadata": {"x": {}}}')
    (d / "manifest.ocdbt").write_text("manifest")
log("end")
"""


def make_queue(tmp_path, phases, durations=None, failures=(), gpu_check=lambda _: {}):
    base, state = tmp_path / "checkpoints", tmp_path / "state"
    events = tmp_path / "events.jsonl"
    script = tmp_path / "dummy.py"
    script.write_text(DUMMY)
    durations = durations or {}

    def command(name, resume, gpu):
        return [
            sys.executable,
            str(script),
            name,
            str(base),
            str(events),
            str(durations.get(name, 0.05)),
            str(resume),
            "yes" if name in failures else "no",
        ]

    queue = runner.Queue(
        state_dir=state,
        checkpoint_base=base,
        exp_name="final_v2",
        gpus=[0, 1],
        max_parallel=2,
        threads=2,
        phases=phases,
        final_step=19,
        command_factory=command,
        gpu_check=gpu_check,
        poll_seconds=0.01,
    )
    return queue, base, events


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_dynamic_exclusive_parallel_and_phase_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    queue, _, path = make_queue(tmp_path, (["a", "b", "c", "d"], ["e", "f"]), {"a": 0.55, "b": 0.1, "c": 0.1, "d": 0.1})
    assert queue.run() == 0
    events = sorted(read_events(path), key=lambda x: x["time"])
    active = {}
    for event in events:
        if event["event"] == "start":
            assert event["gpu"] not in active
            active[event["gpu"]] = event["name"]
            assert len(active) <= 2
            assert event["threads"] == "2"
            assert event["wandb"] == "offline"
        else:
            assert active.pop(event["gpu"]) == event["name"]
    starts = {e["name"]: e["time"] for e in events if e["event"] == "start"}
    ends = {e["name"]: e["time"] for e in events if e["event"] == "end"}
    assert starts["c"] < ends["a"]
    assert min(starts[n] for n in ("e", "f")) >= max(ends[n] for n in ("a", "b", "c", "d"))
    assert all(item["status"] == "completed" for item in queue.state["jobs"].values())


def test_failure_blocks_second_phase(tmp_path):
    queue, _, path = make_queue(tmp_path, (["a", "b"], ["c"]), failures=("a",))
    assert queue.run() == 1
    assert queue.state["jobs"]["a"]["status"] == "failed"
    assert queue.state["jobs"]["a"]["exit_code"] == 3
    assert "c" not in {e["name"] for e in read_events(path)}


def test_complete_skipped_resume_and_empty_directory(tmp_path):
    queue, base, path = make_queue(tmp_path, (["done", "resume", "empty"], []))
    checkpoint(base, "done", 19)
    checkpoint(base, "resume", 5)
    (base / "empty" / "final_v2").mkdir(parents=True)
    assert queue.run() == 0
    starts = {e["name"]: e for e in read_events(path) if e["event"] == "start"}
    assert "done" not in starts
    assert queue.state["jobs"]["done"]["skipped"]
    assert starts["resume"]["resume"] == "True"
    assert starts["empty"]["resume"] == "True"
    assert runner.checkpoint_state(base / "empty" / "final_v2", 19) == ("complete", 19)


def test_invalid_checkpoint_and_busy_gpu_fail_closed(tmp_path):
    queue, base, path = make_queue(tmp_path, (["broken"], ["later"]))
    (base / "broken" / "final_v2" / "5").mkdir(parents=True)
    assert queue.run() == 1
    assert queue.state["jobs"]["broken"]["status"] == "failed"
    assert not path.exists()
    other = tmp_path / "other"
    other.mkdir()
    queue, _, path = make_queue(other, (["a"], []), gpu_check=lambda _: {0: [123]})
    with pytest.raises(RuntimeError, match="running compute"):
        queue.run()
    assert queue.state["status"] == "failed"
    assert not path.exists()


def test_failed_final_checkpoint_requires_explicit_retry(tmp_path):
    queue, base, _ = make_queue(tmp_path, (["a"], []))
    checkpoint(base, "a", 19)
    queue.state["jobs"]["a"] = {"status": "failed"}
    assert queue.classify("a")[0] == "failed"
    queue.retry_failed = True
    assert queue.classify("a")[0] == "resume"


def test_unexpected_gpu_check_error_stops_only_own_children(tmp_path):
    calls = 0

    def check(_):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("GPU query failed")
        return {}

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        queue, _, _ = make_queue(tmp_path, (["a", "b"], []), {"a": 5}, gpu_check=check)
        with pytest.raises(RuntimeError, match="GPU query failed"):
            queue.run()
        assert not queue.running
        assert queue.state["status"] == "failed"
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait()


def test_throughput_anchors_midnight_and_utilization(tmp_path):
    jobs = {}
    for name, start, lines in (
        ("a", "2026-01-01T23:59:00+00:00", "23:59:50.000 [I] Step 100:\n00:00:10.000 [I] Step 200:\n"),
        ("b", "2026-01-02T00:00:00+00:00", "00:00:00.000 [I] Step 100:\n00:00:20.000 [I] Step 200:\n"),
    ):
        path = tmp_path / f"{name}.log"
        path.write_text(lines)
        jobs[name] = {"log": str(path), "started": start, "status": "completed"}
    (tmp_path / "utilization.jsonl").write_text(
        json.dumps(
            {
                "time": "2026-01-02T00:00:05+00:00",
                "cpu_percent": 50,
                "gpus": {"0": {"utilization_percent": 80}, "1": {"utilization_percent": 60}},
                "running": {"0": "a", "1": "b"},
            }
        )
        + "\n"
    )
    report = runner.stable_rates({"jobs": jobs, "state_dir": str(tmp_path)}, 100)
    assert report["models"]["a"]["steps_per_second"] == 5
    assert report["total_steps_per_second"] == pytest.approx(200 / 30)
    assert report["models"]["a"]["gpu_percent_mean"] == 80


def test_real_orbax_checkpoint_zero_resume(tmp_path):
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    from openpi.training.checkpoints import initialize_checkpoint_dir

    directory = tmp_path / "ckpt"
    manager = ocp.CheckpointManager(directory, item_names=("params", "train_state"))
    manager.save(
        0,
        args=ocp.args.Composite(
            params=ocp.args.PyTreeSave({"weights": jnp.array([1])}),
            train_state=ocp.args.PyTreeSave({"step": jnp.array(1)}),
        ),
    )
    manager.wait_until_finished()
    manager.close()
    assert runner.checkpoint_state(directory, 19) == ("resume", 0)
    manager, resuming = initialize_checkpoint_dir(directory, keep_period=5000, overwrite=False, resume=True)
    try:
        assert resuming
    finally:
        manager.close()


def test_stop_checks_process_identity(tmp_path):
    script = tmp_path / "queue_final_v2.py"
    script.write_text("import time\ntime.sleep(30)\n")
    process = subprocess.Popen([sys.executable, str(script), "run"])
    try:
        info = {"pid": process.pid, "proc_start": runner.proc_start(process.pid)}
        (tmp_path / "runner.json").write_text(json.dumps(dict(info, proc_start="wrong")))
        with pytest.raises(RuntimeError, match="stale"):
            runner.send_stop(tmp_path)
        assert process.poll() is None
        (tmp_path / "runner.json").write_text(json.dumps(info))
        runner.send_stop(tmp_path)
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
