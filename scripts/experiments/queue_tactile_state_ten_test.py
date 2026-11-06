"""No GPUs or training jobs are used by these scheduler guard tests."""
import json

import pytest
import queue_tactile_state_ten as queue


def test_run_requires_passing_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "STATE", tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        queue.run()
    (tmp_path / "preflight_report.json").write_text(json.dumps({"status": "blocked", "revision": queue.source_revision()}))
    with pytest.raises(RuntimeError, match="blocked or stale"):
        queue.run()
    (tmp_path / "preflight_report.json").write_text(json.dumps({
        "status": "passed", "revision": queue.source_revision(),
        "pi05_tactile_max_token_len_env": None, "legacy_regression": {"status": "passed"}, "jobs": [],
    }))
    monkeypatch.delenv("OPENPI_PI05_TACTILE_MAX_TOKEN_LEN", raising=False)
    with pytest.raises(RuntimeError, match="all ten configurations"):
        queue.run()


def test_checkpoint_guard_never_overwrites_nonempty_directory(tmp_path):
    path = tmp_path / "job"
    assert queue.classify(path)[0] == "fresh"
    path.mkdir()
    assert queue.classify(path)[0] == "fresh"
    (path / "unrecognized").write_text("keep")
    assert queue.classify(path)[0] == "invalid"
    assert (path / "unrecognized").read_text() == "keep"


def test_gpu_telemetry_failure_never_marks_gpu_free(monkeypatch):
    def fail(_):
        raise RuntimeError("telemetry unavailable")
    monkeypatch.setattr(queue, "gpu_processes", fail)
    with pytest.raises(RuntimeError, match="telemetry"):
        queue.gpu_available()
