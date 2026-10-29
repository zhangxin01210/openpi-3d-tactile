"""CPU-only regression tests for exact-frame Pi0 action comparisons."""
import json

import numpy as np
import pytest

from scripts.deploy.compare_saved_action_runs import (
    _read_runs,
    _measured_joint_positions,
    analyze,
)


def test_identical_input_first_actions_and_pairwise_differences(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    state = np.zeros(1972, dtype=np.float32)
    state[:6] = 0.5
    state[28:40] = 1.0
    np.savez_compressed(frames_dir / "frame_000016.npz", robot_state=state, frame_index=16)
    models = {}
    for name, offset in (("M2", 0.1), ("M2_OC", 0.2)):
        root = tmp_path / name
        root.mkdir()
        (root / "summary.json").write_text(json.dumps({"status": "pass", "dry_run": True}))
        actions = np.tile(
            np.concatenate([state[:6], state[28:40]]) + offset,
            (50, 1),
        )
        np.savez_compressed(root / "actions_000016.npz", actions=actions)
        models[name] = root
    runs = _read_runs([f"{name}={root}" for name, root in models.items()], 16)
    joint, pair, summary = analyze(runs, frames_dir, 16)
    assert len(joint) == 18 * 2
    assert len(pair) == 1
    np.testing.assert_allclose(_measured_joint_positions(frames_dir, 16)[:6], 0.5)
    assert abs(pair[0]["first_arm6_mae_rad"] - 0.1) < 1e-6
    assert abs(pair[0]["first_hand12_mae_rad"] - 0.1) < 1e-6
    assert pair[0]["sign_agreement_fraction"] == 1.0
    assert summary["models"]["M2"]["arm_first_abs_delta_mean_rad"] > 0
    assert summary["pairs"]["M2__vs__M2_OC"]["worst_frame"] == 16


def test_rejects_mismatched_frame_sets(tmp_path):
    roots = []
    for name, frame in (("a", 0), ("b", 16)):
        root = tmp_path / name
        root.mkdir()
        (root / "summary.json").write_text(json.dumps({"status": "pass", "dry_run": True}))
        np.savez_compressed(root / f"actions_{frame:06d}.npz", actions=np.zeros((50, 18)))
        roots.append(f"{name}={root}")
    with pytest.raises(ValueError, match="frame sets differ"):
        _read_runs(roots, 16)
