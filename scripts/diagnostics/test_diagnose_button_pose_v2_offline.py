"""Small synthetic checks for the read-only diagnostic's inference contracts."""

import numpy as np
import pytest
import torch

from scripts.diagnostics import diagnose_button_pose_v2_offline as diag


def test_proxy_and_no_contact_are_not_physical_labels():
    assert diag.event_proxy(np.zeros(30)) is None
    force = np.zeros(30)
    force[10:15] = 1
    event = diag.event_proxy(force)
    assert event["onset"] == 10
    assert event["end"] == 14
    assert [diag.phase(i, event) for i in (9, 10, 15)] == ["before", "during_proxy", "after"]
    assert diag.phase(3, None) == "unclassified"


def test_episode_selection_reproducible_and_covers_splits():
    splits = {"train": list(range(7)), "val": [7, 8], "test": [9, 10]}
    selected = diag.select_episodes(splits, range(11), 8, 42)
    assert selected == diag.select_episodes(splits, range(11), 8, 42)
    assert {s for _, s in selected} == set(splits)
    assert len({ep for ep, _ in selected}) == 8


def test_paired_batch_two_uses_same_rgb_depth_and_ordered_cad():
    class Dummy(torch.nn.Module):
        def forward(self, rgb, depth, cad):
            assert rgb.shape == (2, 3, 4, 5)
            assert depth.shape == (2, 1, 4, 5)
            assert torch.equal(rgb[0], rgb[1])
            assert torch.equal(depth[0], depth[1])
            assert torch.allclose(depth, torch.full_like(depth, 0.5))
            return cad[:, 0, :], torch.eye(3).repeat(2, 1, 1)

    cad = np.stack([np.ones((512, 3)), np.full((512, 3), 2)]).astype(np.float32)
    rgb = np.full((4, 5, 3), 255, dtype=np.uint8)
    depth = np.full((4, 5), 1000, dtype=np.uint16)
    xyz, t, rot = diag.predict_two(Dummy(), {"depth_scale": 0.001}, cad, rgb, depth, torch.device("cpu"))
    np.testing.assert_array_equal(t[:, 0], [1, 2])
    np.testing.assert_array_equal(xyz[:, 0, 0], [2, 4])
    assert diag.rotation_difference(rot[0], rot[1]) == pytest.approx(0)


def test_stats_and_missing_checkpoints(tmp_path):
    report = diag.stats([1, 6, 11, 21, np.nan])
    assert report["count"] == 4
    assert report["over_10mm_fraction"] == 0.5
    with pytest.raises(FileNotFoundError):
        diag.checkpoint_scan(tmp_path)
    for step in (5000, 20000):
        for name in ("assets", "params"):
            (tmp_path / "seed42" / str(step) / name).mkdir(parents=True)
    checkpoint, _ = diag.checkpoint_scan(tmp_path)
    assert checkpoint.name == "20000"


def test_training_input_excludes_actual_and_future_actions():
    original = {
        "spatial": {"object_geometry": {"xyz": np.ones(3)}},
        "action": np.full((50, 18), 123.0),
        "state": np.arange(5),
    }
    captured = {}

    def chain(item):
        captured.update(item)
        return {"state": item["state"], "actions": item["action"]}

    model_input = diag.make_training_input(original, chain, 50)
    assert np.all(captured["action"] == 0)
    assert "actions" not in model_input
    assert model_input["spatial"] is original["spatial"]
    assert np.all(original["action"] == 123.0)


def test_absolute_arm_targets_reuse_verified_fk():
    from openpi.spatial.kinematics import RobotKinematics
    from openpi.spatial.kinematics import load_state_mapping_csv
    from scripts.deploy.tactile_motion_probe import _fk_intent

    fk = RobotKinematics(diag.ROOT / "configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    mapping = load_state_mapping_csv(diag.ROOT / "configs/ur7e_xhand/state_mapping.csv")
    state = np.zeros(1972, dtype=np.float32)
    hold = np.zeros((1, 18), dtype=np.float32)
    intent = _fk_intent(fk, mapping, state, hold, np.zeros((512, 3)))
    assert intent["x_mm"][0] == pytest.approx(0)
    assert intent["y_mm"][0] == pytest.approx(0)
    assert intent["z_mm"][0] == pytest.approx(0)
    move = hold.copy()
    move[0, 0] = 0.1
    changed = _fk_intent(fk, mapping, state, move, np.zeros((512, 3)))
    assert np.hypot(changed["x_mm"][0], changed["y_mm"][0]) > 1
