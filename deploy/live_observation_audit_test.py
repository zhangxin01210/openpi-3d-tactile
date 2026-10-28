"""No hardware: read-only robot observation audit never creates valid native-depth packets."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ur7e_xhand_deploy_3d_tactile_multi as client


class MockArm:
    def get_observation(self):
        # Hint only: this is NOT proof of 6D wrench units/freshness.
        return {"actual_TCP_force_candidate": np.zeros(6, dtype=float)}


class MockRobot:
    def __init__(self, obs):
        self.arm=MockArm()
        self.obs=obs
        self.reads=0
    def get_observation(self):
        self.reads += 1
        return self.obs
    def send_action(self, *_args, **_kwargs):
        raise AssertionError("AUDIT MUST NEVER COMMAND ROBOT")
    def reset_to_home(self):
        raise AssertionError("AUDIT MUST NEVER HOME ROBOT")


def _input():
    names=[f"state_{i}" for i in range(1972)]
    observations={name: 0.0 for name in names}
    observations.update({
        "cam_front": np.zeros((12,16,3),dtype=np.uint8),
        "cam_left": np.zeros((12,16,3),dtype=np.uint8),
        "cam_right": np.zeros((12,16,3),dtype=np.uint8),
        "depths.cam_front": np.ones((12,16),dtype=np.uint16)*1000,
    })
    return observations,names


def test_live_audit_is_readonly_and_never_claims_depth_coordinate_frame(tmp_path,monkeypatch):
    observation,names=_input()
    robot=MockRobot(observation)
    monkeypatch.setattr(client.time,"sleep",lambda _: None)
    args=SimpleNamespace(capture_observation_audit=tmp_path/"audit",
                         capture_audit_frames=2,realsense_height=12,
                         realsense_width=16,fps=15)
    client.capture_observation_audit(robot,args,names)
    assert robot.reads==2
    report=json.loads((args.capture_observation_audit/"observation_schema.json").read_text())
    assert report["no_actions_sent"] is True
    assert report["pi0_server_used"] is False
    assert report["frames"]==2
    assert report["depth_provenance"].startswith("UNVERIFIED")
    assert "actual_TCP_force_candidate" in report["records"][0]["torque_or_wrench_key_candidates_NOT_VERIFIED"]
    with np.load(args.capture_observation_audit/"audit_0000.npz",allow_pickle=False) as arrays:
        assert arrays["robot_state"].shape==(1972,)
        assert "front_depth_coordinate_system_UNVERIFIED" in arrays.files
        assert "front_depth_native" not in arrays.files
        assert "front_depth_color_aligned" not in arrays.files
        assert "front_rgb" in arrays.files


def test_live_audit_missing_depth_is_explicit_not_synthesized(tmp_path):
    observation,names=_input()
    observation.pop("depths.cam_front")
    robot=MockRobot(observation)
    args=SimpleNamespace(capture_observation_audit=tmp_path/"audit",
                         capture_audit_frames=1,realsense_height=12,
                         realsense_width=16,fps=15)
    client.capture_observation_audit(robot,args,names)
    report=json.loads((args.capture_observation_audit/"observation_schema.json").read_text())
    assert report["records"][0]["diagnostics"]["front_depth"]["status"]=="missing_or_invalid"


def test_live_audit_does_not_overwrite(tmp_path):
    observation,names=_input()
    path=tmp_path/"audit"
    path.mkdir()
    args=SimpleNamespace(capture_observation_audit=path,
                         capture_audit_frames=1,realsense_height=12,
                         realsense_width=16,fps=15)
    with pytest.raises(FileExistsError):
        client.capture_observation_audit(MockRobot(observation),args,names)
