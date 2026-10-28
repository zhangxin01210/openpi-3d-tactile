"""No-GPU tests: robot-side scene 1..6 switches SAME server's causal worker only."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deploy.websocket_hand_scene_bridge import (
    SELECTABLE_SCENE_POSITIONS, HandScenePerceptionBridge, validate_history,
)


def frame(i):
    return {
        "front_rgb":np.zeros((480,640,3),dtype=np.uint8),
        "left_rgb":np.zeros((480,640,3),dtype=np.uint8),
        "right_rgb":np.zeros((480,640,3),dtype=np.uint8),
        "front_depth_native":np.ones((480,640),dtype=np.uint16),
        "robot_state":np.zeros(1972,dtype=np.float32),
        "timestamp_s":np.asarray(100.+i/15.),
        "frame_index":np.asarray(i),
    }


def observation(session,position,history):
    last=history[-1]
    return {
        "hand_scene_session":session,"hand_scene_position":position,
        "hand_scene_history":history,"observation.state":last["robot_state"],
        "observation.depths.cam_front":last["front_depth_native"],
        **{f"observation.images.cam_{name}":last[f"{name}_rgb"]
           for name in ("front","left","right")},
    }


def test_history_is_contiguous_and_refuses_stale_or_invalid_inputs():
    seq=[frame(i) for i in range(16)]
    assert len(validate_history(seq,next_index=0,previous_timestamp=float("-inf")))==16
    with pytest.raises(ValueError,match="discontinuity"):
        validate_history(seq,next_index=1,previous_timestamp=float("-inf"))
    with pytest.raises(ValueError,match="timestamp"):
        validate_history(seq,next_index=0,previous_timestamp=100.01)
    bad=frame(0)
    bad.pop("left_rgb")
    with pytest.raises(KeyError):
        validate_history([bad],next_index=0,previous_timestamp=float("-inf"))


def setup_bridge(tmp_path,monkeypatch):
    class Policy:
        metadata={"spatial_camera_roles":["front"]}
        def __init__(self):
            self.received=[]
            self.input_transform=SimpleNamespace(reset_tactile_ema=lambda:None)
        def infer(self,obs):
            self.received.append(obs)
            return {"actions":np.zeros((50,18),dtype=np.float32)}
    pi0=Policy()
    py=tmp_path/"python";py.write_text("stub executable")
    bridge=HandScenePerceptionBridge(
        pi0,repo_root=tmp_path,perception_python=py,
        workdir=tmp_path/"websocket-sessions",anchor_json=None
    )
    def launch():
        bridge._started=True
    def fake_wait(index):
        p=bridge.session_root/"perception"/f"result_{index}.npz"
        np.savez_compressed(p,dummy=np.ones(1))
        return p
    monkeypatch.setattr(bridge,"_launch",launch)
    monkeypatch.setattr(bridge,"_await_output",fake_wait)
    monkeypatch.setattr("scripts.deploy.websocket_hand_scene_bridge._geometry_from_npz",
                        lambda _:{"hand_scene":{"ok":True}})
    return pi0,bridge


def test_robot_can_switch_all_six_scenes_without_reloading_pi0(monkeypatch,tmp_path):
    from scripts.deploy.position_initialization import SCENE_POSITION_TO_EPISODE
    assert SELECTABLE_SCENE_POSITIONS==(1,2,3,4,5,6)
    policy,bridge=setup_bridge(tmp_path,monkeypatch)
    old_dirs=[]
    for scene in SELECTABLE_SCENE_POSITIONS:
        raw=frame(0)
        out=bridge.infer(observation(f"session-{scene}",scene,[raw]))
        assert out["actions"].shape==(50,18)
        assert policy.received[-1]["object_geometry"]["hand_scene"]["ok"]
        assert bridge.scene_position==scene
        assert bridge.next_index==1
        assert bridge.session_counter==scene
        assert (bridge.session_root/"raw"/"frame_000000.npz").exists()
        cmd=bridge._worker_command()
        assert cmd[cmd.index("--position-template-episode")+1]==str(SCENE_POSITION_TO_EPISODE[scene])
        assert cmd[cmd.index("--episode")+1]==str(SCENE_POSITION_TO_EPISODE[scene])
        assert "--auto-static-flask-from-frame0" in cmd
        assert "--register-missing-color-depth" in cmd
        assert "--static-target-anchor-json" not in cmd
        old_dirs.append(bridge.session_root)
    assert len(set(old_dirs))==6
    assert len(policy.received)==6
    assert bridge.policy is policy
    assert bridge.metadata["hand_scene_positions"]==[1,2,3,4,5,6]


def test_rejects_scene_change_midrun_and_missing_selection(monkeypatch,tmp_path):
    policy,bridge=setup_bridge(tmp_path,monkeypatch)
    first=frame(0)
    bridge.infer(observation("first-session",1,[first]))
    second=frame(1)
    with pytest.raises(ValueError,match="midway"):
        bridge.infer(observation("first-session",3,[second]))
    assert bridge.next_index==1
    broken=observation("second-session",1,[frame(0)])
    broken["hand_scene_position"]=None
    with pytest.raises(ValueError,match="hand_scene_position"):
        bridge.infer(broken)
    broken["hand_scene_position"]=23
    with pytest.raises(ValueError,match="hand_scene_position"):
        bridge.infer(broken)


def test_geometry_and_policy_must_share_current_frame(monkeypatch,tmp_path):
    _,bridge=setup_bridge(tmp_path,monkeypatch)
    obs=observation("S",2,[frame(0)])
    obs["observation.state"]=np.ones(1972,dtype=np.float32)
    with pytest.raises(ValueError,match="state and last geometry"):
        bridge.infer(obs)
    assert bridge.next_index==0
