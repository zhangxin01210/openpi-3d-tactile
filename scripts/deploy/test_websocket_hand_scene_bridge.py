"""Offline checks for original WebSocket serving path with causal geometry history."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deploy.websocket_hand_scene_bridge import (
    FRAME_KEYS,HandScenePerceptionBridge,validate_history
)


def _frame(i):
    return {
        "front_rgb":np.zeros((480,640,3),np.uint8),
        "left_rgb":np.zeros((480,640,3),np.uint8),
        "right_rgb":np.zeros((480,640,3),np.uint8),
        "front_depth_native":np.ones((480,640),np.uint16),
        "robot_state":np.zeros(1972,np.float32),
        "timestamp_s":np.asarray(100+i/15),
        "frame_index":np.asarray(i),
    }


def test_history_preserves_16_contiguous_raw_frames_and_fails_closed():
    batch=[_frame(i) for i in range(16)]
    out=validate_history(batch,next_index=0,previous_timestamp=float("-inf"))
    assert len(out)==16
    assert out[0]["front_depth_native"].dtype==np.uint16
    with pytest.raises(ValueError,match="discontinuity"):
        validate_history(batch,next_index=1,previous_timestamp=float("-inf"))
    with pytest.raises(ValueError,match="timestamp"):
        validate_history(batch,next_index=0,previous_timestamp=100.01)
    bad=[_frame(0)]
    bad[0].pop("left_rgb")
    with pytest.raises(KeyError):
        validate_history(bad,next_index=0,previous_timestamp=float("-inf"))


def test_bridge_injects_exact_geometry_into_existing_policy_and_forbids_session_reuse(monkeypatch,tmp_path):
    class StubPi0:
        metadata={"spatial_camera_roles":["front"]}
        def __init__(self):self.received=[]
        def infer(self,obs):
            self.received.append(obs)
            return {"actions":np.zeros((50,18),np.float32)}
    pi0=StubPi0()
    anchor=tmp_path/"reviewed.json"
    anchor.write_text("{}")
    python=tmp_path/"perception_python"
    python.write_text("not executed by stub")
    worker=HandScenePerceptionBridge(pi0,repo_root=tmp_path,
            perception_python=python,workdir=tmp_path/"work",anchor_json=anchor)
    def fake_launch():
        worker._started=True
    def fake_wait(index):
        path=worker.workdir/"perception"/f"result_{index}.npz"
        np.savez_compressed(path,dummy=np.ones(1))
        return path
    monkeypatch.setattr(worker,"_launch",fake_launch)
    monkeypatch.setattr(worker,"_await_output",fake_wait)
    monkeypatch.setattr("scripts.deploy.websocket_hand_scene_bridge._geometry_from_npz",
                        lambda _:{"hand_scene":{"test":True}})
    first=_frame(0)
    obs={
        "hand_scene_session":"session-A",
        "hand_scene_history":[first],
        "observation.state":first["robot_state"],
        "observation.depths.cam_front":first["front_depth_native"],
        "observation.images.cam_front":first["front_rgb"],
    }
    pred=worker.infer(obs)
    assert pred["actions"].shape==(50,18)
    assert pi0.received[0]["object_geometry"]["hand_scene"]["test"]
    assert "hand_scene_history" not in pi0.received[0]
    assert worker.next_index==1
    assert (worker.workdir/"raw"/"frame_000000.npz").is_file()
    assert worker.metadata["hand_scene_history_required"] is True
    bad=dict(obs,hand_scene_session="session-B",hand_scene_history=[_frame(1)])
    with pytest.raises(RuntimeError,match="server restart"):
        worker.infer(bad)
    second=_frame(1)
    obs.update(hand_scene_history=[second],
               **{"observation.state":second["robot_state"]})
    worker.infer(obs)
    assert worker.next_index==2
    assert (worker.workdir/"raw"/"frame_000001.npz").is_file()


def test_geometry_frame_must_match_pi0_observation(monkeypatch,tmp_path):
    anchor=tmp_path/"a.json";anchor.write_text("{}")
    python=tmp_path/"python";python.write_text("test")
    pi0=SimpleNamespace(metadata={},infer=lambda o:{"actions":np.zeros((50,18))})
    worker=HandScenePerceptionBridge(pi0,repo_root=tmp_path,
            perception_python=python,workdir=tmp_path/"w",anchor_json=anchor)
    frame=_frame(0)
    obs={
        "hand_scene_session":"S","hand_scene_history":[frame],
        "observation.state":np.ones(1972),
        "observation.depths.cam_front":frame["front_depth_native"],
        "observation.images.cam_front":frame["front_rgb"],
    }
    with pytest.raises(ValueError,match="state and last geometry"):
        worker.infer(obs)
    assert worker.next_index==0
