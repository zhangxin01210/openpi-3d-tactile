"""No GPU: upstream static target never calls dynamic RGBTrack after frame zero."""
import hashlib
import json

import numpy as np
import pytest

from scripts.deploy.static_flask_runtime import StaticFlaskRuntime, load_reviewed_anchor


class Tracker:
    def __init__(self, pose):
        self.pose=pose
        self.calls=0
    def estimate(self, **kwargs):
        self.calls+=1
        return type("Record",(),{"valid":True,"T_base_object":self.pose,"confidence":0.4})()


def test_static_target_f0_observation_then_467_priors():
    anchor=np.eye(4)
    anchor[:3,3]=[0.3,0.4,0.2]
    runtime=StaticFlaskRuntime(anchor)
    tracker=Tracker(anchor.copy())
    first=runtime.next(frame_index=0,tracker=tracker,mask=np.ones((4,4),bool))
    assert first["pose_source"]=="reviewed_initial_geometry"
    assert first["observed_pose_valid"]
    for frame in range(1,468):
        row=runtime.next(frame_index=frame,tracker=None)
        assert row["pose_source"]=="static_geometry_prior"
        assert row["status"]=="flask_static_anchor_prior"
        assert row["valid"] and not row["observed_pose_valid"]
        np.testing.assert_array_equal(row["T_base_link_object"],anchor)
    assert tracker.calls==1


def test_static_target_requires_current_f0_close_to_reviewed_anchor():
    anchor=np.eye(4)
    tracker=Tracker(np.eye(4))
    tracker.pose[0,3]=0.25
    with pytest.raises(ValueError,match="differs from human-reviewed anchor"):
        StaticFlaskRuntime(anchor).next(frame_index=0,tracker=tracker,mask=np.ones((2,2),bool))


def test_prior_cannot_run_without_f0():
    with pytest.raises(ValueError,match="cannot precede"):
        StaticFlaskRuntime(np.eye(4)).next(frame_index=1)


def test_anchor_has_explicit_review_and_calibration_provenance(tmp_path):
    cal=tmp_path/"calibration.json"
    cal.write_text("{}")
    record={
        "reviewed":True,"source":"human_reviewed","episode_index":23,
        "front_calibration_sha256":hashlib.sha256(cal.read_bytes()).hexdigest(),
        "T_base_link_object":np.eye(4).tolist()
    }
    anchor=tmp_path/"anchor.json"
    anchor.write_text(json.dumps(record))
    T,_=load_reviewed_anchor(anchor,episode=23,calibration_profile=cal)
    np.testing.assert_array_equal(T,np.eye(4))
    with pytest.raises(ValueError,match="episode mismatch"):
        load_reviewed_anchor(anchor,episode=38,calibration_profile=cal)
    record["reviewed"]=False
    anchor.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="human-reviewed"):
        load_reviewed_anchor(anchor,episode=23,calibration_profile=cal)
