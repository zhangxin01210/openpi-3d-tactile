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
        "arm_tcp_force":np.array([1.,2.,3.],dtype=float),
        "arm_tcp_torque":np.array([0.1,0.2,0.3],dtype=float),
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
        np.testing.assert_allclose(arrays["arm_tcp_force"],[1,2,3])
        np.testing.assert_allclose(arrays["arm_tcp_torque"],[.1,.2,.3])
        assert arrays["robot_state"].shape==(1972,)


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


def test_audit_absent_ur_wrench_is_not_faked(tmp_path):
    observation,names=_input()
    observation.pop("arm_tcp_force")
    observation.pop("arm_tcp_torque")
    robot=MockRobot(observation)
    args=SimpleNamespace(capture_observation_audit=tmp_path/"audit",
                         capture_audit_frames=1,realsense_height=12,
                         realsense_width=16,fps=15)
    client.capture_observation_audit(robot,args,names)
    report=json.loads((args.capture_observation_audit/"observation_schema.json").read_text())
    d=report["records"][0]["diagnostics"]
    assert d["arm_tcp_force"]["status"]=="missing"
    assert d["arm_tcp_torque"]["status"]=="missing"
    with np.load(args.capture_observation_audit/"audit_0000.npz",allow_pickle=False) as arrays:
        assert "arm_tcp_force" not in arrays.files
        assert "arm_tcp_torque" not in arrays.files


def test_build_robot_enables_ur_wrench_without_mutating_training_state(monkeypatch):
    import types
    from types import SimpleNamespace
    objects=[]
    def record_config(**kwargs):
        result=SimpleNamespace(**kwargs)
        objects.append(result)
        return result
    mapping={
      "lerobot.cameras.configs":SimpleNamespace(ColorMode=SimpleNamespace(RGB="RGB")),
      "lerobot.cameras.realsense":SimpleNamespace(RealSenseCameraConfig=record_config),
      "lerobot.robots.ur7e.ur7e_config":SimpleNamespace(UR7eConfig=record_config),
      "lerobot.robots.ur7e_xhand.ur7e_xhand":SimpleNamespace(UR7eXHand=lambda cfg:cfg),
      "lerobot.robots.ur7e_xhand.ur7e_xhand_config":SimpleNamespace(UR7eXHandConfig=record_config),
      "lerobot.robots.xhand.xhand_config":SimpleNamespace(XHandConfig=record_config),
    }
    original=client.importlib.import_module
    monkeypatch.setattr(client.importlib,"import_module",lambda name: mapping[name] if name in mapping else original(name))
    args=SimpleNamespace(
        realsense_fps=15,realsense_width=640,realsense_height=480,
        realsense_front_serial="front",realsense_left_serial="left",realsense_right_serial="right",
        spatial_cameras=("front",),arm_ip="127.0.0.1",arm_control_freq=15.,
        arm_max_relative_target=.25,hand_protocol="EtherCAT",hand_serial_port="",
        hand_ethercat_interface="",hand_control_freq=None,fps=15,
        camera_read_timeout_ms=80.)
    robot_cfg=client.build_robot(args)
    assert robot_cfg.arm_config.record_tcp_force_torque is True
    assert robot_cfg.cameras["cam_front"].use_depth is True
    assert robot_cfg.cameras["cam_right"].use_depth is False


def test_single_script_sensor_only_http_forwarder_no_actions(monkeypatch):
    import io
    from types import SimpleNamespace

    obs,names=_input()
    robot=MockRobot(obs)
    token="only-sensors-no-actions-12345"
    monkeypatch.setenv("OPENPI_LIVE_PACKET_TOKEN",token)
    requests=[]
    class Reply:
        def __init__(self,frame): self.frame=frame
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def read(self):
            return json.dumps({"accepted_frame":self.frame,"no_robot_action":True}).encode()

    def fake_urlopen(request,timeout):
        assert request.get_header("X-live-packet-token")==token
        with np.load(io.BytesIO(request.data),allow_pickle=False) as packet:
            assert packet["robot_state"].shape==(1972,)
            assert "front_depth_native" in packet.files
            assert "front_depth_color_aligned" not in packet.files
            assert packet["front_depth_native"].dtype==np.uint16
            idx=int(packet["frame_index"])
        requests.append(idx)
        return Reply(idx)
    monkeypatch.setattr(client.urllib.request,"urlopen",fake_urlopen)
    monkeypatch.setattr(client.time,"sleep",lambda _:None)
    args=SimpleNamespace(live_packet_url="http://127.0.0.1:8765/frame",
                         live_packet_frames=2,realsense_height=12,realsense_width=16,fps=15)
    client.capture_live_packet_stream(robot,args,names)
    assert requests==[0,1]
    assert robot.reads==2


def test_websocket_geometry_history_reuses_original_live_depth_and_state():
    observation,names=_input()
    observation["cam_front"]=np.zeros((480,640,3),dtype=np.uint8)
    observation["cam_left"]=np.zeros((480,640,3),dtype=np.uint8)
    observation["cam_right"]=np.zeros((480,640,3),dtype=np.uint8)
    observation["depths.cam_front"]=np.full((480,640),1000,dtype=np.uint16)
    args=SimpleNamespace(realsense_height=480,realsense_width=640)
    state=client.build_env_state(observation,names)
    frame=client.raw_hand_scene_history_frame(observation,state,args,0)
    assert int(frame["frame_index"])==0
    assert frame["timestamp_s"].shape==()
    assert frame["robot_state"].shape==(1972,)
    assert frame["front_depth_native"].dtype==np.uint16
    assert "front_depth_color_aligned" not in frame
    assert "object_geometry" not in frame
    np.testing.assert_array_equal(frame["front_depth_native"],observation["depths.cam_front"])
    np.testing.assert_array_equal(frame["front_rgb"],observation["cam_front"])
    observation["depths.cam_front"][0,0]=9999
    assert frame["front_depth_native"][0,0]==1000  # no aliasing across steps
