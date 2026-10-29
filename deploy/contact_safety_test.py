import numpy as np
import pytest
from deploy.contact_safety import ContactSafetyGuard, SafetyTrip, raw_tactile_finger_peaks

def config():
    return {
      "tactile_peak_limits_raw":[10.]*5,
      "tactile_emergency_step_change_limits_raw":[8.]*5,
      "tactile_replan_step_change_limits_raw":[4.]*5,
      "replan_enabled":False,
      "replan_min_interval_s":0.25,
      "arm_tcp_force_norm_limit_n":12.0,
      "arm_tcp_torque_norm_limit_nm":2.0,
      "max_observation_gap_s":1.0,
    }

def state(raw=0):
    s=np.zeros(1972,dtype=np.float32)
    s[52+24]=raw
    return s

def test_raw_tactile_index_and_peak():
    assert raw_tactile_finger_peaks(state(3))[0]==pytest.approx(3)
    assert np.all(raw_tactile_finger_peaks(state(3))[1:]==0)

def test_no_excess_safe_reading():
    g=ContactSafetyGuard(config(),clock=lambda:0.)
    out=g.check(observation={"arm_tcp_force":np.zeros(3),"arm_tcp_torque":np.zeros(3)},state=state(3))
    assert out["tactile_finger_peaks_raw"][0]==pytest.approx(3)

@pytest.mark.parametrize("case",["tactile","jump","force","torque","no_arm","nonfinite","gap"])
def test_trip_latches_and_blocks_future_steps(case):
    t=[0.]
    g=ContactSafetyGuard(config(),clock=lambda:t[0])
    obs={"arm_tcp_force":np.zeros(3),"arm_tcp_torque":np.zeros(3)}
    g.check(observation=obs,state=state(9 if case=="jump" else 2))
    if case=="tactile": s=state(12)
    elif case=="jump": s=state(0)  # dangerous sudden loss, no absolute peak violation
    else: s=state(2)
    if case=="force": obs["arm_tcp_force"]=np.array([13,0,0])
    if case=="torque": obs["arm_tcp_torque"]=np.array([0,0,3])
    if case=="no_arm": obs={}
    if case=="nonfinite": s[52+24]=float("nan")
    if case=="gap": t[0]=2.
    with pytest.raises(SafetyTrip):
        g.check(observation=obs,state=s)
    with pytest.raises(SafetyTrip,match="latched"):
        g.check(observation={"arm_tcp_force":np.zeros(3),"arm_tcp_torque":np.zeros(3)},state=state(0))

def test_contact_event_replans_without_latched_stop():
    cfg=config()
    cfg["replan_enabled"]=True
    t=[0.]
    guard=ContactSafetyGuard(cfg,clock=lambda:t[0])
    observation={"arm_tcp_force":np.zeros(3),"arm_tcp_torque":np.zeros(3)}
    guard.check(observation=observation,state=state(1))
    t[0]=0.1
    event=guard.check(observation=observation,state=state(6))
    assert event["replan_recommended"] is True
    assert event["replan_triggered_fingers"]==[0]
    assert guard.tripped is None
    t[0]=0.15
    assert guard.check(observation=observation,state=state(1))["replan_recommended"] is False
    t[0]=0.5
    assert guard.check(observation=observation,state=state(6))["replan_recommended"] is True


def test_fixed_16_step_schedule_default_has_no_contact_replan():
    cfg=config()
    guard=ContactSafetyGuard(cfg,clock=lambda:0.)
    obs={"arm_tcp_force":np.zeros(3),"arm_tcp_torque":np.zeros(3)}
    guard.check(observation=obs,state=state(1))
    event=guard.check(observation=obs,state=state(6))
    assert event["replan_recommended"] is False
    with pytest.raises(SafetyTrip):
        guard.check(observation=obs,state=state(16))


def test_split_rtde_tcp_wrench_and_raw_tactile_keeps_state_1972():
    cfg=config()
    guard=ContactSafetyGuard(cfg,clock=lambda:0.)
    v=state(1)
    obs={"arm_tcp_force":np.array([3.,4.,0.]),
         "arm_tcp_torque":np.array([0.,0.,1.])}
    status=guard.check(observation=obs,state=v)
    assert v.shape==(1972,)
    assert status["arm_tcp_force_norm_n"]==pytest.approx(5.)
    assert status["arm_tcp_torque_norm_nm"]==pytest.approx(1.)


@pytest.mark.parametrize("invalid",["missing_force","missing_torque","invalid_force","nonfinite_torque"])
def test_arm_tcp_force_torque_fail_closed_when_invalid(invalid):
    obs={"arm_tcp_force":np.zeros(3),"arm_tcp_torque":np.zeros(3)}
    if invalid=="missing_force": obs.pop("arm_tcp_force")
    if invalid=="missing_torque": obs.pop("arm_tcp_torque")
    if invalid=="invalid_force": obs["arm_tcp_force"]=np.zeros(6)
    if invalid=="nonfinite_torque": obs["arm_tcp_torque"]=np.array([0.,float("nan"),0.])
    guard=ContactSafetyGuard(config(),clock=lambda:0.)
    with pytest.raises(SafetyTrip):
        guard.check(observation=obs,state=state(1))


def test_reviewed_safety_config_uses_split_rtde_contract(tmp_path):
    import json
    from deploy.contact_safety import load_safety_limits
    c=config()
    c.update(schema_version=1,reviewed_for_glass_task=True)
    file=tmp_path/"limits.json"
    file.write_text(json.dumps(c))
    assert load_safety_limits(file)["replan_enabled"] is False
    c["arm_wrench_observation_key"]="arm_tcp_wrench"  # forbidden legacy contract
    file.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="exactly"):
        load_safety_limits(file)
