import numpy as np
import pytest
from deploy.contact_safety import ContactSafetyGuard, SafetyTrip, raw_tactile_finger_peaks

def config():
    return {
      "tactile_peak_limits_raw":[10.]*5,
      "tactile_step_change_limits_raw":[4.]*5,
      "arm_wrench_observation_key":"arm_tcp_wrench",
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
    out=g.check(observation={"arm_tcp_wrench":np.zeros(6)},state=state(3))
    assert out["tactile_finger_peaks_raw"][0]==pytest.approx(3)

@pytest.mark.parametrize("case",["tactile","jump","force","torque","no_arm","nonfinite","gap"])
def test_trip_latches_and_blocks_future_steps(case):
    t=[0.]
    g=ContactSafetyGuard(config(),clock=lambda:t[0])
    obs={"arm_tcp_wrench":np.zeros(6)}
    g.check(observation=obs,state=state(2))
    if case=="tactile": s=state(12)
    elif case=="jump": s=state(8)
    else: s=state(2)
    if case=="force": obs["arm_tcp_wrench"]=np.array([13,0,0,0,0,0])
    if case=="torque": obs["arm_tcp_wrench"]=np.array([0,0,0,0,0,3])
    if case=="no_arm": obs={}
    if case=="nonfinite": s[52+24]=float("nan")
    if case=="gap": t[0]=2.
    with pytest.raises(SafetyTrip):
        g.check(observation=obs,state=s)
    with pytest.raises(SafetyTrip,match="latched"):
        g.check(observation={"arm_tcp_wrench":np.zeros(6)},state=state(0))
