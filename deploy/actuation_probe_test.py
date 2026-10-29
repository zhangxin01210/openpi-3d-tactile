"""No robot: validate fail-closed bounds for plastic-prop short motion probe."""
import numpy as np
import pytest
from deploy.ur7e_xhand_deploy_3d_tactile_multi import (
    FALLBACK_ACTION_NAMES,
    PROBE_ARM_STEP_RAD,
    PROBE_HAND_STEP_RAD,
    PROBE_ARM_TOTAL_RAD,
    PROBE_HAND_TOTAL_RAD,
    SafetyTrip,
    check_probe_inference_idle_drift,
    limit_probe_joint_targets,
)


def _observation(values):
    return dict(zip(FALLBACK_ACTION_NAMES, map(float,values),strict=True))


def test_large_model_targets_are_bounded_by_measured_joint_state():
    start=np.zeros(18,dtype=np.float32)
    target=np.full(18,3.,dtype=np.float32)
    result=limit_probe_joint_targets(
        requested=target,observed=start,starting=start,
        action_names=FALLBACK_ACTION_NAMES)
    np.testing.assert_allclose(result[:6],PROBE_ARM_STEP_RAD,rtol=1e-5)
    np.testing.assert_allclose(result[6:],PROBE_HAND_STEP_RAD,rtol=1e-5)


def test_bounds_remain_within_start_envelope_despite_repeated_requests():
    start=np.zeros(18,dtype=np.float32)
    target=np.full(18,3.,dtype=np.float32)
    current=start.copy()
    for _ in range(100):
        current=limit_probe_joint_targets(
            requested=target,observed=current,starting=start,
            action_names=FALLBACK_ACTION_NAMES)
    assert np.max(np.abs(current[:6]-start[:6]))<=PROBE_ARM_TOTAL_RAD+1e-6
    assert np.max(np.abs(current[6:]-start[6:]))<=PROBE_HAND_TOTAL_RAD+1e-6


def test_rejects_missing_nonfinite_unsafe_encoders_and_invalid_order():
    v=np.zeros(18)
    kwargs=dict(requested=v,observed=v,starting=v,action_names=FALLBACK_ACTION_NAMES)
    with pytest.raises(SafetyTrip,match="outside bounded"):
        limit_probe_joint_targets(**dict(kwargs,observed=np.full(18,1.)))
    bad=v.copy();bad[6]=np.nan
    with pytest.raises(SafetyTrip,match="non-finite"):
        limit_probe_joint_targets(**dict(kwargs,requested=bad))
    with pytest.raises(SafetyTrip,match="ordering"):
        limit_probe_joint_targets(**dict(kwargs,action_names=FALLBACK_ACTION_NAMES[::-1]))


def test_blocked_inference_rechecks_external_robot_drift():
    origin=np.zeros(18,dtype=np.float32)
    same=origin.copy()
    check_probe_inference_idle_drift(_observation(origin),_observation(same),FALLBACK_ACTION_NAMES)
    moved=origin.copy(); moved[0]=0.04
    with pytest.raises(SafetyTrip,match="moved during blocked"):
        check_probe_inference_idle_drift(_observation(origin),_observation(moved),FALLBACK_ACTION_NAMES)
    moved=origin.copy(); moved[6]=0.07
    with pytest.raises(SafetyTrip,match="moved during blocked"):
        check_probe_inference_idle_drift(_observation(origin),_observation(moved),FALLBACK_ACTION_NAMES)



def test_plastic_probe_requires_explicit_plastic_review_without_claiming_glass(tmp_path):
    import json
    from deploy.ur7e_xhand_deploy_3d_tactile_multi import load_safety_limits
    config={
        "schema_version":1,
        "reviewed_for_glass_task":False,
        "reviewed_for_plastic_probe":True,
        "tactile_peak_limits_raw":[8.]*5,
        "tactile_emergency_step_change_limits_raw":[6.]*5,
        "tactile_replan_step_change_limits_raw":[3.]*5,
        "replan_enabled":False,
        "replan_min_interval_s":0.25,
        "arm_tcp_force_norm_limit_n":10.,
        "arm_tcp_torque_norm_limit_nm":1.5,
        "max_observation_gap_s":1.,
    }
    file=tmp_path/"limits.json"
    file.write_text(json.dumps(config))
    assert load_safety_limits(file,plastic_probe=True)["reviewed_for_plastic_probe"] is True
    with pytest.raises(ValueError,match="exactly"):
        load_safety_limits(file)  # plastic sign-off must NEVER enable glass
    config["reviewed_for_plastic_probe"]=False
    file.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="reviewed_for_plastic_probe"):
        load_safety_limits(file,plastic_probe=True)
