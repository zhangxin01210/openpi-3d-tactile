"""CPU-only contract tests: per-taxel force swaps, no motion commands, exact FK delta."""
import numpy as np
import pytest

from scripts.deploy.tactile_motion_probe import (
    _change_raw_taxel_force, _vary, _classify_z, _fk_intent,
)
from openpi.spatial.kinematics import RobotKinematics, load_state_mapping_csv


def _geometry(value):
    return {
        "target_xyz_m": np.full((512,3), value, dtype=np.float32),
        "manipulated_xyz_m": np.full((512,3), value, dtype=np.float32),
        "hand_scene": {
            "grasp_reference_force_norm":np.full((600,),value,dtype=np.float32),
            "grasp_short_delta_force_norm":np.full((600,),value,dtype=np.float32),
            "grasp_contact_valid":np.ones((600,),dtype=bool),
            "grasp_quality":np.asarray(0.0,dtype=np.float32),
            "causal_tactile":{
                "ema_before_force_norm":np.full((600,),value,dtype=np.float32),
                "ema_after_force_norm":np.full((600,),value,dtype=np.float32),
                "current_minus_ema_force_norm":np.zeros((600,),dtype=np.float32),
                "current_minus_previous_force_norm":np.zeros((600,),dtype=np.float32),
                "reference_valid":np.ones((600,),dtype=bool),
                "previous_valid":np.ones((600,),dtype=bool),
                "sample_valid":np.ones((600,),dtype=bool),
                "alpha":np.asarray(.1,dtype=np.float32),
            },
            "funnel_mask":np.zeros((32,32),dtype=bool),
        }
    }


def test_raw_force_swap_preserves_every_other_state_field():
    from openpi.spatial.config import make_baseline_config
    c=make_baseline_config().tactile
    before=np.arange(1972,dtype=np.float32)
    donor=np.full(1972,5,dtype=np.float32)
    modified=_change_raw_taxel_force(before,donor)
    expected=before.copy()
    for i in range(5):
        a=52+i*c.tactile_block_size+c.raw_force_offset
        expected[a:a+360]=5
    np.testing.assert_array_equal(modified,expected)
    null=_change_raw_taxel_force(before,donor,zero=True)
    np.testing.assert_array_equal(null[:52],before[:52])
    assert np.count_nonzero(null[52:]!=before[52:]) == 5*360


def test_intervention_copies_historical_tactile_only():
    actual={"robot_state":np.arange(1972,dtype=np.float32)}
    donor={"robot_state":np.ones(1972,dtype=np.float32)}
    geo=_geometry(0.5)
    dgeo=_geometry(0.8)
    obs,g=_vary(actual,geo,donor,dgeo)
    assert np.array_equal(g["target_xyz_m"],geo["target_xyz_m"])
    assert np.array_equal(g["hand_scene"]["funnel_mask"],geo["hand_scene"]["funnel_mask"])
    assert np.all(g["hand_scene"]["grasp_reference_force_norm"]==0.8)
    assert np.all(geo["hand_scene"]["grasp_reference_force_norm"]==0.5)
    assert np.array_equal(obs["robot_state"][:52],actual["robot_state"][:52])
    zero,g0=_vary(actual,geo,donor,dgeo,zero=True)
    assert np.all(g0["hand_scene"]["grasp_reference_force_norm"]==0)
    assert np.all(g0["hand_scene"]["causal_tactile"]["sample_valid"])
    assert not np.any(g0["hand_scene"]["grasp_contact_valid"])


def test_fk_same_target_has_zero_palm_movement():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    kin=RobotKinematics(root/"configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    mapping=load_state_mapping_csv(root/"configs/ur7e_xhand/state_mapping.csv")
    state=np.zeros(1972,dtype=np.float32)
    positions=np.concatenate([state[:6],state[28:52:2]])
    result=_fk_intent(kin,mapping,state,np.tile(positions,(16,1)),np.zeros((512,3)))
    assert np.max(np.abs(result["z_mm"])) < 1e-6
    assert np.max(np.abs(result["x_mm"])) < 1e-6
    assert np.max(np.abs(result["rotation_deg"])) < 1e-3
    assert _classify_z(-1.0)=="DOWN"
    assert _classify_z(1.0)=="UP"
    assert _classify_z(0.1)=="LEVEL"
