#!/usr/bin/env python3
"""No GPU/no pytest; run with .venv/bin/python directly."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'scripts/preprocess'))
# The current static policy is intentionally reused rather than replaced.
# Local bundle tests can import its standalone verified file from the merged zip.
try:
    import rgbtrack_static_policy as static
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3] / '_inspect_pkg/press_button_pipeline_merged_20261031/scripts/preprocess'))
    import rgbtrack_static_policy as static
import rgbtrack_dynamic_init as dynamic


def cube_pts():
    return np.asarray([[x,y,z] for x in [-.03,.03] for y in [-.015,.015] for z in [-.06,.06]],float)


def pose(r):
    out=np.eye(4)
    out[:3,:3]=r
    out[:3,3]=[0,0,1]
    return out


def test_select_correct_upright():
    p0=pose(Rotation.from_euler('y',90,degrees=True).as_matrix())
    p1=pose(np.eye(3))
    p=np.stack([p0,p1])
    K=np.array([[150.,0,50.],[0,150.,50.],[0,0,1.]])
    mask=static.mesh_projection_mask(cube_pts(),p1,K,(100,100))
    candidate,entries=dynamic.select_first_dynamic_pose(
        poses=p,scores=np.array([10.,9.]),center=np.eye(4),vertices=cube_pts(),
        mask=mask,K=K,base_R_camera=np.eye(3),upright_axis='+z',max_tilt_deg=25,
        min_iou=.2)
    assert candidate is not None and candidate['rank']==1,(candidate,entries)


def test_preserve_upright_native():
    p0=pose(np.eye(3))
    p1=pose(Rotation.from_euler('z',7,degrees=True).as_matrix())
    K=np.array([[150.,0,50.],[0,150.,50.],[0,0,1.]])
    mask=static.mesh_projection_mask(cube_pts(),p0,K,(100,100))
    candidate,_=dynamic.select_first_dynamic_pose(
        poses=np.stack([p0,p1]),scores=np.array([10.,11.]),center=np.eye(4),
        vertices=cube_pts(),mask=mask,K=K,base_R_camera=np.eye(3),
        upright_axis='+z')
    assert candidate['rank']==0


def test_axis_is_object_relative():
    # For a different mesh frame, its local +X rather than +Z is vertical.
    up=pose(Rotation.from_euler('y',-90,degrees=True).as_matrix())
    flat=pose(np.eye(3))
    K=np.array([[150.,0,50.],[0,150.,50.],[0,0,1.]])
    mask=static.mesh_projection_mask(cube_pts(),up,K,(100,100))
    candidate,_=dynamic.select_first_dynamic_pose(
        poses=np.stack([up,flat]),scores=np.array([10.,9.]),center=np.eye(4),
        vertices=cube_pts(),mask=mask,K=K,base_R_camera=np.eye(3),
        upright_axis='+x')
    assert candidate and candidate['rank']==0


def test_no_upright_candidate_fails_closed():
    flat=pose(Rotation.from_euler('y',90,degrees=True).as_matrix())
    K=np.array([[150.,0,50.],[0,150.,50.],[0,0,1.]])
    mask=static.mesh_projection_mask(cube_pts(),flat,K,(100,100))
    candidate,_=dynamic.select_first_dynamic_pose(
        poses=np.stack([flat]),scores=np.array([10.]),center=np.eye(4),
        vertices=cube_pts(),mask=mask,K=K,base_R_camera=np.eye(3),
        upright_axis='+z')
    assert candidate is None


def test_policy_config_and_runner_flags():
    import importlib.util
    spec=importlib.util.spec_from_file_location('task',repo/'scripts/preprocess/rgbtrack_task.py')
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    from argparse import Namespace
    args=Namespace(static_object='box',dynamic_object='button',mode='depth',
        init_tilt_deg=25.,init_rank=150,min_mask_iou=.35,
        guard_max_step_cm=2.,guard_max_turn_deg=25.,guard_recovery_max_turn_deg=60.,
        dynamic_init_up_axis='+z',dynamic_init_max_tilt_deg=25.,dynamic_init_min_iou=.2)
    cfg=task.policy_settings(args,{'objects':['button','box']})
    assert cfg['dynamic_init_up_axis']=='+z'
    assert cfg['guard_max_step_m']==.02
    assert cfg['static_object']=='box'


if __name__=='__main__':
    for name,obj in list(globals().items()):
        if name.startswith('test_') and callable(obj):
            obj();print('[PASS]',name)
    print('PASS dynamic upright initialization')
