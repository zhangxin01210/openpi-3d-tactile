"""CPU-only tests for dynamic observation gate and internal reset."""
from __future__ import annotations
import importlib.util
import tempfile
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import types

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('guard',ROOT/'scripts/preprocess/rgbtrack_motion_guard.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def pose(x=0., angle=0.):
    a=np.deg2rad(angle);p=np.eye(4)
    p[0,0]=np.cos(a);p[0,1]=-np.sin(a)
    p[1,0]=np.sin(a);p[1,1]=np.cos(a);p[0,3]=x
    return p


def test_threshold_math():
    assert m.check_motion(pose(),pose(.005,10),elapsed_frames=1,
                          max_step_m=.012,max_turn_deg=35)['accepted']
    assert m.check_motion(pose(),pose(.017,10),elapsed_frames=1,
                          max_step_m=.012,max_turn_deg=35)['reason']=='translation'
    assert m.check_motion(pose(),pose(.005,120),elapsed_frames=1,
                          max_step_m=.012,max_turn_deg=35)['reason']=='rotation'
    assert m.check_motion(pose(),pose(.02,25),elapsed_frames=2,
                          max_step_m=.012,max_turn_deg=35)['accepted']
    assert m.check_motion(pose(),pose(.03,160),elapsed_frames=2,
                          max_step_m=.012,max_turn_deg=35)['reason']=='translation'
    assert m.check_motion(None,pose(),elapsed_frames=1,max_step_m=.012,max_turn_deg=35)['accepted']

@dataclass
class Rec:
    frame_index:int
    object_name:str
    T_color_object:np.ndarray
    T_base_object:np.ndarray
    valid:bool=True
    confidence:float=1.
    status:str='observed'

class FakeState:
    def __init__(self):self.last_pose=None;self.last_frame=None
    def observe(self,p,f):self.last_pose=p.copy();self.last_frame=f

class FakeObject:
    def __init__(self,name,poses):
        self.object_name=name;self.poses=poses;self._state=FakeState()
        self._initialized=False;self._estimator=types.SimpleNamespace(track_good=True)
        self.called=0
    def estimate(self,*,frame_index,rgb,depth_m,mask,K,T_base_color):
        self.called+=1
        self._initialized=True
        p=self.poses[frame_index]
        self._state.observe(p,frame_index) # simulate upstream pollution
        return Rec(frame_index,self.object_name,p.copy(),p.copy())


def test_reject_resets_both_states():
    adapter=types.SimpleNamespace(RGBTrackObject=FakeObject)
    with tempfile.TemporaryDirectory() as tmp:
        writer=m.install_motion_guard(adapter,name='button',output=Path(tmp),
                                  max_step_m=.012,max_turn_deg=35.)
        button=FakeObject('button',[pose(),pose(.005,5),pose(.08,135),pose(.007,8)])
        box=FakeObject('box',[pose(),pose(.1,150)])
        kw=dict(rgb=None,depth_m=None,mask=None,K=None,T_base_color=None)
        assert button.estimate(frame_index=0,**kw).valid
        assert button.estimate(frame_index=1,**kw).valid
        bad=button.estimate(frame_index=2,**kw)
        assert not bad.valid and bad.status=='motion_rejected_translation'
        assert not button._initialized and not button._estimator.track_good
        assert button._state.last_frame==1
        assert np.allclose(button._state.last_pose,pose(.005,5))
        good=button.estimate(frame_index=3,**kw)
        assert good.valid
        # Static object bypasses all motion checks.
        assert box.estimate(frame_index=1,**kw).valid
        writer()
        import json
        report=json.loads((Path(tmp)/'motion_guard_report.json').read_text())
        assert report['counters']['rejected']==1

if __name__=='__main__':
    test_threshold_math();test_reject_resets_both_states();print('PASS motion guard')
