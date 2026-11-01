import importlib.util
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2
import tempfile

spec = importlib.util.spec_from_file_location('selector', Path(__file__).resolve().parents[1] / 'scripts/preprocess/rgbtrack_static_policy.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

K = np.array([[180.,0,160.],[0,180.,120.],[0,0,1.]])
vertices = np.array([[x,y,z] for x in (-.1,.1) for y in (-.05,.05) for z in (-.04,.04)],float)
T0 = np.eye(4);T0[:3,3]=[0.,0.,1.]
Tbad = T0.copy();Tbad[0,3]=.2
Tinv = T0.copy();Tinv[1,1]=-1;Tinv[2,2]=-1

def test_mask_selects_better_not_first():
    reference = m.mesh_projection_mask(vertices,T0,K,(240,320))
    selected,entries,tilt=m.evaluate_candidates(np.stack((Tbad,T0)),[100.,99.],np.eye(4),vertices,reference,K,np.eye(3),max_rank=2)
    assert selected['rank']==1, selected
    assert selected['mask_iou']>.95
    assert entries[0]['rank']==1

def test_tilt_rejects_upside_down_even_if_iou_equal():
    reference=m.mesh_projection_mask(vertices,T0,K,(240,320))
    selected,entries,tilt=m.evaluate_candidates(np.stack((Tinv,T0)),[100.,99.],np.eye(4),vertices,reference,K,np.eye(3),max_rank=2)
    assert selected['rank']==1
    assert len(entries)==1
    assert abs(tilt[0]-180)<1e-5

def test_fails_closed_when_no_match():
    reference=np.zeros((240,320),np.uint8);reference[:10,:10]=1
    selected,entries,_=m.evaluate_candidates(np.stack((T0,)),[100.],np.eye(4),vertices,reference,K,np.eye(3),max_rank=1)
    assert selected is None

@dataclass
class Rec:
    frame_index:int=0
    T_color_object:np.ndarray=None
    T_base_object:np.ndarray=None
    valid:bool=True
    status:str='observed'
    mask_area_px:int=100
    elapsed_ms:float=10.

class FakeEstimator:
    def __init__(self):
        self.poses=np.stack((Tbad,T0))
        self.scores=np.array([100.,99.])
        self.pose_last=None
        self.called=0
        self.tracker=type('Tr',(),{'initialize':lambda self,*args:None})()
    def register(self, *,K,rgb,depth,ob_mask,iteration):
        self.called+=1
        self.pose_last=self.poses[0]
        return self.poses[0]
    def get_tf_to_centered_mesh(self):return np.eye(4)

class FakeObject:
    def __init__(self,name):
        self.object_name=name
        self._estimator=FakeEstimator()
        self._mesh=type('M',(),{'vertices':vertices})()
        self.counter=0
    def _ensure_estimator(self):return
    def estimate(self, *,frame_index,rgb,depth_m,mask,K,T_base_color):
        self._ensure_estimator()
        self.counter+=1
        p=self._estimator.register(K=K,rgb=rgb,depth=depth_m,ob_mask=mask,iteration=1)
        return Rec(frame_index,np.asarray(p),np.asarray(p),True,'observed',int(mask.sum()))

def test_freezing_and_dynamic_passthrough():
    class Adapter: RGBTrackObject=FakeObject
    fake = Adapter()
    with tempfile.TemporaryDirectory() as tmp:
        m.install_static_policy(fake,'box',np.eye(3),Path(tmp),25.,150,.35)
        box=FakeObject('box')
        btn=FakeObject('button')
        rgb=np.zeros((240,320,3),np.uint8)
        mask=m.mesh_projection_mask(vertices,T0,K,(240,320))
        args=dict(rgb=rgb,depth_m=np.ones((240,320)),mask=mask,K=K,T_base_color=np.eye(4))
        r0=box.estimate(frame_index=0,**args)
        r1=box.estimate(frame_index=1,**args)
        assert box.counter==1
        assert r0.status=='static_initial_observation'
        assert r1.status=='static_frozen'
        assert np.allclose(r1.T_color_object,T0)
        btn.estimate(frame_index=0,**args)
        btn.estimate(frame_index=1,**args)
        assert btn.counter==2
        assert (Path(tmp)/'box_init_candidates.png').exists()
        assert (Path(tmp)/'box_init_candidates.json').exists()
        assert __import__('json').loads((Path(tmp)/'box_init_candidates.json').read_text())['selected']['rank']==1

if __name__=='__main__':
    test_mask_selects_better_not_first()
    test_tilt_rejects_upside_down_even_if_iou_equal()
    test_fails_closed_when_no_match()
    test_freezing_and_dynamic_passthrough()
    print('PASS')
