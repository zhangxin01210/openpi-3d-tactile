"""Offline contract tests: fake video/SAM2 only; never claims live GPU validation."""
from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'preprocess' / 'sam2_task.py'
spec = importlib.util.spec_from_file_location('sam2_task_integration', SCRIPT)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def args(**kw):
    return types.SimpleNamespace(**kw)


class FakeRawSpatialDataset:
    FRAMES={}

    @staticmethod
    def discover_episode_ids(_root):
        return tuple(sorted(FakeRawSpatialDataset.FRAMES))

    def __init__(self, root, episode=0):
        self.episode=episode

    def iter_video_frames(self, role):
        assert role=='front'
        for fi, img in enumerate(self.FRAMES[self.episode]):
            yield types.SimpleNamespace(frame_index=fi, rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


def fake_sam_predict(_pred, frame, pos, neg, box=None):
    x,y=map(int,np.rint(pos[0]))
    center=frame[y,x].astype(int)
    if center[2]>center[1]+50 and center[2]>center[0]+50:
        mask=(frame[:,:,2]>160)&(frame[:,:,1]<100)
    elif center[1]>center[2]+50 and center[1]>center[0]+50:
        mask=(frame[:,:,1]>160)&(frame[:,:,2]<100)
    else:
        raise ValueError(f'Prompt not on synthesized target: {center}')
    return mask.astype(np.uint8), .93


class FakeVideoPredictor:
    def init_state(self, path, **kwargs):
        self.path=path;self.masks={}
        return self

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.masks[obj_id]=mask

    def propagate_in_video(self,state,start_frame_idx,max_frame_num_to_track,reverse):
        import torch
        for i in range(max_frame_num_to_track):
            logits=torch.stack([self.masks[k].float()*2-1 for k in sorted(self.masks)])
            yield i, sorted(self.masks), logits


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.dataset=self.base/'data'/'task_a';self.dataset.mkdir(parents=True)
        self.out=self.base/'output'/'preprocess'/'task_a'/'sam2'
        FakeRawSpatialDataset.FRAMES={}
        for ep in range(3):
            image=np.full((120,200,3),40,np.uint8)
            x=20+ep*60
            image[65:79,x:x+13]=(0,0,255)
            image[24:49,125:161]=(0,255,0)
            FakeRawSpatialDataset.FRAMES[ep]=[image.copy() for _ in range(3)]
        rawmodule=types.ModuleType('openpi.spatial_dataset.source')
        rawmodule.RawSpatialDataset=FakeRawSpatialDataset
        self.module_patch=mock.patch.dict(sys.modules,{'openpi':types.ModuleType('openpi'),
            'openpi.spatial_dataset':types.ModuleType('openpi.spatial_dataset'),
            'openpi.spatial_dataset.source':rawmodule})
        self.module_patch.start();self.addCleanup(self.module_patch.stop)
        def extract(_,camera,ep,dest):
            task._write_jpg(dest,FakeRawSpatialDataset.FRAMES[ep][0])
        self.extract_patch=mock.patch.object(task,'extract_frame0',extract)
        self.extract_patch.start();self.addCleanup(self.extract_patch.stop)
        self.predict_patch=mock.patch.object(task,'sam_predict',side_effect=fake_sam_predict)
        self.predict_patch.start();self.addCleanup(self.predict_patch.stop)
        self.predictor_patch=mock.patch.object(task,'image_predictor',return_value=object())
        self.predictor_patch.start();self.addCleanup(self.predictor_patch.stop)

    def prepare_and_seed(self):
        task.prepare(args(dataset=self.dataset,out=self.out,objects='button,fixture',
                          camera='front',reference_episode=0,sample_count=3))
        data={'schema':task.SCHEMA,'reference_episode':0,'image_width':200,'image_height':120,
              'objects':{'button':{'positive':[[26,70]],'negative':[]},
                         'fixture':{'positive':[[140,34]],'negative':[]}}}
        task.save_json(self.out/'reference'/'prompts.json',data)
        task.seed(args(out=self.out,checkpoint=self.base/'no_gpu'))

    def test_full_workflow_and_integrity(self):
        self.prepare_and_seed()
        self.assertTrue((self.out/'reference'/'preview.jpg').is_file())
        with self.assertRaises(FileNotFoundError):
            task.firstframes(args(out=self.out,checkpoint=Path('none'),episodes='sampled',name='sampled'))
        task.approve_reference(args(out=self.out,reviewer='tester'))
        task.firstframes(args(out=self.out,checkpoint=Path('none'),episodes='sampled',name='sampled'))
        for ep in range(3):
            for name in ('button','fixture'):
                frame=task.img(self.out/'firstframes'/'source'/f'e{ep:05d}.jpg')
                mask=task.read_mask(self.out/'firstframes'/f'e{ep:05d}'/f'{name}.png',frame.shape[:2])
                self.assertGreater(int(mask.sum()),50)
        # Cannot start video before human first-frame review.
        with self.assertRaises(ValueError):
            task.video_prepare(args(out=self.out,episodes='0'))
        task.approve_firstframes(args(out=self.out,name='sampled',reviewer='tester',accept_warnings=True))
        task.video_prepare(args(out=self.out,episodes='0,2'))
        import torch
        fake_module=types.ModuleType('sam2.build_sam')
        fake_module.build_sam2_video_predictor=lambda *a,**k:FakeVideoPredictor()
        fake_parent=types.ModuleType('sam2');fake_parent.__path__=[]
        fake_ckpt=self.base/'ckpt.pt';fake_ckpt.write_bytes(b'fake')
        with mock.patch.dict(sys.modules,{'sam2':fake_parent,'sam2.build_sam':fake_module}), \
             mock.patch.object(torch.cuda,'is_available',return_value=True), \
             mock.patch.object(torch.cuda,'empty_cache'), \
             mock.patch.object(torch,'autocast',return_value=contextlib.nullcontext()):
            task.video_track(args(out=self.out,episodes='0,2',checkpoint=fake_ckpt))
        for ep in (0,2):
            for name in ('button','fixture'):
                self.assertEqual(len(list((self.out/'episodes'/f'e{ep:05d}'/'masks'/name).glob('*.png'))),3)
            self.assertEqual(task.validate_video_track(self.out,task.manifest(self.out),ep)['frames'],3)
        task.video_review(args(out=self.out,episodes='0,2',reviewer='tester',accept_warnings=True))
        # Mask changes after review are detectable.
        mask_file=self.out/'episodes'/'e00000'/'masks'/'button'/'00001.png'
        mask_file.write_bytes(b'invalidated')
        with self.assertRaises(ValueError):
            task.validate_video_track(self.out,task.manifest(self.out),0)

    def test_sampled_then_all_reuses_completed_masks(self):
        # 2 sampled episodes; third only processed in the second, full pass.
        task.prepare(args(dataset=self.dataset,out=self.out,objects='button,fixture',
                          camera='front',reference_episode=0,sample_count=2))
        data={'schema':task.SCHEMA,'reference_episode':0,'image_width':200,'image_height':120,
              'objects':{'button':{'positive':[[26,70]],'negative':[]},
                         'fixture':{'positive':[[140,34]],'negative':[]}}}
        task.save_json(self.out/'reference'/'prompts.json',data)
        task.seed(args(out=self.out,checkpoint=self.base/'no_gpu'))
        task.approve_reference(args(out=self.out,reviewer='tester'))
        task.firstframes(args(out=self.out,checkpoint=Path('none'),episodes='sampled',name='sampled'))
        kept=self.out/'firstframes'/'e00000'/'button.png'
        first_hash=task.sha(kept)
        task.approve_firstframes(args(out=self.out,name='sampled',reviewer='tester',accept_warnings=True))
        task.firstframes(args(out=self.out,checkpoint=Path('none'),episodes='all',name='all'))
        self.assertEqual(first_hash,task.sha(kept))
        self.assertTrue((self.out/'firstframes'/'e00001'/'button.png').is_file())
        task.approve_firstframes(args(out=self.out,name='all',reviewer='tester',accept_warnings=True))
        self.assertEqual(task.approved_firstframe_set(self.out,task.manifest(self.out)),{0,1,2})
        with self.assertRaises(FileExistsError):
            task.firstframes(args(out=self.out,checkpoint=Path('none'),episodes='all',name='all'))

    def test_provenance_and_no_overwrite(self):
        self.prepare_and_seed()
        with self.assertRaises(FileExistsError):
            task.prepare(args(dataset=self.dataset,out=self.out,objects='button,fixture',
                              camera='front',reference_episode=0,sample_count=3))
        task.approve_reference(args(out=self.out,reviewer='tester'))
        bad=task.load_json(self.out/'reference'/'prompts.json')
        bad['objects']['button']['positive']=[[30,70]]
        task.save_json(self.out/'reference'/'prompts.json',bad)
        with self.assertRaises(ValueError):
            task.approved_reference(self.out,task.manifest(self.out))

    def test_task_separation_and_command_selection(self):
        self.prepare_and_seed()
        self.assertEqual(task.ids_arg('0-2',task.manifest(self.out)),[0,1,2])
        with self.assertRaises(ValueError):
            task.ids_arg('0-3',task.manifest(self.out))
        with self.assertRaises(ValueError):
            task.prepare(args(dataset=self.dataset,out=self.dataset/'sam2',objects='button,fixture',
                              camera='front',reference_episode=0,sample_count=3))


if __name__=='__main__':
    unittest.main()
