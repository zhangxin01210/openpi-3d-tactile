import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/preprocess/rgbtrack_parallel.py'
if not MODULE.is_file():
    MODULE = Path(__file__).with_name('rgbtrack_parallel.py')
spec = importlib.util.spec_from_file_location('parallel', MODULE)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)

class TestScheduler(unittest.TestCase):
    def test_parse_jobs_and_fair_queue(self):
        self.assertEqual(p.parse_job('a:b:c:v1'), p.Job('a','b','c','v1'))
        for x in ['a:b:b:v1', 'a:b:v1', '../a:b:c:v1']:
            with self.assertRaises(ValueError): p.parse_job(x)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for task, n in [('one',3),('two',2)]:
                dest=root/'output/preprocess'/task/'sam2';dest.mkdir(parents=True)
                (dest/'manifest.json').write_text(json.dumps({'objects':['red','yellow'], 'episode_ids':list(range(n))}))
            jobs=[p.Job('one','red','yellow','v1'),p.Job('two','red','yellow','v1')]
            got=p.round_robin_tasks(root,jobs)
            self.assertEqual([(j.dataset,e) for j,e in got],
                             [('one',0),('two',0),('one',1),('two',1),('one',2)])

    def test_gpu_parse(self):
        old=p.gpu_stats
        try:
            p.gpu_stats=lambda:[(0,95,4200),(1,0,3),(2,4,300),(3,0,4096)]
            self.assertEqual(p.choose_gpus('auto',workers=3,util_limit=15,mem_limit=1536),[1,2])
            self.assertEqual(p.choose_gpus('3,1',workers=2,util_limit=15,mem_limit=100),[3,1])
        finally: p.gpu_stats=old

    def test_only_partial_without_provenance_is_backed_up(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            sam=root/'output/preprocess/a/sam2';sam.mkdir(parents=True)
            (sam/'manifest.json').write_text(json.dumps({'objects':['button','box'],'episode_ids':[0,1]}))
            ep_base=root/'output/preprocess/a/rgbtrack/runs/v1/episodes'
            (ep_base/'e00000').mkdir(parents=True)
            (ep_base/'e00000'/'manifest.json').write_text('{}')
            (ep_base/'e00001').mkdir()
            (ep_base/'e00001'/'task_input_provenance.json').write_text('{}')
            n=p.quarantine_partial_outputs(root,[p.Job('a','button','box','v1')],'test')
            self.assertEqual(n,1)
            self.assertFalse((ep_base/'e00000').exists())
            self.assertTrue((root/'output/preprocess/a/rgbtrack/interrupted_scheduler_test/v1/e00000/manifest.json').exists())
            self.assertTrue((ep_base/'e00001').exists())

    def test_end_to_end_two_jobs_no_gpu_required(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            dest=root/'scripts/preprocess';dest.mkdir(parents=True)
            script='''import argparse, json, os, pathlib, sys, time
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('mode');p.add_argument('--episodes',default='all');a,extras=p.parse_known_args()
if a.mode=='check': sys.exit(0)
tag=extras[extras.index('--run-tag')+1];root=pathlib.Path(__file__).resolve().parents[2]
out=root/'output/preprocess'/a.dataset/'rgbtrack/runs'/tag/'episodes'/f'e{int(a.episodes):05d}'
if (out/'task_input_provenance.json').exists():print('[SKIP VERIFIED]');sys.exit(0)
out.mkdir(parents=True,exist_ok=True)
(out/'task_input_provenance.json').write_text('ok'); print('[TRACKED]',a.dataset,a.episodes,os.environ.get('CUDA_VISIBLE_DEVICES'))
'''
            (dest/'rgbtrack_task.py').write_text(script)
            for name in ['one','two']:
                dest2=root/'output/preprocess'/name/'sam2';dest2.mkdir(parents=True)
                (dest2/'manifest.json').write_text(json.dumps({'objects':['red','yellow'],'episode_ids':[0,1,2]}))
            result=subprocess.run([sys.executable,str(MODULE),'--repo',str(root),'--job','one:red:yellow:v1',
                                   '--job','two:red:yellow:v1','--workers','2','--gpus','0,1'],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertIn('"failed": 0',result.stdout)
            for task in ['one','two']:
                for ep in range(3):
                    self.assertTrue((root/'output/preprocess'/task/'rgbtrack/runs/v1/episodes'/f'e{ep:05d}'/'task_input_provenance.json').is_file())
            again=subprocess.run([sys.executable,str(MODULE),'--repo',str(root),'--job','one:red:yellow:v1',
                                   '--workers','2','--gpus','0,1'],capture_output=True,text=True)
            self.assertEqual(again.returncode,0,again.stdout+'\n'+again.stderr)
            self.assertIn('[SKIP VERIFIED]', ''.join((x.read_text() for x in (root/'output/preprocess/rgbtrack_parallel_logs').rglob('*.log'))))

if __name__=='__main__': unittest.main(verbosity=2)
