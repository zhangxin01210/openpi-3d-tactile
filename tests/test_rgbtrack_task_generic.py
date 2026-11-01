"""CPU-only tests for the new task-to-RGBTrack handoff and guarded adapter patch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

TASK = load('rgbtrack_generic_task_test', ROOT / 'scripts/preprocess/rgbtrack_task.py')
PATCH = load('upgrade_rgbtrack_adapter_test', ROOT / 'scripts/preprocess/upgrade_rgbtrack_adapter.py')

class GenericRgbTrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.task = 'task_a'
        self.dataset = self.repo / 'data' / self.task
        self.dataset.mkdir(parents=True)
        base = self.repo / 'output/preprocess' / self.task
        self.base = base
        sam = base / 'sam2'
        geo = base / 'geometry/approved'
        sam.mkdir(parents=True)
        geo.mkdir(parents=True)
        objects = ['peg', 'socket']
        for n in objects:
            (geo / f'{n}.glb').write_bytes(f'fake-{n}'.encode())
        assets = {n: {'scale_m_per_unit': 1., 'sha256': hashlib.sha256((geo / f'{n}.glb').read_bytes()).hexdigest()} for n in objects}
        (geo / 'manifest.json').write_text(json.dumps({'status':'human_accepted','objects':objects,'dataset_id':self.task,'assets':assets}))
        (sam / 'manifest.json').write_text(json.dumps({'dataset':str(self.dataset),'objects':objects,'camera':'front','episode_ids':[2,7,11]}))
        for ep in [2,7,11]:
            eroot = sam / 'episodes' / f'e{ep:05d}'
            (eroot/'tracking_report.json').parent.mkdir(parents=True)
            mask_hashes = {}
            for n in objects:
                d = eroot/'masks'/n
                d.mkdir(parents=True)
                stream = hashlib.sha256()
                for f in range(3):
                    content = b'png'
                    (d/f'{f:05d}.png').write_bytes(content)
                    stream.update(content)
                mask_hashes[n] = stream.hexdigest()
            (eroot/'tracking_report.json').write_text(json.dumps({'status':'tracking_complete_unreviewed','episode':ep,'objects':objects,'frames':3,'mask_stream_sha256':mask_hashes}))

    def test_task_scoping_and_metric_provenance(self):
        info = TASK.task_inputs(self.repo, self.task)
        self.assertEqual(info['objects'], ['peg','socket'])
        for ep in [2,7,11]:
            root, count = TASK.verify_episode(info,ep)
            self.assertEqual(count,3)
            self.assertEqual(root.name,'masks')
        s = TASK.scale_manifest(info)
        self.assertEqual(TASK.read_json(s)['objects'],{'peg':1.0,'socket':1.0})
        self.assertEqual(TASK.select_episodes(None, info['episode_ids'], trial=True), [2,7,11])

    def test_no_acceptance_needed_and_no_cross_task(self):
        info = TASK.task_inputs(self.repo,self.task)
        self.assertEqual(TASK.verify_episode(info,2)[1],3) # no video-review.json needed
        data = TASK.read_json(self.base/'geometry/approved/manifest.json')
        data['dataset_id']='different'
        (self.base/'geometry/approved/manifest.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'not approved'):
            TASK.task_inputs(self.repo,self.task)

    def test_mesh_tamper_blocks_handoff(self):
        (self.base/'geometry/approved/peg.glb').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'changed approved mesh'):
            TASK.task_inputs(self.repo,self.task)

    def test_missing_mask_frame_blocks_handoff(self):
        (self.base/'sam2/episodes/e00002/masks/peg/00001.png').unlink()
        info = TASK.task_inputs(self.repo,self.task)
        with self.assertRaisesRegex(ValueError,'mask frame ids'):
            TASK.verify_episode(info,2)

    def test_mask_content_tamper_blocks_handoff(self):
        (self.base/'sam2/episodes/e00002/masks/peg/00001.png').write_bytes(b'changed-png')
        info = TASK.task_inputs(self.repo,self.task)
        with self.assertRaisesRegex(ValueError,'mask bytes changed'):
            TASK.verify_episode(info,2)

    def test_cmd_uses_real_object_names_and_no_new_tracker_logic(self):
        info = TASK.task_inputs(self.repo,self.task)
        mask_root, _ = TASK.verify_episode(info,2)
        cmd=TASK.build_command(info,ep=2,mask_root=mask_root,
                output=self.base/'rgbtrack/episodes/e00002',
                scale_path=self.base/'rgbtrack/scale_manifest.json',
                calibration=self.repo/'front.json',viz=False,mode='depth',max_frames=None)
        self.assertIn('peg',cmd)
        self.assertIn('socket',cmd)
        self.assertIn('--skip-preview-html',cmd)
        self.assertEqual(cmd.count('--mesh'),2)
        self.assertFalse(any(x.endswith('funnel.glb') for x in cmd))


    def test_completed_checks_mesh_mask_calibration_provenance(self):
        info = TASK.task_inputs(self.repo,self.task)
        mask_root, frames = TASK.verify_episode(info,2)
        calibration = self.repo/'front.json'
        calibration.write_text('{"version":1}')
        output = self.base/'rgbtrack/episodes/e00002'
        output.mkdir(parents=True)
        m={'episode':2, 'frames_written':frames, 'mask_root':str(mask_root),
           'mesh_conversions':{n:{'source_sha256':TASK.digest(path)} for n,path in info['meshes'].items()}}
        (output/'manifest.json').write_text(json.dumps(m))
        for name in info['objects']:
            (output/f'{name}_poses.jsonl').write_text('\n'.join('{' + '}' for _ in range(frames))+'\n')
        (output/'task_input_provenance.json').write_text(json.dumps(TASK.source_provenance(info,2,calibration)))
        self.assertTrue(TASK.completed(info,2,output,mask_root,frames,calibration))
        calibration.write_text('{"version":2}')
        with self.assertRaisesRegex(FileExistsError,'calibration changed'):
            TASK.completed(info,2,output,mask_root,frames,calibration)

    def test_guarded_patch_matches_upstream_baseline_and_keeps_legacy(self):
        original='''def build_arg_parser():
    parser.add_argument("--funnel-glb", type=Path, required=True)
    parser.add_argument("--flask-glb", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", choices=("funnel", "target_flask"), default=("funnel",))
    help="Per-object causal initialization frame; repeat for funnel/target_flask",
    parser.add_argument("--prepare-only", action="store_true", help="Convert meshes and calibration only; do not import RGBTrack")

def run(args):
    mesh_sources = {"funnel": args.funnel_glb, "target_flask": args.flask_glb}
    write_preview_html(output / "preview.html", all_records, mesh_manifests)
'''
        result = PATCH.transform(original)
        self.assertIn('"--mesh", action="append"',result)
        self.assertIn('"--skip-preview-html"',result)
        self.assertIn('"funnel": args.funnel_glb', result)
        self.assertIn('choices=', original)
        self.assertNotIn('choices=("funnel", "target_flask")',result)
        self.assertEqual(PATCH.transform(result),result)
        with self.assertRaisesRegex(RuntimeError,'Adapter changed'):
            PATCH.transform('no adapter here')

if __name__=='__main__':
    unittest.main()
