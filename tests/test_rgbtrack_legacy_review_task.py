import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / 'scripts/preprocess/rgbtrack_legacy_review_task.py'
spec = importlib.util.spec_from_file_location('rgbtrack_legacy_review_task', SRC)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class ViewerBridgeTests(unittest.TestCase):
    def test_renders_old_viewer_labels_with_generic_names(self):
        html = "data.object_xyz.funnel data.object_xyz.target_flask 'funnel' 'target_flask' 漏斗 mesh 目标烧瓶 mesh"
        out = task.remap_original_html(html, ['button','box'])
        self.assertIn('object_xyz.button',out)
        self.assertIn('object_xyz.box',out)
        self.assertNotIn('target_flask',out)
        self.assertNotIn('funnel',out)
        self.assertIn('button mesh',out)
        self.assertIn('box mesh',out)

    def test_maps_aliases_via_existing_pose_parser(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['object_a','object_b']:
                (root/f'{name}_poses.jsonl').write_text('\n'.join(json.dumps({'frame_index':i,'object_name':name}) for i in range(3)))
            def reader(p, name):
                return {x['frame_index']:x for x in map(json.loads,Path(p).read_text().splitlines())}
            old=types.SimpleNamespace(read_poses=reader)
            result=task.load_pose_streams(old,root,['object_a','object_b'],3)
            self.assertEqual(result['funnel'][2]['object_name'],'object_a')
            self.assertEqual(result['target_flask'][1]['object_name'],'object_b')
            with self.assertRaises(ValueError):
                task.load_pose_streams(old,root,['object_a','object_b'],4)

    def test_render_remaps_legacy_names_safely_when_reused(self):
        h="funnel target_flask f.object_xyz.funnel f.object_xyz.target_flask"
        out=task.remap_original_html(h,['funnel','box'])
        self.assertIn('object_xyz.funnel',out)
        self.assertIn('object_xyz.box',out)
        self.assertNotIn('target_flask',out)

    def test_source_sha256_uses_byte_stream(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.bin';p.write_bytes(b'hello')
            self.assertEqual(task.hash_file(p),'2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824')


if __name__ == '__main__': unittest.main()
