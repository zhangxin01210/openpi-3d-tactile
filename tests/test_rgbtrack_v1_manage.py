#!/usr/bin/env python3
"""No pytest/GPU required: python3 tests/test_rgbtrack_v1_manage.py"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SRC=Path(__file__).resolve().parents[1]/'scripts/preprocess/rgbtrack_v1_manage.py'
spec=importlib.util.spec_from_file_location('manage',SRC)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

POLICY={
    'static_object':'box','dynamic_object':'button','dynamic_init_up_axis':'+z',
    'mode':'depth','dynamic_init_max_tilt_deg':25.,'guard_max_step_m':.02,
    'guard_max_turn_deg':25.,'guard_recovery_max_turn_deg':60.,
}

class ReleaseTests(unittest.TestCase):
    def test_policy(self):
        self.assertTrue(m._is_release_policy(POLICY))
        d=dict(POLICY, guard_max_step_m=.012)
        self.assertFalse(m._is_release_policy(d))
        d=dict(POLICY, dynamic_init_up_axis='-z')
        self.assertFalse(m._is_release_policy(d))

    def test_promote_and_safe_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'rgbtrack'
            tag='press_button_v1'; ids=[0,23,50]
            run=root/'runs'/tag/'episodes'
            for ep in ids:
                (run/f'e{ep:05d}').mkdir(parents=True,exist_ok=True)
            other=root/'runs'/'old_experiment';other.mkdir(parents=True)
            (other/'do_not_keep').write_text('x')
            old=root/'experiments';old.mkdir();(old/'old').write_text('x')
            reviews=root/'reviews';reviews.mkdir();(reviews/'old').write_text('x')
            # Dry run cannot change anything.
            m._promote(root,tag,ids,{},POLICY,apply=False)
            self.assertFalse((root/'release_v1.json').exists())
            m._promote(root,tag,ids,{},POLICY,apply=True)
            m._verify_published(root/'episodes',tag,ids)
            release=json.loads((root/'release_v1.json').read_text())
            self.assertEqual(release['episodes'],ids)
            m._cleanup(root,tag,ids,apply=False)
            self.assertTrue(other.exists())
            m._cleanup(root,tag,ids,apply=True)
            self.assertFalse(other.exists())
            self.assertFalse(reviews.exists())
            self.assertFalse(old.exists())
            self.assertTrue((root/'episodes'/'e00000').is_dir())
            self.assertTrue((root/'runs'/tag).exists())

    def test_deny_occupied_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'rgbtrack';(root/'episodes'/'e00000').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                m._promote(root,'press_button_v1',[0],{},POLICY,apply=True)

    def test_cleanup_needs_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                m._cleanup(Path(tmp)/'rgbtrack','press_button_v1',[0],apply=True)

if __name__=='__main__':
    unittest.main(verbosity=2)
