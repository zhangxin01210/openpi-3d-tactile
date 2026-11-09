#!/usr/bin/env python3
"""Read-only V2 candidate Geometry -> HO V2 input-contract smoke test.

Checks prediction geometry availability, local ordered 512-point correspondence,
role/index alignment against native ObjectGeometryDerivedDataset v1, and source
manifest identity. No pi0 configuration is changed and NO candidate is published
as an accepted/production sidecar.

Run in the openpi-3d-tactile repo root:
 PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u v2_ho_candidate_reader_smoke.py

Requires v2_geometry_full_export.py --verify-only to have passed first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TASKS = {
    '1018_peg_all': ('funnel', 'target_flask'),
    '1028_press_button': ('button', 'box'),
    '1030_yellow_socket': ('red', 'yellow'),
    '1102_eraser': ('eraser', 'whiteboard'),
}
ROLES = ('manipulated', 'target')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class V2GeometrySample:
    episode_index: int
    frame_index: int
    manipulated_xyz_m: np.ndarray
    target_xyz_m: np.ndarray
    manipulated_mask: np.ndarray
    target_mask: np.ndarray
    # These masks mean model produced finite 512-point predictions. They are NOT
    # visibility/confidence estimates and cannot certify object detection.
    validity_semantics: str = 'finite_prediction_not_visibility'


class V2CandidateReader:
    """Read candidate NPZ without using old GT labels in prediction geometry.

    For a short QA/smoke test only: NPZ is not efficient random-access training
    storage. Native HO V2 training integration should use a memory-mapped sidecar
    or a streaming reader with persistent per-Episode caching.
    """
    def __init__(self, candidate_root: str | Path, task: str, max_cached: int = 2):
        self.root = Path(candidate_root).resolve()
        self.task = task
        if task not in TASKS:
            raise ValueError(f'unknown task: {task}')
        self.provenance = json.loads((self.root / 'provenance.json').read_text())
        self.summary = json.loads((self.root / 'summary.json').read_text())
        if self.provenance.get('status') != 'candidate_unaccepted_not_native_object_geometry':
            raise RuntimeError('unexpected V2 source status')
        if self.summary.get('status') != 'candidate_complete_unaccepted' or not self.summary.get('all_split_episodes_exported'):
            raise RuntimeError('full candidate export incomplete')
        if self.provenance.get('coordinate_frame') != 'base_link' or self.provenance.get('length_unit') != 'meter':
            raise RuntimeError('candidate coordinate units invalid')
        if self.provenance.get('role_to_object_by_task', {}).get(task) != list(TASKS[task]):
            raise RuntimeError(f'CAD role mapping changed for {task}')
        self._files = {}
        for path in (self.root / task).glob('e*_candidate.npz'):
            ep = int(path.stem[1:6])
            if ep in self._files:
                raise ValueError('duplicate episode')
            self._files[ep] = path
        self.episode_ids = tuple(sorted(self._files))
        self.cache = OrderedDict()
        self.max_cached = max_cached

    def load_episode(self, episode: int) -> dict[str, np.ndarray]:
        if episode in self.cache:
            self.cache.move_to_end(episode)
            return self.cache[episode]
        path = self._files[episode]
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        if meta.get('task') != self.task or meta.get('episode') != episode:
            raise RuntimeError(f'{path}: metadata mismatch')
        if meta.get('checkpoint_sha256') != self.provenance['checkpoint_sha256']:
            raise RuntimeError(f'{path}: checkpoint mismatch')
        # Per-episode hash costs disk IO, but this QA reader only opens a few episodes.
        if digest(path) != meta.get('file_sha256'):
            raise RuntimeError(f'{path}: sha256 mismatch')
        with np.load(path, allow_pickle=False) as a:
            keys = ['frame_index', 'candidate_xyz_m', 'pred_T_base_object',
                    'cad_xyz_local_m', 'role_names', 'object_names']
            item = {k: np.asarray(a[k]).copy() for k in keys}
        xyz, tf, cad, ids = (item[k] for k in ('candidate_xyz_m', 'pred_T_base_object',
                                                  'cad_xyz_local_m', 'frame_index'))
        n = len(ids)
        if (xyz.shape != (n, 2, 512, 3) or xyz.dtype != np.float32 or
            tf.shape != (n, 2, 4, 4) or cad.shape != (2, 512, 3) or
            not np.array_equal(ids, np.arange(n))):
            raise RuntimeError(f'{path}: shape, dtype, or frame index mismatch')
        if not np.array_equal(item['role_names'], np.array(ROLES)) or not np.array_equal(item['object_names'], np.array(TASKS[self.task])):
            raise RuntimeError(f'{path}: roles/objects changed')
        if not np.isfinite(xyz).all() or not np.isfinite(tf).all():
            raise RuntimeError(f'{path}: nonfinite candidate')
        for f in {0, n // 2, n-1}:
            xyz_check = np.einsum('rij,rkj->rki', tf[f, :, :3, :3], cad) + tf[f, :, None, :3, 3]
            if not np.allclose(xyz[f], xyz_check, atol=1e-5, rtol=0):
                raise RuntimeError(f'{path}: CAD/Pose mismatch at f{f}')
        self.cache[episode] = item
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return item

    def frame_ids(self, episode: int) -> tuple[int, ...]:
        ids = self.load_episode(episode)['frame_index']
        return tuple(int(v) for v in ids)

    def get(self, *, episode_index: int, frame_index: int) -> V2GeometrySample:
        a = self.load_episode(episode_index)
        if not (0 <= frame_index < len(a['frame_index'])) or int(a['frame_index'][frame_index]) != frame_index:
            raise IndexError('episode/frame index mismatch')
        xyz = a['candidate_xyz_m'][frame_index]
        # Explicit validity policy: model always regresses a finite pose,
        # even when the object is occluded. No GT mask is reused.
        return V2GeometrySample(
            episode_index=episode_index, frame_index=frame_index,
            manipulated_xyz_m=xyz[0].copy(), target_xyz_m=xyz[1].copy(),
            manipulated_mask=np.ones(512, dtype=bool),
            target_mask=np.ones(512, dtype=bool),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-root', default='output/cad_pose_rgbd_v2/geometry_full_candidate')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--episodes-per-task', type=int, default=2)
    args = ap.parse_args()
    if args.episodes_per_task < 1:
        ap.error('--episodes-per-task must be positive')
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    total = 0
    for task in TASKS:
        pred = V2CandidateReader(args.candidate_root, task)
        old = ObjectGeometryDerivedDataset(Path(args.data_root) / task, version='v1')
        old_eps = set(old.episode_ids)
        # The pose cache may intentionally exclude previously cleaned Episodes.
        extra = set(pred.episode_ids) - old_eps
        if extra:
            raise RuntimeError(f'{task}: predicted Episodes absent in native geometry: {sorted(extra)[:5]}')
        missing = old_eps - set(pred.episode_ids)
        # Check first/middle/last frames in a few episodes per task.
        chosen = sorted(set([pred.episode_ids[0], pred.episode_ids[len(pred.episode_ids)//2], pred.episode_ids[-1]]))[:args.episodes_per_task]
        for ep in chosen:
            ids = pred.frame_ids(ep)
            old_ids = tuple(int(x) for x in old.frame_ids(ep))
            if ids != old_ids:
                raise RuntimeError(f'{task} e{ep}: frame indexing mismatch')
            for f in sorted({0, len(ids)//2, len(ids)-1}):
                a = pred.get(episode_index=ep, frame_index=f)
                b = old.get(episode_index=ep, frame_index=f)
                for role in ROLES:
                    xyz, mask = getattr(a, role+'_xyz_m'), getattr(a, role+'_mask')
                    orig = getattr(b, role+'_xyz_m')
                    if xyz.shape != np.asarray(orig).shape or mask.shape != (512,):
                        raise RuntimeError(f'{task} e{ep} f{f} {role}: HO V2 contract mismatch')
                total += 1
        print(f'[PASS] {task}: {len(pred.episode_ids)} predicted episodes (native-only={len(missing)}); '
              f'checked {len(chosen)} episodes / {len(chosen)*3} frames; 512-point HO fields match', flush=True)
    print(f'[FINAL] PASS {total} sampled frames across four tasks. Candidate reader is usable for integration SMOKE only.', flush=True)
    print('[CAUTION] Prediction valid != object detected/visible. No GT masks used. Native sidecar still unpublished.', flush=True)


if __name__ == '__main__':
    main()
