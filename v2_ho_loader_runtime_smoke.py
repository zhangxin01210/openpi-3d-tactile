#!/usr/bin/env python3
"""Real pi0 HO v2 DataLoader shadow smoke with V2 geometry (NO repo edits).

The V2 candidate reader has previously passed QA. This script patches only the
in-process ObjectGeometryDerivedDataset.get during a single DataLoader sample
read, then restores it in a finally block. It checks the real pi0 transformed
sample, and verifies that only object_geometry changes.

This is NOT a production switch or a published native sidecar; do not train using
this temporary patch. The baseline native v1 remains fully untouched.

From repo root:
 JAX_PLATFORMS=cpu PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u \
   v2_ho_loader_runtime_smoke.py
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import random
import sys
import traceback
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

from v2_ho_candidate_reader_smoke import V2CandidateReader

TASK_CONFIGS = {
    '1028_press_button': 'pi0_xhand_pc_v2_tactile_mlp_ho_v2/press_the_button',
    '1030_yellow_socket': 'pi0_xhand_pc_v2_tactile_mlp_ho_v2/yellow_socket',
}
ROLES = ('manipulated', 'target')
ATTRS = tuple(f'{role}_{kind}' for role in ROLES for kind in ('xyz_m', 'mask'))


def hash_value(val):
    """Small deterministic fingerprint of input array/dict without dumping tensors."""
    if isinstance(val, Mapping):
        return {str(k): hash_value(val[k]) for k in sorted(val, key=str)}
    if isinstance(val, (tuple, list)):
        return [hash_value(x) for x in val]
    try:
        a = np.asarray(val)
    except (TypeError, ValueError):
        return {'type': type(val).__name__, 'repr': repr(val)[:100]}
    if a.dtype.kind not in 'biufcUS':
        return {'type': type(val).__name__, 'repr': repr(val)[:100]}
    a = np.ascontiguousarray(a)
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def get_obj(item):
    if not isinstance(item, Mapping) or not isinstance(item.get('spatial'), Mapping):
        raise RuntimeError(f'transformed sample lacks spatial mapping: {type(item)}')
    obj = item['spatial'].get('object_geometry')
    if not isinstance(obj, Mapping):
        raise RuntimeError(f'transformed sample lacks spatial.object_geometry: {type(obj)}')
    for key in ATTRS:
        if key not in obj:
            raise RuntimeError(f'missing spatial.object_geometry[{key}]')
    return obj


def native_replace(native, update):
    keys = tuple(update)
    for k in keys:
        present = k in native if isinstance(native, Mapping) else hasattr(native, k)
        if not present:
            raise RuntimeError(f'native sample {type(native).__name__} lacks {k}')
    if dataclasses.is_dataclass(native):
        return dataclasses.replace(native, **update)
    if hasattr(native, '_replace'):
        return native._replace(**update)
    if isinstance(native, Mapping):
        return dict(native, **update)
    # No silent mutation or reuse of GT fields on unknown sample types.
    raise RuntimeError(f'cannot safely replace native geometry result {type(native)}; inspect src/openpi/object_geometry_dataset/derived.py')


@contextmanager
def swapped_geometry(reader):
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    orig = ObjectGeometryDerivedDataset.get
    calls = []

    def pred_get(self, *args, **kwargs):
        try:
            ep = int(kwargs.get('episode_index', args[0] if len(args) > 0 else None))
            f = int(kwargs.get('frame_index', args[1] if len(args) > 1 else None))
        except Exception as exc:
            raise RuntimeError(f'unsupported native get() call args={args!r} kwargs={kwargs!r}') from exc
        # Old sample only serves as a structural template, not geometry/masks.
        original = orig(self, *args, **kwargs)
        pred = reader.get(episode_index=ep, frame_index=f)
        update = {key: getattr(pred, key) for key in ATTRS}
        calls.append({'episode': ep, 'frame': f})
        return native_replace(original, update)

    ObjectGeometryDerivedDataset.get = pred_get
    try:
        yield calls
    finally:
        ObjectGeometryDerivedDataset.get = orig


def compare_baseline_and_v2(base, pred, task, cfgname, idx, calls):
    obj_base, obj_v2 = get_obj(base), get_obj(pred)
    if set(base) != set(pred):
        raise RuntimeError('top-level key set changed')
    other_before = {k: hash_value(v) for k, v in base.items() if k != 'spatial'}
    other_after = {k: hash_value(v) for k, v in pred.items() if k != 'spatial'}
    if other_before != other_after:
        raise RuntimeError('state/image/actions or other non-spatial fields changed (nondeterministic transformations?); do not integrate yet')
    spatial_base, spatial_v2 = base['spatial'], pred['spatial']
    if set(spatial_base) != set(spatial_v2):
        raise RuntimeError('spatial keys changed')
    unchanged = {k: hash_value(v) == hash_value(spatial_v2[k])
                 for k, v in spatial_base.items() if k != 'object_geometry'}
    if not all(unchanged.values()):
        raise RuntimeError(f'non-object Geometry spatial fields differ: {[k for k,v in unchanged.items() if not v]}')
    for key in ATTRS:
        p = np.asarray(obj_v2[key])
        e = (512, 3) if key.endswith('xyz_m') else (512,)
        if p.shape != e or not np.isfinite(p).all():
            raise RuntimeError(f'{key}: expected {e} finite; got {p.shape}')
        if key.endswith('_mask') and not p.astype(bool).all():
            raise RuntimeError(f'{key}: V2 finite prediction policy expects all-valid masks')
    if not calls:
        raise RuntimeError('DataLoader did not call patched ObjectGeometryDerivedDataset.get; V2 input NOT actually injected')
    # Some policies may pre-process positions and change geometry shape values.
    changed = [role for role in ROLES if not np.array_equal(
        np.asarray(obj_base[role+'_xyz_m']), np.asarray(obj_v2[role+'_xyz_m']))]
    if not changed:
        raise RuntimeError('DataLoader V2 geometry unchanged; patch not visibly applied')
    nongeom_obj = {k: hash_value(obj_base[k]) == hash_value(obj_v2[k])
                   for k in obj_base.keys() & obj_v2.keys() if k not in ATTRS}
    if not all(nongeom_obj.values()):
        raise RuntimeError(f'other object_geometry fields not replaced: {[k for k,v in nongeom_obj.items() if not v]}')
    result = {'task': task, 'config': cfgname, 'dataset_index': idx,
              'get_calls': calls[:12], 'get_call_count': len(calls),
              'changed_roles': changed, 'object_geometry_keys': list(obj_v2),
              'unmodified_spatial_keys': sorted(unchanged),
              'object_mask_semantics': 'finite_prediction_not_visibility',
              'state_action_rgb_tactile_scene_hand_unchanged': True}
    return result


def native_schema_info(task):
    base = Path('data') / task / 'object_geometry' / 'v1'
    if not base.is_dir():
        return {'root': str(base), 'exists': False}
    children = sorted(base.iterdir(), key=lambda p: p.name)
    manifest = base / 'manifest.json'
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    path = inspect.getsourcefile(ObjectGeometryDerivedDataset)
    result = {'root': str(base), 'exists': True,
              'top_level': [{'name': p.name, 'type': 'dir' if p.is_dir() else 'file',
                             'bytes': p.stat().st_size if p.is_file() else None} for p in children[:20]],
              'native_get_signature': str(inspect.signature(ObjectGeometryDerivedDataset.get)),
              'native_source_file': path}
    if manifest.exists():
        result['manifest_bytes'] = manifest.stat().st_size
        if manifest.stat().st_size < 1_000_000:
            try:
                m = json.loads(manifest.read_text())
                result['manifest_top_keys'] = list(m) if isinstance(m, dict) else ['non-dict']
            except Exception as ex:
                result['manifest_parse_error'] = str(ex)[:150]
    return result


def seed_sample(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-root', default='output/cad_pose_rgbd_v2/geometry_full_candidate')
    ap.add_argument('--configs', nargs='*', help='Exact training config names; default two known HO v2 configs')
    ap.add_argument('--indices', default='0,35,175', help='Dataset item indices (not frame indices)')
    ap.add_argument('--output', default='output/cad_pose_rgbd_v2/ho_v2_loader_shadow_smoke.json')
    args = ap.parse_args()
    indices = sorted(set(int(i) for i in args.indices.split(',') if i.strip()))
    from openpi.training import config, data_loader

    cfgs = args.configs or list(TASK_CONFIGS.values())
    reverse = {v: k for k, v in TASK_CONFIGS.items()}
    # Determine task from repo-id path only if using a non-default config.
    outcome = []
    layouts = {}
    for cfgname in cfgs:
        print(f'[CONFIG] {cfgname}', flush=True)
        cfg = config.get_config(cfgname)
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        dataset_path = str(getattr(data, 'repo_id', ''))
        task = reverse.get(cfgname)
        if not task:
            matches = [x for x in ('1018_peg_all', '1028_press_button', '1030_yellow_socket', '1102_eraser')
                       if x in dataset_path]
            if len(matches) != 1:
                raise RuntimeError(f'cannot infer task from repo_id={dataset_path!r} for {cfgname!r}')
            task = matches[0]
        reader = V2CandidateReader(args.candidate_root, task, max_cached=4)
        layouts[task] = native_schema_info(task)
        ds = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
        ds = data_loader.transform_dataset(ds, data)
        max_n = len(ds)
        valid_indices = [i for i in indices if 0 <= i < max_n]
        if not valid_indices:
            raise RuntimeError(f'no index in dataset of len={max_n}')
        print(f'[ROUTE] task={task} train_items={max_n} native_object_version={getattr(getattr(data,"object_geometry", None),"version",None)}', flush=True)
        for idx in valid_indices:
            seed_sample(12345 + idx)
            base = ds[idx]
            seed_sample(12345 + idx)
            with swapped_geometry(reader) as calls:
                pred = ds[idx]
            diag = compare_baseline_and_v2(base, pred, task, cfgname, idx, calls)
            outcome.append(diag)
            print(f'[PASS] {task} idx={idx} native_get_calls={len(calls)} changed={",".join(diag["changed_roles"])}; other modalities unchanged', flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'status':'pass', 'note':'in-process shadow test only, NOT a published training sidecar',
                               'items': outcome, 'native_schema': layouts}, indent=2, ensure_ascii=False)+'\n')
    print(f'[FINAL] REAL HO V2 DATALOADER SMOKE PASS {len(outcome)} samples, file={out}', flush=True)
    print('[SAFE] No repo/config/sidecar/checkpoint edits. Production integration still pending.', flush=True)


if __name__ == '__main__':
    main()
