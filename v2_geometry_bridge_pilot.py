#!/usr/bin/env python3
"""Read-only V2 -> two-object 512-point geometry pilot for HO V2 diagnostics.

Does NOT publish Object Geometry v1, alter the existing dataset, infer object
visibility/confidence, train a policy, or make a deployment-ready sidecar.
All data is written under a separate --output-dir.

Example:
 PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u v2_geometry_bridge_pilot.py \
   --checkpoint-json output/cad_pose_rgbd_v2/train_20261109_162628/best.json \
   --device cuda:0 --episodes-per-task 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import cad_pose_rgbd_single_v2 as v2

TASKS = v2.TASK_NAMES
ROLES = ('manipulated', 'target')
OBJECTS = {
    '1018_peg_all': ('funnel', 'target_flask'),
    '1028_press_button': ('button', 'box'),
    '1030_yellow_socket': ('red', 'yellow'),
    '1102_eraser': ('eraser', 'whiteboard'),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summary(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return {'n': int(len(a)), 'mean': round(float(a.mean()), 3),
            'median': round(float(np.median(a)), 3),
            'p90': round(float(np.quantile(a, .90)), 3),
            'p95': round(float(np.quantile(a, .95)), 3)}


def select_eps(cache: Path, split: str, per_task: int, overrides: str):
    splits = json.loads((cache / 'episode_splits.json').read_text())
    selected = {t: list(map(int, splits[t][split][:per_task])) for t in TASKS}
    if overrides:
        for expr in overrides.split(';'):
            if not expr.strip():
                continue
            task, sep, ids = expr.partition('=')
            if not sep or task not in TASKS:
                raise ValueError(f'Bad --episodes item {expr!r}. Use task=ep,ep;task=ep')
            selected[task] = [int(e.strip()) for e in ids.split(',') if e.strip()]
    if not all(selected.values()):
        raise ValueError('At least one episode per task required')
    return selected


def infer_episode(model, checkpoint, cache: Path, task: str, ep: int,
                  device: torch.device, batch_size: int):
    path = cache / task / f'e{ep:05d}'
    if not (path / 'meta.json').is_file():
        raise FileNotFoundError(f'Cache episode missing: {path}')
    cad = np.load(cache / task / 'cad.npy', allow_pickle=False).astype(np.float32)
    if cad.shape != (2, 512, 3):
        raise ValueError(f'{task} CAD {cad.shape}; expected (2,512,3)')
    rgb = np.load(path / 'rgb.npy', mmap_mode='r', allow_pickle=False)
    depth = np.load(path / 'depth.npy', mmap_mode='r', allow_pickle=False)
    gt = np.load(path / 'transform.npy', mmap_mode='r', allow_pickle=False)
    gt_valid = np.load(path / 'valid.npy', mmap_mode='r', allow_pickle=False)
    n = len(rgb)
    if n == 0 or depth.shape[0] != n or gt.shape != (n, 2, 4, 4) or gt_valid.shape != (n, 2):
        raise ValueError(f'{task}/e{ep}: cache size mismatch')
    if rgb.shape[1:3] != (checkpoint['height'], checkpoint['width']):
        raise ValueError(f'{task}/e{ep}: RGB resolution mismatch')
    if depth.shape[1:3] != (checkpoint['height'], checkpoint['width']):
        raise ValueError(f'{task}/e{ep}: raw depth resolution mismatch')
    # Predict both CAD queries independently, even on unlabeled frames. There
    # is no detector/visibility score here: finite output != detected object.
    pred_T = np.zeros((n, 2, 4, 4), dtype=np.float32)
    pred_T[:, :, 3, 3] = 1.0
    tic = time.monotonic()
    torch.set_grad_enabled(False)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        k = stop - start
        # B = 2*k, each frame queried separately by both single-object CADs.
        r = np.repeat(np.asarray(rgb[start:stop]), 2, axis=0).copy()
        d = np.repeat(np.asarray(depth[start:stop]), 2, axis=0).copy()
        c = np.tile(cad, (k, 1, 1))
        r_t = torch.from_numpy(r).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        d_t = torch.from_numpy(d).to(device).unsqueeze(1).float()
        d_t = (d_t * float(checkpoint['depth_scale'])).clamp_(0, 2.0) / 2.0
        c_t = torch.from_numpy(c).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == 'cuda', dtype=torch.bfloat16
        ):
            t, rot = model(r_t, d_t, c_t)
        tf = np.broadcast_to(np.eye(4, dtype=np.float32), (2 * k, 4, 4)).copy()
        tf[:, :3, :3] = rot.float().cpu().numpy()
        tf[:, :3, 3] = t.float().cpu().numpy()
        pred_T[start:stop] = tf.reshape(k, 2, 4, 4)
    pred_xyz = np.einsum('nrij,rkj->nrki', pred_T[:, :, :3, :3], cad)
    pred_xyz += pred_T[:, :, None, :3, 3]
    # Candidate roles are NOT flagged as detected/visible.
    if not np.isfinite(pred_T).all() or not np.isfinite(pred_xyz).all():
        raise ValueError(f'{task}/e{ep}: non-finite V2 prediction')
    return dict(cad=cad, pred_T=pred_T, pred_xyz=pred_xyz.astype(np.float32),
                gt_T=np.asarray(gt).copy(), gt_valid=np.asarray(gt_valid).copy(),
                seconds=time.monotonic()-tic)


def analyze(data, task, ep):
    cad = data['cad']
    pred_t = data['pred_T'][:, :, :3, 3]
    gt_t = data['gt_T'][:, :, :3, 3]
    valid = data['gt_valid']
    report = {'task': task, 'episode': ep, 'frames': len(pred_t),
              'valid_labels': {ROLES[r]: int(valid[:, r].sum()) for r in range(2)},
              'metrics_against_existing_labels_not_verified_truth': {},
              'suspect_gt_jump_with_smoother_prediction': []}
    for role in range(2):
        good = valid[:, role]
        err = np.linalg.norm(pred_t[good, role] - gt_t[good, role], axis=-1) * 1000.0
        report['metrics_against_existing_labels_not_verified_truth'][ROLES[role] + '_translation_mm'] = summary(err)
        # Only adjacent valid frames, not across missing stretches.
        adjacent = good[1:] & good[:-1]
        pd = np.linalg.norm(np.diff(pred_t[:, role], axis=0), axis=-1) * 1000.0
        gd = np.linalg.norm(np.diff(gt_t[:, role], axis=0), axis=-1) * 1000.0
        report['metrics_against_existing_labels_not_verified_truth'][ROLES[role] + '_pred_step_mm'] = summary(pd[adjacent])
        report['metrics_against_existing_labels_not_verified_truth'][ROLES[role] + '_gt_step_mm'] = summary(gd[adjacent])
        suspect = np.where(adjacent & (gd > 40) & (pd < 20))[0]
        for fid in suspect[:8]:
            report['suspect_gt_jump_with_smoother_prediction'].append({
                'role': ROLES[role], 'from_frame': int(fid), 'to_frame': int(fid + 1),
                'gt_step_mm': round(float(gd[fid]), 2),
                'pred_step_mm': round(float(pd[fid]), 2),
            })
    both = valid.all(axis=1)
    # Simple object-object relation surrogate: 3D displacement of CAD surface
    # centroids. NOT the actual trained HO encoder features/tokens.
    cad_center = cad.mean(axis=1)
    pred_centers = np.einsum('nrij,rj->nri', data['pred_T'][:, :, :3, :3], cad_center)
    pred_centers += pred_t
    gt_centers = np.einsum('nrij,rj->nri', data['gt_T'][:, :, :3, :3], cad_center)
    gt_centers += gt_t
    oo_delta = (pred_centers[:, 1] - pred_centers[:, 0]) - (gt_centers[:, 1] - gt_centers[:, 0])
    report['metrics_against_existing_labels_not_verified_truth']['object_object_centroid_relation_error_mm'] = (
        summary(np.linalg.norm(oo_delta[both], axis=-1) * 1000.0))
    # A wrong-object score is not reliable on wrong labels, but useful for flagging.
    wrong = {}
    for r in range(2):
        current = np.linalg.norm(pred_t[both, r] - gt_t[both, r], axis=-1)
        other = np.linalg.norm(pred_t[both, r] - gt_t[both, 1 - r], axis=-1)
        wrong[ROLES[r]] = {'wrong_gt_closer_count': int(np.count_nonzero(other < current)),
                           'both_labeled_frames': int(both.sum())}
    report['wrong_object_proxy_label_dependent'] = wrong
    report['inference_seconds'] = round(data['seconds'], 2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-json', type=Path, default=Path('output/cad_pose_rgbd_v2/train_20261109_162628/best.json'))
    p.add_argument('--cache-dir', type=Path, default=Path('output/cad_pose_rgbd_v2/cache'))
    p.add_argument('--output-dir', type=Path, default=Path('output/cad_pose_rgbd_v2/geometry_bridge_pilot'))
    p.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    p.add_argument('--episodes-per-task', type=int, default=1)
    p.add_argument('--episodes', default='', help='Optional: task=ep,ep;task=ep overrides defaults')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if not (1 <= a.episodes_per_task <= 20 and 1 <= a.batch_size <= 256):
        p.error('--episodes-per-task must be 1..20 and --batch-size 1..256')
    ckpt_path = Path(json.loads(a.checkpoint_json.read_text())['checkpoint']).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # trusted local checkpoint only
    if ckpt.get('schema') != v2.SCHEMA or not {'width', 'height', 'depth_scale'} <= ckpt.keys():
        raise RuntimeError('Checkpoint does not match V2 RGB-D Single-CAD contract')
    model = v2.SingleCadRGBDPose()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA but CUDA not available')
    model = model.to(device).eval()
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    chosen = select_eps(a.cache_dir, a.split, a.episodes_per_task, a.episodes)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    all_reports = []
    provenance = {
        'status': 'candidate_unaccepted_diagnostic_only',
        'note': 'NOT a published Geometry v1 sidecar, NO visibility/confidence estimation, NOT for pi0 training/deployment',
        'checkpoint': str(ckpt_path), 'checkpoint_sha256': sha256(ckpt_path),
        'cache_dir': str(a.cache_dir.resolve()), 'cache_config_sha256': sha256(a.cache_dir / 'cache_config.json'),
        'split': a.split, 'episodes': chosen,
        'coordinate_frame': 'base_link', 'length_unit': 'meter',
        'role_order': list(ROLES), 'cad_points_per_role': 512,
    }
    (a.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for task in TASKS:
        (a.output_dir / task).mkdir(parents=True, exist_ok=True)
        for ep in chosen[task]:
            data = infer_episode(model, ckpt, a.cache_dir, task, ep, device, a.batch_size)
            rep = analyze(data, task, ep)
            dest = a.output_dir / task / f'e{ep:05d}_candidate.npz'
            if dest.exists():
                raise FileExistsError(f'Will not overwrite candidate: {dest}')
            np.savez(dest,
                     frame_index=np.arange(len(data['pred_T']), dtype=np.int64),
                     pred_T_base_object=data['pred_T'],
                     candidate_xyz_m=data['pred_xyz'],
                     cad_xyz_local_m=data['cad'],
                     existing_gt_T_base_object=data['gt_T'],
                     existing_gt_label_valid=data['gt_valid'],
                     role_names=np.asarray(ROLES),
                     object_names=np.asarray(OBJECTS[task]),
                     )
            (dest.with_suffix('.json')).write_text(json.dumps(rep, indent=2) + '\n')
            all_reports.append(rep)
            stats = rep['metrics_against_existing_labels_not_verified_truth']
            trans = ', '.join(f'{r}={stats[r + "_translation_mm"].get("mean", "n/a")}mm' for r in ROLES)
            print(f'[OK] {task} e{ep:05d} frames={rep["frames"]} {trans} '
                  f'OO_error={stats["object_object_centroid_relation_error_mm"].get("mean", "n/a")}mm '
                  f'suspect_gt_jumps={len(rep["suspect_gt_jump_with_smoother_prediction"])} '
                  f'time={data["seconds"]:.1f}s', flush=True)
    (a.output_dir / 'summary.json').write_text(json.dumps({
        **provenance, 'episodes_summary': all_reports,
        'interpretation': 'GT-relative errors on potentially noisy labels; OO centroid is a geometry proxy, NOT HO V2 tokens',
    }, indent=2) + '\n')
    print('[DONE]', a.output_dir / 'summary.json', flush=True)
    print('[SAFE] Original datasets, pi0 configs, checkpoints, and object_geometry/v1 unchanged.', flush=True)


if __name__ == '__main__':
    main()
