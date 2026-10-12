#!/usr/bin/env python3
"""Offline, episode-held-out audit of ContactWorld USB events and tactile patterns.

Physics labels are used only as targets and for selecting video examples. Model
features contain state, previously executed actions, RGB features, and tactile
readings available at the observation frame. This is descriptive analysis:
observed descent outcomes cannot certify an untried action would work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import subprocess
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from run_contactworld_history_audit import DEFAULT_OUTPUT as DEFAULT_HISTORY
from run_contactworld_sensor_discovery import DEFAULT_DATASET, fit_pca, image_paths
from run_contactworld_matched_dynamics import atomic_json, encode_frames, color_force

DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_event_sensor_atlas_20261008')
VERSION = 'cw-event-sensor-atlas-v1'
TASKS = ('socket_contact', 'blocked_after_push', 'progress_when_pushed')
MODES = ('state', 'only_rgb', 'only_fz_abs', 'only_fz_signed', 'only_fxy',
         'only_force_map', 'state_rgb', 'state_fz_abs', 'state_fz_signed',
         'state_fxy', 'state_force_summary', 'state_force_map',
         'state_force_map_history', 'state_rgb_force_map')


def load(args):
    entries = json.loads((args.history / 'episode_index.json').read_text())
    manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
    capture_root = Path(manifest['capture_root'])
    if args.smoke:
        entries = [e for split, cap in [('train', 5), ('val', 2), ('test', 2)]
                   for e in [x for x in entries if x['split'] == split][:cap]]
    fields = ('source', 'episode', 'frame', 'valid', 'state', 'rgb', 'tactile',
              'tactile_past', 'force_base', 'force_local', 'action',
              'action_history', 'relative', 'plug_vel', 'motion1',
              'contact', 'socket_force', 'grasp_pos', 'success')
    by_split = {s: {f: [] for f in fields} for s in ('train', 'val', 'test')}
    episodes = {}
    for i, e in enumerate(entries, 1):
        p = args.history / 'episodes' / f"{e['split']}_source_{e['source_episode']:03d}.npz"
        if not p.is_file():
            raise FileNotFoundError(f'Missing cached history features: {p}')
        with np.load(p) as z:
            d = {f: z[f].copy() for f in fields if f != 'success'}
        replay = json.loads((capture_root / f'insertion_usb_episode_{e["source_episode"]:03d}' / 'replay.json').read_text())
        rows = replay['rows']
        n = len(d['frame'])
        if len(rows) != e['length'] or n != e['length'] - 4:
            raise ValueError(f'Replay/cache frame mismatch for source {e["source_episode"]}')
        d['success'] = np.r_[False, [bool(r['source_success']) for r in rows[:n-1]]].astype(bool)
        for f in fields:
            by_split[e['split']][f].append(d[f])
        episodes[(e['split'], e['source_episode'])] = (e, d)
        print(f'load {i}/{len(entries)} {e["split"]} source={e["source_episode"]}', flush=True)
    joined = {s: {f: np.concatenate(parts) for f, parts in blocks.items()}
              for s, blocks in by_split.items()}
    return entries, episodes, joined


def labels(d):
    """Frame t is pre-action; contact[t] is from physics after action[t-1]."""
    n = len(d['frame'])
    valid = d['valid'].astype(bool)
    contact = d['contact'] > 0
    z = d['relative'][:, 2]
    previous_down = np.r_[False, d['action'][:-1, 2] < -2.0]
    current_down = d['action'][:, 2] < -2.0
    previous_push = valid & contact & previous_down & (z > 3)
    observed_push = valid & contact & current_down & (z > 3)
    actual_z = d['plug_vel'][:, 2]
    next_z = d['motion1'][:, 2]
    # Keep an explicit gray band. Boundary cases cannot be forced into either class.
    blocked = previous_push & (actual_z > -0.25)
    progressed = previous_push & (actual_z < -0.4)
    next_blocked = observed_push & (next_z > -0.25)
    next_progress = observed_push & (next_z < -0.4)
    if n > 0:
        previous_push[0] = blocked[0] = progressed[0] = False
    tasks = {
        'socket_contact': (valid, contact),
        'blocked_after_push': (blocked | progressed, blocked),
        'progress_when_pushed': (next_blocked | next_progress, next_progress),
    }
    # This remains observational. Only count correction followed by verified progress.
    lateral = np.linalg.norm(d['action'][:, :2], axis=1) >= 2.5
    rotation = np.linalg.norm(d['action'][:, 3:], axis=1) >= .4
    recovery = np.zeros(n, bool)
    for t in np.flatnonzero(blocked & (lateral | rotation)):
        end = min(n, t + 9)
        while end > t + 1 and d['source'][end-1] != d['source'][t]:
            end -= 1
        if end < t + 3:
            continue
        later = np.arange(t + 1, end)
        reduced_xy = np.linalg.norm(d['relative'][t, :2]) - np.min(
            np.linalg.norm(d['relative'][later, :2], axis=1))
        actual_progress = d['relative'][t, 2] - np.min(d['relative'][later, 2])
        recovery[t] = reduced_xy > .5 and actual_progress > 1.0
    # First-ever contact for each complete demonstration. Brief contact-force
    # dropouts are re-contact events, not new first-contact events.
    onset = np.zeros(n, bool)
    for source in np.unique(d['source']):
        first = np.flatnonzero((d['source'] == source) & valid & contact)
        if len(first): onset[first[0]] = True
    success = d['success'].astype(bool) & valid
    success_onset = np.zeros(n, bool)
    for source in np.unique(d['source']):
        first = np.flatnonzero((d['source'] == source) & success)
        if len(first): success_onset[first[0]] = True
    phases = {'contact': contact & valid, 'contact_onset': onset,
              'success_reached': success, 'completion_onset': success_onset, 'blocked': blocked,
              'progress': progressed, 'down_command': observed_push,
              'gray_previous': previous_push & ~(blocked | progressed),
              'gray_next': observed_push & ~(next_blocked | next_progress),
              'lateral_command': contact & valid & lateral,
              'rotation_command': contact & valid & rotation,
              'recovery_candidate': recovery}
    return tasks, phases


def all_labels(data):
    out = {}
    for split, d in data.items():
        out[split] = labels(d)
    return out


def force_blocks(d):
    f = d['force_base']
    mean = f.mean(axis=2)
    absmean = np.abs(f).mean(axis=2)
    std = f.std(axis=2)
    area = (np.linalg.norm(f, axis=-1) > 1e-4).mean(axis=2)
    # Explicit left/right order is retained. No GT object pose in predictors.
    return {
        'fz_abs': absmean[:, :, 2].reshape(len(f), -1),
        'fz_signed': np.concatenate((mean[:, :, 2], absmean[:, :, 2]), axis=1),
        'fxy': np.concatenate((mean[:, :, :2].reshape(len(f), -1),
                              absmean[:, :, :2].reshape(len(f), -1)), axis=1),
        'summary': np.concatenate((mean.reshape(len(f), -1),
                                   absmean.reshape(len(f), -1),
                                   std.reshape(len(f), -1), area), axis=1),
    }


def prepare_features(data):
    train = data['train']; good = train['valid'].astype(bool)
    pca = {}
    for name, width, seed in [('rgb', 48, 20261008), ('tactile', 48, 20261009)]:
        pca[name] = fit_pca(train[name][good], width, seed)
    raw = {}; normalized = {}
    for split, d in data.items():
        f = force_blocks(d)
        raw[split] = {
            'state': np.concatenate((d['state'], d['action_history'].reshape(len(d['frame']), -1)), axis=1),
            'rgb': (d['rgb'] - pca['rgb'][0]) @ pca['rgb'][1],
            'fz_abs': f['fz_abs'], 'fz_signed': f['fz_signed'],
            'fxy': f['fxy'], 'summary': f['summary'],
            'force_map': (d['tactile'] - pca['tactile'][0]) @ pca['tactile'][1],
            'force_history': d['tactile_past'],
            'current_action': d['action'],
        }
    for key in raw['train']:
        reference = raw['train'][key][good]
        mu = reference.mean(axis=0); sigma = reference.std(axis=0)
        sigma = np.where(sigma < 1e-8, 1, sigma)
        for split in data:
            normalized.setdefault(split, {})[key] = np.clip((raw[split][key] - mu) / sigma, -8, 8).astype(np.float32)
    return normalized


def design(blocks, task, mode):
    names = [] if mode.startswith('only_') else ['state']
    if 'rgb' in mode: names.append('rgb')
    if 'fz_abs' in mode: names.append('fz_abs')
    if 'fz_signed' in mode: names.append('fz_signed')
    if 'fxy' in mode: names.append('fxy')
    if 'summary' in mode: names.append('summary')
    if 'force_map' in mode: names.append('force_map')
    if 'history' in mode: names.append('force_history')
    if task == 'progress_when_pushed': names.append('current_action')
    # Equalize blocks so high-dimensional RGB/map do not dominate by default.
    return np.concatenate([blocks[k] / np.sqrt(blocks[k].shape[1]) for k in names] +
                          [np.ones((len(blocks['state']), 1), np.float32)], axis=1)


def logistic_fit(x, y, strength):
    y = y.astype(np.float64)
    base = np.clip(y.mean(), 1e-4, 1-1e-4)
    initial = np.zeros(x.shape[1], np.float64)
    initial[-1] = np.log(base / (1 - base))
    def objective(w):
        z = x @ w
        p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        loss = np.logaddexp(0, z).sum() - y @ z + .5*strength*(w[:-1] @ w[:-1])
        grad = x.T @ (p-y)
        grad[:-1] += strength*w[:-1]
        return loss, grad
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', options={'maxiter': 160, 'ftol': 1e-8})
    return result.x


def probability(x, w):
    return 1 / (1 + np.exp(-np.clip(x @ w, -50, 50)))


def binary_metrics(y, p, threshold):
    y = np.asarray(y, bool); p = np.asarray(p)
    safe = np.clip(p, 1e-6, 1-1e-6)
    pred = p >= threshold
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum()); fn = int((~pred & y).sum())
    return dict(n=len(y), positives=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
                sensitivity=tp/max(1,tp+fn), specificity=tn/max(1,tn+fp),
                balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),
                precision=tp/max(1,tp+fp), brier=float(np.mean((p-y)**2)),
                logloss=float(-np.mean(y*np.log(safe)+(1-y)*np.log(1-safe))))


def threshold_from_val(y, p):
    best = (.5, -.1)
    for t in np.linspace(.1, .9, 33):
        score = binary_metrics(y, p, float(t))['balanced_accuracy']
        if score > best[1]: best = (float(t), score)
    return best[0]


def episode_bootstrap(y, p, sources, threshold, seed=42, repeats=300):
    unique = np.unique(sources); rng = np.random.default_rng(seed)
    if len(unique) < 2: return None
    values = []
    for _ in range(repeats):
        chosen = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([np.flatnonzero(sources == s) for s in chosen])
        if len(np.unique(y[ids])) < 2: continue
        values.append(binary_metrics(y[ids], p[ids], threshold)['balanced_accuracy'])
    return [float(x) for x in np.percentile(values, [2.5, 97.5])] if values else None


def fit_probes(args, data, labelset, blocks):
    out = {}; saved = {}
    for task in TASKS:
        out[task] = {}; saved[task] = {}
        masks = {s: labelset[s][0][task][0] for s in data}
        yy = {s: labelset[s][0][task][1][masks[s]].astype(np.float64) for s in data}
        if min(len(yy[s]) for s in data) == 0:
            raise ValueError(f'No eligible frames for {task}; inspect event thresholds')
        for mode in MODES:
            xx = {s: design(blocks[s], task, mode)[masks[s]].astype(np.float64) for s in data}
            candidates = []
            for strength in (0.1, 1., 10., 100.):
                w = logistic_fit(xx['train'], yy['train'], strength)
                pv = probability(xx['val'], w)
                # Validation selects regularization; test is never used for tuning.
                score = binary_metrics(yy['val'], pv, .5)['logloss']
                candidates.append((score, strength, w, pv))
            _, strength, weight, pv = min(candidates, key=lambda x: x[0])
            threshold = threshold_from_val(yy['val'], pv)
            pt = probability(xx['test'], weight)
            metric = binary_metrics(yy['test'], pt, threshold)
            sources = data['test']['source'][masks['test']]
            metric['episode_bootstrap_balanced_accuracy_95'] = episode_bootstrap(yy['test'], pt, sources, threshold)
            metric.update(mode=mode, regularization=strength, threshold=threshold,
                          val_logloss=float(binary_metrics(yy['val'], pv, threshold)['logloss']))
            out[task][mode] = metric
            saved[task][mode] = {'y': yy['test'].astype(np.int8), 'p': pt.astype(np.float32),
                                 'source': sources, 'frame': data['test']['frame'][masks['test']]}
            print(f'probe {task} {mode} n={metric["n"]} BA={metric["balanced_accuracy"]:.3f}', flush=True)
    atomic_json(args.output/'probes.json', out)
    (args.output/'predictions').mkdir(exist_ok=True)
    for task in TASKS:
        np.savez_compressed(args.output/'predictions'/f'{task}.npz', **{
            f'{mode}_{key}': value for mode, d in saved[task].items() for key, value in d.items()})
    return out, saved


def cluster_features(data):
    out = {}
    for split, d in data.items():
        f = d['tactile'].reshape(len(d['frame']), 2, 140, 3)
        # Separate total response from normalized spatial pattern. Near-zero
        # maps are not assigned an arbitrary pattern; they form an explicit bin.
        rms = np.sqrt(np.mean(f*f, axis=(1,2,3)))
        pattern = (f / np.maximum(rms[:, None, None, None], .02)).reshape(len(f), -1)
        local = d['force_local'].reshape(len(f), -1)
        local_rms = np.sqrt(np.mean(local*local, axis=1))
        local_pattern = local / np.maximum(local_rms[:, None], 2e-5)
        out[split] = {'rms': rms, 'pattern': pattern, 'local_pattern': local_pattern,
                      'summary': force_blocks(d)['summary'],
                      'history': d['tactile_past']}
    return out


def kmeans(x, k, seed=42, restarts=5, iterations=60):
    rng = np.random.default_rng(seed); best = None
    for _ in range(restarts):
        centers = x[rng.choice(len(x), k, replace=False)].copy()
        for _ in range(iterations):
            distances = cdist(x, centers, 'sqeuclidean')
            assignments = distances.argmin(axis=1)
            new = np.stack([x[assignments == j].mean(0) if (assignments == j).any()
                            else x[rng.integers(len(x))] for j in range(k)])
            if np.max(np.abs(new-centers)) < 1e-4: break
            centers = new
        inertia = float(np.min(cdist(x, centers, 'sqeuclidean'), axis=1).mean())
        if best is None or inertia < best[0]: best = (inertia, centers)
    return best[1]


def fit_clusters(args, data, labelset):
    cb = cluster_features(data)
    train_mask = labelset['train'][1]['contact'] & data['train']['valid'] & (data['train']['relative'][:,2] > 3)
    train_rms = cb['train']['rms'][train_mask]
    low = float(max(.025, np.percentile(train_rms, 10)))
    train_active = train_mask & (cb['train']['rms'] >= low)
    pca_mean, pca_proj = fit_pca(cb['train']['pattern'][train_active], 24, 77)
    local_mean, local_proj = fit_pca(cb['train']['local_pattern'][train_active], 24, 78)
    transforms = {}
    for name in ('amplitude', 'pattern', 'local_pattern', 'pattern_history'):
        transforms[name] = {}
        for split in data:
            b = cb[split]
            if name == 'amplitude': x = b['summary']
            elif name == 'local_pattern': x = (b['local_pattern'] - local_mean) @ local_proj
            else:
                x = (b['pattern'] - pca_mean) @ pca_proj
                if name == 'pattern_history': x = np.concatenate((x, b['history']), axis=1)
            transforms[name][split] = x.astype(np.float32)
        ref = transforms[name]['train'][train_active]
        mean = ref.mean(0); scale = np.where(ref.std(0) < 1e-5, 1., ref.std(0))
        for split in data:
            transforms[name][split] = np.clip((transforms[name][split]-mean)/scale, -6, 6) / np.sqrt(len(mean))
    results = {}
    for name, splits in transforms.items():
        centers = kmeans(splits['train'][train_active], 6, seed=601)
        result = {'low_response_rms_cutoff': low, 'k': 6, 'splits': {}}
        for split in data:
            d = data[split]; b = cb[split]
            eligible = labelset[split][1]['contact'] & (d['relative'][:,2] > 3) & d['valid']
            distance = cdist(splits[split], centers, 'sqeuclidean')
            assigned = distance.argmin(axis=1).astype(np.int16)
            assigned[b['rms'] < low] = -1
            assigned[~eligible] = -2
            np.save(args.output/'predictions'/f'cluster_{name}_{split}.npy', assigned)
            group = []
            for k in [-1, *range(6)]:
                mask = assigned == k
                if not mask.any(): continue
                blocked = labelset[split][1]['blocked'][mask]
                progress = labelset[split][1]['progress'][mask]
                lateral = labelset[split][1]['lateral_command'][mask]
                recovery = labelset[split][1]['recovery_candidate'][mask]
                down = d['action'][mask,2] < -2
                next_z = d['motion1'][mask,2]
                down_progress = down & (next_z < -.4)
                down_block = down & (next_z > -.25)
                group.append(dict(id=k, frames=int(mask.sum()), demonstrations=int(len(np.unique(d['source'][mask]))),
                    down_trials=int(down.sum()), down_progress=int(down_progress.sum()),
                    down_block=int(down_block.sum()),
                    down_progress_fraction=float(down_progress.sum()/max(1,down_progress.sum()+down_block.sum())),
                    mean_rms=float(b['rms'][mask].mean()), blocked_fraction=float(blocked.mean()),
                    progress_fraction=float(progress.mean()), lateral_command_fraction=float(lateral.mean()),
                    recovery_candidate_fraction=float(recovery.mean()),
                    mean_next_motion_xy_mm=d['motion1'][mask,:2].mean(0).tolist(),
                    mean_next_motion_z_mm=float(d['motion1'][mask,2].mean())))
            result['splits'][split] = group
        results[name] = result
        print(f'cluster {name}: train active={train_active.sum()} test groups={len(result["splits"]["test"])}', flush=True)
    atomic_json(args.output/'clusters.json', results)
    return results


def event_summary(args, data, labelset):
    results = {}
    for split, d in data.items():
        groups = labelset[split][1]
        results[split] = {'demonstrations': int(len(np.unique(d['source']))),
                          'frames': int(d['valid'].sum()),
                          'events': {k: int(v.sum()) for k,v in groups.items()},
                          'demonstrations_with_events': {k:int(len(np.unique(d['source'][v])))
                           for k,v in groups.items()}}
    atomic_json(args.output/'events.json', results)
    return results


def select_cases(args, episodes, labelset, data):
    selected=[]
    for split, source, e, d in [(s,src,*pair) for (s,src),pair in episodes.items()]:
        if split != 'test' and not args.smoke: continue
        _, phase = labels(d)
        for category in ('contact_onset','blocked','progress','recovery_candidate','completion_onset','lateral_command'):
            candidates=np.flatnonzero(phase[category] & (d['relative'][:,2]>3))
            if len(candidates)==0:continue
            # First event in each episode prevents selecting force extrema.
            t=int(candidates[0]);selected.append(dict(category=category,split=split,source=source,
                episode=e['episode_index'],frame=t,contact=int(d['contact'][t]),
                relative_mm=d['relative'][t].round(2).tolist(),
                previous_action=d['action_history'][t,0].round(2).tolist(),
                current_action=d['action'][t].round(2).tolist(),
                previous_plug_delta_mm=d['plug_vel'][t].round(2).tolist(),
                next_plug_delta_mm=d['motion1'][t].round(2).tolist()))
    # Balance event types; source de-duplication within each category.
    taken=[]
    per=max(1,args.max_cases//6)
    for category in ('contact_onset','blocked','progress','recovery_candidate','completion_onset','lateral_command'):
        candidates=[x for x in selected if x['category']==category]
        for x in candidates[:per]:taken.append(x)
    return taken[:args.max_cases]


def choose_cluster_cases(args, episodes, data):
    """Within each tactile cluster, seek an action-matched opposite-outcome pair."""
    result=[]
    assign=np.load(args.output/'predictions/cluster_pattern_test.npy')
    d=data['test']
    rms=np.sqrt(np.mean(d['tactile']**2,axis=1))
    pattern=d['tactile']/np.maximum(rms[:,None],.02)
    for cluster in (-1,*range(6)):
        base=(assign==cluster) & d['valid'] & (d['action'][:,2] < -2)
        good=np.flatnonzero(base & (d['motion1'][:,2]<-.4))
        bad=np.flatnonzero(base & (d['motion1'][:,2]>-.25))
        if not len(good) and not len(bad):continue
        chosen=[];pair_note='single outcome in this cluster'
        if len(good) and len(bad):
            # A joint tactile/action match prevents comparing a mild descent
            # with a completely different push. Prefer distinct demonstrations.
            a=good[:300];b=bad[:300]
            distance=cdist(pattern[a],pattern[b],'euclidean')/np.sqrt(pattern.shape[1])
            cross_source=d['source'][a,None]!=d['source'][None,b]
            z_gap=np.abs(d['action'][a,None,2]-d['action'][None,b,2])
            xy_gap=np.linalg.norm(d['action'][a,None,:2]-d['action'][None,b,:2],axis=2)
            gated=cross_source & (z_gap<=3) & (xy_gap<=4)
            pose_xy=cdist(d['relative'][a,:2],d['relative'][b,:2])
            pose_z=np.abs(d['relative'][a,None,2]-d['relative'][None,b,2])
            grasp_gap=cdist(d['grasp_pos'][a],d['grasp_pos'][b])
            pose_gated=gated & (pose_xy<=1.5) & (pose_z<=2.5) & (grasp_gap<=5)
            if gated.any():
                pair_level='also near in pose/grasp' if pose_gated.any() else 'action-matched only; poses differ'
                allowed=pose_gated if pose_gated.any() else gated
                candidate=np.where(allowed,distance,np.inf)
                i,j=np.unravel_index(candidate.argmin(),candidate.shape)
                chosen=[int(a[i]),int(b[j])]
                pair_note=(f'{pair_level}; tactile distance {distance[i,j]:.3f}; '
                           f'pose XY gap {pose_xy[i,j]:.2f} mm, Z gap {pose_z[i,j]:.2f} mm, '
                           f'grasp gap {grasp_gap[i,j]:.2f} mm; '
                           f'command XY gap {xy_gap[i,j]:.2f} mm, Z gap {z_gap[i,j]:.2f} mm')
            else:
                pair_note='no cross-episode pair with matched down command; clips are illustrative only'
        if not chosen:
            chosen=[int(x[len(x)//2]) for x in (good,bad) if len(x)]
        for idx in chosen:
            source=int(d['source'][idx]);frame=int(d['frame'][idx]);e,_=episodes[('test',source)]
            result.append(dict(category=f'pattern cluster {cluster}',split='test',source=source,
                               episode=e['episode_index'],frame=frame,
                               next_z_mm=float(d['motion1'][idx,2]),
                               relative_mm=d['relative'][idx].round(2).tolist(),
                               current_action=d['action'][idx].round(2).tolist(),
                               pair_note=pair_note,pair_group=f'cluster {cluster}'))
    return result


def video_frames(path, indices):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise FileNotFoundError(path)
    output=[]
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=cap.read()
            if not ok:raise RuntimeError(f'{path}: cannot read frame {i}')
            output.append(cv2.resize(frame,(320,320),interpolation=cv2.INTER_AREA))
    finally:cap.release()
    return output


def compose(front,wrist,tac_left,tac_right,d,t,scale,category,marked_frame):
    canvas=np.full((730,1200,3),26,np.uint8)
    canvas[50:370,0:320]=front;canvas[50:370,320:640]=wrist
    cv2.putText(canvas,'FRONT', (5,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(250,250,250),2)
    cv2.putText(canvas,'WRIST', (330,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(250,250,250),2)
    canvas[405:725,0:320]=tac_left;canvas[405:725,320:640]=tac_right
    cv2.putText(canvas,'LEFT TACTILE PREVIEW',(5,390),cv2.FONT_HERSHEY_SIMPLEX,.58,(250,250,250),1)
    cv2.putText(canvas,'RIGHT TACTILE PREVIEW',(330,390),cv2.FONT_HERSHEY_SIMPLEX,.58,(250,250,250),1)
    f=d['force_base'][t]
    for finger in range(2):
        for axis in range(3):
            x=645+axis*185;y=50+finger*160
            image=color_force(f[finger,:,axis].reshape(10,14),scale[axis])
            canvas[y+24:y+164,x:x+180]=cv2.resize(image,(180,140),interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas,f"{'L' if finger==0 else 'R'} F{'xyz'[axis]}",(x,y+17),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(250,250,250),1)
    _,ph=labels(d)
    rel=d['relative'][t]
    prev=d['action_history'][t,0,:3]
    current=d['action'][t,:3]
    fields=[
        f'CASE {category} | event {marked_frame} | shown {t} ({t-marked_frame:+d})',
        f"socket {int(d['contact'][t])} | blocked {int(ph['blocked'][t])} | descent {int(ph['progress'][t])} | success {int(d['success'][t])}",
        f'plug relative mm: x {rel[0]:+.1f} y {rel[1]:+.1f} z {rel[2]:+.1f}',
        f"last plug dz {d['plug_vel'][t,2]:+.2f} | next dz {d['motion1'][t,2]:+.2f}",
        f'previous cmd xyz: {prev[0]:+.1f}, {prev[1]:+.1f}, {prev[2]:+.1f}',
        f'current cmd xyz: {current[0]:+.1f}, {current[1]:+.1f}, {current[2]:+.1f}',
        'BASE TacFF: red positive, blue negative; common scale',
    ]
    for i,line in enumerate(fields):
        cv2.putText(canvas,line,(650,395+i*43),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
    return canvas


def timeline(d,center,path):
    lo=max(0,center-8);hi=min(len(d['frame']),center+13);x=np.arange(lo,hi)-center
    f=d['force_base'][lo:hi]
    fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True,layout='constrained')
    colors=['#dc4849','#2196ae','#7745ac']
    for hand in range(2):
        for j in range(3):
            axes[0].plot(x,f[:,hand,:,j].mean(axis=1),color=colors[j],ls='-' if hand==0 else '--',
                         label=f"{'L' if hand==0 else 'R'} F{'xyz'[j]}")
    axes[0].legend(ncol=6,fontsize=7);axes[0].set_ylabel('mean TacFF')
    for j in range(3):
        axes[1].plot(x,d['action'][lo:hi,j],color=colors[j],label=f'cmd {"xyz"[j]}')
        axes[2].plot(x,d['motion1'][lo:hi,j],color=colors[j],label=f'actual plug Δ{"xyz"[j]}')
    axes[1].legend(ncol=3,fontsize=8);axes[2].legend(ncol=3,fontsize=8)
    axes[1].set_ylabel('command mm/step');axes[2].set_ylabel('next displacement mm/step')
    axes[2].set_xlabel('control steps relative to marked frame')
    for ax in axes:ax.axvline(0,color='black',ls=':');ax.grid(alpha=.2)
    fig.savefig(path,dpi=125);plt.close(fig)


def render_cases(args, episodes, data, labels_all):
    cases=select_cases(args,episodes,labels_all,data)+choose_cluster_cases(args,episodes,data)
    out=args.output/'cases';out.mkdir(exist_ok=True)
    train=data['train'];valid=train['valid'].astype(bool)
    scale=np.quantile(np.abs(train['force_base'][valid]),.995,axis=(0,1,2))
    scale=np.maximum(scale,1e-6)
    for i,case in enumerate(cases):
        e,d=episodes[(case['split'],case['source'])]
        t=case['frame'];lo=max(0,t-8);hi=min(len(d['frame']),t+13);indices=list(range(lo,hi))
        paths=image_paths(args.dataset,e['episode_index'])
        front=video_frames(paths['front'],indices);wrist=video_frames(paths['wrist'],indices)
        tac_left=video_frames(paths['tacff_left_preview'],indices)
        tac_right=video_frames(paths['tacff_right_preview'],indices)
        stem=f'{i:02d}_{case["category"].replace(" ","_")}_source{case["source"]:03d}_f{t:03d}'
        movie=out/f'{stem}.mp4';plot=out/f'{stem}.png'
        if not movie.exists():
            encode_frames(movie,1200,730,(compose(a,b,c,h,d,j,scale,case['category'],t)
                        for a,b,c,h,j in zip(front,wrist,tac_left,tac_right,indices)),fps=10)
        # The poster is the marked event frame, not the pre-event first frame.
        cv2.imwrite(str(movie.with_suffix('.jpg')),compose(front[t-lo],wrist[t-lo],
                    tac_left[t-lo],tac_right[t-lo],d,t,scale,case['category'],t))
        if not plot.exists():timeline(d,t,plot)
        case['movie']=f'cases/{stem}.mp4';case['timeline']=f'cases/{stem}.png'
        case['source_tacff_left']=str(paths['tacff_left_preview'])
        case['source_tacff_right']=str(paths['tacff_right_preview'])
        case['source_front']=str(paths['front']);case['source_wrist']=str(paths['wrist'])
        print(f'render {i+1}/{len(cases)} {stem}',flush=True)
    atomic_json(args.output/'cases.json',cases)
    return cases


def plots(args, events, probes, clusters):
    figdir=args.output/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for i,task in enumerate(TASKS):
        names=list(probes[task]); values=[probes[task][m]['balanced_accuracy'] for m in names]
        axes[i].barh(names,values,color='#4188b4');axes[i].set_xlim(0,1);axes[i].axvline(.5,color='gray',ls=':')
        axes[i].set_title(task.replace('_',' '));axes[i].invert_yaxis()
    fig.savefig(figdir/'probe_balanced_accuracy.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,3.6),layout='constrained')
    for ax,name in zip(axes,('amplitude','pattern','local_pattern','pattern_history')):
        rows=clusters[name]['splits']['test'];names=[str(r['id']) for r in rows]
        ax.bar(names,[r['down_progress_fraction'] for r in rows],label='next descent progresses')
        ax.set_ylabel('among actual downward pushes')
        ax.set_title(name);ax.set_ylim(0,1);ax.set_xlabel('cluster (-1 = weak force)')
    axes[0].legend(fontsize=7);fig.savefig(figdir/'cluster_outcomes.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    names=list(events['test']['events']);vals=[events['test']['events'][n] for n in names]
    ax.bar(names,vals,color='#b46a42');ax.tick_params(axis='x',rotation=35);ax.set_ylabel('test frames')
    fig.savefig(figdir/'test_event_coverage.png',dpi=145);plt.close(fig)



def diagnostic_plots(args, episodes, data, labelset):
    """Descriptive force overlap and event-aligned timing; no frame IID claims."""
    out=args.output/'figures';out.mkdir(exist_ok=True)
    d=data['test']; phase=labelset['test'][1]
    mean=d['force_base'].mean(axis=2)
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for hand in range(2):
        for axis in range(3):
            ax=axes[hand,axis]
            a=mean[phase['blocked'],hand,axis]
            b=mean[phase['progress'],hand,axis]
            if len(a):ax.hist(a,bins=28,density=True,alpha=.48,color='#d1495b',label=f'blocked n={len(a)}')
            if len(b):ax.hist(b,bins=28,density=True,alpha=.48,color='#30a88e',label=f'progress n={len(b)}')
            ax.set_title(f"{'Left' if hand==0 else 'Right'} F{'xyz'[axis]} signed mean")
            ax.set_xlabel('base force (simulation units)');ax.legend(fontsize=8)
    fig.savefig(out/'signed_force_overlap.png',dpi=145);plt.close(fig)
    windows={name:[] for name in ('contact_onset','blocked')}
    for (split,_),(_,ep) in episodes.items():
        if split!='test':continue
        _,ph=labels(ep)
        n=len(ep['frame']); force=ep['force_base']
        channels=np.stack((np.abs(force[:,:,:,2]).mean(axis=(1,2)),
            force[:,:,:,2].mean(axis=(1,2)),ep['action'][:,2],ep['plug_vel'][:,2]),axis=1)
        for name in windows:
            starts=np.flatnonzero(ph[name] & ~np.r_[False,ph[name][:-1]])
            for t in starts:
                if t>=5 and t+6<=n:
                    windows[name].append(channels[t-5:t+6])
    fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True,layout='constrained')
    titles=('mean |Fz| of both fingers','signed mean Fz','current command dz','actual previous-step plug dz')
    for row,name in enumerate(windows):
        if not windows[name]:continue
        arr=np.stack(windows[name]); x=np.arange(-5,6)
        for j,ax in enumerate(axes[row]):
            med=np.median(arr[:,:,j],axis=0)
            lo,hi=np.percentile(arr[:,:,j],[25,75],axis=0)
            ax.plot(x,med,color='#407eb4');ax.fill_between(x,lo,hi,alpha=.23,color='#407eb4')
            ax.axvline(0,color='#ba503d',ls=':');ax.grid(alpha=.2)
            ax.set_title(f'{name} n={len(arr)}\n{titles[j]}',fontsize=9)
    fig.savefig(out/'event_aligned_signals.png',dpi=145);plt.close(fig)
    return {k:len(v) for k,v in windows.items()}

def page(args,events,probes,clusters,cases):
    def table(task):
        rows=[]
        for mode,m in probes[task].items():
            ci=m['episode_bootstrap_balanced_accuracy_95']
            interval=f'{ci[0]:.2f}–{ci[1]:.2f}' if ci else 'n/a'
            rows.append(f'<tr><td>{html.escape(mode)}</td><td>{m["n"]} / {m["positives"]}</td><td>{m["balanced_accuracy"]:.3f} [{interval}]</td><td>{m["sensitivity"]:.2f}</td><td>{m["specificity"]:.2f}</td><td>{m["precision"]:.2f}</td><td>{m["tp"]}/{m["fp"]}/{m["tn"]}/{m["fn"]}</td></tr>')
        return '<table><tr><th>Input</th><th>N / positive</th><th>Balanced accuracy [episode bootstrap 95%]</th><th>Recall</th><th>Specificity</th><th>Precision</th><th>TP/FP/TN/FN</th></tr>'+''.join(rows)+'</table>'
    cluster_rows=[]
    for name,record in clusters.items():
        for row in record['splits']['test']:
            cluster_rows.append(f'<tr><td>{name}</td><td>{row["id"]}</td><td>{row["frames"]}</td><td>{row["demonstrations"]}</td><td>{row["mean_rms"]:.3f}</td><td>{row["down_trials"]}</td><td>{row["down_progress"]}/{row["down_block"]}</td><td>{row["down_progress_fraction"]:.2f}</td><td>{row["lateral_command_fraction"]:.2f}</td></tr>')
    cards=[]
    for c in cases:
        if 'pair_group' in c:
            continue
        caption=f"{c['category']} | {c['split']} source {c['source']} | LeRobot episode {c['episode']} | frame {c['frame']}"
        note=f'<p>{html.escape(c["pair_note"])}</p>' if 'pair_note' in c else ''
        cards.append(f'<article><h3>{html.escape(caption)}</h3>{note}<video controls preload="metadata" poster="{html.escape(c["movie"].replace(".mp4",".jpg"))}" src="{html.escape(c["movie"])}"></video><img src="{html.escape(c["timeline"])}" loading="lazy"><details><summary>原始双视图文件与逐帧记录</summary><p>{html.escape(str(c))}</p></details></article>')
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ContactWorld 事件与触觉模式图谱</title>
<style>body{{font:16px/1.55 system-ui;background:#111827;color:#e7eef8;max-width:1450px;margin:auto;padding:26px}}a{{color:#91c9ff}}h1,h2{{color:#eff7ff}}section,article{{background:#1d293b;padding:20px;margin:20px 0;border-radius:12px}}video{{width:100%;max-width:1200px;display:block}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #4a5871;padding:6px;text-align:left}}th{{background:#30425d}}code{{color:#bad5f8}}.warn{{background:#493823;padding:12px;border-radius:8px}}</style>
<h1>ContactWorld USB：事件、可见信息和双指触觉模式</h1>
<p>先看事件定义，再看各输入能否识别事件，最后用同步原始视频和力图检查支持例及反例。数据划分：{events['train']['demonstrations']} train / {events['val']['demonstrations']} val / {events['test']['demonstrations']} test 条完整示范。每个探针只在 train 拟合，val 选参数与阈值，test 只用于报告。</p>
<section><h2>1. 标签到底表示什么</h2><p>每帧观测在控制动作执行前。插座接触来自上一步仿真碰撞力，不用触觉定义。完成状态来自仿真重放记录的 <code>source_success</code>，映射到下一帧观测；缓存只覆盖每条轨迹的前 <code>length−4</code> 帧，晚于该窗口的完成不会出现在逐帧统计里。<b>刚才下推后受阻</b>：上一步轴向指令 &lt; −2 mm/步，插件实际向下位移 &gt; −0.25 mm，已经接触且轴向距离 &gt; 3 mm。<b>刚才下推且有进展</b>：同样条件下实际位移 &lt; −0.4 mm。两阈值之间的帧排除。<b>这次下推的结果</b>：当前执行下推后下一帧实际下降 &lt; −0.4 mm vs &gt; −0.25 mm；仅对确实下推过的帧有标签。</p><p class="warn">“恢复候选”表示受阻后出现横向/旋转指令，后续 8 步内横向误差减小且轴向有进展；它不证明这一步动作造成恢复，也不说明未试动作的效果。聚类只用触觉、完全不看动作或物理标签。图表是观察性规律，不是因果结论。</p><img src="figures/test_event_coverage.png"><p>Test 事件帧数：{html.escape(str(events['test']['events']))}</p></section>
<section><h2>2. 各种输入能识别什么</h2><p>state = 18 维机器人状态 + 过去三步已执行动作；<code>only_</code> 行没有机器人状态，用来检验该模态自身是否能区分。RGB = 冻结 ResNet 的双视图及中心裁剪特征；触觉有双指，保持左右、base Fx/Fy/Fz。<code>force_map</code> 是双指全部 taxel 的有符号力，经 train-only PCA 48 维。接触后视觉仍有用与否看同一行任务的 state、state_rgb、state_rgb_force_map；不能只凭低维冻结视觉特征的负结果宣布相机无信息。</p><img src="figures/probe_balanced_accuracy.png"><h3>插座是否已接触</h3>{table('socket_contact')}<h3>刚才下推是否受阻</h3>{table('blocked_after_push')}<h3>执行当前下推动作后是否有进展</h3>{table('progress_when_pushed')}<p>第三个任务在所有输入中都加入当前动作，避免把指令差异误算成传感器收益。触觉历史是最近三帧左右指平均三维力；它只能用于该历史确实已发生的时刻。</p></section>
<section><h2>3. 力的符号、重叠与时间顺序</h2><p>先看两指 Fx/Fy/Fz 有符号均值在“刚才下推受阻”与“刚才下推有进展”之间的重叠，再看首次接触及受阻开始前后各五步的原始信号。线是测试示范中事件窗的中位数，阴影是 25–75 分位。Fz 的正负与接触结果是否对应、Fx/Fy 是否提供额外方向信息，应结合上方留出预测和下方连续视频判断。这里的曲线只是时间先后；上一步动作与力可同时影响后续运动。</p><img src="figures/signed_force_overlap.png"><img src="figures/event_aligned_signals.png"></section>
<section><h2>4. 用触觉模式找相似接触</h2><p>只对仿真确认 USB 接触插座、尚未完全插入的帧聚类。<code>amplitude</code> 看强度与粗分布；<code>pattern</code> 把每帧力图按自身 RMS 归一化，再用训练集 PCA 保留形状；<code>local_pattern</code> 在传感器局部坐标重复模式检索，用来核查机器人姿态造成的 base 力旋转；<code>pattern_history</code> 在 base 形状之外加前三帧力均值。弱力另列 -1，未接触不进入聚类。中心只在训练示范拟合，test 帧直接分配。类别编号没有预设物理含义。</p><img src="figures/cluster_outcomes.png"><table><tr><th>聚类</th><th>类</th><th>帧</th><th>示范数</th><th>RMS</th><th>下推帧</th><th>下推后推进/受阻</th><th>推进比例*</th><th>横移指令比例</th></tr>{''.join(cluster_rows)}</table><p>同一类若存在相反下插结果，下面优先展示不同示范中触觉模式最近、且下推动作相近的一对；若无此配对会明确标注。两段后续不同仍不能单凭这一对归因于触觉。*推进比例只在实际下推且结果不在灰区的帧中计算；下推帧少的类不作解释。类别内部即便动作同为下推，仍可能有不同姿态和速度，需看视频与物理量。</p></section>
<section><h2>5. 连续原始案例</h2><p>视频封面对应标记事件帧，播放时从事件前 8 步开始。每段同步展示原始前视、腕视、左右两侧原始触觉预览与双指六张 Fx/Fy/Fz 力图；红为正、蓝为负，所有案例使用同一训练集幅值标尺。下方图同时放两指三轴均值、当步指令与实际插件位移。视频标记在 pre-action 帧，下一步位移是动作执行后的结果。优先核对受阻、正常推进、恢复候选和不同聚类下的反例。</p>{''.join(cards)}</section>
<section><h2>边界与下一步</h2><p>现有示范未对相同初态尝试所有修正方向，因此无法单靠离线相关性判断“哪边调整最有效”或“此刻直接下插是否可行”。需要选取出现模态歧义的少量状态，在仿真中从同一状态分支执行动作。页面保存 <a href="events.json">事件覆盖</a>、<a href="probes.json">预测统计</a>、<a href="clusters.json">聚类统计</a>、<a href="cases.json">案例索引</a> 和逐帧预测数组，便于晚间继续分析。</p></section></html>'''
    cluster_count=sum('pair_group' in c for c in cases)
    navigation=(f'<nav><a class="primary" href="cluster_pairs.html">直接看触觉聚类配对视频（{cluster_count} 段）→</a> '
                '<a href="#scores">模态预测结果</a> <a href="#events">事件视频</a></nav>')
    content=content.replace('<h1>ContactWorld USB：事件、可见信息和双指触觉模式</h1>',
                            '<h1>ContactWorld USB：事件、可见信息和双指触觉模式</h1>'+navigation)
    content=content.replace('<section><h2>2. 各种输入能识别什么</h2>',
                            '<section id="scores"><h2>2. 各种输入能识别什么</h2>')
    content=content.replace('<section><h2>5. 连续原始案例</h2>',
                            '<section id="events"><h2>5. 连续原始事件案例</h2>')
    content=content.replace('优先核对受阻、正常推进、恢复候选和不同聚类下的反例。',
                            '这里展示事件案例。触觉聚类反例在页面顶部链接的专门配对页。')
    content=content.replace('</style>',
                            'nav{display:flex;gap:14px;flex-wrap:wrap;background:#263e5c;padding:16px;'
                            'border-radius:10px;margin:18px 0}.primary{background:#2c75af;color:white;'
                            'padding:8px 13px;border-radius:7px;font-weight:700}</style>')
    (args.output/'index.html').write_text(content)


def cluster_pair_page(args,cases,clusters):
    grouped={}
    for c in cases:
        if 'pair_group' in c:
            grouped.setdefault(c['pair_group'],[]).append(c)
    ids=sorted(grouped,key=lambda x:int(x.split()[-1]))
    nav=' '.join(f'<a href="#c{cid.split()[-1]}">{html.escape(cid)}</a>' for cid in ids)
    sections=[]
    counts={row['id']:row for row in clusters['pattern']['splits']['test']}
    for group in ids:
        cid=int(group.split()[-1]); pair=grouped[group]
        r=counts.get(cid)
        summary=(f"测试集 {r['frames']} 帧 / {r['demonstrations']} 条示范；实际下推 "
                 f"{r['down_trials']} 帧，其中推进 {r['down_progress']}、受阻 {r['down_block']}。") if r else ''
        cards=[]
        for case in pair:
            outcome='推进' if case['next_z_mm']<-.4 else '受阻' if case['next_z_mm']>-.25 else '灰区'
            rel=', '.join(f'{v:+.2f}' for v in case['relative_mm'])
            cmd=', '.join(f'{v:+.2f}' for v in case['current_action'][:3])
            cards.append(f'''<div class="card"><h3>{outcome}：source {case['source']} / frame {case['frame']}</h3>
<p>插件相对插座 XYZ = {rel} mm；当前指令 XYZ = {cmd} mm/步；实际下一步 Z = {case['next_z_mm']:+.2f} mm。</p>
<video controls preload="none" poster="{html.escape(case['movie'].replace('.mp4','.jpg'))}" src="{html.escape(case['movie'])}"></video>
<details><summary>动作、位移和三轴力时间线</summary><img loading="lazy" src="{html.escape(case['timeline'])}"></details></div>''')
        button=(f'<button type="button" onclick="syncPair(&quot;c{cid}&quot;)">两段从头同步播放</button> '
                f'<button type="button" onclick="pausePair(&quot;c{cid}&quot;)">暂停两段</button>') if len(pair)==2 else ''
        sections.append(f'''<section id="c{cid}"><h2>base 力图模式类 {cid}</h2><p>{summary}</p>
<p class="note">{html.escape(pair[0]['pair_note'])}</p>{button}
<div class="pair" data-pair="c{cid}">{''.join(cards)}</div></section>''')
    style='''<style>body{font:16px/1.55 system-ui;background:#101827;color:#eaf1fa;max-width:1900px;margin:auto;padding:22px}
 a{color:#9bd1ff}nav{display:flex;gap:14px;flex-wrap:wrap;background:#263d5c;padding:14px;position:sticky;top:0;z-index:3}
 section{background:#1b293d;margin:22px 0;padding:20px;border-radius:11px}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
 .card{background:#25364e;padding:13px;border-radius:8px}video,img{width:100%;height:auto}button{background:#2c75af;color:white;border:0;border-radius:6px;padding:9px 14px;margin:7px 5px 13px 0;cursor:pointer}
 .note{color:#ffe1a4}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>'''
    script='''<script>function syncPair(id){const v=document.querySelectorAll('[data-pair="'+id+'"] video');v.forEach(x=>{x.pause();x.currentTime=0});v.forEach(x=>x.play().catch(()=>{}))}
function pausePair(id){document.querySelectorAll('[data-pair="'+id+'"] video').forEach(x=>x.pause())}</script>'''
    content=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>双指触觉聚类案例</title>'+style+
       '<h1>双指触觉聚类：相似模式与不同下插结果</h1><p><a href="index.html">← 返回总报告</a></p>'+
       '<p>这里有 '+str(sum(map(len,grouped.values())))+' 段视频，按同一触觉类别成对展示。每段含前视、腕视、左右触觉预览、六张 Fx/Fy/Fz 力图。'
       '优先比较不同示范中下推动作及物理状态也相近的一对；若只能匹配动作，会明确标明位姿差异。'
       '两段视频长度可能不同，结束时停在末帧。视频封面是标记帧，播放从标记前 8 步开始。</p>'+
       '<nav><a href="index.html">总报告</a> '+nav+'</nav>'+''.join(sections)+script+'</html>')
    (args.output/'cluster_pairs.html').write_text(content)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,default=DEFAULT_HISTORY)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-cases',type=int,default=24)
    args=p.parse_args(); start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'predictions').mkdir(exist_ok=True)
    entries, episodes, data=load(args)
    identity={'version':VERSION,'smoke':args.smoke,'episodes':[[e['split'],e['source_episode']] for e in entries]}
    path=args.output/'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Output identity differs; choose another --output')
    atomic_json(path,identity)
    (args.output/'complete.json').unlink(missing_ok=True)
    labelset=all_labels(data)
    events=event_summary(args,data,labelset)
    blocks=prepare_features(data)
    probes,_=fit_probes(args,data,labelset,blocks)
    clusters=fit_clusters(args,data,labelset)
    plots(args,events,probes,clusters)
    aligned_counts=diagnostic_plots(args,episodes,data,labelset)
    atomic_json(args.output/'event_aligned_counts.json',aligned_counts)
    cases=render_cases(args,episodes,data,labelset)
    page(args,events,probes,clusters,cases)
    cluster_pair_page(args,cases,clusters)
    atomic_json(args.output/'complete.json',{'version':VERSION,'elapsed_s':round(time.monotonic()-start,1),
                                              'episodes':len(entries),'cases':len(cases)})
    print(f'DONE elapsed_s={time.monotonic()-start:.1f} report={args.output/"index.html"}',flush=True)

if __name__=='__main__':main()
