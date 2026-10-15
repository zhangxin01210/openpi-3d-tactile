#!/usr/bin/env python3
"""Validate the reviewed raw K=14 tactile classes against independent USB/socket physics.

Observation t contains the effect of executed action t-1. Class labels are
read from the frozen raw K=14 atlas, and physical labels never use tactile
force. Outputs include per-frame labels, episode-level counts, sensitivity
checks, and continuous video examples for support and counterexamples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from run_contactworld_matched_dynamics import atomic_json, encode_frames
from run_contactworld_sensor_discovery import DEFAULT_DATASET, image_paths
from run_contactworld_tactile_cluster_atlas import (
    assignment, load_episodes, read_video_frames, six_force_image,
)

DEFAULT_ATLAS = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_clusters_k14_20261008')
DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_raw14_physics_audit_20261008')
VERSION = 'cw-raw14-physics-audit-v1'
CANDIDATE = (5, 7, 9)
DISPLAY_PHASES = ('pair force below threshold', 'contact + downward progress',
                  'contact + USB stalled', 'contact + intermediate progress',
                  'contact + no decisive down command')
SPLITS = ('train', 'val', 'test')
PHASES = ('no_socket_contact', 'contact_smooth_down', 'contact_blocked_down',
          'contact_gray_down', 'contact_without_decisive_down')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_assignments(path):
    rows = list(csv.DictReader(path.open(newline='')))
    result = {}
    for row in rows:
        key = (row['split'], int(row['source']))
        result.setdefault(key, []).append(row)
    return result, len(rows)


def load(args):
    manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
    all_entries = manifest['episodes']
    if args.smoke:
        selected = [e for split, count in (('train', 6), ('val', 2), ('test', 2))
                    for e in [x for x in all_entries if x['split'] == split][:count]]
        selected_keys = {(x['split'], int(x['source_episode'])) for x in selected}
    else:
        selected_keys = None
    assignments, total_assignments = read_assignments(args.atlas / 'frame_assignments.csv')
    with np.load(args.atlas / 'cluster_model.npz') as z:
        centers = z['centers'].copy()
        scale = z['scale'].copy()
    if centers.shape != (14, 840):
        raise ValueError(f'Expected reviewed raw K=14 cluster model, got {centers.shape}')
    entries, episodes, _ = load_episodes(SimpleNamespace(dataset=args.dataset, smoke=args.smoke))
    if not args.smoke and sum(len(d['force']) for d in episodes.values()) != total_assignments:
        raise ValueError('Atlas frame count differs from dataset')
    records = {}
    capture_root = Path(manifest['capture_root'])
    for entry in entries:
        split, source = entry['split'], int(entry['source_episode'])
        key = (split, source)
        if selected_keys is not None and key not in selected_keys:
            continue
        d = episodes[key]
        n = len(d['force'])
        assigned = assignments.get(key, [])
        if len(assigned) != n:
            raise ValueError(f'Atlas assignment length mismatch: {key}: {len(assigned)} != {n}')
        classes = np.asarray([int(x['class']) for x in assigned], np.int16)
        if any(int(row['frame']) != t or int(row['episode']) != int(entry['episode_index'])
               for t, row in enumerate(assigned)):
            raise ValueError(f'Atlas source/frame order mismatch: {key}')
        # Verify frozen assignments, so a stale/mixed atlas cannot pass silently.
        normalized = np.clip(d['force'] / scale[None, :, None, :], -3, 3).reshape(n, 840)
        recomputed, _ = assignment(normalized, centers)
        if not np.array_equal(classes, recomputed):
            raise ValueError(f'Frozen class labels do not match model and force: {key}')
        folder = capture_root / f'insertion_usb_episode_{source:03d}'
        replay = json.loads((folder / 'replay.json').read_text())['rows']
        with np.load(folder / 'v2_capture.npz') as z:
            plug_pose = z['plug_pose'].copy()
            ee_pose = z['ee_pose'].copy()
            socket_pose = z['socket_pose'].copy()
        if len(replay) != n or len(plug_pose) != n:
            raise ValueError(f'Physics capture length mismatch: {key}')
        force_post = np.asarray([np.linalg.norm(row['plug_socket_force_post']) for row in replay], np.float32)
        contact_force = np.r_[0, force_post[:-1]]
        if not np.array_equal(contact_force > .1, d['contact']):
            raise ValueError(f'Contact force and source-aligned contact disagree: {key}')
        # Signed mm/step in the socket frame. At t, compare the previous
        # executed command with the actual motion observed by frame t.
        previous_action = np.vstack([np.full((1, 6), np.nan), d['action'][:-1]])
        observed_motion = np.vstack([np.full((1, 3), np.nan),
                                     np.diff(d['relative'], axis=0)])
        ee_rel = Rotation.from_quat(socket_pose[:, 3:]).inv().apply(
            ee_pose[:, :3] - socket_pose[:, :3]) * 1000
        ee_motion = np.vstack([np.full((1, 3), np.nan), np.diff(ee_rel, axis=0)])
        grasp_rel = Rotation.from_quat(ee_pose[:, 3:]).inv().apply(
            plug_pose[:, :3] - ee_pose[:, :3]) * 1000
        grasp_step = np.r_[np.nan, np.linalg.norm(np.diff(grasp_rel, axis=0), axis=1)]
        before_success = ~np.maximum.accumulate(d['success'])
        record = dict(entry=entry, split=split, source=source, data=d, classes=classes,
                      force=contact_force, force_post=force_post, previous_action=previous_action,
                      observed_motion=observed_motion, ee_motion=ee_motion,
                      grasp_step=grasp_step, before_success=before_success)
        records[key] = record
    return entries, records, scale


def phase_labels(record, threshold=.1, down_threshold=-2.0,
                 smooth_threshold=-.4, blocked_threshold=-.25):
    n = len(record['classes'])
    force = record['force']
    contact = force > threshold
    previous_down = record['previous_action'][:, 2] < down_threshold
    z = record['data']['relative'][:, 2]
    dz = record['observed_motion'][:, 2]
    eligible_down = contact & previous_down & (z > 3.0) & np.isfinite(dz)
    smooth = eligible_down & (dz < smooth_threshold)
    blocked = eligible_down & (dz > blocked_threshold)
    gray = eligible_down & ~(smooth | blocked)
    phase = np.full(n, 4, np.int8)
    phase[~contact] = 0
    phase[smooth] = 1
    phase[blocked] = 2
    phase[gray] = 3
    # The first observation has no previous action and no aligned contact.
    phase[0] = 0
    return phase, contact, eligible_down


def episode_counts(records, threshold=.1, before_only=False):
    out = []
    for key, rec in records.items():
        phase, contact, down = phase_labels(rec, threshold)
        valid = np.arange(len(phase)) > 0
        if before_only:
            valid &= rec['before_success']
        group = np.isin(rec['classes'], CANDIDATE)
        counts = np.zeros((2, len(PHASES)), np.int32)
        for g in (0, 1):
            for p in range(len(PHASES)):
                counts[g, p] = np.sum(valid & (group == bool(g)) & (phase == p))
        ee_down = rec['ee_motion'][:,2] < -.4
        differential = (phase == 2) & ee_down
        differential_counts = [int(np.sum(valid & (group == bool(g)) & differential)) for g in (0,1)]
        row = dict(split=key[0], source=key[1], frames=int(valid.sum()),
                   group_counts=counts.tolist(), differential_counts=differential_counts,
                   class_counts=np.bincount(rec['classes'][valid], minlength=14).astype(int).tolist(),
                   first_contact=int(np.flatnonzero(contact)[0]) if contact.any() else None)
        out.append(row)
    return out


def ratios_from_counts(counts):
    c = np.asarray(counts, np.int64)  # [other, candidate] x [5 phases]
    group = c[1]
    other = c[0]
    contact = c[:, 1:].sum()
    blocked = c[:, 2].sum()
    down_group = group[1:4].sum()
    down_other = other[1:4].sum()
    return dict(group_frames=int(group.sum()), other_frames=int(other.sum()),
                candidate_contact_rate=float(group[1:].sum() / max(1, group.sum())),
                other_contact_rate=float(other[1:].sum() / max(1, other.sum())),
                candidate_blocked_given_decisive_down=float(group[2] / max(1, group[1]+group[2])),
                other_blocked_given_decisive_down=float(other[2] / max(1, other[1]+other[2])),
                candidate_smooth_given_decisive_down=float(group[1] / max(1, group[1]+group[2])),
                other_smooth_given_decisive_down=float(other[1] / max(1, other[1]+other[2])),
                contact_recall=float(group[1:].sum() / max(1, contact)),
                blocked_recall=float(group[2] / max(1, blocked)),
                candidate_tested_down=int(down_group), other_tested_down=int(down_other),
                contact_frames=int(contact), blocked_frames=int(blocked))


def differential_stats(counts, differential):
    c = np.asarray(counts, np.int64)
    d = np.asarray(differential, np.int64)  # other, candidate
    denominators = c[:,1] + c[:,2]  # contact, decisive downward progress or stall
    return dict(other_differential_stall=int(d[0]), candidate_differential_stall=int(d[1]),
                all_differential_stall=int(d.sum()),
                candidate_differential_given_decisive_down=float(d[1]/max(1,denominators[1])),
                other_differential_given_decisive_down=float(d[0]/max(1,denominators[0])),
                differential_recall=float(d[1]/max(1,d.sum())))


def bootstrap(episode_rows, draws=1000):
    if not episode_rows:
        return {}
    arrays = np.asarray([x['group_counts'] for x in episode_rows], np.int64)
    differential = np.asarray([x['differential_counts'] for x in episode_rows], np.int64)
    rng = np.random.default_rng(20261008)
    names = ('candidate_contact_rate','other_contact_rate',
             'candidate_blocked_given_decisive_down','other_blocked_given_decisive_down',
             'contact_recall','blocked_recall',
             'candidate_differential_given_decisive_down',
             'other_differential_given_decisive_down','differential_recall')
    values = {n: [] for n in names}
    for _ in range(draws):
        ids = rng.integers(len(arrays), size=len(arrays))
        sample = arrays[ids].sum(axis=0)
        m = ratios_from_counts(sample)
        m.update(differential_stats(sample, differential[ids].sum(axis=0)))
        for n in names:
            values[n].append(m[n])
    return {n:[round(float(x), 4) for x in np.quantile(v, [.025,.975])]
            for n,v in values.items()}


def summarize(records, rows):
    all_counts = np.asarray([x['group_counts'] for x in rows], np.int64).sum(axis=0)
    all_differential = np.asarray([x['differential_counts'] for x in rows], np.int64).sum(axis=0)
    summary = dict(all=dict(counts=all_counts.tolist(), **ratios_from_counts(all_counts),
                            **differential_stats(all_counts, all_differential),
                            episode_bootstrap_95=bootstrap(rows)))
    for split in SPLITS:
        local = [x for x in rows if x['split'] == split]
        counts = np.asarray([x['group_counts'] for x in local], np.int64).sum(axis=0)
        differential = np.asarray([x['differential_counts'] for x in local], np.int64).sum(axis=0)
        summary[split] = dict(episodes=len(local), counts=counts.tolist(),
                              **ratios_from_counts(counts),
                              **differential_stats(counts, differential),
                              episode_bootstrap_95=bootstrap(local, draws=500))
    # Per class, including class 9's quiet Fz subgroup. The threshold was
    # fixed by the earlier K=14 audit and never tuned against physics labels.
    class_rows = []
    for k in range(14):
        counts = np.zeros(len(PHASES), np.int64)
        differential_count = 0
        demos = set()
        for key, rec in records.items():
            phase, _, _ = phase_labels(rec)
            mask = (rec['classes'] == k) & (np.arange(len(phase)) > 0)
            counts += np.bincount(phase[mask], minlength=len(PHASES))
            differential_count += int(np.sum(mask & (phase == 2) & (rec['ee_motion'][:,2] < -.4)))
            if mask.any():
                demos.add(key)
        class_rows.append(dict(class_id=k, frames=int(counts.sum()),
                               episodes=len(demos), counts=counts.tolist(),
                               differential_stall=differential_count,
                               contact_rate=float(counts[1:].sum()/max(1,counts.sum())),
                               blocked_given_down=float(counts[2]/max(1,counts[1]+counts[2])),
                               smooth_given_down=float(counts[1]/max(1,counts[1]+counts[2]))))
    return summary, class_rows



def temporal_contact_review(records):
    result = {}
    for width in (0, 1, 2, 3):
        total = near = past = future = 0
        for rec in records.values():
            group = np.isin(rec['classes'], CANDIDATE)
            contact = rec['force'] > .1
            for t in np.flatnonzero(group & ~contact):
                if t == 0:
                    continue
                total += 1
                past += int(contact[max(0,t-width):t+1].any())
                future += int(contact[t:min(len(contact),t+width+1)].any())
                near += int(contact[max(0,t-width):min(len(contact),t+width+1)].any())
        result[str(width)] = dict(candidate_below_threshold=total, nearby_contact=near,
                                  past_contact=past, future_contact=future)
    return result


def analyses(records, scale):
    # Threshold sensitivity is predeclared; no threshold is chosen to
    # maximize agreement with tactile classes.
    thresholds = {}
    for threshold in (.05, .1, .2):
        rows = episode_counts(records, threshold)
        counts = np.asarray([r['group_counts'] for r in rows], np.int64).sum(axis=0)
        thresholds[str(threshold)] = ratios_from_counts(counts)
    # Contact labels at t come from physics post-action t-1; this table
    # shows whether tactile classes lead or lag nearby contact observations.
    lags = {}
    for lag in range(-2, 3):
        g_contact, g_n, o_contact, o_n = 0,0,0,0
        for rec in records.values():
            cls = np.isin(rec['classes'], CANDIDATE)
            physical = rec['force'] > .1
            t = np.arange(2, len(cls)-2)
            selected = cls[t]
            observed = physical[t+lag]
            g_contact += int(observed[selected].sum()); g_n += int(selected.sum())
            o_contact += int(observed[~selected].sum()); o_n += int((~selected).sum())
        lags[str(lag)] = dict(candidate_contact_rate=g_contact/max(1,g_n),
                              other_contact_rate=o_contact/max(1,o_n),
                              candidate_frames=g_n, other_frames=o_n)
    strata = []
    for lo,hi in ((3,10),(10,20),(20,1000)):
        counts = np.zeros((2,2), np.int64)  # candidate/other x smooth/blocked
        demos = [set(),set()]
        for key,rec in records.items():
            phase,_,_ = phase_labels(rec)
            z=rec['data']['relative'][:,2]
            group=np.isin(rec['classes'],CANDIDATE)
            xy=np.linalg.norm(rec['data']['relative'][:,:2],axis=1)
            mask=(z>=lo)&(z<hi)&(xy<5)
            for g in (0,1):
                for outcome,p in enumerate((1,2)):
                    ids=np.flatnonzero(mask & (group==bool(g)) & (phase==p))
                    counts[g,outcome]+=len(ids)
                    if len(ids):demos[g].add(key)
        strata.append(dict(z_mm=[lo,hi], xy_max_mm=5, candidate_smooth=int(counts[1,0]),
                           candidate_blocked=int(counts[1,1]),other_smooth=int(counts[0,0]),
                           other_blocked=int(counts[0,1]),
                           candidate_episodes=len(demos[1]),other_episodes=len(demos[0])))
    quiet = {}
    threshold = float(scale[:,2].mean()) * .05
    for label in ('quiet','active'):
        counts=np.zeros(len(PHASES), np.int64); demos=set()
        for key,rec in records.items():
            fz=rec['data']['force'][:,:,:,2]
            rms=np.sqrt(np.mean(fz*fz,axis=(1,2)))
            selection=(rec['classes']==9)&((rms<threshold) if label=='quiet' else (rms>=threshold))
            selection[0]=False
            phase,_,_=phase_labels(rec)
            counts+=np.bincount(phase[selection],minlength=len(PHASES))
            if selection.any():demos.add(key)
        quiet[label]=dict(frames=int(counts.sum()),episodes=len(demos),
                          counts=counts.tolist(),contact_rate=float(counts[1:].sum()/max(1,counts.sum())))
    return dict(threshold_sensitivity=thresholds, temporal_lags=lags,
                zero_force_temporal_review=temporal_contact_review(records),
                matched_down_strata=strata, class9_fz_cutoff=threshold,
                class9_by_fz=quiet)


def select_cases(records, count_per_type):
    groups = {name:[] for name in ('candidate_blocked','candidate_smooth',
             'candidate_no_contact','other_blocked','other_smooth','other_no_contact')}
    for key, rec in records.items():
        phase, _, _ = phase_labels(rec)
        cls = np.isin(rec['classes'], CANDIDATE)
        for t in range(4, len(phase)-4):
            if not rec['before_success'][t]:
                continue
            if cls[t] and phase[t]==2 and rec['ee_motion'][t,2]<-.4: name='candidate_blocked'
            elif cls[t] and phase[t]==1: name='candidate_smooth'
            elif cls[t] and phase[t]==0: name='candidate_no_contact'
            elif not cls[t] and phase[t]==2 and rec['ee_motion'][t,2]<-.4: name='other_blocked'
            elif not cls[t] and phase[t]==1: name='other_smooth'
            elif not cls[t] and phase[t]==0: name='other_no_contact'
            else: continue
            groups[name].append((key,t))
    rng=np.random.default_rng(20261008)
    picked={}
    for name,cases in groups.items():
        # Fixed seed, one case per source; favor evaluation episodes but do not
        # rank by force magnitude or cherry-pick visually dramatic events.
        shuffled=rng.permutation(len(cases))
        buckets={split:[] for split in SPLITS}
        for i in shuffled:
            key,t=cases[int(i)]
            buckets[key[0]].append((key,t))
        used=set();selected=[]
        for split in ('test','val','train'):
            for key,t in buckets[split]:
                if key in used:continue
                used.add(key);selected.append((key,t))
                if len(selected)>=count_per_type:break
            if len(selected)>=count_per_type:break
        picked[name]=selected
    return picked, {name:len(items) for name,items in groups.items()}


def write_frame_csv(args, records):
    path=args.output/'all_frame_physics.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(('split','source','episode','frame','raw_class','candidate_5_7_9',
                    'physical_phase','socket_contact_force','socket_contact',
                    'previous_cmd_x_mm','previous_cmd_y_mm','previous_cmd_z_mm',
                    'observed_plug_dz_mm','observed_ee_dz_mm','grasp_change_mm',
                    'plug_stall_with_ee_down',
                    'relative_x_mm','relative_y_mm','relative_z_mm','before_success'))
        for key,rec in records.items():
            phase,contact,_=phase_labels(rec)
            d=rec['data']
            for t in range(len(phase)):
                w.writerow([key[0],key[1],rec['entry']['episode_index'],t,
                            int(rec['classes'][t]),int(rec['classes'][t] in CANDIDATE),
                            PHASES[phase[t]],round(float(rec['force'][t]),5),int(contact[t]),
                            *[round(float(v),3) for v in rec['previous_action'][t,:3]],
                            round(float(rec['observed_motion'][t,2]),3),
                            round(float(rec['ee_motion'][t,2]),3),
                            round(float(rec['grasp_step'][t]),3),
                            int(phase[t]==2 and rec['ee_motion'][t,2]<-.4),
                            *[round(float(v),3) for v in d['relative'][t]],int(rec['before_success'][t])])


def case_frame(front, wrist, left, right, rec, scale, t, marked):
    d=rec['data']
    canvas=np.full((750,1280,3),25,np.uint8)
    canvas[50:370,0:320]=front;canvas[50:370,320:640]=wrist
    canvas[425:745,0:320]=left;canvas[425:745,320:640]=right
    for x,y,label in ((8,33,'FRONT'),(327,33,'WRIST'),
                       (8,408,'LEFT TACFF PREVIEW'),(327,408,'RIGHT TACFF PREVIEW')):
        cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.56,(240,240,240),1,cv2.LINE_AA)
    canvas[35:385,645:1275]=six_force_image(d['force'][t],scale)
    phase,_,_=phase_labels(rec)
    prev=rec['previous_action'][t,:3]
    dz=rec['observed_motion'][t,2]
    lines=[f'CLASS {rec["classes"][t]} | {"CANDIDATE 5/7/9" if rec["classes"][t] in CANDIDATE else "OTHER"} | source {rec["source"]}',
           f'marked {marked} | shown {t} ({t-marked:+d}) | {rec["split"]}',
           f'RECORD: {DISPLAY_PHASES[phase[t]]}',
           f'USB-socket pair force {rec["force"][t]:.3f} (post-action t-1)',
           f'previous command XYZ mm: {prev[0]:+.2f}, {prev[1]:+.2f}, {prev[2]:+.2f}',
           f'actual USB dz {dz:+.2f} | EE dz {rec["ee_motion"][t,2]:+.2f} mm',
           f'grasp relative change {rec["grasp_step"][t]:.2f} mm',
           f'relative socket Z {d["relative"][t,2]:.2f} mm | raw base Fx/Fy/Fz']
    for j,line in enumerate(lines):
        cv2.putText(canvas,line,(649,426+j*43),cv2.FONT_HERSHEY_SIMPLEX,.51,(238,238,238),1,cv2.LINE_AA)
    return canvas


def timeline(path, rec, t, lo, hi):
    indices=np.arange(lo,hi)
    phase,_,_=phase_labels(rec)
    fig,axes=plt.subplots(4,1,figsize=(11,7),sharex=True,layout='constrained')
    axes[0].step(indices,rec['classes'][lo:hi],where='mid',color='#5551a7')
    axes[0].set(ylabel='raw class',yticks=range(14),ylim=(-.5,13.5))
    axes[1].plot(indices,rec['force'][lo:hi],color='#bd5b3b')
    axes[1].axhline(.1,ls='--',color='gray',lw=1)
    axes[1].set_ylabel('USB/socket force')
    axes[2].plot(indices,rec['previous_action'][lo:hi,2],label='previous commanded Z',color='#d28a32')
    axes[2].plot(indices,rec['observed_motion'][lo:hi,2],label='actual USB dz',color='#32859b')
    axes[2].legend(fontsize=8,ncol=2)
    axes[2].set_ylabel('mm per control step')
    axes[3].step(indices,phase[lo:hi],where='mid',color='#5d9056')
    axes[3].set(ylabel='physical phase',yticks=range(5),
                yticklabels=['force below threshold','downward progress','USB stalled','gray','contact/unknown'],
                xlabel='source demonstration frame')
    for ax in axes:
        ax.axvline(t,color='#c63131',lw=1)
        ax.grid(alpha=.18)
    fig.savefig(path,dpi=125);plt.close(fig)


def render_cases(args, records, selected, scale):
    root=args.output/'cases';root.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,items in selected.items():
        manifest[name]=[]
        for index,(key,t) in enumerate(items):
            rec=records[key];d=rec['data'];n=len(d['force'])
            lo=max(0,t-8);hi=min(n,t+13);frames=list(range(lo,hi))
            paths=image_paths(args.dataset,int(rec['entry']['episode_index']))
            basename=f'{name}_{index:02d}_{key[0]}_source{key[1]:03d}_f{t:03d}'
            video=root/f'{basename}.mp4';poster=root/f'{basename}.jpg';chart=root/f'{basename}_timeline.png'
            front=read_video_frames(paths['front'],frames)
            wrist=read_video_frames(paths['wrist'],frames)
            left=read_video_frames(paths['tacff_left_preview'],frames)
            right=read_video_frames(paths['tacff_right_preview'],frames)
            if args.refresh_videos or not video.exists() or not poster.exists():
                encode_frames(video,1280,750,
                    (case_frame(a,b,c,e,rec,scale,j,t)
                     for a,b,c,e,j in zip(front,wrist,left,right,frames)),fps=10)
                cv2.imwrite(str(poster),case_frame(front[t-lo],wrist[t-lo],
                            left[t-lo],right[t-lo],rec,scale,t,t))
            if args.refresh_videos or not chart.exists():timeline(chart,rec,t,lo,hi)
            phase,_,_=phase_labels(rec)
            manifest[name].append(dict(split=key[0],source=key[1],episode=int(rec['entry']['episode_index']),
                frame=t,raw_class=int(rec['classes'][t]),physical_phase=PHASES[phase[t]],
                pair_force=float(rec['force'][t]),previous_action_xyz_mm=rec['previous_action'][t,:3].tolist(),
                plug_dz_mm=float(rec['observed_motion'][t,2]),
                ee_dz_mm=float(rec['ee_motion'][t,2]),grasp_change_mm=float(rec['grasp_step'][t]),
                video=str(video.relative_to(args.output)),poster=str(poster.relative_to(args.output)),
                timeline=str(chart.relative_to(args.output)),mark=t-lo))
            print(f'render {name} {index+1}/{len(items)} source={key[1]} frame={t}',flush=True)
    atomic_json(args.output/'cases.json',manifest)
    return manifest


def plot_summary(args, per_class, summary, additional):
    directory=args.output/'figures';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
    xs=np.arange(14)
    axes[0].bar(xs,[r['contact_rate'] for r in per_class],color=['#d25c4b' if k in CANDIDATE else '#5781a7' for k in xs])
    axes[0].set(ylim=(0,1),xticks=xs,xlabel='original raw tactile class',ylabel='USB/socket contact fraction')
    axes[1].bar(xs,[r['blocked_given_down'] for r in per_class],color=['#d25c4b' if k in CANDIDATE else '#5781a7' for k in xs])
    axes[1].set(ylim=(0,1),xticks=xs,xlabel='original raw tactile class',ylabel='blocked / (blocked + smooth down)')
    axes[2].bar(xs,[r['differential_stall']/max(1,r['counts'][1]+r['counts'][2]) for r in per_class],
                color=['#d25c4b' if k in CANDIDATE else '#5781a7' for k in xs])
    axes[2].set(ylim=(0,1),xticks=xs,xlabel='original raw tactile class',
                ylabel='EE down + USB stalled / decisive down')
    fig.savefig(directory/'class_physics.png',dpi=145);plt.close(fig)
    names=['force below threshold','downward progress','USB stalled','gray down','contact/untested']
    counts=np.asarray(summary['all']['counts'])
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    bottom=np.zeros(2)
    for i,name in enumerate(names):
        ax.bar(['other classes','5 / 7 / 9'],counts[:,i],bottom=bottom,label=name)
        bottom+=counts[:,i]
    ax.set_ylabel('observation frames')
    ax.legend(ncol=3,fontsize=9)
    fig.savefig(directory/'group_counts.png',dpi=145);plt.close(fig)
    lags=additional['temporal_lags']
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for label,key in [('5/7/9','candidate_contact_rate'),('other','other_contact_rate')]:
        ax.plot(range(-2,3),[lags[str(j)][key] for j in range(-2,3)],marker='o',label=label)
    ax.set(xlabel='physics contact frame relative to tactile class frame',
           ylabel='fraction with USB/socket contact',ylim=(0,1),xticks=range(-2,3))
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(directory/'contact_lag.png',dpi=145);plt.close(fig)


def report(args, summary, class_rows, extra, cases, case_counts):
    css='''<style>body{font:16px/1.5 system-ui;background:#101827;color:#eaf2fd;max-width:1650px;margin:auto;padding:22px}
a{color:#a5d5ff}section{background:#1d2b40;border-radius:10px;margin:18px 0;padding:17px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #526783;padding:5px;text-align:left}
th{background:#314a6b}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
article{background:#293b55;padding:12px;border-radius:8px}.grid video,.grid img{width:100%;height:auto}
button{padding:8px 12px;background:#2c77b4;color:white;border:0;border-radius:6px;cursor:pointer;margin:4px}
.note{background:#473b29;padding:9px;border-radius:6px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>'''
    rates=summary['all'];ci=rates['episode_bootstrap_95']
    before=extra['before_success']
    near=extra['zero_force_temporal_review']
    table=''.join(f'<tr><td>{r["class_id"]}</td><td>{r["frames"]}</td><td>{r["episodes"]}</td>'
                  f'<td>{r["contact_rate"]:.1%}</td><td>{r["counts"][1]}</td><td>{r["counts"][2]}</td>'
                  f'<td>{r["counts"][4]}</td><td>{r["differential_stall"]}</td>'
                  f'<td>{r["blocked_given_down"]:.1%}</td></tr>'
                  for r in class_rows)
    group_table=''.join(f'<tr><td>{label}</td>'+''.join(f'<td>{x}</td>' for x in vals)+'</tr>'
        for label,vals in [('其余类',rates['counts'][0]),('5／7／9',rates['counts'][1])])
    subsets=''.join(f'<tr><td>{split}</td><td>{summary[split]["episodes"]}</td>'
                    f'<td>{summary[split]["candidate_contact_rate"]:.1%}</td>'
                    f'<td>{summary[split]["candidate_blocked_given_decisive_down"]:.1%}</td>'
                    f'<td>{summary[split]["candidate_differential_given_decisive_down"]:.1%}</td>'
                    f'<td>{summary[split]["other_differential_given_decisive_down"]:.1%}</td>'
                    f'<td>{summary[split]["differential_recall"]:.1%}</td></tr>' for split in SPLITS)
    strata=''.join(f'<tr><td>{r["z_mm"][0]}–{r["z_mm"][1]}</td><td>{r["candidate_episodes"]}</td>'
        f'<td>{r["candidate_smooth"]}</td><td>{r["candidate_blocked"]}</td>'
        f'<td>{r["other_episodes"]}</td><td>{r["other_smooth"]}</td><td>{r["other_blocked"]}</td></tr>'
        for r in extra['matched_down_strata'])
    case_sections=[]
    titles={'candidate_blocked':'5／7／9：夹爪下移但 USB 停滞',
            'candidate_smooth':'5／7／9：该步有接触且仍在下降（可能碰侧壁或触底）',
            'candidate_no_contact':'5／7／9：标记帧接触力记录低于阈值（不能据此称自由运动）',
            'other_blocked':'标记帧属其他类：夹爪下移但 USB 停滞（片段内可出现 5／7／9）',
            'other_smooth':'标记帧属其他类：有接触且仍在下降（前后可停滞）',
            'other_no_contact':'标记帧属其他类：接触力记录低于阈值'}
    for name,items in cases.items():
        cards=[]
        for item in items:
            cmd=', '.join(f'{v:+.1f}' for v in item['previous_action_xyz_mm'])
            cards.append(f'''<article><h3>原始类 {item["raw_class"]} · {item["split"]} source {item["source"]} · 帧 {item["frame"]}</h3>
<p>标记帧记录 {html.escape(DISPLAY_PHASES[PHASES.index(item["physical_phase"])])}；USB—插座接触力 {item["pair_force"]:.3f}；
此前动作 XYZ {cmd} mm；USB 实际 dz {item["plug_dz_mm"]:+.2f} mm；
夹爪实际 dz {item["ee_dz_mm"]:+.2f} mm；相对夹持变化 {item["grasp_change_mm"]:.2f} mm。</p>
<button onclick="const v=this.nextElementSibling;v.pause();v.currentTime={item["mark"]/10+.03:.2f};">停在分类依据帧 {item["frame"]}</button>
<video controls preload="metadata" poster="{html.escape(item["poster"])}" src="{html.escape(item["video"])}"></video>
<img loading="lazy" src="{html.escape(item["timeline"])}"></article>''')
        case_sections.append(f'''<section id="{name}"><h2>{titles[name]}：候选帧 {case_counts[name]}，展示 {len(items)}</h2>
<div class="grid">{''.join(cards) if cards else '<p>没有满足条件的帧。</p>'}</div></section>''')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ContactWorld 原始14类物理检验</title>{css}
<h1>ContactWorld 原始 14 类：5／7／9 是否对应运动受限</h1>
<p>类别固定为此前你审阅的原始 base 坐标三维力图 K=14；完整示范的逐帧类别不重新聚类。
接触标签来自 USB 与插座的仿真接触记录，运动结果来自物体实测位姿，不使用触觉定义。</p>
<section><h2>视频复核后的解释修正</h2>
<p><b>仍在下降并不表示没有碰撞：</b>这些帧本来就检测到 USB—插座接触，下降可与侧壁擦碰、倾斜接触同时发生。
“顺畅”旧标题过强，现改为“该步仍在下降”。现有日志不能单凭 dz 判断碰的是侧壁、孔口还是底部。</p>
<p><b>单帧零力不证明整段没有接触：</b>全部候选类中力记录未超阈值的 {near['0']['candidate_below_threshold']} 帧，
其中 {near['1']['nearby_contact']} 帧在前后 1 步检测到接触，{near['3']['nearby_contact']} 帧在前后 3 步检测到接触。
这支持它们处于接触过程附近，但不能由此确定零力那个采样时刻的接触状态或零值原因。
窗口统计是对本次质疑的事后复核，不能替代原始逐帧指标。</p>
<p><b>标题分类只指一帧：</b>点击每例的“停在分类依据帧”，再播放上下文。
相邻帧可以从 9 跳到 3／8／11，同一段接触过程可跨多个聚类。肉眼看到同类模式时，应核对是否就是标记帧；
不能用整个片段的出现情况去更改那个帧的数值类别，也不能把数值类别当作互斥物理状态。</p></section>
<section><h2>读法与时间对齐</h2><p>视频上的第 t 帧触觉，和执行第 t−1 步动作后记录的 USB—插座力、USB 位移对齐。
“仍在下降／USB 停滞”只在接触且此前明显向下指令时判断：USB dz &lt; −0.4 mm 表示该步仍在下降；
dz &gt; −0.25 mm 为 USB 停滞；中间留灰区。USB 停滞再按夹爪是否继续下移区分；
无明显下推动作的接触帧标作无法判断运动约束。
接触阈值沿用 0.1，附有 0.05／0.2 敏感性分析。</p>
<p class="note">5／7／9 的类别可以提示状态，但类别与插座接触不会一一对应。视频在标记帧前后类别会变化，
请看右侧当前类别和物理状态；图片、视频和统计都记录反例。</p></section>
<section><h2>全量对照</h2>
<p>5／7／9 的 USB—插座接触比例 <b>{rates["candidate_contact_rate"]:.1%}</b>
（按示范重采样 95% 范围 {ci["candidate_contact_rate"][0]:.1%}–{ci["candidate_contact_rate"][1]:.1%}）；
其余类别 {rates["other_contact_rate"]:.1%}。
全部符合单步 USB 停滞阈值的帧中，5／7／9 覆盖 <b>{rates["blocked_recall"]:.1%}</b>
（{ci["blocked_recall"][0]:.1%}–{ci["blocked_recall"][1]:.1%}）。
只看有明确下推结果的帧，5／7／9 中受限比例 {rates["candidate_blocked_given_decisive_down"]:.1%}，
其余类 {rates["other_blocked_given_decisive_down"]:.1%}。
这些比率的分母分别是：候选帧 {rates["group_frames"]}、USB 停滞帧 {rates["blocked_frames"]}、
候选组明确下推帧 {rates["counts"][1][1]+rates["counts"][1][2]}、
其他类明确下推帧 {rates["counts"][0][1]+rates["counts"][0][2]}。</p>
<p class="note"><b>关键拆分：</b>USB 停滞时，夹爪可能也停住；这与夹爪继续下移而 USB 被顶住是不同情形。
后一种“差动停滞”要求同一步中夹爪实际 dz &lt; −0.4 mm、USB dz &gt; −0.25 mm，并有 USB—插座接触。
5／7／9 组中差动停滞 {rates["candidate_differential_stall"]} 帧／明确下推 {rates["counts"][1][1]+rates["counts"][1][2]} 帧
（{rates["candidate_differential_given_decisive_down"]:.1%}；示范重采样区间 {ci["candidate_differential_given_decisive_down"][0]:.1%}–{ci["candidate_differential_given_decisive_down"][1]:.1%}）；
其他类 {rates["other_differential_stall"]} 帧／{rates["counts"][0][1]+rates["counts"][0][2]} 帧
（{rates["other_differential_given_decisive_down"]:.1%}）。
所有差动停滞中候选类覆盖 {rates["differential_recall"]:.1%}
（{ci["differential_recall"][0]:.1%}–{ci["differential_recall"][1]:.1%}）。
这仍是短时运动学标签，不等于唯一障碍原因。</p>
<p>只看首次成功前：候选组接触比例 {before["candidate_contact_rate"]:.1%}，
明确下推中受限比例 {before["candidate_blocked_given_decisive_down"]:.1%}，
覆盖全部 USB 停滞帧的比例 {before["blocked_recall"]:.1%}，
差动停滞覆盖 {before["differential_recall"]:.1%}；
成功后的触觉模式不应单独支撑纠偏结论。</p>
<img style="max-width:48%" src="figures/group_counts.png"><img style="max-width:48%" src="figures/class_physics.png">
<table><tr><th>分组</th><th>力记录低于阈值</th><th>有接触且仍下降</th><th>接触且 USB 停滞</th><th>接触且灰区</th><th>接触但没有明确下推</th></tr>{group_table}</table>
<h3>按完整示范划分</h3><table><tr><th>划分</th><th>示范数</th><th>5／7／9 接触比例</th>
<th>5／7／9 明确下推中 USB 停滞比例</th>
<th>5／7／9 差动停滞比例</th><th>其他类差动停滞比例</th>
<th>差动停滞帧覆盖</th></tr>{subsets}</table>
<p>连续帧相关，区间按整条示范抽样。完整计数见 <a href="analysis.json">analysis.json</a>；
每帧类别和独立物理量见 <a href="all_frame_physics.csv">all_frame_physics.csv</a>。</p></section>
<section><h2>逐类与重要例外</h2>
<table><tr><th>类</th><th>帧</th><th>示范</th><th>接触比例</th><th>有接触且下降帧</th><th>USB 停滞帧</th>
<th>接触但未明确下推</th><th>夹爪下移／USB 停滞</th><th>明确下推中 USB 停滞比例</th></tr>{table}</table>
<p>类 9 中 Fz 接近零的帧数 {extra["class9_by_fz"]["quiet"]["frames"]}，
其接触比例 {extra["class9_by_fz"]["quiet"]["contact_rate"]:.1%}；
其余类 9 帧数 {extra["class9_by_fz"]["active"]["frames"]}，
接触比例 {extra["class9_by_fz"]["active"]["contact_rate"]:.1%}。
Fz 分界沿用先前审计的训练集标尺 5%，没有根据物理标签调整。</p>
<h3>相同几何与下推条件的粗分层</h3>
<p>仅看插座相对 XY 偏差小于 5 mm 且此前明显下推的帧。Z 为 USB 相对插座轴向高度。</p>
<table><tr><th>Z mm</th><th>5／7／9 示范</th><th>仍下降</th><th>USB 停滞</th><th>其他类示范</th>
<th>仍下降</th><th>USB 停滞</th></tr>{strata}</table>
<h3>接触记录前后两步</h3><img style="max-width:750px" src="figures/contact_lag.png"></section>
{''.join(case_sections)}
<section><h2>原始数据与边界</h2>
<p><a href="{html.escape(str(args.atlas/'index.html'))}">原始 K=14 触觉图库</a> ·
<a href="cases.json">视频案例索引</a> · <a href="episode_counts.csv">逐示范计数</a>。
这里分析的是已执行示范；它不证明从同一状态执行另一动作会怎样。</p></section></html>'''
    (args.output/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--atlas',type=Path,default=DEFAULT_ATLAS)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--cases-per-type',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-videos',action='store_true')
    p.add_argument('--refresh-videos',action='store_true',help='Regenerate cached case videos and timelines')
    args=p.parse_args()
    if args.cases_per_type<1:p.error('--cases-per-type must be positive')
    started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(version=VERSION,smoke=args.smoke,cases_per_type=args.cases_per_type,
                  dataset_manifest_sha256=sha256(args.dataset/'spatial/manifest.json'),
                  atlas_assignments_sha256=sha256(args.atlas/'frame_assignments.csv'))
    identity_file=args.output/'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text())!=identity:
        raise ValueError('Output identity differs; choose another --output')
    atomic_json(identity_file,identity)
    (args.output/'complete.json').unlink(missing_ok=True)
    entries,records,scale=load(args)
    counts=episode_counts(records)
    summary,classes=summarize(records,counts)
    extra=analyses(records,scale)
    before_counts=episode_counts(records,before_only=True)
    before_total=np.asarray([x['group_counts'] for x in before_counts]).sum(axis=0)
    before_differential=np.asarray([x['differential_counts'] for x in before_counts]).sum(axis=0)
    extra['before_success']=dict(counts=before_total.tolist(),**ratios_from_counts(before_total),
                                 **differential_stats(before_total,before_differential))
    atomic_json(args.output/'analysis.json',dict(version=VERSION,summary=summary,
                per_class=classes,additional=extra,episodes=len(entries),frames=sum(len(r['classes']) for r in records.values())))
    with (args.output/'episode_counts.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','source','frames']+
             [f'other_{x}' for x in PHASES]+[f'candidate_{x}' for x in PHASES]+
             ['other_ee_down_plug_stalled','candidate_ee_down_plug_stalled'])
        for row in counts:w.writerow([row['split'],row['source'],row['frames'],
                           *row['group_counts'][0],*row['group_counts'][1],*row['differential_counts']])
    write_frame_csv(args,records)
    selected,case_counts=select_cases(records,args.cases_per_type)
    cases=render_cases(args,records,selected,scale) if not args.no_videos else {k:[] for k in selected}
    plot_summary(args,classes,summary,extra)
    report(args,summary,classes,extra,cases,case_counts)
    atomic_json(args.output/'complete.json',dict(version=VERSION,elapsed_s=round(time.monotonic()-started,1),
                episodes=len(entries),frames=sum(len(r['classes']) for r in records.values()),
                cases=sum(map(len,cases.values())),report=str(args.output/'index.html')))
    print(f'DONE episodes={len(entries)} frames={sum(len(r["classes"]) for r in records.values())} '
          f'cases={sum(map(len,cases.values()))} report={args.output/"index.html"}',flush=True)


if __name__=='__main__':main()

