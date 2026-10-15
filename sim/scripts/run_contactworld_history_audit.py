#!/usr/bin/env python3
"""ContactWorld USB: matched dynamic states, tactile timing, and action effects.

Read-only with respect to data and the first sensor-discovery run. Produces a
new resumable output directory. This is an observational audit: a matched
example is never called a causal intervention or the uniquely correct action.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from run_contactworld_sensor_discovery import DEFAULT_DATASET, image_paths, select_entries, fit_pca

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_sensor_discovery_20261004')
DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_history_audit_20261008')
VERSION = 'cw-usb-history-audit-v1'
SPLITS = ('train', 'val', 'test')
STAGES = ('prepare', 'fit', 'match', 'report')
MATCH_LIMITS = {
    'strict': dict(xy=1.5, z=2., rot=2., grasp=2., grasp_rot=3., width=.8,
                   plug_vel=1.5, ee_vel=2., action_hist=3., action_rot=2.),
    'relaxed': dict(xy=4., z=5., rot=5., grasp=4., grasp_rot=6., width=2.,
                    plug_vel=4., ee_vel=5., action_hist=8., action_rot=5.),
}
MODES = {
    'G_geometry': ('geometry',),
    'H_geometry_history': ('geometry', 'history'),
    'H_RGB': ('geometry', 'history', 'rgb'),
    'H_T': ('geometry', 'history', 'tactile'),
    'H_RGB_T': ('geometry', 'history', 'rgb', 'tactile'),
    'H_RGB_Tpast': ('geometry', 'history', 'rgb', 'tactile', 'tactile_past'),
}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cache_path(out, entry):
    return out / 'episodes' / f"{entry['split']}_source_{entry['source_episode']:03d}.npz"


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def pair_angle_deg(left, right):
    left, right = np.asarray(left), np.asarray(right)
    dot = np.sum(left * right, axis=-1)
    return np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))


def make_episode(args, entry, manifest):
    source, episode = entry['source_episode'], entry['episode_index']
    previous = args.prior / 'episodes' / f"{entry['split']}_source_{source:03d}.npz"
    if not previous.is_file():
        raise FileNotFoundError(f'First-run features are missing: {previous}')
    with np.load(previous) as old:
        n = len(old['frame'])
        old_fields = {key: old[key].copy() for key in
                      ('state', 'rgb', 'crop', 'force_base', 'force_local', 'rel', 'contact',
                       'socket_force', 'action', 'map', 'summary', 'slip')}
    folder = Path(manifest['capture_root']) / f'insertion_usb_episode_{source:03d}'
    with np.load(folder / 'v2_capture.npz') as raw:
        plug, socket, ee, action = [raw[k].copy() for k in
                                    ('plug_pose', 'socket_pose', 'ee_pose', 'action_executed')]
    if not (len(plug) == len(socket) == len(ee) == len(action) == entry['length'] and n == entry['length']-4):
        raise ValueError(f'Source {source}: replay length mismatch')
    sock_rotation = Rotation.from_quat(socket[:, 3:])
    ee_rotation = Rotation.from_quat(ee[:, 3:])
    plug_rotation = Rotation.from_quat(plug[:, 3:])
    relative_rotation = sock_rotation.inv() * plug_rotation
    grasp_rotation = ee_rotation.inv() * plug_rotation
    grasp_pos = ee_rotation.inv().apply(plug[:, :3] - ee[:, :3]) * 1000
    relative_pos = sock_rotation.inv().apply(plug[:, :3] - socket[:, :3]) * 1000
    plug_vel = sock_rotation.inv().apply(np.vstack((np.zeros((1, 3)), np.diff(plug[:, :3], axis=0)))) * 1000
    ee_vel = sock_rotation.inv().apply(np.vstack((np.zeros((1, 3)), np.diff(ee[:, :3], axis=0)))) * 1000
    # 6.667e-2 seconds per control step. Velocities retain mm/control-step
    # units so matching tolerances are readable and do not assume FPS=10.
    action_scaled = np.concatenate((sock_rotation.inv().apply(action[:, :3] * .01) * 1000,
                                    action[:, 3:] * (.05 * 180 / np.pi)), axis=1)
    action_history = np.stack([np.roll(action_scaled, k, axis=0) for k in (1, 2, 3)], axis=1)
    tactile_mean = old_fields['force_base'].mean(axis=2).reshape(n, 6)
    tactile_history = np.stack([np.roll(tactile_mean, k, axis=0) for k in (1, 2, 3)], axis=1)
    # Missing history at the first three frames is excluded downstream.
    valid = np.arange(n) >= 3
    geom = np.concatenate((relative_pos[:n], relative_rotation.as_matrix()[:n].reshape(n, 9),
                           grasp_pos[:n], grasp_rotation.as_matrix()[:n].reshape(n, 9),
                           old_fields['state'][:, :9]), axis=1).astype(np.float32)
    hist = np.concatenate((plug_vel[:n], ee_vel[:n], action_history[:n].reshape(n, 18)), axis=1).astype(np.float32)
    target = cache_path(args.output, entry)
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, source=np.full(n, source, np.int32),
        episode=np.full(n, episode, np.int32), frame=np.arange(n, dtype=np.int32), valid=valid,
        geometry=geom, history=hist, rgb=np.concatenate((old_fields['rgb'],old_fields['crop']), axis=1),
        tactile=old_fields['map'], tactile_mean=tactile_mean.astype(np.float32),
        tactile_past=tactile_history[:n].reshape(n, 18).astype(np.float32),
        state=old_fields['state'], force_base=old_fields['force_base'],
        force_local=old_fields['force_local'], relative=relative_pos[:n].astype(np.float32),
        rel_quat=relative_rotation.as_quat()[:n].astype(np.float32),
        grasp_pos=grasp_pos[:n].astype(np.float32),
        grasp_quat=grasp_rotation.as_quat()[:n].astype(np.float32),
        plug_vel=plug_vel[:n].astype(np.float32), ee_vel=ee_vel[:n].astype(np.float32),
        action=action_scaled[:n].astype(np.float32),
        action_history=action_history[:n].astype(np.float32),
        action_sequence4=np.stack([action_scaled[i:i+4] for i in range(n)]).astype(np.float32),
        motion1=(relative_pos[1:n+1]-relative_pos[:n]).astype(np.float32),
        motion4=(relative_pos[4:n+4]-relative_pos[:n]).astype(np.float32),
        slip_proxy=old_fields['slip'], contact=old_fields['contact'],
        socket_force=old_fields['socket_force'])
    temporary.replace(target)


def prepare(args, entries, manifest):
    old_identity = args.prior / 'identity.json'
    identity = dict(version=VERSION, manifest_sha256=file_hash(args.dataset/'spatial/manifest.json'),
                    prior_identity_sha256=file_hash(old_identity), smoke=args.smoke,
                    episodes=[[e['split'],e['source_episode']] for e in entries])
    path = args.output/'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Output identity differs; use another output directory')
    atomic_json(path, identity)
    (args.output/'episodes').mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries, 1):
        dest = cache_path(args.output, entry)
        if not dest.exists():
            make_episode(args, entry, manifest)
        print(f"prepare {i}/{len(entries)} {entry['split']} source={entry['source_episode']}", flush=True)
    atomic_json(args.output/'episode_index.json', entries)


def join(args, entries, split):
    pieces = [load_npz(cache_path(args.output, e)) for e in entries if e['split'] == split]
    fields = ('source','frame','valid','geometry','history','rgb','tactile','tactile_past',
              'tactile_mean','relative','rel_quat','grasp_pos','grasp_quat','plug_vel','ee_vel',
              'action','action_history','action_sequence4','motion1','motion4','slip_proxy',
              'contact','socket_force','state')
    return {field: np.concatenate([p[field] for p in pieces]) for field in fields}


def ridge_fit(x, target, alpha):
    mu, sigma = target.mean(0), target.std(0)
    sigma = np.where(sigma < 1e-5, 1., sigma)
    yn = (target-mu)/sigma
    mat = x.T @ x
    mat.flat[::len(mat)+1] += alpha
    weight = np.linalg.solve(mat.astype(np.float64), (x.T@yn).astype(np.float64)).astype(np.float32)
    return weight,mu,sigma


def build_blocks(data):
    train = data['train']
    pca = {}
    for name,width,seed in [('rgb',128,5100),('tactile',96,5101)]:
        mean,projection=fit_pca(train[name][train['valid']],width,seed)
        pca[name]=(mean,projection)
    blocks={split:{} for split in SPLITS}
    for split in SPLITS:
        d=data[split]
        for name in ['rgb','tactile']:
            mean,projection=pca[name]
            blocks[split][name]=(d[name]-mean)@projection
        blocks[split]['geometry']=d['geometry']
        blocks[split]['history']=d['history']
        blocks[split]['tactile_past']=d['tactile_past']
    # Scale each block using train frames only. RGB and tactile arrays were
    # already fixed by a separate train-only PCA.
    transforms={}
    for name in ('geometry','history','rgb','tactile','tactile_past'):
        x=blocks['train'][name][train['valid']]
        mean=x.mean(0);std=x.std(0);std=np.where(std<1e-6,1.,std)
        transforms[name]=(mean,std)
        for split in SPLITS:
            blocks[split][name]=np.clip((blocks[split][name]-mean)/std,-8,8)/np.sqrt(x.shape[1])
    return blocks,{k:dict(input_dim=len(v[0]),components=v[1].shape[1]) for k,v in pca.items()}


def design(blocks, mode, action_condition, data):
    fields=MODES[mode]
    out={}
    for split in SPLITS:
        x=[blocks[split][name] for name in fields]
        if action_condition:
            act=data[split]['action'] if action_condition=='current' else data[split]['action_sequence4'].reshape(len(x[0]),24)
            # Action terms use fixed physical-unit scales. They are available
            # offline only for the four-step physical-response diagnostic.
            scale=np.array([10,10,10,3,3,3]*(1 if action_condition=='current' else 4),np.float32)
            x.append(np.clip(act/scale,-8,8)/np.sqrt(len(scale)))
        out[split]=np.concatenate(x,axis=1).astype(np.float32)
    return out


def task_target(d,task):
    return d['action'] if task=='action' else d['motion1'] if task=='motion1' else d['motion4']


def evaluate(target,pred,mask):
    if not mask.any():return {}
    error=pred[mask]-target[mask]
    result={'frames':int(mask.sum()),'xy_rmse_mm':float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
            'xyz_rmse_mm':float(np.sqrt(np.mean(np.sum(error[:,:3]**2,axis=1))))}
    if target.shape[1]==6:
        result['rotation_rmse_deg']=float(np.sqrt(np.mean(np.sum(error[:,3:]**2,axis=1))))
    return result


def fit(args,entries):
    out=args.output/'fit';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        print('fit cached',flush=True);return
    data={split:join(args,entries,split) for split in SPLITS}
    blocks,pca=build_blocks(data)
    rows=[];predictions={}
    # Contact models answer contact-related questions. No near-contact GT is
    # part of the input; it only chooses the train/val/test frame subset.
    for task,condition in [('action',None),('motion1','current'),('motion4','sequence4')]:
        for mode in MODES:
            X=design(blocks,mode,condition,data)
            tr=data['train'];va=data['val'];te=data['test']
            tr_mask=tr['valid']&(tr['contact']>0);va_mask=va['valid']&(va['contact']>0)
            te_mask=te['valid']&(te['contact']>0)
            if tr_mask.sum()<20 or va_mask.sum()<10 or te_mask.sum()<10:raise RuntimeError('Too few contact frames')
            ytr=task_target(tr,task)[tr_mask];yval=task_target(va,task)[va_mask]
            best=None
            for alpha in (.1,1.,10.,100.):
                w,mu,sig=ridge_fit(X['train'][tr_mask],ytr,alpha)
                p=X['val'][va_mask]@w*sig+mu
                score=float(np.mean(((p-yval)/sig)**2))
                if best is None or score<best[0]:best=(score,alpha,w,mu,sig)
            _,alpha,w,mu,sig=best
            ptest=X['test']@w*sig+mu
            target=task_target(te,task)
            rows.append(dict(task=task,mode=mode,alpha=alpha,validation_scaled_mse=best[0],
                             **evaluate(target,ptest,te_mask)))
            predictions[f'{task}_{mode}']=ptest.astype(np.float32)
            print(f'fit {task} {mode} alpha={alpha}',flush=True)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    atomic_json(out/'metrics.json',dict(splits={split:dict(episodes=len({int(v) for v in data[split]['source']}),
        frames=int(data[split]['valid'].sum()),contact_frames=int((data[split]['valid']&(data[split]['contact']>0)).sum()))
        for split in SPLITS},pca=pca,rows=rows))
    np.savez_compressed(out/'test_predictions.npz',**predictions,
                        source=data['test']['source'],frame=data['test']['frame'],
                        valid=data['test']['valid'],contact=data['test']['contact'],
                        action=data['test']['action'],motion1=data['test']['motion1'],motion4=data['test']['motion4'])
    atomic_json(out/'complete.json',dict(version=VERSION,rows=len(rows)))


def paired_intervals(pred,task,baseline,variant):
    source=pred['source'];mask=pred['valid']&(pred['contact']>0)
    y=pred[task];a=pred[f'{task}_{baseline}'];b=pred[f'{task}_{variant}']
    ids=np.unique(source[mask]);pairs=[]
    for episode in ids:
        m=mask&(source==episode)
        pairs.append([np.mean(np.sum((a[m,:2]-y[m,:2])**2,axis=1)),
                      np.mean(np.sum((b[m,:2]-y[m,:2])**2,axis=1))])
    pairs=np.asarray(pairs);rng=np.random.default_rng(20261008)
    sample=rng.integers(len(ids),size=(3000,len(ids)))
    boot=pairs[sample].mean(1);diff=np.sqrt(boot[:,1])-np.sqrt(boot[:,0])
    return dict(variant_minus_baseline_mm=float(np.sqrt(pairs[:,1].mean())-np.sqrt(pairs[:,0].mean())),
                episode_bootstrap95_mm=np.quantile(diff,[.025,.975]).tolist(),
                episodes_improved=int(np.sum(pairs[:,1]<pairs[:,0])),episodes=len(ids))


def match_vector(d):
    n=len(d['source'])
    pose_matrix=Rotation.from_quat(d['rel_quat']).as_matrix().reshape(n,9)
    grip_matrix=Rotation.from_quat(d['grasp_quat']).as_matrix().reshape(n,9)
    width=d['state'][:,7:9].sum(axis=1)[:,None]*1000
    parts=[d['relative']/[4.,4.,5.],pose_matrix/.2,d['grasp_pos']/4.,
           grip_matrix/.2,width/2.,d['plug_vel']/4.,d['ee_vel']/5.,
           d['action_history'][:,:,:3].reshape(n,9)/8.]
    return np.concatenate([x/np.sqrt(x.shape[1]) for x in parts],axis=1).astype(np.float32)


def match_distances(query,pool,qi,possible):
    q=lambda key:query[key][qi]
    p=lambda key:pool[key][possible]
    distance={
      'xy_mm':np.linalg.norm(p('relative')[:,:2]-q('relative')[:2],axis=1),
      'z_mm':np.abs(p('relative')[:,2]-q('relative')[2]),
      'rot_deg':pair_angle_deg(p('rel_quat'),q('rel_quat')),
      'grasp_mm':np.linalg.norm(p('grasp_pos')-q('grasp_pos'),axis=1),
      'grasp_rot_deg':pair_angle_deg(p('grasp_quat'),q('grasp_quat')),
      'width_mm':np.abs(p('state')[:,7:9].sum(axis=1)-q('state')[7:9].sum())*1000,
      'plug_vel_mm':np.linalg.norm(p('plug_vel')-q('plug_vel'),axis=1),
      'ee_vel_mm':np.linalg.norm(p('ee_vel')-q('ee_vel'),axis=1),
      'action_hist_mm':np.sqrt(np.mean(np.sum((p('action_history')[:,:,:3]-q('action_history')[:,:3])**2,axis=2),axis=1)),
      'action_rot_deg':np.sqrt(np.mean(np.sum((p('action_history')[:,:,3:]-q('action_history')[:,3:])**2,axis=2),axis=1)),
      'socket_load':np.abs(p('socket_force')-q('socket_force')),
    }
    return distance


def passing(distance,limits,history):
    selected=['xy_mm','z_mm','rot_deg','grasp_mm','grasp_rot_deg','width_mm']
    if history:selected += ['plug_vel_mm','ee_vel_mm','action_hist_mm','action_rot_deg']
    keys=['xy','z','rot','grasp','grasp_rot','width']
    if history:keys += ['plug_vel','ee_vel','action_hist','action_rot']
    return np.logical_and.reduce([distance[col]<=limits[lim] for col,lim in zip(selected,keys)])


def candidate_score(distance):
    col_to_limit={'xy_mm':'xy','z_mm':'z','rot_deg':'rot','grasp_mm':'grasp',
                  'grasp_rot_deg':'grasp_rot','width_mm':'width',
                  'plug_vel_mm':'plug_vel','ee_vel_mm':'ee_vel',
                  'action_hist_mm':'action_hist','action_rot_deg':'action_rot'}
    return np.mean(np.stack([(distance[k]/MATCH_LIMITS['strict'][v])**2
                             for k,v in col_to_limit.items()],axis=1),axis=1)


def choose_match_cases(records,max_cases):
    # All records were ranked by geometry/history before force and action are read.
    selected=[];used_query=set()
    for tier in ('strict_history','relaxed_history','nearest_unmatched'):
        options=[r for r in records if r['best'][tier] is not None]
        if tier!='nearest_unmatched':
            # Alternate high/low tactile difference only AFTER state/history
            # gates pass. These examples illustrate range, not prevalence.
            options.sort(key=lambda r:(r['best'][tier]['tactile_difference'],r['best'][tier]['score']))
            alternating=[]
            while options:
                alternating.append(options.pop(-1))
                if options:alternating.append(options.pop(0))
            options=alternating
        else:options.sort(key=lambda r:r['best'][tier]['score'])
        for record in options:
            query=int(record['query_source'])
            if query in used_query:continue
            if tier=='relaxed_history' and sum(c['tier']==tier for c in selected)>=3:break
            if tier=='nearest_unmatched' and sum(c['tier']==tier for c in selected)>=3:break
            selected.append(dict(tier=tier,query_split=record['query_split'],query_source=query,
                                 query_frame=record['query_frame'],**record['best'][tier]))
            used_query.add(query)
            if len(selected)>=max_cases:return selected
    return selected


def match(args,entries):
    out=args.output/'match';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():print('match cached',flush=True);return
    train=join(args,entries,'train')
    val=join(args,entries,'val');test=join(args,entries,'test')
    query={k:np.concatenate([val[k],test[k]]) for k in val}
    def eligible(d):
        return d['valid']&(d['contact']>0)&(d['relative'][:,2]>=5)&(d['relative'][:,2]<=30)
    ti=np.flatnonzero(eligible(train));qi=np.flatnonzero(eligible(query))
    tree=cKDTree(match_vector(train)[ti])
    _,neighbors=tree.query(match_vector(query)[qi],k=min(128,len(ti)),workers=1)
    neighbors=np.asarray(neighbors).reshape(len(qi),-1)
    records=[];counts=Counter()
    for item,(index,row) in enumerate(zip(qi,neighbors),1):
        possible=ti[np.atleast_1d(row)]
        distance=match_distances(query,train,index,possible)
        # A shortlist cannot prove zero coverage; expand if it has no strict match.
        if not np.any(passing(distance,MATCH_LIMITS['strict'],True)):
            possible=ti
            distance=match_distances(query,train,index,possible)
        score=candidate_score(distance)
        choices={}
        for tier,limits,hist in [('strict_geometry',MATCH_LIMITS['strict'],False),
                                 ('strict_history',MATCH_LIMITS['strict'],True),
                                 ('relaxed_geometry',MATCH_LIMITS['relaxed'],False),
                                 ('relaxed_history',MATCH_LIMITS['relaxed'],True)]:
            accepted=np.flatnonzero(passing(distance,limits,hist))
            choices[tier]=int(accepted[np.argmin(score[accepted])]) if len(accepted) else None
            if len(accepted):counts[tier]+=1
        choices['nearest_unmatched']=int(np.argmin(score))
        record=dict(query_split='val' if index<len(val['source']) else 'test',
                    query_source=int(query['source'][index]),query_frame=int(query['frame'][index]),best={})
        for tier,local_index in choices.items():
            if local_index is None:record['best'][tier]=None;continue
            j=int(possible[local_index]);r={k:round(float(v[local_index]),5) for k,v in distance.items()}
            r.update(train_source=int(train['source'][j]),train_frame=int(train['frame'][j]),
                     score=round(float(score[local_index]),5),
                     tactile_difference=round(float(np.linalg.norm(train['tactile_mean'][j]-query['tactile_mean'][index])),7),
                     tactile_map_rms=round(float(np.sqrt(np.mean((train['tactile'][j]-query['tactile'][index])**2))),5),
                     action_xy_difference_mm=round(float(np.linalg.norm(train['action'][j,:2]-query['action'][index,:2])),4),
                     train_mean_Fz=[float(v) for v in train['tactile_mean'][j,[2,5]]],
                     query_mean_Fz=[float(v) for v in query['tactile_mean'][index,[2,5]]])
            record['best'][tier]=r
        records.append(record)
        if item%100==0:print(f'match {item}/{len(qi)}',flush=True)
    cases=choose_match_cases(records,args.max_pairs)
    atomic_json(out/'matches.json',dict(version=VERSION,limits=MATCH_LIMITS,
        queries=len(qi),train_pool=len(ti),counts=dict(counts),
        coverage={k:round(counts[k]/len(qi),4) for k in
                  ('strict_geometry','strict_history','relaxed_geometry','relaxed_history')},
        records=records,case_choices=cases))
    atomic_json(out/'complete.json',dict(queries=len(qi),cases=len(cases)))
    print(f"match complete: {len(qi)} queries; strict history {counts['strict_history']}; relaxed history {counts['relaxed_history']}",flush=True)


def episode_lag_curves(args,entries):
    rows=[]
    for split in ('val','test'):
        for entry in [e for e in entries if e['split']==split]:
            d=load_npz(cache_path(args.output,entry));n=len(d['frame'])
            good=d['valid']&(d['contact']>0)&(d['relative'][:,2]>=5)&(d['relative'][:,2]<=30)
            for signal,target in [('left_Fz','slip_proxy'),('right_Fz','slip_proxy'),
                                  ('left_Fx','action_x'),('right_Fx','action_x'),
                                  ('left_Fy','action_y'),('right_Fy','action_y'),
                                  ('left_Fz','action_z'),('right_Fz','action_z'),
                                  ('left_Fz','plug_vel_z'),('right_Fz','plug_vel_z')]:
                channel={'left_Fx':0,'left_Fy':1,'left_Fz':2,
                         'right_Fx':3,'right_Fy':4,'right_Fz':5}[signal]
                f=d['tactile_mean'][:,channel]
                y=d['slip_proxy'] if target=='slip_proxy' else (d['plug_vel'][:,2] if target=='plug_vel_z' else d['action'][:,{'action_x':0,'action_y':1,'action_z':2}[target]])
                for lag in range(-4,5):
                    idx=np.flatnonzero(good & (np.arange(n)+lag>=3) & (np.arange(n)+lag<n))
                    yy=y[idx+lag];xx=f[idx]
                    rho=float(spearmanr(xx,yy).statistic) if len(idx)>=8 and np.std(xx)>1e-9 and np.std(yy)>1e-9 else None
                    rows.append(dict(split=split,source=entry['source_episode'],signal=signal,
                                     target=target,lag=lag,frames=len(idx),spearman=rho))
    return rows


def read_clip(video,frames):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise FileNotFoundError(f'Cannot decode {video}')
    result=[]
    try:
        for number in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(number))
            ok,image=cap.read()
            if not ok:raise RuntimeError(f'Frame {number} missing from {video}')
            result.append(cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA))
    finally:cap.release()
    return result


def paired_video(args,case,entry_by_source,target):
    sources=[(case['query_source'],case['query_frame']),
             (case['train_source'],case['train_frame'])]
    groups=[]
    for source,frame in sources:
        entry=entry_by_source[source]
        vids=image_paths(args.dataset,entry['episode_index'])
        lo,hi=frame-3,frame+5
        frames=np.clip(np.arange(lo,hi),0,entry['length']-1)
        groups.append([read_clip(vids[name],frames) for name in
                       ('front','wrist','tacff_left_preview','tacff_right_preview')])
    width,height=1024,572
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r','3','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    for k in range(8):
        canvas=np.full((height,width,3),20,np.uint8)
        for row,(source,frame) in enumerate(sources):
            y=0 if row==0 else 286
            canvas[y+30:y+286]=np.concatenate([g[k] for g in groups[row]],axis=1)
            cv2.putText(canvas,f"{'QUERY' if row==0 else 'TRAIN'} source {source} frame {frame+k-3}  relative {k-3:+d} step",
                        (7,y+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed to encode H.264 pair clip')


def force_panel(args,case,target,scale):
    samples=[(case['query_split'],case['query_source'],case['query_frame']),
             ('train',case['train_source'],case['train_frame'])]
    fig,axes=plt.subplots(4,3,figsize=(12,10),layout='constrained')
    for side,(split,source,frame) in enumerate(samples):
        d=load_npz(args.output/'episodes'/f'{split}_source_{source:03d}.npz')
        for finger in range(2):
            for axis in range(3):
                ax=axes[side*2+finger,axis]
                grid=d['force_base'][frame,finger,:,axis].reshape(10,14)
                ax.imshow(grid,cmap='RdBu_r',vmin=-scale[axis],vmax=scale[axis],interpolation='nearest')
                ax.set(title=f"{'Query' if side==0 else 'Train'} {['left','right'][finger]} base F{'XYZ'[axis]} mean {grid.mean():+.5f}")
                ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Center-frame force maps · fixed train P99.5 scale · red positive / blue negative')
    fig.savefig(target,dpi=120);plt.close(fig)


def trajectory_panel(args,case,target):
    sources=[(case['query_split'],case['query_source'],case['query_frame']),
             ('train',case['train_source'],case['train_frame'])]
    fig,axes=plt.subplots(3,2,figsize=(13,9),layout='constrained',sharex='col')
    for col,(split,source,frame) in enumerate(sources):
        d=load_npz(args.output/'episodes'/f'{split}_source_{source:03d}.npz')
        lo=max(0,frame-8);hi=min(len(d['frame']),frame+9);x=np.arange(lo,hi)-frame
        f=d['tactile_mean'][lo:hi]
        axes[0,col].plot(x,f[:,2],label='left base Fz');axes[0,col].plot(x,f[:,5],label='right base Fz')
        axes[0,col].set_ylabel('Mean Fz (synthetic)');axes[0,col].legend(fontsize=8)
        action=d['action'][lo:hi]
        axes[1,col].plot(x,action[:,0],label='action X');axes[1,col].plot(x,action[:,1],label='action Y')
        axes[1,col].set_ylabel('Current control XY (mm)');axes[1,col].legend(fontsize=8)
        rel=d['relative'][lo:hi]
        axes[2,col].plot(x,rel[:,0],label='plug X');axes[2,col].plot(x,rel[:,1],label='plug Y')
        axes[2,col].plot(x,rel[:,2],label='plug Z',alpha=.5)
        axes[2,col].plot(x,d['plug_vel'][lo:hi,2],label='prior plug ΔZ',ls='--',alpha=.7)
        axes[2,col].set_ylabel('Relative state and prior motion (mm)');axes[2,col].legend(fontsize=7)
        for ax in axes[:,col]:
            ax.axvline(0,color='#bb5533',lw=1,ls='--');ax.grid(alpha=.15)
            ax.set_xlabel('Control-step offset (one step ≈ 66.7 ms)')
        axes[0,col].set_title(f"{'Query' if col==0 else 'Train'} source {source} / frame {frame}")
    fig.suptitle('Tactile, action and geometry timeline · current tactile precedes current action')
    fig.savefig(target,dpi=120);plt.close(fig)


def lag_chart(rows,target):
    items=[('left_Fz','slip_proxy'),('right_Fz','slip_proxy'),
           ('left_Fx','action_x'),('right_Fx','action_x'),
           ('left_Fy','action_y'),('right_Fy','action_y'),
           ('left_Fz','action_z'),('right_Fz','action_z'),
           ('left_Fz','plug_vel_z'),('right_Fz','plug_vel_z')]
    fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained',sharex=True)
    for ax,(signal,label) in zip(axes.flat,items):
        points=[r for r in rows if r['signal']==signal and r['target']==label]
        values=[];low=[];high=[];counts=[]
        for lag in range(-4,5):
            z=np.array([r['spearman'] for r in points if r['lag']==lag and r['spearman'] is not None],float)
            values.append(float(np.median(z)) if len(z) else np.nan)
            l,h=np.quantile(z,[.25,.75]) if len(z) else (np.nan,np.nan)
            low.append(l);high.append(h);counts.append(len(z))
        x=np.arange(-4,5)
        ax.plot(x,values,marker='o');ax.fill_between(x,low,high,alpha=.2)
        ax.axvline(0,color='black',lw=.7)
        ax.set(title=f'{signal} vs {label} (up to {max(counts)} episodes)',
               xlabel='Target offset from tactile t (control steps)',ylabel='Within-episode Spearman')
        ax.grid(alpha=.15)
    fig.suptitle('Temporal association, not a sensor-delay estimate; +1 action is chosen after tactile t')
    fig.savefig(target,dpi=130);plt.close(fig)



def matched_pair_chart(records,target):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,tier in zip(axes,('strict_history','relaxed_history')):
        values=[r['best'][tier] for r in records if r['best'][tier] is not None]
        if values:
            x=np.array([v['tactile_map_rms'] for v in values]);y=np.array([v['action_xy_difference_mm'] for v in values])
            ax.scatter(x,y,s=9,alpha=.24,color='#3c72a4')
            rho=float(spearmanr(x,y).statistic) if len(x)>=8 and np.std(x)>1e-9 and np.std(y)>1e-9 else np.nan
            ax.set_title(f'{tier}: {len(values)} query frames; rho={rho:.2f}')
        else:
            ax.text(.5,.5,'No matched query frames',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(tier)
        ax.set(xlabel='Bilateral full-force-map log-feature RMS difference',
               ylabel='Demonstrator current XY action difference (mm)')
        ax.grid(alpha=.15)
    fig.suptitle('Action difference versus tactile difference after pre-defined matching; descriptive only')
    fig.savefig(target,dpi=130);plt.close(fig)


def metric_table(rows):
    by={(r['task'],r['mode']):r for r in rows}
    labels={'G_geometry':'full pose + proprioception',
            'H_geometry_history':'+ prior motion/actions',
            'H_RGB':'+ two-view RGB',
            'H_T':'+ current bilateral TacFF',
            'H_RGB_T':'+ RGB + current TacFF',
            'H_RGB_Tpast':'+ RGB + current/prior TacFF'}
    return ''.join('<tr><td>'+html.escape(labels[m])+'</td>'+''.join(
        f"<td>{by[(task,m)]['xy_rmse_mm']:.3f}</td>" for task in ('action','motion1','motion4'))+'</tr>' for m in MODES)


def diff_table(case):
    mapping=[('XY offset','xy_mm','xy'),('Z height','z_mm','z'),('full orientation','rot_deg','rot'),
             ('plug in gripper','grasp_mm','grasp'),('grasp orientation','grasp_rot_deg','grasp_rot'),
             ('gripper opening','width_mm','width'),('plug velocity','plug_vel_mm','plug_vel'),
             ('EE velocity','ee_vel_mm','ee_vel'),('prior 3 actions','action_hist_mm','action_hist'),
             ('prior rotations','action_rot_deg','action_rot'),('socket load (post-match check)','socket_load',None)]
    rows=[]
    for label,key,lim in mapping:
        value=float(case[key]);strict=MATCH_LIMITS['strict'][lim] if lim else None
        relaxed=MATCH_LIMITS['relaxed'][lim] if lim else None
        unit='°' if key in ('rot_deg','grasp_rot_deg','action_rot_deg') else 'N*' if key=='socket_load' else 'mm'
        threshold=f'{strict:g} / {relaxed:g} {unit}' if lim else 'not used for matching'
        status=('strict' if value<=strict else 'relaxed' if value<=relaxed else 'fails') if lim else 'diagnostic'
        rows.append(f'<tr><td>{label}</td><td>{value:.3f} {unit}</td><td>{threshold}</td><td>{status}</td></tr>')
    return ''.join(rows)


def report(args,entries):
    out=args.output
    model=json.loads((out/'fit/metrics.json').read_text());match_result=json.loads((out/'match/matches.json').read_text())
    preds=load_npz(out/'fit/test_predictions.npz')
    contrasts={}
    for task in ('action','motion1','motion4'):
        for baseline,variant in [('G_geometry','H_geometry_history'),
                                 ('H_geometry_history','H_T'),
                                 ('H_geometry_history','H_RGB'),
                                 ('H_RGB','H_RGB_T'),
                                 ('H_RGB_T','H_RGB_Tpast')]:
            contrasts[f'{task}:{baseline}->{variant}']=paired_intervals(preds,task,baseline,variant)
    atomic_json(out/'paired_intervals.json',contrasts)
    lag=episode_lag_curves(args,entries)
    with (out/'lag_by_episode.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=lag[0].keys());writer.writeheader();writer.writerows(lag)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    lag_chart(lag,figdir/'lag_associations.png')
    matched_pair_chart(match_result['records'],figdir/'matched_tactile_action.png')
    entry_by_source={e['source_episode']:e for e in entries}
    cases=[]
    scale=np.asarray(json.loads((args.prior/'cases.json').read_text())['force_color_scale_train_p995'],float)
    for i,choice in enumerate(match_result['case_choices'],1):
        prefix=f'pair_{i:02d}'
        video=out/'cases'/f'{prefix}.mp4';video.parent.mkdir(exist_ok=True)
        if not video.exists():paired_video(args,choice,entry_by_source,video)
        force=out/'cases'/f'{prefix}_force.png';timeline=out/'cases'/f'{prefix}_timeline.png'
        force_panel(args,choice,force,scale);trajectory_panel(args,choice,timeline)
        choice['video']=str(video.relative_to(out));choice['force_image']=str(force.relative_to(out))
        choice['timeline_image']=str(timeline.relative_to(out));cases.append(choice)
        print(f'report pair {i}/{len(match_result["case_choices"])} tier={choice["tier"]}',flush=True)
    atomic_json(out/'cases.json',cases)
    count=match_result['counts'];total=match_result['queries']
    cards=[]
    for i,case in enumerate(cases,1):
        tier=case['tier'];strong=tier=='strict_history'
        label='Full history thresholds passed' if strong else (
            'Relaxed history thresholds only' if tier=='relaxed_history' else 'Nearest example; matching thresholds NOT passed')
        cards.append(f'''<details class="case" {'open' if i<=2 else ''}><summary>Pair {i:02d} · {label} · test/val source {case['query_source']} → train source {case['train_source']}</summary>
<p>Center frame: query {case['query_frame']}, train {case['train_frame']}. Force difference is inspected after matching. The demonstrator action is one choice, not an oracle.</p>
<video controls preload="metadata" playsinline src="{case['video']}"></video>
<div class="split"><img loading="lazy" src="{case['force_image']}"><img loading="lazy" src="{case['timeline_image']}"></div>
<table><thead><tr><th>Match variable</th><th>Difference</th><th>strict / relaxed limit</th><th>Status</th></tr></thead><tbody>{diff_table(case)}</tbody></table>
<p>Left/right mean Fz query: {case['query_mean_Fz']}; train: {case['train_mean_Fz']}. Current control XY difference: {case['action_xy_difference_mm']:.2f} mm.</p></details>''')
    table=metric_table(model['rows'])
    policy_facts='''<p>For action prediction, the input contains no current action. For one-step response, a[t] is supplied and the target is plug motion t→t+1. For four-step response, the complete a[t:t+4] sequence is supplied offline; it is not available to a deployed policy. All metrics are on contact frames of 10 held-out demonstrations, with model regularization chosen on 15 separate validation demonstrations. Matching uses simulator ground truth for analysis only.</p>'''
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ContactWorld dynamic tactile audit</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101923;color:#e9f1fa;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1460px;margin:auto;padding:24px}}a{{color:#9bd5ff}}h1{{font-size:29px}}h2{{margin-top:38px}}.notice{{background:#20354a;border-left:4px solid #86c7ff;padding:15px 20px}}.case{{border:1px solid #425d73;padding:15px;margin:20px 0;border-radius:9px}}summary{{cursor:pointer;font-size:18px;font-weight:650}}video{{width:100%;max-width:1024px;display:block;margin:16px auto}}img{{max-width:100%;background:white}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}table{{border-collapse:collapse;margin:12px 0;max-width:1050px;width:100%}}th,td{{border-bottom:1px solid #415268;text-align:left;padding:7px 9px}}.small{{color:#c0d2e2;font-size:14px}}@media(max-width:820px){{.split{{grid-template-columns:1fr}}main{{padding:10px}}}}
</style></head><body><main><h1>ContactWorld USB：历史影响和匹配状态审查</h1>
<div class="notice"><p>先看匹配覆盖，再看短片。匹配条件由状态和历史预先定义，不用触觉或下一动作选邻居。若门槛未通过，视频仅展示最近候选，不能解释为“相同状态却给出不同触觉”。</p><p>接触是 U 盘—插座合力判据；手指 TacFF 来自夹持接触。每个画面的触觉在标注的当前动作执行前采集。每一控制步约 66.7 ms，视频以 3 帧／秒慢放，不作为物理时间刻度。</p></div>
<h2>1. 可比状态覆盖</h2><p>从训练示范 {match_result['train_pool']} 帧已接触且接近插口的状态中，为独立验证/测试示范的 {total} 帧查找候选。</p>
<table><thead><tr><th>门槛</th><th>有候选的查询帧</th><th>覆盖率</th></tr></thead><tbody>
{''.join(f'<tr><td>{name}</td><td>{count.get(name,0)} / {total}</td><td>{100*count.get(name,0)/total:.1f}%</td></tr>' for name in ('strict_geometry','strict_history','relaxed_geometry','relaxed_history'))}
</tbody></table><img src="figures/matched_tactile_action.png" alt="matched tactile and action differences"><p class="small">上图仅为通过预定门槛的示范差异分布；相邻帧相关，ρ 不能当作独立样本显著性。姿态门槛使用三维旋转测地角；夹持位置相对末端，速度和前三步动作单独核对。插座接触合力仅在匹配后核查差异，不参与检索或门槛。完整门槛见 <a href="match/matches.json">matches.json</a>。</p>
<h2>2. 原始双视图、双指触觉和连续动作</h2><p>每条短片上排为待查示范、下排为训练候选；每排依次是前视、腕视、左指、右指。中心前后各显示数步，保持同一控制步偏移。其下是固定色标的左右 Fx/Fy/Fz 和连续曲线。案例按匹配通过级别标注。高/低触觉差异仅用于在通过门槛的案例中挑选展示例子，不能代表这类状态的发生频率。</p>
{''.join(cards)}
<h2>3. 动作历史、触觉与一步／四步运动</h2>{policy_facts}
<table><thead><tr><th>输入</th><th>下一动作 XY RMSE mm ↓</th><th>给定 a[t] 的一步 XY RMSE mm ↓</th><th>给定四步动作的四步 XY RMSE mm ↓</th></tr></thead><tbody>{table}</tbody></table>
<p>每组差值和以完整示范为单位的区间可见 <a href="paired_intervals.json">paired_intervals.json</a>；完整指标见 <a href="fit/metrics.json">metrics.json</a>。RGB 是冻结 ResNet 特征；触觉力场和 RGB 的 PCA 均只在训练示范拟合。探针说明当前特征能否预测留出示范，不能替代 π0 成功率。</p>
<h2>4. 时间关系</h2><p>下图在每条验证/测试示范内计算相关，再汇总中位数与四分位范围。横轴 0 表示与当前触觉同一观测时刻；正偏移的动作在触觉之后才执行。相关峰值可能受动力学与动作连续性影响，不能直接当作传感器延迟。</p>
<img src="figures/lag_associations.png" alt="lag curves"><p><a href="lag_by_episode.csv">逐示范逐滞后数据 CSV</a> · <a href="cases.json">所示案例数值</a></p>
<p class="small">“可以直接向下插”需要恢复同一动态初态并施加同一动作验证；离线示范没有这种受控反事实，本页不为案例打这个标签。</p>
</main></body></html>'''
    (out/'index.html').write_text(page)
    atomic_json(out/'report_summary.json',dict(version=VERSION,coverage=match_result['coverage'],
        cases=len(cases),metrics=len(model['rows']),lag_rows=len(lag)))
    print(f'REPORT {out/"index.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--prior',type=Path,default=DEFAULT_PRIOR)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--stage',choices=('all',*STAGES),default='all')
    parser.add_argument('--smoke',action='store_true',help='6 train, 2 val, 2 test; use a separate output directory')
    parser.add_argument('--max-pairs',type=int,default=8)
    args=parser.parse_args()
    args.dataset=args.dataset.resolve();args.prior=args.prior.resolve();args.output=args.output.resolve()
    if not (args.prior/'identity.json').exists():parser.error(f'Missing first-run cache {args.prior}')
    if not (args.dataset/'spatial/manifest.json').exists():parser.error(f'Missing dataset {args.dataset}')
    manifest=json.loads((args.dataset/'spatial/manifest.json').read_text())
    entries=select_entries(manifest,args.smoke)
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if args.stage in ('all','prepare'):prepare(args,entries,manifest)
    if args.stage in ('all','fit'):
        if not (args.output/'identity.json').exists():parser.error('prepare stage required')
        fit(args,entries)
    if args.stage in ('all','match'):
        if not (args.output/'identity.json').exists():parser.error('prepare stage required')
        match(args,entries)
    if args.stage in ('all','report'):
        if not ((args.output/'fit/complete.json').exists() and (args.output/'match/complete.json').exists()):
            parser.error('fit and match stages required')
        report(args,entries)
    print(f'DONE elapsed_s={time.monotonic()-started:.1f} output={args.output}',flush=True)


if __name__=='__main__':main()
