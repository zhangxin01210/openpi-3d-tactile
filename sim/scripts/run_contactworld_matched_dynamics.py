#!/usr/bin/env python3
"""Matched ContactWorld USB dynamics: action coverage, bilateral force movies, paired probes.

All matching uses prior pose/grasp/velocity/action history, plus current action
for the one-step response question. Current tactile and outcomes are examined
only after pairing. This is an observational audit, not a causal intervention.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from run_contactworld_history_audit import (
    DEFAULT_DATASET, DEFAULT_OUTPUT as DEFAULT_HISTORY, DEFAULT_PRIOR,
    MATCH_LIMITS, candidate_score, join, load_npz, match_distances, passing,
)
from run_contactworld_sensor_discovery import image_paths

DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_matched_dynamics_20261008')
VERSION = 'cw-matched-dynamics-v1'
TAGS = ('lateral_adjust', 'rotate_adjust', 'stalled_push', 'withdraw',
        'pause', 'descent', 'other_contact', 'pre_contact')
ACTION_GATES = {
    'strict': {'xy_mm': 1.0, 'z_mm': 1.0, 'rot_deg': 0.2},
    'relaxed': {'xy_mm': 3.0, 'z_mm': 3.0, 'rot_deg': 0.6},
}


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def episode_path(history: Path, split: str, source: int):
    return history / 'episodes' / f'{split}_source_{source:03d}.npz'


def action_tags(d):
    """Overlapping descriptive tags; thresholds in physical units per control step."""
    contact = d['contact'] > 0
    action = d['action']
    xy = np.linalg.norm(action[:, :2], axis=1)
    rot = np.linalg.norm(action[:, 3:], axis=1)
    moving_down = action[:, 2] < -2.0
    actual_down = d['motion1'][:, 2] < -0.4
    tags = {
        'lateral_adjust': contact & (xy >= 2.5),
        'rotate_adjust': contact & (rot >= 0.4),
        'stalled_push': contact & moving_down & (d['motion1'][:, 2] > -0.25),
        'withdraw': contact & ((action[:, 2] > 0.5) | (d['motion1'][:, 2] > 0.5)),
        'pause': contact & (np.linalg.norm(action[:, :3], axis=1) < 1.0) & (rot < 0.15),
        'descent': contact & moving_down & actual_down,
        'pre_contact': ~contact,
    }
    union = np.logical_or.reduce([tags[k] for k in tags if k != 'pre_contact'])
    tags['other_contact'] = contact & ~union
    return tags


def segments(mask):
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b-a >= 2]


def eligible(d):
    return d['valid'] & (d['contact'] > 0) & (d['relative'][:, 2] >= 5) & (d['relative'][:, 2] <= 30)


def action_gate_distances(query_action, train_actions):
    diff = train_actions - query_action
    return {'xy_mm': np.linalg.norm(diff[:, :2], axis=1),
            'z_mm': np.abs(diff[:, 2]),
            'rot_deg': np.linalg.norm(diff[:, 3:], axis=1)}


def action_gate_pass(distances, limits):
    return np.logical_and.reduce([distances[k] <= v for k, v in limits.items()])


def summarized_force(d, frame):
    force = d['force_base'][frame]
    return np.r_[force.mean(axis=1).reshape(-1), np.abs(force).mean(axis=1).reshape(-1)].astype(np.float32)


def pose_features(query, pool, qi, pj):
    qrot = Rotation.from_quat(query['rel_quat'][qi]); prot = Rotation.from_quat(pool['rel_quat'][pj])
    qgrip = Rotation.from_quat(query['grasp_quat'][qi]); pgrip = Rotation.from_quat(pool['grasp_quat'][pj])
    return np.r_[query['relative'][qi]-pool['relative'][pj],
                 (prot.inv()*qrot).as_rotvec()*180/np.pi,
                 query['grasp_pos'][qi]-pool['grasp_pos'][pj],
                 (pgrip.inv()*qgrip).as_rotvec()*180/np.pi,
                 (query['state'][qi, 7:9]-pool['state'][pj, 7:9])*1000,
                 query['plug_vel'][qi]-pool['plug_vel'][pj],
                 query['ee_vel'][qi]-pool['ee_vel'][pj],
                 (query['action_history'][qi]-pool['action_history'][pj]).reshape(-1)].astype(np.float32)


def classify_match(record):
    if record['best']['strict_history'] is not None:
        return 'strict_history', record['best']['strict_history']
    if record['best']['relaxed_history'] is not None:
        return 'relaxed_history', record['best']['relaxed_history']
    return None, None


def add_features(row, q, t, qi, ti, current_action=False):
    row['state_delta'] = pose_features(q, t, qi, ti).tolist()
    row['force_summary_delta'] = (summarized_force(q, qi)-summarized_force(t, ti)).tolist()
    row['force_map_delta'] = (q['tactile'][qi]-t['tactile'][ti]).tolist()
    row['rgb_delta'] = (q['rgb'][qi]-t['rgb'][ti]).tolist()
    row['target_action_xy'] = (q['action'][qi, :2]-t['action'][ti, :2]).tolist()
    row['target_action_z'] = float(q['action'][qi, 2]-t['action'][ti, 2])
    row['target_action_rot'] = (q['action'][qi, 3:]-t['action'][ti, 3:]).tolist()
    row['target_motion1_xy'] = (q['motion1'][qi, :2]-t['motion1'][ti, :2]).tolist()
    row['target_motion1_z'] = float(q['motion1'][qi, 2]-t['motion1'][ti, 2])
    has_next = qi+1 < len(q['frame']) and ti+1 < len(t['frame'])
    row['target_grasp_drift_next'] = float(
        np.linalg.norm(q['grasp_pos'][qi+1]-q['grasp_pos'][qi])-
        np.linalg.norm(t['grasp_pos'][ti+1]-t['grasp_pos'][ti])) if has_next else 0.0
    row['action_delta'] = (q['action'][qi]-t['action'][ti]).tolist()
    row['force_map_rms'] = float(np.sqrt(np.mean((q['tactile'][qi]-t['tactile'][ti])**2)))
    row['force_mean_query'] = q['tactile_mean'][qi].tolist()
    row['force_mean_train'] = t['tactile_mean'][ti].tolist()
    row['force_abs_query'] = np.abs(q['force_base'][qi]).mean(axis=1).reshape(-1).tolist()
    row['force_abs_train'] = np.abs(t['force_base'][ti]).mean(axis=1).reshape(-1).tolist()
    row['motion1_query'] = q['motion1'][qi].tolist()
    row['motion1_train'] = t['motion1'][ti].tolist()
    if current_action:
        row['state_delta'].extend(row['action_delta'])
    return row


def build(args):
    root = args.output/'build'
    if (root/'complete.json').exists():
        print('build cached', flush=True)
        return
    entries = json.loads((args.history/'episode_index.json').read_text())
    cache = {}
    def ep(split, source):
        key = split, int(source)
        if key not in cache:
            cache[key] = load_npz(episode_path(args.history, *key))
        return cache[key]
    prior = json.loads((args.history/'match/matches.json').read_text())
    records = prior['records']
    if args.smoke:
        keep = sorted({r['query_source'] for r in records})[:4]
        records = [r for r in records if r['query_source'] in keep][:90]
    train = join(SimpleNamespace(output=args.history), entries, 'train')
    pool_ids = np.flatnonzero(eligible(train))
    rows_a = []
    rows_b = []
    class_stats = defaultdict(lambda: defaultdict(lambda: dict(frames=0, episodes=set(), segments=0)))
    for entry in entries:
        split, source = entry['split'], int(entry['source_episode'])
        d = ep(split, source)
        tags = action_tags(d)
        for tag, mask in tags.items():
            count = int(np.sum(mask & d['valid']))
            if count:
                item = class_stats[split][tag]
                item['frames'] += count
                item['episodes'].add(source)
                item['segments'] += len(segments(mask & d['valid']))
    coverage = Counter()
    for index, rec in enumerate(records, 1):
        split, source, frame = rec['query_split'], int(rec['query_source']), int(rec['query_frame'])
        q = ep(split, source)
        if not eligible(q)[frame]:
            raise ValueError(f'query outside eligible pool: {split} {source} {frame}')
        tags = [name for name, mask in action_tags(q).items() if mask[frame]]
        tier, chosen = classify_match(rec)
        if tier:
            t_source, t_frame = int(chosen['train_source']), int(chosen['train_frame'])
            t = ep('train', t_source)
            base = dict(question='A_action', gate=tier, query_split=split, query_source=source,
                        query_frame=frame, train_source=t_source, train_frame=t_frame,
                        categories=tags, score=float(chosen['score']),
                        match_distances={k: chosen[k] for k in chosen if k in (
                            'xy_mm','z_mm','rot_deg','grasp_mm','grasp_rot_deg','width_mm',
                            'plug_vel_mm','ee_vel_mm','action_hist_mm','action_rot_deg','socket_load')})
            rows_a.append(add_features(base, q, t, frame, t_frame))
            coverage['A_'+tier] += 1
            for tag in tags: coverage['A_'+tag] += 1
        # For B, choose a candidate that matches state/history AND the current
        # executed command. The response is examined only after pairing.
        distances = match_distances(q, train, frame, pool_ids)
        a_dist = action_gate_distances(q['action'][frame], train['action'][pool_ids])
        accepted = None
        for level in ('strict', 'relaxed'):
            mask = passing(distances, MATCH_LIMITS[level], True) & action_gate_pass(a_dist, ACTION_GATES[level])
            indices = np.flatnonzero(mask)
            if len(indices):
                score = candidate_score({k: v[indices] for k, v in distances.items()})
                action_score = sum((a_dist[k][indices]/lim)**2 for k, lim in ACTION_GATES[level].items())
                accepted = (level, int(indices[np.argmin(score+action_score/3)]))
                break
        if accepted:
            level, local = accepted
            t_id = int(pool_ids[local]); t_source = int(train['source'][t_id]); t_frame = int(train['frame'][t_id])
            t = ep('train', t_source)
            if frame+1 >= len(q['frame']) or t_frame+1 >= len(t['frame']):
                continue  # the next-step grasp-drift target must exist for both episodes
            base = dict(question='B_response', gate=level, query_split=split, query_source=source,
                        query_frame=frame, train_source=t_source, train_frame=t_frame,
                        categories=tags, score=float(candidate_score({k:v[local:local+1] for k,v in distances.items()})[0]),
                        match_distances={k:float(v[local]) for k,v in distances.items()},
                        current_action_distance={k:float(v[local]) for k,v in a_dist.items()})
            rows_b.append(add_features(base, q, t, frame, t_frame, current_action=True))
            coverage['B_'+level] += 1
            for tag in tags: coverage['B_'+tag] += 1
        if index%100 == 0: print(f'build matched {index}/{len(records)}', flush=True)
    simple_stats = {split:{tag:dict(frames=x['frames'], episodes=len(x['episodes']), segments=x['segments'])
                          for tag,x in modes.items()} for split,modes in class_stats.items()}
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'coverage.json', dict(version=VERSION, queries=len(records),
        question_A=len(rows_a), question_B=len(rows_b), counts=dict(coverage),
        action_gates=ACTION_GATES, class_thresholds=dict(lateral_action_xy_mm=2.5,
        rotation_action_deg=.4, downward_action_mm=-2, actual_descent_mm=-.4,
        stalled_descent_mm=-.25), class_counts=simple_stats,
        note='Categories can overlap. A matches state/history; B also matches the current action.'))
    atomic_json(root/'pairs_A.json', rows_a)
    atomic_json(root/'pairs_B.json', rows_b)
    atomic_json(root/'complete.json',dict(version=VERSION,A=len(rows_a),B=len(rows_b)))
    print(f'build complete: queries {len(records)}, A {len(rows_a)}, B {len(rows_b)}',flush=True)


def group_folds(groups, maximum=5, seed=20261008):
    distinct=np.unique(groups)
    if len(distinct)<3:
        return []
    rng=np.random.default_rng(seed)
    shuffled=rng.permutation(distinct)
    out=[]
    for held in np.array_split(shuffled,min(maximum,len(distinct))):
        eval_indices=np.flatnonzero(np.isin(groups,held))
        train_indices=np.flatnonzero(~np.isin(groups,held))
        if len(eval_indices) and len(train_indices)>=8:
            out.append((train_indices,eval_indices))
    return out


def pair_arrays(rows):
    return {name:np.asarray([r[name] for r in rows],np.float32) for name in
            ('state_delta','rgb_delta','force_summary_delta','force_map_delta',
             'target_action_xy','target_action_z','target_action_rot',
             'target_motion1_xy','target_motion1_z','target_grasp_drift_next')}


def make_design(arrays,train_ids,eval_ids,mode):
    blocks=['state_delta']
    if 'RGB' in mode:blocks.append('rgb_delta')
    if 'Tsummary' in mode:blocks.append('force_summary_delta')
    if 'Tmap' in mode:blocks.append('force_map_delta')
    train_parts=[];eval_parts=[]
    for name in blocks:
        x=arrays[name][train_ids];e=arrays[name][eval_ids]
        if name in ('rgb_delta','force_map_delta'):
            from run_contactworld_sensor_discovery import fit_pca
            mean,proj=fit_pca(x,32,20261008+(0 if name=='rgb_delta' else 1))
            x=(x-mean)@proj;e=(e-mean)@proj
        mean=x.mean(0);scale=x.std(0);scale=np.maximum(scale,1e-5)
        width=max(1,x.shape[1])
        train_parts.append(np.clip((x-mean)/scale,-8,8)/np.sqrt(width))
        eval_parts.append(np.clip((e-mean)/scale,-8,8)/np.sqrt(width))
    return np.concatenate(train_parts,axis=1),np.concatenate(eval_parts,axis=1)


def ridge_predict(x,y,xe,alpha):
    mu=y.mean(0);centered=y-mu
    gram=(x.T@x).astype(np.float64)
    gram.flat[::len(gram)+1]+=alpha
    weights=np.linalg.solve(gram,(x.T@centered).astype(np.float64))
    return (xe@weights+mu).astype(np.float32)


def choose_alpha(arrays,rows,indices,mode,target):
    groups=np.asarray([r['query_source'] for r in rows])[indices]
    folds=group_folds(groups,maximum=3,seed=20261009)
    if len(folds)<2:return 1.0
    scores=defaultdict(list)
    y=arrays[target]
    if y.ndim==1:y=y[:,None]
    for local_train,local_eval in folds:
        inner_train=indices[local_train];inner_eval=indices[local_eval]
        x,xe=make_design(arrays,inner_train,inner_eval,mode)
        for alpha in (.1,1.,10.,100.):
            pred=ridge_predict(x,y[inner_train],xe,alpha)
            scores[alpha].append(float(np.mean(np.sum((pred-y[inner_eval])**2,axis=1))))
    return min(scores,key=lambda alpha:np.mean(scores[alpha]))


def episode_rmse(y,pred,groups,mask=None):
    if mask is None:mask=np.ones(len(y),bool)
    vals=[]
    for source in np.unique(groups[mask]):
        m=mask&(groups==source)
        vals.append(float(np.mean(np.sum((pred[m]-y[m])**2,axis=1))))
    return float(np.sqrt(np.mean(vals))) if vals else None,vals


def bootstrap_delta(y,left,right,groups,seed=20261008):
    ids=np.unique(groups)
    means=[]
    for source in ids:
        mask=groups==source
        means.append([np.mean(np.sum((left[mask]-y[mask])**2,axis=1)),
                      np.mean(np.sum((right[mask]-y[mask])**2,axis=1))])
    means=np.asarray(means)
    rng=np.random.default_rng(seed)
    pick=rng.integers(len(ids),size=(1500,len(ids)))
    sample=means[pick].mean(axis=1)
    changes=np.sqrt(sample[:,1])-np.sqrt(sample[:,0])
    return dict(delta_mm=float(np.sqrt(means[:,1].mean())-np.sqrt(means[:,0].mean())),
                episode_bootstrap95_mm=np.quantile(changes,[.025,.975]).tolist(),
                improved_episodes=int((means[:,1]<means[:,0]).sum()),episodes=len(ids))


def analyze_one_question(rows,task,smoke):
    if len(rows)<15:return dict(status='insufficient_pairs',pairs=len(rows))
    arrays=pair_arrays(rows)
    target={'A_action_xy':'target_action_xy',
            'A_action_z':'target_action_z',
            'A_action_rot':'target_action_rot',
            'B_motion_xy':'target_motion1_xy',
            'B_motion_z':'target_motion1_z',
            'B_grasp_drift':'target_grasp_drift_next'}[task]
    y=arrays[target]
    if y.ndim==1:y=y[:,None]
    groups=np.asarray([r['query_source'] for r in rows],np.int32)
    folds=group_folds(groups,maximum=3 if smoke else 5)
    if len(folds)<2:return dict(status='insufficient_episodes',pairs=len(rows),episodes=len(np.unique(groups)))
    modes=('state','state_RGB','state_Tsummary','state_Tmap','state_RGB_Tmap')
    predictions={'neighbor':np.zeros_like(y)}
    alpha_by_fold={}
    for mode in modes:
        pred=np.full_like(y,np.nan)
        used=[]
        for train_ids,eval_ids in folds:
            alpha=choose_alpha(arrays,rows,train_ids,mode,target)
            x,xe=make_design(arrays,train_ids,eval_ids,mode)
            pred[eval_ids]=ridge_predict(x,y[train_ids],xe,alpha)
            used.append(alpha)
        if not np.isfinite(pred).all():raise RuntimeError(f'Incomplete grouped predictions for {task} {mode}')
        predictions[mode]=pred
        alpha_by_fold[mode]=used
        print(f'analyze {task} {mode} alphas={used}',flush=True)
    metrics={mode:episode_rmse(y,pred,groups)[0] for mode,pred in predictions.items()}
    contrasts={f'state→{mode}':bootstrap_delta(y,predictions['state'],pred,groups)
               for mode,pred in predictions.items() if mode not in ('neighbor','state')}
    coverage_by_class={}
    for tag in TAGS:
        mask=np.asarray([tag in r['categories'] for r in rows])
        if mask.sum()<10 or len(np.unique(groups[mask]))<3:continue
        coverage_by_class[tag]=dict(frames=int(mask.sum()),episodes=len(np.unique(groups[mask])),
            rmse_mm={mode:episode_rmse(y,pred,groups,mask)[0] for mode,pred in predictions.items()})
    results=dict(status='ok',pairs=len(rows),episodes=len(np.unique(groups)),
                 outcome=target,grouped_rmse_mm=metrics,paired_deltas=contrasts,
                 alpha_by_fold=alpha_by_fold,by_class=coverage_by_class,
                 note='Five outer folds by complete query episode; alpha chosen inside each outer training fold. Each query frame predicted once.')
    prediction_rows=[]
    for i,r in enumerate(rows):
        prediction_rows.append(dict(query_source=r['query_source'],query_frame=r['query_frame'],
            categories=r['categories'],target=y[i].tolist(),
            **{k:v[i].tolist() for k,v in predictions.items()}))
    return results,prediction_rows


def association(rows,question):
    if len(rows)<8:return {}
    x=np.asarray([r['force_map_rms'] for r in rows],float)
    key='target_action_xy' if question=='A' else 'target_motion1_xy'
    y=np.asarray([np.linalg.norm(r[key]) for r in rows],float)
    groups=np.asarray([r['query_source'] for r in rows])
    rho=float(spearmanr(x,y).statistic) if x.std()>1e-9 and y.std()>1e-9 else None
    rng=np.random.default_rng(20261008)
    episodes=np.unique(groups)
    boot=[]
    for _ in range(800):
        sampled=rng.choice(episodes,len(episodes),replace=True)
        ids=np.concatenate([np.flatnonzero(groups==g) for g in sampled])
        if x[ids].std()>1e-9 and y[ids].std()>1e-9:
            boot.append(float(spearmanr(x[ids],y[ids]).statistic))
    return dict(pairs=len(rows),episodes=len(episodes),spearman=rho,
                episode_bootstrap95=np.quantile(boot,[.025,.975]).tolist() if boot else None,
                note='Descriptive association. Pairs share neighboring frames and training examples.')



def signed_force_association(rows,question):
    if len(rows)<8:return {}
    x=np.asarray([np.asarray(r['force_mean_query'])-np.asarray(r['force_mean_train']) for r in rows])
    if question=='A':
        y=np.asarray([np.r_[r['target_action_xy'],r['target_action_z'],r['target_action_rot']] for r in rows])
        labels=['action X','action Y','action Z','action Rx','action Ry','action Rz']
    else:
        y=np.asarray([np.r_[r['target_motion1_xy'],r['target_motion1_z'],r['target_grasp_drift_next']] for r in rows])
        labels=['plug ΔX','plug ΔY','plug ΔZ','grasp drift']
    matrix=np.full((6,y.shape[1]),np.nan)
    for i in range(6):
        for j in range(y.shape[1]):
            if x[:,i].std()>1e-9 and y[:,j].std()>1e-9:
                matrix[i,j]=float(spearmanr(x[:,i],y[:,j]).statistic)
    return dict(force_labels=['L Fx','L Fy','L Fz','R Fx','R Fy','R Fz'],
                outcome_labels=labels,spearman=matrix.tolist(),pairs=len(rows),
                note='Signed query-minus-training differences, after matching. Descriptive correlation; query frames cluster within episodes.')


def signed_force_chart(associations,path):
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,(key,title) in zip(axes,[('A','A: demonstrator action'),('B','B: next-step response')]):
        item=associations[key]
        if not item:
            ax.text(.5,.5,'Insufficient matched pairs',transform=ax.transAxes,ha='center')
            ax.axis('off');continue
        data=np.asarray(item['spearman'],float)
        ax.imshow(data,cmap='RdBu_r',vmin=-.5,vmax=.5,aspect='auto')
        ax.set_xticks(range(len(item['outcome_labels'])),item['outcome_labels'],rotation=35,ha='right')
        ax.set_yticks(range(6),item['force_labels'])
        ax.set_title(f'{title} · {item["pairs"]} matched frames')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if np.isfinite(data[row,col]):
                    ax.text(col,row,f'{data[row,col]:+.2f}',ha='center',va='center',fontsize=8)
    fig.suptitle('Signed TacFF difference versus action/response difference · Spearman ρ · descriptive')
    fig.savefig(path,dpi=130);plt.close(fig)



def policy_observable_summary(args):
    """Re-score earlier held-out predictions by the new action classes.

    The earlier probe used only sensor and robot inputs. Its test episodes have
    been inspected repeatedly in this project, so these are exploratory.
    """
    path=args.prior/'fit/heldout_predictions.npz'
    if not path.exists():return {'status':'prior_predictions_missing'}
    with np.load(path) as z:
        pred={k:z[k].copy() for k in z.files}
    rows=[];modes=('S','S_R','S_Fz','S_Fxy','S_Tsummary','S_Tmap',
                   'S_R_Tmap','S_R_Tmap_history')
    cache={}
    sources=pred['source'].astype(int);frames=pred['frame'].astype(int)
    valid=np.zeros(len(sources),bool)
    class_mask={k:np.zeros(len(sources),bool) for k in TAGS}
    for source in np.unique(sources):
        d=load_npz(episode_path(args.history,'test',int(source)))
        cache[int(source)]=d
        ids=np.flatnonzero(sources==source)
        good=frames[ids]<len(d['frame'])
        valid[ids[good]]=d['valid'][frames[ids[good]]]
        tags=action_tags(d)
        for tag,mask in tags.items():
            class_mask[tag][ids[good]]=mask[frames[ids[good]]]
    actual=pred['target_action']
    categories=('all_contact','lateral_adjust','rotate_adjust','stalled_push',
                'withdraw','pause','descent')
    for category in categories:
        mask=valid & (pred['contact']>0) if category=='all_contact' else valid & class_mask[category]
        if mask.sum()<10 or len(np.unique(sources[mask]))<3:continue
        for mode in modes:
            key=f'{mode}_action_all'
            if key not in pred:continue
            estimate=pred[key]
            score={}
            for name,sl in [('XY',slice(0,2)),('Z',slice(2,3)),('rotation',slice(3,6))]:
                score[name]=episode_rmse(actual[:,sl],estimate[:,sl],sources,mask)[0]
            rows.append(dict(category=category,mode=mode,frames=int(mask.sum()),
                             episodes=int(len(np.unique(sources[mask]))),**score))
    return dict(status='ok',rows=rows,note='Re-analysis of prior test predictions, grouped by complete source episode. Inputs S, RGB and tactile are policy-observable; repeated prior test inspection makes the result exploratory.')


def analyze(args):
    root=args.output/'analysis'
    if (root/'complete.json').exists():print('analysis cached',flush=True);return
    a=json.loads((args.output/'build/pairs_A.json').read_text())
    b=json.loads((args.output/'build/pairs_B.json').read_text())
    root.mkdir(parents=True,exist_ok=True)
    results={};association_results={}
    for task,rows in (('A_action_xy',a),('A_action_z',a),('A_action_rot',a),
                      ('B_motion_xy',b),('B_motion_z',b),('B_grasp_drift',b)):
        outcome=analyze_one_question(rows,task,args.smoke)
        if isinstance(outcome,tuple):
            results[task], predictions=outcome
            atomic_json(root/f'{task}_predictions.json',predictions)
        else:results[task]=outcome
    association_results['A']=association(a,'A');association_results['B']=association(b,'B')
    association_results['signed_A']=signed_force_association(a,'A')
    association_results['signed_B']=signed_force_association(b,'B')
    observable=policy_observable_summary(args)
    atomic_json(root/'metrics.json',dict(version=VERSION,tasks=results,associations=association_results,
        policy_observable=observable,
        interpretation='Observational pair comparison. A predicts demonstrator action. B conditions on the executed current action and predicts next-step plug motion.'))
    atomic_json(root/'complete.json',dict(version=VERSION,A=len(a),B=len(b)))


def choose_cases(a,b,limit):
    selected=[];used=set()
    def take(rows,question,cap):
        picked=0
        for tag in TAGS:
            if tag=='pre_contact':continue
            possibilities=[r for r in rows if tag in r['categories'] and r['query_source'] not in used]
            possibilities.sort(key=lambda r:(r['score'],r['query_source'],r['query_frame']))
            if possibilities:
                r=possibilities[0]
                selected.append({k:r[k] for k in ('question','gate','query_split','query_source',
                    'query_frame','train_source','train_frame','categories','score','match_distances')}
                    | {'selection_tag':tag,'current_action_distance':r.get('current_action_distance')})
                used.add(r['query_source']);picked+=1
                if picked>=cap or len(selected)>=limit:return
        # Fill remaining slots with the closest state matches from new episodes.
        for r in sorted(rows,key=lambda r:r['score']):
            if r['query_source'] in used:continue
            selected.append({k:r[k] for k in ('question','gate','query_split','query_source',
                'query_frame','train_source','train_frame','categories','score','match_distances')}
                | {'selection_tag':'closest_remaining','current_action_distance':r.get('current_action_distance')})
            used.add(r['query_source']);picked+=1
            if picked>=cap or len(selected)>=limit:return
    take(b,'B_response',min(5,limit))
    take(a,'A_action',max(0,limit-len(selected)))
    return selected[:limit]


def read_video_frames(video,indices):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise FileNotFoundError(f'Cannot decode {video}')
    output=[]
    try:
        for frame in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame))
            ok,picture=cap.read()
            if not ok:raise RuntimeError(f'Missing frame {frame} in {video}')
            output.append(cv2.resize(picture,(256,256),interpolation=cv2.INTER_AREA))
    finally:cap.release()
    return output


def encode_frames(path,width,height,frames,fps=5):
    tmp=path.with_suffix('.tmp.mp4')
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','22',
        '-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE)
    first=None
    for frame in frames:
        if frame.shape!=(height,width,3):raise ValueError('Bad frame shape')
        if first is None:first=frame.copy()
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'ffmpeg failed for {path}')
    tmp.replace(path)
    cv2.imwrite(str(path.with_suffix('.jpg')),first,[cv2.IMWRITE_JPEG_QUALITY,88])


def color_force(grid,scale):
    norm=np.clip(grid/scale,-1,1)
    positive=np.maximum(norm,0)[...,None];negative=np.maximum(-norm,0)[...,None]
    white=np.full((*grid.shape,3),245.0,np.float32)
    red=np.array([50,55,225],np.float32)
    blue=np.array([220,90,55],np.float32)
    bgr=white*(1-positive-negative)+positive*red+negative*blue
    return cv2.resize(bgr.astype(np.uint8),(196,140),interpolation=cv2.INTER_NEAREST)


def force_movie_frame(samples,offset,scale):
    tile_w,row_h=210,235
    canvas=np.full((row_h*2,tile_w*6,3),24,np.uint8)
    for row,(split,source,center,d) in enumerate(samples):
        index=int(np.clip(center+offset,0,len(d['frame'])-1));y=row*row_h
        for finger in range(2):
            for axis in range(3):
                col=finger*3+axis;x=col*tile_w
                force=d['force_base'][index,finger,:,axis].reshape(10,14)
                canvas[y+55:y+195,x+7:x+203]=color_force(force,scale[axis])
                name=f"{'QUERY' if row==0 else 'TRAIN'} {['L','R'][finger]} F{'xyz'[axis]}"
                cv2.putText(canvas,name,(x+8,y+20),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(canvas,f'frame {index} | {offset:+d}',(x+8,y+41),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1)
                cv2.putText(canvas,f'mean {force.mean():+.5f}',(x+8,y+215),cv2.FONT_HERSHEY_SIMPLEX,.42,(228,228,228),1)
    return canvas


def rgb_movie_frame(rgb_groups,sources,offset,step):
    width,height=512,572
    canvas=np.full((height,width,3),20,np.uint8)
    for row,((source,center),group) in enumerate(zip(sources,rgb_groups)):
        y=row*286
        canvas[y+30:y+286]=np.concatenate([group[0][step],group[1][step]],axis=1)
        cv2.putText(canvas,f"{'QUERY' if row==0 else 'TRAIN'} source {source} frame {center+offset}  {offset:+d} step",
                    (7,y+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
    return canvas


def timeline_plot(samples,path):
    fig,axes=plt.subplots(3,2,figsize=(13,8),layout='constrained',sharex='col')
    colors=('#d6422b','#2595af','#7f5cbb')
    for col,(split,source,center,d) in enumerate(samples):
        lo=max(0,center-10);hi=min(len(d['frame']),center+16);x=np.arange(lo,hi)-center
        for finger in range(2):
            f=d['tactile_mean'][lo:hi,finger*3:finger*3+3]
            for axis in range(3):
                axes[0,col].plot(x,f[:,axis],color=colors[axis],ls='-' if finger==0 else '--',
                                 label=f"{'L' if finger==0 else 'R'} F{'xyz'[axis]}",alpha=.82)
        for axis in range(3):
            axes[1,col].plot(x,d['action'][lo:hi,axis],color=colors[axis],label=f"a{'xyz'[axis]}")
            axes[2,col].plot(x,d['motion1'][lo:hi,axis],color=colors[axis],label=f"plug Δ{'xyz'[axis]}")
        for row in range(3):
            ax=axes[row,col];ax.axvline(0,color='black',ls=':',lw=1);ax.grid(alpha=.2)
            ax.set_xlabel('Control steps relative to matched frame')
            ax.legend(fontsize=7,ncol=3,loc='best')
        axes[0,col].set_title(f"{'Query' if col==0 else 'Train'} source {source} / frame {center}")
        axes[0,col].set_ylabel('Signed mean TacFF (synthetic)')
        axes[1,col].set_ylabel('Executed control (mm/step)')
        axes[2,col].set_ylabel('Actual plug displacement (mm/step)')
    fig.savefig(path,dpi=125);plt.close(fig)


def render_cases(args):
    out=args.output/'cases';out.mkdir(parents=True,exist_ok=True)
    a=json.loads((args.output/'build/pairs_A.json').read_text())
    b=json.loads((args.output/'build/pairs_B.json').read_text())
    cases=choose_cases(a,b,args.max_cases)
    entries=json.loads((args.history/'episode_index.json').read_text())
    index={int(e['source_episode']):e for e in entries}
    scale=np.asarray(json.loads((args.prior/'cases.json').read_text())['force_color_scale_train_p995'],float)
    output=[]; offsets=list(range(-8,17))
    for number,case in enumerate(cases,1):
        stem=f'case_{number:02d}'
        sources=[(case['query_source'],case['query_frame']),
                 (case['train_source'],case['train_frame'])]
        samples=[];rgb_groups=[]
        for row,(source,frame) in enumerate(sources):
            split=case['query_split'] if row==0 else 'train'
            d=load_npz(episode_path(args.history,split,source))
            samples.append((split,source,frame,d))
            entry=index[source];paths=image_paths(args.dataset,int(entry['episode_index']))
            frames=np.clip(np.asarray(offsets)+frame,0,len(d['frame'])-1)
            rgb_groups.append([read_video_frames(paths[k],frames) for k in ('front','wrist')])
        rgb_path=out/f'{stem}_rgb.mp4';force_path=out/f'{stem}_force.mp4'
        if not rgb_path.exists():
            encode_frames(rgb_path,512,572,
                (rgb_movie_frame(rgb_groups,sources,offset,step) for step,offset in enumerate(offsets)))
        if not force_path.exists():
            encode_frames(force_path,1260,470,
                (force_movie_frame(samples,offset,scale) for offset in offsets))
        trace=out/f'{stem}_timeline.png'
        if not trace.exists():timeline_plot(samples,trace)
        case['rgb_video']=str(rgb_path.relative_to(args.output))
        case['force_video']=str(force_path.relative_to(args.output))
        case['rgb_poster']=str(rgb_path.with_suffix('.jpg').relative_to(args.output))
        case['force_poster']=str(force_path.with_suffix('.jpg').relative_to(args.output))
        case['timeline']=str(trace.relative_to(args.output))
        case['steps']=len(offsets);case['center_video_frame']=8
        output.append(case)
        print(f'render {number}/{len(cases)} {case["question"]} {case["selection_tag"]}',flush=True)
    atomic_json(out/'cases.json',output)


def coverage_chart(coverage,path):
    counts=coverage['class_counts'];pair=coverage['counts']
    tags=[t for t in TAGS if t!='pre_contact']
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    split_colors={'train':'#7489b5','val':'#d6a147','test':'#4eaa87'}
    x=np.arange(len(tags));bottom=np.zeros(len(tags))
    for split in ('train','val','test'):
        value=np.array([counts.get(split,{}).get(tag,{}).get('segments',0) for tag in tags])
        axes[0].bar(x,value,bottom=bottom,label=split,color=split_colors[split]);bottom+=value
    axes[0].set(title='All complete demonstrations: segments ≥2 frames',ylabel='Segments')
    axes[0].legend()
    width=.38
    axes[1].bar(x-width/2,[pair.get('A_'+tag,0) for tag in tags],width,label='A: state+history',color='#5588bb')
    axes[1].bar(x+width/2,[pair.get('B_'+tag,0) for tag in tags],width,label='B: +current action',color='#d39353')
    axes[1].set(title='Matched query frames by category',ylabel='Frames')
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x,tags,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Categories can overlap; counts are descriptive, not independent samples')
    fig.savefig(path,dpi=135);plt.close(fig)


def metrics_chart(results,path):
    tasks=[k for k,v in results['tasks'].items() if v.get('status')=='ok']
    cols=3;rows=max(1,int(np.ceil(len(tasks)/cols)))
    fig,axes=plt.subplots(rows,cols,figsize=(16,4.6*rows),layout='constrained')
    for ax in np.asarray(axes).reshape(-1)[len(tasks):]:ax.axis('off')
    for ax,task in zip(np.asarray(axes).reshape(-1),tasks):
        m=results['tasks'][task]['grouped_rmse_mm']
        labels=list(m);values=[m[k] for k in labels]
        ax.barh(np.arange(len(labels)),values,color=['#8594a8' if k=='neighbor' else '#578eb9' if k=='state' else '#d6a147' for k in labels])
        ax.set_yticks(np.arange(len(labels)),labels);ax.invert_yaxis()
        ax.set(title=task,xlabel='Query-episode-weighted RMSE (task units) ↓')
        ax.grid(axis='x',alpha=.2)
    fig.savefig(path,dpi=130);plt.close(fig)


def case_difference_html(case):
    mapping=[('XY position','xy_mm','xy'),('Z height','z_mm','z'),('relative 3D rotation','rot_deg','rot'),
             ('plug grasp position','grasp_mm','grasp'),('grasp rotation','grasp_rot_deg','grasp_rot'),
             ('gripper width','width_mm','width'),('prior plug velocity','plug_vel_mm','plug_vel'),
             ('prior EE velocity','ee_vel_mm','ee_vel'),('prior 3 translational actions','action_hist_mm','action_hist'),
             ('prior 3 rotational actions','action_rot_deg','action_rot')]
    level='strict' if case['gate'].startswith('strict') else 'relaxed'
    rows=[]
    for label,key,lim in mapping:
        value=case['match_distances'][key];threshold=MATCH_LIMITS[level][lim]
        unit='°' if 'rot' in key else 'mm'
        rows.append(f'<tr><td>{label}</td><td>{value:.2f} {unit}</td><td>{threshold:g} {unit}</td></tr>')
    if case['question']=='B_response':
        for key,label in [('xy_mm','current action XY'),('z_mm','current action Z'),('rot_deg','current action rotation')]:
            value=case['current_action_distance'][key];threshold=ACTION_GATES[level][key]
            unit='°' if 'rot' in key else 'mm'
            rows.append(f'<tr><td>{label}</td><td>{value:.2f} {unit}</td><td>{threshold:g} {unit}</td></tr>')
    rows.append(f'<tr><td>Socket load (post-match check)</td><td>{case["match_distances"]["socket_load"]:.3f}</td><td>not a gate</td></tr>')
    return ''.join(rows)


def report(args):
    out=args.output
    coverage=json.loads((out/'build/coverage.json').read_text())
    results=json.loads((out/'analysis/metrics.json').read_text())
    cases=json.loads((out/'cases/cases.json').read_text())
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    coverage_chart(coverage,figdir/'action_coverage.png')
    metrics_chart(results,figdir/'grouped_predictions.png')
    signed_force_chart({'A':results['associations']['signed_A'],
                        'B':results['associations']['signed_B']},figdir/'signed_force_associations.png')
    assoc=results['associations']
    pair_stat=f'''<p>A 有 {coverage['question_A']} 个匹配查询帧，B 在额外匹配当前动作后有 {coverage['question_B']} 个；两者分母均为 {coverage['queries']}。这些帧来自多条示范，连续帧不是独立实验。A 的触觉图差异与动作 XY 差异的 Spearman ρ={assoc['A'].get('spearman')}；B 的触觉图差异与下一步位移 XY 差异的 ρ={assoc['B'].get('spearman')}。该相关只描述已匹配候选，不能证明触觉导致了动作或位移。</p>'''
    metric_rows=[]
    for task,record in results['tasks'].items():
        if record['status']!='ok':
            metric_rows.append(f'<tr><td>{html.escape(task)}</td><td colspan="3">{record["status"]}: {record["pairs"]} pairs</td></tr>')
            continue
        for mode,value in record['grouped_rmse_mm'].items():
            delta=record['paired_deltas'].get('state→'+mode)
            extra='' if not delta else f'{delta["delta_mm"]:+.3f} [{delta["episode_bootstrap95_mm"][0]:+.3f},{delta["episode_bootstrap95_mm"][1]:+.3f}]'
            metric_rows.append(f'<tr><td>{html.escape(task)}</td><td>{html.escape(mode)}</td><td>{value:.3f}</td><td>{extra}</td></tr>')
    observable_rows=[]
    for item in results.get('policy_observable',{}).get('rows',[]):
        observable_rows.append(f'<tr><td>{item["category"]}</td><td>{item["mode"]}</td><td>{item["episodes"]}</td><td>{item["XY"]:.3f}</td><td>{item["Z"]:.3f}</td><td>{item["rotation"]:.3f}</td></tr>')
    cards=[]
    for number,case in enumerate(cases,1):
        title=f'{case["question"]} · {case["selection_tag"]} · query {case["query_source"]}:{case["query_frame"]} vs train {case["train_source"]}:{case["train_frame"]}'
        cards.append(f'''<details class="case" {'open' if number<=2 else ''}><summary>{html.escape(title)} · {html.escape(case['gate'])}</summary>
<p>上方视频：左右两排分别为查询与训练；两排共享相对控制步。RGB 前视与腕视、双指 base Fx/Fy/Fz 同步显示。蓝色为负、白色为零、红色为正；所有案例使用训练集固定 P99.5 色标。</p>
<div class="controls"><button type="button" class="play">播放／暂停</button><button type="button" class="center">跳到配对帧</button><input type="range" min="0" max="{case['steps']-1}" value="0"><span class="frame">−8</span></div>
<div class="movies"><video muted playsinline preload="metadata" poster="{case['rgb_poster']}" src="{case['rgb_video']}"></video><video muted playsinline preload="metadata" poster="{case['force_poster']}" src="{case['force_video']}"></video></div>
<img class="timeline" loading="lazy" src="{case['timeline']}">
<table><thead><tr><th>匹配变量</th><th>实际差异</th><th>对应门槛</th></tr></thead><tbody>{case_difference_html(case)}</tbody></table>
<p class="small">示例按动作类别分层后选几何分数较近者；用于观察，不代表该类的发生频率。配对仅支持观测关联，不能验证同初态反事实。</p></details>''')
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ContactWorld matched dynamics</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#101a25;color:#e9f0f7;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:24px}}h1{{font-size:30px}}h2{{margin-top:42px}}.intro{{background:#21374b;padding:16px 20px;border-left:4px solid #73b8e7}}a{{color:#97d1fa}}.case{{border:1px solid #48617a;padding:16px;margin:25px 0;border-radius:9px}}summary{{font-size:18px;font-weight:650;cursor:pointer}}img{{max-width:100%;background:#fff}}.movies{{display:grid;grid-template-columns:40% 60%;gap:12px;align-items:start}}video{{display:block;width:100%;background:#111}}.timeline{{display:block;margin:16px auto}}.controls{{display:flex;align-items:center;gap:10px;margin:15px 0;flex-wrap:wrap}}button{{background:#2e4b64;color:#fff;border:1px solid #6687a5;border-radius:5px;padding:8px 11px;cursor:pointer}}input[type=range]{{flex:1;min-width:180px}}table{{border-collapse:collapse;width:100%;max-width:1000px}}th,td{{border-bottom:1px solid #405366;padding:7px 10px;text-align:left}}.small{{color:#b9c9d9;font-size:14px}}@media(max-width:850px){{.movies{{grid-template-columns:1fr}}main{{padding:10px}}}}
</style></head><body><main><h1>ContactWorld USB：匹配状态下的触觉、动作与实际运动</h1>
<div class="intro"><p><strong>先看覆盖，再看案例。</strong>A 在相对位姿、夹持、速度与前三步动作上配对，比较随后示范动作；B 进一步配对当前控制指令，比较执行后的下一步物体位移。两种检验都只用离线示范。触觉和结果均不参与配对。动作类别可以重叠，片段长度至少两个控制步。</p><p>固定训练集力图色标：Fx ±{json.loads((args.prior/'cases.json').read_text())['force_color_scale_train_p995'][0]:.5f}、Fy ±{json.loads((args.prior/'cases.json').read_text())['force_color_scale_train_p995'][1]:.5f}、Fz ±{json.loads((args.prior/'cases.json').read_text())['force_color_scale_train_p995'][2]:.5f}，合成力单位。视频以 5 帧／秒播放，每一帧仍代表一个约 66.7 ms 的控制步。</p></div>
<h2>1. 动作类别与可配对覆盖</h2><img src="figures/action_coverage.png" alt="action category coverage">
<p>完整训练／验证／测试示范中各类片段数和配对查询帧数。A {coverage['question_A']}/{coverage['queries']}，B {coverage['question_B']}/{coverage['queries']}。类别阈值、每类帧数／轨迹数／片段数和门槛见 <a href="build/coverage.json">coverage.json</a>。持续下插会产生大量相邻帧，不能按帧数当作证据强度。</p>{pair_stat}<img src="figures/signed_force_associations.png" alt="signed tactile component associations"><p class="small">每格是已匹配配对的“查询减训练”双指 base 力分量与动作／下一步位移差异的相关；符号随坐标轴方向变化，相关不表示单帧可判定纠偏方向。完整数值见 analysis/metrics.json。</p>
<h2>2. 连续 RGB 和双指三分量力</h2><p>先看非下插类，再看下插类。每个视频从中心帧前 8 步放到后 16 步；进度条和按钮同步控制 RGB 与力图。曲线显示双方 Fx/Fy/Fz 的有符号均值、实际执行的控制量和物体下一步实际位移。</p>{''.join(cards)}
<h2>3. 整条示范留出的配对预测</h2><img src="figures/grouped_predictions.png" alt="episode-grouped prediction errors">
<p>目标是查询动作相对训练邻居的横向、轴向和旋转差值（A），或在当前控制匹配后下一步位移与插件在夹爪坐标系内的相对位移差值（B）。state 输入保留配对后的位姿、夹持、速度与动作历史残差；新增 RGB 为双视图冻结特征；Tsummary 为双指有符号均值与绝对强度；Tmap 为双指完整力场的训练折内 PCA。每一外层折留出整条查询示范；正则系数仅在其余示范内部选择。误差按示范等权汇总。</p>
<table><thead><tr><th>问题</th><th>输入</th><th>RMSE（平移 mm／旋转 °）↓</th><th>相对 state 差值及示范级 95% 区间</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<h2>4. 实际策略可获得的模态</h2><p>复用此前基于本体状态、双视图 RGB 和双指触觉训练的留出预测，按本页动作类别重新计分。S 为关节状态，R 为 RGB，T 为触觉；这一节没有使用精确插件相对位姿作为输入。每条示范等权；类别少于三条示范时不显示。</p><table><thead><tr><th>动作类别</th><th>输入</th><th>示范数</th><th>XY RMSE mm</th><th>Z RMSE mm</th><th>旋转 RMSE °</th></tr></thead><tbody>{''.join(observable_rows)}</tbody></table>
<p class="small">这些是探索性关联检验。测试集已用于本项目多轮观察；区间反映本批示范间波动，并非最终可发表的独立测试置信区间。单指或单分量的更细消融需根据本页覆盖情况选择。</p>
<p><a href="analysis/metrics.json">全部模型与分层指标</a> · <a href="build/pairs_A.json">A 配对数值</a> · <a href="build/pairs_B.json">B 配对数值</a> · <a href="cases/cases.json">展示案例索引</a> · <a href="{args.history/'index.html'}">前次历史匹配审查</a></p>
</main><script>
for(const card of document.querySelectorAll('.case')){{
 const videos=card.querySelectorAll('video'),slider=card.querySelector('input[type=range]'),label=card.querySelector('.frame');
 function seek(index){{const frame=Math.max(0,Math.min(+slider.max,index));slider.value=frame;const second=(frame+.08)/5;for(const v of videos){{v.pause();v.currentTime=second}}label.textContent=(frame-8>=0?'+':'')+(frame-8)+' 控制步';}}
 slider.addEventListener('input',()=>seek(+slider.value));
 card.querySelector('.center').addEventListener('click',()=>seek(8));
 card.querySelector('.play').addEventListener('click',()=>{{if(videos[0].paused){{videos[1].currentTime=videos[0].currentTime;videos[0].play();videos[1].play();}}else{{for(const v of videos)v.pause();}}}});
 videos[0].addEventListener('timeupdate',()=>{{if(videos[0].paused)return;const t=videos[0].currentTime;if(Math.abs(videos[1].currentTime-t)>.12)videos[1].currentTime=t;const frame=Math.min(+slider.max,Math.floor(t*5));slider.value=frame;label.textContent=(frame-8>=0?'+':'')+(frame-8)+' 控制步';}});
 videos[0].addEventListener('ended',()=>videos[1].pause());
}}
</script></body></html>'''
    (out/'index.html').write_text(page)
    atomic_json(out/'report_summary.json',dict(version=VERSION,queries=coverage['queries'],
        A_pairs=coverage['question_A'],B_pairs=coverage['question_B'],shown_cases=len(cases),
        tasks={k:v['status'] for k,v in results['tasks'].items()}))
    print(f'REPORT {out/"index.html"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,default=DEFAULT_HISTORY)
    parser.add_argument('--prior',type=Path,default=DEFAULT_PRIOR)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--stage',choices=('all','build','analyze','render','report'),default='all')
    parser.add_argument('--max-cases',type=int,default=12)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    for key in ('history','prior','dataset','output'):
        setattr(args,key,getattr(args,key).resolve())
    if not (args.history/'match/complete.json').exists():parser.error('Completed history audit is required')
    if not (args.history/'episode_index.json').exists():parser.error('History episode index is missing')
    if not (args.dataset/'spatial/manifest.json').exists():parser.error('Bilateral dataset is missing')
    args.output.mkdir(parents=True,exist_ok=True)
    identity=dict(version=VERSION,history_match_sha256=sha256(args.history/'match/matches.json'),
        history_episode_sha256=sha256(args.history/'episode_index.json'),
        dataset_manifest_sha256=sha256(args.dataset/'spatial/manifest.json'),smoke=args.smoke)
    identity_path=args.output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        parser.error('Output identity differs. Use a new output directory.')
    atomic_json(identity_path,identity)
    started=time.monotonic()
    if args.stage in ('all','build'):build(args)
    if args.stage in ('all','analyze'):
        if not (args.output/'build/complete.json').exists():parser.error('build stage required')
        analyze(args)
    if args.stage in ('all','render'):
        if not (args.output/'build/complete.json').exists():parser.error('build stage required')
        render_cases(args)
    if args.stage in ('all','report'):
        if not ((args.output/'build/complete.json').exists() and (args.output/'analysis/complete.json').exists()
                and (args.output/'cases/cases.json').exists()):
            parser.error('build, analyze and render stages required')
        report(args)
    print(f'DONE elapsed_s={time.monotonic()-started:.1f} output={args.output}',flush=True)


if __name__=='__main__':main()
