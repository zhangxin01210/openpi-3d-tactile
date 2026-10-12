#!/usr/bin/env python3
"""Grouped light probes of recorded short action effects; no policy retraining."""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_contactworld_tactile_action_branches_v3 as replay
from contactworld_neighbor_features import Features, DEFAULT_WEIGHTS
from run_contactworld_action_effect import BRANCHES, VERSION, read, write

MODES=('S','T','TH','C','CT','CTH')
TARGETS=('extra_mm','slip_mm','xy_final_mm','angle_change_deg')
PENALTY=20.


def utility(target):
    extra,slip,xy,angle=np.asarray(target).T
    return extra-2*np.maximum(0,slip-.75)-np.maximum(0,xy-3)-.5*np.maximum(0,angle-3)


def whiten_fit(x):
    mu=x.mean(0);sd=x.std(0)
    sd=np.where(sd<1e-5,1,sd)
    return mu,sd


def whiten(x,mu,sd):
    return np.clip((x-mu)/sd,-5,5)


def estimate(train_x,train_y,test_x,method):
    mu,sd=whiten_fit(train_x)
    x=whiten(train_x,mu,sd);q=whiten(test_x,mu,sd)
    if method=='ridge':
        ym=train_y.mean(0)
        alpha=np.linalg.solve(x@x.T+PENALTY*np.eye(len(x)),train_y-ym)
        return ym+q@x.T@alpha
    if method=='neighbor':
        d=np.sum((q[:,None,:]-x[None,:,:])**2,axis=2)
        near=np.argsort(d,axis=1)[:,:min(5,len(x))]
        weights=1/(1+np.take_along_axis(d,near,axis=1))
        weights/=weights.sum(1,keepdims=True)
        return (weights[:,:,None]*train_y[near]).sum(1)
    raise ValueError(method)


def build_features(args,report):
    selected=[s for s in report['states'] if s['usable']]
    if not selected:
        return None
    extractor=Features(weights=DEFAULT_WEIGHTS,device='cpu',batch=8)
    front=[];wrist=[];cloud=[]
    state=[];past=[];raw=[];history=[];sources=[]
    manifest=read(args.dataset/'spatial/manifest.json')
    capture_root=Path(manifest['capture_root'])
    baseline_cache={}
    for s in selected:
        file=s['accepted_branches']['hold']['file']
        with np.load(file) as z:
            rgb=z['rgb'][0].copy()
            front.append(rgb[:,:256]);wrist.append(rgb[:,256:])
            cloud.append(z['cloud_fused_xyz'].copy())
            state.append(z['state'].copy())
            ah=z['action_history'].copy()
            past.append(np.pad(ah.reshape(-1)[-18:],(max(0,18-ah.size),0))[-18:])
            force_local=z['force_local'][0].reshape(-1).copy()
            force_base=z['force_base'][0].reshape(-1).copy()
            force_hist=z['force_history_local'].reshape(-1,840)
        raw.append(np.r_[force_local,force_base])
        source=s['source']
        if source not in baseline_cache:
            folder=capture_root/f'insertion_usb_episode_{source:03d}'
            capture=replay.load_capture(folder)
            local,_=replay.load_forces(folder)
            contact=replay.source_contact(folder,len(local))
            baseline_cache[source]=replay.stable_baseline(local,contact,capture['action_executed'])
        baseline=baseline_cache[source]
        delta=force_local-(baseline['local'].reshape(-1) if baseline else force_local)
        force_change=force_hist[-1]-force_hist[0]
        history.append(np.r_[delta,force_change,float(baseline is not None)])
        sources.append(source)
    vis=np.concatenate([extractor.rgb(np.stack(front)),extractor.rgb(np.stack(wrist))],1)
    geo=extractor.cloud(np.stack(cloud))
    state=np.asarray(state,np.float32);past=np.asarray(past,np.float32)
    raw=np.asarray(raw,np.float32);history=np.asarray(history,np.float32)
    rows=[];sample_modes={k:[] for k in MODES}
    for i,s in enumerate(selected):
        for kind,entry in s['accepted_branches'].items():
            onehot=np.eye(len(BRANCHES),dtype=np.float32)[BRANCHES.index(kind)]
            context=np.r_[state[i],past[i],vis[i],geo[i]]
            traw=np.r_[raw[i],past[i]]
            th=np.r_[raw[i],history[i],past[i]]
            variants=dict(S=np.r_[state[i],past[i],onehot],
                          T=np.r_[traw,onehot],TH=np.r_[th,onehot],
                          C=np.r_[context,onehot],CT=np.r_[context,raw[i],onehot],
                          CTH=np.r_[context,raw[i],history[i],onehot])
            for mode in MODES:sample_modes[mode].append(variants[mode])
            rows.append(dict(source=s['source'],frame=s['frame'],kind=kind,
                k4=s['selected']['k4'],band=s['selected']['depth_band'],
                target=[entry[k] for k in TARGETS],
                effective=entry['effective'],n=entry['n']))
    arrays={k:np.stack(v).astype(np.float32) for k,v in sample_modes.items()}
    y=np.asarray([r['target'] for r in rows],np.float32)
    groups=np.asarray([r['source'] for r in rows])
    np.savez_compressed(args.output/'probe_features.npz',**arrays,target=y,groups=groups,
                        sources=np.asarray(sources),raw=raw,history=history)
    return rows,arrays,y,groups


def score_predictions(rows,predicted):
    by_source=defaultdict(list)
    for i,r in enumerate(rows):
        by_source[r['source']].append(i)
    table={}
    for source,idx in by_source.items():
        ix=np.asarray(idx)
        actual=np.asarray([rows[j]['target'] for j in ix])
        actual_u=utility(actual)
        estimated_u=utility(predicted[ix])
        kinds=[rows[j]['kind'] for j in ix]
        hold=kinds.index('hold') if 'hold' in kinds else None
        if hold is not None:
            actual_u[hold]=0
            estimated_u[hold]=0
        best=int(np.argmax(actual_u))
        chosen=int(np.argmax(estimated_u))
        actual_effective=bool(any(rows[j]['effective'] for j in ix))
        picked_effective=bool(rows[ix[chosen]]['effective'])
        nonhold=ix[np.asarray(kinds)!='hold']
        table[source]=dict(source=source,rank_regret_mm=float(actual_u[best]-actual_u[chosen]),
            chosen=kinds[chosen],best=kinds[best],
            chosen_utility_mm=float(actual_u[chosen]),
            best_utility_mm=float(actual_u[best]),
            false_stop=bool(kinds[chosen]=='hold' and actual_effective),
            false_continue=bool(kinds[chosen]!='hold' and not actual_effective),
            chose_effective=picked_effective,
            effects_mae_mm=float(np.abs(predicted[nonhold,0]-np.asarray(
                [rows[j]['target'][0] for j in nonhold])).mean()) if len(nonhold) else None,
            slip_mae_mm=float(np.abs(predicted[nonhold,1]-np.asarray(
                [rows[j]['target'][1] for j in nonhold])).mean()) if len(nonhold) else None,
            actions=len(ix))
    return table


def baseline_prediction(rows,y,groups):
    pred=np.zeros_like(y)
    for source in sorted(set(groups)):
        train=groups!=source;test=~train
        for i in np.flatnonzero(test):
            kind=rows[i]['kind']
            pool=[j for j in np.flatnonzero(train) if rows[j]['kind']==kind]
            if not pool:pool=list(np.flatnonzero(train))
            pred[i]=y[pool].mean(0)
    return pred


def bootstrap_contrast(base,other,seed=520,draws=2000):
    shared=sorted(set(base)&set(other))
    if not shared:return None
    diff=np.asarray([base[s]['rank_regret_mm']-other[s]['rank_regret_mm'] for s in shared])
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(diff),size=(draws,len(diff)))
    means=diff[idx].mean(1)
    return dict(improvement_mean_mm=float(diff.mean()),
                interval95_mm=np.quantile(means,[.025,.975]).tolist(),
                improved_states=int((diff>0).sum()),states=len(shared))


def run(args):
    report=read(args.output/'analysis.json')
    if report['version']!=VERSION or report['stage'] not in ('full','pilot'):
        raise ValueError('Analyze action branches before probing')
    usable=[s for s in report['states'] if s['usable']]
    if len(usable)<6:
        result=dict(version=VERSION,status='insufficient_replayable_states',
                    usable=len(usable),minimum=6,
                    note='No cross-state modality ranking is produced. Inspect reconstruction and branch coverage.')
        write(args.output/'probe.json',result)
        (args.output/'probe.html').write_text('<!doctype html><meta charset="utf-8">'
            f'<h1>预测实验未运行</h1><p>可比较状态 {len(usable)}，至少需要 6。'
            '<a href="index.html">查看分支与重建门槛</a>。</p>')
        print('PROBE_INSUFFICIENT',len(usable),flush=True)
        return result
    made=build_features(args,report)
    rows,arrays,y,groups=made
    states=sorted(set(groups))
    rng=np.random.default_rng(20261009)
    rng.shuffle(states)
    folds=[set(x) for x in np.array_split(states,min(5,len(states))) if len(x)]
    prediction={}
    base=baseline_prediction(rows,y,groups)
    prediction['action_mean']=base
    for mode in MODES:
        for method in ('ridge','neighbor'):
            out=np.full_like(y,np.nan)
            for fold in folds:
                test=np.isin(groups,list(fold));train=~test
                out[test]=estimate(arrays[mode][train],y[train],arrays[mode][test],method)
            if not np.isfinite(out).all():
                raise ValueError(f'Nonfinite probe prediction {mode}/{method}')
            prediction[f'{mode}_{method}']=out
    all_metrics=[];per_state={}
    for name,pred in prediction.items():
        scores=score_predictions(rows,pred);per_state[name]=scores
        all_metrics.append(dict(mode=name,states=len(scores),candidate_rows=len(rows),
            effect_mae_mm=float(np.mean([x['effects_mae_mm'] for x in scores.values()])),
            slip_mae_mm=float(np.mean([x['slip_mae_mm'] for x in scores.values()])),
            mean_regret_mm=float(np.mean([x['rank_regret_mm'] for x in scores.values()])),
            false_stop=int(sum(x['false_stop'] for x in scores.values())),
            false_continue=int(sum(x['false_continue'] for x in scores.values())),
            chosen_effective=int(sum(x['chose_effective'] for x in scores.values()))))
    contrasts={name:bootstrap_contrast(per_state['C_ridge'],per_state[name])
               for name in ('CT_ridge','CTH_ridge','T_ridge','TH_ridge')}
    out=dict(version=VERSION,status='exploratory_grouped_complete',
             independent_states=len(states),candidate_rows=len(rows),
             fold_sources=[sorted(map(int,f)) for f in folds],
             metrics=all_metrics,paired_contrasts=contrasts,
             per_state=per_state,
             note='Five grouped folds, fixed model settings, one mean outcome per state/action. '
                  'Only actions that passed replay checks enter a state; missing actions are not negatives. '
                  'Prior pilot selected replayable states, so results condition on reconstruction. '
                  'A linear/nearest-neighbor failure does not prove sensor information is absent.')
    write(args.output/'probe.json',out)
    np.savez_compressed(args.output/'probe_predictions.npz',target=y,groups=groups,
                        **{k:v for k,v in prediction.items()})
    with (args.output/'probe_rows.json').open('w') as stream:
        json.dump(rows,stream,ensure_ascii=False,indent=2)
    metric_html=''.join(f'<tr><td>{html.escape(m["mode"])}</td><td>{m["effect_mae_mm"]:.2f}</td>'
        f'<td>{m["slip_mae_mm"]:.2f}</td><td>{m["mean_regret_mm"]:.2f}</td>'
        f'<td>{m["false_stop"]}</td><td>{m["false_continue"]}</td>'
        f'<td>{m["chosen_effective"]}/{m["states"]}</td></tr>' for m in all_metrics)
    case_html=[]
    for source in sorted(states):
        entry=next(s for s in usable if s['source']==source)
        sid=f's{source:03d}f{entry["frame"]:03d}'
        base_choice=per_state['C_ridge'][source]
        tactile_choice=per_state['CTH_ridge'][source]
        case_html.append(f'<tr><td><a href="index.html#{sid}">{source}</a></td>'
            f'<td>K4 {entry["selected"]["k4"]} · {entry["selected"]["depth_band"]}</td>'
            f'<td>{base_choice["best"]}</td><td>{base_choice["chosen"]}</td>'
            f'<td>{tactile_choice["chosen"]}</td>'
            f'<td>{base_choice["rank_regret_mm"]:.2f}</td>'
            f'<td>{tactile_choice["rank_regret_mm"]:.2f}</td></tr>')
    contrast_html=''.join(f'<tr><td>{k}</td><td>{v["improvement_mean_mm"]:+.2f}</td>'
        f'<td>{[round(z,2) for z in v["interval95_mm"]]}</td>'
        f'<td>{v["improved_states"]}/{v["states"]}</td></tr>' for k,v in contrasts.items())
    (args.output/'probe.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>动作效果预测</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:auto;padding:20px;'
        'background:#142135;color:#e8f2fd}a{color:#9bd7ff}'
        'td,th{border-bottom:1px solid #4f6b84;padding:6px;text-align:left}</style>'
        f'<h1>候选动作的短时效果：按初态留出</h1><p>{len(states)} 个独立来源状态，'
        f'{len(rows)} 个通过重放门槛的状态／动作。'
        '所有同一初态的动作都在同一个折内。旧 K4 8 状态试点按重建质量选取，'
        '本表适用于可重建状态；区间按初态重采样。</p>'
        '<p>S=本体＋历史动作；T=双指当前力图＋历史动作；TH=再加基线残差和力变化；'
        'C=本体＋历史动作＋双视图冻结 ResNet18 特征＋base 点云；'
        'CT/CTH 在 C 上加触觉。每项都给定候选动作。'
        'RGB 和点云编码器不使用任务结果训练。</p>'
        '<p>选择遗憾值：实际最好候选的短时效果减去预测器所选动作的实际效果，越低越好。'
        '“停止”对应保持分支。效果含轴向增量和滑移、偏轴惩罚；'
        '能预测示范动作与能选出有效动作是不同问题。</p>'
        '<h2>留出结果</h2><table><tr><th>输入／方法</th><th>效果误差 mm</th>'
        '<th>滑移误差 mm</th><th>平均选择遗憾 mm</th><th>错停</th><th>错继续</th>'
        f'<th>选中有效动作</th></tr>{metric_html}</table>'
        '<h2>相对 C_ridge 的成对变化</h2><p>正值表示降低选择遗憾；'
        f'区间跨 0 表示本批证据不稳。</p><table><tr><th>比较</th><th>平均改善 mm</th>'
        f'<th>重采样 95% 区间</th><th>改善状态</th></tr>{contrast_html}</table>'
        '<h2>逐状态：看动作和原始视频</h2><table><tr><th>source</th><th>状态</th>'
        '<th>实测最佳</th><th>上下文选择</th><th>加历史触觉选择</th>'
        f'<th>上下文损失</th><th>加触觉损失</th></tr>{"".join(case_html)}</table>'
        '<p><a href="probe.json">全部数值和折</a> · '
        '<a href="probe_rows.json">状态／动作原始标签</a> · '
        '<a href="failure_audit.html">520 条旧评测的症状分层</a></p>')
    print('PROBE_DONE',len(states),'states',len(rows),'candidate actions',flush=True)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=replay.DATA)
    run(p.parse_args())
