"""Seed-grouped, validation-selected linear/MLP probes using existing torch."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from contactworld_slip_core import read, write, save, history

MODES = ('S', 'T', 'T_A', 'TH', 'C', 'CT', 'CTH')
TASKS = ('detect', 'forecast1', 'forecast2', 'forecast4', 'worsen4')


def components(force, proprio, action, baseline, baseline_valid, t):
    past = history(np.r_[np.zeros((1, 6), np.float32), action], t, (0, 1, 2, 3))
    current = force[t]
    change = np.concatenate([current-force[np.maximum(0, t-lag)] for lag in (1, 2, 4)], axis=1)
    delta = (current-baseline[t])*baseline_valid[t, None]
    return dict(S=np.c_[proprio[t], past], T=current, past=past,
                H=np.c_[change, delta, baseline_valid[t]])


def matrix(c, mode):
    planned = mode.endswith('_plan')
    mode = mode.removesuffix('_plan')
    parts = {
        'S': ['S'], 'T': ['T'], 'T_A': ['T', 'past'], 'TH': ['T', 'H', 'past'],
        'C': ['S', 'visual', 'cloud'], 'CT': ['S', 'visual', 'cloud', 'T'],
        'CTH': ['S', 'visual', 'cloud', 'T', 'H'],
    }[mode]
    x = np.concatenate([c[k] for k in parts] + ([c['planned']] if planned else []), axis=1)
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite probe inputs: {mode}')
    return np.ascontiguousarray(x, dtype=np.float32)


def load_data(args):
    from run_contactworld_slip_study import cache_path
    blocks={}; labels={}; valid={}; rows=[]
    for item in read(args.output/'manifest.json'):
        with np.load(cache_path(args,item)) as z, np.load(cache_path(args,item).with_suffix('.views.npz')) as view:
            t=z['steps']; c=components(z['force'],z['proprio'],z['action'],z['baseline'],z['baseline_valid'],t)
            c.update(visual=view['visual'],cloud=view['cloud'],planned=z['planned'][t].reshape(len(t),-1))
            for k,v in c.items():blocks.setdefault(k,[]).append(v)
            for task in TASKS:
                labels.setdefault(task,[]).append(z['y_'+task][t])
                valid.setdefault(task,[]).append(z['valid_'+task][t])
            for h in (1,2,4):valid.setdefault('quiet'+str(h),[]).append(z['valid_quiet'+str(h)][t])
            rows.extend([dict(model=item['model'],seed=item['seed'],step=int(step),dt=item['dt']) for step in t])
    return ({k:np.concatenate(v).astype(np.float32) for k,v in blocks.items()},
            {k:np.concatenate(v) for k,v in labels.items()},
            {k:np.concatenate(v).astype(bool) for k,v in valid.items()},rows)


def splits(groups):
    unique=np.unique(groups).copy();np.random.default_rng(10010).shuffle(unique)
    if len(unique)<5:raise ValueError('Need at least five independent groups')
    folds=np.array_split(unique,5)
    for i in range(5):
        test=np.isin(groups,folds[i]);val=np.isin(groups,folds[(i+1)%5])
        yield i,~(test|val),val,test


def classification_metrics(y,score,threshold):
    y=np.asarray(y,bool);score=np.asarray(score);pred=score>=threshold
    tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum());tn=int((~pred&~y).sum())
    order=np.argsort(-score,kind='stable');ys=y[order];s=score[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1] if len(s) else np.array([],int)
    ap=None;auc=None
    if y.any() and len(ends):
        recall=np.cumsum(ys)[ends]/y.sum();precision=np.cumsum(ys)[ends]/(ends+1)
        ap=float(np.sum(np.diff(np.r_[0,recall])*precision))
    if y.any() and (~y).any():
        ranks=rankdata(score);n1=y.sum();n0=(~y).sum()
        auc=float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else None,ap=ap,auc=auc)


def fit(x,y,train,val,method,epochs,seed,regression=False):
    import torch
    torch.manual_seed(seed)
    mu=x[train].mean(0);sd=x[train].std(0);sd=np.where(sd<1e-8,1.,sd)
    xx=np.clip((x-mu)/sd,-6,6).astype(np.float32)
    ym=np.mean(y[train],0) if regression else 0.
    yd=np.maximum(np.std(y[train],0),.1) if regression else 1.
    yy=((y-ym)/yd).astype(np.float32)
    if yy.ndim==1:yy=yy[:,None]
    outputs=yy.shape[1]
    model=(torch.nn.Linear(x.shape[1],outputs) if method=='linear' else torch.nn.Sequential(
        torch.nn.Linear(x.shape[1],64),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(64,outputs)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    xtr=torch.from_numpy(xx[train]);ytr=torch.from_numpy(yy[train]);xval=torch.from_numpy(xx[val]);yval=torch.from_numpy(yy[val])
    positive=float(ytr.sum())
    weight=torch.tensor(min(20.,max(.1,(len(ytr)-positive)/max(positive,1.))))
    loss=(torch.nn.SmoothL1Loss() if regression else torch.nn.BCEWithLogitsLoss(pos_weight=weight))
    best=float('inf');saved=None;bad=0
    for epoch in range(epochs):
        model.train();indices=torch.randperm(len(xtr))
        for batch in indices.split(512):
            optimizer.zero_grad(set_to_none=True);err=loss(model(xtr[batch]),ytr[batch]);err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        model.eval()
        with torch.no_grad():value=float(loss(model(xval),yval))
        if value<best-1e-5:
            best=value;saved={k:v.detach().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=6:break
    if saved is None:raise ValueError('Probe training produced no finite validation loss')
    model.load_state_dict(saved);model.eval()
    with torch.no_grad():raw=torch.cat([model(batch) for batch in torch.from_numpy(xx).split(1024)]).numpy()
    pred=raw*yd+ym if regression else 1/(1+np.exp(-np.clip(raw[:,0],-30,30)))
    bundle=dict(state_dict=saved,mean=mu,scale=sd,target_mean=ym,target_scale=yd,
                method=method,regression=regression,validation_loss=best)
    return pred,bundle


def infer_bundle(x,bundle):
    import torch
    sdct=bundle['state_dict'];output=sdct['weight'].shape[0] if bundle['method']=='linear' else sdct['3.weight'].shape[0]
    model=(torch.nn.Linear(x.shape[1],output) if bundle['method']=='linear' else torch.nn.Sequential(
        torch.nn.Linear(x.shape[1],64),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(64,output)))
    model.load_state_dict(sdct);model.eval()
    x=np.clip((x-bundle['mean'])/bundle['scale'],-6,6).astype(np.float32)
    with torch.no_grad():raw=torch.cat([model(batch) for batch in torch.from_numpy(x).split(1024)]).numpy()
    return raw*bundle['target_scale']+bundle['target_mean'] if bundle['regression'] else 1/(1+np.exp(-np.clip(raw[:,0],-30,30)))


def fast_tactile(args,tasks):
    """Same TH fold models and validation cutoffs, sampled every control step."""
    import torch
    from run_contactworld_slip_study import cache_path
    manifest=read(args.output/'manifest.json');results=[]
    for task in tasks:
        models={};lookup={}
        for path in sorted((args.output/'probes'/f'{task}_TH').glob('fold_*.json')):
            info=read(path);fold=info['fold']
            models[fold]=(torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False),info['threshold'])
            for seed in info['test_seeds']:lookup[seed]=fold
        rows=[];ys=[];scores=[];cuts=[];masks=[];quiet=[]
        for item in manifest:
            if item['seed'] not in lookup:continue
            bundle,threshold=models[lookup[item['seed']]]
            with np.load(cache_path(args,item)) as z:
                t=np.arange(item['n'])
                c=components(z['force'],z['proprio'],z['action'],z['baseline'],z['baseline_valid'],t)
                pred=infer_bundle(matrix(c,'TH'),bundle)
                ys.append(z['y_'+task][t]);masks.append(z['valid_'+task][t]);scores.append(pred);cuts.append(np.full(len(t),threshold))
                quiet.append(z['valid_quiet'+task[-1]][t] if task.startswith('forecast') else np.ones(len(t),bool))
            rows.extend([dict(model=item['model'],seed=item['seed'],step=int(step),dt=item['dt']) for step in t])
        if not rows:continue
        y=np.concatenate(ys);score=np.concatenate(scores);cut=np.concatenate(cuts);mask=np.concatenate(masks);q=np.concatenate(quiet)
        metrics=classification_metrics(y[mask],score[mask],cut[mask]);metrics['mean_fold_ap']=None
        metrics['rollouts_with_false_alarm']=len({(rows[i]['model'],rows[i]['seed']) for i in np.flatnonzero(mask&(score>=cut)&(y==0))})
        metrics.update(event_metrics(rows,mask,score>=cut,manifest,task))
        if task.startswith('forecast'):metrics['quiet_subset']=classification_metrics(y[mask&q],score[mask&q],cut[mask&q])
        results.append(dict(task=task,mode='TH_every_step',metrics=metrics,
                            note='Same TH query-trained models/thresholds. Denser decision cadence; no extra training or test threshold tuning.'))
        save(args.output/'probes'/f'{task}_TH_every_step.npz',score=score,threshold=cut,target=y,eligible=mask,
             groups=np.array([r['seed'] for r in rows]),steps=np.array([r['step'] for r in rows]))
        print('fast tactile',task,'rows',len(rows),flush=True)
    return results


def event_metrics(rows,mask,pred,manifest,task):
    by={}
    for i in np.flatnonzero(mask):by.setdefault((rows[i]['model'],rows[i]['seed']),[]).append(i)
    hits=events=0;timings=[];false_alarms=0
    for item in manifest:
        indices=by.get((item['model'],item['seed']),[])
        alarm_steps=[rows[i]['step'] for i in indices if pred[i]]
        observed_steps=[rows[i]['step'] for i in indices]
        if not observed_steps:continue
        for onset in (item['onsets'] if task=='detect' else item['event_starts']):
            h=4 if task=='detect' else int(task[-1]) if task.startswith('forecast') else 0
            eligible=[t for t in observed_steps if (onset<=t<=onset+4 if task=='detect' else onset-h<=t<onset)]
            if not eligible:continue
            events+=1
            alarm=[t for t in alarm_steps if t in eligible]
            if alarm:hits+=1;timings.append((onset-min(alarm))*item['dt']*1000)
    return dict(observable_events=events,event_hits=hits,event_recall=hits/events if events else None,
                median_lead_ms=float(np.median(timings)) if timings else None,
                timing_note='Forecast lead is before first threshold-exceeding frame of a confirmed event; detection delay is from causal confirmation. Conditional on eligible frames.')


def bootstrap_error(y,a,b,groups,iterations=500):
    # Each seed contributes its confusion counts, not an independent count per frame.
    unique=np.unique(groups);counts=[]
    for seed in unique:
        ix=groups==seed
        counts.append([[(y[ix]&~p[ix]).sum(),y[ix].sum(),(~y[ix]&p[ix]).sum(),(~y[ix]).sum()] for p in (a,b)])
    counts=np.asarray(counts,float);rng=np.random.default_rng(810);delta=[]
    for _ in range(iterations):
        c=counts[rng.integers(0,len(unique),len(unique))].sum(0)
        if (c[:,[1,3]]==0).any():continue
        err=(c[:,0]/c[:,1]+c[:,2]/c[:,3])/2
        delta.append(err[1]-err[0])
    return np.quantile(delta,[.025,.5,.975]).tolist() if delta else None


def probes(args):
    import torch
    from run_contactworld_slip_study import status
    torch.set_num_threads(args.threads)
    c,ys,valid,rows=load_data(args);groups=np.array([r['seed'] for r in rows])
    manifest=read(args.output/'manifest.json');results=[];case_candidates=[];predictions={}
    tasks=('detect','forecast4') if args.smoke else TASKS
    modes=MODES
    for task in tasks:
        mask=valid[task];y=ys[task]
        for mode in list(modes)+([] if not task.startswith('forecast') else ['S_plan','CTH_plan']):
            output=args.output/'probes'/f'{task}_{mode}';output.mkdir(parents=True,exist_ok=True)
            x=matrix(c,mode);score=np.full(len(y),np.nan);cut=np.full(len(y),np.nan);fold_reports=[]
            for fold,train,val,test in splits(groups):
                train &= mask;val &= mask;test &= mask
                dest=output/f'fold_{fold}.npz';meta=output/f'fold_{fold}.json'
                if dest.exists() and meta.exists():
                    with np.load(dest) as old:score[test]=old['score'];cut[test]=old['threshold']
                    fold_reports.append(read(meta));continue
                if len(np.unique(y[train]))<2 or len(np.unique(y[val]))<2 or not test.any():
                    fold_reports.append(dict(fold=fold,status='insufficient_class_coverage'));continue
                best=None
                for method in ('linear','mlp'):
                    pred,bundle=fit(x,y,train,val,method,args.epochs,fold)
                    if best is None or bundle['validation_loss']<best[1]['validation_loss']:best=(pred,bundle)
                pred,bundle=best
                threshold=float(np.nextafter(np.quantile(pred[val & (y==0)],.95),np.inf))
                score[test]=pred[test];cut[test]=threshold
                info=dict(fold=fold,method=bundle['method'],threshold=threshold,
                          train_seeds=np.unique(groups[train]).tolist(),val_seeds=np.unique(groups[val]).tolist(),
                          test_seeds=np.unique(groups[test]).tolist(),metrics=classification_metrics(y[test],score[test],threshold))
                save(dest,score=score[test],threshold=np.full(test.sum(),threshold));write(meta,info)
                temporary=output/f'fold_{fold}.tmp.pt';torch.save(bundle,temporary);temporary.replace(output/f'fold_{fold}.pt')
                fold_reports.append(info);print('probe',task,mode,'fold',fold,flush=True)
            eligible=mask&np.isfinite(score);pred=score>=cut
            metrics=classification_metrics(y[eligible],score[eligible],cut[eligible])
            fold_ap=[r['metrics']['ap'] for r in fold_reports if r.get('metrics',{}).get('ap') is not None]
            metrics['mean_fold_ap']=float(np.mean(fold_ap)) if fold_ap else None
            alarms=set((rows[i]['model'],rows[i]['seed']) for i in np.flatnonzero(eligible&pred&(y==0)))
            metrics['rollouts_with_false_alarm']=len(alarms)
            metrics.update(event_metrics(rows,eligible,pred,manifest,task))
            if task.startswith('forecast'):
                ix=eligible&valid['quiet'+task[-1]]
                metrics['quiet_subset']=classification_metrics(y[ix],score[ix],cut[ix])
            info=dict(task=task,mode=mode,metrics=metrics,folds=fold_reports)
            results.append(info);predictions[(task,mode)]=(eligible,pred)
            save(output/'predictions.npz',score=score,threshold=cut,target=y,eligible=eligible,groups=groups)
            for title,idx in [('误报：触觉探针报警但窗口内未出现标签事件',np.flatnonzero(eligible&pred&(y==0))),
                              ('漏检：标签事件未被探针识别',np.flatnonzero(eligible&~pred&(y==1))),
                              ('命中：请核对力变化和物理事件的先后',np.flatnonzero(eligible&pred&(y==1)))]:
                if mode=='TH':
                    for i in idx[::max(1,len(idx)//3)][:3]:case_candidates.append(dict(**rows[i],title=task+' · '+title))
            write(args.output/'probe_results.json',dict(results=results,rows=rows))
            del x
    contrasts=[]
    for task in tasks:
        for a,b in [('S','TH'),('T_A','TH'),('C','CT'),('C','CTH'),('CT','CTH')]:
            if (task,a) not in predictions or (task,b) not in predictions:continue
            ma,pa=predictions[(task,a)];mb,pb=predictions[(task,b)];ix=ma&mb
            ci=bootstrap_error(ys[task][ix].astype(bool),pa[ix],pb[ix],groups[ix]) if ix.any() else None
            contrasts.append(dict(task=task,reference=a,added=b,delta_balanced_error_95ci=ci))
    results.extend(fast_tactile(args,tasks))
    write(args.output/'probe_results.json',dict(results=results,rows=rows))
    write(args.output/'probe_contrasts.json',contrasts);write(args.output/'probe_cases.json',case_candidates)
    table=[]
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    for r in results:
        m=r['metrics'];q=m.get('quiet_subset',{})
        table.append(f'<tr><td>{r["task"]}</td><td>{r["mode"]}</td><td>{m["positive"]}/{m["n"]}</td>'
                     f'<td>{fmt(m["mean_fold_ap"])}</td><td>{fmt(m["recall"])}</td><td>{fmt(m["fpr"])}</td>'
                     f'<td>{m["rollouts_with_false_alarm"]}</td><td>{fmt(m["median_lead_ms"])}</td>'
                     f'<td>{fmt(q.get("recall"))} / {fmt(q.get("fpr"))} ({q.get("positive",0)}/{q.get("n",0)})</td></tr>')
    page='<html><meta charset="utf-8"><style>body{font:16px system-ui;margin:25px}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}</style>'
    page+='<h1>滑移检测与预警：初态分组留出</h1><a href="index.html">案例与时间线</a>'
    page+='<p>S=关节状态＋过去4步动作；T=完整双指局部840维力场；T_A=T＋动作历史；TH=T_A＋1/2/4步变化＋有效静态基线差分；'
    page+='C=S＋冻结双视图全图及固定裁剪特征＋点云；CT/CTH追加触觉。plan追加已生成的4步动作。'
    page+='所有对照在相同重规划时刻取样，触觉历史为每控制步记录。没有输入USB真值位姿、接触真值或未来观测。</p>'
    page+='<p>5折按seed划分：每折60%训练、20%验证、20%测试；验证选择线性/小MLP与早停，并将验证负例误报率目标设为5%。'
    page+='测试误报率可能不同。分数未做概率校准。quiet列只含最近两帧均未超过运动阈值的预警样本；预测提前量相对已确认事件的首个超阈值帧，'
    page+='不是相对晚一拍的确认时刻。检测延迟仍相对因果确认时刻。末尾不足以验证连续确认的样本已排除；首超阈值标签仅用于事后评分。</p>'
    page+='<p>TH_every_step使用相同已训练TH模型与阈值，但每一步运行检测；请同时看误报和提前量，不能只比较召回。'
    page+='视觉为冻结ResNet18全图/固定裁剪特征，点云为空间投影描述；这些探针无法利用某模态，不等于该模态没有信息。</p>'
    page+='<table><tr><th>任务</th><th>输入</th><th>正例/有效样本</th><th>平均折AP</th><th>召回</th><th>误报率</th><th>有误报轨迹数</th><th>事件中位提前量ms</th><th>quiet召回/误报</th></tr>'+''.join(table)+'</table>'
    page+='<h2>同一初态配对变化</h2><p>Δ平衡错误率小于0为改善；按seed重采样95%区间。模型、相邻帧均不是独立初态。</p><pre>'+html.escape(str(contrasts))+'</pre>'
    (args.output/'probe_report.html').write_text(page)
    status(args,'probes','complete',comparisons=len(results),note='Exploratory labels need video audit; no manual acceptance assumed.')


def cost(y,reference):
    # Explicit diagnostic tradeoff, not a physical law or final task success.
    floor=max(.5,.75*float(reference[2]))
    return y[:,0]+.25*y[:,1]+2*np.maximum(0,floor-y[:,2])+np.maximum(0,y[:,3]-3.)


def branch_probes(args):
    from run_contactworld_slip_study import status
    result_files=sorted((args.output/'branches').glob('*/result.json'))
    states=[read(p) for p in result_files]
    usable=[s for s in states if s['usable']]
    if not usable:
        status(args,'branch-probes','pending',reason='No accepted matched action tables yet')
        return
    from contactworld_neighbor_features import Features
    encoder=Features(device='cpu',batch=16)
    import torch
    torch.set_num_threads(args.threads)
    blocks={};y=[];groups=[];kinds=[];descriptive=[]
    for s in usable:
        with np.load(Path(s['folder'])/'pre.npz') as z:
            force=z['force_history'];actions=z['action_history'];state=z['proprio_history'];t=np.array([len(force)-1])
            c=components(force,state,actions,z['baseline_history'],z['baseline_valid'],t)
            c['visual']=np.concatenate([encoder.rgb(z['front'][None]),encoder.rgb(z['wrist'][None]),
                                        encoder.rgb(z['front'][None,64:224,48:208]),encoder.rgb(z['wrist'][None,48:208,48:208])],1)
            c['cloud']=encoder.cloud(z['cloud_xyz'][None])
        outcomes=s['effects'];ref=np.array([outcomes['continue'][k] for k in ('slip_mm','rotation_deg','extra_progress_mm','xy_final_mm')])
        for i,kind in enumerate(s['kinds']):
            outcome=outcomes[kind];target=np.array([outcome[k] for k in ('slip_mm','rotation_deg','extra_progress_mm','xy_final_mm')])
            for name,v in c.items():blocks.setdefault(name,[]).append(v[0])
            record=next(r for r in s['records'] if r['kind']==kind and r['accepted'])
            with np.load(record['file']) as z:blocks.setdefault('candidate_action',[]).append(z['action'].reshape(-1))
            kinds.append(kind);groups.append(s['seed']);y.append(target)
        helpful=[kind for kind in s['kinds'] if kind not in ('continue','hold') and
                 all(r['slip_mm']<ref[0]-.5 and r['progress_mm']-outcomes['hold']['progress_mm']>=max(.5,.75*ref[2]) and
                     r['xy_final_mm']<=max(3.,ref[3]+.5) and r['rotation_deg']<=ref[1]+1.
                     for r in outcomes[kind]['all_repeats'])]
        descriptive.append(dict(seed=s['seed'],state=s['id'],group=s['group'],helpful=helpful,effects=outcomes))
    c={k:np.asarray(v,np.float32) for k,v in blocks.items()};y=np.asarray(y,np.float32);groups=np.asarray(groups)
    results=[];details=[]
    if len(np.unique(groups))>=8:
        for mode in ('S','T','TH','C','CT','CTH'):
            x0=np.c_[matrix(c,mode),c['candidate_action']]
            # State/action interactions are necessary for state-dependent action rankings.
            encoded=np.eye(len(usable[0]['kinds']),dtype=np.float32)[[usable[0]['kinds'].index(k) for k in kinds]]
            x=np.c_[x0,encoded,np.einsum('ni,nj->nij',x0[:,::max(1,x0.shape[1]//128)],encoded).reshape(len(x0),-1)]
            prediction=np.full_like(y,np.nan)
            for fold,train,val,test in splits(groups):
                best=None
                for method in ('linear','mlp'):
                    pred,bundle=fit(x,y,train,val,method,args.epochs,fold,regression=True)
                    if best is None or bundle['validation_loss']<best[1]['validation_loss']:best=(pred,bundle)
                prediction[test]=best[0][test]
                path=args.output/'branch_probe_models'/f'{mode}_fold_{fold}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(best[1],path)
            regrets=[];chosen=[]
            for seed in np.unique(groups):
                ix=np.flatnonzero(groups==seed);names=[kinds[i] for i in ix];ref=names.index('continue')
                true_cost=cost(y[ix],y[ix[ref]]);pred_cost=cost(prediction[ix],prediction[ix[ref]])
                forecast=prediction[ix];floor=max(.5,.75*float(forecast[ref,2]))
                admissible=(forecast[:,2]>=floor)&(forecast[:,3]<=max(3.,float(forecast[ref,3])+.5))
                admissible &= np.array([k!='hold' for k in names])
                risk=forecast[:,0]+.25*forecast[:,1]
                selected=int(np.argmin(np.where(admissible,risk,np.inf))) if admissible.any() else ref
                regret=float(true_cost[selected]-true_cost.min());regrets.append(regret)
                retained=bool(y[ix[selected],2]>=max(.5,.75*float(y[ix[ref],2])))
                helped=names[selected] in next(r['helpful'] for r in descriptive if r['seed']==seed)
                chosen.append(dict(seed=int(seed),action=names[selected],regret=regret,
                                   abstained=not bool(admissible.any()),progress_retained=retained,helped=helped,
                                   actual=y[ix[selected]].tolist(),oracle=names[int(np.argmin(true_cost))]))
            rng=np.random.default_rng(10010);boot=[np.mean(rng.choice(regrets,len(regrets),replace=True)) for _ in range(500)]
            result=dict(mode=mode,mean_regret=float(np.mean(regrets)),regret_95ci=np.quantile(boot,[.025,.975]).tolist(),
                        mean_slip_mae_mm=float(np.abs(y[:,0]-prediction[:,0]).mean()),choices=chosen,
                        progress_retention_rate=float(np.mean([r['progress_retained'] for r in chosen])),
                        helped_states=sum(r['helped'] for r in chosen),abstained=sum(r['abstained'] for r in chosen))
            results.append(result)
            save(args.output/'branch_probe_predictions'/f'{mode}.npz',target=y,prediction=prediction,groups=groups)
        # Fixed-action baselines use exactly the same states and cost definition.
        fixed=[]
        for kind in usable[0]['kinds']:
            regret=[]
            for seed in np.unique(groups):
                ix=np.flatnonzero(groups==seed);names=[kinds[i] for i in ix];values=cost(y[ix],y[ix[names.index('continue')]])
                regret.append(float(values[names.index(kind)]-values.min()))
            fixed.append(dict(kind=kind,mean_regret=float(np.mean(regret))))
    else:fixed=[]
    write(args.output/'branch_probe_results.json',dict(usable=len(usable),descriptive=descriptive,models=results,fixed=fixed,
          note='Targeted states, open-loop fixed commands, continuous relative-motion labels; not closed-loop pi0 rescue rates.'))
    page='<html><meta charset="utf-8"><style>body{font:16px system-ui;margin:25px}td,th{padding:5px;border:1px solid #aaa}video{width:48%}</style>'
    page+='<h1>同状态动作对照</h1><a href="index.html">返回总报告</a><p>滑移为夹爪坐标中的相对运动。所有分支16步；'
    page+='continue重复起点已知的4步指令；resume返回该固定指令，不是重新查询π0。暂停和保持只作为对照，减少运动不自动算防滑控制成功。</p>'
    page+='<p>效用：滑移mm＋0.25×转角°＋2×推进不足mm＋离轴超出3mm的部分。推进目标为max(0.5mm,原动作推进×75%)；'
    page+='此权重是诊断约定，各连续量均保留。可用初态少于8时只显示实测效果，不拟合动作选择器。</p>'
    page+='<p>模型的推进目标已扣除同初态保持分支的平均推进，避免把惯性或被动释放当成动作收益。helped要求该动作所有接受的重复都减少滑移并保留推进。</p>'
    page+='<p>模型选动作先要求预测推进量不低于上述目标、离轴不明显恶化，排除一直保持；再在合格动作里最小化滑移与转动风险。'
    page+='没有合格候选时标记拒绝推荐并沿用continue作评分参照。效用遗憾只是辅助；同时检查实际推进保留率及真正减少滑移的状态数。</p>'
    page+='<pre>'+html.escape(str([dict(mode=r['mode'],regret=r['mean_regret'],ci=r['regret_95ci'],
                progress_retention=r['progress_retention_rate'],helped=r['helped_states'],abstained=r['abstained']) for r in results]))+'</pre>'
    page+='<p>固定动作对照：</p><pre>'+html.escape(str(fixed))+'</pre>'
    for s in states:
        page+=f'<hr><h2>{s["id"]} · seed {s["seed"]} · {s["group"]} · 验收 {s["usable"]}</h2>'
        page+='<p>'+html.escape(str(s.get('gate',{})))+'</p><table><tr><th>动作</th><th>相对平移mm</th><th>转角°</th><th>推进mm</th><th>比保持多推进mm</th><th>后段相对位移mm</th><th>视频</th></tr>'
        for kind,e in s.get('effects',{}).items():
            video=Path(s['folder'])/f'{kind}.mp4'
            import os
            link=os.path.relpath(video,args.output)
            extra='—' if e.get('extra_progress_mm') is None else f'{e["extra_progress_mm"]:.2f}'
            page+=f'<tr><td>{kind}</td><td>{e["slip_mm"]:.2f}</td><td>{e["rotation_deg"]:.2f}</td><td>{e["progress_mm"]:.2f}</td><td>{extra}</td><td>{e["late_slip_mm"]:.2f}</td><td><a href="{link}">双视图＋双指触觉</a></td></tr>'
        page+='</table>'
        if s['usable']:
            shown=['continue','slow','pause_resume']
            import os
            for kind in shown:
                link=os.path.relpath(Path(s['folder'])/f'{kind}.mp4',args.output)
                page+=f'<p>{kind}</p><video controls preload="metadata" src="{link}"></video>'
    (args.output/'branch_report.html').write_text(page)
    status(args,'branch-probes','complete' if results else 'descriptive_only',usable=len(usable),models=len(results))
