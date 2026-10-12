#!/usr/bin/env python3
"""Audit action and object-motion similarity before/after first plug–socket contact."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_contactworld_training_neighbors import geometry


def resample(x,n):
    x=np.asarray(x,float)
    return np.stack([np.interp(np.linspace(0,len(x)-1,n),np.arange(len(x)),x[:,i]) for i in range(x.shape[1])],axis=1)


def similarity(a,b):
    mag=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    active=mag>.01
    cos=np.sum(a[active]*b[active],axis=1)/mag[active] if active.any() else np.array([])
    return float(np.mean(cos)) if len(cos) else None,int(active.sum())


def phase_actions(a,contact,mode):
    if mode=='pre':return resample(a[:contact,:3],24) if contact>=4 else None
    return resample(a[contact:contact+20,:3],20) if len(a)-contact>=8 else None


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--dataset',type=Path,default=Path('data/contactworld_usb_v3_bilateral'))
    a=p.parse_args();out=a.root/'training_neighbors'/'phase_audit';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.dataset/'spatial/manifest.json').read_text());nearest={x['seed']:x['nearest_source_episode'] for x in json.loads((a.root/'training_neighbors'/'analysis.json').read_text())['neighbors']}
    demos=[]
    for e in manifest['episodes']:
        if e['split']!='train':continue
        folder=Path(manifest['capture_root'])/f'insertion_usb_episode_{e["source_episode"]:03d}'
        with np.load(folder/'v2_capture.npz') as z:
            actions=z['action_executed'].copy();g=geometry(z['plug_pose'],z['socket_pose'],z['ee_pose'])
        rr=json.loads((folder/'replay.json').read_text())['rows'];forces=np.linalg.norm([r['plug_socket_force_post'] for r in rr],axis=1)
        ix=np.flatnonzero(forces>.1);contact=int(ix[0])+1 if len(ix) else None
        demos.append({'source':e['source_episode'],'actions':actions,'contact':contact,'g':g,
                      'pre':phase_actions(actions,contact,'pre') if contact else None,
                      'post':phase_actions(actions,contact,'post') if contact else None})
    demo={d['source']:d for d in demos};paths=sorted(a.root.glob('pi0_cw3_usb_*/summary.json'));rows=[]
    for model_index,path in enumerate(paths,1):
        for seed in nearest:
            t=json.loads((path.parent/f'seed_{seed}/trajectory.json').read_text());rr=t['rows'];init=t['result']['initial_physical_state']
            gg=geometry(np.array([r['plug_pos']+r['plug_quat'] for r in [init]+rr]),
                        np.array([r['socket_pos_gt']+r['socket_quat'] for r in [init]+rr]),
                        np.array([r['ee_pos']+r['ee_quat'] for r in [init]+rr]))
            actions=np.asarray([r['action_sim_clipped'] for r in rr]);force=np.linalg.norm([r['plug_socket_contact_force_xyz'] for r in rr],axis=1)
            ix=np.flatnonzero(force>.1);contact=int(ix[0])+1 if len(ix) else None
            d=demo[nearest[seed]]
            record={'seed':seed,'model':model_index,'train_source':d['source'],'ever':int(t['result']['source_ever']),'final':int(t['result']['source_final']),
                    'test_contact_step':contact,'train_contact_step':d['contact']}
            for mode in ['pre','post']:
                qa=phase_actions(actions,contact,mode) if contact else None
                da=d[mode]
                if qa is None or da is None:
                    record.update({f'{mode}_action_cos3d':None,f'{mode}_action_cos_xy':None,f'{mode}_rank_nearest_of95':None,
                                   f'{mode}_test_plug_delta_z_mm':None,f'{mode}_train_plug_delta_z_mm':None,
                                   f'{mode}_test_ee_delta_z_mm':None,f'{mode}_train_ee_delta_z_mm':None})
                    continue
                co,_=similarity(qa,da);cxy,_=similarity(qa[:,:2],da[:,:2]);sims=[]
                for other in demos:
                    if other[mode] is None:continue
                    x,_=similarity(qa,other[mode]);sims.append((other['source'],x))
                rank=1+sum(x is not None and co is not None and x>co for _,x in sims)
                record.update({f'{mode}_action_cos3d':co,f'{mode}_action_cos_xy':cxy,f'{mode}_rank_nearest_of95':rank})
                if mode=='post':
                    end=min(contact+20,len(actions),len(gg['relative'])-1)
                    dend=min(d['contact']+20,len(d['actions']),len(d['g']['relative'])-1)
                    for label,g,start,stop in [('test',gg,contact,end),('train',d['g'],d['contact'],dend)]:
                        record[f'post_{label}_plug_delta_z_mm']=float((g['relative'][stop,2]-g['relative'][start,2])*1000)
                        record[f'post_{label}_ee_delta_z_mm']=float((g['ee_relative'][stop,2]-g['ee_relative'][start,2])*1000)
                else:
                    for label in ['test','train']:
                        record[f'pre_{label}_plug_delta_z_mm']=None;record[f'pre_{label}_ee_delta_z_mm']=None
            for label in ('pre8','post8'):
                if contact is None or d['contact'] is None:
                    record[f'{label}_action_cos3d']=None
                    record[f'{label}_action_cos_xy']=None
                    record[f'{label}_active_xy_samples']=None
                    continue
                start=contact-8 if label=='pre8' else contact
                train_start=d['contact']-8 if label=='pre8' else d['contact']
                if start<0 or train_start<0 or start+8>len(actions) or train_start+8>len(d['actions']):
                    record[f'{label}_action_cos3d']=None
                    record[f'{label}_action_cos_xy']=None
                    record[f'{label}_active_xy_samples']=None
                    continue
                x=actions[start:start+8,:3];y=d['actions'][train_start:train_start+8,:3]
                c3,_=similarity(x,y);cxy,nxy=similarity(x[:,:2],y[:,:2])
                record[f'{label}_action_cos3d']=c3
                record[f'{label}_action_cos_xy']=cxy
                record[f'{label}_active_xy_samples']=nxy
            rows.append(record)
    write_csv(out/'all_520_contact_phase.csv',rows)
    valid=[r for r in rows if r['pre_action_cos3d'] is not None and r['post_action_cos3d'] is not None]
    def arr(key):return np.array([r[key] for r in valid],float)
    byseed=[]
    for seed in nearest:
        sub=[r for r in valid if r['seed']==seed]
        if sub:
            byseed.append({'seed':seed,'n':len(sub),'pre_cos3d':np.mean([r['pre_action_cos3d'] for r in sub]),
                           'post_cos3d':np.mean([r['post_action_cos3d'] for r in sub]),
                           'pre_cos_xy':np.mean([r['pre_action_cos_xy'] for r in sub if r['pre_action_cos_xy'] is not None]),
                           'post_cos_xy':np.mean([r['post_action_cos_xy'] for r in sub if r['post_action_cos_xy'] is not None]),
                           'plug_z_gap_mm':np.mean([r['post_test_plug_delta_z_mm']-r['post_train_plug_delta_z_mm'] for r in sub]),
                           'ee_z_gap_mm':np.mean([r['post_test_ee_delta_z_mm']-r['post_train_ee_delta_z_mm'] for r in sub])})
    write_csv(out/'per_seed_contact_phase.csv',byseed)
    rng=np.random.default_rng(20261004)
    boot=rng.integers(len(byseed),size=(10000,len(byseed)))
    def summarize(v):
        v=np.array(v,float)
        return {'mean':float(v.mean()),'median':float(np.median(v)),'bootstrap95_seed':np.quantile(v[boot].mean(axis=1),[.025,.975]).tolist()}
    summary={'train_episodes':len(demos),'test_trajectories':len(rows),'both_phase_comparable':len(valid),
             'no_test_contact':sum(r['test_contact_step'] is None for r in rows),
             'no_train_contact':sum(d['contact'] is None for d in demos),
             'eight_step_comparable':sum(r['pre8_action_cos3d'] is not None and r['post8_action_cos3d'] is not None for r in rows),
             'unit':'40 seeds; seed-level bootstrap, 13 model trajectories nested within seed',
             'action_similarity':{'pre_cos3d':summarize([r['pre_cos3d'] for r in byseed]),'post_cos3d':summarize([r['post_cos3d'] for r in byseed]),
                                  'post_minus_pre_cos3d':summarize([r['post_cos3d']-r['pre_cos3d'] for r in byseed]),
                                  'pre_cos_xy':summarize([r['pre_cos_xy'] for r in byseed]),'post_cos_xy':summarize([r['post_cos_xy'] for r in byseed]),
                                  'post_minus_pre_cos_xy':summarize([r['post_cos_xy']-r['pre_cos_xy'] for r in byseed])},
             'object_and_ee_z_gap':{'post_plug_mm':summarize([r['plug_z_gap_mm'] for r in byseed]),
                                    'post_ee_mm':summarize([r['ee_z_gap_mm'] for r in byseed])},
             'nearest_demo_action_rank_median_pre':float(np.median(arr('pre_rank_nearest_of95'))),
             'nearest_demo_action_rank_median_post':float(np.median(arr('post_rank_nearest_of95'))),
             'nearest_demo_pre_rank_top10_fraction':float(np.mean(arr('pre_rank_nearest_of95')<=10)),
             'nearest_demo_post_rank_top10_fraction':float(np.mean(arr('post_rank_nearest_of95')<=10)),
             'success_breakdown':{str(s):{'n':sum(r['ever']==s for r in valid),
                                          'pre_cos3d_median':float(np.median([r['pre_action_cos3d'] for r in valid if r['ever']==s])),
                                          'post_cos3d_median':float(np.median([r['post_action_cos3d'] for r in valid if r['ever']==s])),
                                          'post_plug_delta_z_mm_median':float(np.median([r['post_test_plug_delta_z_mm'] for r in valid if r['ever']==s]))}
                                  for s in [0,1]}}
    for label in ('pre8','post8'):
        for component in ('cos3d','cos_xy'):
            key=f'{label}_action_{component}'
            summary['action_similarity'][key]=summarize([
                np.mean([r[key] for r in rows if r['seed']==seed and r[key] is not None])
                for seed in nearest
            ])
    summary['eight_step_active_xy_samples_median']={
        label:float(np.median([r[f'{label}_active_xy_samples'] for r in rows
                            if r[f'{label}_active_xy_samples'] is not None]))
        for label in ('pre8','post8')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for i,(before,after,title) in enumerate([('pre_action_cos3d','post_action_cos3d','3D action direction'),('pre_action_cos_xy','post_action_cos_xy','XY action direction'),('post_test_plug_delta_z_mm','post_train_plug_delta_z_mm','Postcontact plug Z displacement')]):
        x=np.array([r[before] for r in valid if r[before] is not None and r[after] is not None]);y=np.array([r[after] for r in valid if r[before] is not None and r[after] is not None]);ax[i].scatter(x,y,s=9,alpha=.3);ax[i].set(xlabel=before,ylabel=after,title=title);ax[i].grid(alpha=.2)
    fig.savefig(out/'phase_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
