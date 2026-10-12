#!/usr/bin/env python3
"""Contact-aligned action-direction similarity and support by relative step."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003')
OUT=ROOT/'training_neighbors'/'phase_audit'
CAP=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_bilateral_full_20261002/replay')

def cosine(x,y):
    n=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/n) if n>.01 else None

def main():
    nearest={x['seed']:x['nearest_source_episode'] for x in json.loads((ROOT/'training_neighbors'/'analysis.json').read_text())['neighbors']}
    demos={}
    for source in set(nearest.values()):
        folder=CAP/f'insertion_usb_episode_{source:03d}'
        with np.load(folder/'v2_capture.npz') as z:actions=z['action_executed']
        rr=json.loads((folder/'replay.json').read_text())['rows'];force=np.linalg.norm([r['plug_socket_force_post'] for r in rr],axis=1)
        first=np.flatnonzero(force>.1);demos[source]=(actions,int(first[0]) if len(first) else None)
    records=[]
    for m,summary in enumerate(sorted(ROOT.glob('pi0_cw3_usb_*/summary.json')),1):
        for seed,source in nearest.items():
            rr=json.loads((summary.parent/f'seed_{seed}/trajectory.json').read_text())['rows']
            actions=np.array([r['action_sim_clipped'] for r in rr]);f=np.linalg.norm([r['plug_socket_contact_force_xyz'] for r in rr],axis=1)
            first=np.flatnonzero(f>.1);testc=int(first[0]) if len(first) else None;train,demo_c=demos[source]
            if testc is None or demo_c is None:continue
            for offset in range(-16,17):
                ti=testc+offset;di=demo_c+offset
                if ti<0 or di<0 or ti>=len(actions) or di>=len(train):continue
                records.append({'seed':seed,'model':m,'offset':offset,
                                'cos3d':cosine(actions[ti,:3],train[di,:3]),
                                'cosxy':cosine(actions[ti,:2],train[di,:2])})
    with (OUT/'aligned_step_similarity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    agg=[]
    for off in range(-16,17):
        sub=[r for r in records if r['offset']==off]
        item={'offset':off,'trajectories':len(sub)}
        for name in ['cos3d','cosxy']:
            seedmeans=[]
            for seed in nearest:
                x=[r[name] for r in sub if r['seed']==seed and r[name] is not None]
                if x:seedmeans.append(np.mean(x))
            item[name+'_mean']=float(np.mean(seedmeans)) if seedmeans else None
            item[name+'_seeds']=len(seedmeans)
            if len(seedmeans)>1:
                rng=np.random.default_rng(20261004+off)
                bs=np.array(seedmeans)[rng.integers(len(seedmeans),size=(5000,len(seedmeans)))].mean(axis=1)
                item[name+'_bootstrap95']=np.quantile(bs,[.025,.975]).tolist()
        agg.append(item)
    (OUT/'aligned_step_similarity.json').write_text(json.dumps(agg,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for ax,name,label in zip(axes,['cos3d','cosxy'],['3D translation direction cosine','XY translation direction cosine']):
        x=np.array([r['offset'] for r in agg]);y=np.array([r[name+'_mean'] for r in agg]);lo=np.array([r[name+'_bootstrap95'][0] for r in agg]);hi=np.array([r[name+'_bootstrap95'][1] for r in agg])
        ax.plot(x,y,color='#1f6ca1');ax.fill_between(x,lo,hi,color='#1f6ca1',alpha=.17)
        ax.axvline(0,color='#d0643a',linestyle='--');ax.set(ylabel=label);ax.grid(alpha=.2)
    axes[-1].set(xlabel='Steps relative to first plug-socket contact')
    fig.savefig(OUT/'aligned_step_similarity.png',dpi=170);plt.close(fig)
    print(json.dumps({x['offset']:x for x in agg if x['offset'] in [-16,-8,-4,-1,0,1,4,8,16]},indent=2))

if __name__=='__main__':main()
