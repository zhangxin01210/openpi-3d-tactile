#!/usr/bin/env python3
"""Small, transparent QA summary for RGBTrack cross-episode validation.

The summary does NOT auto-approve training. Valid means tracker accepted, not verified pose accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import numpy as np


def read(path):
    data=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not data:raise ValueError('No poses: '+str(path))
    ids=[int(d['frame_index']) for d in data]
    if ids!=list(range(len(ids))):raise ValueError('Noncontiguous 0-based frames: '+str(path))
    return data

def longest_false(a):
    best=cur=0; start=0; longest=(None,None)
    for i,b in enumerate(a):
        if not b:
            if not cur:start=i
            cur+=1
            if cur>best:best=cur;longest=(start,i)
        else:cur=0
    return {'length':best,'frames':longest}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pose-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();folder=args.pose_run
    manifest=json.loads((folder/'manifest.json').read_text())
    f=read(folder/'funnel_poses.jsonl');t=read(folder/'target_flask_poses.jsonl')
    if len(f)!=len(t):raise ValueError('Frame count mismatch')
    valid=[bool(x.get('valid')) for x in f]
    stats={'episode':manifest.get('episode'),'n_frames':len(f),
       'funnel':{'valid_frames':sum(valid),'valid_fraction':round(sum(valid)/len(valid),4),
                 'longest_invalid_streak':longest_false(valid),'statuses':dict(Counter(x.get('status') for x in f))},
       'target_flask':{'statuses':dict(Counter(x.get('status') for x in t)),
                       'all_static_prior_after_f0':all(x.get('status')=='flask_static_anchor_prior' for x in t[1:])},
       'qa':'NOT automatically accepted: inspect frame0, first contact, object lift, insertion, release and tracking recovery.'}
    # Validate flask truly held fixed in base, and all retained poses are homogeneous.
    init=np.asarray(t[0]['T_base_link_object'],float)
    if init.shape!=(4,4):raise ValueError('Flask anchor not 4x4')
    for row in t:
        X=np.asarray(row['T_base_link_object'],float)
        if not np.allclose(X,init,atol=1e-6):raise ValueError('Flask static prior drifted')
    stats['target_flask']['static_anchor_exact']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print('[SUMMARY]',args.output)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
