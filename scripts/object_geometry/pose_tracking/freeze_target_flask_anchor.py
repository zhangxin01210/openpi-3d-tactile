#!/usr/bin/env python3
"""Freeze a mechanically supported target flask at its reviewed frame-0 pose.

Only reprocesses existing JSONL data: no GPU, SAM2, RGBTrack or robot FK.
Byte-for-byte preserves the validated dynamic funnel track.  Every target
pose after the anchor is explicitly tagged as a STATIC GEOMETRY PRIOR, not
an observed 6D estimate.  Suitable as a baseline when the target is fixed
in a clamp and contact-triggered pivot estimation is not yet validated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows:
        raise ValueError(f'Empty JSONL: {path}')
    frames = [int(r['frame_index']) for r in rows]
    if len(set(frames)) != len(frames) or frames != sorted(frames):
        raise ValueError(f'Duplicate or non-monotonic frames: {path}')
    return rows


def transform(record: dict, name: str) -> np.ndarray:
    t = np.asarray(record[name], dtype=float)
    if t.shape != (4,4) or not np.isfinite(t).all():
        raise ValueError(f'Invalid {name}')
    if not np.allclose(t[3], [0,0,0,1], atol=1e-5):
        raise ValueError(f'Invalid bottom row of {name}')
    r=t[:3,:3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-2) or np.linalg.det(r)<0.98:
        raise ValueError(f'Non-rigid rotation in {name}')
    return t


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    source=args.source_pose_run.resolve();dest=args.output.resolve()
    if dest.exists():
        raise FileExistsError(f'Output already exists: {dest}')
    poses=source/'target_flask_poses.jsonl'; funnel=source/'funnel_poses.jsonl'; manifest=source/'manifest.json'
    for path in (poses,funnel,manifest):
        if not path.is_file(): raise FileNotFoundError(path)
    rows=read_jsonl(poses); funnel_rows=read_jsonl(funnel)
    if [int(r['frame_index']) for r in rows] != [int(r['frame_index']) for r in funnel_rows]:
        raise ValueError('Funnel and target frame indices mismatch')
    anchors=[r for r in rows if int(r['frame_index'])==args.anchor_frame]
    if len(anchors)!=1 or not anchors[0].get('valid',False):
        raise ValueError(f'Missing/invalid anchor at frame {args.anchor_frame}')
    anchor=anchors[0]
    tb=transform(anchor,'T_base_link_object'); tc=transform(anchor,'T_color_object')
    drift=[]
    statuses=Counter()
    output=[]
    for r in rows:
        r=dict(r)
        f=int(r['frame_index'])
        if r.get('valid',False):
            try:
                current=transform(r,'T_base_link_object')
                delta=current[:3,3]-tb[:3,3]
                relative=tb[:3,:3].T@current[:3,:3]
                rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)*0.5,-1,1))))
                drift.append((f,float(np.linalg.norm(delta)),rotation_deg))
            except (TypeError,ValueError,KeyError): pass
        statuses[str(r.get('status','missing'))]+=1
        r['T_base_link_object']=tb.tolist()
        r['T_color_object']=tc.tolist()  # front camera is static; same calibration for all frames
        r['valid']=True
        r['status']='flask_upright_anchor_observed' if f==args.anchor_frame else 'flask_static_anchor_prior'
        r['pose_source']='reviewed_initial_geometry' if f==args.anchor_frame else 'static_geometry_prior'
        r['visual_measurement_accepted']=bool(f==args.anchor_frame)
        r['confidence']=float(anchor.get('confidence',0.35)) if f==args.anchor_frame else 0.08
        r['geometry_prior_valid']=True
        r['observed_pose_valid']=bool(f==args.anchor_frame)
        r['static_prior_anchor_frame']=int(args.anchor_frame)
        output.append(r)
    md=json.loads(manifest.read_text(encoding='utf-8'))
    if md.get('episode') not in (None,args.episode):
        raise ValueError(f'Manifest episode mismatch: {md.get("episode")}')
    if args.anchor_frame != 0:
        raise ValueError('Only frame-0 anchor supported until camera motion provenance is verified')
    dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(funnel, dest/funnel.name)
    with (dest/poses.name).open('w',encoding='utf-8') as f:
        for r in output:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    if sha256(funnel)!=sha256(dest/funnel.name):raise AssertionError('Funnel trajectory changed')
    md['target_flask_static_prior']={
        'anchor_frame':args.anchor_frame, 'source_pose_run':str(source),
        'assumption':'Mechanically clamped flask is stationary; collision-induced tilt ignored in this baseline',
        'representation':'frame-0 6D pose held for all frames, geometry prior; NOT per-frame visual observation',
        'funnel_poses_unchanged_sha256':sha256(funnel),
    }
    (dest/'manifest.json').write_text(json.dumps(md,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report={
        'source_pose_run':str(source), 'anchor_frame':args.anchor_frame,
        'frames':len(output), 'funnel_sha256_unchanged':sha256(funnel),
        'flask_source_status_counts':dict(statuses),
        'visual_drift_from_anchor_when_valid':{
            'count':len(drift),
            'max_translation_cm':round(max((x[1] for x in drift),default=0)*100,2),
            'max_rotation_deg':round(max((x[2] for x in drift),default=0),2),
        },
        'warning':'Treat target flask as static scene prior; not a measured dynamic 6D pose. For physical tilting during contact add a causally triggered update after sensor validation.'
    }
    (dest/'flask_static_prior_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('[DONE] Funnel trajectory preserved byte-for-byte, target flask held at f0')
    print('[OUTPUT]', dest)
    print('[QA]', dest/'flask_static_prior_report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-pose-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,default=23)
    p.add_argument('--anchor-frame',type=int,default=0)
    run(p.parse_args())

if __name__=='__main__':main()
