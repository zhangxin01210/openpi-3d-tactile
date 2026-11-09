#!/usr/bin/env python3
"""V2 RGB-D + single CAD evaluation: standalone HTML with embedded images.

Does not change the model, checkpoint, training data, or original evaluation. Uses
V2 cached RGB for the background (256x192 by default); projects predicted and
GT CAD surfaces through the project's calibrated front color camera.

Example:
  .venv/bin/python -u visualize_cad_pose_rgbd_v2.py \
    --checkpoint output/cad_pose_rgbd_v2/train_.../best.pt \
    --rows output/cad_pose_rgbd_v2/eval_best_test.rows.jsonl \
    --output output/cad_pose_rgbd_v2/v2_predictions_standalone.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'src'))

from cad_pose_rgbd_single_v2 import (SingleCadRGBDPose, SCHEMA, TASK_NAMES, ROLE_NAMES)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo',type=Path,default=HERE)
    p.add_argument('--cache-dir',type=Path,default=HERE/'output/cad_pose_rgbd_v2/cache')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--rows',type=Path,default=HERE/'output/cad_pose_rgbd_v2/eval_best_test.rows.jsonl')
    p.add_argument('--metrics',type=Path,default=HERE/'output/cad_pose_rgbd_v2/eval_best_test.json')
    p.add_argument('--output',type=Path,default=HERE/'output/cad_pose_rgbd_v2/v2_predictions_standalone.html')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--per-role',type=int,default=6)
    p.add_argument('--ref-width',type=int,default=640,help='Original RGB calibration width, before image cache resize')
    p.add_argument('--ref-height',type=int,default=480,help='Original RGB calibration height, before image cache resize')
    p.add_argument('--peg-intrinsics',choices=('legacy','current'),default='legacy')
    p.add_argument('--seed',type=int,default=42)
    return p.parse_args()


def draw_rgb_overlay(rgb, xy, *, title, color, max_points=512):
    base=cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2BGR)
    h,w=base.shape[:2]
    ink=base.copy()
    uv=np.asarray(xy)
    good=np.isfinite(uv).all(axis=1)
    xys=np.rint(uv[good]).astype(np.int32)
    xys=xys[(xys[:,0]>=0)&(xys[:,0]<w)&(xys[:,1]>=0)&(xys[:,1]<h)]
    if len(xys)>max_points:
        xys=xys[np.linspace(0,len(xys)-1,num=max_points,dtype=int)]
    for x,y in xys:
        cv2.circle(ink,(int(x),int(y)),1,color,-1,lineType=cv2.LINE_AA)
    res=cv2.addWeighted(ink,0.78,base,0.22,0)
    cv2.putText(res,title,(7,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(res,f'visible CAD points: {len(xys)}/{len(uv)}',(7,h-9),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
    return res,len(xys)


def jpg_img(rgb_panels):
    bgr=np.concatenate(rgb_panels,axis=1)
    ok,encoded=cv2.imencode('.jpg',bgr,[int(cv2.IMWRITE_JPEG_QUALITY),91])
    if not ok:raise RuntimeError('JPEG encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(encoded.tobytes()).decode('ascii')


def project_xyz(points_base, cam, out_w, out_h, ref_w, ref_h):
    # Reuse repository math and calibration, not a hand-written extrinsic convention.
    from openpi.spatial.geometry import invert_transform, transform_points, project_pinhole
    pts=transform_points(np.asarray(points_base,dtype=np.float64),invert_transform(cam.T_base_color))
    uv=project_pinhole(pts,cam.color_intrinsics)
    uv=np.asarray(uv,dtype=np.float64)
    uv[pts[:,2]<=1e-5]=np.nan
    uv[:,0]*=out_w/ref_w
    uv[:,1]*=out_h/ref_h
    return uv


def front_camera(repo):
    from openpi.spatial.calibration import load_camera_calibrations
    from openpi.spatial.config import make_baseline_config
    from openpi.spatial.geometry import apply_diagnostic_overrides
    cfg=make_baseline_config().with_camera_roles('front').with_front_diagnostics(extrinsic=True,intrinsic=False)
    bundle=load_camera_calibrations(repo_root=repo,calibration_config=cfg.calibration,camera_roles=('front',))
    return apply_diagnostic_overrides(bundle.cameras,cfg.diagnostics)['front']


def peg_camera(repo, cam):
    from openpi.spatial.config import PinholeIntrinsics
    p=repo/'data/1018_peg_all/object_geometry/v1/manifest.json'
    d=json.loads(p.read_text(encoding='utf-8'))
    k=d.get('calibration',{}).get('applied_front_color_intrinsics')
    if not isinstance(k,dict):
        raise RuntimeError('Peg legacy intrinsics unavailable; use --peg-intrinsics current')
    intr=PinholeIntrinsics(**{s:float(k[s]) for s in ('fx','fy','cx','cy')})
    return replace(cam,color_intrinsics=intr)


def transformed(cad, t, r):
    return np.asarray(cad,dtype=np.float64)@np.asarray(r,dtype=np.float64).T+np.asarray(t,dtype=np.float64)


def tilt_deg(r_pred, r_gt):
    # Directed CAD-local Z-axis mismatch. Ignore only yaw around this local axis.
    a=np.asarray(r_pred,dtype=float)[:,2]
    b=np.asarray(r_gt,dtype=float)[:,2]
    return float(np.degrees(np.arccos(np.clip(float(a@b),-1,1))))


def select_rows(rows,per_role=6,seed=42):
    rng=np.random.default_rng(seed)
    groups=defaultdict(list)
    for row in rows:
        name=row.get('task');role=row.get('role')
        if name in TASK_NAMES and role in ROLE_NAMES:
            groups[(name,role)].append(row)
    selected=[]
    for task in TASK_NAMES:
        for role in ROLE_NAMES:
            data=groups.get((task,role),[])
            if not data:raise ValueError(f'Missing evaluation rows for {task}/{role}')
            data.sort(key=lambda r:float(r['translation_mm']))
            wrong=[r for r in data if r.get('other_object_closer')]
            picks=[data[0],data[len(data)//2],data[int(.90*(len(data)-1))],data[-1]]
            if wrong:picks.append(wrong[len(wrong)//2])
            # Include a random episode rather than only absolute outliers.
            picks.append(data[int(rng.integers(len(data)))])
            seen=set()
            for p in picks:
                k=(p['episode'],p['frame'])
                if k not in seen and len(seen)<per_role:
                    selected.append(p);seen.add(k)
            extra=[r for r in data if (r['episode'],r['frame']) not in seen]
            rng.shuffle(extra)
            for p in extra:
                if len(seen)>=per_role:break
                selected.append(p);seen.add((p['episode'],p['frame']))
    return selected


def load_models(ckpt_path,device,cache_dir):
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    if ckpt.get('schema')!=SCHEMA:
        raise ValueError('Checkpoint is not Single CAD RGB-D V2')
    cachecfg=json.loads((cache_dir/'cache_config.json').read_text())
    if (ckpt.get('width'),ckpt.get('height'))!=(cachecfg.get('width'),cachecfg.get('height')):
        raise RuntimeError('Checkpoint/cache resolution mismatch')
    model=SingleCadRGBDPose().to(device)
    model.load_state_dict(ckpt['model_state_dict'],strict=True)
    model.eval()
    return model,ckpt


def load_sample(cache_dir,row,cad_cache):
    task=row['task'];role=ROLE_NAMES.index(row['role']);ep=int(row['episode']);frame=int(row['frame'])
    path=cache_dir/task/f'e{ep:05d}'
    if task not in cad_cache:
        cad_cache[task]=np.load(cache_dir/task/'cad.npy',allow_pickle=False).astype('float32')
    rgb=np.load(path/'rgb.npy',mmap_mode='r',allow_pickle=False)
    dep=np.load(path/'depth.npy',mmap_mode='r',allow_pickle=False)
    pose=np.load(path/'transform.npy',mmap_mode='r',allow_pickle=False)
    valid=np.load(path/'valid.npy',mmap_mode='r',allow_pickle=False)
    if not bool(valid[frame,role]):raise RuntimeError(f'Invalid selected label: {task} e{ep} f{frame} {role}')
    return np.asarray(rgb[frame]).copy(),np.asarray(dep[frame]).copy(),cad_cache[task][role],np.asarray(pose[frame,role]).copy()


def predict(model, ckpt, rgb, depth, cad, device):
    r=torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(device).permute(0,3,1,2).float().div(255)
    d=torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).unsqueeze(1).to(device).float()
    d=(d*float(ckpt.get('depth_scale',.001))).clamp(0,2.0)/2.0
    c=torch.from_numpy(np.ascontiguousarray(cad)).unsqueeze(0).to(device).float()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):
            t,rot=model(r,d,c)
    return t[0].float().cpu().numpy(),rot[0].float().cpu().numpy()


def metric_table(metrics):
    if not metrics:return ''
    items=[]
    for task in TASK_NAMES:
        for role in ROLE_NAMES:
            m=metrics.get(f'{task}/{role}',{})
            if not m:continue
            t=m.get('translation_mm',{}); rot=m.get('rotation_deg',{}); add=m.get('add_mm',{})
            wr=m.get('wrong_object_closer',{})
            try:txt=f'<tr><td>{html.escape(task)}</td><td>{role}</td><td>{t["mean"]:.1f}</td><td>{rot["mean"]:.1f}</td><td>{add["mean"]:.1f}</td><td>{wr.get("mean","N/A")}</td></tr>'
            except Exception:continue
            items.append(txt)
    return '<h2>Test metrics (8 objects)</h2><table><thead><tr><th>Task</th><th>Role</th><th>t mean (mm)</th><th>raw rotation mean (deg)</th><th>ADD mean (mm)</th><th>Other closer rate</th></tr></thead><tbody>'+''.join(items)+'</tbody></table>'


def main():
    a=parse_args()
    a.repo=a.repo.resolve();a.cache_dir=a.cache_dir.resolve();a.output=a.output.resolve()
    if not a.rows.is_file():raise FileNotFoundError(f'Missing rows JSONL: {a.rows}; rerun V2 evaluate --save-rows')
    if a.per_role<1 or a.per_role>24:raise ValueError('--per-role must be 1..24')
    if a.ref_width<100 or a.ref_height<100:raise ValueError('Bad original RGB dimensions')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    device=torch.device(a.device)
    model,ckpt=load_models(a.checkpoint,device,a.cache_dir)
    data=[]
    with a.rows.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():data.append(json.loads(line))
    selected=select_rows(data,per_role=a.per_role,seed=a.seed)
    if not selected:raise RuntimeError('No valid samples selected')
    camera=front_camera(a.repo)
    peg_cam=peg_camera(a.repo,camera) if a.peg_intrinsics=='legacy' else camera
    cads={}
    imgs=[]
    for n,row in enumerate(selected,1):
        task=row['task'];role=row['role'];ep=int(row['episode']);frame=int(row['frame'])
        rgb,depth,cad,gt=load_sample(a.cache_dir,row,cads)
        pred_t,pred_r=predict(model,ckpt,rgb,depth,cad,device)
        gt_t=gt[:3,3];gt_r=gt[:3,:3]
        cam=peg_cam if task=='1018_peg_all' else camera
        h,w=rgb.shape[:2]
        guv=project_xyz(transformed(cad,gt_t,gt_r),cam,w,h,a.ref_width,a.ref_height)
        puv=project_xyz(transformed(cad,pred_t,pred_r),cam,w,h,a.ref_width,a.ref_height)
        # Green GT, magenta predicted. Show both separately and side-by-side.
        original=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).copy()
        cv2.putText(original,'RGB (cached)',(7,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
        gt_image,ng=draw_rgb_overlay(rgb,guv,title='GT - GREEN',color=(60,230,60))
        pred_image,npred=draw_rgb_overlay(rgb,puv,title='PRED - MAGENTA',color=(205,40,235))
        tmm=float(np.linalg.norm(pred_t-gt_t)*1000)
        tilt=tilt_deg(pred_r,gt_r)
        shown=jpg_img([original,gt_image,pred_image])
        wrong=bool(row.get('other_object_closer',False))
        label=' <b class="wrong">OTHER-OBJECT CLOSER</b>' if wrong else ''
        imgs.append(f'<article><header><b>{html.escape(task)} / {role}</b> &nbsp; e{ep:05d} f{frame:04d} &nbsp; '
                    f'| t={tmm:.1f}mm &nbsp; local-Z tilt={tilt:.1f}° {label}</header>'
                    f'<img loading="lazy" alt="Three-panel prediction overlay" src="{shown}">'
                    f'<div class="meta">GT visible CAD points: {ng}/512; Pred: {npred}/512; '
                    f'test reported t={row.get("translation_mm",0):.1f} mm</div></article>')
        print(f'[VIZ] {n}/{len(selected)} {task}/{role} e{ep} f{frame} t={tmm:.1f}mm tilt={tilt:.1f}deg',flush=True)
    metrics={}
    if a.metrics.is_file():
        metrics=json.loads(a.metrics.read_text()).get('metrics',{})
    doc='''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>V2 RGB-D single CAD — prediction vs GT</title>
    <meta name="viewport" content="width=device-width,initial-scale=1"><style>
    body{font:15px/1.5 system-ui,-apple-system,sans-serif;background:#10141d;color:#e1e6f0;margin:24px auto;padding:0 16px;max-width:1150px}
    h1{font-size:25px;margin-bottom:7px}p{color:#c0c9d9}h2{font-size:20px;margin-top:28px}
    article{border:1px solid #394050;border-radius:11px;margin:16px 0;padding:13px;background:#1a2130}
    article header{margin-bottom:9px}article img{width:100%;height:auto;image-rendering:auto;border-radius:5px}
    .meta{font-size:12px;color:#9eabc3;margin-top:4px}.wrong{color:#ff9b9b}
    table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #3b4351;padding:7px;text-align:left}
    </style></head><body>'''
    doc+='<h1>V2 RGB-D + Single CAD — Pred vs GT</h1>'
    doc+='<p><b>Single-file HTML; all images embedded.</b> Panels: cached RGB | green GT CAD | magenta predicted CAD. '+\
         'Test samples selected by low/median/p90/high error, cross-object errors and random examples; these are NOT an unbiased frequency distribution.</p>'
    doc+='<p><b>Symmetry:</b> raw 6D rotation errors include unobservable CAD-local Z yaw for axially symmetric objects. '+\
         'The per-image “local-Z tilt” angle compares the CAD local Z axes under predicted vs GT rotation, ignoring yaw around that axis. '+\
         'It is meaningful only when the relevant physical symmetry axis is CAD-local Z and its direction is observable. '+\
         'GT annotation noise may remain.</p>'
    doc+=f'<p>Checkpoint: <code>{html.escape(str(a.checkpoint))}</code><br>Background: cached RGB {ckpt["width"]}×{ckpt["height"]}; '+\
         f'original calibration reference {a.ref_width}×{a.ref_height}; Peg intrinsics mode: <b>{a.peg_intrinsics}</b>. '+\
         'The cache downsampling limits tiny-object visual inspection; image projection uses project camera calibration.</p>'
    doc+=metric_table(metrics)
    doc+='<h2>Selected prediction overlays</h2>'+''.join(imgs)
    doc+='</body></html>'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(doc,encoding='utf-8')
    if '<img ' not in doc or 'data:image/jpeg;base64,' not in doc:raise RuntimeError('No embedded images written')
    if '<img ' in doc and doc.count('<img ')!=len(selected):raise RuntimeError('Embedded image count mismatch')
    print(f'[OK] standalone HTML {a.output} | {len(selected)} embedded images | {a.output.stat().st_size/1048576:.1f} MiB',flush=True)

if __name__=='__main__':main()