#!/usr/bin/env python3
"""Cluster the *entire* USB demonstration by six signed tactile force maps.

No contact, action, pose or success labels enter clustering or K selection.
Each taxel of left/right base-frame Fx/Fy/Fz contributes to the distance.
The report uses those withheld quantities only to describe the resulting
classes and the subsequent behavior. Results are observational.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from run_contactworld_matched_dynamics import atomic_json, encode_frames, color_force
from run_contactworld_sensor_discovery import DEFAULT_DATASET, image_paths

DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_clusters_full_20261008')
VERSION = 'cw-tactile-whole-trajectory-v1'
SPLITS = ('train','val','test')


def load_episodes(args):
    manifest=json.loads((args.dataset/'spatial/manifest.json').read_text())
    entries=manifest['episodes']
    if args.smoke:
        entries=[e for s,n in [('train',6),('val',2),('test',2)]
                 for e in [x for x in entries if x['split']==s][:n]]
    capture_root=Path(manifest['capture_root'])
    episodes={};parts={s:[] for s in SPLITS}
    for i,e in enumerate(entries,1):
        ep=int(e['episode_index']);source=int(e['source_episode']);n=int(e['length'])
        spatial=args.dataset/f'spatial/episodes/episode_{ep:06d}'
        forces=np.stack([np.load(spatial/f'tactile_force_{hand}_base.npy')
                         for hand in ('left','right')],axis=1).astype(np.float32)
        if forces.shape!=(n,2,140,3):raise ValueError(f'Bad bilateral force shape: source {source}: {forces.shape}')
        folder=capture_root/f'insertion_usb_episode_{source:03d}'
        with np.load(folder/'v2_capture.npz') as raw:
            plug=raw['plug_pose'].copy();socket=raw['socket_pose'].copy();ee=raw['ee_pose'].copy()
            action=raw['action_executed'].copy()
        replay=json.loads((folder/'replay.json').read_text())['rows']
        if any(len(x)!=n for x in (plug,socket,ee,action,replay)):
            raise ValueError(f'Capture/replay length mismatch: source {source}')
        sock_rotation=Rotation.from_quat(socket[:,3:])
        rel=(sock_rotation.inv().apply(plug[:,:3]-socket[:,:3])*1000).astype(np.float32)
        action_xyz=sock_rotation.inv().apply(action[:,:3]*.01)*1000
        action_rot=action[:,3:]*(.05*180/np.pi)
        action_scaled=np.concatenate((action_xyz,action_rot),axis=1).astype(np.float32)
        next_motion=np.zeros((n,3),np.float32)
        next_motion[:-1]=rel[1:]-rel[:-1]
        contact=np.r_[False,[np.linalg.norm(r['plug_socket_force_post'])>.1 for r in replay[:-1]]]
        success=np.r_[False,[bool(r['source_success']) for r in replay[:-1]]]
        item=dict(entry=e,force=forces,relative=rel,action=action_scaled,
                  next_motion=next_motion,contact=contact,success=success,
                  grasp=(plug[:,:3]-ee[:,:3]).astype(np.float32))
        episodes[(e['split'],source)]=item
        parts[e['split']].append((e,item))
        print(f'load {i}/{len(entries)} {e["split"]} source={source} frames={n}',flush=True)
    return entries,episodes,parts


def concatenate(parts):
    data={}
    for split in SPLITS:
        e=parts[split]
        data[split]={name:np.concatenate([d[name] for _,d in e])
                     for name in ('force','relative','action','next_motion','contact','success')}
        data[split]['source']=np.concatenate([np.full(len(d['force']),x['source_episode'],np.int32) for x,d in e])
        data[split]['episode']=np.concatenate([np.full(len(d['force']),x['episode_index'],np.int32) for x,d in e])
        data[split]['frame']=np.concatenate([np.arange(len(d['force']),dtype=np.int32) for _,d in e])
    return data


def normalized_force(data):
    # Equal scale per physical finger/axis, fitted on *all train frames*.
    # No per-frame RMS normalization: overall force strength remains available.
    scale=np.quantile(np.abs(data['train']['force']),.995,axis=(0,2)).astype(np.float32)
    scale=np.maximum(scale,1e-6)
    features={}
    for split,d in data.items():
        force=d['force']/scale[None,:,None,:]
        features[split]=np.clip(force,-3,3).reshape(len(force),840).astype(np.float32)
    return features,scale


def assignment(features,centers):
    distance=cdist(features,centers,'sqeuclidean')/features.shape[1]
    labels=distance.argmin(axis=1).astype(np.int16)
    nearest=np.sqrt(distance[np.arange(len(labels)),labels]).astype(np.float32)
    return labels,nearest


def silhouette_sample(x,labels):
    # Sampled validation silhouette uses the same 840-D metric as clustering.
    if len(np.unique(labels))<2:return -1.0
    dm=cdist(x,x,'euclidean')/np.sqrt(x.shape[1])
    groups={int(k):np.flatnonzero(labels==k) for k in np.unique(labels)}
    scores=[]
    for i in range(len(x)):
        own=groups[int(labels[i])]
        if len(own)<2:continue
        intra=dm[i,own].sum()/(len(own)-1)
        inter=min(dm[i,ids].mean() for k,ids in groups.items() if k!=int(labels[i]))
        scores.append((inter-intra)/max(intra,inter,1e-9))
    return float(np.mean(scores)) if scores else -1.0


def adjusted_rand(a,b):
    # Permutation-invariant cluster agreement. A low value warns of unstable boundaries.
    a=np.asarray(a);b=np.asarray(b);na=np.unique(a,return_inverse=True)[1];nb=np.unique(b,return_inverse=True)[1]
    counts=np.zeros((na.max()+1,nb.max()+1),np.int64)
    np.add.at(counts,(na,nb),1)
    comb=lambda x:x*(x-1)/2
    both=comb(counts).sum();left=comb(counts.sum(1)).sum();right=comb(counts.sum(0)).sum()
    total=comb(len(a));expected=left*right/max(total,1);maximum=.5*(left+right)
    return float((both-expected)/max(maximum-expected,1e-9))


def choose_k(args,features):
    train=features['train'];val=features['val'];rng=np.random.default_rng(20261008)
    sample_idx=rng.choice(len(val),min(args.silhouette_frames,len(val)),replace=False)
    candidates=list(range(args.k_min,args.k_max+1,args.k_step))
    if args.smoke:candidates=[k for k in candidates if k<=min(10,len(train)//20)]
    if args.k is not None and args.k not in candidates:candidates.append(args.k)
    candidates=sorted(set(candidates))
    if not candidates:raise ValueError('No valid candidate K; change --k-min/--k-max')
    results=[];models={}
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,80,1e-4)
    for k in candidates:
        cv2.setRNGSeed(6100+k)
        compactness,labels,centers=cv2.kmeans(train,k,None,criteria,4,cv2.KMEANS_PP_CENTERS)
        centers=centers.astype(np.float32);val_labels,val_distance=assignment(val,centers)
        cv2.setRNGSeed(8100+k)
        _,_,alternative=cv2.kmeans(train,k,None,criteria,1,cv2.KMEANS_PP_CENTERS)
        val_alternative,_=assignment(val,alternative)
        train_count=np.bincount(labels[:,0],minlength=k)
        val_count=np.bincount(val_labels,minlength=k)
        sil=silhouette_sample(val[sample_idx],val_labels[sample_idx])
        eligible=(train_count.min()>=max(10,int(.01*len(train))) and
                  val_count.min()>=max(4,int(.01*len(val))))
        record=dict(k=k,val_silhouette=sil,val_rms=float(np.sqrt(np.mean(val_distance**2))),
                    train_rms=float(np.sqrt(compactness/(len(train)*train.shape[1]))),
                    stability_ari=adjusted_rand(val_labels,val_alternative),
                    train_min_count=int(train_count.min()),val_min_count=int(val_count.min()),
                    eligible=bool(eligible))
        results.append(record);models[k]=centers
        print(f'K sweep {k}: silhouette={sil:.3f} val_rms={record["val_rms"]:.3f} '
              f'ARI={record["stability_ari"]:.3f} min_val={val_count.min()} eligible={eligible}',flush=True)
    if args.k is not None:chosen=args.k;reason='explicit --k override'
    else:
        eligible=[r for r in results if r['eligible']]
        options=eligible if eligible else results
        best=max(r['val_silhouette'] for r in options)
        # Near-ties favor fewer groups, avoiding a false claim that K is exact.
        chosen=min(r['k'] for r in options if r['val_silhouette']>=best-.005)
        reason='highest held-out validation silhouette within 0.005, with >=1% train/val occupancy when feasible'
    return chosen,models[chosen],results,reason


def annotate(data,features,centers):
    out={}
    for split in SPLITS:
        label,distance=assignment(features[split],centers)
        out[split]=dict(label=label,distance=distance)
    return out


def action_summary(d,mask):
    a=d['action'][mask];motion=d['next_motion'][mask];contact=d['contact'][mask]
    down=a[:,2]<-2;lateral=np.linalg.norm(a[:,:2],axis=1)>=2.5
    rotate=np.linalg.norm(a[:,3:],axis=1)>=.4
    progress=down & (motion[:,2]<-.4);blocked=down & (motion[:,2]>-.25)
    return dict(frames=int(mask.sum()),contact_fraction=float(contact.mean()),
      down_command_fraction=float(down.mean()),lateral_command_fraction=float(lateral.mean()),
      rotation_command_fraction=float(rotate.mean()),
      down_progress=int(progress.sum()),down_block=int(blocked.sum()),
      progress_given_decisive_down=float(progress.sum()/max(1,progress.sum()+blocked.sum())),
      median_action_xyz_mm=np.median(a[:,:3],axis=0).tolist(),
      median_next_motion_xyz_mm=np.median(motion,axis=0).tolist())


def summarize(data,assigned,k):
    result={}
    for split in SPLITS:
        d=data[split];label=assigned[split]['label'];rows=[]
        for cluster in range(k):
            mask=label==cluster
            if not mask.any():
                rows.append(dict(cluster=cluster,frames=0,demonstrations=0));continue
            row=dict(cluster=cluster,demonstrations=int(np.unique(d['source'][mask]).size),
                     mean_center_distance=float(assigned[split]['distance'][mask].mean()),
                     median_center_distance=float(np.median(assigned[split]['distance'][mask])))
            row.update(action_summary(d,mask));rows.append(row)
        result[split]=rows
    return result


def six_map_max_rms(features,center,ids):
    # A gallery frame must be close on every finger/axis map, not merely on
    # the 840-dimensional average. This score is used for display selection;
    # clustering and class assignment still use the declared Euclidean metric.
    diff=features[ids].reshape(-1,2,140,3)-center.reshape(1,2,140,3)
    return np.sqrt(np.mean(diff*diff,axis=2)).max(axis=(1,2))


def select_representatives(args,episodes,data,assigned,features,centers,k):
    selected={}
    for cluster in range(k):
        cases=[];used=set()
        # Held-out visual examples first, then validation and training examples.
        quotas={'test':min(3,args.examples_per_cluster),'val':1,'train':args.examples_per_cluster}
        for split in ('test','val','train'):
            d=data[split];mask=np.flatnonzero(assigned[split]['label']==cluster)
            order=mask[np.argsort(six_map_max_rms(features[split],centers[cluster],mask))]
            for idx in order:
                source=int(d['source'][idx]);key=(split,source)
                if key in used:continue
                used.add(key)
                cases.append(dict(split=split,source=source,episode=int(d['episode'][idx]),
                    frame=int(d['frame'][idx]),cluster=cluster,
                    center_distance=float(assigned[split]['distance'][idx]),
                    relative_mm=d['relative'][idx].round(2).tolist(),
                    contact=bool(d['contact'][idx]),success=bool(d['success'][idx]),
                    action=d['action'][idx].round(2).tolist(),
                    next_motion=d['next_motion'][idx].round(2).tolist()))
                if sum(x['split']==split for x in cases)>=quotas[split] or len(cases)>=args.examples_per_cluster:
                    break
            if len(cases)>=args.examples_per_cluster:break
        # If a split quota did not fill the gallery, add the nearest remaining
        # members from any split while preserving distinct demonstrations.
        if len(cases)<args.examples_per_cluster:
            pool=[]
            for split in SPLITS:
                d=data[split];ids=np.flatnonzero(assigned[split]['label']==cluster)
                scores=six_map_max_rms(features[split],centers[cluster],ids)
                for idx,score in zip(ids,scores):pool.append((float(score),split,int(idx)))
            for _,split,idx in sorted(pool):
                d=data[split];source=int(d['source'][idx]);key=(split,source)
                if key in used:continue
                used.add(key)
                cases.append(dict(split=split,source=source,episode=int(d['episode'][idx]),
                    frame=int(d['frame'][idx]),cluster=cluster,
                    center_distance=float(assigned[split]['distance'][idx]),
                    relative_mm=d['relative'][idx].round(2).tolist(),
                    contact=bool(d['contact'][idx]),success=bool(d['success'][idx]),
                    action=d['action'][idx].round(2).tolist(),
                    next_motion=d['next_motion'][idx].round(2).tolist()))
                if len(cases)>=args.examples_per_cluster:break
        selected[cluster]=cases
    return selected


def read_video_frames(path,indices):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise FileNotFoundError(f'Cannot decode {path}')
    frames=[]
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Missing frame {i}: {path}')
            frames.append(cv2.resize(frame,(320,320),interpolation=cv2.INTER_AREA))
    finally:cap.release()
    return frames


def six_force_image(force,scale):
    tile_w=210;row_h=175
    canvas=np.full((row_h*2,tile_w*3,3),28,np.uint8)
    for hand in range(2):
        for axis in range(3):
            x=axis*tile_w;y=hand*row_h
            grid=force[hand,:,axis].reshape(10,14)
            canvas[y+27:y+167,x+7:x+203]=color_force(grid,float(scale[hand,axis]))
            cv2.putText(canvas,f"{'L' if hand==0 else 'R'} F{'xyz'[axis]}",(x+8,y+19),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
    return canvas


def composite(front,wrist,left,right,episode_data,scale,case,shown):
    canvas=np.full((750,1280,3),25,np.uint8)
    canvas[50:370,0:320]=front;canvas[50:370,320:640]=wrist
    cv2.putText(canvas,'FRONT',(8,33),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1)
    cv2.putText(canvas,'WRIST',(327,33),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1)
    tile=six_force_image(episode_data['force'][shown],scale)
    canvas[35:385,645:1275]=tile
    canvas[425:745,0:320]=left;canvas[425:745,320:640]=right
    cv2.putText(canvas,'LEFT TACTILE PREVIEW',(8,408),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1)
    cv2.putText(canvas,'RIGHT TACTILE PREVIEW',(327,408),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1)
    xyz=episode_data['relative'][shown];a=episode_data['action'][shown]
    m=episode_data['next_motion'][shown]
    lines=[f"CLUSTER {case['cluster']} | source {case['source']} | marked {case['frame']} | shown {shown} ({shown-case['frame']:+d})",
           f"contact {int(episode_data['contact'][shown])} | success {int(episode_data['success'][shown])} | {case['split']}",
           f'plug relative xyz mm: {xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f}',
           f'command xyz mm/step: {a[0]:+.1f}, {a[1]:+.1f}, {a[2]:+.1f}',
           f'next actual plug dz mm: {m[2]:+.2f}',
           f'center distance: {case["center_distance"]:.3f}',
           'Base-frame TacFF: red +, blue -; same scale for every clip']
    for j,line in enumerate(lines):
        cv2.putText(canvas,line,(650,427+j*43),cv2.FONT_HERSHEY_SIMPLEX,.53,(235,235,235),1,cv2.LINE_AA)
    return canvas


def render_examples(args,selected,episodes,scale,centers):
    out=args.output/'examples';out.mkdir(parents=True,exist_ok=True)
    for cluster,cases in selected.items():
        # The center image is the mean six-map centroid; cards show actual frames.
        center=centers[cluster].reshape(2,140,3)*scale[:,None,:]
        cv2.imwrite(str(out/f'cluster_{cluster:02d}_center.png'),six_force_image(center,scale))
        for i,case in enumerate(cases):
            d=episodes[(case['split'],case['source'])]
            n=len(d['force']);t=case['frame'];lo=max(0,t-8);hi=min(n,t+13)
            normalized=np.clip(d['force'][t]/scale[:,None,:],-3,3)
            center_map=centers[cluster].reshape(2,140,3)
            case['six_map_rms']=np.sqrt(np.mean((normalized-center_map)**2,axis=1)).round(3).tolist()
            indices=list(range(lo,hi));paths=image_paths(args.dataset,case['episode'])
            stem=f'cluster_{cluster:02d}_{i:02d}_{case["split"]}_source{case["source"]:03d}_f{t:03d}'
            movie=out/f'{stem}.mp4';marker=out/f'{stem}_force.png'
            cv2.imwrite(str(marker),six_force_image(d['force'][t],scale))
            if not movie.is_file():
                front=read_video_frames(paths['front'],indices)
                wrist=read_video_frames(paths['wrist'],indices)
                left=read_video_frames(paths['tacff_left_preview'],indices)
                right=read_video_frames(paths['tacff_right_preview'],indices)
                encode_frames(movie,1280,750,(composite(a,b,c,h,d,scale,case,j)
                    for a,b,c,h,j in zip(front,wrist,left,right,indices)),fps=10)
                # Poster is the class representative frame, not the clip start.
                cv2.imwrite(str(movie.with_suffix('.jpg')),composite(front[t-lo],wrist[t-lo],
                    left[t-lo],right[t-lo],d,scale,case,t))
            case['video']=str(movie.relative_to(args.output));case['marker_force']=str(marker.relative_to(args.output))
            case['poster']=str(movie.with_suffix('.jpg').relative_to(args.output))
            case['marked_video_frame']=t-lo
            print(f'render class {cluster} example {i+1}/{len(cases)} {stem}',flush=True)
    atomic_json(args.output/'representatives.json',selected)


def plot_diagnostics(args,k_results,chosen,centers,data,assigned,summary):
    out=args.output/'figures';out.mkdir(exist_ok=True)
    ks=[r['k'] for r in k_results]
    fig,axes=plt.subplots(1,3,figsize=(14,3.5),layout='constrained')
    axes[0].plot(ks,[r['val_silhouette'] for r in k_results],marker='o');axes[0].set_ylabel('validation silhouette')
    axes[1].plot(ks,[r['val_rms'] for r in k_results],marker='o');axes[1].set_ylabel('validation map RMS')
    axes[2].plot(ks,[r['stability_ari'] for r in k_results],marker='o');axes[2].set_ylabel('validation stability ARI')
    for ax in axes:ax.axvline(chosen,color='#d45a43',ls=':');ax.set_xlabel('K');ax.grid(alpha=.2)
    fig.savefig(out/'k_selection.png',dpi=145);plt.close(fig)
    dist=cdist(centers,centers,'euclidean')/np.sqrt(centers.shape[1])
    fig,ax=plt.subplots(figsize=(6.5,5.6),layout='constrained');im=ax.imshow(dist,cmap='viridis')
    ax.set_xticks(range(chosen));ax.set_yticks(range(chosen));ax.set_xlabel('class');ax.set_ylabel('class')
    fig.colorbar(im,ax=ax,label='six-map normalized RMS distance')
    fig.savefig(out/'class_separation.png',dpi=145);plt.close(fig)
    rows=summary['test'];x=np.arange(chosen);width=.25
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    ax.bar(x-width,[r.get('contact_fraction',0) for r in rows],width,label='socket contact')
    ax.bar(x,[r.get('down_command_fraction',0) for r in rows],width,label='downward command')
    ax.bar(x+width,[r.get('lateral_command_fraction',0) for r in rows],width,label='lateral command')
    ax.set_xticks(x);ax.set_xlabel('tactile class');ax.set_ylabel('fraction of class frames');ax.set_ylim(0,1)
    ax.legend(ncol=3);fig.savefig(out/'action_associations.png',dpi=145);plt.close(fig)
    # The test timeline makes pre-contact and contact-era cluster occupancy visible.
    test=data['test'];label=assigned['test']['label'];sources=np.unique(test['source'])
    fig,ax=plt.subplots(figsize=(13,max(3,len(sources)*.38)),layout='constrained')
    canvas=np.full((len(sources),max(np.bincount(test['source']))),np.nan)
    for row,src in enumerate(sources):
        ids=np.flatnonzero(test['source']==src)
        canvas[row,:len(ids)]=label[ids]
    cm=plt.get_cmap('tab20',chosen);im=ax.imshow(canvas,aspect='auto',interpolation='nearest',cmap=cm,vmin=-.5,vmax=chosen-.5)
    ax.set_yticks(range(len(sources)),[f'source {s}' for s in sources]);ax.set_xlabel('control frame')
    fig.colorbar(im,ax=ax,ticks=range(chosen),label='tactile class')
    fig.savefig(out/'test_timelines.png',dpi=145);plt.close(fig)


def save_assignments(args,data,assigned):
    path=args.output/'frame_assignments.csv'
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','source','episode','frame','class','distance','socket_contact',
                                    'cmd_x_mm','cmd_y_mm','cmd_z_mm','next_plug_dz_mm','success'])
        for split in SPLITS:
            d=data[split];a=assigned[split]
            for i in range(len(a['label'])):
                w.writerow([split,int(d['source'][i]),int(d['episode'][i]),int(d['frame'][i]),
                    int(a['label'][i]),round(float(a['distance'][i]),5),int(d['contact'][i]),
                    *[round(float(x),3) for x in d['action'][i,:3]],
                    round(float(d['next_motion'][i,2]),3),int(d['success'][i])])


def report(args,selected,summary,k_results,chosen,reason,scale):
    css='''<style>body{font:16px/1.55 system-ui;background:#101827;color:#ecf3fc;max-width:1900px;margin:auto;padding:22px}
nav{display:flex;gap:11px;flex-wrap:wrap;background:#243b59;padding:13px;position:sticky;top:0;z-index:5}a{color:#9bd1ff}
section{background:#1d2a3e;margin:20px 0;padding:20px;border-radius:12px;scroll-margin-top:70px}article{background:#263750;padding:13px;border-radius:9px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.grid video,.grid img{max-width:100%;height:auto}.force{width:630px}
button{background:#2d75af;color:white;border:0;border-radius:6px;padding:9px 13px;margin:5px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #53677f;padding:6px;text-align:left}th{background:#354a68}
.note{background:#4a3927;padding:10px;border-radius:7px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>'''
    nav='<nav><a href="#method">聚类方法</a>'+' '.join(
        f'<a href="#class-{i}">类 {i}</a>' for i in range(chosen))+' <a href="#selection">类数选择</a></nav>'
    ks=''.join(f'<tr><td>{r["k"]}</td><td>{r["val_silhouette"]:.3f}</td><td>{r["val_rms"]:.3f}</td><td>{r["stability_ari"]:.3f}</td><td>{r["val_min_count"]}</td><td>{"✓" if r["eligible"] else "—"}</td></tr>'
               for r in k_results)
    sections=[]
    for cluster in range(chosen):
        cases=selected[cluster];r=summary['test'][cluster]
        examples=[]
        for case in cases:
            delta=', '.join(f'{v:+.1f}' for v in case['action'][:3])
            map_rms='；'.join(f'{hand} Fx {values[0]:.2f} / Fy {values[1]:.2f} / Fz {values[2]:.2f}'
                             for hand,values in zip(('左','右'),case['six_map_rms']))
            ex=f'''<article><h3>{html.escape(case['split'])} · source {case['source']} · 帧 {case['frame']}</h3>
<p>中心距离 {case['center_distance']:.3f}；插座接触 {int(case['contact'])}；当前动作 XYZ {delta} mm/步；
下一步实际 Z 位移 {case['next_motion'][2]:+.2f} mm。<br>六张图各自到类中心的归一化 RMS：{map_rms}。</p>
<img class="force" width="630" height="350" loading="lazy" src="{html.escape(case['marker_force'])}" alt="left/right Fx Fy Fz at marked frame">
<video controls preload="none" width="1280" height="750" poster="{html.escape(case['poster'])}" src="{html.escape(case['video'])}" data-mark="{case['marked_video_frame']}"></video></article>'''
            examples.append(ex)
        cases_note=''.join(examples) if examples else '<p>此类没有可展示的独立示范。</p>'
        sections.append(f'''<section id="class-{cluster}"><h2>类 {cluster}：{r['frames']} 个测试帧，{r['demonstrations']} 条测试示范</h2>
<p>插座接触 {r.get('contact_fraction',0):.0%}；下推指令 {r.get('down_command_fraction',0):.0%}；
横向指令 {r.get('lateral_command_fraction',0):.0%}。实际下推后：推进 {r.get('down_progress',0)}、受阻 {r.get('down_block',0)}；
只在有明确结果的下推帧中，推进比例 {r.get('progress_given_decisive_down',0):.0%}。</p>
<p>类中心的六张图（左/右指，各 Fx/Fy/Fz）：</p><img class="force" width="630" height="350" src="examples/cluster_{cluster:02d}_center.png">
<p><button onclick="playAligned({cluster})">从各自标记帧同步播放</button><button onclick="playStart({cluster})">从片段开头同步播放</button>
<button onclick="pauseAll({cluster})">暂停</button><label>共同偏移 <input type="range" min="-8" max="12" value="0" oninput="scrub({cluster},this.value)"> <span id="offset-{cluster}">0</span> 步</label></p>
<div class="grid" data-class="{cluster}">{cases_note}</div></section>''')
    script='''<script>
function videos(k){return [...document.querySelectorAll('[data-class="'+k+'"] video')]}
function ready(v){return new Promise(resolve=>{if(v.readyState>=1){resolve();return}v.addEventListener('loadedmetadata',resolve,{once:true});v.load()})}
async function playAligned(k){const a=videos(k);await Promise.all(a.map(ready));a.forEach(v=>{v.pause();v.currentTime=Number(v.dataset.mark)/10});await Promise.all(a.map(v=>v.play().catch(()=>{})))}
async function playStart(k){const a=videos(k);await Promise.all(a.map(ready));a.forEach(v=>{v.pause();v.currentTime=0});await Promise.all(a.map(v=>v.play().catch(()=>{})))}
function pauseAll(k){videos(k).forEach(v=>v.pause())}
async function scrub(k,offset){document.getElementById('offset-'+k).textContent=offset;const a=videos(k);await Promise.all(a.map(ready));a.forEach(v=>{v.pause();v.currentTime=Math.max(0,Math.min((Number(v.dataset.mark)+Number(offset))/10,v.duration-.05))})}
window.addEventListener('load',()=>{if(location.hash){setTimeout(()=>{const e=document.querySelector(location.hash);if(e)e.scrollIntoView({block:'start'})},100)}})
</script>'''
    scale_text='；'.join(f"{hand} Fx {scale[h,0]:.6f}, Fy {scale[h,1]:.6f}, Fz {scale[h,2]:.6f}"
                        for h,hand in enumerate(('左指','右指')))
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ContactWorld 全程双指触觉聚类</title>{css}
<h1>ContactWorld USB：全程双指三维力图聚类</h1><p>这次从接近、接触、调整到完成都参与聚类；没有先按插座接触筛帧。</p>{nav}
<section id="method"><h2>怎么定义“触觉相似”</h2><p>每帧输入左指 Fx/Fy/Fz 与右指 Fx/Fy/Fz，共 840 个有符号 taxel 值。
每个手指/分量用训练集绝对力的 99.5% 分位缩放，再限制极端值在 ±3；之后直接对六张图做欧氏距离和 K-means。
<b>没有逐帧除以力强度，也没有先做 PCA</b>，所以强度、方向和六张图的空间分布都会影响相似度。
所有例子以相同的六分量颜色标尺渲染：红为正，蓝为负。接触、动作、真实推进只用于事后解读。</p>
<p class="note">同类表示在这个度量下比较接近，不保证每张图像素级相同。看每张标记帧图、类中心图和中心距离；
如果某类内部明显不像，说明类数或距离定义还需要改。触觉可能滞后于前一步动作；类别与动作的关系不是因果关系。</p>
<p>六分量固定标尺（仿真力单位）：{scale_text}。</p></section>
{''.join(sections)}
<section id="selection"><h2>为什么选 {chosen} 类</h2><p>训练集拟合候选 K，验证集（整条示范隔离）比较原始六图距离的 silhouette；
同时要求每类覆盖至少约 1% 的训练和验证帧。验证 silhouette 相差不超过 0.005 时选较小 K。
这是一条公开的选择规则，不表示存在唯一“真实类数”。本次：{html.escape(reason)}。</p>
<img src="figures/k_selection.png"><table><tr><th>K</th><th>验证 silhouette ↑</th><th>验证 RMS ↓</th><th>重启稳定性 ARI ↑</th><th>验证集最小类帧数</th><th>合格</th></tr>{ks}</table>
<h3>类之间和整段轨迹</h3><img src="figures/class_separation.png"><img src="figures/action_associations.png"><img src="figures/test_timelines.png"></section>
<section><h2>数据与边界</h2><p><a href="k_selection.json">类数比较</a> · <a href="cluster_summary.json">逐类统计</a> ·
<a href="representatives.json">代表帧</a> · <a href="frame_assignments.csv">全部逐帧类别与动作</a>。</p>
<p>类中心和代表帧只根据触觉选择；所有类别的接触和动作比例都是观察性统计。若要知道某一触觉类该做哪个修正动作，
需从同一仿真状态执行不同动作分支，并比较实际结果。</p></section>{script}</html>'''
    (args.output/'index.html').write_text(content)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--k',type=int,help='Override automatically selected K')
    p.add_argument('--k-min',type=int,default=4);p.add_argument('--k-max',type=int,default=18)
    p.add_argument('--k-step',type=int,default=2)
    p.add_argument('--silhouette-frames',type=int,default=700)
    p.add_argument('--examples-per-cluster',type=int,default=5)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();start=time.monotonic()
    if args.k_min<2 or args.k_step<1 or (args.k is not None and args.k<2):p.error('K must be >=2')
    args.output.mkdir(parents=True,exist_ok=True)
    entries,episodes,parts=load_episodes(args)
    identity=dict(version=VERSION,smoke=args.smoke,entries=[[e['split'],e['source_episode']] for e in entries],
                  examples_per_cluster=args.examples_per_cluster,k=args.k,k_range=[args.k_min,args.k_max,args.k_step])
    identity_path=args.output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Output identity differs; choose another --output')
    atomic_json(identity_path,identity)
    (args.output/'complete.json').unlink(missing_ok=True)
    data=concatenate(parts)
    features,scale=normalized_force(data)
    chosen,centers,sweep,reason=choose_k(args,features)
    assigned=annotate(data,features,centers)
    summary=summarize(data,assigned,chosen)
    selected=select_representatives(args,episodes,data,assigned,features,centers,chosen)
    atomic_json(args.output/'k_selection.json',dict(chosen_k=chosen,reason=reason,candidates=sweep,
                distance='RMS across all 840 signed force-map entries after fixed per-finger/axis train scale'))
    atomic_json(args.output/'cluster_summary.json',summary)
    np.savez_compressed(args.output/'cluster_model.npz',centers=centers,scale=scale)
    save_assignments(args,data,assigned)
    plot_diagnostics(args,sweep,chosen,centers,data,assigned,summary)
    render_examples(args,selected,episodes,scale,centers)
    report(args,selected,summary,sweep,chosen,reason,scale)
    atomic_json(args.output/'complete.json',dict(version=VERSION,elapsed_s=round(time.monotonic()-start,1),
                frames={s:len(data[s]['force']) for s in SPLITS},chosen_k=chosen,
                example_videos=sum(map(len,selected.values()))))
    print(f'DONE k={chosen} elapsed_s={time.monotonic()-start:.1f} report={args.output/"index.html"}',flush=True)


if __name__=='__main__':main()
