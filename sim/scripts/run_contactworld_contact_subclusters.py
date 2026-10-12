#!/usr/bin/env python3
"""Explore raw K14 contact-enriched frames with tactile-only subclusters and held-out action probes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from contactworld_neighbor_features import DEFAULT_WEIGHTS, Features, decode
from run_contactworld_sensor_discovery import fit_pca, image_paths
from run_contactworld_tactile_cluster_atlas import (
    adjusted_rand, assignment, load_episodes, read_video_frames,
    silhouette_sample, six_force_image,
)
from run_contactworld_matched_dynamics import atomic_json, encode_frames

REPO=Path(__file__).resolve().parents[2]
DEFAULT_DATA=REPO/'data/contactworld_usb_v3_bilateral'
DEFAULT_ATLAS=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_clusters_k14_20261008')
DEFAULT_CACHE=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_sensor_discovery_20261004')
DEFAULT_OUT=Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_contact_subclusters_20261009')
VERSION='contact-subclusters-and-predictability-v1'
PARENT=(5,7,9)
SPLITS=('train','val','test')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read_parent(path):
    out=defaultdict(list)
    for row in csv.DictReader(path.open(newline='')):
        out[(row['split'],int(row['source']))].append((int(row['frame']),int(row['class'])))
    return out


def load(args):
    manifest_path=args.dataset/'spatial/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assignment_file=args.atlas/'frame_assignments.csv'
    parents=read_parent(assignment_file)
    with np.load(args.atlas/'cluster_model.npz') as z:
        original_centers=z['centers'].copy();scale=z['scale'].copy()
    if original_centers.shape!=(14,840) or scale.shape!=(2,3):raise ValueError('Unexpected frozen raw K14 model')
    if args.rgb_cache.exists():
        cache_id=json.loads((args.rgb_cache/'identity.json').read_text())
        if cache_id['manifest_sha256']!=digest(manifest_path) or cache_id.get('smoke'):
            raise ValueError('Frozen RGB cache does not match dataset manifest')
    entries,episodes,_=load_episodes(SimpleNamespace(dataset=args.dataset,smoke=args.smoke))
    extractor=None;records={};data={s:[] for s in SPLITS}
    for e in entries:
        key=(e['split'],int(e['source_episode']));d=episodes[key];n=len(d['force'])
        original=np.array([c for t,c in parents[key]],np.int16)
        if len(original)!=n or any(t!=i for i,(t,_) in enumerate(parents[key])):
            raise ValueError(f'Atlas source/frame mismatch: {key}')
        normal=np.clip(d['force']/scale[None,:,None,:],-3,3).reshape(n,840).astype(np.float32)
        checked,_=assignment(normal,original_centers)
        if not np.array_equal(checked,original):raise ValueError(f'K14 force mismatch: {key}')
        ep=int(e['episode_index']);source=key[1]
        cache=args.rgb_cache/'episodes'/f'{key[0]}_source_{source:03d}.npz'
        if cache.exists():
            with np.load(cache) as z:
                rgb=np.concatenate((z['rgb'],z['crop']),axis=1)
                state=z['state'].copy()
                if not np.allclose(z['force_base'],d['force'][:len(state)],atol=1e-6):
                    raise ValueError(f'RGB cache force mismatch: {key}')
        else:
            if extractor is None:extractor=Features(args.vision_weights,args.device,args.batch)
            paths=image_paths(args.dataset,ep)
            front=decode(paths['front'],n-4);wrist=decode(paths['wrist'],n-4)
            rgb=np.concatenate((extractor.rgb(front),extractor.rgb(wrist),
                                extractor.rgb(front[:,32:224,32:224]),
                                extractor.rgb(wrist[:,32:224,32:224])),axis=1)
            folder=Path(manifest['capture_root'])/f'insertion_usb_episode_{source:03d}'
            with np.load(folder/'v2_capture.npz') as z:state=z['state'][:n-4].copy()
        if len(rgb)<n-4 or len(state)<n-4:raise ValueError(f'RGB/state cache too short: {key}')
        # Include all K14 contact-enriched frames in clustering. For action
        # probes, require history, future and pre-success observations.
        times=np.flatnonzero(np.isin(original,PARENT))
        record=dict(entry=e,d=d,normal=normal,raw=original,rgb=rgb,state=state)
        records[key]=record
        for t in times:
            item=dict(split=key[0],source=source,episode=ep,frame=int(t),raw=int(original[t]),
                      force=normal[t],contact=int(d['contact'][t]),
                      relative=d['relative'][t].copy(),action=d['action'][t].copy(),
                      dz=float(d['next_motion'][t,2]),
                      eligible=bool(3<=t<n-4 and not d['success'][:t+1].any()))
            data[key[0]].append(item)
    print('Selected original K14 5/7/9 frames:',{s:len(data[s]) for s in SPLITS},flush=True)
    if min(map(len,data.values()))<3:raise ValueError('A train/val/test split has fewer than three selected frames')
    return records,data,scale,manifest


def cluster(args,data):
    features={s:np.stack([r['force'] for r in data[s]]).astype(np.float32) for s in SPLITS}
    x=features['train'];v=features['val'];rng=np.random.default_rng(20261009)
    if args.smoke:ks=[k for k in (2,3,4) if k<len(x)//5]
    else:ks=list(range(2,args.k_max+1))
    if args.k is not None and args.k not in ks:ks.append(args.k)
    result=[];models={};criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,1e-4)
    for k in sorted(set(ks)):
        cv2.setRNGSeed(5200+k)
        compact,train_label,centers=cv2.kmeans(x,k,None,criteria,4,cv2.KMEANS_PP_CENTERS)
        centers=centers.astype(np.float32);val_label,val_dist=assignment(v,centers)
        cv2.setRNGSeed(6900+k)
        _,_,alt=cv2.kmeans(x,k,None,criteria,1,cv2.KMEANS_PP_CENTERS)
        alt_label,_=assignment(v,alt)
        train_count=np.bincount(train_label[:,0],minlength=k)
        val_count=np.bincount(val_label,minlength=k)
        silhouette=silhouette_sample(v,val_label)
        occupancy=int(train_count.min())>=max(8,len(x)//100) and int(val_count.min())>=2
        row=dict(k=k,val_silhouette=float(silhouette),val_rms=float(np.sqrt(np.mean(val_dist**2))),
                 stability_ari=float(adjusted_rand(val_label,alt_label)),
                 min_train=int(train_count.min()),min_val=int(val_count.min()),occupancy_ok=bool(occupancy))
        result.append(row);models[k]=centers
        print('K sweep',row,flush=True)
    if args.k is not None:chosen=args.k;reason='explicit --k'
    else:
        viable=[r for r in result if r['occupancy_ok'] and r['stability_ari']>=.35]
        if viable:quality='occupied and stable'
        else:
            viable=[r for r in result if r['occupancy_ok']]
            quality='occupied; stability criterion not met'
        if not viable:
            viable=result
            quality='exploratory; minimum occupancy criterion not met'
        best=max(r['val_silhouette'] for r in viable)
        chosen=min(r['k'] for r in viable if r['val_silhouette']>=best-.01)
        reason=f'validation silhouette within .01 of best ({quality}); simpler K wins ties'
    centers=models[chosen]
    assigned={}
    for s in SPLITS:
        label,distance=assignment(features[s],centers)
        assigned[s]=dict(label=label,distance=distance)
        for i,row in enumerate(data[s]):row['subcluster']=int(label[i]);row['distance']=float(distance[i])
    print('Selected subcluster K',chosen,reason,flush=True)
    return centers,result,chosen,reason,assigned,models


def features_and_targets(records,data):
    # Only frames before the first success are prediction examples; all parent
    # frames remain in the descriptive cluster atlas.
    result={}
    for s in SPLITS:
        selected=[r for r in data[s] if r['eligible']]
        if not selected:raise ValueError(f'No pre-success examples in {s}')
        arrays={k:[] for k in ('state','rgb','past_action','tactile','delta1','delta3','action','lateral','rotation','next_dz','source','frame','cluster')}
        for row in selected:
            t=row['frame'];d=records[(s,row['source'])]
            a=d['d']['action'];f=d['normal'];tactile=f[t]
            values=dict(state=d['state'][t],rgb=d['rgb'][t],
                        past_action=a[t-3:t].reshape(-1),tactile=tactile,
                        delta1=tactile-f[t-1],delta3=tactile-f[t-3],
                        action=a[t],lateral=a[t,:2],rotation=a[t,3:6],
                        next_dz=np.asarray([d['d']['next_motion'][t,2]]),
                        source=np.asarray(row['source']),frame=np.asarray(t),
                        cluster=np.asarray(row['subcluster']))
            for name,v in values.items():arrays[name].append(v)
        result[s]={name:np.asarray(items,dtype=np.float32 if name not in ('source','frame','cluster') else np.int32)
                   for name,items in arrays.items()}
    return result


def reduce_features(data,smoke):
    # PCA and all scale parameters are fit on selected training demonstrations.
    widths={'rgb':64 if smoke else 96,'tactile':32 if smoke else 64}
    transformed={s:{} for s in SPLITS}
    for name,width in widths.items():
        mu,projection=fit_pca(data['train'][name],width,11000+width)
        for s in SPLITS:transformed[s][name]=(data[s][name]-mu)@projection
        if name=='tactile':
            for delta in ('delta1','delta3'):
                for s in SPLITS:transformed[s][delta]=data[s][delta]@projection
    for name in ('state','past_action','action'):
        for s in SPLITS:transformed[s][name]=data[s][name]
    for name in ('state','past_action','rgb','tactile','delta1','delta3','action'):
        train=transformed['train'][name];mu=train.mean(0);std=np.maximum(train.std(0),1e-5)
        for s in SPLITS:
            transformed[s][name]=np.clip((transformed[s][name]-mu)/std,-8,8)/np.sqrt(train.shape[1])
    return transformed


def design(transformed,mode,outcome):
    fields={
      'T':['tactile'],
      'T_history':['tactile','delta1','delta3'],
      'context':['state','past_action','rgb'],
      'context_T':['state','past_action','rgb','tactile'],
      'context_T_history':['state','past_action','rgb','tactile','delta1','delta3'],
    }[mode].copy()
    if outcome:fields.append('action')
    return {s:np.concatenate([transformed[s][f] for f in fields],axis=1).astype(np.float32)
            for s in SPLITS}


def episode_rmse(actual,pred,source):
    errs=[]
    for ep in np.unique(source):
        mask=source==ep
        errs.append(float(np.mean(np.sum((actual[mask]-pred[mask])**2,axis=1))))
    return float(np.sqrt(np.mean(errs))),np.asarray(errs)


def ridge(x,y,query,alpha):
    mu=y.mean(0);std=np.maximum(y.std(0),1e-5);z=(y-mu)/std
    gram=np.asarray(x.T@x,np.float64);gram.flat[::len(gram)+1]+=alpha
    beta=np.linalg.solve(gram,np.asarray(x.T@z,np.float64))
    return (query@beta*std+mu).astype(np.float32)


def neighbors(x,y,query,k,weighted):
    dist=cdist(query,x,'sqeuclidean')
    idx=np.argsort(dist,axis=1)[:,:k]
    actions=y[idx]
    if not weighted:return actions.mean(1)
    d=np.take_along_axis(dist,idx,axis=1)
    w=1/np.maximum(np.sqrt(d),.05)
    w/=w.sum(1,keepdims=True)
    return np.sum(actions*w[:,:,None],axis=1)


def bootstrap_pair(actual,a,b,source,seed=20261009):
    ids=np.unique(source);means=[]
    for ep in ids:
        mask=source==ep
        means.append([np.mean(np.sum((actual[mask]-a[mask])**2,axis=1)),
                      np.mean(np.sum((actual[mask]-b[mask])**2,axis=1))])
    values=np.asarray(means);rng=np.random.default_rng(seed)
    sample=values[rng.integers(len(ids),size=(1000,len(ids)))].mean(1)
    delta=np.sqrt(sample[:,1])-np.sqrt(sample[:,0])
    return dict(delta_rmse=float(np.sqrt(values[:,1].mean())-np.sqrt(values[:,0].mean())),
                bootstrap95=np.quantile(delta,[.025,.975]).tolist(),
                improved_episodes=int((values[:,1]<values[:,0]).sum()),episodes=int(len(ids)))


def evaluate(data,transformed,smoke):
    modes=('T','T_history','context','context_T','context_T_history')
    tasks=('lateral','rotation','next_dz')
    rows=[];predictions={};contrasts=[]
    for mode in modes:
        for task in tasks:
            x=design(transformed,mode,task=='next_dz')
            y={s:data[s][task] for s in SPLITS};source=data['test']['source']
            candidates=[]
            for alpha in (.1,1.,10.,100.):
                q=ridge(x['train'],y['train'],x['val'],alpha)
                candidates.append((episode_rmse(y['val'],q,data['val']['source'])[0],
                                   'ridge',alpha,None))
            for k in (1,3,5,10):
                if k>len(x['train']):continue
                for weighted in (False,True):
                    q=neighbors(x['train'],y['train'],x['val'],k,weighted)
                    candidates.append((episode_rmse(y['val'],q,data['val']['source'])[0],
                                       'neighbor',k,weighted))
            score,method,param,weighted=min(candidates,key=lambda item:item[0])
            if method=='ridge':pred=ridge(x['train'],y['train'],x['test'],param)
            else:pred=neighbors(x['train'],y['train'],x['test'],param,weighted)
            actual=y['test'];rmse,errors=episode_rmse(actual,pred,source)
            record=dict(mode=mode,task=task,method=method,hyperparameter=param,
                        distance_weighted=weighted,val_episode_rmse=score,
                        test_episode_rmse=rmse,test_frames=len(actual),
                        test_episodes=len(np.unique(source)))
            if task=='lateral':
                length=np.linalg.norm(actual,axis=1);valid=length>2
                record['lateral_direction_agreement']=float(np.mean(np.sum(actual[valid]*pred[valid],axis=1)>0)) if valid.any() else None
                record['lateral_direction_frames']=int(valid.sum())
            rows.append(record);predictions[(mode,task)]=pred
            print('probe',mode,task,method,param,'val',round(score,3),'test',round(rmse,3),flush=True)
    for task in tasks:
        for challenger in ('context_T','context_T_history'):
            contrasts.append(dict(task=task,baseline='context',challenger=challenger,
              **bootstrap_pair(data['test'][task],predictions[('context',task)],
                               predictions[(challenger,task)],data['test']['source'])))
    return rows,contrasts,predictions

def describe(data,k):
    rows=[]
    for split in SPLITS:
        for cluster in range(k):
            picked=[r for r in data[split] if r['subcluster']==cluster]
            if not picked:
                rows.append(dict(split=split,cluster=cluster,frames=0,episodes=0));continue
            act=np.stack([r['action'] for r in picked]);rel=np.stack([r['relative'] for r in picked])
            dz=np.array([r['dz'] for r in picked]);contact=np.array([r['contact'] for r in picked])
            lateral=np.linalg.norm(act[:,:2],axis=1)>2
            direction=act[lateral,:2]/np.maximum(np.linalg.norm(act[lateral,:2],axis=1,keepdims=True),1e-6)
            rows.append(dict(split=split,cluster=cluster,frames=len(picked),
               episodes=len({r['source'] for r in picked}),
               parent_counts={str(parent):sum(r['raw']==parent for r in picked) for parent in PARENT},
               pair_contact_fraction=float(contact.mean()),
               median_relative_xyz_mm=np.median(rel,axis=0).tolist(),
               mean_action_xy_mm=act[:,:2].mean(0).tolist(),
               lateral_direction_concentration=float(np.linalg.norm(direction.mean(0))) if len(direction) else None,
               lateral_actions=int(lateral.sum()),
               down_command_fraction=float(np.mean(act[:,2]<-2)),
               mean_next_plug_dz_mm=float(dz.mean()),
               next_plug_progress_fraction=float(np.mean(dz<-.4)),
               next_plug_stall_fraction=float(np.mean(dz>-.25))))
    return rows


def select_gallery(data,centers,count):
    chosen={}
    for k,center in enumerate(centers):
        candidates=[]
        for split in ('test','val','train'):
            for row in data[split]:
                if row['subcluster']!=k:continue
                diff=(row['force']-center).reshape(2,140,3)
                # Closeness must hold across all six maps, not just 840-D average.
                worst=float(np.sqrt(np.mean(diff**2,axis=1)).max())
                candidates.append((worst,row))
        order=sorted(candidates,key=lambda x:x[0])
        result=[];used=set()
        # Representative frames across distinct demonstrations, preferring
        # test/val examples when reasonably close to the center.
        for split in ('test','val','train'):
            subset=[(distance,row) for distance,row in order if row['split']==split]
            for distance,row in subset:
                key=(row['split'],row['source'])
                if key in used:continue
                used.add(key);result.append(dict(row=row,selection='near center',worst_map_rms=distance))
                if len(result)>=count-1:break
            if len(result)>=count-1:break
        if order and len(result)<count:
            # One explicitly labelled action-divergent example exposes when
            # nearly related force patterns precede different lateral actions.
            reference=np.mean([x['row']['action'][:2] for x in result],axis=0) if result else np.zeros(2)
            alt=max((pair for pair in candidates if (pair[1]['split'],pair[1]['source']) not in used),
                    key=lambda pair:np.linalg.norm(pair[1]['action'][:2]-reference),default=None)
            if alt:result.append(dict(row=alt[1],selection='lateral-action contrast',worst_map_rms=alt[0]))
        chosen[k]=result
    return chosen


def case_composite(front,wrist,left,right,record,scale,centers,marked,t):
    d=record['d'];canvas=np.full((750,1280,3),25,np.uint8)
    canvas[50:370,0:320]=front;canvas[50:370,320:640]=wrist
    canvas[425:745,0:320]=left;canvas[425:745,320:640]=right
    for x,y,title in ((8,32,'FRONT RGB'),(328,32,'WRIST RGB'),
                      (8,409,'LEFT TACFF PREVIEW'),(328,409,'RIGHT TACFF PREVIEW')):
        cv2.putText(canvas,title,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.54,(235,235,235),1,cv2.LINE_AA)
    canvas[35:385,645:1275]=six_force_image(d['force'][t],scale)
    x=d['relative'][t];a=d['action'][t];dz=d['next_motion'][t,2]
    subtype=int(assignment(record["normal"][t:t+1],centers)[0][0]) if record["raw"][t] in PARENT else "outside"
    lines=[f'PARENT K14={record["raw"][t]} | CURRENT subtype={subtype}',
           f'source {record["entry"]["source_episode"]} | marked {marked} | shown {t} ({t-marked:+d})',
           f'USB/socket pair force > .1: {int(d["contact"][t])}',
           f'plug/socket XYZ: {x[0]:+.1f}, {x[1]:+.1f}, {x[2]:+.1f} mm',
           f'NEXT demonstrated action XY: {a[0]:+.1f}, {a[1]:+.1f} mm',
           f'NEXT action Z {a[2]:+.1f} mm | actual plug dz {dz:+.2f} mm',
           'Six raw base-frame force maps; red +, blue -']
    for i,line in enumerate(lines):
        cv2.putText(canvas,line,(649,427+i*43),cv2.FONT_HERSHEY_SIMPLEX,.51,(235,235,235),1,cv2.LINE_AA)
    return canvas


def render_gallery(args,selected,records,centers,scale,subfolder='examples'):
    out=args.output/subfolder;out.mkdir(parents=True,exist_ok=True)
    gallery={}
    for k,examples in selected.items():
        center=np.asarray(centers[k]).reshape(2,140,3)*scale[:,None,:]
        cv2.imwrite(str(out/f'subcluster_{k:02d}_center.png'),six_force_image(center,scale))
        gallery[str(k)]=[]
        for i,item in enumerate(examples):
            row=item['row'];key=(row['split'],row['source']);record=records[key]
            n=len(record['d']['force']);t=row['frame'];lo=max(0,t-8);hi=min(n,t+13)
            idx=list(range(lo,hi));paths=image_paths(args.dataset,row['episode'])
            views=[read_video_frames(paths[name],idx)
                   for name in ('front','wrist','tacff_left_preview','tacff_right_preview')]
            stem=f'k{k:02d}_{i:02d}_{row["split"]}_source{row["source"]:03d}_f{t:03d}'
            video=out/f'{stem}.mp4';poster=out/f'{stem}.jpg'
            if not video.exists():
                encode_frames(video,1280,750,
                    (case_composite(*(v[j] for v in views),record,scale,centers,t,frame)
                     for j,frame in enumerate(idx)),fps=10)
            # encode_frames writes the first video frame as a JPEG at this path;
            # overwrite it with the actual marked subcluster frame.
            cv2.imwrite(str(poster),case_composite(*(v[t-lo] for v in views),record,scale,centers,t,t))
            gallery[str(k)].append(dict(split=row['split'],source=row['source'],episode=row['episode'],
                frame=t,parent=row['raw'],subcluster=k,selection=item['selection'],
                worst_map_rms=item['worst_map_rms'],contact=row['contact'],
                action_xy_mm=row['action'][:2].tolist(),next_dz_mm=row['dz'],
                video=str(video.relative_to(args.output)),poster=str(poster.relative_to(args.output)),
                mark=t-lo))
            print('gallery',k,i+1,'/',len(examples),key,'frame',t,flush=True)
    atomic_json(args.output/(subfolder.replace('/','_')+'_gallery.json'),gallery)
    return gallery


def create_all_members(args,data,records,scale):
    root=args.output/'maps';root.mkdir(exist_ok=True)
    rows=[]
    for split in SPLITS:
        for r in data[split]:
            name=f'{split}_source{r["source"]:03d}_frame{r["frame"]:03d}.jpg'
            path=root/name
            if not path.exists():
                raw=records[(split,r['source'])]['d']['force'][r['frame']]
                cv2.imwrite(str(path),six_force_image(raw,scale),[cv2.IMWRITE_JPEG_QUALITY,80])
            video=image_paths(args.dataset,r['episode'])
            rows.append(dict(split=split,source=r['source'],episode=r['episode'],frame=r['frame'],
                parent=r['raw'],subcluster=r['subcluster'],distance=round(r['distance'],4),
                contact=r['contact'],action_xy=[round(float(x),2) for x in r['action'][:2]],
                next_dz=round(r['dz'],2),map=f'maps/{name}',
                front=str(video['front']),wrist=str(video['wrist'])))
    atomic_json(args.output/'all_members.json',rows)
    rows_json=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全部接触子类帧</title>
<style>body{font:15px/1.5 system-ui;background:#101827;color:#ebf2fd;max-width:1500px;margin:auto;padding:20px}
button,select{padding:8px;background:#284761;color:white;border:1px solid #6380a6;border-radius:5px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}video,img{width:100%}section{background:#1d2a40;padding:14px;margin:12px 0;border-radius:8px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #4c6485;padding:6px;text-align:left}a{color:#a4d4ff}</style>
<h1>5／7／9 的全部逐帧子类结果</h1><p>点击行查看双视图和该帧六张原始力图。聚类只使用该帧力图；动作、物理量用于事后检查。</p>
<p><a href="index.html">返回主报告</a>　筛选子类 <select id="filter"></select>　<span id="count"></span></p>
<section class="grid"><div><img id="map"></div><div><video id="front" controls preload="none"></video><video id="wrist" controls preload="none"></video></div></section>
<section><table><thead><tr><th>子类</th><th>来源</th><th>帧</th><th>原类</th><th>接触</th><th>下步 XY</th><th>下步 dz</th><th>中心距离</th></tr></thead><tbody id="body"></tbody></table></section>
<script>const rows=ROWS;const classes=[...new Set(rows.map(r=>r.subcluster))].sort((a,b)=>a-b);
const filter=document.getElementById('filter'),body=document.getElementById('body');
filter.innerHTML='<option value="all">全部</option>'+classes.map(k=>'<option value="'+k+'">'+k+'</option>').join('');
function seek(video,path,time){video.src=path;video.load();video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(time,Math.max(0,video.duration-.1));},{once:true})}
function show(i){const r=rows[i];document.getElementById('map').src=r.map;seek(document.getElementById('front'),r.front,r.frame/10);seek(document.getElementById('wrist'),r.wrist,r.frame/10)}
function refresh(){const selected=rows.map((r,i)=>[r,i]).filter(([r])=>filter.value==='all'||r.subcluster==filter.value);
document.getElementById('count').textContent=selected.length+' 帧';
body.innerHTML=selected.map(([r,i])=>'<tr onclick="show('+i+')"><td>'+r.subcluster+'</td><td>'+r.split+' '+r.source+'</td><td>'+r.frame+'</td><td>'+r.parent+'</td><td>'+r.contact+'</td><td>'+r.action_xy.join(', ')+'</td><td>'+r.next_dz+'</td><td>'+r.distance+'</td></tr>').join('')}
filter.onchange=refresh;refresh();if(rows.length)show(0);</script></html>'''.replace('ROWS',rows_json)
    (args.output/'all_members.html').write_text(page)


def plots(args,sweep,descriptions,metrics,data,chosen):
    out=args.output/'figures';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.plot([r['k'] for r in sweep],[r['val_silhouette'] for r in sweep],'o-',label='validation silhouette')
    ax.axvline(chosen,color='red',ls='--',label='selected K');ax.set(xlabel='K',ylabel='validation silhouette')
    ax.legend();ax.grid(alpha=.25);fig.savefig(out/'k_sweep.png',dpi=140);plt.close(fig)
    test=[r for r in data['test'] if r['eligible']]
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for k in range(chosen):
        a=np.array([r['action'][:2] for r in test if r['subcluster']==k])
        if len(a):ax.scatter(a[:,0],a[:,1],s=20,alpha=.7,label=f'K{k} n={len(a)}')
    ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
    ax.set(xlabel='next demonstrator action X mm',ylabel='next demonstrator action Y mm',title='Test episodes; down action excluded')
    ax.legend(fontsize=8);fig.savefig(out/'heldout_actions.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for i,task in enumerate(('lateral','rotation','next_dz')):
        subset=[r for r in metrics if r['task']==task]
        axes[i].barh([r['mode'] for r in subset],[r['test_episode_rmse'] for r in subset])
        axes[i].invert_yaxis();axes[i].set(xlabel='per-episode RMSE',title=task)
    fig.savefig(out/'action_probe.png',dpi=140);plt.close(fig)


def report(args,chosen,sweep,reason,descriptions,metrics,contrasts,gallery,data,alternative_galleries,alternative_descriptions):
    selection_labels = {
        'near center': '靠近类中心的候选例（仍需核对六张力图）',
        'lateral-action contrast': '类内动作极端例：未限制触觉相似度，不能用作相似触觉的动作对照',
    }
    stylesheet='''<style>body{font:16px/1.55 system-ui;background:#101827;color:#eaf2fd;max-width:1620px;margin:auto;padding:20px}
a{color:#9fd5ff}section{background:#1c2a40;padding:18px;margin:17px 0;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.card{background:#293b55;padding:12px;border-radius:8px}
video,img{max-width:100%}.card video,.card img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #5d708b;padding:6px;text-align:left}
button{padding:8px 12px;background:#28639b;color:white;border:0;border-radius:5px;margin-bottom:7px}
.notice{background:#493c27}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>'''
    bysplit={s:sum(r['eligible'] for r in data[s]) for s in SPLITS}
    sweep_table=''.join(f'<tr><td>{r["k"]}</td><td>{r["val_silhouette"]:.3f}</td><td>{r["stability_ari"]:.3f}</td><td>{r["min_train"]}</td><td>{r["min_val"]}</td><td>{"yes" if r["occupancy_ok"] else "no"}</td></tr>' for r in sweep)
    metric_table=''.join(f'<tr><td>{r["task"]}</td><td>{r["mode"]}</td><td>{r["method"]}</td><td>{r["val_episode_rmse"]:.3f}</td><td>{r["test_episode_rmse"]:.3f}</td><td>{r.get("lateral_direction_agreement") if r.get("lateral_direction_agreement") is not None else "—"}</td></tr>' for r in metrics)
    contrast_table=''.join(f'<tr><td>{r["task"]}</td><td>{r["challenger"]}</td><td>{r["delta_rmse"]:+.3f}</td><td>{r["bootstrap95"][0]:+.3f}–{r["bootstrap95"][1]:+.3f}</td><td>{r["improved_episodes"]}/{r["episodes"]}</td></tr>' for r in contrasts)
    sections=[]
    for k in range(chosen):
        desc=[r for r in descriptions if r['cluster']==k]
        descrows=''.join(f'<tr><td>{r["split"]}</td><td>{r["frames"]}</td><td>{r["episodes"]}</td><td>{r.get("parent_counts",{})}</td><td>{r.get("pair_contact_fraction",0):.1%}</td><td>{r.get("lateral_direction_concentration") if r.get("lateral_direction_concentration") is not None else "—"}</td><td>{r.get("mean_next_plug_dz_mm",0):+.2f}</td></tr>' for r in desc)
        cards=[]
        for item in gallery[str(k)]:
            cards.append(f'''<div class="card"><h3>{selection_labels.get(item['selection'], item['selection'])} · {item['split']} source {item['source']} frame {item['frame']} · raw {item['parent']}</h3>
<p>接触记录 {item['contact']}；下一动作 XY={item['action_xy_mm'][0]:+.1f},{item['action_xy_mm'][1]:+.1f} mm；下一步实际 dz={item['next_dz_mm']:+.2f} mm；六图最大 RMS={item['worst_map_rms']:.3f}</p>
<button onclick="const v=this.nextElementSibling;v.pause();v.currentTime={item['mark']/10+.03:.2f}">停在所属子类帧</button>
<video controls preload="metadata" poster="{html.escape(item['poster'])}" src="{html.escape(item['video'])}"></video></div>''')
        sections.append(f'''<section id="cluster{k}"><h2>子类 {k}</h2><img src="examples/subcluster_{k:02d}_center.png" alt="六张平均力图">
<p>上图是全类平均图，不是真实帧；视频中的六张力图才是当前实际帧。动作极端例没有触觉相似度门槛，不能据此推断相似触觉存在动作歧义。</p>
<table><tr><th>划分</th><th>帧</th><th>示范</th><th>原 5/7/9 组成</th><th>插座接触</th><th>横向动作方向集中度</th><th>下一步 dz mm</th></tr>{descrows}</table>
<div class="grid">{''.join(cards)}</div></section>''')
    alternative_sections=[]
    for alt_k,alt_gallery in alternative_galleries.items():
        blocks=[]
        for subtype,cases in alt_gallery.items():
            stats=[r for r in alternative_descriptions[alt_k] if r['cluster']==int(subtype)]
            stats_table=''.join(f"<tr><td>{r['split']}</td><td>{r['frames']}</td><td>{r['episodes']}</td><td>{r.get('parent_counts',{})}</td><td>{r.get('pair_contact_fraction',0):.1%}</td><td>{[round(x,1) for x in r.get('median_relative_xyz_mm',[])]}</td><td>{[round(x,1) for x in r.get('mean_action_xy_mm',[])]}</td><td>{r.get('lateral_direction_concentration')}</td><td>{r.get('mean_next_plug_dz_mm',0):+.2f}</td></tr>" for r in stats)
            cards=''.join(f"""<div class='card'><p>{item['split']} source {item['source']} frame {item['frame']} · raw {item['parent']} · {selection_labels.get(item['selection'], item['selection'])}</p>
<p><b>下图是全类平均图，不是这段视频的真实帧。</b>请在视频标记帧核对实际六张力图。</p>
            <img src='alternatives/k{alt_k}/subcluster_{int(subtype):02d}_center.png' alt='subtype mean maps'>
            <button onclick=\"const v=this.nextElementSibling;v.pause();v.currentTime={item['mark']/10+.03:.2f}\">停在所属子类帧</button>
            <video controls preload='metadata' poster='{html.escape(item['poster'])}' src='{html.escape(item['video'])}'></video></div>""" for item in cases)
            blocks.append(f"<h3>K={alt_k} 子类 {subtype}</h3><table><tr><th>划分</th><th>帧</th><th>示范</th><th>原类组成</th><th>接触比例</th><th>相对 XYZ mm</th><th>下步 XY mm</th><th>横向方向集中度</th><th>下步 dz mm</th></tr>{stats_table}</table><div class='grid'>{cards}</div>")
        alternative_sections.append(f"<section id='alternative{alt_k}'><h2>K={alt_k} 细分浏览</h2><p>同一原始力图度量；用于看更细类别是否清楚。这里的 K 未由验证集选为主方案。每类显示中心图和来自独立示范的实际短片。</p>{''.join(blocks)}</section>")
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ContactWorld 接触内部触觉子类</title>{stylesheet}
<h1>5／7／9 内部：触觉子类与下一步动作</h1><p>原始 K14 的 5／7／9 帧合并后，仅用当帧双指六张 base Fx/Fy/Fz 力图聚类。统计和预测使用独立的动作、状态与视觉记录。</p>
<section class="notice"><b>判断边界：</b>子类看起来不同，不能独自证明调整方向可判别；子类重叠也不能证明连续触觉无信息。当前分组按六张力图的总体距离分配，没有逐图相似度或类内最大差异门槛；同类不保证六图都相似。轮廓系数偏好的粗分组尚未满足细模式审查要求。
下方预测表检验触觉加入已知视觉、本体状态、前三步动作后，对未用于拟合的完整示范是否有额外价值。
接触位置的“孔口／侧壁／底部”没有可靠逐帧真值，不自动给这些子类命名。示范动作也不是唯一正确动作。
本数据的 test 示范此前已有人工查看，本次 test 结果为探索性证据。</section>
<section><h2>入口与数据量</h2><p>所有入选帧：{', '.join(f'{s} {len(data[s])}' for s in SPLITS)}。
可用于预测且成功前、有前三步历史与下一步结果：{', '.join(f'{s} {bysplit[s]}' for s in SPLITS)}。
<a href="all_members.html">逐帧查看所有入选样本、双视图与原始力图</a> · <a href="analysis.json">完整统计 JSON</a>。
更细的原始力图分组：<a href="#alternative4">K=4</a>、<a href="#alternative6">K=6</a>；它们作为视觉审查，不用于挑选有利的预测分数。</p></section>
<section><h2>类数选择：K={chosen}</h2><p>{html.escape(reason)}。K 选择只看验证示范的触觉几何与类别稳定性，未看动作或接触标签。
若指标接近、类内力图仍连续过渡，应理解为触觉变化连续，而非物理状态恰好有 K 种。</p>
<img src="figures/k_sweep.png"><table><tr><th>K</th><th>val silhouette</th><th>稳定性 ARI</th><th>最小 train</th><th>最小 val</th><th>覆盖合格</th></tr>{sweep_table}</table></section>
<section><h2>留出示范的动作预测</h2><p>每个配置只从 train 示范拟合，val 示范选择邻居数或岭回归正则，test 示范按整条来源汇总误差。
T=当前六张力图；T_history=当前＋过去 1／3 步力图变化；context=状态＋前三步动作＋双视图；context_T 和 context_T_history 依次加入触觉。
横向动作和旋转是示范的下一动作；实际 dz 是执行该动作后的单步 USB 运动，所有 dz 预测器都输入当前动作。横向方向指标只在真实动作长度大于 2 mm 的帧上计算：预测与示范 XY 向量点积大于 0 即算同向；它不检查幅度，也不是纠偏是否成功的比例。RMSE 按完整来源示范平均每步向量平方误差后开方。
负的 ΔRMSE 表示加入触觉降低误差。相邻帧不按独立样本计算置信区间。</p>
<img src="figures/action_probe.png"><img src="figures/heldout_actions.png">
<table><tr><th>目标</th><th>输入</th><th>最佳轻量方法</th><th>val 整条示范 RMSE</th><th>test 整条示范 RMSE</th><th>横向夹角&lt;90°（真实动作&gt;2 mm；测试 45 帧）</th></tr>{metric_table}</table>
<h3>相比 context 的配对变化</h3><table><tr><th>目标</th><th>加入</th><th>test ΔRMSE</th><th>示范重采样 95% 区间</th><th>改善示范</th></tr>{contrast_table}</table></section>
{''.join(sections)}{''.join(alternative_sections)}<section><h2>解释边界</h2><p>模型未重新训练 π0。预测的是已有示范中的动作及单步结果；
加入触觉不改善此探针，只能说明当前表示、样本与轻量方法下尚未证实额外预测信息。实验不能证明所有有效动作，也不能单独确定碰撞部位。
<a href="all_members.json">全部逐帧索引</a> · <a href="test_predictions.csv">留出预测逐帧记录</a></p></section></html>'''
    (args.output/'index.html').write_text(page)


def write_predictions(args,data,predictions):
    d=data['test'];path=args.output/'test_predictions.csv'
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['source','frame','subcluster','action_x_mm','action_y_mm','next_dz_mm',
                         'context_x','context_y','context_T_x','context_T_y',
                         'context_T_history_x','context_T_history_y',
                         'context_dz','context_T_dz','context_T_history_dz'])
        for i in range(len(d['source'])):
            vals=lambda mode,task: predictions[(mode,task)][i].reshape(-1)
            writer.writerow([int(d['source'][i]),int(d['frame'][i]),int(d['cluster'][i]),
               *d['lateral'][i].tolist(),float(d['next_dz'][i,0]),
               *vals('context','lateral').tolist(),*vals('context_T','lateral').tolist(),
               *vals('context_T_history','lateral').tolist(),
               float(vals('context','next_dz')[0]),float(vals('context_T','next_dz')[0]),
               float(vals('context_T_history','next_dz')[0])])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--atlas',type=Path,default=DEFAULT_ATLAS)
    parser.add_argument('--rgb-cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--vision-weights',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--k-max',type=int,default=10)
    parser.add_argument('--k',type=int)
    parser.add_argument('--examples-per-cluster',type=int,default=5)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.k_max<2 or args.examples_per_cluster<2:parser.error('k-max and examples-per-cluster must be at least 2')
    args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    identity=dict(version=VERSION,smoke=args.smoke,k_max=args.k_max,k=args.k,
        examples_per_cluster=args.examples_per_cluster,
        data_sha256=digest(args.dataset/'spatial/manifest.json'),
        parent_sha256=digest(args.atlas/'frame_assignments.csv'))
    ident=args.output/'identity.json'
    if ident.exists() and json.loads(ident.read_text())!=identity:
        raise ValueError('Output identity mismatch; choose a new --output')
    atomic_json(ident,identity);(args.output/'complete.json').unlink(missing_ok=True)
    records,data,scale,_=load(args)
    centers,sweep,chosen,reason,assigned,models=cluster(args,data)
    np.savez_compressed(args.output/'subcluster_model.npz',centers=centers,scale=scale,parent_classes=np.asarray(PARENT))
    descriptions=describe(data,chosen)
    prediction_data=features_and_targets(records,data)
    print('Prediction counts:',{s:len(prediction_data[s]['source']) for s in SPLITS},flush=True)
    transformed=reduce_features(prediction_data,args.smoke)
    metrics,contrasts,predictions=evaluate(prediction_data,transformed,args.smoke)
    write_predictions(args,prediction_data,predictions)
    cases=select_gallery(data,centers,args.examples_per_cluster)
    gallery=render_gallery(args,cases,records,centers,scale)
    alternative_galleries={}
    alternative_descriptions={}
    alternative_assignments=[]
    if not args.smoke:
        for alt_k in (4,6):
            if alt_k==chosen or alt_k not in models:continue
            alt_centers=models[alt_k]
            alt_data={split:[dict(row) for row in data[split]] for split in SPLITS}
            for split in SPLITS:
                x=np.stack([row['force'] for row in alt_data[split]])
                labels,distance=assignment(x,alt_centers)
                for i,row in enumerate(alt_data[split]):
                    row['subcluster']=int(labels[i]);row['distance']=float(distance[i])
                    alternative_assignments.append(dict(k=alt_k,split=split,source=row['source'],
                        frame=row['frame'],raw=row['raw'],subcluster=int(labels[i])))
            alternative_descriptions[str(alt_k)]=describe(alt_data,alt_k)
            subfolder=f'alternatives/k{alt_k}'
            selected_alt=select_gallery(alt_data,alt_centers,3)
            alternative_galleries[str(alt_k)]=render_gallery(args,selected_alt,records,alt_centers,scale,subfolder)
            np.savez_compressed(args.output/f'alternative_k{alt_k}_model.npz',centers=alt_centers,scale=scale)
    atomic_json(args.output/'alternative_assignments.json',alternative_assignments)
    create_all_members(args,data,records,scale)
    plots(args,sweep,descriptions,metrics,data,chosen)
    atomic_json(args.output/'analysis.json',dict(version=VERSION,selection=PARENT,
        cluster_k=chosen,k_reason=reason,k_sweep=sweep,alternative_ks=sorted(map(int,alternative_galleries)),
        alternative_stats=alternative_descriptions,subcluster_stats=descriptions,
        prediction_metrics=metrics,paired_contrasts=contrasts,
        selected_frames={s:len(data[s]) for s in SPLITS},
        prediction_frames={s:len(prediction_data[s]['source']) for s in SPLITS}))
    report(args,chosen,sweep,reason,descriptions,metrics,contrasts,gallery,data,alternative_galleries,alternative_descriptions)
    atomic_json(args.output/'complete.json',dict(version=VERSION,elapsed_s=round(time.monotonic()-started,1),
       k=chosen,frames=sum(len(data[s]) for s in SPLITS),gallery_cases=sum(map(len,gallery.values())),
       alternative_gallery_cases=sum(len(items) for alt in alternative_galleries.values() for items in alt.values()),
       report=str(args.output/'index.html')))
    print('DONE',args.output/'index.html',flush=True)


if __name__=='__main__':main()
