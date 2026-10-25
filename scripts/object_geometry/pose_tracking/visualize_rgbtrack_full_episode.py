#!/usr/bin/env python3
"""Full-episode RGBTrack pose audit: interactive 3D XHand/mesh + aligned RGB frames.

Read-only, except writing a NEW output directory. Operates on already-created
<name>_poses.jsonl from RGBTrack (does not run RGBTrack or infer poses itself).
Tracks all frames; flags missing/invalid/predicted poses and never silently
fills an invalid pose with the previous valid transform. Uses deterministic
samples of calibrated metric GLB meshes, and native XHand FK via the repo's
SpatialPreprocessor.preprocess_tactile() (all in base_link metres).

One self-contained HTML contains a frame slider, replay and synchronized front
RGB with *sampled mesh surface* reprojection. Additional CSV and JSON diagnostics
cover every tracked frame (regardless of --viz-stride).

Example:
  PYTHONPATH=src .venv/bin/python scripts/object_geometry/pose_tracking/visualize_rgbtrack_full_episode.py \
    --repo-root . --dataset data/1018_peg_all --episode 23 \
    --pose-run output/pose_eval/<full-track-run> \
    --output output/hand_scene_qa/e23_full_track_review
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np

MESH_PATHS = {
    'funnel': 'assets/object_geometry/synthware_19_22/funnel_F181975.glb',
    'target_flask': 'assets/object_geometry/synthware_19_22/flask_F301950.glb',
}
PALETTE = ['#f97316','#d946ef','#facc15','#10b981','#f43f5e']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(pts, T):
    return (np.asarray(pts, dtype=np.float64) @ T[:3,:3].T + T[:3,3]).astype(np.float32)


def check_pose(val, what):
    t=np.asarray(val, np.float64)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1], atol=1e-5):
        raise ValueError(f'Bad 4x4 pose: {what}')
    if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=3e-2) or not np.isclose(np.linalg.det(t[:3,:3]),1.0,atol=3e-2):
        raise ValueError(f'Invalid rotation: {what}')
    return t


def get_mesh(path, count, seed):
    import trimesh
    original=trimesh.load(str(path), force='scene', process=False)
    mesh=original.dump(concatenate=True) if isinstance(original,trimesh.Scene) else original
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'Invalid mesh: {path}')
    tris=np.asarray(mesh.triangles,np.float64)
    a=np.linalg.norm(np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0]),axis=-1)*.5
    if not np.isfinite(a).all() or a.sum()<=0:raise ValueError(f'Bad triangle areas: {path}')
    rng=np.random.default_rng(seed)
    tri=rng.choice(len(tris),count,p=a/a.sum())
    u=np.sqrt(rng.random(count));v=rng.random(count)
    p=(1-u)[:,None]*tris[tri,0]+(u*(1-v))[:,None]*tris[tri,1]+(u*v)[:,None]*tris[tri,2]
    return np.asarray(p,np.float32)


def finite_score(value):
    try:
        f=float(value)
        return f if np.isfinite(f) else None
    except (TypeError,ValueError):
        return None


def read_poses(path, label):
    rows={}
    for ln,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        r=json.loads(line); f=int(r['frame_index'])
        if f in rows: raise ValueError(f'{label}: duplicate frame {f}')
        if r.get('object_name') != label:raise ValueError(f'{path}:{ln} object_name mismatch')
        # Matrix only trusted if frame is valid; invalid tracking output can contain fallback/NaNs.
        if r.get('valid'):
            try:
                r['_T'] = check_pose(r['T_base_link_object'],f'{label} frame {f}')
            except (ValueError,KeyError,TypeError) as exc:
                # Preserve corrupt output as an invalid observation for human audit.
                r['valid']=False
                r['status']='corrupt_matrix_in_pose_output'
                r['_validation_error']=str(exc)
        r['confidence']=finite_score(r.get('confidence'))
        rows[f]=r
    if not rows:raise ValueError(f'Empty pose stream: {path}')
    return rows


def validate_pose_run(repo, run, episode, calibration):
    from openpi.object_geometry.calibration import front_calibration_provenance, calibration_provenance_error
    manifest=json.loads((run/'manifest.json').read_text())
    if int(manifest.get('episode',-1))!=episode:raise ValueError('Wrong episode in pose manifest')
    expected=front_calibration_provenance(repo, calibration)
    err=calibration_provenance_error(manifest.get('calibration'),expected)
    if err:raise ValueError(f'Pose-run calibration mismatch: {err}')
    for name,rel in MESH_PATHS.items():
        source=repo/rel;conv=manifest.get('mesh_conversions',{}).get(name,{})
        if not source.is_file():raise FileNotFoundError(source)
        if Path(conv.get('source_glb','MISSING')).resolve()!=source.resolve():
            raise ValueError(f'{name} mesh source != current metric asset; cannot reuse poses')
        if conv.get('source_sha256')!=digest(source):
            raise ValueError(f'{name} mesh hash differs')
        if not np.isclose(float(conv.get('scale_m_per_unit',float('nan'))),1.0):
            raise ValueError(f'{name} GLB scale is not 1.0')
    return manifest


def downsample_taxels(xyz, fid, force, per_finger):
    parts=[]
    for finger in range(5):
        ids=np.flatnonzero(fid==finger)
        if not len(ids):raise ValueError(f'finger {finger} has no taxels')
        # fixed deterministic IDs, not reactive to force
        ss=ids[np.linspace(0,len(ids)-1,min(per_finger,len(ids))).round().astype(int)]
        parts.append(xyz[ss].round(4).tolist())
    return parts


def _rgb_to_jpeg_data(rgb, xyz, T_cam_base, K, quality=68, max_width=640):
    """RGB plus surface point projections. Requires RGB and pose in same calibration."""
    try:
        import cv2
    except ImportError:
        return None
    image=np.asarray(rgb,np.uint8).copy()
    H,W=image.shape[:2]
    for name,color in [('funnel',(40,245,40)),('target_flask',(40,180,245))]:
        pp=xyz.get(name)
        if pp is None:continue
        xyz_cam=transform(pp,T_cam_base)
        ok=np.isfinite(xyz_cam).all(axis=1)&(xyz_cam[:,2]>0.05)
        if not np.any(ok):continue
        projected=(xyz_cam[ok,:2]/xyz_cam[ok,2:3]) @ K[:2,:2].T + K[:2,2]
        for u,v in projected:
            if 0<=u<W and 0<=v<H:
                cv2.circle(image,(int(round(u)),int(round(v))),1,color,-1,cv2.LINE_AA)
    if W>max_width:
        H2=int(round(H*max_width/W)); image=cv2.resize(image,(max_width,H2),interpolation=cv2.INTER_AREA)
    ok,blob=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY),quality])
    return 'data:image/jpeg;base64,'+base64.b64encode(blob.tobytes()).decode('ascii') if ok else None


def rgb_frames_from_folder(directory, selected):
    import cv2
    for f in selected:
        path=directory/f'{f:05d}.jpg'
        if not path.is_file():
            path=directory/f'{f:06d}.jpg'
        im=cv2.imread(str(path)) if path.is_file() else None
        if im is None:raise FileNotFoundError(f'No RGB JPEG for f{f} in {directory}')
        yield f,cv2.cvtColor(im,cv2.COLOR_BGR2RGB)


def html_document(frames, manifest, title):
    from plotly.offline import get_plotlyjs
    # Plotly code is embedded: works without internet access after copying the HTML.
    plotly_js=get_plotlyjs()
    data_json=json.dumps(frames,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    meta=json.dumps(manifest,ensure_ascii=False,indent=2)
    # Scripts only contain our own numerics and fixed labels; escaping < prevents HTML/script breakage.
    data_json=data_json.replace('</','<\\/')
    meta_html=meta.replace('&','&amp;').replace('<','&lt;')
    return '''<!doctype html><html lang="zh"><head><meta charset="utf-8"/>
<title>'''+title+'''</title><style>
body{margin:0;font-family:system-ui,Arial;color:#dce5ef;background:#0d1420}
header{padding:12px 20px;background:#142337;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
strong{font-size:17px} button,select{background:#253a50;color:white;border:1px solid #5b6e80;border-radius:5px;padding:8px;cursor:pointer}
input[type=range]{width:min(62vw,760px)} #status{font-size:13px;line-height:1.6;color:#f8fafc}
main{display:grid;grid-template-columns:minmax(480px,1fr) minmax(320px,0.85fr);gap:12px;margin:8px 12px}
#scene{height:76vh;border-radius:9px;border:1px solid #24364b;background:#fff} #rgb{width:100%;height:auto;max-height:64vh;object-fit:contain;background:#091018;border-radius:9px}
aside{background:#172436;padding:14px;border-radius:8px} #details{font:12px/1.6 monospace;white-space:pre-wrap;overflow-wrap:anywhere}
footer{padding:10px 20px;font-size:12px;color:#adbdd0} code{color:#a8d8ff} @media(max-width:1000px){main{grid-template-columns:1fr}#scene{height:60vh}}
</style><script>'''+plotly_js+'''</script></head><body>
<header><strong>Episode 3D 追踪审计</strong><button id="play">▶ 播放</button>
<button id="back">◀</button><input type="range" id="slider" min="0" max="'''+str(len(frames)-1)+'''" value="0" step="1"/>
<button id="next">▶</button><span id="seek"></span><label>速度<select id="speed"><option value="1200">慢</option><option selected value="450">正常</option><option value="200">快</option></select></label></header>
<main><div id="scene"></div><aside><img id="rgb"/><p id="status"></p><div id="details"></div></aside></main>
<footer>绿色：漏斗 mesh 采样点；蓝色：目标烧瓶 mesh 采样点；五色：XHand 五指触觉位置。<br>
红字/缺失：追踪 invalid；黄字：预测姿态非观测；<strong>所有状态仅表示 RGBTrack 的输出，不等于位姿精度得到验证。</strong>
点云在 base_link 坐标，单位米；3D 图可旋转缩放，切换帧时保持视角。RGB 图的彩色点为 Mesh 真实表面采样点重投影，而不是包围框。</footer>
<script>
const FRAMES='''+data_json+''';
const fingerColors=['#f97316','#d946ef','#facc15','#10b981','#f43f5e'];
const F=FRAMES.length;let current=0,timer=null;
const slider=document.getElementById('slider'), seek=document.getElementById('seek');
function cloud(pos,name,color,size=2.8,opacity=1){if(!pos)pos=[];return {type:'scatter3d',mode:'markers',name,x:pos.map(p=>p[0]),y:pos.map(p=>p[1]),z:pos.map(p=>p[2]),marker:{size,color,opacity},hovertemplate:'x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra>'+name+'</extra>'};}
function traceSet(f){let d=[cloud(f.object_xyz.funnel,'funnel', '#36c86f',3.0,f.pose.funnel.valid?0.9:0.06),cloud(f.object_xyz.target_flask,'target_flask','#27aceb',3.1,f.pose.target_flask.valid?0.9:0.06)];
  for(let i=0;i<5;i++)d.push(cloud(f.hand[i],'finger '+i,fingerColors[i],3));
  const tr=f.trajectory||{};
  for(let n of ['funnel','target_flask'])d.push({type:'scatter3d',mode:'lines',name:n+' movement',x:(tr[n]||[]).map(x=>x[0]),y:(tr[n]||[]).map(x=>x[1]),z:(tr[n]||[]).map(x=>x[2]),line:{color:n==='funnel'?'#15a64c':'#0a73ac',width:3},showlegend:false,hoverinfo:'skip'});
  return d;
}
const layout={paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',margin:{l:0,r:0,t:20,b:0},
  legend:{x:0.02,y:0.98,bgcolor:'rgba(255,255,255,.5)'},uirevision:'same_camera',
  scene:{aspectmode:'data',xaxis:{title:'X (m)'},yaxis:{title:'Y (m)'},zaxis:{title:'Z (m)'},camera:{eye:{x:1.4,y:1.2,z:1.05}}}};
const config={responsive:true,displaylogo:false};
function updateStatus(f){let p=f.pose;let str=[];
  for(let n of ['funnel','target_flask']){let x=p[n];let color=x.valid?(x.status==='observed'?'#73e695':'#ffda69'):'#ff7171';
    let c=x.confidence===null?'N/A':Number(x.confidence).toFixed(3);
    str.push('<span style="color:'+color+'">'+n+': '+x.status+' valid='+x.valid+' score='+c+'</span>');}
  document.getElementById('status').innerHTML=str.join('<br>');
  let extra=['frame = '+f.frame,'3D pose is used only if valid. No pose imputation.'];
  for(let n of ['funnel','target_flask']){extra.push('['+n+'] min distance per finger (cm): '+(f.distance_cm[n]||[]).map(v=>v===null?'—':v.toFixed(1)).join(' / '));}
  document.getElementById('details').textContent=extra.join('\\n');
}
function draw(idx){idx=Math.max(0,Math.min(F-1,idx));current=idx;let f=FRAMES[idx];slider.value=idx;seek.textContent='帧 '+f.frame+' ('+(idx+1)+'/'+F+')';
  document.getElementById('rgb').src=f.jpg||'';updateStatus(f);
  Plotly.react('scene',traceSet(f),layout,config);}
document.getElementById('next').onclick=()=>draw(current+1);
document.getElementById('back').onclick=()=>draw(current-1);
slider.oninput=()=>draw(Number(slider.value));
document.getElementById('play').onclick=()=>{let b=document.getElementById('play');if(timer){clearInterval(timer);timer=null;b.textContent='▶ 播放';return;}
  b.textContent='⏸ 暂停';timer=setInterval(()=>{if(current>=F-1){clearInterval(timer);timer=null;b.textContent='▶ 播放';return;}draw(current+1)},Number(document.getElementById('speed').value));};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')draw(current+1);if(e.key==='ArrowLeft')draw(current-1)});
draw(0);
</script></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--episode',type=int,default=23)
    p.add_argument('--pose-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration-profile',default='configs/ur7e_xhand/front_calibration_candidate.json')
    p.add_argument('--frames-dir',type=Path,help='Optional already-decoded front RGB frames directory (e.g. historic episode frames)')
    p.add_argument('--viz-stride',type=int,default=1,help='Use 1 to show every tracked frame in interactive HTML')
    p.add_argument('--mesh-points',type=int,default=220,help='Sampled points per mesh for interactive HTML; metric mesh source unchanged')
    p.add_argument('--taxels-per-finger',type=int,default=30)
    p.add_argument('--no-rgb',action='store_true',help='Skip synchronized RGB if video unavailable')
    a=p.parse_args()
    if a.viz_stride<1 or a.mesh_points<16 or a.taxels_per_finger<1: p.error('Invalid sample/stride')
    repo=a.repo_root.expanduser().resolve();out=a.output.expanduser().resolve();run=a.pose_run.expanduser().resolve()
    data=a.dataset.expanduser().resolve(); profile=Path(a.calibration_profile)
    if not profile.is_absolute():profile=repo/profile
    if out.exists():raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    if not (run/'manifest.json').is_file():raise FileNotFoundError(run/'manifest.json')
    validate_pose_run(repo,run,a.episode,profile)
    poses={name:read_poses(run/f'{name}_poses.jsonl',name) for name in MESH_PATHS}
    all_ids=sorted(set(poses['funnel'])|set(poses['target_flask']))
    if set(poses['funnel'])!=set(poses['target_flask']):
        raise ValueError('Funnel/target pose streams have different frame IDs')
    if all_ids!=list(range(len(all_ids))):
        raise ValueError(f'Tracking frame coverage is not continuous 0..N-1 ({len(all_ids)} frames).')
    chosen=sorted(set(all_ids[::a.viz_stride]+[all_ids[-1]]))
    from openpi.object_geometry.calibration import make_front_geometry_config
    from openpi.spatial.preprocess import SpatialPreprocessor
    from openpi.spatial_dataset.source import RawSpatialDataset
    cfg,_=make_front_geometry_config(repo,profile)
    prep=SpatialPreprocessor.from_repo_root(repo_root=repo,config=cfg)
    ds=RawSpatialDataset(data,episode=a.episode)
    local={name:get_mesh(repo/rel,a.mesh_points,seed=20261025+i) for i,(name,rel) in enumerate(MESH_PATHS.items())}
    # use same points for distances and projection to avoid hidden geometry differences
    csv_records=[];frame_payload=[];history={'funnel':[],'target_flask':[]}
    try:
        from scipy.spatial import cKDTree
        def minimum_dist(pts,hand):return cKDTree(pts).query(hand,k=1,workers=1)[0]
    except ImportError:
        def minimum_dist(pts,hand):return np.min(np.linalg.norm(hand[:,None,:]-pts[None,:,:],axis=-1),axis=1)
    viz_set=set(chosen)
    for row in ds.iter_rows(depth_roles=()):
        frame=int(row['frame_index'])
        if frame not in poses['funnel']:raise ValueError(f'Dataset frame {frame} missing in RGBTrack poses')
        tactile=prep.preprocess_tactile(np.asarray(row['observation.state']).reshape(-1))
        hand=np.asarray(tactile.xyz_m,np.float32)
        fid=np.asarray(tactile.finger_id,np.int32)
        if hand.shape!=(600,3) or not np.isfinite(hand).all():raise ValueError(f'Invalid FK/taxels at {frame}: {hand.shape}')
        rec={'frame_index':frame};xyz={};dist={}
        for name in MESH_PATHS:
            pose=poses[name][frame];valid=bool(pose.get('valid'))
            rec[f'{name}_valid']=valid;rec[f'{name}_status']=pose.get('status','unknown')
            rec[f'{name}_score']=pose.get('confidence',None)
            if valid:
                xyz[name]=transform(local[name],pose['_T'])
                if not np.isfinite(xyz[name]).all():raise ValueError('Nonfinite mesh xyz')
                ds_min=minimum_dist(xyz[name],hand)
                dist[name]=[float(np.min(ds_min[fid==f])) for f in range(5)]
                for f in range(5):rec[f'{name}_finger{f}_nearest_m']=dist[name][f]
                ctr=transform(np.zeros((1,3),np.float32),pose['_T'])[0]
                for k,c in zip('xyz',ctr):rec[f'{name}_origin_{k}_m']=float(c)
                if pose.get('status')=='observed':history[name].append(ctr.round(4).tolist())
            else:
                xyz[name]=None;dist[name]=[None]*5
                for f in range(5):rec[f'{name}_finger{f}_nearest_m']=''
        csv_records.append(rec)
        if frame in viz_set:
            frame_payload.append({
                'frame':frame,
                'pose':{name:{'valid':bool(poses[name][frame].get('valid')),
                       'status':str(poses[name][frame].get('status','unknown')),
                       'confidence':poses[name][frame].get('confidence',None)} for name in MESH_PATHS},
                'object_xyz':{name:(xyz[name].round(4).tolist() if xyz[name] is not None else []) for name in MESH_PATHS},
                'hand':downsample_taxels(hand,fid,np.asarray(tactile.force_norm),a.taxels_per_finger),
                'distance_cm':{name:[round(100*d,3) if d is not None else None for d in dist[name]] for name in MESH_PATHS},
                'trajectory':{name:history[name][-100:] for name in MESH_PATHS},
            })
        if len(csv_records)%100==0:print(f'[3D] processed {len(csv_records)} frames',flush=True)
    if len(csv_records)!=len(all_ids):raise ValueError(f'Dataset/pose length mismatch: {len(csv_records)} vs {len(all_ids)}')
    jump_report={}
    for name,stream in poses.items():
        last=None;jumps=[]
        for f in all_ids:
            r=stream[f]
            if not r.get('valid'):
                last=None
                continue
            if last is not None:
                fp, prev=last
                delta=float(np.linalg.norm(r['_T'][:3,3]-prev[:3,3]))
                rot=r['_T'][:3,:3]@prev[:3,:3].T
                angle=float(np.rad2deg(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1))))
                # Heuristic QA alarms only. Never silently reject usable poses.
                if delta>0.05*(f-fp) or angle>30*(f-fp):
                    jumps.append({'from_frame':fp,'to_frame':f,'translation_m':round(delta,4),'rotation_deg':round(angle,2)})
            last=(f,r['_T'])
        jump_report[name]=jumps

    # RGB overlay is synchronized by exact frame index, not by sequential order or raw-depth direct indexing.
    if not a.no_rgb:
        sys.path.insert(0,str(repo/'scripts/object_geometry/pose_tracking'))
        from rgbtrack_adapter import load_front_calibration
        cam=load_front_calibration(repo,profile)
        T_cam_base=np.linalg.inv(check_pose(cam.T_base_color,'front camera T_base_color'))
        if a.frames_dir:
            rgbs=rgb_frames_from_folder(a.frames_dir,chosen)
        else:
            rgbs=((int(v.frame_index),np.asarray(v.rgb)) for v in ds.iter_video_frames('front',selected=viz_set))
        dic={frame:i for i,frame in enumerate(chosen)}
        seen=set()
        for f,rgb in rgbs:
            if f not in dic:raise ValueError(f'Unexpected RGB frame {f}')
            item=frame_payload[dic[f]]
            xyz={name:(np.asarray(item['object_xyz'][name],dtype=np.float32) if item['object_xyz'][name] else None) for name in MESH_PATHS}
            item['jpg']=_rgb_to_jpeg_data(rgb,xyz,T_cam_base,cam.K_color)
            seen.add(f)
        if seen!=viz_set:raise ValueError(f'Missing RGB frames {sorted(viz_set-seen)[:10]}')
    else:
        for item in frame_payload:item['jpg']=None
    out.mkdir(parents=True)
    with (out/'tracked_distance_by_finger.csv').open('w',newline='',encoding='utf-8') as f:
        keys=sorted(set().union(*(r.keys() for r in csv_records)))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(csv_records)
    report={'episode':a.episode,'pose_run':str(run),'pose_run_manifest_sha256':digest(run/'manifest.json'),
            'frames_all':len(all_ids),'frames_html':len(chosen),'viz_stride':a.viz_stride,
            'mesh_points_per_object':a.mesh_points,'taxels_per_finger':a.taxels_per_finger,
            'model_scores_are_not_calibrated_pose_accuracy':True,
            'calibration_profile':str(profile),'objects':{},'pose_jump_warnings_heuristic_not_ground_truth':jump_report,
            'warning':'RGBTrack poses may drift when occluded. Invalid poses are NOT imputed. No ground-truth accuracy claimed. Full pose and distances remain in JSONL/CSV.'}
    for name,prs in poses.items():
        statuses=Counter(r.get('status','unknown') for r in prs.values())
        report['objects'][name]={'valid_frames':sum(bool(r.get('valid')) for r in prs.values()),
                                 'invalid_frames':sum(not bool(r.get('valid')) for r in prs.values()),
                                 'status_counts':dict(statuses),
                                 'mesh_sha256':digest(repo/MESH_PATHS[name]),
                                 'first_invalid_frames':[f for f in all_ids if not prs[f].get('valid')][:30]}
    (out/'tracking_quality_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (out/'hand_scene_dynamic_e23.html').write_text(html_document(frame_payload,report,f'Hand-Scene dynamic tracking E{a.episode}'),encoding='utf-8')
    print('[DONE]',len(all_ids),'pose frames;',len(chosen),'visualized frames',flush=True)
    for name,r in report['objects'].items():print(f'  {name}: {r["status_counts"]}; invalid={r["invalid_frames"]}',flush=True)
    print('[HTML]',out/'hand_scene_dynamic_e23.html')
    print('[CSV]',out/'tracked_distance_by_finger.csv')
    print('[REPORT]',out/'tracking_quality_report.json')
    print('[NOTE] Mesh is shown ONLY for valid pose frames. Score is not pose accuracy. Check RGB overlay and 3D jointly.')

if __name__=='__main__':main()
