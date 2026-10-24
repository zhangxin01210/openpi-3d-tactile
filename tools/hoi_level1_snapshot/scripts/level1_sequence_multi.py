#!/usr/bin/env python3
"""Level-1 sequence: clickable flask prompts, video frames, object-labelled RGB-D candidates, temporal 3-D playback.

Run under the OpenPI Python environment (not the SAM2 environment).
Never modifies source data, calibration, or training pipeline.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HOI = Path('/workspace/mnt/sqzhang26/hoi')
REPO = Path('/workspace/mnt/sqzhang26/openpi-3d-tactile')
DATASET = REPO / 'data/1018_peg_all'
ROOT = HOI / 'outputs/level1_sequence_e0'
CLIP = HOI / 'data/level1_sequence_e0'
COLORS = {'funnel':(245,138,39),'target_flask':(156,81,245)} # RGB


def imports(repo:Path):
    sys.path.insert(0,str(repo/'src'))
    # Formal generation must use the versioned helper beside this script, not
    # an older copied helper under the external HOI workspace.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    sys.path.insert(0,str(HOI/'scripts'))
    from openpi.spatial_dataset.source import RawSpatialDataset
    import level1_object_cloud_probe as common
    return RawSpatialDataset, common


def save_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def refuse(path):
    if Path(path).exists():
        raise FileExistsError(f'Will not overwrite: {path}')


def get_frame0(repo,dataset,episode):
    Reader,_ = imports(repo)
    d=Reader(dataset,episode=episode)
    return d.load_video_frames('front',selected={0})[0]


def make_annotation(args):
    import cv2
    rgb=get_frame0(Path(args.repo),Path(args.dataset),args.episode)
    h,w=rgb.shape[:2]
    success,jpg=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92])
    if not success:raise RuntimeError('could not encode first frame')
    img='data:image/jpeg;base64,'+base64.b64encode(jpg).decode('ascii')
    page=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>目标烧瓶 | Frame 0 点标注</title>
<style>body{font:15px system-ui,sans-serif;background:#f7f8fb;color:#202632;margin:18px;}
main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:20px;max-width:1400px;margin:auto}
#canvas{max-width:100%;height:auto;border:1px solid #9da6b5;cursor:crosshair;touch-action:none;background:#151820}
button,select{border:1px solid #9da6b5;border-radius:5px;background:white;padding:9px 12px;cursor:pointer;margin:5px 3px}
button.active{border:2px solid #1354bb;background:#e5edff}.tip{color:#6d3441;background:#fff5f4;padding:11px}
small{color:#546175}#points{max-height:350px;overflow:auto;white-space:pre-wrap;font:12px monospace}
@media(max-width:850px){main{display:block}}</style></head><body><main><section><h2>Episode 0 / Frame 0 — 右侧目标烧瓶</h2>
<p>左侧桌上的锥形瓶是<strong>初始放漏斗的容器</strong>，不要选它；这里要标的是<strong>右侧亮光附近的目标烧瓶</strong>。放大浏览器后可以精确点选。</p>
<canvas id="canvas" width="__W__" height="__H__"></canvas>
<p><small>点击会标记原始 RGB 像素坐标。绿色为正点，红色为负点；导出 JSON 后由 SAM2 生成 Mask，并先检查预览再追踪。</small></p></section>
<aside><h3>标注模式</h3><button id="positive" class="active">✓ 正点（属于目标烧瓶）</button><button id="negative">✕ 负点（不属于目标烧瓶）</button>
<p>建议：在烧瓶的瓶口、瓶颈、瓶身可辨认位置放正点；在白色固定夹具、桌面、漏斗及其他物体放负点。正点不要点到透明瓶身<strong>背后的背景</strong>。</p>
<button id="undo">撤销上一点</button><button id="clear">清空所有点</button>
<p class="tip">负点越多不一定越准。曝光严重的区域若边界不可辨识，后续必须人工核对 SAM2 的预览，不能把白色夹具当瓶身。</p>
<p><strong id="count">正点 0 / 负点 0</strong></p><div id="points"></div>
<button id="download" style="font-weight:bold">下载 flask_prompts_e0.json</button>
<p><small>下载后，把 JSON 上传到服务器 <code>hoi/outputs/level1_sequence_e0/flask_prompts_e0.json</code></small></p>
</aside></main>
<script>const W=__W__,H=__H__,img=new Image();img.src='__IMAGE__';
const c=document.querySelector('#canvas'),ctx=c.getContext('2d');let mode=1,pts=[];
function draw(){ctx.clearRect(0,0,W,H);ctx.drawImage(img,0,0,W,H);ctx.font='bold 13px system-ui';pts.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x,p.y,7,0,2*Math.PI);ctx.fillStyle=p.label===1?'#34de5a':'#f04343';ctx.fill();ctx.lineWidth=2;ctx.strokeStyle='#161616';ctx.stroke();ctx.fillStyle='#fff';ctx.fillText(''+(i+1),p.x+9,p.y-7)});let pos=pts.filter(p=>p.label===1).length;document.querySelector('#count').textContent=`正点 ${pos} / 负点 ${pts.length-pos}`;document.querySelector('#points').textContent=pts.map((p,i)=>`${i+1}. ${p.label===1?'+' : '-'} (${p.x},${p.y})`).join('\n')}
img.onload=draw;
c.addEventListener('click',e=>{let r=c.getBoundingClientRect();let x=Math.max(0,Math.min(W-1,Math.round((e.clientX-r.left)*W/r.width)));let y=Math.max(0,Math.min(H-1,Math.round((e.clientY-r.top)*H/r.height)));pts.push({x,y,label:mode});draw()});
function setmode(v){mode=v;document.querySelector('#positive').classList.toggle('active',v===1);document.querySelector('#negative').classList.toggle('active',v===0)}
document.querySelector('#positive').onclick=()=>setmode(1);document.querySelector('#negative').onclick=()=>setmode(0);
document.querySelector('#undo').onclick=()=>{pts.pop();draw()};document.querySelector('#clear').onclick=()=>{if(confirm('确定清空全部标注点？')){pts=[];draw()}};
document.querySelector('#download').onclick=()=>{let plus=pts.filter(x=>x.label===1);if(!plus.length){alert('至少需要一个正点');return}
const obj={schema_version:1,episode_index:0,frame_index:0,image_width:W,image_height:H,objects:{target_flask:{positive:plus.map(x=>[x.x,x.y]),negative:pts.filter(x=>x.label===0).map(x=>[x.x,x.y])}}};let data=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'});let url=URL.createObjectURL(data);let a=document.createElement('a');a.href=url;a.download='flask_prompts_e0.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></body></html>'''.replace('__W__',str(w)).replace('__H__',str(h)).replace('__IMAGE__',img)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    refuse(path)
    path.write_text(page,encoding='utf8')
    print('[ANNOTATE]',path)
    print('[NEXT] Download HTML, open locally, click green/red points, download JSON, upload back to the server.')


def prepare(args):
    import cv2
    Reader,_=imports(Path(args.repo))
    ds=Reader(args.dataset,episode=args.episode)
    frames=np.asarray(ds.states()['frame_index'],dtype=int).tolist()
    if not frames or frames != list(range(frames[-1]+1)):
        raise ValueError('Expected episode 0 video indices 0..N-1 (contiguous)')
    clip=Path(args.clip)
    refuse(clip)
    (clip/'frames').mkdir(parents=True)
    mapping=[]
    try:
        for v in ds.iter_video_frames('front',selected=set(frames)):
            filename=f'{v.frame_index:05d}.jpg'
            success=cv2.imwrite(str(clip/'frames'/filename),cv2.cvtColor(v.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,96])
            if not success:raise OSError(filename)
            mapping.append({'clip_index':v.frame_index,'episode_frame_index':v.frame_index,'filename':filename})
        if len(mapping)!=len(frames):raise RuntimeError(f'{len(mapping)} frames decoded; expected {len(frames)}')
        save_json(clip/'manifest.json',{'episode':args.episode,'camera_role':'front','mapping':mapping,'num_frames':len(mapping),'resolution':list(v.rgb.shape[1::-1])})
    except Exception:
        import shutil
        shutil.rmtree(clip)
        raise
    print('[PREPARE]',len(mapping),'frames ->',clip/'frames')


def nth_subset(xyz,max_points):
    n=len(xyz)
    if max_points<1:return np.empty((0,3),np.float32)
    return np.asarray(xyz)[np.linspace(0,n-1,min(n,max_points),dtype=int)] if n else np.empty((0,3),np.float32)


def apparent_nearness(a,b):
    """10th percentile candidate-point nearest-neighbor distance (meters).

    Diagnostic only: background-through-glass and hand self-occlusion may yield
    arbitrarily small distances that do NOT imply physical contact.
    """
    if len(a)==0 or len(b)==0:return None
    from scipy.spatial import cKDTree
    # Use deterministic subsampling to cap compute, but do not alter exported clouds.
    query=nth_subset(a,1000)
    target=nth_subset(b,1000)
    dist=cKDTree(target).query(query,k=1,workers=1)[0]
    return round(float(np.quantile(dist,0.10)),4)


def points_js(points,limit):
    s=nth_subset(points,limit)
    # round millimetre-scale for compact HTML; not for saved full NPZ
    return [[round(float(v),4) for v in axis] for axis in s.T] if len(s) else [[],[],[]]


def overlay_jpg(rgb,masks:dict,uv:dict,points:dict):
    import cv2
    im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR).copy()
    for name in ('target_flask','funnel'):
        if name not in masks:continue
        mask=masks[name].astype(np.uint8)
        color=COLORS[name][::-1]
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im,contours,-1,color,2)
        for u,v in nth_subset(uv.get(name, np.zeros((0,2))),250).astype(int):
            if 0<=u<im.shape[1] and 0<=v<im.shape[0]:
                cv2.circle(im,(u,v),1,color,-1)
    ok,data=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,65])
    if not ok:raise RuntimeError('encode overlay failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')


def build_html(items,plotly_src,output):
    # HTML is offline. Numeric data is only for visual inspection, NEVER a geometric ground truth.
    raw=json.dumps(items,ensure_ascii=False,separators=(',',':'))
    html=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Level 1 — 双物体逐帧3D回放</title>
<style>body{font-family:system-ui,Arial,sans-serif;margin:10px 14px;background:#f7f8fb;color:#202632}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px}header h3{margin:4px}
main{display:grid;grid-template-columns:minmax(500px,1.7fr) minmax(320px,1fr);gap:10px}
#plot{width:100%;height:75vh;min-height:520px;background:white}#photo{max-width:100%;max-height:68vh;object-fit:contain;border:1px solid #ccd2de}
button{padding:7px 12px;border-radius:5px;border:1px solid #b0b8c3;cursor:pointer;background:white;margin-right:6px}
label{margin:0 8px}#seek{width:min(72vw,950px)}.warning{color:#874d10}#note{font:13px system-ui;line-height:1.65}
@media(max-width:950px){main{display:block}#plot{min-height:460px;height:55vh}}</style>
<script>__PLOTLY__</script></head><body>
<header><h3>Episode multi · Mask→RGB-D候选点 · 全程回放</h3><button id="prev">◀</button><button id="play">▶ 播放</button><button id="next">▶</button>
<input type="range" id="seek" value="0" min="0" max="0" step="1"/><strong id="frameid"></strong></header>
<header><label><input type="checkbox" id="funnel" checked/>🟠 漏斗</label>
<label><input type="checkbox" id="flask" checked/>🟣 目标烧瓶</label>
<label><input type="checkbox" id="hand" checked/>🟢 XHand</label>
<label><input type="checkbox" id="scene"/>⚪ 当前帧其他深度</label>
<button id="focus">聚焦交互区域</button><button id="all">查看整个场景</button><span class="warning">注意：Mask内深度点可能来自透明物体后方，距离≠接触</span></header>
<main><div id="plot"></div><div><img id="photo"/><div id="note"></div></div></main>
<div id="metrics" style="height:250px;width:100%;background:white"></div><div id="forces" style="height:195px;width:100%;background:white"></div><p class="warning">下方两条曲线只是候选深度点之间的表观接近程度（10%分位数），不是接触标签；玻璃后的背景深度或错误Mask可能使其虚假接近。要结合视频与触觉力判断。</p>
<script>const frames=__DATA__;const max=frames.length-1;let idx=0,timer=null,focused=true;
const s=document.querySelector('#seek');s.max=max;
const checks={funnel:document.querySelector('#funnel'),flask:document.querySelector('#flask'),hand:document.querySelector('#hand'),scene:document.querySelector('#scene')};
function trace(p,name,color,show,size,opacity){let xyz=p||[[],[],[]];return{type:'scatter3d',mode:'markers',name,x:xyz[0],y:xyz[1],z:xyz[2],visible:show,marker:{size,opacity,color},hoverinfo:'skip'}}
function xrange(f){let list=[];for(let key of ['funnel','flask','hand'])if(checks[key].checked){let z=f[key]||[[],[],[]];for(let i=0;i<z[0].length;i+=3)list.push([z[0][i],z[1][i],z[2][i]])}if(!list.length)return null;let range=[];
for(let j=0;j<3;j++){let v=list.map(a=>a[j]).sort((a,b)=>a-b);let lo=v[Math.floor(v.length*.02)],hi=v[Math.floor(v.length*.98)];let mid=(lo+hi)/2,half=Math.max(.09,(hi-lo)/2+.05);range.push([mid-half,mid+half])}return range}
function sceneLayout(f){let lim=focused?xrange(f):null;
return{margin:{l:0,r:0,t:6,b:0},showlegend:false,uirevision:'stay-camera',scene:{aspectmode:'cube',xaxis:{title:'base X (m)',range:lim?lim[0]:[-.2,.92]},yaxis:{title:'base Y (m)',range:lim?lim[1]:[-.15,.80]},zaxis:{title:'base Z (m)',range:lim?lim[2]:[0,.75]}}}}
function draw(){let f=frames[idx];s.value=idx;document.querySelector('#frameid').textContent=`Frame ${f.frame} (${idx+1}/${frames.length})`;
let tr=[trace(f.funnel,'funnel candidates','#f39235',checks.funnel.checked,3.4,.8),trace(f.flask,'target flask candidates','#9651e5',checks.flask.checked,3.4,.8),trace(f.hand,'XHand FK','#16a14c',checks.hand.checked,3.4,.85),trace(f.scene,'other measured depth','#aab2bd',checks.scene.checked,1.3,.18)];
Plotly.react('plot',tr,sceneLayout(f),{responsive:true,displaylogo:false});if(window.Plotly&&document.querySelector('#metrics').data)Plotly.relayout('metrics',{'shapes[0].x0':f.frame,'shapes[0].x1':f.frame});if(window.Plotly&&document.querySelector('#forces').data)Plotly.relayout('forces',{'shapes[0].x0':f.frame,'shapes[0].x1':f.frame});document.querySelector('#photo').src=f.overlay;
let n=f.n,missing=f.missing||[];document.querySelector('#note').innerHTML=`<strong>候选深度点（未验证物体表面）</strong><br>漏斗：${n.funnel}; 目标烧瓶：${n.target_flask}; XHand：${n.hand}<br>未找到Mask：${missing.length?missing.join(', '):'无'}<br><small>白色/灰色点为当前帧其他深度；聚焦只是相机视野裁切，不是改变点云坐标。</small>`}
function metricsInit(){const xs=frames.map(x=>x.frame);
Plotly.newPlot('metrics',[{x:xs,y:frames.map(x=>x.d_hand_funnel_m===null?null:x.d_hand_funnel_m*100),name:'手指—漏斗候选点',mode:'lines',line:{color:'#e59439'}},
{x:xs,y:frames.map(x=>x.d_funnel_flask_m===null?null:x.d_funnel_flask_m*100),name:'漏斗—烧瓶候选点',mode:'lines',line:{color:'#9651e5'}}],
{margin:{l:55,r:20,t:15,b:40},height:250,xaxis:{title:'Episode frame'},yaxis:{title:'表观接近度 (cm)'},
shapes:[{type:'line',x0:frames[0].frame,x1:frames[0].frame,yref:'paper',y0:0,y1:1,line:{color:'#8b95a0',width:2,dash:'dot'}}],legend:{orientation:'h',y:1.18}},{responsive:true,displaylogo:false});
Plotly.newPlot('forces',[{x:xs,y:frames.map(x=>x.tactile_force_p95),name:'XHand 触觉力幅值 P95',mode:'lines',line:{color:'#13a14f'}}],
{margin:{l:55,r:20,t:12,b:40},height:195,xaxis:{title:'Episode frame'},yaxis:{title:'触觉力幅值 P95 (原始单位)'},
shapes:[{type:'line',x0:frames[0].frame,x1:frames[0].frame,yref:'paper',y0:0,y1:1,line:{color:'#8b95a0',width:2,dash:'dot'}}]},{responsive:true,displaylogo:false});}
function set(i){idx=Math.max(0,Math.min(max,i));draw()};document.querySelector('#prev').onclick=()=>set(idx-1);document.querySelector('#next').onclick=()=>set(idx+1);s.oninput=()=>set(+s.value);
function stop(){if(timer!==null){clearInterval(timer);timer=null};document.querySelector('#play').textContent='▶ 播放'}document.querySelector('#play').onclick=()=>{if(timer!==null){stop();return};document.querySelector('#play').textContent='⏸ 暂停';timer=setInterval(()=>{if(idx>=max){stop();return}set(idx+1)},350)};
for(let key in checks)checks[key].onchange=draw;document.querySelector('#focus').onclick=()=>{focused=true;draw()};document.querySelector('#all').onclick=()=>{focused=false;draw()};metricsInit();set(0);
</script></body></html>'''
    html=html.replace('__PLOTLY__',plotly_src).replace('__DATA__',raw.replace('</','<\\/'))
    refuse(output)
    Path(output).write_text(html,encoding='utf8')


def build(args):
    import cv2
    Reader,common=imports(Path(args.repo))
    from openpi.object_geometry.calibration import front_calibration_provenance
    from openpi.object_geometry.calibration import resolve_front_calibration_profile
    from openpi.spatial.preprocess import SpatialPreprocessor
    from plotly.offline import get_plotlyjs
    root=Path(args.output)
    maskroot=root/'masks'
    report_path=maskroot/'masks_report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Full SAM2 tracking missing: {report_path}. Run SAM2 track before building.')
    reviewed_path=root/'tracking_reviewed.json'
    if not reviewed_path.is_file():
        raise RuntimeError(f'Not visually reviewed yet: {reviewed_path}; inspect tracking_contact_sheet.jpg and run review-track')
    reviewed=json.loads(reviewed_path.read_text(encoding='utf8'))
    if int(reviewed.get('episode',-1))!=args.episode or reviewed.get('masks_report_sha256')!=hashlib.sha256(report_path.read_bytes()).hexdigest():
        raise RuntimeError('Tracking report changed since review; must review-track again')
    # Verify all masks still match the tracking report accepted by the reviewer.
    report_meta=json.loads(report_path.read_text(encoding='utf8'))
    if 'masks_sha256' in report_meta:
        mask_sha=hashlib.sha256()
        for name in ('target_flask','funnel'):
            for k in range(int(report_meta['num_frames'])):
                path=maskroot/name/f'{k:05d}.png'
                mask_sha.update(name.encode('utf8'));mask_sha.update(str(k).encode('ascii'))
                mask_sha.update(path.read_bytes())
        if mask_sha.hexdigest()!=report_meta['masks_sha256']:
            raise RuntimeError('Mask bytes changed since tracking; revalidate and review before building')
    if args.stride == 1 and args.front_calibration_profile is None:
        raise ValueError(
            'Formal stride=1 geometry requires --front-calibration-profile; '
            'refusing to create an unprofiled training source.'
        )
    if args.stride == 1 and args.run_name == 'default':
        raise ValueError(
            'Formal stride=1 geometry requires a new --run-name (for example '
            'front_candidate_stride1); refusing to reuse cloud_frames.'
        )
    profile = (
        resolve_front_calibration_profile(Path(args.repo), args.front_calibration_profile)
        if args.front_calibration_profile is not None else None
    )
    ds=Reader(args.dataset,episode=args.episode)
    available=sorted(int(x) for x in ds.states()['frame_index'])
    if not available:raise ValueError(f'episode {args.episode} has no states')
    selected=sorted(set(available[::args.stride])|({0,200,212,240,available[-1]}&set(available)))
    cfg,cam=common._make_calibration(Path(args.repo),profile)
    calibration=front_calibration_provenance(Path(args.repo),profile,config=cfg)
    prep=SpatialPreprocessor.from_repo_root(repo_root=Path(args.repo),config=cfg)
    frames3=root/('cloud_frames' if args.run_name=='default' else f'cloud_frames_{args.run_name}')
    if frames3.exists():raise FileExistsError(f'Will not overwrite {frames3}, choose another --output or back up/delete previous derivatives')
    frames3.mkdir(parents=True)
    rows=ds.iter_rows(selected=selected,depth_roles=('front',))
    image_iter=ds.iter_video_frames('front',selected=selected)
    items=[];report=[]
    try:
        for row,v in zip(rows,image_iter,strict=True):
            fid=int(row['frame_index'])
            if v.frame_index!=fid:raise ValueError(f'Depth/RGB ordinal mismatch {fid} {v.frame_index}')
            rgb=v.rgb;H,W=rgb.shape[:2]
            depth=np.asarray(row['observation.depths.cam_front'])
            base,color,source=common._flatten_depth_to_points(depth,cam)
            z=depth.reshape(-1)[source].astype(np.float32)*cam.depth_scale_m_per_unit
            uv=common._project_mask_filter(base,color,z,np.ones((H,W),bool),cam,cfg)[0]
            masks={};payload={};ux={};selected_union=np.zeros(len(base),bool);counts={};outside_roi={};masked_depth_counts={};missing=[]
            for name in ('funnel','target_flask'):
                p=maskroot/name/f'{fid:05d}.png'
                if not p.is_file():
                    missing.append(name);payload[name]=np.zeros((0,3),np.float32);ux[name]=np.zeros((0,2),dtype=int);counts[name]=0;outside_roi[name]=0;masked_depth_counts[name]=0
                    continue
                m=common._load_image(p,grayscale=True)>127
                if m.shape!=(H,W):raise ValueError(f'{p}: mask {m.shape}, RGB {(H,W)} mismatch')
                masks[name]=m
                px,_,inside,in_range,workspace,valid=common._project_mask_filter(base,color,z,m,cam,cfg)
                masked_depth_counts[name]=int((inside & in_range).sum())
                outside_roi[name]=int((inside & in_range & ~workspace).sum())
                # If overlapping masks, retain both separately for audit; DO NOT silently resolve false ownership.
                payload[name]=base[valid]
                ux[name]=px[valid]
                selected_union |= valid
                counts[name]=int(valid.sum())
            touch=prep.preprocess_tactile(np.asarray(row['observation.state'],dtype=np.float64))
            hand=touch.xyz_m
            forces=np.asarray(touch.force_norm,dtype=np.float32)
            p95=float(np.percentile(forces[np.isfinite(forces)],95)) if np.isfinite(forces).any() else None
            roi=cfg.roi
            inside=((base[:,0]>=roi.x_min_m)&(base[:,0]<=roi.x_max_m)&(base[:,1]>=roi.y_min_m)&(base[:,1]<=roi.y_max_m)&(base[:,2]>=roi.z_min_m)&(base[:,2]<=roi.z_max_m))
            other=base[inside & ~selected_union]
            output_file=frames3/f'{fid:05d}.npz'
            np.savez_compressed(output_file,funnel_xyz_base_m=payload['funnel'],target_flask_xyz_base_m=payload['target_flask'],tactile_xyz_base_m=hand,tactile_force_norm=forces,other_depth_xyz_base_m=nth_subset(other,4000),frame_index=fid)
            counts['hand']=int(len(hand))
            image=overlay_jpg(rgb,masks,ux,payload)
            items.append({'frame':fid,'tactile_force_p95':p95,'d_hand_funnel_m':apparent_nearness(hand,payload['funnel']), 'd_funnel_flask_m':apparent_nearness(payload['funnel'],payload['target_flask']), 'funnel':points_js(payload['funnel'],args.display_points),'flask':points_js(payload['target_flask'],args.display_points),'hand':points_js(hand,600),'scene':points_js(other,args.scene_points),'n':counts,'missing':missing,'overlay':image})
            report.append({'frame':fid,'tactile_force_p95':p95,'funnel_candidates':counts['funnel'],'target_flask_candidates':counts['target_flask'],'missing_mask':missing,'mask_depth_projected':masked_depth_counts,'mask_depth_outside_roi':outside_roi,'frame_npz':str(output_file.relative_to(root))})
            if len(items)%10==0 or fid==selected[-1]:print(f'[BUILD] {len(items)}/{len(selected)} | frame {fid} | funnel {counts["funnel"]} | flask {counts["target_flask"]}',flush=True)
        if not items:raise RuntimeError('No frames processed')
        save_json(root/('sequence_report.json' if args.run_name=='default' else f'sequence_report_{args.run_name}.json'),{
            'schema_version':1,
            'episode':args.episode,
            'frames':report,
            'stride':args.stride,
            'dataset':str(args.dataset),
            'calibration':calibration,
            'warning':'Candidate points are measured depths projected under RGB masks; transparent objects may include background. Neighboring candidate points are not contact ground truth.'
        })
        viewer=root/('sequence_3d.html' if args.run_name=='default' else f'sequence_3d_{args.run_name}.html')
        build_html(items,get_plotlyjs(),viewer)
        print('[DONE]',viewer,'size MB',round(viewer.stat().st_size/1e6,2))
        print('[DATA]',frames3)
    except Exception:
        # Keep partial products visible for debugging, but never claim success.
        print('[INCOMPLETE] Partial files in',frames3,flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare',help='Extract all frames for SAM2 full-video tracking')
    a.add_argument('--repo',type=Path,default=REPO);a.add_argument('--dataset',type=Path,default=DATASET);a.add_argument('--episode',type=int,default=0)
    a.add_argument('--clip',type=Path,default=None);a.set_defaults(fn=prepare)
    a=sub.add_parser('build',help='Read original RGB-D + two object masks; write per-frame point clouds and standalone 3D timeline')
    a.add_argument('--repo',type=Path,default=REPO);a.add_argument('--dataset',type=Path,default=DATASET);a.add_argument('--episode',type=int,default=0)
    a.add_argument('--output',type=Path,default=None);a.add_argument('--stride',type=int,default=3)
    a.add_argument('--front-calibration-profile',type=Path,default=None,
                   help='Required for formal stride=1 output; relative paths are resolved from --repo')
    a.add_argument('--display-points',type=int,default=1000);a.add_argument('--scene-points',type=int,default=1800)
    a.add_argument('--run-name',default='default',help='Different build outputs can coexist, e.g. full to export stride=1')
    a.set_defaults(fn=build)
    args=p.parse_args()
    if args.cmd == "build" and args.output is None:
        args.output = HOI/f'outputs/level1_sequence_e{args.episode}'
    if args.cmd=='prepare' and args.clip is None: args.clip=HOI/f'data/level1_sequence_e{args.episode}'
    if args.cmd=='build' and (args.stride<1 or args.display_points<1 or args.scene_points<0 or not args.run_name.replace('_','').isalnum()):p.error('invalid stride / point limits')
    args.fn(args)

if __name__=='__main__':main()
