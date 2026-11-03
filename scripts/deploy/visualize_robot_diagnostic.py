#!/usr/bin/env python3
"""Visualize all saved policy queries and control states from diagnostic runs.

Images and states come from the recorder. Spatial input and FK overlays are
recomputed with the current checkout and the recorded profile, because the
recorder did not store server-side spatial tensors or calibration hashes.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np
from plotly.offline import get_plotlyjs

from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess
from openpi.spatial.calibration import load_camera_calibrations
from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import make_baseline_config
from openpi.spatial.geometry import apply_diagnostic_overrides

ROOT = Path(__file__).resolve().parents[2]
FINGERS = ('拇指', '食指', '中指', '无名指', '小指')
CHAINS = ((0,1,2,3,4), (0,5,6,7,8), (0,9,10,11), (0,12,13,14), (0,15,16,17))


def b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode('ascii')


def save_jpeg(path, rgb, quality=86):
    bgr = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f'Could not write {path}')


def reprojection(rgb, visual, hand, camera, geometry=None):
    image = rgb.copy()
    xyz = np.asarray(visual['xyz_m'], np.float64)
    hand_xyz = np.asarray(hand, np.float64)
    to_color = np.linalg.inv(camera.T_base_color)
    def project(p):
        q = p @ to_color[:3,:3].T + to_color[:3,3]
        k = camera.color_intrinsics
        z = q[:,2]
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = np.column_stack((k.fx*q[:,0]/z+k.cx,k.fy*q[:,1]/z+k.cy))
        inside = np.isfinite(uv).all(axis=1) & (z > 0.05) & (uv[:,0] >= 0) & (uv[:,0] < image.shape[1]) & (uv[:,1] >= 0) & (uv[:,1] < image.shape[0])
        return uv, inside
    uv, valid = project(xyz)
    overlay = image.copy()
    for x,y in np.rint(uv[valid][::2]).astype(int):
        cv2.circle(overlay,(x,y),1,(0,230,240),-1,cv2.LINE_AA)
    image = cv2.addWeighted(overlay,0.48,image,0.52,0)
    uv, valid = project(hand_xyz)
    for chain in CHAINS:
        for a,b in zip(chain[:-1],chain[1:]):
            if valid[a] and valid[b]:
                cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),(255,40,180),2,cv2.LINE_AA)
    for i in np.flatnonzero(valid):
        x,y=np.rint(uv[i]).astype(int)
        cv2.circle(image,(x,y),4,(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(image,(x,y),4,(200,0,130),1,cv2.LINE_AA)
        cv2.putText(image,str(i),(x+4,y-3),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(image,str(i),(x+4,y-3),cv2.FONT_HERSHEY_SIMPLEX,0.38,(140,0,100),1,cv2.LINE_AA)
    if geometry is not None:
        for role, color in (('manipulated', (255, 40, 40)), ('target', (255, 210, 0))):
            points = geometry[f'{role}_xyz_m'][geometry[f'{role}_mask']]
            points_uv, inside = project(points)
            for x, y in np.rint(points_uv[inside][::2]).astype(int):
                cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
    return image, int(valid.sum())


def depth_image(depth, scale):
    depth_m = depth.astype(np.float32) * scale
    t = np.clip((depth_m - 0.1) / 2.4, 0, 1)
    heat = cv2.applyColorMap(np.rint(t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[depth == 0] = 0
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def control_data(run, manifest):
    blocks = sorted((run/'control').glob('block_*.npz'))
    if not blocks:
        raise ValueError(f'No control blocks in {run}')
    arrays = {}
    names = ('frame_index','observation_elapsed_s','state','command_action','action_sent','fallback')
    for name in names:
        chunks=[]
        for p in blocks:
            with np.load(p,allow_pickle=False) as data:
                chunks.append(np.asarray(data[name]))
        arrays[name]=np.concatenate(chunks)
    idx=arrays['frame_index']
    if len(idx) != len(set(idx.tolist())) or not np.all(np.diff(idx)>0):
        raise ValueError(f'Control frame IDs must be unique and ordered: {run}')
    state=arrays['state']
    if state.shape[1] != 1972:
        raise ValueError(f'Expected 1972 values in {run}')
    tactile=state[:,52:].reshape(-1,5,384)[:,:,24:].reshape(-1,5,120,3)
    force=np.linalg.norm(tactile,axis=-1).max(axis=2)
    joints=np.concatenate((state[:,:6], state[:,28:52:2]),axis=1)
    return arrays, joints, force


def process(run, output):
    manifest=json.loads((run/'manifest.json').read_text())
    meta=manifest['args'].get('server_metadata') or {}
    model=meta.get('training_config') or f"unknown ({meta.get('mode','unknown')})"
    profile=meta.get('front_calibration_profile')
    roles=tuple(meta.get('spatial_camera_roles') or [])
    if roles != ('front',) or not meta.get('spatial_use_visual') or not meta.get('spatial_use_tactile'):
        raise ValueError(f'Unsupported recorded input contract for {run}: {roles}, {meta}')
    if not profile or not (ROOT/profile).is_file():
        raise ValueError(f'Recorded profile unavailable: {profile}')
    is_full = bool(meta.get('object_geometry_required') or meta.get('hand_scene_geometry_service'))
    geometry_by_path = {}
    if is_full:
        matches_path = ROOT/'output/deploy/data_vs_live_20261103/full_actual/matches.json'
        if not matches_path.is_file():
            raise ValueError(f'Full visualization requires byte-verified saved worker geometry: {matches_path}')
        matches = json.loads(matches_path.read_text())
        geometry_by_path = {str(Path(p['input_path']).resolve()): p for p in matches['pairs']}
    processor=XHandSpatialOnlinePreprocess(repo_root=ROOT,camera_roles=roles,front_calibration_profile=profile,
        require_hand_keypoints=True, require_object_geometry=is_full, require_hand_scene=is_full,
        m2_state_from_rgbtrack_v1=is_full)
    config=apply_front_calibration_profile(make_baseline_config().with_camera_roles(*roles),ROOT/profile)
    bundle=load_camera_calibrations(repo_root=ROOT,calibration_config=config.calibration,camera_roles=roles)
    camera=apply_diagnostic_overrides(bundle.cameras,config.diagnostics)['front']
    arrays,joints,forces=control_data(run,manifest)
    inputs=sorted((run/'inference').glob('query_*_input.npz'))
    if not inputs:
        raise ValueError(f'No recorded query inputs in {run}')
    images=output/'images'
    images.mkdir(parents=True,exist_ok=True)
    records=[]
    for number,path in enumerate(inputs):
        with np.load(path,allow_pickle=False) as npz:
            d={k:np.asarray(npz[k]) for k in npz.files}
        query=int(d['query_id']); frame=int(d['frame_index'])
        raw={'observation.state':d['state'],'observation.depths.cam_front':d['depth_cam_front'],
             'frame_index':frame,'timestamp_s':float(d['timestamp_s'])}
        if is_full:
            matched = geometry_by_path.get(str(path.resolve()))
            if matched is None:
                raise ValueError(f'No byte-verified Full worker geometry for {path}')
            with np.load(matched['geometry_path'],allow_pickle=False) as z:
                raw['object_geometry'] = {key: np.asarray(z[key]) for key in z.files if key != 'timestamp_s'}
                raw['timestamp_s'] = float(z['timestamp_s'])
        for role in ('front','left','right'):
            raw[f'observation.images.cam_{role}']=d[f'rgb_cam_{role}']
            save_jpeg(images/f'q{query:04d}_{role}.jpg',d[f'rgb_cam_{role}'])
        save_jpeg(images/f'q{query:04d}_depth.jpg',depth_image(d['depth_cam_front'],camera.depth_scale_m_per_unit))
        spatial=processor(raw)['spatial']
        visual=spatial['visual']; tactile=spatial['tactile']; hand=spatial['hand_keypoints']['xyz_m']
        geometry = spatial.get('object_geometry')
        overlay,visible=reprojection(d['rgb_cam_front'],visual,hand,camera,geometry)
        save_jpeg(images/f'q{query:04d}_projection.jpg',overlay,quality=90)
        xyz=np.asarray(visual['xyz_m'],np.float32)
        colors=np.asarray(visual['rgb'],np.uint8).copy()
        colors[~visual['rgb_valid']]=[150,150,150]
        raw_forces=np.asarray(d['tactile_raw_force_norm'],np.float32)
        control_idx=np.flatnonzero(arrays['frame_index']==frame)
        result_file=run/'inference'/f'query_{query:06d}_output.npz'
        if result_file.exists():
            with np.load(result_file,allow_pickle=False) as out:
                success=bool(out['success']); roundtrip=float(out['client_roundtrip_ms'])
        else:
            success=False;roundtrip=None
        record={'query':query,'frame':frame,'elapsed':float(d['client_elapsed_s']),
            'model_time_s':float(d['timestamp_s']),'success':success,'roundtrip_ms':roundtrip,
            'depth_nonzero_pct':float(np.count_nonzero(d['depth_cam_front'])/d['depth_cam_front'].size*100),
            'rgb_valid':int(visual['rgb_valid'].sum()),'fk_visible':visible,
            'force_max':float(raw_forces.max()),'finger_max':raw_forces.max(axis=1).tolist(),
            'cloud_xyz':b64(xyz,'<f4'),'cloud_rgb':b64(colors,'u1'),
            'tactile_xyz':b64(tactile['xyz_m'],'<f4'),'tactile_force':b64(tactile['force_norm'],'<f4'),
            'hand_xyz':b64(hand,'<f4'),'raw_tactile':b64(raw_forces,'<f4'),
            'state18':joints[control_idx[0]].round(5).tolist() if len(control_idx) else None}
        if geometry is not None:
            for role in ('manipulated', 'target'):
                points=geometry[f'{role}_xyz_m'][geometry[f'{role}_mask']]
                record[f'{role}_xyz']=b64(points,'<f4')
                record[f'{role}_valid_count']=int(len(points))
        records.append(record)
        if number % 10 == 0:
            print(f'{run.name}: {number+1}/{len(inputs)} queries',flush=True)
    # All recorded Front queries on one page, so the full run can be scanned
    # without mistaking the unsaved control frames for available video.
    columns, thumb_width, thumb_height, label_height = 5, 240, 180, 26
    rows = (len(records) + columns - 1) // columns
    sheet = np.full((rows * (thumb_height + label_height), columns * thumb_width, 3), 245, np.uint8)
    for i, record in enumerate(records):
        image_path = images / f'q{record["query"]:04d}_front.jpg'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        y, x = divmod(i, columns)
        y *= thumb_height + label_height
        x *= thumb_width
        sheet[y:y + thumb_height, x:x + thumb_width] = cv2.resize(image, (thumb_width, thumb_height))
        label = f'Q{record["query"]:02d} frame {record["frame"]}  {record["elapsed"]:.1f}s'
        cv2.putText(sheet, label, (x + 4, y + thumb_height + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 40, 50), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output / 'front_contact_sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f'Could not write contact sheet in {output}')
    query_frames={r['frame'] for r in records}
    nonzero=np.max(forces,axis=1)>0
    sampled=np.isin(arrays['frame_index'],list(query_frames))
    summary={'run':run.name,'model':model,'run_index':int(manifest['run_index']),'started_at':manifest['started_at'],
        'status':manifest['status'],'end_reason':manifest.get('end_reason'),'control_frames':len(joints),
        'queries':len(records),'query_first_frame':records[0]['frame'],'query_last_frame':records[-1]['frame'],
        'mode':'full' if is_full else 'st',
        'query_stride':manifest['args'].get('query_frequency'),'dropped_records':manifest.get('dropped_records'),
        'recorded_profile':profile,'force_nonzero_control_frames':int(nonzero.sum()),
        'force_nonzero_query_frames':sum(r['force_max']>0 for r in records),
        'max_tactile_norm':float(forces.max()),'finger_max':forces.max(axis=0).tolist(),
        'notes':['每次模型请求保存 RGB 和深度；控制帧之间的相机图像没有录制。',
                 '三维点云、触觉位置和 FK 投影是用当前本机代码及记录的 profile 离线重算；原记录没有保存服务端输入张量或标定文件哈希。',
                 'Full 物体几何来自与客户端输入逐项匹配的当时服务端 worker 缓存。' if is_full else 'ST 不含物体几何输入。']}
    data={'summary':summary,'frames':records,'control':{
        'frame':arrays['frame_index'].tolist(),'elapsed':np.round(arrays['observation_elapsed_s'],4).tolist(),
        'joints':np.round(joints,5).tolist(),'command':np.round(arrays['command_action'],5).tolist(),
        'force':np.round(forces,5).tolist(),'action_sent':arrays['action_sent'].tolist(),
        'fallback':arrays['fallback'].tolist()},'fingers':FINGERS}
    (output/'data.js').write_text('const DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n')
    (output/'index.html').write_text(PAGE,encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>机器人部署输入检查</title><script src="../plotly.min.js"></script><script src="data.js"></script>
<style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#172535;background:#f3f6fa}body{margin:0}header{padding:20px 3vw;background:#172b3e;color:white}header h1{margin:0 0 6px;font-size:23px}header p{margin:4px 0;color:#d5e6f4}main{max-width:1750px;margin:auto;padding:14px 22px 35px}.note{background:#fff5d9;border-left:4px solid #deaa34;padding:10px 15px;line-height:1.55}.bar{position:sticky;top:0;background:#fff;z-index:4;padding:12px;border:1px solid #d4dfe8;border-radius:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;box-shadow:0 3px 8px #22334412}button,select{padding:7px 12px;border:1px solid #bdd0df;border-radius:6px;background:white;color:#173147;font-size:14px;cursor:pointer}input[type=range]{flex:1;min-width:170px}.tag{font-size:14px;font-weight:600}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}.card{background:white;border:1px solid #dae4ed;border-radius:9px;overflow:hidden}.card h3{font-size:15px;margin:0;padding:9px 12px;background:#edf3f8}.card img{width:100%;display:block}.explain{font-size:13px;color:#526578;padding:7px 12px;margin:0}.wide{grid-column:span 2}.chart{height:520px}.smallchart{height:320px}#status{font-size:14px;color:#475d6d;line-height:1.5}.stats{display:flex;gap:20px;flex-wrap:wrap;padding:12px 2px;font-size:14px}a{color:#087f95}@media(max-width:1000px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:650px){.grid{grid-template-columns:1fr}.wide{grid-column:span 1}main{padding:10px}.chart{height:400px}}
</style></head><body><header><h1 id="title"></h1><p id="subtitle"></p><p><a href="../index.html" style="color:#a8e6ff">← 返回全部记录</a></p></header><main>
<div class="note"><b>数据范围：</b>相机和深度只在每次推理请求时保存；拖动滑块查看所有请求，不代表每个控制帧都有图像。场景点云、触觉位置和手部 FK 是当前代码和记录的 profile 离线重算；Full 物体几何来自与客户端原始输入逐项核对一致的当时服务端缓存。标定文件哈希未录制。</div>
<div class="stats" id="summary"></div><div class="bar"><button id="prev">◀</button><button id="play">▶ 播放</button><button id="next">▶</button><input id="slider" type="range" min="0" step="1" value="0"><span id="label" class="tag"></span></div><div id="status"></div>
<div class="grid"><section class="card"><h3>Front RGB · 记录输入</h3><img id="front"></section><section class="card"><h3>Left RGB · 记录输入</h3><img id="left"></section><section class="card"><h3>Right RGB · 记录输入</h3><img id="right"></section>
<section class="card"><h3>Front 原始深度 · 未对齐</h3><img id="depth"><p class="explain">色标 0.1～2.5 m：紫/蓝近、黄/红远，黑色是零深度；此图与 RGB 像素未直接对齐。</p></section>
<section class="card wide"><h3>Front RGB 投影检查</h3><img id="projection"><p class="explain">青色：重算深度点；紫线和数字：离线 FK 锚点。Full 额外显示原服务端按钮几何（红）和盒子几何（黄）；无效时不会出现对应点。</p></section>
<section class="card wide"><h3>3D 空间输入 · 可旋转、缩放</h3><div id="scene" class="chart"></div><p class="explain">RGB 场景 4096 点 + 600 个触觉位置，触觉点按记录的力值着色；紫色是诊断 FK。坐标为 base_link、单位 m。</p></section>
<section class="card"><h3>模型请求帧的原始触觉强度</h3><div id="heat" class="chart"></div><p class="explain">每行一根手指、每列一个 taxel；显示原始三轴力模长，非物理标定单位。深色表示低值。</p></section>
<section class="card wide"><h3>完整控制状态与采样位置</h3><div style="padding:8px 12px"><select id="metric"></select><span class="explain">蓝线为每个控制帧，橙点为模型请求；关节模式下虚线为发送的目标。</span></div><div id="timeline" class="smallchart"></div></section>
<section class="card"><h3>当前请求详情</h3><div id="detail" style="padding:12px;font-size:14px;line-height:1.75;word-break:break-word"></div></section>
<section class="card wide"><h3>Front 全部已保存请求一览</h3><a href="front_contact_sheet.jpg"><img src="front_contact_sheet.jpg" alt="全部已保存请求的 Front 图像"></a><p class="explain">点击可查看原尺寸；每张图标注模型请求号和控制帧号，未保存的控制帧不在图中。</p></section></div></main>
<script>
const S=DATA.summary,F=DATA.frames,C=DATA.control;let selected=0,timer=null,sceneStarted=false,heatStarted=false;
const by=id=>document.getElementById(id);by('title').textContent=`部署输入检查 · Run ${S.run_index}`;by('subtitle').textContent=`${S.model}　|　${S.started_at}　|　${S.run}`;
by('summary').innerHTML=`<span>控制帧 <b>${S.control_frames}</b></span><span>模型请求 <b>${S.queries}</b></span><span>原始触觉非零：控制帧 <b>${S.force_nonzero_control_frames}</b> / 请求帧 <b>${S.force_nonzero_query_frames}</b></span><span>记录丢失 <b>${S.dropped_records}</b></span>`;
const slider=by('slider');slider.max=F.length-1;
function decode64(s,typ){let b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return typ==='f'?new Float32Array(u.buffer):u}
function xyz(a){let x=[],y=[],z=[];for(let i=0;i<a.length;i+=3){x.push(a[i]);y.push(a[i+1]);z.push(a[i+2])}return{x,y,z}}
function dataScene(d){let v=xyz(decode64(d.cloud_xyz,'f')),rgb=decode64(d.cloud_rgb,'u'),color=[];for(let i=0;i<rgb.length;i+=3)color.push(`rgb(${rgb[i]},${rgb[i+1]},${rgb[i+2]})`);let tactile=xyz(decode64(d.tactile_xyz,'f')),force=Array.from(decode64(d.tactile_force,'f')),hand=xyz(decode64(d.hand_xyz,'f'));
 let traces=[{...v,type:'scatter3d',mode:'markers',name:'场景点',marker:{size:2,color,opacity:.65},hovertemplate:'x=%{x:.3f} y=%{y:.3f} z=%{z:.3f} m<extra>场景</extra>'},{...tactile,type:'scatter3d',mode:'markers',name:'触觉点',marker:{size:4,color:force,colorscale:'Turbo',cmin:0,cmax:Math.max(2,S.max_tactile_norm),colorbar:{title:'力模长'}},hovertemplate:'力=%{marker.color:.2f}<extra>触觉</extra>'},{...hand,type:'scatter3d',mode:'markers+text',name:'手部 FK（诊断）',text:Array.from({length:18},(_,i)=>String(i)),textposition:'top center',marker:{size:4,color:'#d0209a'},textfont:{size:9,color:'#870e60'}}];
 for(const chain of [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11],[0,12,13,14],[0,15,16,17]])traces.push({x:chain.map(i=>hand.x[i]),y:chain.map(i=>hand.y[i]),z:chain.map(i=>hand.z[i]),type:'scatter3d',mode:'lines',showlegend:false,line:{width:5,color:'#d0209a'},hoverinfo:'skip'});for(const [role,label,color] of [['manipulated','按钮几何','#ed344a'],['target','盒子几何','#e0aa00']])if(d[role+'_xyz']!==undefined)traces.push({...xyz(decode64(d[role+'_xyz'],'f')),type:'scatter3d',mode:'markers',name:`${label} ${d[role+'_valid_count']}/512`,marker:{size:3,color}});return traces}
const sceneLayout={margin:{l:0,r:0,b:0,t:0},uirevision:'stay',scene:{aspectmode:'data',xaxis:{title:'base X (m)'},yaxis:{title:'base Y (m)'},zaxis:{title:'base Z (m)'},camera:{eye:{x:1.5,y:-1.5,z:1.1}}},legend:{orientation:'h',x:0,y:1}};
const heatLayout={margin:{l:65,r:20,t:12,b:45},xaxis:{title:'Taxel index (0–119)'},yaxis:{autorange:'reversed'},uirevision:'heat'};
const metrics=by('metric');for(let i=0;i<5;i++)metrics.add(new Option(`触觉最大值 · ${DATA.fingers[i]}`,`f${i}`));for(let i=0;i<6;i++)metrics.add(new Option(`机械臂关节 ${i} · 实测 / 目标`,`j${i}`));for(let i=0;i<12;i++)metrics.add(new Option(`手部关节 ${i} · 实测 / 目标`,`j${i+6}`));metrics.value='f1';
function timeline(){const key=metrics.value,idx=Number(key.slice(1)),isforce=key[0]==='f';let y=C[isforce?'force':'joints'].map(a=>a[idx]);let traces=[{x:C.elapsed,y,type:'scattergl',mode:'lines',name:isforce?'控制帧触觉':'实测',line:{color:'#2876ba',width:2}}];if(isforce){traces.push({x:F.map(x=>x.elapsed),y:F.map(x=>x.finger_max[idx]),mode:'markers',name:'模型请求帧',marker:{size:6,color:'#ef8d32'}})}else{traces.push({x:C.elapsed,y:C.command.map(a=>a[idx]),mode:'lines',name:'命令目标（rad）',line:{color:'#ef8d32',width:1,dash:'dot'}})}let d=F[selected];Plotly.react('timeline',traces,{margin:{l:65,r:20,t:10,b:50},xaxis:{title:'实际经过时间 (s)'},yaxis:{title:isforce?'原始力模长':'关节角 (rad)'},legend:{orientation:'h',x:0,y:1.17},shapes:[{type:'line',x0:d.elapsed,x1:d.elapsed,y0:0,y1:1,yref:'paper',line:{color:'#ce2c77',dash:'dash',width:2}}],uirevision:'timeline'}, {responsive:true,displaylogo:false})}
function show(index){selected=Math.min(F.length-1,Math.max(0,index));slider.value=selected;let d=F[selected],prefix=`images/q${String(d.query).padStart(4,'0')}_`;by('label').textContent=`${selected+1} / ${F.length}　控制帧 ${d.frame}　${d.elapsed.toFixed(2)} s`;for(const k of ['front','left','right','depth','projection'])by(k).src=prefix+k+'.jpg';by('status').textContent=`${d.success?'推理成功':'推理失败或无输出'} · 请求 #${d.query} · Front 非零深度 ${d.depth_nonzero_pct.toFixed(1)}% · 点云有 RGB 的点 ${d.rgb_valid}/4096 · 请求触觉最大值 ${d.force_max.toFixed(3)}`+(S.mode==='full'?` · 按钮 / 盒子有效点 ${d.manipulated_valid_count} / ${d.target_valid_count}`:'');
 let ht=decode64(d.raw_tactile,'f'),z=Array.from({length:5},(_,j)=>Array.from(ht.slice(j*120,(j+1)*120)));Plotly.react('heat',[{type:'heatmap',z,y:DATA.fingers,zmin:0,zmax:Math.max(2,S.max_tactile_norm),colorscale:'Turbo',colorbar:{title:'原始力'}}],heatLayout,{responsive:true,displaylogo:false});
 Plotly.react('scene',dataScene(d),sceneLayout,{responsive:true,displaylogo:false});timeline();by('detail').innerHTML=`请求编号：${d.query}<br>控制帧：${d.frame}<br>实际经过：${d.elapsed.toFixed(3)} s<br>请求时间戳：${d.model_time_s.toFixed(3)} s<br>请求往返：${d.roundtrip_ms===null?'未保存':d.roundtrip_ms.toFixed(1)+' ms'}<br>FK 可投影点：${d.fk_visible}/18<br>请求时各指触觉最大值：<br>${DATA.fingers.map((n,i)=>n+' '+d.finger_max[i].toFixed(3)).join(' / ')}<br><small>显示 3D 输入由本机离线重算；图像为便于浏览的 JPEG，原始数据在 NPZ。</small>`}
by('prev').onclick=()=>show(selected-1);by('next').onclick=()=>show(selected+1);slider.oninput=()=>show(Number(slider.value));metrics.onchange=timeline;by('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;by('play').textContent='▶ 播放'}else{if(selected===F.length-1)show(0);timer=setInterval(()=>{if(selected>=F.length-1){clearInterval(timer);timer=null;by('play').textContent='▶ 播放'}else show(selected+1)},800);by('play').textContent='❚❚ 暂停'}};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')show(selected-1);if(e.key==='ArrowRight')show(selected+1)});show(0);
</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.expanduser().resolve(); output=args.output_dir.expanduser().resolve()
    manifests=sorted(source.rglob('manifest.json'))
    runs=[p.parent for p in manifests if (p.parent/'inference').is_dir() and (p.parent/'control').is_dir()]
    if not runs:
        raise FileNotFoundError(f'No diagnostic runs under {source}')
    output.mkdir(parents=True,exist_ok=True)
    (output/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    summaries=[]
    for i,run in enumerate(runs,1):
        folder=output/f'run_{i:03d}'
        folder.mkdir(exist_ok=True)
        summary=process(run,folder)
        summaries.append(summary)
    cards=''.join(f'<li><a href="run_{i:03d}/index.html">Run {s["run_index"]} · {html.escape(s["model"])}</a> — {s["control_frames"]} 控制帧、{s["queries"]} 模型请求；触觉非零 {s["force_nonzero_control_frames"]} 帧、请求采到 {s["force_nonzero_query_frames"]} 帧</li>' for i,s in enumerate(summaries,1))
    (output/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>1103 部署输入检查</title><style>body{{font:17px system-ui;max-width:960px;margin:50px auto;padding:0 24px;color:#172535;line-height:1.7}}a{{color:#087e94}}.note{{background:#fff5d9;padding:14px}}li{{margin:16px 0}}</style><h1>1103 部署输入检查</h1><p>ST 与 Full 真机记录的全部已保存模型请求，以及全部控制帧的状态、关节目标和触觉时间线。</p><div class="note">相机和深度只在推理请求时录制，控制帧间的图像不存在。场景点云、触觉位置和 FK 投影使用当前代码与记录的 profile 重算；Full 物体几何来自核对一致的当时服务端缓存。</div><ul>{cards}</ul><p>每段页面支持切换请求、播放、旋转 3D 点云，及选择不同手指或关节检查时间线。原始记录保留不变。</p></html>''',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps({'source':str(source),'runs':summaries},ensure_ascii=False,indent=2)+'\n')
    print('VISUALIZATION:',output/'index.html',flush=True)

if __name__=='__main__':
    main()
