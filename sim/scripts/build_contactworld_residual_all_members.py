#!/usr/bin/env python3
"""Add a paged browser for every member of a residual tactile cluster report.

Reuses the fitted cluster model; no retraining or new clustering is performed.
Original per-episode videos are copied once, not one video per frame.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

import run_contactworld_tactile_cluster_atlas as raw_atlas
import run_contactworld_tactile_residual_cluster_atlas as residual_atlas
from run_contactworld_sensor_discovery import DEFAULT_DATASET, image_paths


def copy_media(output, dataset, entries):
    count, total = 0, 0
    for entry in entries:
        ep = int(entry['episode_index'])
        destination = output / 'member_media' / f'episode_{ep:06d}'
        destination.mkdir(parents=True, exist_ok=True)
        for stream, source in image_paths(dataset, ep).items():
            dest = destination / f'{stream}.mp4'
            if not dest.exists() or dest.stat().st_size != source.stat().st_size:
                shutil.copy2(source, dest)
            count += 1
            total += dest.stat().st_size
    return count, total


def build(output: Path, dataset: Path, *, copy_videos=True):
    output = Path(output)
    dataset = Path(dataset)
    model_path = output / 'cluster_model.npz'
    if not model_path.is_file():
        raise FileNotFoundError(f'Cluster model missing: {model_path}')
    identity = json.loads((output / 'identity.json').read_text())
    load_args = SimpleNamespace(dataset=dataset, smoke=bool(identity['smoke']))
    entries, episodes, parts, _ = residual_atlas.load_episodes(load_args)
    if [[e['split'], e['source_episode']] for e in entries] != identity['entries']:
        raise ValueError('Dataset episodes no longer match fitted cluster identity')
    with np.load(model_path) as z:
        centers, scale = z['centers'].copy(), z['scale'].copy()
    k = len(centers)
    data = raw_atlas.concatenate(parts)
    features = {
        split: np.clip(data[split]['force'] / scale[None, :, None, :], -3, 3)
                     .reshape(len(data[split]['force']), 840).astype(np.float32)
        for split in residual_atlas.SPLITS
    }
    assigned = raw_atlas.annotate(data, features, centers)
    expected = list(csv.DictReader((output / 'frame_assignments.csv').open()))
    rows = []
    force_blocks = []
    for split in residual_atlas.SPLITS:
        d = data[split]
        a = assigned[split]
        if len(d['force']) != sum(x['split'] == split for x in expected):
            raise ValueError(f'Frame count mismatch: {split}')
        # Display quantization preserves the same signed per-axis scales.
        display = np.clip(d['force'] / scale[None, :, None, :], -1, 1)
        force_blocks.append(np.rint(display * 127).astype(np.int8).reshape(-1).tobytes())
        for i in range(len(a['label'])):
            cluster = int(a['label'][i])
            marker = expected[len(rows)]
            ep, frame, source = (int(d['episode'][i]), int(d['frame'][i]), int(d['source'][i]))
            if (marker['split'], int(marker['episode']), int(marker['frame']), int(marker['source']), int(marker['class'])) != (
                split, ep, frame, source, cluster):
                raise ValueError(f'Frame assignment changed: {split} source {source} frame {frame}')
            diff = features[split][i].reshape(2, 140, 3) - centers[cluster].reshape(2, 140, 3)
            worst_map_rms = float(np.sqrt(np.mean(diff * diff, axis=1)).max())
            rows.append([len(rows), cluster, split, ep, source, frame,
                         round(float(a['distance'][i]), 5),
                         round(worst_map_rms, 5), int(d['contact'][i]),
                         *[round(float(x), 2) for x in d['action'][i, :3]],
                         round(float(d['next_motion'][i, 2]), 3)])
    packed = b''.join(force_blocks)
    if len(packed) != len(rows) * 840:
        raise ValueError('Packed force map dimensions do not match frame metadata')
    media_count, media_bytes = (0, 0)
    if copy_videos:
        media_count, media_bytes = copy_media(output, dataset, entries)
    script = ('window.CW_ROWS=' + json.dumps(rows, separators=(',', ':'), ensure_ascii=False) + ';\n'
              + 'window.CW_FORCE_B64="' + base64.b64encode(packed).decode('ascii') + '";\n')
    (output / 'all_members_data.js').write_text(script)
    sections = []
    for cluster in range(k):
        counts = {split:sum(row[1] == cluster and row[2] == split for row in rows)
                  for split in residual_atlas.SPLITS}
        sections.append(f"""<section id="class-{cluster}" data-class="{cluster}">
<h2>类 {cluster}：{sum(counts.values())} 帧</h2>
<p>训练 {counts['train']}；验证 {counts['val']}；测试 {counts['test']}。
下表收录本类全部帧，按需翻页；点击一行加入下方对比区，最多同时选 4 帧。</p>
<img width="630" height="350" src="examples/cluster_{cluster:02d}_center.png" alt="六张残差类中心图">
<div class="controls">
<label>划分 <select class="split"><option value="all">全部</option><option>train</option><option>val</option><option>test</option></select></label>
<label>接触 <select class="contact"><option value="all">全部</option><option value="1">接触</option><option value="0">未接触</option></select></label>
<label>source <input class="source" type="number" min="0" placeholder="全部"></label>
<label>排序 <select class="sort"><option value="center">靠近类中心</option>
<option value="boundary">远离类中心</option><option value="map">六图最大差异</option>
<option value="episode">示范与时间</option></select></label>
<button class="apply">筛选</button>
</div>
<div class="count"></div><div class="pager"></div>
<div class="table-wrap"><table><thead><tr><th>加入</th><th>划分</th><th>source</th><th>帧</th>
<th>距中心</th><th>六图最大差异</th><th>接触</th><th>动作 X/Y/Z mm</th><th>下一步实际 dz</th></tr></thead>
<tbody class="members"></tbody></table></div>
<h3>选中的片段</h3><p><button class="aligned">从标记帧同步播放</button>
<button class="start">从片段前 8 帧同步播放</button><button class="pause">暂停</button>
<label>共同偏移 <input class="scrub" type="range" min="-8" max="12" value="0"> <span class="offset">0</span> 帧</label></p>
<div class="cards"></div></section>""")
    nav = ' '.join(f'<a href="#class-{i}">类 {i}</a>' for i in range(k))
    page = HTML.replace('__SECTIONS__', '\n'.join(sections)).replace('__NAV__', nav)
    (output / 'all_members.html').write_text(page)
    index_path = output / 'index.html'
    index = index_path.read_text()
    anchor = '<h1>ContactWorld USB 双指触觉残差聚类</h1>'
    link = '<p><a href="all_members.html">浏览每一类的全部逐帧成员，并选择多帧同步比较</a></p>'
    if link not in index:
        if anchor not in index:
            raise ValueError('Cannot find report heading for all-member browser link')
        index_path.write_text(index.replace(anchor, anchor + link, 1))
    manifest = dict(version='cw-residual-all-members-v1', frames=len(rows), classes=k,
                    source_videos=media_count, video_bytes=media_bytes,
                    force_encoding='int8 clip(residual_base / per-finger-axis scale, -1, 1) * 127',
                    row_columns=['id','class','split','episode','source','frame','center_distance',
                                 'worst_map_rms','socket_contact','action_x_mm','action_y_mm',
                                 'action_z_mm','next_plug_dz_mm'])
    (output / 'all_members_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    complete_file = output / 'complete.json'
    if complete_file.exists():
        complete = json.loads(complete_file.read_text())
        complete.update(all_member_frames=len(rows), all_member_source_videos=media_count)
        complete_file.write_text(json.dumps(complete, ensure_ascii=False, indent=2) + '\n')
    print(f'ALL MEMBERS DONE frames={len(rows)} classes={k} videos={media_count} '
          f'report={output/"all_members.html"}', flush=True)
    return manifest


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ContactWorld 残差触觉全部成员</title>
<style>
body{font:15px/1.45 system-ui;background:#101827;color:#eef4ff;max-width:1850px;margin:auto;padding:20px}
a{color:#a9d8ff}nav{position:sticky;top:0;background:#243b59;padding:12px;z-index:10;display:flex;gap:14px}
section{background:#1d2a3e;border-radius:12px;padding:20px;margin:22px 0;scroll-margin-top:70px}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:15px 0}
input,select,button{background:#2e4260;color:white;border:1px solid #7185a0;border-radius:5px;padding:7px}
button{cursor:pointer;background:#2d75af}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #546981;padding:5px;text-align:left}
thead{background:#354a68;position:sticky;top:50px}.table-wrap{max-height:460px;overflow:auto}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.card{background:#273750;border-radius:8px;padding:12px}
.card .streams{display:grid;grid-template-columns:repeat(2,1fr);gap:4px}
.card video{width:100%;background:#000}
.card canvas{width:100%;image-rendering:pixelated;background:#eee}
.small{color:#bbcee4;font-size:13px}.pager button{margin-right:7px}
@media(max-width:900px){.cards{grid-template-columns:1fr}}
</style>
<h1>双指残差触觉的全部类内成员</h1>
<p><a href="index.html">返回精选视频报告</a>。这里包括训练、验证、测试中的每一帧。
点击成员后可看原始双视图、原始触觉视频及该帧的六张残差力图；选多帧可同步对齐。
视频继续播放时，右侧残差图和当前类别也会随帧更新。</p>
<p>代表帧距类中心近，不能显示类边缘。请切换“远离类中心”核对类内差异。
这里的类别来自同一 K-means 模型；接触和动作只用于事后观察。</p>
<nav><a href="#top">顶部</a> __NAV__</nav>
__SECTIONS__
<script src="all_members_data.js"></script>
<script>
const rows=window.CW_ROWS;
const decoded=atob(window.CW_FORCE_B64);
const bytes=new Int8Array(decoded.length);
for(let i=0;i<decoded.length;i++)bytes[i]=decoded.charCodeAt(i);
const byEpFrame=new Map(rows.map(r=>[r[3]+":"+r[5],r]));
const cardState=new Map();
const pageState=new Map();
function section(k){return document.getElementById("class-"+k)}
function selected(k){return cardState.get(k)||[]}
function candidates(k){
 const s=section(k),split=s.querySelector(".split").value,contact=s.querySelector(".contact").value;
 const source=s.querySelector(".source").value,sort=s.querySelector(".sort").value;
 let a=rows.filter(r=>r[1]===k&&(split==="all"||r[2]===split)&&
   (contact==="all"||String(r[8])===contact)&&(!source||r[4]===Number(source)));
 if(sort==="center")a.sort((x,y)=>x[6]-y[6]);
 if(sort==="boundary")a.sort((x,y)=>y[6]-x[6]);
 if(sort==="map")a.sort((x,y)=>y[7]-x[7]);
 if(sort==="episode")a.sort((x,y)=>x[3]-y[3]||x[5]-y[5]);
 return a
}
function list(k,page=0){
 const s=section(k),a=candidates(k),size=60,pages=Math.max(1,Math.ceil(a.length/size));
 page=Math.max(0,Math.min(page,pages-1));pageState.set(k,page);
 s.querySelector(".count").textContent=a.length+" 帧；第 "+(page+1)+"/"+pages+" 页";
 const shown=a.slice(page*size,(page+1)*size);
 s.querySelector(".members").innerHTML=shown.map(r=>
 '<tr><td><button data-id="'+r[0]+'">查看</button></td><td>'+r[2]+'</td><td>'+r[4]+
 '</td><td>'+r[5]+'</td><td>'+r[6].toFixed(3)+'</td><td>'+r[7].toFixed(3)+
 '</td><td>'+r[8]+'</td><td>'+r.slice(9,12).map(v=>v.toFixed(1)).join(" / ")+
 '</td><td>'+r[12].toFixed(2)+'</td></tr>').join("");
 s.querySelectorAll(".members button").forEach(b=>b.onclick=()=>addCard(k,rows[Number(b.dataset.id)]));
 s.querySelector(".pager").innerHTML='<button class="prev">上一页</button><button class="next">下一页</button>';
 s.querySelector(".prev").onclick=()=>list(k,page-1);
 s.querySelector(".next").onclick=()=>list(k,page+1);
}
function paint(canvas,row){
 const ctx=canvas.getContext("2d");const cell=9,gap=8,top=22;
 canvas.width=3*(14*cell+gap);canvas.height=2*(10*cell+top);
 ctx.fillStyle="#1b283c";ctx.fillRect(0,0,canvas.width,canvas.height);
 for(let hand=0;hand<2;hand++)for(let axis=0;axis<3;axis++){
  const ox=axis*(14*cell+gap),oy=hand*(10*cell+top);
  ctx.fillStyle="#f2f5fa";ctx.font="14px sans-serif";
  ctx.fillText((hand?"R ":"L ")+["Fx","Fy","Fz"][axis],ox+3,oy+16);
  for(let taxel=0;taxel<140;taxel++){
   const offset=row[0]*840+hand*420+taxel*3+axis;
   const v=bytes[offset]/127,p=Math.max(0,v),n=Math.max(0,-v);
   const red=Math.round(245*(1-p-n)+225*p+55*n);
   const green=Math.round(245*(1-p-n)+55*p+90*n);
   const blue=Math.round(245*(1-p-n)+50*p+220*n);
   ctx.fillStyle="rgb("+red+","+green+","+blue+")";
   ctx.fillRect(ox+(taxel%14)*cell,oy+top+Math.floor(taxel/14)*cell,cell,cell)
  }
 }
}
function currentRow(ep,time){const frame=Math.max(0,Math.round(time*10));return byEpFrame.get(ep+":"+frame)}
function mediaPath(ep,stream){return "member_media/episode_"+String(ep).padStart(6,"0")+"/"+stream+".mp4"}
function addCard(k,row){
 let a=selected(k);
 if(a.some(x=>x[0]===row[0]))return;
 if(a.length>=4){alert("每类最多同时比较 4 帧；先移除一张。");return}
 a.push(row);cardState.set(k,a);
 const s=section(k),card=document.createElement("article");card.className="card";card.dataset.id=row[0];
 card.innerHTML='<button class="remove">移除</button><b> source '+row[4]+' / 帧 '+row[5]+'</b>'+
 '<p class="small status">载入中</p><div class="streams">'+
 ['front','wrist','tacff_left_preview','tacff_right_preview'].map((name,i)=>
 '<div><div class="small">'+["前视","腕视","左触觉原始视频","右触觉原始视频"][i]+'</div>'+
 '<video '+(i===0?'controls':'muted')+' playsinline preload="metadata" src="'+mediaPath(row[3],name)+'"></video></div>').join("")+
 '</div><canvas></canvas>';
 s.querySelector(".cards").appendChild(card);
 card.querySelector(".remove").onclick=()=>{card.remove();cardState.set(k,selected(k).filter(x=>x[0]!==row[0]))};
 const v=[...card.querySelectorAll("video")],front=v[0],status=card.querySelector(".status"),canvas=card.querySelector("canvas");
 function update(){
  const cur=currentRow(row[3],front.currentTime)||row;
  status.textContent="标记类 "+row[1]+"；当前类 "+cur[1]+"；当前帧 "+cur[5]+
   "；接触 "+cur[8]+"；动作 X/Y/Z "+cur.slice(9,12).join("/")+" mm；下一步实际 dz "+cur[12]+" mm";
  paint(canvas,cur)
 }
 front.addEventListener("timeupdate",update);front.addEventListener("seeked",update);
 front.addEventListener("play",()=>v.slice(1).forEach(x=>{x.currentTime=front.currentTime;x.play().catch(()=>{})}));
 front.addEventListener("pause",()=>v.slice(1).forEach(x=>x.pause()));
 front.addEventListener("seeked",()=>v.slice(1).forEach(x=>x.currentTime=front.currentTime));
 front.addEventListener("loadedmetadata",()=>{front.currentTime=Math.min(row[5]/10,front.duration-.05);update()},{once:true});
 update()
}
function cards(k){return [...section(k).querySelectorAll(".card")]}
function seekCard(card,sec){
 const v=[...card.querySelectorAll("video")];v.forEach(x=>{x.pause();if(Number.isFinite(x.duration))x.currentTime=Math.max(0,Math.min(sec,x.duration-.05))})
}
function align(k,offset,play){
 cards(k).forEach(card=>{
  const r=rows[Number(card.dataset.id)];seekCard(card,(r[5]+offset)/10);
  if(play){const v=card.querySelector("video");v.play().catch(()=>{})}
 })
}
for(const s of document.querySelectorAll("section[data-class]")){
 const k=Number(s.dataset.class);cardState.set(k,[]);pageState.set(k,0);
 s.querySelector(".apply").onclick=()=>list(k,0);
 s.querySelector(".aligned").onclick=()=>align(k,0,true);
 s.querySelector(".start").onclick=()=>align(k,-8,true);
 s.querySelector(".pause").onclick=()=>cards(k).forEach(c=>c.querySelectorAll("video").forEach(v=>v.pause()));
 s.querySelector(".scrub").oninput=e=>{s.querySelector(".offset").textContent=e.target.value;align(k,Number(e.target.value),false)};
 list(k)
}
</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--no-copy-videos', action='store_true')
    args = p.parse_args()
    build(args.output, args.dataset, copy_videos=not args.no_copy_videos)


if __name__ == '__main__':
    main()

