#!/usr/bin/env python3
"""Human-gated, two-object SAM2 processing for one LeRobot task.

OpenPI env: prepare, video-prepare.
HOI SAM2 env (cwd = hoi/repos/sam2): seed, firstframes, video-track.
Either env: approve-reference, approve-firstframes, video-review, status.

No position groups, no automatic human approval, no changes to source dataset.
Transparent-object legacy six-position pipeline remains independent.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import cv2
import numpy as np

SCHEMA = "task_sam2_two_objects.v1"
OBJECT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,47}$")
IMG_QUALITY = 96


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data: Any) -> None:
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(type(v).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=convert) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def img(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(path)
    return a


def read_mask(path: Path, hw: tuple[int, int], *, nonempty: bool = True) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None or a.shape != hw:
        raise ValueError(f"Bad mask/shape: {path}, got {None if a is None else a.shape}, want {hw}")
    a = (a > 127).astype(np.uint8)
    if nonempty and int(a.sum()) < 8:
        raise ValueError(f"Empty mask: {path}")
    return a


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise IOError(path)


def manifest(root: Path) -> dict:
    m = load_json(root / "manifest.json")
    if m.get("schema") != SCHEMA or len(m.get("objects", [])) != 2:
        raise ValueError(f"Incompatible task SAM2 manifest: {root}")
    if not Path(m["dataset"]).is_dir():
        raise FileNotFoundError(f"Original dataset missing: {m['dataset']}")
    return m


def ids_arg(spec: str | None, m: dict) -> list[int]:
    if spec is None or spec == "sampled":
        result = list(map(int, m["sampled_ids"]))
    elif spec == "all":
        result = list(map(int, m["episode_ids"]))
    else:
        result = []
        for t in spec.split(","):
            t = t.strip()
            if "-" in t:
                a, b = map(int, t.split("-", 1))
                if b < a:
                    raise ValueError(f"Invalid range {t}")
                result.extend(range(a, b + 1))
            else:
                result.append(int(t))
    result = sorted(set(result))
    if not result or any(i not in m["episode_ids"] for i in result):
        raise ValueError("Selected episodes missing or not in task dataset")
    return result


def sheet(cells: list[tuple[str, np.ndarray]], output: Path, *, cols=3, cw=360, ch=270, page_size=30) -> list[str]:
    if not cells:
        raise ValueError("Cannot build an empty sheet")
    outputs=[]
    for start in range(0, len(cells), page_size):
        part = cells[start:start+page_size]
        rows = (len(part)+cols-1)//cols
        canvas = np.full((rows*ch, cols*cw, 3), 232, dtype=np.uint8)
        for j, (label, frame) in enumerate(part):
            h,w = frame.shape[:2]
            f = min(cw/w, ch/h)
            rw,rh = max(1,int(round(w*f))),max(1,int(round(h*f)))
            v = cv2.resize(frame,(rw,rh),interpolation=cv2.INTER_AREA)
            x = (j%cols)*cw+(cw-rw)//2
            y = (j//cols)*ch+(ch-rh)//2
            canvas[y:y+rh,x:x+rw]=v
            bx=(j%cols)*cw; by=(j//cols)*ch
            cv2.rectangle(canvas,(bx,by),(bx+min(cw-1,len(label)*12+20),by+29),(0,0,0),-1)
            cv2.putText(canvas,label[:29],(bx+6,by+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        path = output if len(cells) <= page_size else output.with_name(f"{output.stem}_{start//page_size:02d}{output.suffix}")
        write_png(path,canvas)
        outputs.append(str(path))
    return outputs


def overlay(frame: np.ndarray, masks: dict[str,np.ndarray], label="") -> np.ndarray:
    a = frame.copy()
    colors=[(30,215,40),(225,60,200)]
    for (_,m),color in zip(masks.items(), colors):
        selected = m.astype(bool)
        layer = a.copy()
        layer[selected] = color
        a = cv2.addWeighted(a,.72,layer,.28,0)
        contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(a,contours,-1,color,2)
    if label:
        cv2.putText(a,label,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.66,(0,0,0),3)
        cv2.putText(a,label,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.66,(255,255,255),1)
    return a


def prompts_html(jpg: Path, names: list[str], reference_ep: int, out: Path) -> None:
    im=img(jpg); h,w=im.shape[:2]
    uri="data:image/jpeg;base64,"+base64.b64encode(jpg.read_bytes()).decode("ascii")
    # Local HTML only. Multiple positive/negative clicks supported per object.
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Task SAM2 reference</title>
<style>body{font:15px sans-serif;background:#f8fafc;margin:16px} main{display:flex;gap:18px;flex-wrap:wrap}
canvas{max-width:90vw;height:auto;cursor:crosshair;border:1px solid #777}button{margin:4px;padding:8px}
aside{min-width:290px;max-width:380px}pre{max-height:280px;overflow:auto}</style>
<h3>参考 Episode __EP__：先给两个物体各点一个正点；必要时可增加负点</h3>
<main><canvas id="cv" width="__W__" height="__H__"></canvas><aside><div id="objbuttons"></div>
<button id="pos">正点 (+)</button><button id="neg">负点 (-)</button>
<button id="undo">撤销</button><button id="clear">清空当前物体</button>
<p id="state"></p><button id="download">下载 prompts.json</button>
<p>下载后上传到本任务 sam2/reference/prompts.json。运行 seed 并检查两个 Mask；点错了可以重标。</p><pre id="points"></pre></aside></main>
<script>
const objs=__OBJECTS__, ep=__EP__, W=__W__, H=__H__, data={}, cv=document.getElementById('cv'),ctx=cv.getContext('2d'), pic=new Image();
let selected=objs[0], lab=1;objs.forEach(o=>data[o]={positive:[],negative:[]});pic.src='__IMG__';
function render(){ctx.clearRect(0,0,W,H);ctx.drawImage(pic,0,0,W,H);
objs.forEach((o,i)=>['positive','negative'].forEach(k=>data[o][k].forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.fillStyle=k==='positive'?'#00ff33':'#ff3030';ctx.fill();ctx.strokeStyle=i===0?'#000000':'#ffffff';ctx.lineWidth=2;ctx.stroke()})));
document.getElementById('state').textContent='当前: '+selected+' / '+(lab?'正点':'负点');document.getElementById('points').textContent=JSON.stringify(data,null,2);}
pic.onload=render;objs.forEach(o=>{let b=document.createElement('button');b.textContent=o;b.onclick=()=>{selected=o;render()};document.getElementById('objbuttons').append(b)});
cv.onclick=e=>{let r=cv.getBoundingClientRect();let x=Math.max(0,Math.min(W-1,Math.round((e.clientX-r.left)*W/r.width))),y=Math.max(0,Math.min(H-1,Math.round((e.clientY-r.top)*H/r.height)));data[selected][lab?'positive':'negative'].push([x,y]);render()};
document.getElementById('pos').onclick=()=>{lab=1;render()};document.getElementById('neg').onclick=()=>{lab=0;render()};
document.getElementById('undo').onclick=()=>{let q=data[selected][lab?'positive':'negative'];q.pop();render()};
document.getElementById('clear').onclick=()=>{data[selected]={positive:[],negative:[]};render()};
document.getElementById('download').onclick=()=>{if(objs.some(o=>data[o].positive.length===0)){alert('两个物体都需要至少一个正点');return}
let doc={schema:'__SCHEMA__',reference_episode:ep,image_width:W,image_height:H,objects:data};
let u=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='prompts.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script>'''
    for k,v in {"__EP__":str(reference_ep),"__W__":str(w),"__H__":str(h),"__OBJECTS__":json.dumps(names),"__IMG__":uri,"__SCHEMA__":SCHEMA}.items():
        html=html.replace(k,v)
    out.write_text(html,encoding="utf-8")


def extract_frame0(dataset: Path, camera: str, ep: int, path: Path) -> None:
    # Imported only by OpenPI stage (not installed in HOI SAM2 environment).
    from openpi.spatial_dataset.source import RawSpatialDataset
    import av
    d=RawSpatialDataset(dataset, episode=ep)
    if camera not in d.camera_roles:
        raise ValueError(f"e{ep}: camera {camera} unavailable: {d.camera_roles}")
    with av.open(str(d.video_path(camera))) as c:
        frame=next(c.decode(video=0),None)
        if frame is None:
            raise ValueError(f"Empty e{ep} video")
        image=frame.to_ndarray(format="bgr24")
    write_png(path,image) if path.suffix=='.png' else _write_jpg(path,image)


def _write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,IMG_QUALITY]):
        raise IOError(path)


def prepare(args) -> None:
    from openpi.spatial_dataset.source import RawSpatialDataset
    dataset=args.dataset.expanduser().resolve(); root=args.out.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if root==dataset or dataset in root.parents:
        raise ValueError("Output must remain outside original LeRobot dataset")
    names=[s.strip() for s in args.objects.split(",")]
    if len(names)!=2 or len(set(names))!=2 or any(not OBJECT_RE.fullmatch(s) for s in names):
        raise ValueError("--objects must contain exactly two distinct identifier names, e.g. button,button_base")
    ids=list(RawSpatialDataset.discover_episode_ids(dataset))
    if not ids:
        raise ValueError("Dataset has no episodes")
    if args.reference_episode is not None and args.reference_episode not in ids:
        raise ValueError("Reference episode absent")
    ref=ids[0] if args.reference_episode is None else args.reference_episode
    if args.sample_count<2 or len(ids)<2:
        raise ValueError("Need at least two episodes and sample-count >=2")
    n=min(args.sample_count,len(ids))
    samples=sorted(set(ids[int(round(j))] for j in np.linspace(0,len(ids)-1,n))|{ref})
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Nonempty task output; refusing prepare: {root}")
    refroot=root/'reference'; source=root/'firstframes'/'source'
    refroot.mkdir(parents=True); source.mkdir(parents=True)
    # Prepare all frame-0 images once. Sampling only affects initial SAM2 inference.
    # No full videos or masks are read/written at this stage.
    for i,ep in enumerate(ids):
        extract_frame0(dataset,args.camera,ep,source/f'e{ep:05d}.jpg')
        if (i+1)%20==0 or i+1==len(ids):
            print(f"[FRAME0] {i+1}/{len(ids)}",flush=True)
    ref_image=source/f'e{ref:05d}.jpg'
    shutil.copyfile(ref_image,refroot/'reference_frame.jpg')
    prompts_html(refroot/'reference_frame.jpg',names,ref,refroot/'reference_click.html')
    cells=[(f'e{ep}',img(source/f'e{ep:05d}.jpg')) for ep in samples]
    sheet(cells,root/'firstframes'/'sampled_first_frames.jpg')
    m={"schema":SCHEMA,"dataset":str(dataset),"dataset_id":dataset.name,"camera":args.camera,
       "objects":names,"reference_episode":ref,"episode_ids":ids,"sampled_ids":samples,
       "reference_frame_sha256":sha(refroot/'reference_frame.jpg'),
       "note":"task-specific; no position groups; human review required at every gate"}
    save_json(root/'manifest.json',m)
    print('[PREPARED]',root)
    print('[NEXT] Open',refroot/'reference_click.html','; upload downloaded prompts.json into reference/ then run seed')


def validate_prompts(root: Path, m: dict) -> tuple[dict, np.ndarray, Path]:
    path=root/'reference'/'prompts.json'; data=load_json(path)
    if data.get('schema')!=SCHEMA or data.get('reference_episode')!=m['reference_episode']:
        raise ValueError("Reference prompt task/episode mismatch")
    frame=img(root/'reference'/'reference_frame.jpg'); h,w=frame.shape[:2]
    if data.get('image_width')!=w or data.get('image_height')!=h or set(data.get('objects',{}))!=set(m['objects']):
        raise ValueError("Reference prompt dimensions or object names mismatch")
    for name in m['objects']:
        points=data['objects'][name]
        if not isinstance(points,dict):
            raise ValueError(name)
        for key in ('positive','negative'):
            coords=np.asarray(points.get(key,[]),dtype=np.float32).reshape(-1,2)
            if key=='positive' and not len(coords):
                raise ValueError(f"{name} requires positive point")
            if not np.isfinite(coords).all() or np.any(coords<0) or np.any(coords[:,0]>=w) or np.any(coords[:,1]>=h):
                raise ValueError(f"{name} has invalid {key} points")
    if sha(root/'reference'/'reference_frame.jpg')!=m['reference_frame_sha256']:
        raise ValueError('Reference image changed since prepare')
    return data,frame,path


def image_predictor(checkpoint: Path):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for SAM2')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(checkpoint),device='cuda',apply_postprocessing=True)
    return SAM2ImagePredictor(model)


def sam_predict(predictor, frame: np.ndarray, pos: np.ndarray, neg: np.ndarray, box=None) -> tuple[np.ndarray,float]:
    import torch
    points=np.concatenate([pos,neg]).astype(np.float32)
    labels=np.asarray([1]*len(pos)+[0]*len(neg),dtype=np.int32)
    predictor.set_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    with torch.inference_mode():
        masks,scores,_=predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=True)
    order=np.argsort(np.asarray(scores))[::-1]
    xy=np.rint(pos).astype(int)
    eligible=[int(i) for i in order if all(masks[int(i)][y,x] for x,y in xy)]
    if not eligible:
        raise ValueError("No SAM2 mask contains all positive points; re-annotate")
    idx=eligible[0]
    mask=(np.asarray(masks[idx])>0).astype(np.uint8)
    return pick_component(mask,(int(xy[0,0]),int(xy[0,1]))),float(scores[idx])


def pick_component(mask: np.ndarray, xy: tuple[int,int]) -> np.ndarray:
    x,y=xy
    if not mask[y,x]:
        raise ValueError('Mask excludes positive click')
    count,labeled,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    i=int(labeled[y,x])
    if i==0 or stats[i,cv2.CC_STAT_AREA]<8:
        raise ValueError('Mask component too small')
    return (labeled==i).astype(np.uint8)


def seed(args) -> None:
    root=args.out.resolve(); m=manifest(root)
    data,frame,prompts=validate_prompts(root,m)
    approval=root/'reference'/'review.json'
    if approval.exists():
        raise FileExistsError('Reference already approved. Use a NEW task output version for changed reference prompts.')
    predictor=image_predictor(args.checkpoint.resolve())
    masks={}; metrics={}
    for name in m['objects']:
        p=data['objects'][name]
        pos=np.asarray(p['positive'],dtype=np.float32).reshape(-1,2)
        neg=np.asarray(p.get('negative',[]),dtype=np.float32).reshape(-1,2)
        mask,score=sam_predict(predictor,frame,pos,neg)
        masks[name]=mask
        metrics[name]={"sam2_score":score,"mask_pixels":int(mask.sum())}
        write_png(root/'reference'/'templates'/f'{name}.png',mask*255)
    a,b=(masks[name] for name in m['objects'])
    overlap=int(np.count_nonzero((a>0)&(b>0)))
    vis=overlay(frame,masks,'reference / REVIEW BOTH OBJECTS')
    write_png(root/'reference'/'preview.jpg',vis)
    report={"schema":SCHEMA,"status":"candidate_unreviewed","prompts_sha256":sha(prompts),
            "frame_sha256":m['reference_frame_sha256'],
            "template_hashes":{name:sha(root/'reference'/'templates'/f'{name}.png') for name in m['objects']},
            "objects":metrics,"overlap_pixels":overlap}
    save_json(root/'reference'/'candidate_report.json',report)
    print('[REF CANDIDATES]',root/'reference'/'preview.jpg')
    print('[NEXT] Inspect both masks; correct prompts if needed and re-run seed; else approve-reference.')


def approved_reference(root: Path, m: dict) -> dict:
    ref=root/'reference'; review=load_json(ref/'review.json')
    report=load_json(ref/'candidate_report.json')
    if review.get('status')!='human_accepted' or review.get('candidate_report_sha256')!=sha(ref/'candidate_report.json'):
        raise ValueError('Reference approval missing or stale')
    if report.get('prompts_sha256')!=sha(ref/'prompts.json') or report.get('frame_sha256')!=m['reference_frame_sha256']:
        raise ValueError('Reference prompts/frame changed after approval')
    if set(report.get('objects', {})) != set(m['objects']):
        raise ValueError('Object names changed after reference approval')
    for name in m['objects']:
        p=ref/'templates'/f'{name}.png'
        if report['template_hashes'].get(name)!=sha(p):
            raise ValueError(f'{name}: reference mask changed after approval')
    return report


def approve_reference(args) -> None:
    root=args.out.resolve();m=manifest(root)
    report=load_json(root/'reference'/'candidate_report.json')
    data,frame,prompts=validate_prompts(root,m)
    if report['prompts_sha256']!=sha(prompts):
        raise ValueError('Prompts changed since seed')
    if (root/'reference'/'review.json').exists():
        raise FileExistsError('Reference already reviewed')
    for name in m['objects']:
        p=root/'reference'/'templates'/f'{name}.png'
        read_mask(p,frame.shape[:2]);
        if sha(p)!=report['template_hashes'].get(name):
            raise ValueError(f'{name} reference candidate changed since seed')
    save_json(root/'reference'/'review.json',{
        'schema':SCHEMA,'status':'human_accepted','reviewer':args.reviewer,
        'candidate_report_sha256':sha(root/'reference'/'candidate_report.json'),
        'note':'Manual inspection of reference/preview.jpg explicitly acknowledged.'})
    print('[APPROVED REFERENCE]')


def tight(ref: np.ndarray, mask: np.ndarray, padding=3):
    ys,xs=np.where(mask>0)
    if len(xs)<8:
        raise ValueError('Reference mask empty')
    x0=max(0,int(xs.min())-padding);x1=min(ref.shape[1],int(xs.max())+padding+1)
    y0=max(0,int(ys.min())-padding);y1=min(ref.shape[0],int(ys.max())+padding+1)
    return ref[y0:y1,x0:x1].copy(),(mask[y0:y1,x0:x1]*255).astype(np.uint8),(x0,y0)


def locate(frame: np.ndarray, template: np.ndarray, mask: np.ndarray) -> dict:
    # Reuse tested color-mask template matching; no per-task color thresholds/ROI.
    if template.shape[:2]!=mask.shape or template.dtype!=np.uint8 or mask.dtype!=np.uint8:
        raise ValueError('Invalid template')
    if template.shape[0]>frame.shape[0] or template.shape[1]>frame.shape[1]:
        raise ValueError('Template exceeds image')
    a=cv2.matchTemplate(frame.astype(np.float32),template.astype(np.float32),cv2.TM_SQDIFF,
                        mask=(mask>0).astype(np.float32))
    a=np.maximum(a,0)/max(1,int(np.count_nonzero(mask))*3)
    if not np.isfinite(a).all():
        raise ValueError('Matching returned nonfinite scores')
    val,_,(x,y),_=cv2.minMaxLoc(a)
    ex=max(5,int(round(max(template.shape[:2])*.85)))
    b=a.copy();b[max(0,y-ex):min(b.shape[0],y+ex+1),max(0,x-ex):min(b.shape[1],x+ex+1)]=np.inf
    runner_up=float(np.min(b)) if np.isfinite(b).any() else None
    return {'x':int(x),'y':int(y),'width':int(template.shape[1]),'height':int(template.shape[0]),
            'rgb_rmse':float(np.sqrt(val)),
            'second_vs_best_ratio':None if runner_up is None else float(np.sqrt(runner_up)/max(np.sqrt(val),1e-6))}


def predict_object(pred, frame:np.ndarray, reference_img:np.ndarray, reference_mask:np.ndarray):
    tpl,msk,_=tight(reference_img,reference_mask)
    match=locate(frame,tpl,msk)
    yy,xx=np.nonzero(msk>0)
    ox,oy=match['x'],match['y']
    point=np.asarray([[int(ox+np.rint(np.median(xx))),int(oy+np.rint(np.median(yy))) ]],dtype=np.float32)
    border=max(3,int(.15*max(msk.shape)))
    box=np.asarray([max(0,ox-border),max(0,oy-border),min(frame.shape[1]-1,ox+tpl.shape[1]+border),
                    min(frame.shape[0]-1,oy+tpl.shape[0]+border)],dtype=np.float32)
    seg,score=sam_predict(pred,frame,point,np.zeros((0,2),np.float32),box=box)
    return seg,score,match


def frame_record(root:Path,m:dict,ep:int)->dict:
    folder=root/'firstframes'/f'e{ep:05d}'
    record=load_json(folder/'report.json')
    if record.get('schema')!=SCHEMA or record.get('episode')!=ep:
        raise ValueError(f'e{ep} first-frame report mismatch')
    approved_reference(root,m)
    if record.get('reference_report_sha256')!=sha(root/'reference'/'candidate_report.json') or record.get('source_frame_sha256')!=sha(root/'firstframes'/'source'/f'e{ep:05d}.jpg'):
        raise ValueError(f'e{ep} first-frame sources changed; use a new output version')
    hw=img(root/'firstframes'/'source'/f'e{ep:05d}.jpg').shape[:2]
    for name in m['objects']:
        p=folder/f'{name}.png'
        read_mask(p,hw)
        if record['mask_hashes'].get(name)!=sha(p):
            raise ValueError(f'e{ep} {name} first-frame mask differs from report')
    return record


def firstframes(args) -> None:
    root=args.out.resolve();m=manifest(root); approved_reference(root,m)
    ids=ids_arg(args.episodes,m)
    if (root/'firstframes'/f'review_{args.name}.json').exists():
        raise FileExistsError(f'{args.name} firstframes already approved; refusing regeneration')
    pred=image_predictor(args.checkpoint.resolve())
    ref=img(root/'reference'/'reference_frame.jpg')
    refs={name:read_mask(root/'reference'/'templates'/f'{name}.png',ref.shape[:2]) for name in m['objects']}
    rows=[];cells=[]
    for i,ep in enumerate(ids):
        framepath=root/'firstframes'/'source'/f'e{ep:05d}.jpg'
        frame=img(framepath)
        if frame.shape!=ref.shape:
            raise ValueError(f'e{ep} camera dimensions changed')
        dest=root/'firstframes'/f'e{ep:05d}'
        if (dest/'report.json').is_file():
            existing=frame_record(root,m,ep)
            print(f'[SKIP] e{ep} verified existing firstframe',flush=True)
            row=existing
        else:
            if dest.exists():
                raise FileExistsError(f'e{ep} partial firstframes, inspect: {dest}')
            dest.mkdir()
            out_masks={};scores={};matching={};errors=[]
            try:
                for name in m['objects']:
                    if ep==m['reference_episode']:
                        seg=refs[name].copy();score=None;match={'source':'reference_exact'}
                    else:
                        seg,score,match=predict_object(pred,frame,ref,refs[name])
                    out_masks[name]=seg;scores[name]=None if score is None else float(score)
                    matching[name]=match
                    write_png(dest/f'{name}.png',seg*255)
                a,b=[out_masks[name] for name in m['objects']]
                intersection=int(np.count_nonzero((a>0)&(b>0)))
                overlap_fraction=intersection/max(1,min(int(a.sum()),int(b.sum())))
                if overlap_fraction>0.15:
                    errors.append(f'Two object masks overlap {overlap_fraction:.2%} of smaller object')
                for name,match in matching.items():
                    if match.get('second_vs_best_ratio') is not None and match['second_vs_best_ratio']<1.05:
                        errors.append(f'{name}: ambiguous template location')
                row={'schema':SCHEMA,'episode':ep,'source_frame_sha256':sha(framepath),
                     'reference_report_sha256':sha(root/'reference'/'candidate_report.json'),
                     'mask_hashes':{name:sha(dest/f'{name}.png') for name in m['objects']},
                     'areas':{name:int(out_masks[name].sum()) for name in m['objects']},
                     'sam2_scores':scores,'matching':matching,'warnings':errors}
                save_json(dest/'report.json',row)
                print(f'[FIRSTFRAME] e{ep}: '+', '.join(f'{k}={v}' for k,v in row['areas'].items()),flush=True)
            except Exception:
                # Partial directory deliberately retained; inspect rather than overwrite.
                print(f'[FAILED] e{ep}: partial output retained in {dest}',flush=True)
                raise
        masks={name:read_mask(dest/f'{name}.png',frame.shape[:2]) for name in m['objects']}
        cells.append((f'e{ep}',overlay(frame,masks)))
        rows.append(row)
    sheets=sheet(cells,root/'firstframes'/f'contact_sheet_{args.name}.jpg')
    report={'schema':SCHEMA,'status':'candidate_unreviewed','episodes':[int(ep) for ep in ids],
            'objects':m['objects'],'rows':rows,'sheets':sheets,
            'sheet_hashes':{str(Path(sheet_path).resolve()):sha(Path(sheet_path)) for sheet_path in sheets},
            'warning':'No first-frame labels are accepted until explicit approve-firstframes; check object identity and overlaps.'}
    target=root/'firstframes'/f'report_{args.name}.json'
    if (root/'firstframes'/f'review_{args.name}.json').exists():
        raise FileExistsError(f'{args.name} firstframes already approved; no regeneration allowed')
    save_json(target,report)
    print('[NEXT] Inspect',sheets,'then approve-firstframes --name',args.name)


def approve_firstframes(args) -> None:
    root=args.out.resolve();m=manifest(root);approved_reference(root,m)
    report_path=root/'firstframes'/f'report_{args.name}.json'
    review_path=root/'firstframes'/f'review_{args.name}.json'
    report=load_json(report_path)
    if report.get('schema')!=SCHEMA or report.get('status')!='candidate_unreviewed':
        raise ValueError('Bad firstframe report')
    if review_path.exists():
        raise FileExistsError(review_path)
    rows=report['rows'];ids=report['episodes']
    for sheet_path, expected in report.get('sheet_hashes',{}).items():
        if sha(Path(sheet_path))!=expected:
            raise ValueError(f'Firstframe review sheet changed: {sheet_path}')
    if len(ids)!=len(rows) or len(set(ids))!=len(ids):
        raise ValueError('Firstframe report incomplete')
    for ep,row in zip(ids,rows,strict=True):
        verified=frame_record(root,m,ep)
        if row!=verified:
            raise ValueError(f'e{ep}: first-frame report changed after sheet')
        if row['warnings'] and not args.accept_warnings:
            raise ValueError(f'e{ep}: warnings need explicit --accept-warnings after visual review: {row["warnings"]}')
    save_json(review_path,{'schema':SCHEMA,'status':'human_accepted','name':args.name,
                           'episodes':ids,'reviewer':args.reviewer,
                           'report_sha256':sha(report_path),
                           'note':'User inspected all listed firstframe sheets.'})
    print(f'[APPROVED] {args.name}: {len(ids)} episodes')


def approved_firstframe_set(root:Path,m:dict)->set[int]:
    accepted=set()
    for path in sorted((root/'firstframes').glob('review_*.json')):
        r=load_json(path)
        report_path=root/'firstframes'/f'report_{r["name"]}.json'
        if r.get('schema')!=SCHEMA or r.get('status')!='human_accepted' or r.get('report_sha256')!=sha(report_path):
            raise ValueError(f'Stale firstframe review: {path}')
        report=load_json(report_path)
        for ep in r['episodes']:
            if ep not in report['episodes']:
                raise ValueError('Review reports unknown episode')
            frame_record(root,m,int(ep))
            accepted.add(int(ep))
    return accepted


def video_prepare(args) -> None:
    from openpi.spatial_dataset.source import RawSpatialDataset
    root=args.out.resolve();m=manifest(root)
    ids=ids_arg(args.episodes,m)
    approved=approved_firstframe_set(root,m)
    if not set(ids).issubset(approved):
        raise ValueError(f'Firstframes require review before video: {sorted(set(ids)-approved)}')
    cache=root/'cache'/'video_frames';cache.mkdir(parents=True,exist_ok=True)
    for ep in ids:
        edir=root/'episodes'/f'e{ep:05d}'
        meta=edir/'prepare.json';frames=cache/f'e{ep:05d}'
        if meta.is_file():
            d=load_json(meta)
            if d.get('schema')!=SCHEMA or d['source_mask_hashes']!={name:sha(root/'firstframes'/f'e{ep:05d}'/f'{name}.png') for name in m['objects']}:
                raise ValueError(f'e{ep} staged sources changed; refuse reuse')
            if len(list(frames.glob('*.jpg')))!=d['num_frames']:
                raise ValueError(f'e{ep} frame cache incomplete')
            print(f'[SKIP PREPARED] e{ep}',flush=True)
            continue
        if edir.exists() or frames.exists():
            raise FileExistsError(f'e{ep} partial video-prep output; inspect {edir} and {frames}')
        frames.mkdir();edir.mkdir(parents=True)
        source=root/'firstframes'/'source'/f'e{ep:05d}.jpg'
        frame0=img(source)
        ds=RawSpatialDataset(Path(m['dataset']),episode=ep)
        n=0
        for frame in ds.iter_video_frames(m['camera']):
            if frame.frame_index!=n:
                raise ValueError(f'e{ep}: video index not contiguous')
            bgr=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)
            if bgr.shape!=frame0.shape:
                raise ValueError(f'e{ep}: frame shape differs')
            if n==0:
                # Existing first-frame mask was reviewed on source cache JPG;
                # preserve those exact bytes for alignment.
                mae=float(np.abs(bgr.astype(np.int16)-frame0.astype(np.int16)).mean())
                if mae>7:
                    raise ValueError(f'e{ep}: frame 0 mismatch between cached JPEG and video (MAE {mae:.2f})')
                shutil.copyfile(source,frames/'00000.jpg')
            else:
                _write_jpg(frames/f'{n:05d}.jpg',bgr)
            n+=1
        if n<2:
            raise ValueError(f'e{ep}: insufficient frames')
        save_json(meta,{'schema':SCHEMA,'episode':ep,'num_frames':n,'source_frame0_sha256':sha(source),
                        'source_mask_hashes':{name:sha(root/'firstframes'/f'e{ep:05d}'/f'{name}.png') for name in m['objects']}})
        print(f'[VIDEO PREPARED] e{ep}: {n} frames',flush=True)
    print('[NEXT] run video-track in the HOI SAM2 environment; video frames are cache only')


def _iou(a:np.ndarray,b:np.ndarray)->float:
    union=int(np.count_nonzero((a>0)|(b>0)))
    return float(np.count_nonzero((a>0)&(b>0))/union) if union else 1.0


def video_track(args) -> None:
    import gc
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    root=args.out.resolve();m=manifest(root)
    ids=ids_arg(args.episodes,m)
    if not set(ids).issubset(approved_firstframe_set(root,m)):
        raise ValueError('Video tracking requires approved firstframes')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('SAM2 CUDA not available')
    pred=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(args.checkpoint.resolve()),device='cuda',apply_postprocessing=True)
    if not hasattr(pred,'add_new_mask'):
        raise RuntimeError('Installed SAM2 predictor lacks add_new_mask')
    for ep in ids:
        eroot=root/'episodes'/f'e{ep:05d}'
        prep=load_json(eroot/'prepare.json')
        frames=root/'cache'/'video_frames'/f'e{ep:05d}'
        n=int(prep['num_frames']);hw=img(frames/'00000.jpg').shape[:2]
        seeds={name:read_mask(root/'firstframes'/f'e{ep:05d}'/f'{name}.png',hw) for name in m['objects']}
        if {name:sha(root/'firstframes'/f'e{ep:05d}'/f'{name}.png') for name in m['objects']}!=prep['source_mask_hashes']:
            raise ValueError(f'e{ep}: seed masks changed after video prep')
        if (eroot/'tracking_report.json').is_file():
            validate_video_track(root,m,ep)
            print('[SKIP TRACKED]',ep,flush=True);continue
        incomplete=eroot/'masks.incomplete'
        if incomplete.exists() or (eroot/'masks').exists():
            raise FileExistsError(f'e{ep} partial/existing masks require inspection; refusing overwrite')
        incomplete.mkdir()
        for name in m['objects']:
            (incomplete/name).mkdir()
        count=[-1]*n;areas={name:[-1]*n for name in m['objects']}
        print(f'[TRACK] e{ep}: {n} frames, objects={m["objects"]}',flush=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=pred.init_state(str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for i,name in enumerate(m['objects'],start=1):
                pred.add_new_mask(inference_state=state,frame_idx=0,obj_id=i,
                                  mask=torch.from_numpy(seeds[name].astype(bool)))
            for fi,object_ids,logits in pred.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=n,reverse=False):
                fi=int(fi)
                if not 0<=fi<n or count[fi]!=-1:
                    raise ValueError(f'e{ep}: repeated/out-of-range frame {fi}')
                mapped=list(map(int,object_ids))
                if set(mapped)!=set(range(1,len(m['objects'])+1)):
                    raise ValueError(f'e{ep}: unexpected SAM2 object ids {mapped}')
                for i,name in enumerate(m['objects'],start=1):
                    seg=(logits[mapped.index(i)]>0).detach().cpu().numpy().squeeze().astype(np.uint8)
                    if seg.shape!=hw:
                        raise ValueError(f'e{ep} frame {fi}: {name} shape {seg.shape}, expected {hw}')
                    areas[name][fi]=int(seg.sum())
                    write_png(incomplete/name/f'{fi:05d}.png',seg*255)
                count[fi]=fi
                if fi%100==0 or fi==n-1:
                    print(f'[TRACK] e{ep} f{fi}/{n-1}',flush=True)
        if any(x==-1 for x in count):
            raise ValueError(f'e{ep}: missing video frames')
        # Detect suspicious jumps/near-empty masks; human review still required.
        warnings=[];statistics={};picks={0,n-1}
        picks.update(int(round((n-1)*x)) for x in (0.05,.1,.2,.35,.5,.65,.8,.9,.95))
        for name in m['objects']:
            a=np.asarray(areas[name],dtype=float)
            initial=max(1,int(seeds[name].sum()))
            absent=np.where(a<8)[0].tolist()
            unusual=np.where((a<.15*initial)|(a>3.5*initial))[0].tolist()
            jumps=np.where(np.maximum(a[1:],a[:-1])>3.5*np.maximum(1,np.minimum(a[1:],a[:-1])))[0]+1
            if absent: warnings.append(f'{name}: near-empty frames {absent[:8]} (total={len(absent)})')
            if unusual: warnings.append(f'{name}: area outliers {unusual[:8]} (total={len(unusual)})')
            if len(jumps): warnings.append(f'{name}: area jumps {jumps[:8].tolist()} (total={len(jumps)})')
            picks.update(map(int,jumps[:3]))
            mask0=read_mask(incomplete/name/'00000.png',hw,nonempty=False)
            iou=_iou(seeds[name],mask0)
            if iou<.85: warnings.append(f'{name}: f0 seed IoU={iou:.3f}')
            statistics[name]={"mask0_iou":iou,"min_pixels":int(a.min()),"median_pixels":float(np.median(a)),
                              "max_pixels":int(a.max()),"near_empty_count":len(absent),"jump_count":len(jumps)}
        incomplete.rename(eroot/'masks')
        del state;gc.collect();torch.cuda.empty_cache()
        visual=[]
        for fi in sorted(picks):
            picture=img(frames/f'{fi:05d}.jpg')
            mm={name:read_mask(eroot/'masks'/name/f'{fi:05d}.png',hw,nonempty=False) for name in m['objects']}
            visual.append((f'e{ep} f{fi}',overlay(picture,mm)))
        pages=sheet(visual,eroot/'tracking_contact_sheet.jpg')
        mask_digests={}
        for name in m['objects']:
            h=hashlib.sha256()
            for fi in range(n):
                h.update((eroot/'masks'/name/f'{fi:05d}.png').read_bytes())
            mask_digests[name]=h.hexdigest()
        save_json(eroot/'tracking_report.json',{"schema":SCHEMA,"status":"tracking_complete_unreviewed",
            "episode":ep,"frames":n,"objects":m['objects'],"seed_hashes":prep['source_mask_hashes'],
            "statistics":statistics,"warnings":warnings,"contact_sheets":pages,
            "mask_stream_sha256":mask_digests,
            "contact_sheet_sha256":{p:sha(Path(p)) for p in pages}})
        print(f'[TRACKED UNREVIEWED] e{ep}: warnings={len(warnings)}',flush=True)
    print('[NEXT] Inspect tracking_contact_sheet*.jpg; run video-review on accepted episodes.')


def validate_video_track(root:Path, m:dict, ep:int) -> dict:
    eroot=root/'episodes'/f'e{ep:05d}'
    report=load_json(eroot/'tracking_report.json')
    if report.get('schema')!=SCHEMA or report.get('status')!='tracking_complete_unreviewed' or report.get('episode')!=ep:
        raise ValueError(f'e{ep}: invalid tracking report')
    if report.get('objects')!=m['objects']:
        raise ValueError(f'e{ep}: object contract changed')
    prep=load_json(eroot/'prepare.json')
    if report.get('seed_hashes')!=prep['source_mask_hashes'] or int(report['frames'])!=prep['num_frames']:
        raise ValueError(f'e{ep}: changed seed source or frame count')
    n=int(report['frames'])
    for name in m['objects']:
        target=eroot/'masks'/name
        if len(list(target.glob('*.png')))!=n:
            raise ValueError(f'e{ep}: {name} incomplete tracked masks')
        h=hashlib.sha256()
        for fi in range(n):
            path=target/f'{fi:05d}.png'
            if not path.is_file():
                raise ValueError(f'e{ep}: {name} missing frame {fi}')
            h.update(path.read_bytes())
        if report.get('mask_stream_sha256',{}).get(name)!=h.hexdigest():
            raise ValueError(f'e{ep}: {name} mask stream changed after tracking')
    for path,digest in report.get('contact_sheet_sha256',{}).items():
        if sha(Path(path))!=digest:
            raise ValueError(f'e{ep}: contact sheet changed after tracking')
    return report

def video_review(args) -> None:
    root=args.out.resolve();m=manifest(root)
    ids=ids_arg(args.episodes,m)
    if not set(ids).issubset(approved_firstframe_set(root,m)):
        raise ValueError('Firstframe approval missing')
    for ep in ids:
        eroot=root/'episodes'/f'e{ep:05d}';p=eroot/'tracking_report.json'
        report=validate_video_track(root,m,ep)
        if report.get('seed_hashes')!={name:sha(root/'firstframes'/f'e{ep:05d}'/f'{name}.png') for name in m['objects']}:
            raise ValueError(f'e{ep} seed changed')
        review_path=eroot/'review.json'
        if review_path.exists():
            previous=load_json(review_path)
            if previous.get('status')!='human_accepted' or previous.get('tracking_report_sha256')!=sha(p):
                raise ValueError(f'e{ep}: prior video review provenance mismatch')
            print(f'[SKIP REVIEWED] e{ep}',flush=True)
            continue
        if report['warnings'] and not args.accept_warnings:
            raise ValueError(f'e{ep}: warnings require explicit --accept-warnings after review')
        save_json(review_path,{'schema':SCHEMA,'status':'human_accepted','reviewer':args.reviewer,
            'episode':ep,'tracking_report_sha256':sha(p),
            'note':'Reviewer confirmed sampled/contact/occlusion frames; full numerical masks remain auditable.'})
        print(f'[APPROVED VIDEO] e{ep}',flush=True)


def status(args) -> None:
    root=args.out.resolve();m=manifest(root)
    ref_ok=False
    try:
        approved_reference(root,m);ref_ok=True
    except (FileNotFoundError,ValueError):
        pass
    first=set()
    if ref_ok:
        try:first=approved_firstframe_set(root,m)
        except (FileNotFoundError,ValueError) as e:print('[WARN] stale firstframe review:',e)
    video=[];accepted=[]
    for ep in m['episode_ids']:
        eroot=root/'episodes'/f'e{ep:05d}'
        if (eroot/'tracking_report.json').is_file():video.append(ep)
        if (eroot/'review.json').is_file():accepted.append(ep)
    print(f'[TASK] {m["dataset_id"]} objects={m["objects"]} camera={m["camera"]}')
    print(f'[REFERENCE] {"APPROVED" if ref_ok else "pending"}')
    print(f'[FIRSTFRAMES] approved {len(first)}/{len(m["episode_ids"])}')
    print(f'[VIDEOS] tracked {len(video)} reviewed {len(accepted)} / {len(m["episode_ids"])}')
    print('[MASK ROOT] <out>/episodes/eXXXXX/masks/<object>/00000.png')


def main(argv=None) -> None:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for op in ('prepare','seed','approve-reference','firstframes','approve-firstframes',
               'video-prepare','video-track','video-review','status'):
        sp=sub.add_parser(op)
        sp.add_argument('--out',type=Path,required=True)
        if op=='prepare':
            sp.add_argument('--dataset',type=Path,required=True)
            sp.add_argument('--camera',default='front')
            sp.add_argument('--objects',required=True,help='Two object role names: button,button_base')
            sp.add_argument('--reference-episode',type=int)
            sp.add_argument('--sample-count',type=int,default=27)
        if op in ('seed','firstframes','video-track'):
            sp.add_argument('--checkpoint',type=Path,default=Path('/workspace/mnt/sqzhang26/hoi/weights/sam2/sam2.1_hiera_small.pt'))
        if op in ('firstframes','video-prepare','video-track','video-review'):
            sp.add_argument('--episodes',default='sampled',help='sampled, all, or 0,23,50 / 0-99')
        if op in ('firstframes','approve-firstframes'):
            sp.add_argument('--name',choices=['sampled','all'],default='sampled')
        if op in ('approve-reference','approve-firstframes','video-review'):
            sp.add_argument('--reviewer',default=os.environ.get('USER','unknown'))
        if op in ('approve-firstframes','video-review'):
            sp.add_argument('--accept-warnings',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':prepare(a)
    elif a.command=='seed':seed(a)
    elif a.command=='approve-reference':approve_reference(a)
    elif a.command=='firstframes':
        if a.episodes not in ('sampled','all'):
            p.error('Use --episodes sampled or all; selected video episode subsets are supported separately')
        if a.name=='sampled' and a.episodes=='all':
            p.error('Use --name all when --episodes all')
        if a.name=='all' and a.episodes!='all':
            p.error('Use --episodes all with --name all')
        firstframes(a)
    elif a.command=='approve-firstframes':approve_firstframes(a)
    elif a.command=='video-prepare':video_prepare(a)
    elif a.command=='video-track':video_track(a)
    elif a.command=='video-review':video_review(a)
    else:status(a)


if __name__=='__main__':main()
