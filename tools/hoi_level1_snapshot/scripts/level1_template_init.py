#!/usr/bin/env python3
"""Reuse manually reviewed first-frame object masks as image templates across episodes/locations.
Template matching proposes new first-frame prompts; SAM2 still refines them, human reviews.
Outputs never imply physical surface accuracy or replace source files.
"""
from __future__ import annotations
import argparse,base64,json,math
from pathlib import Path
import numpy as np

HOI=Path('/workspace/mnt/sqzhang26/hoi')
NAMES=('target_flask','funnel')

def read_image(path):
    import cv2
    img=cv2.imread(str(path))
    if img is None:raise FileNotFoundError(path)
    return img

def load_templates(home:Path, skip_episode:int):
    import cv2
    refs=[]
    for epdir in sorted((home/'outputs').glob('level1_sequence_e*')):
        try:eid=int(epdir.name.rsplit('_e',1)[1])
        except ValueError:continue
        if eid==skip_episode:continue
        # Only e0's previously manually checked masks are assumed reviewed.
        # Other episodes must have explicit preview approval and tracked masks.
        if eid!=0 and not (epdir/'approved.json').exists():continue
        imgpath=home/f'data/level1_sequence_e{eid}/frames/00000.jpg'
        if not imgpath.is_file():continue
        rgb=read_image(imgpath)
        for name in NAMES:
            mp=epdir/'masks'/name/'00000.png'
            if not mp.is_file():continue
            mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
            if mask is None or not np.any(mask>127) or mask.shape!=rgb.shape[:2]:continue
            refs.append(dict(episode=eid,name=name,rgb=rgb,mask=mask>127,root=epdir))
    return refs

def roi_mask_bbox(mask,margin_scale=0.8,min_margin=18):
    ys,xs=np.where(mask)
    if not len(xs):raise ValueError('Empty reference mask')
    x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
    w=x1-x0;h=y1-y0
    pad_x=max(min_margin,round(w*margin_scale));pad_y=max(min_margin,round(h*margin_scale))
    return x0-pad_x,y0-pad_y,x1+pad_x,y1+pad_y

def match_image(ref_rgb,mask,target_rgb,scales,margin_scale=0.8):
    """Translation + isotropic scale only. Does not assume a known position ID.
    Returns diagnostic confidence, which is NOT probability of correct segmentation.
    """
    import cv2
    if ref_rgb.shape[:2]!=target_rgb.shape[:2]:raise ValueError('Camera resolution changed; no matching without explicit calibration')
    H,W=target_rgb.shape[:2];x0,y0,x1,y1=roi_mask_bbox(mask,margin_scale)
    # Ensure crop is within source frame; offset kept for full-mask warping.
    cx0=max(0,x0);cy0=max(0,y0);cx1=min(W,x1);cy1=min(H,y1)
    template=ref_rgb[cy0:cy1,cx0:cx1]
    if min(template.shape[:2])<12:raise ValueError('Reference object patch too small')
    # Local contrast normalization mitigates some exposure variation; keep spatial pattern.
    def norm_gray(img):
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(g)
    target=norm_gray(target_rgb)
    candidate=[]
    for scale in scales:
        tw=max(8,round(template.shape[1]*scale));th=max(8,round(template.shape[0]*scale))
        if th>H or tw>W:continue
        tpl=cv2.resize(norm_gray(template),(tw,th),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        # Template may be constant under severe exposure, abort rather than silently fail.
        if float(tpl.std())<3.0:continue
        scoremap=cv2.matchTemplate(target,tpl,cv2.TM_CCOEFF_NORMED)
        if not np.isfinite(scoremap).all():continue
        _min,maxval,_,pos=cv2.minMaxLoc(scoremap)
        # Global top competitor MUST be spatially distinct, not adjacent same-object location.
        cut=max(15,int(max(tw,th)*.6))
        exclusion=(max(0,pos[0]-cut),max(0,pos[1]-cut),min(scoremap.shape[1],pos[0]+cut+1),min(scoremap.shape[0],pos[1]+cut+1))
        sm=scoremap.copy();a,b,c,d=exclusion;sm[b:d,a:c]=-2
        runner=float(np.max(sm)) if sm.size else -1.
        candidate.append({'score':float(maxval),'runner_up':runner,'scale':float(scale),
                          'crop_xy':list(map(int,pos)), 'reference_crop_xy':[cx0,cy0,cx1,cy1],
                          'crop_wh':[tw,th]})
    if not candidate:raise RuntimeError('No non-degenerate visual template matched')
    candidate.sort(key=lambda c:c['score'],reverse=True)
    best=candidate[0]
    # Include competing different-scale maxima, but identical site with different scale is not a competing object.
    competitors=[best['runner_up']]
    bx,by=best['crop_xy'];bw,bh=best['crop_wh']
    for x in candidate[1:]:
        cx,cy=x['crop_xy'];cw,ch=x['crop_wh']
        if abs((cx+cw/2)-(bx+bw/2))>=.6*max(bw,cw) or abs((cy+ch/2)-(by+bh/2))>=.6*max(bh,ch):
            competitors.append(x['score'])
    best['runner_up_different_location']=max(competitors)
    best['margin']=best['score']-best['runner_up_different_location']
    return best

def sample_clicks(mask,limit=7):
    import cv2
    binary=mask.astype(np.uint8)
    H,W=binary.shape
    erode=cv2.erode(binary,np.ones((3,3),np.uint8))
    usable=erode if np.any(erode) else binary
    dist=cv2.distanceTransform(usable,cv2.DIST_L2,3)
    pos=[]
    for _ in range(limit):
        y,x=np.unravel_index(np.argmax(dist),dist.shape)
        if dist[y,x]<=.5:break
        pos.append([float(x),float(y)])
        cv2.circle(dist,(int(x),int(y)),max(6,round(float(dist[y,x])*2)),0,-1)
    if not pos:raise RuntimeError('Template mask too thin to derive a positive point')
    # Background negatives sampled a small distance away from mask bbox, not across other objects.
    # These are prompts ONLY, must be checked by the user in HTML.
    bbox=roi_mask_bbox(mask,margin_scale=0.25,min_margin=12)
    x0,y0,x1,y1=bbox
    options=[((x0+x1)/2,y0),(x0,(y0+y1)/2),(x1-1,(y0+y1)/2),((x0+x1)/2,y1-1)]
    neg=[]
    for xx,yy in options:
        x=min(W-1,max(0,int(xx)));y=min(H-1,max(0,int(yy)))
        if not binary[y,x]:neg.append([float(x),float(y)])
    return pos,neg

def get_ref_points(ref):
    path=ref['root']/'flask_prompts_e0.json' if ref['name']=='target_flask' else None
    if path and path.is_file():
        try:
            obj=json.loads(path.read_text(encoding='utf8'))['objects'][ref['name']]
            return obj['positive'],obj['negative'],'original_clicks'
        except (KeyError,ValueError):pass
    path=ref['root']/'seed_prompts.json'
    if path.is_file():
        try:
            obj=json.loads(path.read_text(encoding='utf8'))['objects'][ref['name']]
            return obj['positive'],obj['negative'],'previous_reviewed_prompts'
        except (KeyError,ValueError):pass
    p,n=sample_clicks(ref['mask'])
    return p,n,'derived_from_previous_mask'

def transfer_points(points,ref,match,shape):
    H,W=shape
    x0,y0,x1,y1=match['reference_crop_xy'];dx,dy=match['crop_xy']
    sx=match['crop_wh'][0]/(x1-x0);sy=match['crop_wh'][1]/(y1-y0)
    out=[]
    for x,y in points:
        nx=dx+(float(x)-x0)*sx;ny=dy+(float(y)-y0)*sy
        if 0<=nx<W and 0<=ny<H:out.append([int(round(nx)),int(round(ny))])
    return list(dict.fromkeys(tuple(p) for p in out))

def translated_mask(ref,match,shape):
    import cv2
    x0,y0,x1,y1=match['reference_crop_xy'];dx,dy=match['crop_xy']
    sx=match['crop_wh'][0]/(x1-x0);sy=match['crop_wh'][1]/(y1-y0)
    T=np.array([[sx,0,dx-sx*x0],[0,sy,dy-sy*y0]],dtype=np.float32)
    return cv2.warpAffine(ref['mask'].astype('uint8')*255,T,(shape[1],shape[0]),flags=cv2.INTER_NEAREST)

def propose(args):
    import cv2
    target=read_image(args.clip/'frames/00000.jpg');H,W=target.shape[:2]
    library=load_templates(args.hoi,args.episode)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'seed_prompts.json').exists():raise FileExistsError(f'{root}/seed_prompts.json exists; preserve human edits. Use --overwrite-seed if intended')
    report={};objects={};masks={}
    for name in NAMES:
        matches=[]
        for ref in library:
            if ref['name']!=name:continue
            try:
                match=match_image(ref['rgb'],ref['mask'],target,[.75,.85,.95,1.0,1.05,1.15,1.30],margin_scale=args.crop_margin)
                matches.append((match['score'],match,ref))
            except (ValueError,RuntimeError) as e:print('[SKIP TEMPLATE]',name,ref['episode'],e)
        if not matches:raise RuntimeError(f'No previous confirmed template for {name}. Need e0 masks and source video.')
        _,m,ref=max(matches,key=lambda v:v[0]);p,n,source=get_ref_points(ref)
        pos=transfer_points(p,ref,m,(H,W));neg=transfer_points(n,ref,m,(H,W))
        if not pos:raise RuntimeError(f'No transferred positive points for {name}; annotate manually')
        objects[name]={'positive':pos,'negative':neg}
        masks[name]=translated_mask(ref,m,(H,W))
        qualified=m['score']>=args.min_score and m['margin']>=args.min_margin
        report[name]={'template_episode':ref['episode'],'ref_point_source':source,**m,'automatic_match_qualified':qualified}
        print(f'[MATCH] {name}: template=e{ref["episode"]} score={m["score"]:.3f} margin={m["margin"]:.3f} qualified={qualified}')
    obj={'schema_version':2,'episode_index':args.episode,'frame_index':0,'image_width':W,'image_height':H,
         'objects':objects,'auto_reference':'Translation/scale from previous reviewed first-frame templates. Unverified: inspect SAM2 preview.'}
    path=root/'seed_prompts.json';path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    (root/'template_match_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    image=target.copy()
    colors={'target_flask':(255,0,215),'funnel':(0,140,255)}
    for name,m in masks.items():
        cn,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,cn,-1,colors[name],2)
        for x,y in objects[name]['positive']:cv2.circle(image,(x,y),4,(0,255,0),-1)
        for x,y in objects[name]['negative']:cv2.circle(image,(x,y),3,(0,0,255),-1)
    cv2.imwrite(str(root/'template_seed_preview.jpg'),image)
    print('[OUTPUT]',path,'preview:',root/'template_seed_preview.jpg')
    if not all(x['automatic_match_qualified'] for x in report.values()):
        print('[LOW CONFIDENCE] Run annotate and correct transferred points before SAM2 preview. Do NOT approve automatically.')

def annotate(args):
    import cv2
    image=read_image(args.clip/'frames/00000.jpg');H,W=image.shape[:2]
    buf=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tobytes()
    encoded='data:image/jpeg;base64,'+base64.b64encode(buf).decode('ascii')
    src=args.output/'seed_prompts.json'
    old=json.loads(src.read_text(encoding='utf8')) if src.is_file() else None
    preload=[]
    if old:
        for name,x in old['objects'].items():
            for xy in x['positive']:preload.append({'o':name,'l':1,'x':xy[0],'y':xy[1]})
            for xy in x['negative']:preload.append({'o':name,'l':0,'x':xy[0],'y':xy[1]})
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>双物体点标注</title><style>body{font:14px sans-serif;background:#f4f6fa;display:flex;gap:18px;margin:15px}canvas{width:min(65vw,100%);height:auto;border:1px solid #777;cursor:crosshair}aside{width:330px}button{padding:9px;margin:4px;cursor:pointer}b{color:#673ab7}pre{max-height:220px;overflow:auto}small{color:#666}</style>
<section><h2>Episode __EP__ / Frame 0</h2><p>选择<strong>右侧目标烧瓶</strong>或<strong>漏斗</strong>，绿点属于选中物体，红点不属于。左侧初始锥形瓶不是目标烧瓶。</p><canvas width="__W__" height="__H__"></canvas><p>已预加载模板建议点，可逐个删掉并重新点。需要两个物体都有正点。</p></section><aside><h3>选中物体</h3><button id="flask">目标烧瓶</button><button id="funnel">漏斗</button><hr><button id="pos">正点</button><button id="neg">负点</button><p id="which"></p><button id="undo">撤销</button><button id="clear">清空当前物体</button><button id="delete">点选删除模式</button><hr><button id="dl"><b>下载 seed_prompts.json</b></button><p><small>下载后上传覆盖服务器该Episode的同名 JSON，再运行 SAM2 preview。必须自己查看 preview，不是模板匹配成功就算成功。</small></p><pre id="points"></pre></aside>
<script>const W=__W__,H=__H__,EP=__EP__,img=new Image(),canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');img.src='__IMAGE__';let data=__PRELOAD__,obj='target_flask',lab=1,deleteMode=false;
function draw(){ctx.clearRect(0,0,W,H);ctx.drawImage(img,0,0,W,H);ctx.font='bold 12px sans-serif';data.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.fillStyle=p.l?'#2aff47':'#ff3232';ctx.fill();ctx.strokeStyle=p.o==='funnel'?'#ffab00':'#b700ff';ctx.lineWidth=3;ctx.stroke();ctx.fillStyle='#222';ctx.fillText(''+(i+1),p.x+7,p.y-8)});document.querySelector('#which').textContent=`当前 ${obj} / ${lab?'正点':'负点'} ${deleteMode?' / 点选删除':''}`;document.querySelector('#points').textContent=data.map((p,i)=>`${i+1} ${p.o} ${p.l?'+':'-'} (${p.x},${p.y})`).join('\n')};img.onload=draw;
canvas.onclick=e=>{let r=canvas.getBoundingClientRect(),x=Math.max(0,Math.min(W-1,Math.round((e.clientX-r.left)*W/r.width))),y=Math.max(0,Math.min(H-1,Math.round((e.clientY-r.top)*H/r.height)));if(deleteMode){let j=data.findIndex(p=>Math.hypot(p.x-x,p.y-y)<13);if(j>=0)data.splice(j,1)}else data.push({o:obj,l:lab,x,y});draw()};
for(const [id,value] of [['flask','target_flask'],['funnel','funnel']])document.getElementById(id).onclick=()=>{obj=value;deleteMode=false;draw()};document.querySelector('#pos').onclick=()=>{lab=1;deleteMode=false;draw()};document.querySelector('#neg').onclick=()=>{lab=0;deleteMode=false;draw()};document.querySelector('#delete').onclick=()=>{deleteMode=!deleteMode;draw()};document.querySelector('#undo').onclick=()=>{data.pop();draw()};document.querySelector('#clear').onclick=()=>{if(confirm('清空当前物体点？'))data=data.filter(x=>x.o!==obj);draw()};
document.querySelector('#dl').onclick=()=>{let objects={};for(let name of ['funnel','target_flask']){objects[name]={positive:data.filter(x=>x.o===name&&x.l===1).map(x=>[x.x,x.y]),negative:data.filter(x=>x.o===name&&x.l===0).map(x=>[x.x,x.y])};if(!objects[name].positive.length){alert(name+' 没有正点');return}}const o={schema_version:2,episode_index:EP,frame_index:0,image_width:W,image_height:H,objects,manual_reviewed_prompts:true};let u=URL.createObjectURL(new Blob([JSON.stringify(o,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='seed_prompts.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script></html>'''
    page=page.replace('__EP__',str(args.episode)).replace('__W__',str(W)).replace('__H__',str(H)).replace('__IMAGE__',encoded).replace('__PRELOAD__',json.dumps(preload))
    path=args.output/'annotate_frame0.html';path.write_text(page,encoding='utf8')
    print('[ANNOTATE]',path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cmd',choices=['seed','annotate']);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--hoi',type=Path,default=HOI);p.add_argument('--clip',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--min-score',type=float,default=.68);p.add_argument('--min-margin',type=float,default=.035);p.add_argument('--crop-margin',type=float,default=.8)
    args=p.parse_args()
    if args.clip is None:args.clip=args.hoi/f'data/level1_sequence_e{args.episode}'
    if args.output is None:args.output=args.hoi/f'outputs/level1_sequence_e{args.episode}'
    args.output.mkdir(parents=True,exist_ok=True)
    {'seed':propose,'annotate':annotate}[args.cmd](args)
if __name__=='__main__':main()
