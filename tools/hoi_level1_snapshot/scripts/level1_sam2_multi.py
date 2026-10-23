#!/usr/bin/env python3
"""Two-object SAM2 frame-0 prompt initializer + causal forward tracking for any episode.
Run under HOI's SAM2 Python from hoi/repos/sam2. Does not alter original data.
Preview does not approve; human must inspect and explicitly run approve before track.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

HOI=Path('/workspace/mnt/sqzhang26/hoi')
NAMES=('target_flask','funnel')
COLORS={'target_flask':(225,90,150),'funnel':(0,135,255)}  # BGR

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def validate(args):
    import cv2
    clip=Path(args.clip); meta=json.loads((clip/'manifest.json').read_text(encoding='utf8'))
    if int(meta['episode'])!=args.episode:raise ValueError('wrong clip/episode')
    n=int(meta['num_frames'])
    if set(int(x['clip_index']) for x in meta['mapping'])!=set(range(n)):
        raise ValueError('video frames must be 0..N-1')
    first=cv2.imread(str(clip/'frames/00000.jpg'))
    if first is None:raise FileNotFoundError(clip/'frames/00000.jpg')
    h,w=first.shape[:2]
    prm=json.loads(Path(args.prompts).read_text(encoding='utf8'))
    if int(prm['episode_index'])!=args.episode or int(prm['frame_index'])!=0:raise ValueError('prompt wrong episode/frame')
    if (int(prm['image_width']),int(prm['image_height']))!=(w,h):raise ValueError('prompt image shape mismatch')
    points={}
    for name in NAMES:
        x=prm['objects'][name]
        pos=np.asarray(x['positive'],dtype=np.float32).reshape(-1,2)
        neg=np.asarray(x['negative'],dtype=np.float32).reshape(-1,2)
        if not len(pos):raise ValueError(f'{name}: at least one positive click required')
        pts=np.concatenate([pos,neg])
        if not (np.isfinite(pts).all() and np.all((0<=pts[:,0])&(pts[:,0]<w)&(0<=pts[:,1])&(pts[:,1]<h))):
            raise ValueError(f'{name}: out-of-bounds or invalid point')
        points[name]=(pts,np.array([1]*len(pos)+[0]*len(neg),np.int32))
    return clip,n,first,points

def predictor(args):
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(args.checkpoint.resolve()),device='cuda',apply_postprocessing=True)

def add_both(pred,st,points):
    for name,objid in zip(NAMES,(1,2)):
        p,labels=points[name]
        pred.add_new_points_or_box(inference_state=st,frame_idx=0,obj_id=objid,points=p,labels=labels)

def arrays(ids,logits):
    ix=[int(x) for x in ids]
    if set(ix)!=set((1,2)):raise RuntimeError(f'Expected object IDs 1,2 got {ix}')
    return {name:(logits[ix.index(objid)]>0).detach().cpu().numpy().squeeze().astype(np.uint8)
            for name,objid in zip(NAMES,(1,2))}

def vis(im,masks,points=None,title=''):
    import cv2
    out=im.copy()
    for name,m in masks.items():
        color=COLORS[name]
        contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        over=out.copy();over[m!=0]=color;out=cv2.addWeighted(over,.22,out,.78,0)
        cv2.drawContours(out,contours,-1,color,2)
        if points and name in points:
            pp,ll=points[name]
            for (x,y),lab in zip(pp,ll):cv2.circle(out,(round(float(x)),round(float(y))),4,(0,255,0) if lab else (0,0,255),-1)
    if title:cv2.putText(out,title,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3);cv2.putText(out,title,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    return out

def preview(args):
    import cv2, torch
    clip,n,first,points=validate(args);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pred=predictor(args)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        st=pred.init_state(str(clip/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        add_both(pred,st,points)
        # SAM2 API may return only updated masks on add; request an explicit frame-0 propagation.
        observations=[]
        for idx,ids,logits in pred.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=1,reverse=False):
            observations.append((idx,arrays(ids,logits)))
    frame0 = [masks for idx, masks in observations if int(idx) == 0]
    if len(frame0) != 1:
        raise RuntimeError(
            f'Expected exactly one frame0 observation, got indices {[int(idx) for idx, _ in observations]}'
        )
    masks = frame0[0]
    overlap=int(np.sum((masks[NAMES[0]]!=0)&(masks[NAMES[1]]!=0)))
    counts={name:int(m.sum()) for name,m in masks.items()}
    for name,m in masks.items():cv2.imwrite(str(out/f'{name}_preview_binary.png'),m*255)
    cv2.imwrite(str(out/'mask_preview_frame0.jpg'),vis(first,masks,points,'E'+str(args.episode)+' frame0: REVIEW before approval'))
    props={name:round(v/(first.shape[0]*first.shape[1]),5) for name,v in counts.items()}
    warnings=[]
    for name,count in counts.items():
        if count<25 or props[name]>.35:warnings.append(f'{name}: implausible initial mask area {count} pixels')
    if overlap>min(counts.values())*.15:warnings.append(f'Large overlapping masks: {overlap}')
    approved=out/'approved.json'
    if approved.exists():approved.unlink()   # invalidate stale manual review after preview regenerate
    save_json(out/'preview_report.json',{'episode':args.episode,'prompts_sha256':digest(args.prompts),'num_frames':n,'counts':counts,'overlap':overlap,'warnings':warnings})
    print('[PREVIEW]',out/'mask_preview_frame0.jpg','mask pixels:',counts,'warnings:',warnings)
    print('[NEXT] Download/review screenshot. If wrong: edit clicks in annotator and rerun preview. If good: explicit approve.')

def approve(args):
    out=Path(args.output);meta=json.loads((out/'preview_report.json').read_text(encoding='utf8'))
    if meta['episode']!=args.episode or meta['prompts_sha256']!=digest(args.prompts):raise ValueError('Preview not current for prompts')
    if meta['warnings'] and not args.accept_warnings:raise ValueError(f'Warnings need explicit --accept-warnings: {meta["warnings"]}')
    save_json(out/'approved.json',{'episode':args.episode,'prompts_sha256':digest(args.prompts),'preview_report':meta,'human_review_required':True,'approved_by_user_command':True})
    print('[APPROVED]',out/'approved.json')

def track(args):
    import cv2,torch
    clip,n,first,points=validate(args);out=Path(args.output)
    approved=json.loads((out/'approved.json').read_text(encoding='utf8'))
    if approved['episode']!=args.episode or approved['prompts_sha256']!=digest(args.prompts):raise ValueError('Approval does not match current prompts')
    maskroot=out/'masks'
    if maskroot.exists():raise FileExistsError(f'Refusing to overwrite existing masks {maskroot}')
    maskroot.mkdir(parents=True)
    for name in NAMES:(maskroot/name).mkdir()
    pred=predictor(args);counts={name:{} for name in NAMES}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        st=pred.init_state(str(clip/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        add_both(pred,st,points)
        for idx,ids,logits in pred.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=n,reverse=False):
            idx=int(idx);masks=arrays(ids,logits)
            for name,m in masks.items():
                cv2.imwrite(str(maskroot/name/f'{idx:05d}.png'),m*255)
                counts[name][idx]=int(m.sum())
            if idx%50==0:print('[TRACK]',args.episode,idx,'/',n-1,flush=True)
    if any(set(counts[name])!=set(range(n)) for name in NAMES):
        raise RuntimeError('Tracking incomplete; do not treat results as ready')
    picks=sorted(set([0,n-1]+[round((n-1)*x) for x in (.1,.25,.4,.55,.7,.85)]))
    thumbs=[]
    for k in picks:
        im=cv2.imread(str(clip/'frames'/f'{k:05d}.jpg'))
        masks={name:(cv2.imread(str(maskroot/name/f'{k:05d}.png'),0)>127).astype(np.uint8) for name in NAMES}
        thumb=cv2.resize(vis(im,masks,title=f'e{args.episode} f{k}'),(480,360))
        thumbs.append(thumb)
    sheet=np.zeros((math.ceil(len(thumbs)/2)*360,960,3),np.uint8)
    for j,im in enumerate(thumbs):sheet[j//2*360:(j//2+1)*360,j%2*480:(j%2+1)*480]=im
    cv2.imwrite(str(out/'tracking_contact_sheet.jpg'),sheet)
    # Flag suspicious mask jumps and losses, but NEVER mark a track human-reviewed.
    flags=[]
    for name in NAMES:
        areas=np.array([counts[name][k] for k in range(n)],dtype=np.float64)
        bad=np.flatnonzero(areas<25)
        if len(bad)>max(2,int(.03*n)):
            flags.append(f'{name}: small/empty masks at {len(bad)} frames (first {bad[:6].tolist()})')
        ratios=np.maximum(np.maximum(areas[1:],1)/np.maximum(areas[:-1],1),
                          np.maximum(areas[:-1],1)/np.maximum(areas[1:],1))
        jumps=np.flatnonzero((ratios>4) & (np.maximum(areas[1:],areas[:-1])>100))
        if len(jumps):flags.append(f'{name}: >4x mask area jumps near frames {jumps[:6].tolist()}')
    hashes=hashlib.sha256()
    for name in NAMES:
        for k in range(n):
            hashes.update(name.encode('utf8'));hashes.update(str(k).encode('ascii'))
            hashes.update((maskroot/name/f'{k:05d}.png').read_bytes())
    save_json(maskroot/'masks_report.json',{'masks_sha256':hashes.hexdigest(),'status':'complete','episode':args.episode,'num_frames':n,'approved_prompt_sha256':digest(args.prompts),'mask_pixels':counts,'warnings':flags,
             'warning':'SAM2 forward masks only; inspect tracking_contact_sheet and contact phase; preview approval does not prove tracking is correct.'})
    print('[DONE]',maskroot,'sheet:',out/'tracking_contact_sheet.jpg','warnings:',flags)

def review_track(args):
    out=Path(args.output);report=out/'masks/masks_report.json'
    data=json.loads(report.read_text(encoding='utf8'))
    if data.get('episode')!=args.episode or data.get('status')!='complete':
        raise ValueError('No complete mask tracking for this episode')
    if data.get('warnings') and not args.accept_warnings:
        raise ValueError('Check tracking_contact_sheet.jpg first. Warnings: '+str(data['warnings'])+'; use --accept-warnings ONLY after reviewing and accepting')
    if not (out/'tracking_contact_sheet.jpg').exists():raise FileNotFoundError('Missing tracking contact sheet')
    save_json(out/'tracking_reviewed.json',{'episode':args.episode,'masks_report_sha256':digest(report),'human_review_required':True,'reviewed_by_explicit_command':True})
    print('[TRACK REVIEWED]',out/'tracking_reviewed.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preview','approve','track','review-track']);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--clip',type=Path,default=None);p.add_argument('--prompts',type=Path,default=None);p.add_argument('--output',type=Path,default=None)
    p.add_argument('--checkpoint',type=Path,default=HOI/'weights/sam2/sam2.1_hiera_small.pt');p.add_argument('--accept-warnings',action='store_true')
    args=p.parse_args()
    if args.episode<0:p.error('episode must be >=0')
    if args.clip is None:args.clip=HOI/f'data/level1_sequence_e{args.episode}'
    if args.output is None:args.output=HOI/f'outputs/level1_sequence_e{args.episode}'
    if args.prompts is None:args.prompts=args.output/'seed_prompts.json'
    {'preview':preview,'approve':approve,'track':track,'review-track':review_track}[args.command](args)
if __name__=='__main__':main()
