#!/usr/bin/env python3
"""Six-position manual-prompt library, for the existing Level-1 SAM2 episode pipeline.

Only human-selected points are used as *position templates*. No image matching,
white-fixture localization, mask copying, or automatic SAM2 approval is performed.

Stages:
  annotate : generate 5 clean frame-0 labeling HTML pages (flask blank; old funnel hint optional)
  install  : import 5 downloaded seed_prompts_e<EP>.json from --inbox
  spread   : copy group template points into every same-location episode metadata
  status   : print source/target coverage

Default chronological group boundaries are provisional and based on the collage.
Change --groups if the episode boundaries are different.
"""
from __future__ import annotations
import argparse, copy, json, os, shutil, sys, tempfile, hashlib
from pathlib import Path

HOI=Path('/workspace/mnt/sqzhang26/hoi')
GROUPS='0-15:0,16-30:23,31-45:38,46-53:49,54-61:58,62-69:65'
NAMES=('target_flask','funnel')


def groups_parse(s):
    groups=[];seen=set();ids=set()
    for z in s.split(','):
        if not z.strip():continue
        rg,rep=z.split(':',1);a,b=[int(v) for v in rg.split('-',1)];rep=int(rep)
        if a<0 or b<a or rep<a or rep>b:raise ValueError('Invalid group '+z)
        part=set(range(a,b+1))
        if seen.intersection(part):raise ValueError('Overlapping group ranges')
        seen|=part;ids.add(rep);groups.append({'start':a,'end':b,'source':rep})
    if not groups:raise ValueError('No groups specified')
    return groups


def ensure_json(src):
    with Path(src).open(encoding='utf8') as f:return json.load(f)


def safe_points(obj,origin):
    for name in NAMES:
        if name not in obj:raise ValueError(f'{origin}: missing {name}')
        for field in ('positive','negative'):
            pts=obj[name].get(field,[])
            if not isinstance(pts,list):raise ValueError(f'{origin} {name}.{field} not list')
            if field=='positive' and not pts:raise ValueError(f'{origin}: {name} must have >=1 positive point')
            for xy in pts:
                if not isinstance(xy,(list,tuple)) or len(xy)!=2 or not all(isinstance(v,(int,float)) for v in xy):
                    raise ValueError(f'{origin}: invalid point {xy}')
    return True


def check_bounds(p,w,h,origin):
    for name in NAMES:
        for k in ('positive','negative'):
            for x,y in p[name][k]:
                if not 0<=x<w or not 0<=y<h:raise ValueError(f'{origin}: {name} {k} {(x,y)} outside {w}x{h}')


def first_shape(hoi,ep):
    import cv2
    p=hoi/f'data/level1_sequence_e{ep}/frames/00000.jpg'
    if not p.is_file():return None
    im=cv2.imread(str(p))
    if im is None:raise RuntimeError(f'Cannot read {p}')
    return int(im.shape[1]),int(im.shape[0])


def jpath(hoi,ep):return hoi/f'outputs/level1_sequence_e{ep}/seed_prompts.json'


def group_for(ep,groups):
    for i,g in enumerate(groups,1):
        if g['start']<=ep<=g['end']:return i,g
    return None,None


def backup_then_write(hoi,ep,payload,replace_auto=True):
    root=hoi/f'outputs/level1_sequence_e{ep}'
    root.mkdir(parents=True,exist_ok=True)
    target=root/'seed_prompts.json'
    data=json.dumps(payload,ensure_ascii=False,indent=2)+'\n'
    if target.exists():
        old=ensure_json(target)
        if old==payload:return 'unchanged'
        if (old.get('manual_reviewed_prompts') is True and
            old.get('manual_source_episode') != payload.get('manual_source_episode')):
            raise RuntimeError(f'e{ep}: existing manually reviewed file; refusing to replace')
        # No silent overwrites of tracked/approved datasets (even if auto initialization was wrong).
        if (root/'masks').exists() or (root/'approved.json').exists() or (root/'tracking_reviewed.json').exists():
            raise RuntimeError(f'e{ep}: has masks or approval. Review and archive old results before changing prompts.')
        if not replace_auto:raise RuntimeError(f'e{ep}: file exists')
        bk=root/'seed_prompts.before_group_copy.json'
        if not bk.exists():shutil.copy2(target,bk)
    tmp=root/'seed_prompts.json.tmp'
    tmp.write_text(data,encoding='utf8')
    os.replace(tmp,target)
    return 'written'


def make_annotation_ui(hoi,ep,with_funnel_hint=True):
    clip=hoi/f'data/level1_sequence_e{ep}'
    if not (clip/'frames/00000.jpg').is_file():
        raise FileNotFoundError(f'e{ep}: no prepared first frame. Run: bash "$HOI/scripts/level1_ep.sh" {ep} prepare')
    sys.path.insert(0,str(hoi/'scripts'))
    try:
        import level1_template_init as base
    except ImportError as e:
        raise ImportError('Upload this script alongside existing level1_template_init.py') from e
    root=hoi/f'outputs/level1_sequence_e{ep}'
    manual=root/'manual_label_ui';manual.mkdir(parents=True,exist_ok=True)
    tmpseed=manual/'seed_prompts.json'
    if with_funnel_hint and jpath(hoi,ep).is_file():
        old=ensure_json(jpath(hoi,ep))
        f=old.get('objects',{}).get('funnel',{})
        if f.get('positive'):
            obj={'objects':{'target_flask':{'positive':[],'negative':[]},
                            'funnel':{'positive':f.get('positive',[]),'negative':f.get('negative',[])}}}
            tmpseed.write_text(json.dumps(obj),encoding='utf8')
    import argparse as ap
    try:
        base.annotate(ap.Namespace(episode=ep,clip=clip,output=manual))
    finally:
        if tmpseed.is_file():tmpseed.unlink()
    path=manual/'annotate_frame0.html';s=path.read_text(encoding='utf8')
    s=s.replace('选择<strong>右侧目标烧瓶</strong>或<strong>漏斗</strong>',
                '选择<strong>目标烧瓶（玻璃瓶本体，不是白色夹具）</strong>或<strong>漏斗</strong>')
    s=s.replace('已预加载模板建议点，可逐个删掉并重新点。需要两个物体都有正点。',
                '烧瓶点全部由你手工标注。漏斗如果有预载点，只是旧的提示，必须核对。两个物体都需要正点。')
    s=s.replace("join('\n')", "join('\\n')")  # original annotator mistakenly emitted a raw newline inside JS string
    s=s.replace("a.download='seed_prompts.json'","a.download='seed_prompts_e'+EP+'.json'")
    s=s.replace('下载 seed_prompts.json','下载当前Episode的JSON')
    s=s.replace('下载后上传覆盖服务器该Episode的同名 JSON，再运行 SAM2 preview。必须自己查看 preview，不是模板匹配成功就算成功。',
                '下载后把 seed_prompts_eN.json 全部上传到 hoi/outputs/level1_manual_labels/，运行 install。首帧SAM2 Mask仍需逐个验收。')
    path.write_text(s,encoding='utf8')
    print(f'[ANNOTATE e{ep:02d}] {path} | flask blank, funnel hint={with_funnel_hint and f.get("positive") is not None if "f" in locals() else False}')


def reference_e0(hoi):
    """Create e0 point template from its existing initial masks + manually clicked flask points.
    This does NOT claim masks are true surfaces; only SAM2 initialization points.
    """
    import cv2
    root=hoi/'outputs/level1_sequence_e0'
    seed=root/'seed_prompts.json'
    fsource=root/'flask_prompts_e0.json'
    flask=None
    if fsource.is_file():
        ff=ensure_json(fsource)
        flask=ff.get('objects',{}).get('target_flask')
    if not flask or not flask.get('positive'):
        if seed.is_file():
            x=ensure_json(seed)
            if all(x.get('objects',{}).get(name,{}).get('positive') for name in NAMES):
                print('[CAUTION] e0 original manual flask prompts missing; falling back to e0 seed. Review group1 first-frame masks.')
                safe_points(x['objects'],seed);return x
        raise RuntimeError('e0 lacks valid flask prompts; manually annotate e7 and change first group source to 7')
    # Use previously reviewed mask, if available, only to propose the first-frame funnel clicks.
    m=root/'masks/funnel/00000.png'
    if not m.is_file():
        if seed.is_file():
            x=ensure_json(seed)
            if all(x.get('objects',{}).get(name,{}).get('positive') for name in NAMES):
                x['objects']['target_flask']=flask
                print('[CAUTION] e0 funnel first-frame mask unavailable; using old funnel seed. Review group1 first-frame masks.')
                safe_points(x['objects'],seed);return x
        raise RuntimeError('e0 lacks initial funnel mask for source prompts; manually annotate e7 and change group source to 7')
    mask=cv2.imread(str(m),cv2.IMREAD_GRAYSCALE)
    if mask is None or not (mask>127).any():raise RuntimeError('e0 funnel mask missing/empty')
    sys.path.insert(0,str(hoi/'scripts'))
    from level1_template_init import sample_clicks
    p,n=sample_clicks(mask>127)
    wh=first_shape(hoi,0)
    if not wh:raise RuntimeError('e0 first frame not prepared. Run episode 0 prepare or manually annotate e7')
    objs={'target_flask':flask, 'funnel':{'positive':p,'negative':n}}
    safe_points(objs,'e0 derived');check_bounds(objs,*wh,'e0 derived')
    return {'schema_version':2,'episode_index':0,'frame_index':0,
            'image_width':wh[0],'image_height':wh[1], 'objects':objs,
            'e0_reference_provenance':'manually clicked flask + funnel sampled from existing e0 initial mask; inspect group1 preview'}


def get_reference(hoi,ep):
    path=jpath(hoi,ep)
    if ep==0:
        return reference_e0(hoi)
    if not path.is_file():raise FileNotFoundError(f'No manual reference prompts for e{ep}: {path}. Run install first.')
    dat=ensure_json(path)
    if not dat.get('manual_reviewed_prompts'):
        raise RuntimeError(f'e{ep} is NOT a manually annotated reference; please run install first')
    safe_points(dat['objects'],f'e{ep}')
    if dat['episode_index']!=ep or dat['frame_index']!=0:raise ValueError(f'e{ep}: incorrect metadata')
    return dat


def install(hoi,reps,inbox,groups):
    import cv2
    for ep in reps:
        i,g=group_for(ep,groups)
        if g is None or g['source']!=ep:raise RuntimeError(f'e{ep} not a group source in --groups')
        source=inbox/f'seed_prompts_e{ep}.json'
        if not source.exists():raise FileNotFoundError(f'Missing {source}; upload exported HTML labels here first')
        dat=ensure_json(source)
        if dat.get('episode_index')!=ep or dat.get('frame_index')!=0:raise ValueError(f'{source}: wrong episode/frame')
        if dat.get('manual_reviewed_prompts') is not True:raise ValueError(f'{source}: not exported from manual annotator')
        wh=first_shape(hoi,ep)
        if wh is None:raise FileNotFoundError(f'e{ep}: missing prepared frame0')
        if (dat.get('image_width'),dat.get('image_height'))!=wh:raise ValueError(f'e{ep}: image dims differ')
        safe_points(dat['objects'],source);check_bounds(dat['objects'],*wh,str(source))
        dat['manual_source_episode']=ep;dat['position_group']=i
        print(f'[INSTALL] e{ep}:',backup_then_write(hoi,ep,dat))


def spread(hoi,eps,groups,include_sources=False):
    for ep in eps:
        i,g=group_for(ep,groups)
        if g is None:
            print(f'[SKIP] e{ep}: outside groups');continue
        src_ep=g['source']
        if ep==src_ep:
            print(f'[SOURCE] e{ep}: preserved');continue
        try:
            ref=get_reference(hoi,src_ep)
            obj=copy.deepcopy(ref['objects']);safe_points(obj,f'e{src_ep}')
            wh=(int(ref['image_width']),int(ref['image_height']))
            actual=first_shape(hoi,ep)
            if actual and actual!=wh:raise RuntimeError(f'dimension mismatch target {actual}, ref {wh}')
            check_bounds(obj,*wh,f'e{ep} copied from e{src_ep}')
            data={'schema_version':2,'episode_index':ep,'frame_index':0,
                  'image_width':wh[0],'image_height':wh[1],
                  'objects':obj,'manual_reviewed_prompts':False,
                  'position_group':i,'manual_source_episode':src_ep,
                  'group_template_points_unverified':True,
                  'warning':'Same-group pixel clicks reused without template matching; must review each SAM2 preview. No geometric truth implied.'}
            result=backup_then_write(hoi,ep,data)
            print(f'[COPY] e{ep:02d} <= e{src_ep:02d} group={i}: {result}'+(' (first frame not prepared yet)' if not actual else ''))
        except Exception as e:
            print(f'[ERROR e{ep:02d}] {e}')
            raise


def status(hoi,groups):
    for i,g in enumerate(groups,1):
        labels=[]
        for ep in range(g['start'],g['end']+1):
            f=jpath(hoi,ep)
            if f.exists():
                x=ensure_json(f);state='manual' if x.get('manual_reviewed_prompts') else 'copy' if x.get('manual_source_episode') is not None else 'auto/old'
                labels.append((ep,state))
        print(f'Group {i}: e{g["start"]}-e{g["end"]} reference=e{g["source"]} count={len(labels)}/{g["end"]-g["start"]+1}')
        print(' ',labels[:12], '...' if len(labels)>12 else '')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd',choices=('annotate','install','spread','status'))
    parser.add_argument('--hoi',type=Path,default=HOI)
    parser.add_argument('--groups',default=GROUPS)
    parser.add_argument('--episodes',default='1-69',help='for spread or annotate, comma-separated episode IDs / intervals')
    parser.add_argument('--inbox',type=Path,default=None)
    parser.add_argument('--no-funnel-hint',action='store_true')
    args=parser.parse_args();groups=groups_parse(args.groups)
    print(f'[START] cmd={args.cmd} hoi={args.hoi}')
    hoi=args.hoi.resolve();inbox=(args.inbox or hoi/'outputs/level1_manual_labels')
    print(f'[PATH] inbox={inbox}')
    # Import the existing episode-ID parser if available, otherwise use our simple equivalent.
    def parse_eps(s):
        out=[]
        for tok in s.split(','):
            if not tok:continue
            if '-' in tok:
                a,b=map(int,tok.split('-',1));out.extend(range(a,b+1))
            else:out.append(int(tok))
        return list(dict.fromkeys(out))
    reps=[g['source'] for g in groups if g['source']!=0]
    if args.cmd=='annotate':
        for ep in (parse_eps(args.episodes) if args.episodes!='1-69' else reps):
            make_annotation_ui(hoi,ep,not args.no_funnel_hint)
    elif args.cmd=='install':install(hoi,reps,inbox,groups)
    elif args.cmd=='spread':spread(hoi,parse_eps(args.episodes),groups)
    else:status(hoi,groups)

if __name__=='__main__':main()
