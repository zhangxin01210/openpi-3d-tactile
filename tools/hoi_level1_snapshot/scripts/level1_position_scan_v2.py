#!/usr/bin/env python3
"""Scan up to 70 robot episodes by a *moving bright fixture proxy*, not a transparent-glass mask.

The right/lower white fixture is used only to locate station configurations.
NEVER treat its pixels as flask geometry, contact truth, or object segmentation.
No SAM2, depth, training files, or manually reviewed masks are modified.

Usage from openpi-3d-tactile:
  PYTHONPATH=src .venv/bin/python /workspace/mnt/sqzhang26/hoi/scripts/level1_position_scan_v2.py --episodes 0-69

If the automatic proxy detector fails to yield exactly six visually correct groups,
use the visible chronological ranges as provisional manual group IDs:
  ... --ranges 0-15,16-30,31-45,46-53,54-61,62-69
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

DEFAULT_HOI=Path('/workspace/mnt/sqzhang26/hoi')
DEFAULT_REPO=Path('/workspace/mnt/sqzhang26/openpi-3d-tactile')


def parse_ranges(spec):
    from level1_position_scan import episode_ids
    bands=[];seen=set()
    for b in spec.split(','):
        if not b:continue
        ids=episode_ids(b)
        if set(ids)&seen:raise ValueError('Overlapping position ranges')
        seen.update(ids);bands.append(ids)
    return bands


def locate_fixture(img, *, min_area_fraction=.0005, debug=False):
    """Find bright moving bottle support in right-lower tabletop, returning *proxy* center.

    This deliberately does not inspect any mask or attempt to detect transparent glass.
    """
    import cv2
    H,W=img.shape[:2]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    sat=hsv[:,:,1];value=hsv[:,:,2]
    # Generous whites to tolerate changing exposure; never use the entire image.
    binary=((value>=183)&(sat<=83)).astype(np.uint8)*255
    # Restrict to moving support's table zone, exclude upper arm and the far-right static base.
    xlo=int(W*.36); xhi=int(W*.865)
    ylo=int(H*.575)
    binary[:ylo,:]=0;binary[:,:xlo]=0;binary[:,xhi:]=0
    # A short closing operation joins fragmented white top/rim of the mount.
    k=max(3,round(min(H,W)*.016))
    if k%2==0:k+=1
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    choices=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<min_area_fraction*H*W:continue
        if h<=6 or w<=6:continue
        cx,cy=map(float,centers[i])
        # Exclude regions glued to far image edge (static bright background).
        if cx>=W*.85:continue
        # The target fixture is compact; very thin highlights can be ignored.
        if w/h>7 or h/w>7:continue
        choices.append({'center_px':[round(cx,2),round(cy,2)],'area':area,
                        'bbox_xywh':[x,y,w,h],'fill':round(area/(w*h),3)})
    choices.sort(key=lambda c:c['area'],reverse=True)
    if not choices:return None,binary
    chosen=choices[0]
    second=choices[1]['area'] if len(choices)>1 else 0
    chosen['area_ratio_to_runner']=round(chosen['area']/max(1,second),2)
    chosen['support_area_fraction']=round(chosen['area']/(H*W),4)
    return chosen,binary


def group_auto(rows,radius):
    """Cluster all located centers, ignoring the previous glass-match margin entirely."""
    clusters=[]
    for row in sorted((r for r in rows if r['detected']),key=lambda r:r['episode']):
        xy=np.array([row['x_px'],row['y_px']])
        best=None
        for i,g in enumerate(clusters):
            pts=np.array([[r['x_px'],r['y_px']] for r in g])
            dist=np.linalg.norm(pts-xy,axis=1)
            if len(dist) and dist.max()<=radius and (best is None or dist.mean()<best[0]):
                best=(dist.mean(),i)
        if best is None:clusters.append([row])
        else:clusters[best[1]].append(row)
    clusters.sort(key=lambda g:min(r['episode'] for r in g))
    return clusters


def group_explicit(rows,spec):
    bands=parse_ranges(spec);byep={r['episode']:r for r in rows}
    clusters=[]
    for ids in bands:
        group=[byep[i] for i in ids if i in byep]
        if group:clusters.append(group)
    return clusters


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',default='0-69')
    p.add_argument('--hoi',type=Path,default=DEFAULT_HOI)
    p.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    p.add_argument('--dataset',type=Path,default=None)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--radius-px',type=float,default=17.,help='Max intra-group centroid distance (pixels)')
    p.add_argument('--expected-groups',type=int,default=6)
    p.add_argument('--ranges',type=str,default='',help='Optional visually reviewed contiguous position ranges; e.g. 0-15,16-30,...')
    args=p.parse_args()
    hoi=args.hoi.resolve();repo=args.repo.resolve();dataset=(args.dataset or repo/'data/1018_peg_all').resolve()
    out=(args.out or hoi/'outputs/level1_position_scan_v2').resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(hoi/'scripts'))
    from level1_position_scan import first_frame, episode_ids, make_sheet
    import cv2
    rows=[];frames={};errors=[]
    for ep in episode_ids(args.episodes):
        try:
            img,src=first_frame(ep,hoi=hoi,dataset=dataset)
            mount,_=locate_fixture(img)
            if mount:
                x,y=mount['center_px']
                row={'episode':ep,'detected':True,'x_px':x,'y_px':y,
                    'support_area_fraction':mount['support_area_fraction'],
                    'area_ratio_to_runner':mount['area_ratio_to_runner'],
                    'source':src,'group_id':None}
                print(f'[PROXY] e{ep:02d} fixture=({x:.1f},{y:.1f}) '
                      f'area={mount["support_area_fraction"]:.3f} runner_ratio={mount["area_ratio_to_runner"]:.2f}',flush=True)
            else:
                row={'episode':ep,'detected':False,'x_px':None,'y_px':None,
                     'support_area_fraction':0,'area_ratio_to_runner':0,'source':src,'group_id':None}
                print(f'[NO PROXY] e{ep:02d}',flush=True)
            rows.append(row);frames[ep]=img
        except Exception as e:
            print(f'[FAIL] e{ep:02d}: {e}',flush=True)
            errors.append({'episode':ep,'error':repr(e)})
    if not rows:
        raise RuntimeError("No episode first frames could be read; check dataset and prepared JPEG paths")
    groups=group_explicit(rows,args.ranges) if args.ranges else group_auto(rows,args.radius_px)
    representatives=[];group_descriptions=[]
    for idx,group in enumerate(groups,1):
        for row in group:row['group_id']=idx
        detected=[r for r in group if r['detected']]
        if detected:
            pts=np.array([[r['x_px'],r['y_px']] for r in detected]);mid=np.median(pts,axis=0)
            choice=min(detected,key=lambda r:(np.linalg.norm(np.array([r['x_px'],r['y_px']])-mid),abs(r['episode']-np.median([a['episode'] for a in group]))))
            center=[round(float(x),2) for x in mid]
        else:
            choice=group[len(group)//2];center=None
        episodes=sorted(r['episode'] for r in group)
        desc={'group_id':idx,'episodes':episodes,'n':len(episodes),'proxy_center_px':center,
              'representative':choice['episode']}
        group_descriptions.append(desc)
        representatives.append(choice['episode'])
        print(f'[GROUP {idx}] n={len(episodes)} range={episodes[0]}-{episodes[-1]} '
              f'fixture_center={center} representative=e{choice["episode"]}',flush=True)
    ungrouped=sorted(r['episode'] for r in rows if r['group_id'] is None)
    if ungrouped:print('[UNGROUPED]',ungrouped,flush=True)
    if len(groups)!=args.expected_groups:
        print(f'[REVIEW] Found {len(groups)} groups, expected {args.expected_groups}. '
              'Do not force the output to six by lowering matching margins. '
              'Use visually verified --ranges if automatic fixture grouping fails.',flush=True)
    with (out/'episode_positions.csv').open('w',encoding='utf8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
    result={'method':'moving_bright_fixture_proxy','object_surface':'UNVERIFIED / NOT A FLASK MASK',
            'mode':'reviewed_episode_ranges' if args.ranges else 'automatic_fixture_proxy',
            'grouping_is_provisional':True,'expected_groups':args.expected_groups,
            'groups':group_descriptions,'representative_episodes':representatives,
            'ungrouped_episodes':ungrouped,'errors':errors,
            'warning':'Bright moving fixture is a location proxy, not transparent flask geometry; check mask manually.'}
    (out/'position_groups.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    def tile(ep):
        img=frames[ep];r=next(x for x in rows if x['episode']==ep)
        H,W=img.shape[:2];view=cv2.resize(img,(400,300),interpolation=cv2.INTER_AREA)
        if r['detected']:
            px=int(r['x_px']*400/W);py=int(r['y_px']*300/H)
            cv2.drawMarker(view,(px,py),(0,200,255),cv2.MARKER_CROSS,21,2)
        cv2.rectangle(view,(0,0),(399,32),(30,30,30),-1)
        caption=f'e{ep:02d} group={r["group_id"]} fixture_proxy'
        cv2.putText(view,caption,(9,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        return view
    make_sheet([tile(ep) for ep in representatives],3,out/'representatives.jpg')
    make_sheet([tile(ep) for ep in sorted(frames)],5,out/'all_episodes_montage.jpg')
    print('[OUTPUT]',out/'position_groups.json',out/'representatives.jpg',out/'episode_positions.csv',flush=True)
    print('[PICK]',representatives,flush=True)

if __name__=='__main__':main()
