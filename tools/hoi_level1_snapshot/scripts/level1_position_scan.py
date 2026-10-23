#!/usr/bin/env python3
"""Fast position audit for Level-1: only the first RGB frame of each episode.

Scan target flask template from reviewed episode 0, cluster its *image* positions,
propose representative episodes. Position groups are hypotheses, not verified labels.
No masks, SAM2, RGB-D, calibration, or training data are modified.

Run in openpi-3d-tactile:
  PYTHONPATH=src .venv/bin/python /workspace/mnt/sqzhang26/hoi/scripts/level1_position_scan.py --episodes 0-69
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

DEFAULT_HOI = Path('/workspace/mnt/sqzhang26/hoi')
DEFAULT_REPO = Path('/workspace/mnt/sqzhang26/openpi-3d-tactile')


def episode_ids(spec: str) -> list[int]:
    ids = set()
    for part in spec.replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            lo, hi = (int(x) for x in part.split('-', 1))
            if lo < 0 or hi < lo:
                raise ValueError(f'invalid episode range {part}')
            ids.update(range(lo, hi + 1))
        else:
            x = int(part)
            if x < 0:
                raise ValueError('episode must be nonnegative')
            ids.add(x)
    if not ids:
        raise ValueError('no episodes selected')
    return sorted(ids)


def first_frame(ep: int, *, hoi: Path, dataset: Path):
    import cv2
    # Prepared episodes already have frame 0; other episodes only decode ONE frame.
    existing = hoi / f'data/level1_sequence_e{ep}/frames/00000.jpg'
    if existing.is_file():
        frame = cv2.imread(str(existing))
        if frame is None:
            raise ValueError(f'Failed to read {existing}')
        return frame, 'prepared_frame0'
    from openpi.spatial_dataset.source import RawSpatialDataset
    ds = RawSpatialDataset(dataset, episode=ep)
    reader = ds.iter_video_frames('front', selected=[0])
    try:
        rgb = next(reader).rgb
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        reader.close()  # Important: do NOT decode the rest of the video!
    return frame, 'decoded_video_frame0'


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        raise ValueError('Empty estimated mask')
    return float(np.median(xx)), float(np.median(yy))


def group_positions(rows: list[dict], radius: float) -> list[dict]:
    """Greedy complete-link groups; avoids DBSCAN chaining nearby positions.

    Input contains only qualified flask matches. Deterministic ordering by score,
    then episode ID. A 'group' is NOT an actual station calibration.
    """
    groups: list[dict] = []
    valid = sorted((r for r in rows if r['qualified']),
                   key=lambda r: (-r['score'], r['episode']))
    for row in valid:
        xy = np.array([row['x_px'], row['y_px']], dtype=float)
        # Every member must be close, not merely group mean.
        options = []
        for j, g in enumerate(groups):
            d = [np.linalg.norm(xy - [a['x_px'], a['y_px']]) for a in g['members']]
            if all(z <= radius for z in d):
                options.append((max(d), j))
        if options:
            groups[min(options)[1]]['members'].append(row)
        else:
            groups.append({'members': [row]})
    for g in groups:
        xy = np.array([[m['x_px'], m['y_px']] for m in g['members']])
        mid = np.median(xy, axis=0)
        g['center_px'] = [round(float(v), 1) for v in mid]
        g['members'].sort(key=lambda r: (np.linalg.norm(np.array([r['x_px'], r['y_px']]) - mid), -r['score'], r['episode']))
    groups.sort(key=lambda g: (g['center_px'][0], g['center_px'][1]))
    for i, g in enumerate(groups, 1):
        g['group_id'] = i
        for row in g['members']:
            row['group_id'] = i
    return groups


def draw_tile(img: np.ndarray, row: dict, *, width=420, height=320) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    sx, sy = width/w, height/h
    thumb = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    c = (25, 220, 25) if row['qualified'] else (35, 40, 230)
    # Mark *estimated* flask position only, not a verified contour.
    xy = (int(round(row['x_px']*sx)), int(round(row['y_px']*sy)))
    cv2.drawMarker(thumb, xy, c, markerType=cv2.MARKER_CROSS, markerSize=17, thickness=3)
    cv2.circle(thumb, xy, 11, c, 2)
    cv2.rectangle(thumb, (0,0), (width-1,32), (22,22,22), -1)
    label = (f"e{row['episode']}  G{row.get('group_id','?')}  "
             f"s={row['score']:.3f}  x={row['x_px']:.0f} y={row['y_px']:.0f}")
    cv2.putText(thumb, label, (7,22), cv2.FONT_HERSHEY_SIMPLEX,.59,(255,255,255),1,cv2.LINE_AA)
    return thumb


def make_sheet(thumbs, cols, path):
    import cv2
    if not thumbs:
        return
    H,W=thumbs[0].shape[:2]
    nrows=(len(thumbs)+cols-1)//cols
    sheet = np.full((H*nrows, W*cols, 3), 242, np.uint8)
    for i,img in enumerate(thumbs):
        sheet[(i//cols)*H:(i//cols+1)*H, (i%cols)*W:(i%cols+1)*W] = img
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),sheet):
        raise RuntimeError(f'Failed writing {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', default='0-69', help='e.g. 0-69 or 0,2,5-8')
    p.add_argument('--hoi', type=Path, default=DEFAULT_HOI)
    p.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    p.add_argument('--dataset', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--radius-px', type=float, default=28.,
                   help='Max pairwise distance within same image-position group')
    p.add_argument('--expected-groups', type=int, default=6)
    p.add_argument('--crop-margin', type=float, default=.25,
                   help='Crop margin relative to Mask bbox (template matcher v2 used .8; .25 is less background-sensitive)')
    p.add_argument('--min-score', type=float, default=.68)
    p.add_argument('--min-margin', type=float, default=.035)
    p.add_argument('--reference-episode', type=int, default=0,
                   help='Known manually approved initial flask template')
    args=p.parse_args()
    if args.radius_px<=0 or args.expected_groups<=0:
        p.error('radius and expected-groups must be positive')
    hoi=args.hoi.resolve(); repo=args.repo.resolve()
    dataset=(args.dataset or repo/'data/1018_peg_all').resolve()
    out=(args.out or hoi/'outputs/level1_position_scan').resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(hoi/'scripts'))
    import cv2
    from level1_template_init import load_templates, match_image, translated_mask

    refs = [r for r in load_templates(hoi,skip_episode=-1)
            if r['episode']==args.reference_episode and r['name']=='target_flask']
    if not refs:
        raise RuntimeError('Missing reference episode 0 first RGB and confirmed mask: '
                           'hoi/data/level1_sequence_e0/frames/00000.jpg and '
                           'hoi/outputs/level1_sequence_e0/masks/target_flask/00000.png')
    ref=refs[0]
    rows=[]; thumb_by_ep={}; errors=[]
    for ep in episode_ids(args.episodes):
        try:
            img,source=first_frame(ep,hoi=hoi,dataset=dataset)
            # Episode 0 uses the original mask position, no matching to itself needed.
            if ep==args.reference_episode:
                cx,cy=mask_center(ref['mask'])
                mask=ref['mask'].astype('uint8')*255
                score,margin=1.,1.
            else:
                match=match_image(ref['rgb'],ref['mask'],img,
                    scales=[.75,.85,.95,1.0,1.05,1.15,1.30],margin_scale=args.crop_margin)
                mask=translated_mask(ref,match,img.shape[:2])
                cx,cy=mask_center(mask>127)
                score,margin=match['score'],match['margin']
            qualified=(score>=args.min_score and margin>=args.min_margin)
            row={'episode':ep,'x_px':round(cx,2),'y_px':round(cy,2),
                 'score':round(score,4),'margin':round(margin,4),
                 'qualified':bool(qualified),'source':source,'group_id':None}
            rows.append(row)
            thumb_by_ep[ep]=img
            print(f'[SCAN] e{ep:02d} ({cx:5.1f},{cy:5.1f}) score={score:.3f} '
                  f'margin={margin:.3f} valid={qualified}',flush=True)
        except Exception as exc:
            errors.append({'episode':ep,'error':repr(exc)})
            print(f'[FAIL] e{ep}: {exc}',flush=True)
    groups=group_positions(rows,args.radius_px)
    unqualified=[r for r in rows if not r['qualified']]
    for g in groups:
        mem=g['members']; print(f"[GROUP {g['group_id']}] n={len(mem)} "
                    f"center={g['center_px']} samples={[r['episode'] for r in mem[:4]]}",flush=True)
    print('[RESULT]',len(groups),'position groups,',len(unqualified),'uncertain matches,',len(errors),'load failures')
    if len(groups)!=args.expected_groups:
        print(f'[REVIEW] Expected {args.expected_groups}, found {len(groups)}. '
              'Review montage: grouping is provisional; use --radius-px to test sensitivity.')
    if unqualified:
        print('[REVIEW] Uncertain episodes:',[r['episode'] for r in unqualified])
    # Prefer one representative per actual position, not one per episode number.
    suggestions=[]
    for g in groups:
        # For the reference location, choose another episode to test transfer.
        candidates=[r for r in g['members'] if r['episode']!=args.reference_episode]
        choice=(candidates or g['members'])[0]
        suggestions.append({'group':g['group_id'],'episode':choice['episode'],
                            'n_episodes':len(g['members']),'center_px':g['center_px'],
                            'score':choice['score']})
    with (out/'episode_positions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['episode','group_id','x_px','y_px','score','margin','qualified','source'])
        writer.writeheader();writer.writerows(sorted(rows,key=lambda r:r['episode']))
    result={'reference_episode':args.reference_episode,'episodes_requested':args.episodes,
            'radius_px':args.radius_px,'crop_margin':args.crop_margin,'expected_groups':args.expected_groups,
            'groups':[{'group':g['group_id'],'center_px':g['center_px'],
                       'episode_ids':sorted(r['episode'] for r in g['members']),
                       'count':len(g['members'])} for g in groups],
            'proposed_representatives':suggestions,
            'uncertain_matches':[r['episode'] for r in unqualified],
            'errors':errors,
            'warning':'Image-location clusters based on e0 RGB template; NOT verified flask pose, '
                      'mask quality or six true robot base coordinates. Manual inspect montage.'}
    (out/'position_groups.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    ranked=[r for g in groups for r in g['members']]
    ranked+=unqualified
    make_sheet([draw_tile(thumb_by_ep[r['episode']],r,width=270,height=205) for r in ranked],
               5,out/'all_episodes_montage.jpg')
    make_sheet([draw_tile(thumb_by_ep[s['episode']],
                          next(r for r in rows if r['episode']==s['episode']))
                for s in suggestions],2,out/'representatives.jpg')
    print('[OUTPUT]',out/'position_groups.json',out/'representatives.jpg',out/'all_episodes_montage.jpg')
    print('[PICK] representative episodes:',[s['episode'] for s in suggestions])

if __name__=='__main__':
    main()
