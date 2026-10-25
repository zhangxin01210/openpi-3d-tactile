#!/usr/bin/env python3
"""Audit whether the existing SAM2 funnel mask can trigger native RGBTrack recovery.

Uses EXACTLY the same mask loader and RGB dataset reader as the project adapter.
Outputs a contact sheet for critical frames and mask occupancy CSV. Read-only.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import rgbtrack_adapter as adapter
except ImportError as exc:
    raise SystemExit("Place alongside rgbtrack_adapter.py") from exc


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=23)
    p.add_argument("--output", type=Path,required=True)
    p.add_argument("--frames", default="0,100,141,148,155,181,203,357,456")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    selected={int(x) for x in args.frames.split(",") if x.strip()}
    cells=[]
    counts=[]
    for frame in adapter.iter_episode_frames(args.dataset,args.episode):
        i=int(frame.frame_index)
        rgb=np.asarray(frame.rgb,dtype=np.uint8)
        mask=adapter.load_mask(args.mask_root,"funnel",i,expected_shape=rgb.shape[:2])
        area=int(mask.sum())
        counts.append((i,area))
        if i not in selected:
            continue
        img=rgb.copy()
        # project RGB to BGR for OpenCV output; blend green SAM2 mask.
        bgr=img[...,::-1].copy()
        bgr[mask]=np.clip(0.55*bgr[mask]+0.45*np.array([40,255,40]),0,255).astype(np.uint8)
        yy,xx=np.where(mask)
        if len(xx):
            x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())
            cv2.rectangle(bgr,(x0,y0),(x1,y1),(0,250,250),1)
        cv2.rectangle(bgr,(0,0),(min(bgr.shape[1]-1,395),34),(10,10,10),-1)
        cv2.putText(bgr,f"frame={i} SAM2 area={area}px  lost_if<=40",(6,23),
                    cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),1,cv2.LINE_AA)
        cells.append((i,cv2.resize(bgr,(480,360))))
    cells.sort(key=lambda x:x[0])
    if not cells:
        raise SystemExit("No requested RGB frames were found")
    cols=3; rows=(len(cells)+cols-1)//cols
    canvas=np.zeros((rows*360,cols*480,3),dtype=np.uint8)
    for index,(_,cell) in enumerate(cells):
        r,c=divmod(index,cols)
        canvas[r*360:(r+1)*360,c*480:(c+1)*480]=cell
    cv2.imwrite(str(args.output/"funnel_sam2_mask_contact_sheet.png"),canvas)
    with (args.output/"mask_areas.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f);w.writerow(["frame_index","mask_area_px","native_lost_if_le_40"])
        w.writerows((i,a,int(a<=40)) for i,a in counts)
    lost=sum(a<=40 for _,a in counts)
    print(f"[MASK AUDIT] {len(counts)} frames; area<=40: {lost} frames",flush=True)
    print("[MASK AUDIT] selected:",[(i,a) for i,a in counts if i in selected],flush=True)
    print("[MASK AUDIT]",args.output/"funnel_sam2_mask_contact_sheet.png",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
