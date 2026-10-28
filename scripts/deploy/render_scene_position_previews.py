#!/usr/bin/env python3
"""Render PREVIEW ONLY: original reference frame 0 with human SAM2 seed clicks.

The reference RGB is read from the *training dataset* solely to show which
physical arrangement each --scene-position corresponds to. The WebSocket
server NEVER reads these reference images for inference: it applies only
the selected position's 2D positive/negative points to CURRENT live RGB f0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
from scripts.deploy.position_initialization import SCENE_POSITION_TO_EPISODE, resolve_prompts

POSITIONS=tuple(SCENE_POSITION_TO_EPISODE)
BGR_COLORS={"funnel":(0,158,255), "target_flask":(245,150,45)}


def draw_reference(rgb: np.ndarray, prompt: dict, position: int) -> np.ndarray:
    image=np.asarray(rgb)
    if image.shape!=(480,640,3) or image.dtype!=np.uint8:
        raise ValueError(f"Reference image for position {position} must be RGB uint8 480x640")
    canvas=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    h,w=canvas.shape[:2]
    if (prompt.get("image_width"),prompt.get("image_height"))!=(w,h):
        raise ValueError(f"Position {position}: reviewed clicks and reference image resolution mismatch")
    for name in ("funnel","target_flask"):
        if name not in prompt["objects"]:
            raise KeyError(f"Missing {name} keypoints in position {position}")
        color=BGR_COLORS[name]
        for kind,points in prompt["objects"][name].items():
            if kind not in ("positive","negative"):
                continue
            for x,y in points:
                px,py=int(round(x)),int(round(y))
                if not (0<=px<w and 0<=py<h):
                    raise ValueError(f"Out-of-bounds point in {name}.{kind}: {px},{py}")
                cv2.circle(canvas,(px,py),8,(0,0,0),-1,cv2.LINE_AA)
                if kind=="positive":
                    cv2.circle(canvas,(px,py),6,color,-1,cv2.LINE_AA)
                    cv2.circle(canvas,(px,py),2,(255,255,255),-1,cv2.LINE_AA)
                else:
                    cv2.line(canvas,(px-6,py-6),(px+6,py+6),color,3,cv2.LINE_AA)
                    cv2.line(canvas,(px-6,py+6),(px+6,py-6),color,3,cv2.LINE_AA)
    header=np.zeros((80,w,3),dtype=np.uint8)
    episode=SCENE_POSITION_TO_EPISODE[position]
    label=f"--scene-position {position}   (ref E{episode})"
    cv2.putText(header,label,(12,31),
                cv2.FONT_HERSHEY_SIMPLEX,.89,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(header,"Funnel: orange   Flask: blue   + filled   - X mark",
                (12,58),cv2.FONT_HERSHEY_SIMPLEX,.54,(225,225,225),1,cv2.LINE_AA)
    return np.concatenate([header,canvas],axis=0)


def generate(*,dataset:Path,output:Path,reader_factory,
             positions=POSITIONS)->dict:
    """Dataset is accessed only HERE, never in the live 8990 server."""
    output=Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preview output: {output}")
    output.mkdir(parents=True,exist_ok=False)
    panels=[]
    meta=[]
    for position in positions:
        if position not in SCENE_POSITION_TO_EPISODE:
            raise ValueError(f"Unknown operator scene position {position}")
        episode=SCENE_POSITION_TO_EPISODE[position]
        ds=reader_factory(dataset,episode=episode)
        image=next(ds.iter_video_frames("front",selected={0})).rgb
        prompts=resolve_prompts(
            repo_root=ROOT,prompt_file=None,template_episode=episode,
            template_confirmed=True,first_frame_rgb_shape=image.shape,
        )
        payload={"image_width":image.shape[1],"image_height":image.shape[0],
                 "objects":prompts["objects"]}
        panel=draw_reference(image,payload,position)
        dst=output/f"scene_position_{position}_reference_points.png"
        if not cv2.imwrite(str(dst),panel):
            raise OSError(f"Cannot write {dst}")
        panels.append(panel)
        meta.append({
            "position":position,
            "reference_training_episode":episode,
            "preview":dst.name,
            "source_training_image_role":"front frame 0 (OPERATOR PREVIEW ONLY)",
            "clicks":prompts["initialization_provenance"],
            "funnel_reference_episode":23 if position==1 else episode,
            "flask_reference_episode":episode,
            "used_by_live_inference":"2D points ONLY; NOT the reference image or its 6D pose",
        })
        print(f"[PREVIEW] --scene-position {position}: {dst}",flush=True)
    height,width=panels[0].shape[:2]
    blank=np.full((height,width,3),35,dtype=np.uint8)
    arranged=panels+[blank]*(6-len(panels))
    overview=np.vstack([
        np.hstack(arranged[:3]),np.hstack(arranged[3:6]),
    ])
    overview_path=output/"scene_positions_overview.png"
    if not cv2.imwrite(str(overview_path),overview):
        raise OSError(f"Cannot write {overview_path}")
    manifest={
        "positions":meta,
        "overview":overview_path.name,
        "reference_images_only_for_operator_identification":True,
        "live_sam2_runs_on_current_live_camera_frame_zero":True,
        "live_flask_rgbtrack_reestimates_current_frame_zero":True,
    }
    (output/"scene_positions_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"[PREVIEW] saved overview: {overview_path}",flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True,help="Original LeRobot dataset root on sai")
    p.add_argument("--output-dir",type=Path,
                   default=ROOT/"output/deploy/scene_position_previews_v1")
    args=p.parse_args()
    if str(ROOT/"src") not in sys.path:
        sys.path.insert(0,str(ROOT/"src"))
    from openpi.spatial_dataset.source import RawSpatialDataset
    generate(dataset=args.dataset.expanduser().resolve(),output=args.output_dir,
             reader_factory=RawSpatialDataset)


if __name__=="__main__":
    main()
