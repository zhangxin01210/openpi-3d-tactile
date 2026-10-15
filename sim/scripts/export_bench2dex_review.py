#!/usr/bin/env python3
"""Export synchronized replay RGB/depth/TacMap video and chest point clouds."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np

from unproject_univtac_depth import write_binary_ply


def label(frame, text, xy=(12,28)):
    cv2.putText(frame,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)


def geometry(group, row, table_z, output):
    d=group['depth'][row]
    k=group['intrinsic'][:]
    transform=group['extrinsic_world_from_cam'][row]
    rgb=cv2.imdecode(group['rgb'][row],cv2.IMREAD_COLOR)[...,::-1]
    v,u=np.indices(d.shape)
    valid=np.isfinite(d)&(d>.01)&(d<3)
    stats={}
    for offset in [0.,.5]:
        z=d[valid]
        x=(u[valid]+offset-k[0,2])*z/k[0,0]
        y=(v[valid]+offset-k[1,2])*z/k[1,1]
        # Exported extrinsics use +X forward, +Y left, +Z up.
        camera_body=np.column_stack([z,-x,-y])
        world=camera_body@transform[:3,:3].T+transform[:3,3]
        near_table=(np.abs(world[:,2]-table_z)<.015)&(np.abs(world[:,0])<.7)&(np.abs(world[:,1])<.45)
        stats[str(offset)]={"near_table_count":int(near_table.sum()),
            "median_table_height_residual_m":float(np.median(world[near_table,2]-table_z)),
            "median_absolute_table_height_residual_m":float(np.median(np.abs(world[near_table,2]-table_z)))}
        if offset==.5:
            write_binary_ply(str(output),world[::4],rgb[valid][::4])
    return {"row":row,"table_z_m":table_z,"pixel_offset_checks":stats,
            "scope":"Chest pinhole camera only; near-table consistency check, not full metric calibration"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.replay,'r') as f:
        n=int(f['meta/frame_count'][()]);fps=int(f['meta/fps'][()])
        homing=f['time/sim_step'][:] >= int(f['meta/homing_start_sim_step'][()])
        cameras=[f['cameras/cam_chest'],f['cameras/cam_wrist_right']]
        tactile=f['robot/tactile/tacmap']
        sites=sorted(tactile.keys())
        video=args.output/'review.mp4'
        command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x960',
                 '-r',str(fps),'-i','-','-an','-c:v','av1_nvenc','-preset','p4','-cq','30','-b:v','0',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        snapshots=set(np.linspace(0,n-1,8,dtype=int));story=[]
        for row in range(n):
            panel=np.zeros((960,1280,3),np.uint8)
            for column,cam in enumerate(cameras):
                rgb=cv2.imdecode(cam['rgb'][row],cv2.IMREAD_COLOR)
                panel[:480,column*640:(column+1)*640]=rgb
            depth=cameras[0]['depth'][row]
            scaled=np.uint8(np.clip(np.nan_to_num(depth,nan=0,posinf=0)/2.,0,1)*255)
            panel[480:,:640]=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO)
            for i,site in enumerate(sites):
                x=640+(i%5)*128;y=540+(i//5)*180
                img=cv2.resize(tactile[site][row],(120,120),interpolation=cv2.INTER_NEAREST)
                panel[y:y+120,x:x+120]=cv2.applyColorMap(img,cv2.COLORMAP_INFERNO)
                name=site.replace('_elastomer','').replace('left_','L ').replace('right_','R ')
                label(panel,name,(x+2,y+140))
            label(panel,f'Chest RGB | frame {row}/{n-1}');label(panel,'Wrist RGB (fisheye)',(652,28))
            label(panel,'Chest depth: 0..2 m',(12,510));label(panel,'TacMap: quantized penetration, 0..255',(652,510))
            label(panel,'State replay; not a policy rollout',(652,940))
            if homing[row]:
                label(panel,'Return-home segment: exclude from training',(12,462))
            process.stdin.write(panel.tobytes())
            if row in snapshots:
                cv2.imwrite(str(args.output/f'frame_{row:04d}.jpg'),panel)
                story.append(cv2.resize(panel,(640,480)))
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed')
        cv2.imwrite(str(args.output/'storyboard.jpg'),np.vstack([np.hstack(story[i:i+2]) for i in range(0,len(story),2)]))
        sample=json.loads(f['meta/scene_generalization_sample'][()].decode())
        table_z=.75+sample['spatial']['table_height']['height_offset_m']
        checks=[geometry(cameras[0],row,table_z,args.output/f'cloud_{row:04d}.ply') for row in [0,n//2,n-1]]
        (args.output/'geometry.json').write_text(json.dumps(checks,indent=2)+'\n')
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    print(json.dumps(dict(video=str(video),frames=n,fps=fps,full_decode_passed=True)))


if __name__=='__main__':
    main()
