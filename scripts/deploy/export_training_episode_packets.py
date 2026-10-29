#!/usr/bin/env python3
"""Export SAME LeRobot episode input (RGB/state/native depth) as online perception packets.

Only the existing training reader and existing training RGB registration
are used. Never writes to the source dataset or commands a robot.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def export_episode(*,dataset_root:Path,episode:int,max_frames:int,output_dir:Path,
                   camera,register_depth,reader_class):
    if max_frames<1:
        raise ValueError("max_frames must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Episode packet output must be a new path: {output_dir}")
    ds=reader_class(dataset_root,episode=episode)
    required_roles=("front","left","right")
    if not set(required_roles).issubset(ds.camera_roles):
        raise ValueError(f"Missing training RGB roles: {set(required_roles)-set(ds.camera_roles)}")
    if "front" not in ds.depth_roles:
        raise ValueError("Training dataset is missing original front depth")
    selected=set(range(max_frames))
    streams={
        role: ds.iter_video_frames(role,selected=selected)
        for role in required_roles
    }
    rows=ds.iter_rows(selected=selected,depth_roles=("front",))
    output_dir.mkdir(parents=True,exist_ok=False)
    written=0
    for row,front,left,right in zip(rows,streams["front"],streams["left"],streams["right"],strict=True):
        frame=int(row["frame_index"])
        if frame!=written or any(int(v.frame_index)!=frame for v in (front,left,right)):
            raise ValueError(f"LeRobot video/parquet alignment mismatch at frame={frame}")
        state=np.asarray(row["observation.state"],dtype=np.float32)
        if state.shape!=(1972,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid original robot state at frame {frame}: {state.shape}")
        native=np.asarray(row["observation.depths.cam_front"])
        if native.ndim!=2 or native.dtype.kind not in "iuf" or not np.isfinite(native).all():
            raise ValueError(f"Invalid original front depth at frame {frame}")
        color=front.rgb
        aligned_m=register_depth(native,camera,color.shape[:2])
        scale=float(camera.depth_scale_m_per_unit)
        if not np.isfinite(scale) or scale<=0:
            raise ValueError("Training depth scale invalid")
        aligned=(np.asarray(aligned_m,dtype=np.float32)/np.float32(scale)).astype(np.float32)
        if aligned.shape!=color.shape[:2] or not np.isfinite(aligned).all():
            raise ValueError(f"Invalid shared registration at frame {frame}")
        packet={
            "frame_index":np.asarray(frame,dtype=np.int64),
            "episode_index":np.asarray(episode,dtype=np.int64),
            "timestamp_s":np.asarray(float(row["timestamp"]),dtype=np.float64),
            "robot_state":state,
            "front_rgb":front.rgb,
            "left_rgb":left.rgb,
            "right_rgb":right.rgb,
            "front_depth_native":native,
            "front_depth_color_aligned":aligned,
        }
        from scripts.deploy.run_online_perception import _validate_packet
        _validate_packet(output_dir/f"frame_{frame:06d}.npz",packet)
        dest=output_dir/f"frame_{frame:06d}.npz"
        tmp=dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("xb") as f:
                np.savez_compressed(f,**packet)
            os.replace(tmp,dest)
        finally:
            tmp.unlink(missing_ok=True)
        written+=1
        if written%8==0:
            print(f"[TRAIN REPLAY] packed {written}/{max_frames} frames",flush=True)
    if written!=max_frames:
        raise ValueError(f"Only {written} of {max_frames} frames exported")
    print(f"[TRAIN REPLAY] ready: {written} frames, {output_dir}",flush=True)
    return written


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--episode",type=int,default=23)
    p.add_argument("--max-frames",type=int,default=32)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--front-calibration-profile",type=Path,
                   default=ROOT/"configs/ur7e_xhand/front_calibration_candidate.json")
    args=p.parse_args()
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/"src"))
    sys.path.insert(0,str(ROOT/"scripts/object_geometry/pose_tracking"))
    from openpi.spatial_dataset.source import RawSpatialDataset
    from rgbtrack_adapter import _rgb_registered_depth,_load_front_camera_for_depth_registration
    cam=_load_front_camera_for_depth_registration(ROOT,args.front_calibration_profile)
    export_episode(
        dataset_root=args.dataset.expanduser().resolve(),episode=args.episode,
        max_frames=args.max_frames,output_dir=args.output_dir.expanduser().resolve(),
        camera=cam,register_depth=_rgb_registered_depth,reader_class=RawSpatialDataset,
    )


if __name__=="__main__":
    main()
