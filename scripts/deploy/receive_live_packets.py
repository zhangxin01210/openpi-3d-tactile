#!/usr/bin/env python3
"""Receive raw LeRobot frames, reuse training depth registration, write NPZ atomically.

This is a SENSOR-ONLY bridge. It never imports robot SDK, Pi0 policy, or
sends actuation requests. Depth source is exactly the live LeRobot
observation.depths.cam_front, the field recorded in the training dataset.
Registration uses rgbtrack_adapter._rgb_registered_depth, NOT a new algorithm.
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MAX_POST_BYTES=12 * 1024 * 1024
RAW_KEYS=("front_rgb","left_rgb","right_rgb","front_depth_native","robot_state","timestamp_s","frame_index")


def construct_training_packet(raw, *, register_depth, depth_scale):
    missing=[k for k in RAW_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Missing original LeRobot fields {missing}")
    packet={key:np.asarray(raw[key]) for key in RAW_KEYS}
    for key in ("front_rgb","left_rgb","right_rgb"):
        arr=packet[key]
        if arr.dtype!=np.uint8 or arr.ndim!=3 or arr.shape[2]!=3:
            raise ValueError(f"{key}: expected RGB uint8 HxWx3")
        if arr.shape[:2]!=(480,640):
            raise ValueError(f"{key}: expected recorded 480x640 camera resolution")
    native=packet["front_depth_native"]
    if native.shape!=(480,640) or native.dtype!=np.uint16:
        raise ValueError("Front native depth must match training LeRobot uint16 480x640 field")
    if packet["robot_state"].shape!=(1972,) or not np.isfinite(packet["robot_state"]).all():
        raise ValueError("Training state must be finite 1972 vector")
    packet["robot_state"]=packet["robot_state"].astype(np.float32,copy=False)
    if any(np.asarray(packet[key]).shape!=() for key in ("timestamp_s","frame_index")):
        raise ValueError("Frame index and timestamp must be scalars")
    frame=int(packet["frame_index"])
    timestamp=float(packet["timestamp_s"])
    if frame<0 or not np.isfinite(timestamp):
        raise ValueError("Invalid frame index or timestamp")
    if not np.isfinite(depth_scale) or depth_scale<=0:
        raise ValueError("Invalid training depth scale")
    registered_m=np.asarray(register_depth(native,packet["front_rgb"].shape[:2]),dtype=np.float32)
    if registered_m.shape!=packet["front_rgb"].shape[:2] or not np.isfinite(registered_m).all() or (registered_m<0).any():
        raise ValueError("Shared RGBTrack depth registration returned an invalid image")
    packet["front_depth_color_aligned"]=(registered_m / np.float32(depth_scale)).astype(np.float32)
    packet["frame_index"]=np.asarray(frame,dtype=np.int64)
    packet["timestamp_s"]=np.asarray(timestamp,dtype=np.float64)
    from scripts.deploy.run_online_perception import _validate_packet
    return _validate_packet(Path(f"frame_{frame:06d}.npz"),packet)


class PacketReceiver:
    def __init__(self,output_dir:Path,*,register_depth,depth_scale:float,max_frames:int=0):
        self.output_dir=output_dir
        self.register_depth=register_depth
        self.depth_scale=depth_scale
        self.max_frames=max_frames
        self.count=0
        self.previous_index=-1
        self.previous_timestamp=float("-inf")
        if self.output_dir.exists():
            raise FileExistsError(f"Receiver output must be new: {self.output_dir}")
        self.output_dir.mkdir(parents=True,exist_ok=False)

    def accept(self,raw):
        packet=construct_training_packet(raw,register_depth=self.register_depth,depth_scale=self.depth_scale)
        frame=int(packet["frame_index"])
        timestamp=float(packet["timestamp_s"])
        if frame!=self.previous_index+1:
            raise ValueError(f"Noncontiguous live frame {frame}; expected {self.previous_index+1}")
        if timestamp<=self.previous_timestamp:
            raise ValueError("Non-increasing live frame timestamp")
        if self.max_frames and self.count>=self.max_frames:
            raise ValueError("Receiver is at configured frame limit")
        dest=self.output_dir/f"frame_{frame:06d}.npz"
        temp=dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        try:
            with temp.open("xb") as fp:
                np.savez_compressed(fp,**packet)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(temp,dest)
        finally:
            temp.unlink(missing_ok=True)
        self.previous_index=frame
        self.previous_timestamp=timestamp
        self.count+=1
        return {"accepted_frame":frame,"received_frames":self.count,
                "training_depth_registration":"rgbtrack_adapter._rgb_registered_depth",
                "no_robot_action":True}


def make_http_handler(receiver:PacketReceiver,token:str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path!="/frame":
                return self._reply(404,{"error":"Unknown endpoint"})
            auth=self.headers.get("X-Live-Packet-Token","")
            if not hmac.compare_digest(auth,token):
                return self._reply(403,{"error":"Invalid packet token"})
            try:
                n=int(self.headers.get("Content-Length","0"))
                if not 0<n<=MAX_POST_BYTES:
                    return self._reply(413,{"error":"Invalid compressed packet length"})
                body=self.rfile.read(n)
                with np.load(io.BytesIO(body),allow_pickle=False) as loaded:
                    raw={k:np.asarray(loaded[k]) for k in loaded.files}
                result=receiver.accept(raw)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                return self._reply(400,{"error":str(exc)})
            self._reply(200,result)
            print(f"[LIVE] frame={result['accepted_frame']} received={receiver.count}",flush=True)

        def _reply(self,status,body):
            data=json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self,*args):
            return
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--front-calibration-profile",type=Path,
                   default=ROOT/"configs/ur7e_xhand/front_calibration_candidate.json")
    p.add_argument("--bind",default="127.0.0.1",
                   help="Default localhost; for trusted robot LAN explicitly set private interface or 0.0.0.0")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--max-frames",type=int,default=0)
    args=p.parse_args()
    if args.max_frames<0:
        raise ValueError("--max-frames must be >= 0")
    token=os.environ.get("OPENPI_LIVE_PACKET_TOKEN","")
    if len(token)<16:
        raise ValueError("Set OPENPI_LIVE_PACKET_TOKEN (at least 16 characters) on receiver and robot sender")
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/"src"))
    sys.path.insert(0,str(ROOT/"scripts/object_geometry/pose_tracking"))
    from rgbtrack_adapter import _load_front_camera_for_depth_registration,_rgb_registered_depth
    cam=_load_front_camera_for_depth_registration(ROOT,args.front_calibration_profile)
    def register(depth,hw):
        return _rgb_registered_depth(depth,cam,hw)
    receiver=PacketReceiver(args.output_dir.expanduser().resolve(),register_depth=register,
                            depth_scale=cam.depth_scale_m_per_unit,max_frames=args.max_frames)
    server=HTTPServer((args.bind,args.port),make_http_handler(receiver,token))
    print(f"[LIVE] sensor-only receiver listening {args.bind}:{args.port}/frame "
          f"output={receiver.output_dir} training_depth_scale={cam.depth_scale_m_per_unit}",flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
