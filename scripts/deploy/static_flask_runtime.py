"""Shared training static-target semantics for online RGBTrack: observed f0, prior f>=1.

The initial RGBTrack registration uses the EXACT upright selector imported from
training's rgbtrack_flask_pivot. A separately human-reviewed, calibration-bound
anchor must agree with that observation; no pose is auto-approved. Later frames
do not call the flask tracker, nor trust later masks to update its pose.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_reviewed_anchor(path, *, episode, calibration_profile):
    path=Path(path).expanduser().resolve()
    entry=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entry,dict) or entry.get("reviewed") is not True:
        raise ValueError("Static flask anchor must explicitly be human-reviewed")
    if entry.get("source")!="human_reviewed":
        raise ValueError("Static flask anchor source must be human_reviewed")
    if int(entry.get("episode_index",-1))!=int(episode):
        raise ValueError("Static flask anchor episode mismatch")
    profile=Path(calibration_profile).expanduser().resolve()
    digest=hashlib.sha256(profile.read_bytes()).hexdigest()
    if entry.get("front_calibration_sha256")!=digest:
        raise ValueError("Static flask anchor calibration mismatch")
    T=np.asarray(entry["T_base_link_object"],dtype=np.float64)
    if (T.shape!=(4,4) or not np.isfinite(T).all() or
        not np.allclose(T[3],[0,0,0,1],atol=1e-6) or
        not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-2) or
        not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-2)):
        raise ValueError("Static flask anchor is not a rigid base_link transform")
    return T,entry


class StaticFlaskRuntime:
    """One-time reviewed registration; strict static geometry for later frames."""

    def __init__(self, anchor, *, first_frame=0, translation_tolerance_m=0.025,
                 rotation_tolerance_deg=20.):
        self.anchor=np.asarray(anchor,dtype=np.float64)
        self.first_frame=int(first_frame)
        self.translation_tolerance_m=float(translation_tolerance_m)
        self.rotation_tolerance_deg=float(rotation_tolerance_deg)
        self.initialized=False
        if self.first_frame!=0:
            raise ValueError("Training static flask must be initialized at frame 0")

    def next(self, *, frame_index, tracker=None, rgb=None, depth_m=None, mask=None,
             K=None, T_base_color=None):
        frame=int(frame_index)
        if frame==self.first_frame:
            if self.initialized:
                raise ValueError("Static flask frame-0 was already initialized")
            if tracker is None or mask is None or not np.asarray(mask).any():
                raise ValueError("Static flask needs a valid frame-0 mask and tracker")
            result=tracker.estimate(frame_index=frame,rgb=rgb,depth_m=depth_m,
                                    mask=mask,K=K,T_base_color=T_base_color)
            if not result.valid:
                raise ValueError("Frame-0 flask registration invalid; anchor must be reviewed")
            observed=np.asarray(result.T_base_object,dtype=np.float64)
            distance=float(np.linalg.norm(observed[:3,3]-self.anchor[:3,3]))
            delta=observed[:3,:3] @ self.anchor[:3,:3].T
            angle=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))))
            if distance>self.translation_tolerance_m or angle>self.rotation_tolerance_deg:
                raise ValueError(f"Current flask registration differs from human-reviewed anchor: "
                                 f"translation {distance:.4f}m rotation {angle:.2f}deg. "
                                 "Do not reuse a different scene anchor.")
            self.initialized=True
            return {"frame_index":frame,"T_base_link_object":observed,
                    "valid":True,"confidence":float(result.confidence),
                    "status":"flask_upright_anchor_observed",
                    "pose_source":"reviewed_initial_geometry",
                    "observed_pose_valid":True}
        if frame<self.first_frame or not self.initialized:
            raise ValueError("Static flask prior cannot precede frame-0 registration")
        return {"frame_index":frame,"T_base_link_object":self.anchor.copy(),
                "valid":True,"confidence":0.08,
                "status":"flask_static_anchor_prior",
                "pose_source":"static_geometry_prior",
                "observed_pose_valid":False}
