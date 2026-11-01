#!/usr/bin/env python3
"""Causal RGBTrack motion guard. Only filters dynamic-object observations.

A rejected observation is INVALID, never treated as ground-truth or clamped.
Rejecting also resets the adapter's causal state and requests new registration
on the next frame (register() reinitializes the RGBTrack Kalman tracker).
Do not apply to static objects; they use the existing frozen static policy.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
import json
import math
import numpy as np


def step_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float,float]:
    """Metres and SO(3) geodesic degrees between base-frame object poses."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != (4,4) or b.shape != (4,4) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf'), float('inf')
    distance=float(np.linalg.norm(a[:3,3]-b[:3,3]))
    R=a[:3,:3].T @ b[:3,:3]
    angle=math.degrees(math.acos(float(np.clip((np.trace(R)-1.)/2.,-1.,1.))))
    return distance,angle


def check_motion(previous: np.ndarray | None, candidate: np.ndarray, *,
                 elapsed_frames: int, max_step_m: float, max_turn_deg: float,
                 max_recovery_turn_deg: float = 60.0) -> dict:
    """Hard candidate acceptance gate, not a trajectory smoother.

    Frame elapsed time is accounted for using index difference.  The caller
    should use a real per-frame dt if datasets have nonuniform frame rates.
    """
    if max_step_m <= 0 or max_turn_deg <= 0 or not 0 < max_turn_deg <= 180 or not max_turn_deg <= max_recovery_turn_deg <= 180:
        raise ValueError('Both bounds must be positive and turn bound <=180 degrees')
    if elapsed_frames < 1:
        raise ValueError('elapsed_frames must be >=1')
    if previous is None:
        finite = np.asarray(candidate).shape == (4,4) and np.isfinite(candidate).all()
        return dict(accepted=bool(finite), reason='first_observation' if finite else 'nonfinite',
                    distance_m=None, angle_deg=None, allowable_m=None,
                    allowable_deg=None)
    d,angle=step_metrics(previous,candidate)
    md=max_step_m*elapsed_frames
    # Re-registration used to relax this to 180 degrees after enough missed
    # updates, accidentally accepting a flipped object. Cap total rotation
    # against the last *accepted* pose, even after repeated reinitializations.
    ma=min(180.0,max_turn_deg*elapsed_frames,max_recovery_turn_deg)
    if not np.isfinite(d) or not np.isfinite(angle):
        reason='nonfinite'
    elif d>md:
        reason='translation'
    elif angle>ma:
        reason='rotation'
    else:
        reason='accepted'
    return dict(accepted=reason=='accepted',reason=reason,
                distance_m=round(d,6),angle_deg=round(angle,4),
                allowable_m=round(md,6),allowable_deg=round(ma,4))


def install_motion_guard(adapter, *, name: str, output: Path,
                         max_step_m: float, max_turn_deg: float,
                         max_recovery_turn_deg: float = 60.0):
    """Install after static policy so its static wrapper remains untouched."""
    if not 0 < max_turn_deg <= max_recovery_turn_deg <= 180:
        raise ValueError('Recovery turn cap must be between per-frame turn limit and 180 degrees')
    original_estimate=adapter.RGBTrackObject.estimate
    last_good={}
    events=[]
    counters={'accepted_observations':0,'rejected':0,'native_bad':0,
              'reinitialization_requested':0}

    def estimate(self, *, frame_index, rgb, depth_m, mask, K, T_base_color):
        if self.object_name != name:
            return original_estimate(self, frame_index=frame_index, rgb=rgb,
                    depth_m=depth_m, mask=mask, K=K, T_base_color=T_base_color)
        key=id(self)
        state_before=copy.deepcopy(self._state)
        rec=original_estimate(self,frame_index=frame_index,rgb=rgb,
                    depth_m=depth_m,mask=mask,K=K,T_base_color=T_base_color)
        # Missing masks cause invalid predictions.  They do not constitute new
        # observations, and the last accepted pose is intentionally unchanged.
        if rec.status in ('occluded_causal_prediction','invalid_no_observation',
                          'before_requested_initialization'):
            return rec
        if not rec.valid:
            decision={'accepted':False,'reason':'native_bad','distance_m':None,
                      'angle_deg':None,'allowable_m':None,'allowable_deg':None}
            counters['native_bad']+=1
        else:
            old=last_good.get(key)
            elapsed=max(1,int(frame_index)-(old[0] if old is not None else int(frame_index)-1))
            decision=check_motion(old[1] if old else None, rec.T_base_object,
                        elapsed_frames=elapsed,max_step_m=max_step_m,max_turn_deg=max_turn_deg,
                        max_recovery_turn_deg=max_recovery_turn_deg)
        if decision['accepted']:
            last_good[key]=(int(frame_index),np.asarray(rec.T_base_object).copy(),
                            np.asarray(rec.T_color_object).copy())
            counters['accepted_observations']+=1
            return rec
        # The original adapter writes _state.observe even when track_good=False.
        # Restore it.  Force a *real* register() next time: simply reverting the
        # returned JSON leaves upstream RGBTrack/Kalman already poisoned.
        self._state=state_before
        self._initialized=False
        if self._estimator is not None:
            self._estimator.track_good=False
        counters['reinitialization_requested']+=1
        counters['rejected']+=1
        event={'frame_index':int(frame_index),'reason':decision['reason'],
               'status_before':rec.status,**{k:v for k,v in decision.items() if k!='accepted'}}
        events.append(event)
        prior=last_good.get(key)
        if prior is not None:
            # Retain last trusted pose for visualization diagnostics only; invalid
            # records must never be consumed as ground-truth by the training pipeline.
            rec=replace(rec,T_base_object=prior[1].copy(),T_color_object=prior[2].copy())
        rec.valid=False
        rec.confidence=0.0
        rec.status='motion_rejected_'+decision['reason']
        return rec

    adapter.RGBTrackObject.estimate=estimate

    def write_report():
        report_dir=Path(output)
        if not report_dir.exists():
            return
        report={'schema_version':1,'dynamic_object':name,
                'max_step_m_per_frame':max_step_m,'max_turn_deg_per_frame':max_turn_deg,
                'max_recovery_turn_deg':max_recovery_turn_deg,
                'counters':counters,'events':events,
                'notes':['Rejected pose is INVALID, not clamped or interpolated.',
                         'First re-registration after rejection must pass the same gate.',
                         'Large genuine motions may be rejected; inspect review HTML.',
                         'Total rotation from last trusted pose is capped during recovery, not just per-frame.',
                         'True fast rotations beyond the cap are rejected and require review.',
                         'Continued small errors below the gate are not eliminated.']}
        (report_dir/'motion_guard_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return write_report
