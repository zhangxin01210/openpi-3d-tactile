#!/usr/bin/env python3
"""Causal upright/temporal acceptance gate for the *existing* RGBTrack.

Wraps rgbtrack_upright_full.py; does NOT change FoundationPose network or models.
First frame: existing ranked upright initialization. Next frames: accept/refuse
RGBTrack predictions based on world-up axis, per-frame axis/angular change and
geometric-center displacement. Accepted corrections are fed back to RGBTrack.

Rotation-only repairs have status constrained_rotation_low_confidence and are
NOT 6D ground truth. Large translation jumps are INVALID, not made up.

Run alongside rgbtrack_upright_full.py and rgbtrack_adapter.py.
Requires the same pre-existing RGBTrack/CUDA environment.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import rgbtrack_adapter as adapter
import rgbtrack_upright_full as upright


def as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def angle_deg(u, v):
    u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
    return math.degrees(math.acos(float(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1., 1.))))


def evaluate_step(previous: np.ndarray, observation: np.ndarray, *, base_R_cam: np.ndarray,
                  mesh_center: np.ndarray, frame_delta: int, max_tilt_deg: float,
                  max_axis_step_deg: float, max_rot_step_deg: float,
                  max_center_step_m: float) -> tuple[np.ndarray, str, dict[str,float]]:
    """Return a causal corrected camera<-object transform plus explicit provenance.

    Only use information available at current time.  Local mesh +Z is bowl-up.
    Constant-R updates preserve the *observed object geometric center*.
    """
    prev = np.asarray(previous, np.float64)
    obs = np.asarray(observation, np.float64)
    c = np.asarray(mesh_center, np.float64).reshape(3)
    if prev.shape != (4,4) or obs.shape != (4,4) or not np.isfinite(obs).all():
        return prev.copy(), 'rejected_invalid', {}
    p0 = prev[:3,3] + prev[:3,:3] @ c
    p1 = obs[:3,3] + obs[:3,:3] @ c
    shift = float(np.linalg.norm(p1-p0))
    z0 = base_R_cam @ prev[:3,2]
    z1 = base_R_cam @ obs[:3,2]
    tilt = angle_deg(z1, np.array([0., 0., 1.]))
    axis_step = angle_deg(z0,z1)
    from scipy.spatial.transform import Rotation
    rot_step = float(np.degrees(Rotation.from_matrix(prev[:3,:3].T @ obs[:3,:3]).magnitude()))
    stats = {'tilt_deg':tilt, 'axis_step_deg':axis_step, 'full_rot_step_deg':rot_step,
             'center_step_m':shift}
    # Scale movement limit with frame gap, but never accept an unbounded leap.
    move_budget = max_center_step_m * min(max(1,int(frame_delta)), 2)
    if shift > move_budget:
        return prev.copy(), 'rejected_position_invalid', stats
    # Since the funnel is near-axisymmetric, yaw jumps can be entirely unobservable.
    # Keep the previous orientation, but follow the CURRENT candidate geometric center.
    if tilt > max_tilt_deg or axis_step > max_axis_step_deg or rot_step > max_rot_step_deg:
        corrected = prev.copy()
        corrected[:3,3] = p1 - corrected[:3,:3] @ c
        return corrected, 'constrained_rotation_low_confidence', stats
    return obs.copy(), 'accepted_rgbtrack', stats


def reset_upstream_pose(est, T_cam_object):
    """Always reset the estimator to its documented *centered-mesh* convention."""
    import torch
    from scipy.spatial.transform import Rotation
    center_T = as_np(est.get_tf_to_centered_mesh())
    centered = np.asarray(T_cam_object) @ np.linalg.inv(center_T)
    est.pose_last = torch.as_tensor(centered, device='cuda', dtype=torch.float32)
    est.xyz = est.pose_last[:3,3]
    xyz = centered[:3,3].reshape(3,1)
    euler = Rotation.from_matrix(centered[:3,:3]).as_euler('xyz').reshape(3,1)
    est.tracker.initialize(xyz, euler)


def main():
    own_parser=argparse.ArgumentParser(add_help=False)
    own_parser.add_argument('--temporal-max-tilt-deg',type=float,default=35.)
    own_parser.add_argument('--temporal-max-axis-step-deg',type=float,default=25.)
    own_parser.add_argument('--temporal-max-rot-step-deg',type=float,default=50.)
    own_parser.add_argument('--temporal-max-center-step-m',type=float,default=0.06)
    own_parser.add_argument('--temporal-fix-upstream-recovery-center',choices=('auto','yes','no'),default='auto')
    own, remaining = own_parser.parse_known_args()
    if not all(x>0 for x in [own.temporal_max_tilt_deg,own.temporal_max_axis_step_deg,
                             own.temporal_max_rot_step_deg,own.temporal_max_center_step_m]):
        own_parser.error('Temporal thresholds must be >0')
    # Let the previous fully-tested upright script parse every existing CLI option.
    sys.argv=[sys.argv[0]]+remaining
    base_parser = adapter.build_arg_parser()
    base_parser.add_argument('--upright-max-tilt-deg',type=float,default=25.)
    base_parser.add_argument('--upright-max-rank',type=int,default=150)
    parsed = base_parser.parse_args(remaining)
    out = Path(parsed.output).expanduser().resolve()
    T_base_cam = np.asarray(adapter.load_front_calibration(parsed.repo_root,parsed.front_calibration_profile).T_base_color, np.float64)
    state: dict[int,dict[str,Any]] = {}
    audit=[]
    stats=Counter()
    original_ensure=adapter.RGBTrackObject._ensure_estimator
    original_estimate=adapter.RGBTrackObject.estimate

    def get_state(tracker):
        return state.setdefault(id(tracker), {'last_accepted':None,'last_frame':None,'result':None,
                                            'pending_frame':None, 'estimator_id':None})

    def ensure_temporal(self):
        original_ensure(self)
        if self.object_name!='funnel':
            return
        est=self._estimator
        st=get_state(self)
        if st['estimator_id']==id(est):
            return
        st['estimator_id']=id(est)
        track_original=est.track_one_new
        # Assert before intervening with a potentially different upstream version.
        import inspect
        try:
            source=inspect.getsource(track_original)
        except (OSError,TypeError):
            source=''
        buggy_recovery=('self.pose_last = pose' in source and
                        'best_pose = poses[0] @ self.get_tf_to_centered_mesh()' in source)
        fix_recovery=(own.temporal_fix_upstream_recovery_center=='yes' or
                    (own.temporal_fix_upstream_recovery_center=='auto' and buggy_recovery))
        print(f'[TEMPORAL] funnel recovery double-center signature={buggy_recovery}; '
              f'correction_enabled={fix_recovery}',flush=True)

        def track_with_gate(*args,**kwargs):
            previous=st['last_accepted']
            if previous is None:
                raise RuntimeError('Temporal gate has no accepted first-frame upright pose')
            old_good=bool(getattr(est,'track_good',True))
            raw=as_np(track_original(*args,**kwargs)).reshape(4,4)
            # Upstream recover branch stores already de-centered pose then returns
            # another application of get_tf_to_centered_mesh(). Undo exactly one.
            if not old_good and fix_recovery:
                raw = raw @ np.linalg.inv(as_np(est.get_tf_to_centered_mesh()))
            frame=int(st['pending_frame'])
            delta=frame-int(st['last_frame'])
            proposed,decision,measure=evaluate_step(
                previous,raw,base_R_cam=T_base_cam[:3,:3],
                mesh_center=as_np(est.model_center),frame_delta=delta,
                max_tilt_deg=own.temporal_max_tilt_deg,
                max_axis_step_deg=own.temporal_max_axis_step_deg,
                max_rot_step_deg=own.temporal_max_rot_step_deg,
                max_center_step_m=own.temporal_max_center_step_m,
            )
            if decision=='accepted_rgbtrack':
                # RGBTrack sometimes uses different centered pose shapes or a
                # different convention after re-registration. Canonicalize it.
                if not old_good:
                    reset_upstream_pose(est,proposed)
                st['last_accepted']=proposed.copy()
                st['last_frame']=frame
                est.track_good=True
            elif decision=='constrained_rotation_low_confidence':
                reset_upstream_pose(est,proposed)
                est.track_good=True  # next frame should refine near accepted pose
                st['last_accepted']=proposed.copy()
                st['last_frame']=frame
            else:
                reset_upstream_pose(est,previous)
                # allow upstream re-register to seek a recovered observation
                est.track_good=False
                # NOTE: do not advance last_frame / last_accepted.
            st['result']=(decision,measure,old_good)
            audit.append({'frame':frame,'decision':decision,'upstream_was_track_good':old_good,
                          'recovery_center_correction_applied':bool(not old_good and fix_recovery),
                          **measure})
            stats[decision]+=1
            if decision!='accepted_rgbtrack' and (stats[decision]<=12 or frame%30==0):
                print(f'[TEMPORAL] f={frame} {decision} axis={measure.get("axis_step_deg",None)} '
                      f'tilt={measure.get("tilt_deg",None)} '
                      f'center_step={measure.get("center_step_m",None)}',flush=True)
            return proposed
        est.track_one_new=track_with_gate

    def estimate_with_status(self,*,frame_index,rgb,depth_m,mask,K,T_base_color):
        if self.object_name!='funnel':
            return original_estimate(self,frame_index=frame_index,rgb=rgb,depth_m=depth_m,
                                     mask=mask,K=K,T_base_color=T_base_color)
        st=get_state(self)
        st['pending_frame']=int(frame_index)
        st['result']=None
        rec=original_estimate(self,frame_index=frame_index,rgb=rgb,depth_m=depth_m,
                              mask=mask,K=K,T_base_color=T_base_color)
        if st['last_accepted'] is None and rec.valid:
            st['last_accepted']=np.asarray(rec.T_color_object).copy()
            st['last_frame']=int(frame_index)
            stats['initial_upright']+=1
        if st['result'] is not None:
            decision, measure, was_good=st['result']
            if decision=='constrained_rotation_low_confidence':
                rec.status=decision
                rec.confidence=min(float(rec.confidence),0.15)
                rec.valid=True  # geometry available; clearly LOW confidence + not true observed
            elif decision.startswith('rejected'):
                rec.status=decision
                rec.confidence=0.
                rec.valid=False
        else:
            # Adapter skips missing masks and upstream track would otherwise not
            # learn it was occluded. Set lost so next valid frame can re-register.
            if (self._initialized and self._estimator is not None and
                    np.asarray(mask,dtype=bool).sum()<self.min_mask_area_px):
                self._estimator.track_good=False
                stats['lost_mask']+=1
        return rec

    adapter.RGBTrackObject._ensure_estimator=ensure_temporal
    adapter.RGBTrackObject.estimate=estimate_with_status
    print('[TEMPORAL] Existing upright first-frame registration + per-frame causal physics gate',flush=True)
    print('[TEMPORAL] limits:',vars(own),flush=True)
    result=upright.main()
    # Save versioned diagnostic alongside pose-run. raw JSONL remains untouched.
    out.mkdir(parents=True,exist_ok=True)
    (out/'temporal_gate_report.json').write_text(json.dumps({
      'schema_version':1,'frame_causal':True,'frame_constraints':vars(own),
      'summary':dict(stats),'frames':audit,
      'provenance_note':'constrained_rotation_low_confidence is a prior-based estimate, not a vision-confirmed ground truth; adapter mask-area confidence is not calibrated.',
    },indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('[TEMPORAL] status summary:',dict(stats),flush=True)
    print('[TEMPORAL] report:',out/'temporal_gate_report.json',flush=True)
    return result

if __name__=='__main__':
    raise SystemExit(main())
