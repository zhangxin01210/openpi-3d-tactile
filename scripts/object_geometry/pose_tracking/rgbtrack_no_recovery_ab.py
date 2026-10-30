#!/usr/bin/env python3
"""Controlled RGBTrack tracking-only A/B, built on verified upright INITIALIZATION.

A: --follow-mode direct        : never enter RGBTrack lost/re-registration path.
B: --follow-mode upright_gate  : A plus a per-frame causal +Z-axis tilt rejection.

This is an experimental diagnostic, not a trustworthy 6D truth generator.
It makes no edits to the RGBTrack submodule or model weights.
Only called from upright.main(), which restricts the FIRST initialization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np

try:
    import rgbtrack_adapter as adapter
    import rgbtrack_upright_full as upright
except ImportError as exc:
    raise SystemExit('Install alongside rgbtrack_adapter.py and rgbtrack_upright_full.py') from exc


def to_np(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def axis_tilt_deg(T_cam_obj: np.ndarray, R_base_cam: np.ndarray) -> float:
    axis = R_base_cam @ T_cam_obj[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(axis[2] / np.linalg.norm(axis), -1, 1))))


def main() -> int:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--follow-mode', required=True, choices=('direct', 'upright_gate'))
    extra.add_argument('--follow-max-tilt-deg', type=float, default=35.0)
    switches, remaining = extra.parse_known_args()
    if not 0 < switches.follow_max_tilt_deg < 90:
        raise SystemExit('follow-max-tilt-deg must be between 0 and 90')
    sys.argv = [sys.argv[0], *remaining]
    parser = adapter.build_arg_parser()
    # Full parsing only to get output/calibration; upright.main parses its own options.
    partial, _ = parser.parse_known_args(remaining)
    root = Path(partial.output).expanduser().resolve()
    calib = adapter.load_front_calibration(partial.repo_root, partial.front_calibration_profile)
    R_base_cam = np.asarray(calib.T_base_color, dtype=float)[:3, :3]

    original_ensure = adapter.RGBTrackObject._ensure_estimator
    original_estimate = adapter.RGBTrackObject.estimate
    hooked_ids: set[int] = set()
    stats: Counter = Counter()
    events: list[dict] = []
    runtime: dict[int, dict] = {}

    def ensure_no_recovery(self):
        original_ensure(self)
        if self.object_name != 'funnel' or id(self._estimator) in hooked_ids:
            return
        est = self._estimator
        hooked_ids.add(id(est))
        track_original = est.track_one_new
        rt = runtime.setdefault(id(self), {})

        def track_follow(*args, **kwargs):
            import torch
            previous_pose = est.pose_last
            if previous_pose is None:
                raise RuntimeError('Tracking without first registration')
            previous_pose = previous_pose.detach().clone()
            previous_depth = est.last_depth
            # CRITICAL: upstream track_one_new branches on track_good. For this
            # controlled experiment, use only one-pose refinement; NEVER invoke
            # its Kalman/lost/re-register branch.
            est.track_good = True
            observed = np.asarray(to_np(track_original(*args, **kwargs)), dtype=float).reshape(4, 4)
            native_depth_good = bool(getattr(est, 'track_good', False))
            tilt = axis_tilt_deg(observed, R_base_cam)
            accepted = bool(np.all(np.isfinite(observed)))
            if switches.follow_mode == 'upright_gate':
                accepted = accepted and tilt <= switches.follow_max_tilt_deg
            rt['tilt_deg'] = tilt
            rt['depth_check_passed'] = native_depth_good
            if not accepted:
                # Reject, do not silently present a stale previous pose as an
                # observed geometry or pretend the rejected pose is accurate.
                # Restore the last accepted centered pose for the NEXT refinement.
                est.pose_last = previous_pose
                est.last_depth = previous_depth
                est.track_good = True
                rt['last_status'] = 'rejected_upright_invalid' if np.isfinite(tilt) else 'rejected_nonfinite_invalid'
                stats[rt['last_status']] += 1
                return np.asarray(to_np(previous_pose @ est.get_tf_to_centered_mesh()), dtype=float).reshape(4, 4)
            # Keep the raw (unconstrained) translation and orientation of each
            # accepted normal single-pose refinement. If upstream depth sanity
            # check failed, mark this observation invalid but continue normal
            # tracking in the next frame (no recovery search).
            est.track_good = True
            rt['last_status'] = 'direct_tracking_observed' if native_depth_good else 'direct_tracking_depth_warning'
            stats[rt['last_status']] += 1
            return observed

        est.track_one_new = track_follow
        print(f'[FOLLOW] Installed tracking-only mode={switches.follow_mode}. No upstream re-registration.', flush=True)

    def estimate_follow(self, *, frame_index, rgb, depth_m, mask, K, T_base_color):
        rt = runtime.setdefault(id(self), {})
        rt['last_status'] = None
        rec = original_estimate(self, frame_index=frame_index, rgb=rgb,
                                depth_m=depth_m, mask=mask, K=K,
                                T_base_color=T_base_color)
        if self.object_name != 'funnel':
            return rec
        status = rt['last_status']
        if status is not None:
            if status.startswith('rejected_') or status.endswith('_warning'):
                rec.valid = False
                rec.confidence = 0.0
            rec.status = status
            events.append({'frame_index': int(frame_index), 'status': status,
                           'tilt_deg': rt['tilt_deg'],
                           'depth_check_passed': rt['depth_check_passed'],
                           'mask_area_px': int(np.asarray(mask, dtype=bool).sum())})
        return rec

    adapter.RGBTrackObject._ensure_estimator = ensure_no_recovery
    adapter.RGBTrackObject.estimate = estimate_follow
    print('[FOLLOW] Retaining upstream RGBTrack refinement and already verified rank-based upright FIRST initialization.', flush=True)
    print('[FOLLOW] No re-registration, no FK interpolation, no fabricated visible poses.', flush=True)
    if switches.follow_mode == 'upright_gate':
        print(f'[FOLLOW] Reject poses with CAD +Z base-frame tilt > {switches.follow_max_tilt_deg:.1f} degrees', flush=True)
    result = upright.main()
    report = {
        'schema_version': 1,
        'experiment': 'one-pose tracking-only; controlled vertical-gate A/B',
        'mode': switches.follow_mode,
        'upright_gate_threshold_deg': switches.follow_max_tilt_deg if switches.follow_mode == 'upright_gate' else None,
        'no_native_reregistration': True,
        'summary': dict(stats),
        'events': events,
        'warnings': [
            'Mask area confidence is not pose-estimator confidence.',
            'This is a diagnostic: accepted observations can still have wrong translation.',
            'Rejected poses are invalid and cannot be used as training geometry.',
            'RGBTrack normal track refinement can drift without loss/re-registration.',
        ],
    }
    (root / 'tracking_only_ab_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('[FOLLOW] report:', root/'tracking_only_ab_report.json', flush=True)
    return int(result)


if __name__ == '__main__':
    raise SystemExit(main())
