#!/usr/bin/env python3
"""Episode RGBTrack: vetted upright init + upstream loss/recovery + upright recovery candidates.

Uses existing RGBTrack weights and the adapter; no model training, mesh rebuilding,
FK fusion, or smoothing.  Only the funnel is constrained; the target flask uses
unchanged RGBTrack.  This is a *diagnostic*; no unverified pose is training truth.

The funnel's calibrated CAD has object +Z pointing from stem toward bowl.
Assumption for Episode 23 only: bowl remains upward, at most --max-tilt-deg.

Requires rgbtrack_adapter.py, rgbtrack_upright_full.py and existing RGBTrack env.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import rgbtrack_adapter as adapter
import rgbtrack_upright_full as upright


def arr(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def tilt_many(poses: np.ndarray, R_base_cam: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim == 2:
        poses = poses[None]
    axes = np.einsum("ij,njk,k->ni", R_base_cam, poses[:, :3, :3], np.array([0., 0., 1.]))
    return np.degrees(np.arccos(np.clip(axes[:, 2] / np.maximum(np.linalg.norm(axes, axis=1), 1e-9), -1, 1)))


def mask_center_error_px(centered_pose: np.ndarray, mask: np.ndarray, K: np.ndarray) -> float:
    """Cheap image-space sanity check. Mask center may shift under occlusion."""
    pixels = np.argwhere(np.asarray(mask) != 0)
    if len(pixels) == 0:
        return float("inf")
    p = np.asarray(centered_pose, np.float64).reshape(4, 4)[:3, 3]
    if not np.isfinite(p).all() or p[2] <= 0.05:
        return float("inf")
    K = np.asarray(K, dtype=np.float64)
    projected = K @ p
    projected = projected[:2] / projected[2]
    # BBox midpoint is consistent with upstream guess_translation(); for a
    # severely occluded or asymmetric mask, this is ONLY a conservative gate.
    ymin, xmin = pixels.min(axis=0)
    ymax, xmax = pixels.max(axis=0)
    center = np.array([(xmin+xmax)*0.5, (ymin+ymax)*0.5])
    return float(np.linalg.norm(projected - center))


def main() -> int:
    optparser = argparse.ArgumentParser(add_help=False)
    optparser.add_argument("--follow-max-tilt-deg", type=float, default=35.0)
    optparser.add_argument("--recovery-center-max-px", type=float, default=90.0)
    optparser.add_argument("--recovery-depth-l1-max-m", type=float, default=0.18)
    switches, remaining = optparser.parse_known_args()
    if not 0.0 < switches.follow_max_tilt_deg < 90.0:
        raise SystemExit("--follow-max-tilt-deg must be in (0,90)")
    if not 10 <= switches.recovery_center_max_px <= 300:
        raise SystemExit("--recovery-center-max-px must be in [10,300]")
    if not 0.02 <= switches.recovery_depth_l1_max_m <= 1.0:
        raise SystemExit("--recovery-depth-l1-max-m must be in [0.02,1]")
    sys.argv = [sys.argv[0], *remaining]
    args, _ = adapter.build_arg_parser().parse_known_args(remaining)
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"Refuse overwrite: {out}")
    calib = adapter.load_front_calibration(args.repo_root, args.front_calibration_profile)
    R_base_cam = np.asarray(calib.T_base_color, float)[:3, :3]

    original_ensure = adapter.RGBTrackObject._ensure_estimator
    original_estimate = adapter.RGBTrackObject.estimate
    installed = set()
    runtime: dict[int, dict] = {}
    stats = Counter()
    events: list[dict] = []

    def ensure_with_recovery(self):
        original_ensure(self)
        if self.object_name != "funnel" or id(self._estimator) in installed:
            return
        est = self._estimator
        installed.add(id(est))
        original_track = est.track_one_new
        original_generate = est.generate_random_pose_hypo
        original_score = est.scorer.predict
        rt = runtime.setdefault(id(self), {})
        rt["in_recovery"] = False
        rt["last_status"] = None

        def constrained_hypotheses(*gen_args, **gen_kwargs):
            poses = original_generate(*gen_args, **gen_kwargs)
            if not rt["in_recovery"]:
                return poses
            tilts = tilt_many(arr(poses), R_base_cam)
            eligible = np.flatnonzero(tilts <= switches.follow_max_tilt_deg)
            rt["recovery_initial_candidate_total"] = len(tilts)
            rt["recovery_initial_candidate_eligible"] = len(eligible)
            if not len(eligible):
                rt["recovery_no_initial_candidates"] = True
                return poses
            # The upstream recovery always takes 10 evenly spaced samples.
            # Return exactly 10 physically admissible hypotheses to retain the
            # original compute budget while covering distinct yaw alternatives.
            take = eligible[np.linspace(0, len(eligible)-1, 10).astype(int)]
            stats["upright_hypothesis_recovery_calls"] += 1
            return poses[take.tolist()]

        def constrained_scores(*score_args, **score_kwargs):
            scores, vis = original_score(*score_args, **score_kwargs)
            if not rt["in_recovery"]:
                return scores, vis
            # IMPORTANT: filter candidates *after refinement*, not just before.
            ob = score_kwargs.get("ob_in_cams")
            if ob is None:
                rt["recovery_no_pose_scores"] = True
                return scores, vis
            tilts = tilt_many(arr(ob), R_base_cam)
            eligible = tilts <= switches.follow_max_tilt_deg
            rt["recovery_refined_candidate_eligible"] = int(eligible.sum())
            if not eligible.any():
                rt["recovery_no_refined_candidates"] = True
                return scores, vis
            import torch
            if isinstance(scores, torch.Tensor):
                new = scores.clone()
                mask = torch.as_tensor(~eligible, device=new.device, dtype=torch.bool)
                new[mask] = -1.0e9
            else:
                new = np.asarray(scores).copy()
                new[~eligible] = -1.0e9
            return new, vis

        est.generate_random_pose_hypo = constrained_hypotheses
        est.scorer.predict = constrained_scores

        def track_with_recovery(*track_args, **track_kwargs):
            import torch
            from scipy.spatial.transform import Rotation
            previous = np.asarray(arr(est.pose_last), dtype=np.float64).reshape(4, 4).copy()
            prev_depth = est.last_depth
            was_good = bool(est.track_good)
            mask = track_kwargs.get("mask", track_args[4] if len(track_args) > 4 else None)
            depth = track_kwargs.get("depth", track_args[1] if len(track_args) > 1 else None)
            K = track_kwargs.get("K", track_args[2] if len(track_args) > 2 else None)
            area = int(np.count_nonzero(mask)) if mask is not None else 0
            rt["in_recovery"] = not was_good and area > 40
            rt["recovery_no_initial_candidates"] = False
            rt["recovery_no_refined_candidates"] = False
            try:
                ret = np.asarray(arr(original_track(*track_args, **track_kwargs)), np.float64).reshape(4, 4)
            finally:
                rt["in_recovery"] = False
            if area <= 40:
                rt["last_status"] = "lost_small_mask"
                stats["lost_small_mask"] += 1
                return ret

            center_transform = np.asarray(arr(est.get_tf_to_centered_mesh()), np.float64).reshape(4, 4)
            if not was_good:
                # Upstream bug: recovery did public=(internal@C), stores public
                # in pose_last, and returns public@C.  Undo double uncentering.
                public = ret @ np.linalg.inv(center_transform)
                centered = public @ np.linalg.inv(center_transform)
                rt["recovery_uncenter_correction_applied"] = True
            else:
                public = ret
                centered = np.asarray(arr(est.pose_last), np.float64).reshape(4, 4)

            tilt = float(tilt_many(public, R_base_cam)[0])
            center_px = mask_center_error_px(centered, mask, K)
            rt["tilt_deg"] = tilt
            rt["mask_center_error_px"] = center_px
            rt["native_depth_good"] = bool(est.track_good)
            valid_geom = bool(np.isfinite(public).all() and np.isfinite(tilt) and
                              tilt <= switches.follow_max_tilt_deg and
                              center_px <= switches.recovery_center_max_px)

            if not was_good:
                # Also guard transparent-object depth ambiguity.  Unlike opaque
                # benchmark objects, glass depth can be BACKGROUND, so this is
                # advisory: a failed depth gate is reported, not used to force
                # acceptance or rejection if silhouette/axis remain plausible.
                try:
                    guessed = np.asarray(est.guess_translation(np.asarray(depth),
                                                               np.asarray(mask), np.asarray(K)), float).reshape(3)
                    depth_error = float(np.abs(centered[:3, 3] - guessed).sum())
                except (ValueError, TypeError):
                    depth_error = float("inf")
                rt["recovery_depth_l1_m"] = depth_error
                rt["recovery_depth_warning"] = bool(depth_error > switches.recovery_depth_l1_max_m)
                if not valid_geom or rt["recovery_no_refined_candidates"] or rt["recovery_no_initial_candidates"]:
                    why = "recovery_upright_rejected" if tilt > switches.follow_max_tilt_deg else "recovery_geometry_rejected"
                    if rt["recovery_no_refined_candidates"]:
                        why = "recovery_no_upright_candidate"
                    rt["last_status"] = why
                    stats[why] += 1
                    # Re-entry on next frame: preserve previous plausible pose.
                    est.pose_last = torch.as_tensor(previous, device="cuda", dtype=torch.float32)
                    est.last_depth = prev_depth
                    est.track_good = False
                    return previous @ center_transform
                est.pose_last = torch.as_tensor(centered, device="cuda", dtype=torch.float32)
                xyz = centered[:3, 3]
                est.last_depth = np.full_like(np.asarray(mask), xyz[2], dtype=np.float32)
                euler = Rotation.from_matrix(centered[:3, :3]).as_euler("xyz")
                est.tracker.initialize(xyz.reshape(3, 1), euler.reshape(3, 1))
                est.track_good = True
                rt["last_status"] = "native_recovered_upright_depth_warning" if rt["recovery_depth_warning"] else "native_recovered_upright"
                stats[rt["last_status"]] += 1
                return public

            if not valid_geom:
                why = "track_upright_rejected" if tilt > switches.follow_max_tilt_deg else "track_position_rejected"
                rt["last_status"] = why
                stats[why] += 1
                est.pose_last = torch.as_tensor(previous, device="cuda", dtype=torch.float32)
                est.last_depth = prev_depth
                est.track_good = False
                # Invalid observation; returns previous solely for adapter
                # serialization. visualize_rgbtrack_full_episode hides invalid.
                return previous @ center_transform
            if not rt["native_depth_good"]:
                rt["last_status"] = "lost_native_depth_warning"
                stats["lost_native_depth_warning"] += 1
                # Keep the candidate for diagnosis, but allow native recovery
                # next frame.  This record remains invalid for train data.
                return public
            rt["last_status"] = "native_tracking_upright"
            stats["native_tracking_upright"] += 1
            return public

        est.track_one_new = track_with_recovery
        print("[COMBINED] installed upright recovery candidates + native tracking/loss logic", flush=True)

    def estimate_with_quality(self, *, frame_index, rgb, depth_m, mask, K, T_base_color):
        if self.object_name != "funnel":
            return original_estimate(self, frame_index=frame_index, rgb=rgb,
                depth_m=depth_m, mask=mask, K=K, T_base_color=T_base_color)
        rt = runtime.setdefault(id(self), {})
        rt["last_status"] = None
        old_threshold = self.min_mask_area_px
        if self._initialized:
            # Pass small masks through to upstream lost-state machinery.
            self.min_mask_area_px = 0
        try:
            rec = original_estimate(self, frame_index=frame_index, rgb=rgb,
                depth_m=depth_m, mask=mask, K=K, T_base_color=T_base_color)
        finally:
            self.min_mask_area_px = old_threshold
        status = rt.get("last_status")
        if status:
            rec.status = status
            if status.startswith(("lost_", "track_", "recovery_")):
                rec.valid = False
                rec.confidence = 0.0
            elif "warning" in status:
                # Display of recovered orientation is only for visual QA.
                rec.confidence = min(rec.confidence, 0.15)
            events.append({
                "frame_index": int(frame_index), "status": status,
                "valid": bool(rec.valid),
                "mask_area_px": int(np.count_nonzero(mask)),
                "tilt_deg": rt.get("tilt_deg"),
                "mask_center_error_px": rt.get("mask_center_error_px"),
                "native_depth_good": rt.get("native_depth_good"),
                "recovery_depth_l1_m": rt.get("recovery_depth_l1_m"),
                "recovery_initial_candidate_total": rt.get("recovery_initial_candidate_total"),
                "recovery_initial_candidate_eligible": rt.get("recovery_initial_candidate_eligible"),
                "recovery_refined_candidate_eligible": rt.get("recovery_refined_candidate_eligible"),
            })
        return rec

    adapter.RGBTrackObject._ensure_estimator = ensure_with_recovery
    adapter.RGBTrackObject.estimate = estimate_with_quality
    print("[COMBINED] First frame: existing 252-candidate upright rank selection.", flush=True)
    print("[COMBINED] Later: upstream single-pose refinement + official Kalman/re-registration.", flush=True)
    print("[COMBINED] Recovery: same 10 scoring hypotheses, selected from upright CAD poses.", flush=True)
    print("[COMBINED] Tilt/2D-center sanity gates reject inconsistent updates; no pose clamping.", flush=True)
    result = upright.main()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "experiment": "RGBTrack native recovery + constrained upright candidate search",
        "parameters": {
            "follow_max_tilt_deg": switches.follow_max_tilt_deg,
            "recovery_center_max_px": switches.recovery_center_max_px,
            "recovery_depth_l1_max_m": switches.recovery_depth_l1_max_m,
            "initial_upright_selection": "delegated to rgbtrack_upright_full.py",
            "recovery_hypotheses": 10,
        },
        "summary": dict(stats), "funnel_events": events,
        "warnings": [
            "This is an experiment, NOT a verified 6D pose source.",
            "SAM2 mask centroid can be biased by occlusion.",
            "Glass depth can be background; depth warnings are reported separately.",
            "valid=True does not imply precise hand-to-funnel metric relation.",
        ],
    }
    (out / "native_upright_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("[COMBINED] report:", out / "native_upright_report.json", flush=True)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
