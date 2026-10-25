#!/usr/bin/env python3
"""RGBTrack official loss/re-registration path on top of verified upright first init.

This is a narrow adapter patch, NOT a temporal pose smoother or FK fusion.
- Forward small/empty masks to upstream track_one_new so its original Kalman
  lost-state is updated (the project adapter previously short-circuited).
- Preserve the upstream re-registration/refiner/scorer and all pose hypotheses.
- Fix two confirmed upstream recovery-branch bugs: duplicate uncentering and
  unconditional track_good=True after failed translation consistency check.
- Preserve invalid status instead of silently treating Kalman prediction as
  a valid observation.
- Keep the upright filter ONLY on the initial register, via existing script.

This only improves recovery when the SAM2 mask is genuinely trustworthy.
If the SAM2 mask sticks to the background/stand, RGBTrack cannot recover from it.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import rgbtrack_adapter as adapter
    import rgbtrack_upright_full as upright
except ImportError as exc:
    raise SystemExit("Place beside rgbtrack_adapter.py and rgbtrack_upright_full.py") from exc


def as_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def recover_centered_pose(returned: np.ndarray, tf_to_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correct the exact upstream erroneous recovery return convention.

    Upstream recovery: best_pose = centered @ C; self.pose_last = best_pose;
    return best_pose @ C.  Thus returned = centered @ C @ C.
    Correct public pose is centered @ C, internal pose is centered.
    """
    c_inv = np.linalg.inv(tf_to_center)
    public_pose = returned @ c_inv
    internal_pose = public_pose @ c_inv
    return public_pose, internal_pose


def main() -> int:
    # Parse only this script's switches; upright.main parses the remaining
    # unchanged project adapter CLI arguments.
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--native-recovery-depth-l1-max-m", type=float, default=0.10,
                   help="Upstream depth/center gate is L1 <= 0.10 m by default")
    opts, remaining = p.parse_known_args()
    if not (0.02 <= opts.native_recovery_depth_l1_max_m <= 1.0):
        raise SystemExit("recovery depth gate must be in [0.02,1.0] m")
    sys.argv = [sys.argv[0], *remaining]

    # upright.main() parses its own args, so locate output without consuming
    # the added upright switches.
    try:
        out_path = Path(remaining[remaining.index("--output") + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("Missing standard --output DIR argument") from exc
    if out_path.exists():
        raise SystemExit(f"Output already exists; refuse overwrite: {out_path}")

    original_ensure = adapter.RGBTrackObject._ensure_estimator
    original_estimate = adapter.RGBTrackObject.estimate
    statistics: Counter = Counter()
    events: list[dict] = []
    installed: set[int] = set()
    runtime: dict[int, dict] = {}

    def ensure_native(self):
        original_ensure(self)
        if self.object_name != "funnel":
            return
        est = self._estimator
        if id(est) in installed:
            return
        installed.add(id(est))
        orig_track = est.track_one_new
        # Guard the exact upstream recovery convention before patching it.
        import inspect
        try:
            source = inspect.getsource(orig_track)
        except (OSError,TypeError) as exc:
            raise RuntimeError("Cannot audit installed RGBTrack recovery method") from exc
        if not ("best_pose = poses[0] @ self.get_tf_to_centered_mesh()" in source
                and "self.pose_last = pose" in source
                and "self.track_good = True" in source):
            raise RuntimeError("RGBTrack upstream recovery code differs; refuse unsafe auto-fix")
        rt = runtime.setdefault(id(self), {})
        rt["estimator"] = est

        def track_with_native_recovery(*track_args, **kwargs):
            import torch
            from scipy.spatial.transform import Rotation

            old_good = bool(getattr(est, "track_good", True))
            previous = as_numpy(est.pose_last).copy() if est.pose_last is not None else None
            mask = kwargs.get("mask", track_args[4] if len(track_args) > 4 else None)
            depth = kwargs.get("depth", track_args[1] if len(track_args) > 1 else None)
            K = kwargs.get("K", track_args[2] if len(track_args) > 2 else None)
            area = int(np.count_nonzero(mask)) if mask is not None else -1
            ret = as_numpy(orig_track(*track_args, **kwargs)).reshape(4, 4)

            # The official Kalman prediction path already handles small masks.
            if area <= 40:
                rt["last_transition"] = "lost_native_mask"
                statistics["lost_native_mask"] += 1
                return ret

            if old_good:
                # Official normal one-pose refinement.  Upstream may flag a
                # center mismatch as lost; do not overwrite that decision.
                if not bool(getattr(est, "track_good", True)):
                    rt["last_transition"] = "lost_native_depth_mismatch"
                    statistics["lost_native_depth_mismatch"] += 1
                else:
                    rt["last_transition"] = "normal_native_track"
                    statistics["normal_native_track"] += 1
                return ret

            # Native lost-state recovery: upstream already selected 10 pose
            # candidates, refined/scored them, and returned a chosen pose.
            # Fix the known double-centering error before publishing the pose.
            tf_center = as_numpy(est.get_tf_to_centered_mesh()).reshape(4, 4)
            public_pose, centered = recover_centered_pose(ret, tf_center)
            xyz = centered[:3, 3]

            try:
                guessed = np.asarray(est.guess_translation(np.asarray(depth),
                                    np.asarray(mask), np.asarray(K)), np.float64).reshape(3)
            except (ValueError, TypeError) as exc:
                guessed = np.full(3, np.nan)
                rt["last_recovery_warning"] = repr(exc)
            depth_err = float(np.abs(xyz - guessed).sum())
            plausible = (np.all(np.isfinite(centered)) and
                         np.isfinite(depth_err) and guessed[2] > 0 and
                         depth_err <= opts.native_recovery_depth_l1_max_m)
            rt["last_recovery_error_m_l1"] = depth_err
            if not plausible:
                # Do not contaminate the centered pose for the next iteration.
                if previous is not None:
                    est.pose_last = torch.as_tensor(previous,device="cuda",dtype=torch.float32)
                    public_pose = previous @ tf_center
                est.track_good = False
                rt["last_transition"] = "native_recovery_failed_depth_check"
                statistics["native_recovery_failed_depth_check"] += 1
                print(f"[NATIVE] recovery rejected: L1_depth={depth_err:.4f} m",flush=True)
                return public_pose

            est.pose_last = torch.as_tensor(centered, device="cuda", dtype=torch.float32)
            rotation = Rotation.from_matrix(centered[:3, :3]).as_euler("xyz").reshape(3, 1)
            est.tracker.initialize(xyz.reshape(3, 1), rotation)
            est.track_good = True
            rt["last_transition"] = "native_recovered"
            statistics["native_recovered"] += 1
            print(f"[NATIVE] recovery accepted: L1_depth={depth_err:.4f} m",flush=True)
            return public_pose

        est.track_one_new = track_with_native_recovery
        print("[NATIVE] installed official loss/Kalman/re-register path; fixed recovery coordinate convention",flush=True)

    def estimate_native(self, *, frame_index, rgb, depth_m, mask, K, T_base_color):
        if self.object_name != "funnel":
            return original_estimate(self,frame_index=frame_index,rgb=rgb,depth_m=depth_m,
                                     mask=mask,K=K,T_base_color=T_base_color)
        rt = runtime.setdefault(id(self), {})
        rt["last_transition"] = None
        was_initialized = self._initialized
        old_limit = self.min_mask_area_px
        # Only AFTER initial registration, ensure upstream receives masks
        # smaller than 40 px so it actually triggers its lost-state machinery.
        if was_initialized:
            self.min_mask_area_px = 0
        try:
            rec = original_estimate(self,frame_index=frame_index,rgb=rgb,depth_m=depth_m,
                                    mask=mask,K=K,T_base_color=T_base_color)
        finally:
            self.min_mask_area_px = old_limit
        trans = rt.get("last_transition")
        if trans is not None:
            if trans.startswith("lost_") or trans.startswith("native_recovery_failed"):
                rec.valid = False
                rec.confidence = 0.0
                rec.status = trans
            elif trans == "native_recovered":
                rec.status = "native_recovered"
            events.append({"frame":int(frame_index),"status":trans,
                          "mask_area_px":int(np.count_nonzero(mask)),
                          "depth_recovery_l1_m":rt.get("last_recovery_error_m_l1")
                                      if trans.startswith("native_recover") else None})
        return rec

    adapter.RGBTrackObject._ensure_estimator = ensure_native
    adapter.RGBTrackObject.estimate = estimate_native
    print("[NATIVE] Only initial registration uses upright selection; no temporal/vertical pose clamping.",flush=True)
    print("[NATIVE] Upstream lost-state detection, Kalman prediction, and 10-candidate re-register enabled.",flush=True)
    print("[NATIVE] Invalid recovery remains invalid; model score still NOT calibrated confidence.",flush=True)
    result = upright.main()
    out_path.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version":1,"method":"upstream RGBTrack loss/recovery with minimal state/coordinate fixes",
        "official_mask_threshold_px":40,
        "recovery_depth_l1_max_m":opts.native_recovery_depth_l1_max_m,
        "summary":dict(statistics),"funnel_events":events,
        "warnings":[
            "SAM2 masks are NOT verified by this script; incorrect masks prevent reliable recovery.",
            "A valid=True or mask_area-based confidence does not prove 6D pose quality.",
            "No temporal pose smoothing, vertical clamp or FK constraint is applied.",
        ],
    }
    (out_path/"native_recovery_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("[NATIVE] report:",out_path/"native_recovery_report.json",flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
