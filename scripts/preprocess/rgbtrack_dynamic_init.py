#!/usr/bin/env python3
"""Optional *first-ever* upright registration for a dynamic RGBTrack object.

The static policy applies ONLY to the static object. This separate opt-in
selects an upright hypothesis for the dynamic object at its first successful
registration, then leaves subsequent RGBTrack tracking/re-registration to the
existing motion guard. No pose clamping or constant-upright constraint.

The object-space upright axis must be explicitly supplied (e.g. +z). It should
be audited against an accepted reference run before enabling the policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def arr(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def parse_axis(spec: str) -> np.ndarray:
    axes = {'+x': (1, 0, 0), '-x': (-1, 0, 0),
            '+y': (0, 1, 0), '-y': (0, -1, 0),
            '+z': (0, 0, 1), '-z': (0, 0, -1)}
    if spec not in axes:
        raise ValueError(f'Bad upright axis {spec!r}; choose one of {tuple(axes)}')
    return np.asarray(axes[spec], np.float64)


def select_first_dynamic_pose(*, poses: np.ndarray, scores: np.ndarray,
                              center: np.ndarray, vertices: np.ndarray,
                              mask: np.ndarray, K: np.ndarray,
                              base_R_camera: np.ndarray, upright_axis: str,
                              max_tilt_deg: float = 25., max_rank: int = 150,
                              min_iou: float = 0.20):
    """Return selected rank & diagnostics, fail if no valid upright candidate.

    Preserve native #0 when already upright and visually plausible, so a
    previously good registration is not changed just to optimize an IoU tie.
    All candidates use RGBTrack's centered -> public pose transform.
    The convex-hull check is a coarse 2D sanity gate, not a true rendered IoU.
    """
    from rgbtrack_static_policy import mesh_projection_mask

    p = np.asarray(arr(poses), np.float64)
    s = np.asarray(arr(scores), np.float64).reshape(-1)
    c = np.asarray(center, np.float64)
    M = np.asarray(mask).astype(bool)
    K = np.asarray(K, np.float64)
    Rbc = np.asarray(base_R_camera, np.float64)
    if p.ndim != 3 or p.shape[1:] != (4,4) or c.shape != (4,4) or len(p) != len(s):
        raise ValueError('Invalid RGBTrack pose candidates / scorer / mesh-center transform')
    if not np.any(M):
        raise ValueError('Empty dynamic-object mask at first registration')
    if Rbc.shape != (3,3) or not 0 < max_tilt_deg < 90 or max_rank < 1 or not 0 <= min_iou <= 1:
        raise ValueError('Invalid dynamic upright settings')
    axis = parse_axis(upright_axis)
    h, w = M.shape
    entries = []
    for rank in range(min(len(p), max_rank)):
        if not np.isfinite(p[rank]).all():
            continue
        up_base = Rbc @ (p[rank, :3,:3] @ axis)
        tilt = float(np.rad2deg(np.arccos(np.clip(up_base[2]/max(np.linalg.norm(up_base),1e-12),-1.,1.))))
        if tilt > max_tilt_deg:
            continue
        public = p[rank] @ c
        project = mesh_projection_mask(vertices, public, K, (h,w)).astype(bool)
        inter = int(np.count_nonzero(project & M))
        union = int(np.count_nonzero(project | M))
        iou = inter / union if union else 0.
        # Tie break by original rank; IoU dominates only when needed.
        quality = 0.95*iou + 0.05*(1-rank/max(len(p)-1,1))
        entries.append({'rank': int(rank), 'tilt_deg': round(tilt,3),
                        'mask_hull_iou': round(float(iou),5),
                        'native_score': round(float(s[rank]),6),
                        'selection_quality': round(float(quality),6)})
    viable = [x for x in entries if x['mask_hull_iou'] >= min_iou]
    # Rank 0 is a proven native output; preserve it if the upright+mask gates pass.
    candidate = next((x for x in viable if x['rank']==0), None)
    if candidate is None and viable:
        candidate = max(viable,key=lambda x:(x['selection_quality'],-x['rank']))
    return candidate, sorted(entries,key=lambda x:(-x['selection_quality'],x['rank']))


def install_dynamic_first_upright(adapter, *, name: str, base_R_camera: np.ndarray,
                                  output: Path, upright_axis: str,
                                  max_tilt_deg: float = 25., max_rank: int = 150,
                                  min_iou: float = .20):
    """Wrap only the dynamic object's register(); first-ever init only.

    The existing static-object policy is installed first. The motion guard is
    installed afterwards and handles all later poses and recoveries.
    """
    parse_axis(upright_axis)
    original_ensure = adapter.RGBTrackObject._ensure_estimator
    wrapped_estimator_ids: set[int] = set()
    selected_by_id: dict[int,dict] = {}

    def ensure(self):
        original_ensure(self)
        if self.object_name != name or id(self._estimator) in wrapped_estimator_ids:
            return
        est = self._estimator
        wrapped_estimator_ids.add(id(est))
        native_register = est.register
        first_selected = False

        def register(*args,**kwargs):
            nonlocal first_selected
            native = np.asarray(arr(native_register(*args,**kwargs)),np.float64).reshape(4,4)
            # Re-registrations after motion rejection should be controlled by
            # the temporal motion guard, not the initial world-upright prior.
            if first_selected:
                return native
            center = np.asarray(arr(est.get_tf_to_centered_mesh()),np.float64).reshape(4,4)
            poses = np.asarray(arr(est.poses),np.float64)
            if not np.allclose(poses[0] @ center, native, atol=1e-3,rtol=1e-3):
                raise RuntimeError('RGBTrack centered/public pose contract changed')
            rgb = kwargs.get('rgb', args[1] if len(args)>1 else None)
            mask = kwargs.get('ob_mask',args[3] if len(args)>3 else None)
            K = kwargs.get('K',args[0] if len(args)>0 else None)
            if rgb is None or mask is None or K is None:
                raise RuntimeError('register missing rgb/mask/K')
            chosen, ranked = select_first_dynamic_pose(
                poses=poses, scores=est.scores, center=center, vertices=self._mesh.vertices,
                mask=mask, K=K, base_R_camera=base_R_camera,
                upright_axis=upright_axis, max_tilt_deg=max_tilt_deg,
                max_rank=max_rank, min_iou=min_iou)
            audit = {'object':name,'axis':upright_axis,'max_tilt_deg':max_tilt_deg,
                     'max_rank':max_rank,'min_hull_iou':min_iou,'selected':chosen,
                     'eligible_sorted':ranked,
                     'note':'Upright applies only to FIRST registration; temporal guard checks subsequent tracking and recoveries.'}
            Path(output).mkdir(parents=True,exist_ok=True)
            (Path(output)/f'{name}_dynamic_init_candidates.json').write_text(
                json.dumps(audit,indent=2)+'\n',encoding='utf-8')
            # Compact first-frame visual QA with selected mesh projection.
            if chosen is not None:
                from rgbtrack_static_policy import mesh_projection_mask,overlay
                pub = poses[chosen['rank']]@center
                rendered=mesh_projection_mask(self._mesh.vertices,pub,K,mask.shape)
                annotated=overlay(rgb,mask,rendered,
                         f'{name} selected #{chosen["rank"]} axis={upright_axis} '
                         f'tilt={chosen["tilt_deg"]:.1f} IoU={chosen["mask_hull_iou"]:.3f}')
                cv2.imwrite(str(Path(output)/f'{name}_dynamic_init_selected.png'),
                            cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR))
            if chosen is None:
                raise RuntimeError(f'{name}: no upright first-frame RGBTrack candidate '
                    f'(axis={upright_axis}, max_tilt={max_tilt_deg}, min_iou={min_iou}); '
                    f'inspect {output}/{name}_dynamic_init_candidates.json. '
                    f'Check the model axis, do NOT force-flip pose output.')
            rank=chosen['rank']
            if rank!=0:
                import torch
                from scipy.spatial.transform import Rotation
                old=est.pose_last
                options={'device':old.device,'dtype':old.dtype} if isinstance(old,torch.Tensor) else {'dtype':torch.float32}
                est.pose_last=torch.as_tensor(poses[rank],**options)
                xyz=poses[rank,:3,3]
                euler=Rotation.from_matrix(poses[rank,:3,:3]).as_euler('xyz')
                est.tracker.initialize(xyz.reshape(3,1),euler.reshape(3,1))
                est.xyz=est.pose_last[:3,3]
                est.best_id=rank
                est.track_good=True
            first_selected=True
            selected_by_id[id(self)]=chosen
            print(f'[DYNAMIC INIT UPRIGHT] {name} axis={upright_axis} '
                  f'rank={rank} tilt={chosen["tilt_deg"]} IoU={chosen["mask_hull_iou"]}',flush=True)
            return poses[rank]@center

        est.register=register

    adapter.RGBTrackObject._ensure_estimator=ensure
    return selected_by_id
