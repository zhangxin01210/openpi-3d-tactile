#!/usr/bin/env python3
"""Static-object initializer used by the formal RGBTrack task pipeline.

Score RGBTrack registration candidates by upright + mask-projection agreement.

Dynamic objects use untouched RGBTrack. A trusted initial static pose is frozen.
The current repository's generalized adapter and shared RGB-D calibration are reused.
Use --max-frames 1 to inspect initial candidates before a full-episode run.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np


def arr(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def mesh_projection_mask(vertices, T, K, hw):
    """Fill projected convex hull of a box-like mesh. Suitable for this box,
    not a general concave-object silhouette renderer. Returns binary HxW mask.
    """
    vertices = np.asarray(vertices, np.float64)
    T = np.asarray(T, np.float64)
    K = np.asarray(K, np.float64)
    h, w = map(int, hw)
    out = np.zeros((h, w), np.uint8)
    xyz = vertices @ T[:3, :3].T + T[:3, 3]
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0.05):
        return out
    uv = np.column_stack((K[0, 0]*xyz[:, 0]/xyz[:, 2]+K[0, 2],
                          K[1, 1]*xyz[:, 1]/xyz[:, 2]+K[1, 2]))
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1.e5:
        return out
    hull = cv2.convexHull(np.rint(uv).astype(np.int32))
    if len(hull) >= 3:
        cv2.fillConvexPoly(out, hull, 1)
    return out


def tilt_degrees(poses, base_R_camera):
    R = np.asarray(poses)[:, :3, :3]
    z_cam = R[:, :, 2]
    z_base = z_cam @ np.asarray(base_R_camera, np.float64).T
    norms = np.linalg.norm(z_base, axis=1)
    return np.rad2deg(np.arccos(np.clip(z_base[:, 2] / np.maximum(norms, 1.e-12), -1., 1.)))


def evaluate_candidates(poses, scores, center, vertices, rgb_mask, K,
                        base_R_camera, max_tilt=25., max_rank=150,
                        min_iou=0.35):
    """Select candidate by visible 2D silhouette, retaining upright & RGBTrack rank.

    RGBTrack internally stores centered poses. Public pose = centered @ center.
    Never fabricate pose by rotating/flipping the final output.
    """
    p = np.asarray(arr(poses), np.float64)
    s = np.asarray(arr(scores), np.float64).reshape(-1)
    center = np.asarray(center, np.float64)
    mask = np.asarray(rgb_mask) > 0
    if p.ndim != 3 or p.shape[1:] != (4, 4) or len(s) != len(p):
        raise ValueError('Invalid RGBTrack candidate tensor shape')
    if not mask.any():
        raise ValueError('Static-object first-frame mask is empty')
    if center.shape != (4,4):
        raise ValueError('Invalid centered-mesh transform')
    if not 0 < max_tilt < 90 or max_rank < 1:
        raise ValueError('Invalid upright selection settings')
    angles = tilt_degrees(p, base_R_camera)
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    mask_center = np.array([xx.mean(), yy.mean()])
    equivalent_radius = np.sqrt(len(xx) / np.pi)
    entries = []
    for rank in range(min(max_rank, len(p))):
        if not np.isfinite(angles[rank]) or angles[rank] > max_tilt:
            continue
        public_pose = p[rank] @ center
        projected = mesh_projection_mask(vertices, public_pose, K, (h, w))
        pred_area = int(projected.sum())
        intersection = int(np.count_nonzero(projected & mask))
        union = pred_area + int(mask.sum()) - intersection
        iou = float(intersection / union) if union else 0.
        if pred_area:
            y, x = np.nonzero(projected)
            centroid = np.array([x.mean(), y.mean()])
            centroid_px = float(np.linalg.norm(centroid - mask_center))
        else:
            centroid_px = float('inf')
        size_ratio = pred_area / float(mask.sum())
        # IoU is primary; centroid and scorer's original ranking break close ties.
        center_term = 1. / (1. + centroid_px / max(equivalent_radius, 1.))
        rank_term = 1. - rank / max(len(p) - 1, 1)
        quality = 0.88*iou + 0.10*center_term + 0.02*rank_term
        entries.append({'rank': rank, 'tilt_deg': round(float(angles[rank]), 3),
                        'rgbtrack_score': round(float(s[rank]), 6),
                        'mask_iou': round(iou, 5), 'center_error_px': round(centroid_px, 3),
                        'projected_area_px': pred_area, 'area_ratio': round(size_ratio, 4),
                        'selection_quality': round(float(quality), 6)})
    entries.sort(key=lambda x: (-x['selection_quality'], x['rank']))
    selected = entries[0] if entries and entries[0]['mask_iou'] >= min_iou else None
    return selected, entries, angles


def overlay(rgb, mask, rendered_mask, caption):
    """RGB view: predicted mesh fill green, SAM2 contour magenta."""
    image = np.asarray(rgb, np.uint8).copy()
    pred = np.asarray(rendered_mask, bool)
    image[pred] = (0.55*image[pred] + 0.45*np.array([40, 230, 50])).astype(np.uint8)
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (245, 20, 245), 2)
    cv2.rectangle(image, (0, 0), (image.shape[1]-1, 36), (12, 14, 18), -1)
    cv2.putText(image, caption[:77], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.51,
                (255,255,255), 1, cv2.LINE_AA)
    return image


def write_review(output, object_name, rgb, mask, p, center, vertices, K, entries, angles):
    """Review eight alternatives in one contact sheet (including native rank 0)."""
    shortlist = [e['rank'] for e in entries[:7]]
    ranks = list(dict.fromkeys([0] + shortlist))[:8]
    panels = []
    for rank in ranks:
        public = p[rank] @ center
        projected = mesh_projection_mask(vertices, public, K, mask.shape)
        e = next((e for e in entries if e['rank'] == rank), None)
        desc = (f'#{rank} tilt={angles[rank]:.1f} IoU={e["mask_iou"]:.3f}' if e
                else f'#{rank} native tilt={angles[rank]:.1f} (NOT upright)')
        panels.append(overlay(rgb, mask, projected, desc))
    while len(panels) < 8:
        panels.append(np.zeros_like(panels[0]))
    sheet = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:8])])
    cv2.imwrite(str(output / f'{object_name}_init_candidates.png'),
                cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def install_static_policy(adapter, name, base_R_camera, output, max_tilt, max_rank, min_iou):
    original_ensure = adapter.RGBTrackObject._ensure_estimator
    original_estimate = adapter.RGBTrackObject.estimate
    installed = set()
    frozen = {}
    decisions = {}

    def ensure(self):
        original_ensure(self)
        if self.object_name != name or id(self) in installed:
            return
        installed.add(id(self))
        est = self._estimator
        original_register = est.register

        def register(*args, **kwargs):
            original_pose = np.asarray(arr(original_register(*args, **kwargs)), np.float64).reshape(4,4)
            center = np.asarray(arr(est.get_tf_to_centered_mesh()), np.float64).reshape(4,4)
            poses = np.asarray(arr(est.poses), np.float64)
            if not np.allclose(poses[0] @ center, original_pose, atol=1.e-3, rtol=1.e-3):
                raise RuntimeError('RGBTrack pose centering convention changed; not selecting')
            rgb = kwargs.get('rgb', args[1] if len(args)>1 else None)
            mask = kwargs.get('ob_mask', args[3] if len(args)>3 else None)
            K = kwargs.get('K', args[0] if len(args)>0 else None)
            if rgb is None or mask is None or K is None:
                raise RuntimeError('RGBTrack register missing RGB/mask/K')
            selected, entries, angles = evaluate_candidates(
                poses, est.scores, center, self._mesh.vertices, mask, K,
                base_R_camera, max_tilt=max_tilt, max_rank=max_rank, min_iou=min_iou)
            audit = {'static_object': name, 'selected': selected, 'max_tilt_deg': max_tilt,
                     'max_rank': max_rank, 'min_iou': min_iou,
                     'rgbtrack_score_note': 'original RGBTrack scorer, not mask-area confidence',
                     'candidate_silhouette': 'projected convex hull; use for box-like meshes only',
                     'eligible_sorted': entries}
            (output / f'{name}_init_candidates.json').write_text(json.dumps(audit, indent=2)+'\n')
            write_review(output, name, rgb, mask, poses, center, self._mesh.vertices, K,
                         entries, angles)
            if selected is None:
                raise RuntimeError(f'{name}: no convincing upright candidate with mask IoU >= {min_iou}; '
                                   f'see {output/name}_init_candidates.png')
            i = int(selected['rank'])
            public = poses[i] @ center
            if i != 0:
                import torch
                from scipy.spatial.transform import Rotation
                old = est.pose_last
                options = {'device': old.device, 'dtype': old.dtype} if isinstance(old, torch.Tensor) else {'dtype': torch.float32}
                est.pose_last = torch.as_tensor(poses[i], **options)
                xyz = poses[i, :3, 3]
                euler = Rotation.from_matrix(poses[i, :3, :3]).as_euler('xyz')
                est.tracker.initialize(xyz.reshape(3, 1), euler.reshape(3, 1))
                est.xyz = est.pose_last[:3, 3]
                est.best_id = i
                est.track_good = True
            decisions[id(self)] = selected
            print(f'[SELECT] {name} rank={i} tilt={selected["tilt_deg"]} '
                  f'IoU={selected["mask_iou"]} center_error={selected["center_error_px"]}px',flush=True)
            return public

        est.register = register

    def estimate(self, *, frame_index, rgb, depth_m, mask, K, T_base_color):
        if self.object_name != name:
            return original_estimate(self, frame_index=frame_index, rgb=rgb, depth_m=depth_m,
                                     mask=mask, K=K, T_base_color=T_base_color)
        if id(self) in frozen:
            first = frozen[id(self)]
            return replace(first, frame_index=int(frame_index),
                           T_color_object=first.T_color_object.copy(),
                           T_base_object=first.T_base_object.copy(),
                           mask_area_px=int(np.count_nonzero(mask)),
                           elapsed_ms=0., status='static_frozen')
        rec = original_estimate(self, frame_index=frame_index, rgb=rgb, depth_m=depth_m,
                                mask=mask, K=K, T_base_color=T_base_color)
        if rec.valid and id(self) in decisions:
            frozen[id(self)] = replace(rec, T_color_object=rec.T_color_object.copy(),
                                       T_base_object=rec.T_base_object.copy())
            rec.status = 'static_initial_observation'
            print(f'[STATIC FROZEN] {name} frame={frame_index}', flush=True)
        return rec

    adapter.RGBTrackObject._ensure_estimator = ensure
    adapter.RGBTrackObject.estimate = estimate
    return decisions, frozen


