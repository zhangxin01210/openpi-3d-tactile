#!/usr/bin/env python3
"""Read-only HTML QA: old/new 1018 spatial point clouds vs accepted legacy object geometry.

Example:
 PYTHONPATH=src .venv/bin/python scripts/preprocess/inspect_1018_e23_spatial_alignment.py \
   --dataset data/1018_peg_all --episode 23 --frames 0,120,240,360,467 \
   --new-spatial v1_probe_1018 --old-spatial v1_front_candidate \
   --object-version v1_hand_scene_q2

Writes HTML and report.json under output/preprocess/1018_peg_all/migration_audit/overlay_e23.
No source or derived sidecars are changed. Does not evaluate RGBTrack metric accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from openpi.spatial_dataset.derived import SpatialDerivedDataset
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('data/1018_peg_all'))
    p.add_argument('--episode', type=int, default=23)
    p.add_argument('--frames', default='0,120,240,360,467')
    p.add_argument('--old-spatial', default='v1_front_candidate')
    p.add_argument('--new-spatial', default='v1_probe_1018')
    p.add_argument('--object-version', default='v1_hand_scene_q2')
    p.add_argument('--output', type=Path, default=Path('output/preprocess/1018_peg_all/migration_audit/overlay_e23'))
    return p.parse_args()


def stats(x):
    a=np.asarray(x, dtype=np.float64)
    return {'min_m': np.round(a.min(0), 5).tolist(), 'max_m': np.round(a.max(0), 5).tolist(), 'centroid_m': np.round(a.mean(0), 5).tolist()}


def scatter(go, arr, name, color, size, opacity=1., skip=1):
    a=np.asarray(arr, dtype=np.float64)[::skip]
    return go.Scatter3d(x=a[:,0], y=a[:,1], z=a[:,2], name=name, mode='markers',
           marker={'size': size, 'color':color, 'opacity':opacity},
           hovertemplate=(name+'<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>'))


def main():
    args=parse_args()
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit('Requires plotly for HTML QA (existing spatial viewer also uses it).') from e
    frames=[int(s.strip()) for s in args.frames.split(',') if s.strip()]
    if not frames or len(frames)!=len(set(frames)) or any(f<0 for f in frames):
        raise ValueError('Provide unique, nonnegative frame IDs')
    old=SpatialDerivedDataset(args.dataset, version=args.old_spatial)
    new=SpatialDerivedDataset(args.dataset, version=args.new_spatial)
    obj=ObjectGeometryDerivedDataset(args.dataset, version=args.object_version)
    for label, ds in [('old',old),('new',new)]:
        if ds.coordinate_frame != 'base_link' or ds.xyz_unit != 'm':
            raise ValueError(f'{label}: unexpected spatial units/frame {ds.coordinate_frame}/{ds.xyz_unit}')
        if ds.camera_roles != ('front',):
            raise ValueError(f'{label}: expected front-only cameras, saw {ds.camera_roles}')
    out=args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report={'episode':args.episode,'versions':{'old':args.old_spatial,'new':args.new_spatial,'object':args.object_version},'frames':[]}
    for frame in frames:
        a=old.get(episode_index=args.episode,frame_index=frame)
        b=new.get(episode_index=args.episode,frame_index=frame)
        o=obj.get(episode_index=args.episode,frame_index=frame)
        if a.frame_index != frame or b.frame_index != frame or o.frame_index != frame:
            raise ValueError('Frame join mismatch')
        if abs(float(a.timestamp_s)-float(b.timestamp_s))>1e-5:
            raise ValueError(f'Old/new spatial timestamp mismatch f{frame}: {a.timestamp_s} vs {b.timestamp_s}')
        manipulated=np.asarray(o.manipulated_xyz_m)[np.asarray(o.manipulated_mask,dtype=bool)]
        target=np.asarray(o.target_xyz_m)[np.asarray(o.target_mask,dtype=bool)]
        fig=go.Figure()
        fig.add_trace(scatter(go,a.visual_xyz_m, 'OLD spatial - grey', '#a3a3a3', 1.6, 0.25))
        fig.add_trace(scatter(go,b.visual_xyz_m, 'NEW spatial - blue', '#2474ef', 1.8, 0.5))
        fig.add_trace(scatter(go,b.tactile_xyz_m, 'XHand tactile taxels', '#9329cc',2.6,0.8))
        if len(manipulated):fig.add_trace(scatter(go,manipulated,'Legacy funnel geometry', '#f0781e',3.3))
        if len(target):fig.add_trace(scatter(go,target,'Legacy target flask geometry','#2d9b4b',3.3))
        fig.update_layout(title=f'1018 episode {args.episode} frame {frame}: NEW/OLD spatial + OLD object',
            scene={'xaxis_title':'base_link X (m)','yaxis_title':'base_link Y (m)',
                   'zaxis_title':'base_link Z (m)','aspectmode':'data'},
            legend={'orientation':'h','y':-0.06}, margin={'l':0,'r':0,'t':45,'b':0})
        html=out/f'e{args.episode:05d}_f{frame:05d}.html'
        fig.write_html(str(html), include_plotlyjs=True)
        old_points=np.asarray(a.visual_xyz_m)
        new_points=np.asarray(b.visual_xyz_m)
        same_shape=old_points.shape==new_points.shape
        row_max_abs=(float(np.max(np.abs(old_points-new_points))) if same_shape else None)
        row_rmse=(float(np.sqrt(np.mean((old_points-new_points)**2))) if same_shape else None)
        r={'frame':frame,'timestamp_s':float(b.timestamp_s),'old_scene':stats(old_points),'new_scene':stats(new_points),
           'same_order_per_point_max_abs_m':row_max_abs,'same_order_per_point_rmse_m':row_rmse,
           'note':'Same-order comparison is diagnostic only; sampling order may vary.',
           'funnel_valid_points':int(len(manipulated)),'flask_valid_points':int(len(target)),'html':str(html)}
        report['frames'].append(r)
        print(f'[FRAME] {frame} old/new scene max paired delta {row_max_abs} m; funnel={len(manipulated)} flask={len(target)} -> {html}',flush=True)
    path=out/'report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('[DONE]',len(frames),'frames; report:',path)
    print('[IMPORTANT] Visual agreement alone does not prove registration accuracy; compare original front RGB masks/overlays too.')

if __name__=='__main__':main()
