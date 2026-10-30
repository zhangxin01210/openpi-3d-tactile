#!/usr/bin/env python3
"""Read-only geometric audit of calibrated funnel mesh + existing RGBTrack poses.

No CUDA/model inference, no modification of source output. All pose/frame
values are read from existing RGBTrack JSONL. This deliberately does not
'fix' a pose until mesh axes and initial physical orientation are confirmed.

Example:
 PYTHONPATH=src python scripts/object_geometry/pose_tracking/audit_funnel_axis_rgbtrack.py \
    --repo-root . --pose-run output/pose_eval/hand_scene_e23_metric_full_YYYYMMDD_HHMMSS \
    --output output/hand_scene_qa/funnel_axis_audit_e23
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def _angle_deg(a, b, undirected=False):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    a=a/max(np.linalg.norm(a),1e-12); b=b/max(np.linalg.norm(b),1e-12)
    dot=float(np.clip(np.dot(a,b),-1,1))
    if undirected: dot=abs(dot)
    return float(np.degrees(np.arccos(dot)))


def _proper_transform(raw, name):
    t=np.asarray(raw,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all(): raise ValueError(f'{name}: invalid matrix')
    if not np.allclose(t[3],[0,0,0,1],atol=1e-5): raise ValueError(f'{name}: bad SE3 footer')
    if np.linalg.norm(t[:3,:3].T @ t[:3,:3]-np.eye(3))>.04 or abs(np.linalg.det(t[:3,:3])-1)>.02:
        raise ValueError(f'{name}: rotation not SO(3)')
    return t


def _load_mesh(path):
    import trimesh
    scene=trimesh.load(str(path),force='scene',process=False)
    mesh=scene.dump(concatenate=True) if isinstance(scene,trimesh.Scene) else scene
    if not isinstance(mesh,trimesh.Trimesh) or len(mesh.vertices)==0: raise RuntimeError('Empty mesh')
    return mesh


def _mesh_geometry(mesh):
    verts=np.asarray(mesh.vertices,dtype=float)
    lo=verts.min(axis=0); hi=verts.max(axis=0)
    centered=verts-np.mean(verts,axis=0)
    vals,vecs=np.linalg.eigh(centered.T @ centered/max(1,len(verts)))
    pca_order=np.argsort(vals)[::-1]
    info={
      'bounds_low_m':lo.tolist(), 'bounds_high_m':hi.tolist(),
      'extents_m':(hi-lo).tolist(),
      'vertex_pca_eigenvalues':vals[pca_order].tolist(),
      'vertex_pca_axes_columns':vecs[:,pca_order].tolist(),
      'note':'PCA describes mesh vertex spread; it does NOT certify the physical funnel axis.',
      'local_axes_width_end_comparison':{}
    }
    # Radians of visible shell near each end. For a funnel the bowl-side radial
    # spread should normally be greater than stem-side spread. Descriptive, NOT a solver.
    for ax, name in enumerate('xyz'):
        length=hi[ax]-lo[ax]
        if length<1e-7:continue
        f=(verts[:,ax]-lo[ax])/length
        radial=np.linalg.norm(verts[:,[i for i in range(3) if i!=ax]],axis=1)
        out={}
        for part,mask in [('negative_end',f<.12),('positive_end',f>.88)]:
            out[part+'_n']=int(mask.sum())
            out[part+'_radial_p90_m']=float(np.percentile(radial[mask],90)) if mask.any() else None
        info['local_axes_width_end_comparison'][name]=out
    return info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--pose-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--funnel-asset',type=Path,default=None)
    ap.add_argument('--axis',choices=['x','y','z'],default='z',help='Candidate funnel long axis in mesh local frame, not assumed to be correct')
    ap.add_argument('--frames',default='0,100,141,144,151,155,203,358,467')
    ap.add_argument('--jump-deg',type=float,default=35.)
    ap.add_argument('--jump-m',type=float,default=.06)
    args=ap.parse_args()
    repo=args.repo_root.expanduser().resolve(); run=args.pose_run.expanduser().resolve(); out=args.output.expanduser().resolve()
    if out.exists(): raise FileExistsError(f'Refusing existing output: {out}')
    manifest=json.loads((run/'manifest.json').read_text('utf8'))
    asset=args.funnel_asset or repo/'assets/object_geometry/synthware_19_22/funnel_F181975.glb'
    asset=asset.expanduser().resolve()
    mconv=manifest['mesh_conversions']['funnel']
    saved=Path(mconv['source_glb']).resolve()
    if saved!=asset: raise RuntimeError(f'Mesh mismatch: pose run used {saved}, requested {asset}')
    if not np.isclose(float(mconv['scale_m_per_unit']),1.):raise RuntimeError('Pose run did not use metric asset scale 1.0')
    if 'source_sha256' in mconv:
        digest=hashlib.sha256(asset.read_bytes()).hexdigest()
        if digest!=mconv['source_sha256']:raise RuntimeError('Mesh SHA256 mismatch')
    geom=_mesh_geometry(_load_mesh(asset))
    axis_index='xyz'.index(args.axis)
    local_axis=np.eye(3)[:,axis_index]
    z=np.array([0.,0.,1.])
    entries={}
    for line in (run/'funnel_poses.jsonl').read_text('utf8').splitlines():
        if not line.strip():continue
        r=json.loads(line); f=int(r['frame_index'])
        if f in entries:raise ValueError(f'Duplicate frame {f}')
        entries[f]=r
    if not entries or 0 not in entries:raise RuntimeError('Missing frame-0 pose')
    requested=set(int(f) for f in args.frames.split(',') if f.strip())
    rows=[]; prev=None
    for frame in sorted(entries):
        row=entries[frame]
        valid=bool(row.get('valid',False));T=None
        if valid:
            T=_proper_transform(row['T_base_link_object'],f'frame {frame}')
        axis_base=T[:3,:3]@local_axis if T is not None else None
        tilt=_angle_deg(axis_base,z,undirected=True) if T is not None else None
        signed=_angle_deg(axis_base,z) if T is not None else None
        step_ang=None;step_axis=None;step_m=None
        if T is not None and prev is not None:
            Tp,fprev=prev
            step_axis=_angle_deg(Tp[:3,:3]@local_axis,axis_base,undirected=True)
            rr=Tp[:3,:3].T@T[:3,:3]
            step_ang=float(np.degrees(np.arccos(np.clip((np.trace(rr)-1)/2,-1,1))))
            step_m=float(np.linalg.norm(Tp[:3,3]-T[:3,3]))
        item={'frame':frame,'valid':valid,'status':row.get('status'),'reported_confidence':row.get('confidence'),
              'mask_area_px':row.get('mask_area_px'), 'axis_local':args.axis,
              'tilt_to_vertical_unsigned_deg':tilt,'tilt_to_vertical_signed_deg':signed,
              'rotation_step_deg':step_ang,'axis_step_unsigned_deg':step_axis,'translation_step_m':step_m,
              'suspected_jump':bool((step_ang is not None and step_ang>args.jump_deg) or (step_m is not None and step_m>args.jump_m))}
        rows.append(item)
        # Last valid observation, regardless of gaps; frame gap must be accounted separately.
        if T is not None:prev=(T,frame)
    frame0=_proper_transform(entries[0]['T_base_link_object'],'f0')
    axes={name:{'world_vector':(frame0[:3,:3]@np.eye(3)[:,i]).round(6).tolist(),
                'unsigned_tilt_from_vertical_deg':_angle_deg(frame0[:3,:3]@np.eye(3)[:,i],z,True)}
          for i,name in enumerate('xyz')}
    report={
      'schema_version':1,'input_pose_run':str(run),'mesh':str(asset),
      'mesh_geometry':geom,'initial_world_axes':axes,
      'frame_0_rotation_matrix':frame0[:3,:3].round(6).tolist(),
      'frame_0_translation_m':frame0[:3,3].round(6).tolist(),
      'evaluated_funnel_axis':args.axis,
      'f0_axis_unsigned_tilt_deg':_angle_deg(frame0[:3,:3]@local_axis,z,True),
      'f0_axis_signed_tilt_deg':_angle_deg(frame0[:3,:3]@local_axis,z,False),
      'suspected_jump_frames':[r['frame'] for r in rows if r['suspected_jump']],
      'frames_inspected':[r for r in rows if r['frame'] in requested],
      'important_notes':[
       'The local CAD axis must be verified from mesh shape (stem vs bowl).',
       'Unsigned tilt treats the axis as a line and does not detect stem-up vs bowl-up reversal.',
       'A high mask-area-based confidence is not a probability of correct 6D pose.',
       'Frame-to-frame jumps are warnings, not automatic proof of error: rapid true robot rotations exist.',
       'This tool does not alter poses; do not use its outputs as repaired training ground truth.'
      ]}
    out.mkdir(parents=True)
    (out/'axis_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf8')
    with (out/'pose_orientation_steps.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('== Mesh extent (m):',np.round(geom['extents_m'],4))
    print('== Mesh end widths:', json.dumps(geom['local_axes_width_end_comparison'],indent=2))
    print('== Frame-0 mesh local axes vs base_link vertical:')
    for axis,d in axes.items():print(f"   {axis}: {d['unsigned_tilt_from_vertical_deg']:.1f} deg, base_dir={d['world_vector']}")
    print('== Reviewed frames (relative to local',args.axis,')')
    for r in report['frames_inspected']:
        print(' f={:3d} valid={} tilt={} rot_step={} axis_step={} t_step={}'.format(
          r['frame'],r['valid'],
          round(r['tilt_to_vertical_unsigned_deg'],1) if r['tilt_to_vertical_unsigned_deg'] is not None else 'NA',
          round(r['rotation_step_deg'],1) if r['rotation_step_deg'] is not None else 'NA',
          round(r['axis_step_unsigned_deg'],1) if r['axis_step_unsigned_deg'] is not None else 'NA',
          round(r['translation_step_m'],3) if r['translation_step_m'] is not None else 'NA'))
    print('== Suspected jumps:',len(report['suspected_jump_frames']))
    print('== Wrote',out/'axis_audit.json','and',out/'pose_orientation_steps.csv')

if __name__=='__main__':main()
