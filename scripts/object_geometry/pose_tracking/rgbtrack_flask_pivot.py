#!/usr/bin/env python3
"""Episode 23 target-flask constraint: one upright RGBTrack registration + causal pivot prior.

DOES NOT re-run or modify the successful funnel trajectory. Initial flask pose
comes from RGBTrack's own refined/scored 252-pose candidates. Subsequent raw
flask poses from the existing full run are admitted only if consistent with a
nearly fixed flask rotating slightly around its bottom. An invalid observation
falls back to the last constrained pose with an explicit PRIOR status.

All generated JSONL outputs remain compatible with visualize_rgbtrack_full_episode.py.
Not a source of metrically precise flask ground truth. Camera and mesh provenance
are inherited from the original run and checked against the new registration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

import rgbtrack_adapter as adapter


def as_array(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value, dtype=np.float64)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty pose JSONL: {path}')
    return rows


def vec_unit(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError('Near-zero/invalid direction')
    return a / n


def deg_between(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(vec_unit(a), vec_unit(b)), -1, 1))))


def z_end_stats(verts: np.ndarray) -> dict:
    """Infer the flask's mouth end: narrower end along local Z is likely neck.

    This is a HEURISTIC; expose a manual override if the mesh is ambiguous.
    """
    vertices = np.asarray(verts, dtype=np.float64)
    xyz_min = vertices.min(axis=0)
    xyz_max = vertices.max(axis=0)
    zmin, zmax = float(xyz_min[2]), float(xyz_max[2])
    slab = max((zmax-zmin)*0.14, 0.002)
    xy_ctr = 0.5*(xyz_min[:2]+xyz_max[:2])
    neg = vertices[vertices[:,2] <= zmin+slab]
    pos = vertices[vertices[:,2] >= zmax-slab]
    def p90(points):
        return float(np.percentile(np.linalg.norm(points[:,:2]-xy_ctr, axis=1),90)) if len(points)>5 else float('nan')
    nwidth, pwidth = p90(neg), p90(pos)
    if not np.isfinite(nwidth+pwidth):
        raise ValueError('Cannot infer flask Z-end widths; specify --flask-mouth-axis')
    mouth_sign = +1 if pwidth < nwidth else -1
    ambiguous = max(nwidth,pwidth)/max(min(nwidth,pwidth),1e-9) < 1.12
    pivot = np.array([*xy_ctr, zmin if mouth_sign > 0 else zmax], dtype=np.float64)
    return {'mouth_sign':mouth_sign, 'ambiguous':ambiguous,
            'negative_z_radial_p90_m':nwidth, 'positive_z_radial_p90_m':pwidth,
            'local_bottom_pivot_m':pivot.tolist(), 'z_min_m':zmin, 'z_max_m':zmax}


def rot_from_axis_change(a, b):
    """Minimal Rodrigues rotation a->b, preserving initial twist around flask axis."""
    a, b = vec_unit(a), vec_unit(b)
    v = np.cross(a,b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a,b),-1,1))
    if s<1e-10:
        if c<0:
            raise ValueError('Opposite axis disallowed by physical gates')
        return np.eye(3)
    K = np.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]], dtype=np.float64) / s
    return np.eye(3) + s*K + (1-c)*(K@K)


def project_anchor_rotation(T0, pivot_local, current_axis, first_axis):
    """Project an axis measurement onto fixed-pivot motion without spurious yaw."""
    R = rot_from_axis_change(first_axis, current_axis) @ T0[:3,:3]
    out = np.eye(4, dtype=np.float64)
    out[:3,:3] = R
    fixed_pivot = T0[:3,:3] @ pivot_local + T0[:3,3]
    out[:3,3] = fixed_pivot - R @ pivot_local
    return out


def compute_target_records(raw, initial, mesh_verts, sign, args):
    """Pure NumPy deterministic filter; independent of CUDA, video and RGBTrack."""
    T0 = np.asarray(initial['T_base_link_object'], dtype=float).reshape(4,4)
    if not np.isfinite(T0).all() or not initial.get('valid',False):
        raise ValueError('Flask upright initialization must be finite and valid')
    info = z_end_stats(mesh_verts)
    pivot = np.asarray(info['local_bottom_pivot_m'],dtype=float)
    # Override inferred mouth sign and pivot if user specified a different end.
    if sign != info['mouth_sign']:
        pivot[2] = info['z_min_m'] if sign>0 else info['z_max_m']
    axis_local = np.array([0.,0.,float(sign)])
    axis0 = vec_unit(T0[:3,:3] @ axis_local)
    if deg_between(axis0,[0,0,1]) > args.flask_max_tilt_deg:
        raise ValueError('Initialization not upright: verify local mouth axis selection')
    fixed_pivot = T0[:3,:3] @ pivot + T0[:3,3]
    T_base_color = T0 @ np.linalg.inv(np.asarray(initial['T_color_object'],float).reshape(4,4))
    T_color_base = np.linalg.inv(T_base_color)
    previous_axis=axis0.copy()
    last=T0.copy()
    output=[]; qa=[]; stats=Counter()
    for index, original in enumerate(raw):
        frame = int(original['frame_index'])
        if index and frame <= int(raw[index-1]['frame_index']):
            raise ValueError('Non-increasing frame indices')
        t = np.asarray(original['T_base_link_object'],dtype=float).reshape(4,4)
        visible_valid = bool(original.get('valid',False)) and np.isfinite(t).all()
        raw_axis = vec_unit(t[:3,:3] @ axis_local) if visible_valid else None
        tilt = deg_between(raw_axis,[0,0,1]) if visible_valid else float('nan')
        axis_jump = deg_between(raw_axis,previous_axis) if visible_valid else float('nan')
        raw_pivot=t[:3,:3]@pivot+t[:3,3] if visible_valid else None
        pivot_shift = float(np.linalg.norm(raw_pivot-fixed_pivot)) if visible_valid else float('nan')
        if frame == int(initial['frame_index']):
            refined=T0.copy()
            status='flask_upright_anchor_observed'
            confidence=min(float(initial.get('confidence',0.2)),0.45)
            stats['anchor']+=1
        else:
            accept = (visible_valid and tilt <= args.flask_max_tilt_deg and
                      axis_jump <= args.flask_max_axis_jump_deg and
                      pivot_shift <= args.flask_max_pivot_shift_m)
            if accept:
                blended_axis=vec_unit((1.-args.flask_axis_alpha)*previous_axis+args.flask_axis_alpha*raw_axis)
                refined=project_anchor_rotation(T0,pivot,blended_axis,axis0)
                previous_axis=blended_axis
                last=refined.copy()
                status='flask_pivot_visual_conditioned'
                confidence=min(float(original.get('confidence',0.1)),0.35)
                stats['visual_conditioned']+=1
            else:
                refined=last.copy()
                if not visible_valid: reason='raw_invalid'
                elif tilt > args.flask_max_tilt_deg: reason='tilt'
                elif pivot_shift > args.flask_max_pivot_shift_m: reason='pivot_shift'
                else: reason='axis_jump'
                status='flask_pivot_prior_'+reason
                confidence=0.08
                stats['prior_'+reason]+=1
        if frame==int(initial['frame_index']):
            last=refined.copy()
        record=dict(original)
        record.update({
            'T_base_link_object':refined.tolist(),
            'T_color_object':(T_color_base@refined).tolist(),
            'valid':True,  # Usable GEOMETRIC PRIOR; not necessarily observed!
            'status':status,
            'confidence':round(float(confidence),4),
            'pose_source':'visual_constrained' if 'observed' in status or 'visual_conditioned' in status else 'geometric_prior',
            'visual_measurement_accepted':status in ('flask_upright_anchor_observed','flask_pivot_visual_conditioned'),
            'raw_visual_status':original.get('status'),
            'raw_visual_valid':bool(original.get('valid',False)),
        })
        output.append(record)
        qa.append({'frame_index':frame, 'status':status,
                   'tilt_raw_deg':round(tilt,3) if np.isfinite(tilt) else None,
                   'axis_jump_raw_deg':round(axis_jump,3) if np.isfinite(axis_jump) else None,
                   'pivot_shift_raw_m':round(pivot_shift,4) if np.isfinite(pivot_shift) else None})
    report = {'summary':dict(stats), 'initial_pivot_base_m':fixed_pivot.tolist(),
              'initial_mouth_axis_base':axis0.tolist(),
              'initial_axis_tilt_deg':deg_between(axis0,[0,0,1]),
              'inferred_mesh_geometry':info,
              'mouth_axis_used_local':('+z' if sign>0 else '-z'),
              'assumption':'flask bottom pivot approximately stationary; small axis tilt allowed',
              'warning':'PRIOR frames are not observed 6D poses. valid=True means usable geometry only. Use pose_source and confidence for training.'}
    return output, qa, report


def install_flask_upright_register(est, mesh_vertices, R_base_cam, *,
                                   flask_mouth_axis='auto', flask_init_tilt_deg=20.,
                                   flask_upright_max_rank=180, decisions=None):
    """Shared TRAINING/DEPLOYMENT frame-0 flask candidate selection.

    A wrapped RGBTrack register still uses its scored/refined hypotheses. Only
    one selected upright candidate replaces its top-ranked pose and tracker
    state. Never use this as an unrestricted pose optimizer or as review proof.
    """
    if decisions is None:
        decisions = []
    width=z_end_stats(np.asarray(mesh_vertices,dtype=float))
    if flask_mouth_axis=='auto':
        sign=width['mouth_sign']
        if width['ambiguous']:
            print('[FLASK WARNING] mouth direction ambiguous; confirm candidate image',flush=True)
    elif flask_mouth_axis in ('+z','-z'):
        sign=+1 if flask_mouth_axis=='+z' else -1
    else:
        raise ValueError('flask_mouth_axis must be auto, +z, or -z')
    if not 0 < float(flask_init_tilt_deg) < 90 or int(flask_upright_max_rank)<1:
        raise ValueError('Invalid flask upright selection thresholds')
    decision={'mouth_sign':sign,'mesh_axis_inference':width}
    decisions.append(decision)
    orig_register=est.register

    def constrained_register(*rargs,**rkwargs):
        default=orig_register(*rargs,**rkwargs)
        poses=as_array(est.poses)
        scores=as_array(est.scores).reshape(-1)
        C=as_array(est.get_tf_to_centered_mesh())
        if poses.shape[1:]!=(4,4) or len(poses)!=len(scores):
            raise RuntimeError('Unexpected RGBTrack candidate set')
        if not np.allclose(poses[0]@C,as_array(default),rtol=1e-3,atol=1e-3):
            raise RuntimeError('Internal centered-mesh transform differs')
        axis_local=np.array([0.,0.,float(sign)])
        axes=np.einsum('ij,njk,k->ni',R_base_cam,poses[:,:3,:3],axis_local)
        tilts=np.degrees(np.arccos(np.clip(axes[:,2]/np.maximum(np.linalg.norm(axes,axis=1),1e-9),-1,1)))
        eligible=np.flatnonzero(tilts[:flask_upright_max_rank]<=flask_init_tilt_deg)
        if not len(eligible):
            raise RuntimeError('No upright target flask candidate in top-ranked set; inspect mesh axis')
        rank=int(eligible[0])
        chosen=poses[rank]
        selected_public=chosen@C
        import torch
        from scipy.spatial.transform import Rotation
        est.pose_last=torch.as_tensor(chosen,device='cuda',dtype=torch.float32)
        est.xyz=est.pose_last[:3,3]
        est.tracker.initialize(chosen[:3,3].reshape(3,1),Rotation.from_matrix(chosen[:3,:3]).as_euler('xyz').reshape(3,1))
        est.best_id=rank
        est.track_good=True
        decision.update({'chosen_rank':rank,'chosen_tilt_deg':float(tilts[rank]),
                         'default_tilt_deg':float(tilts[0]),
                         'chosen_raw_score':float(scores[rank]),'default_raw_score':float(scores[0]),
                         'eligible_in_top_rank':int(len(eligible)), 'candidate_count':len(poses)})
        print(f'[FLASK INIT] target upright rank={rank} tilt={tilts[rank]:.2f}deg vs default {tilts[0]:.2f}deg',flush=True)
        return selected_public
    est.register=constrained_register
    return decision



def run(args):
    repo=Path(args.repo_root).resolve()
    source=Path(args.source_pose_run).resolve()
    output=Path(args.output).resolve()
    init_output=output/'target_init'
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite: {output}')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if int(manifest.get('episode',-1)) != int(args.episode):
        raise ValueError('Source pose run is not the requested episode')
    raw_funnel=source/'funnel_poses.jsonl'
    raw_target=source/'target_flask_poses.jsonl'
    if not raw_funnel.is_file():raise FileNotFoundError(raw_funnel)
    src_target=load_jsonl(raw_target)
    full_args=adapter.build_arg_parser().parse_args([
        '--repo-root',str(repo), '--dataset',str(args.dataset), '--episode',str(args.episode),
        '--mask-root',str(args.mask_root),
        '--funnel-glb',str(args.funnel_glb), '--flask-glb',str(args.flask_glb),
        '--scale-manifest',str(args.scale_manifest),
        '--front-calibration-profile',str(args.front_calibration_profile),
        '--output',str(init_output), '--objects','target_flask',
        '--mode','depth','--max-frames','1','--viz-frames','0'])
    calib=adapter.load_front_calibration(repo,args.front_calibration_profile)
    R_base_cam=np.asarray(calib.T_base_color,float)[:3,:3]
    ensure_orig=adapter.RGBTrackObject._ensure_estimator
    decisions=[]
    hooked=set()
    def ensure_flask_upright(self):
        ensure_orig(self)
        if self.object_name!='target_flask' or id(self._estimator) in hooked:return
        est=self._estimator
        hooked.add(id(est))
        install_flask_upright_register(
            est, self._mesh.vertices, R_base_cam,
            flask_mouth_axis=args.flask_mouth_axis,
            flask_init_tilt_deg=args.flask_init_tilt_deg,
            flask_upright_max_rank=args.flask_upright_max_rank,
            decisions=decisions,
        )
    adapter.RGBTrackObject._ensure_estimator=ensure_flask_upright
    try:
        adapter.run(full_args)
    finally:
        adapter.RGBTrackObject._ensure_estimator=ensure_orig
    if not decisions or 'chosen_rank' not in decisions[-1]:
        raise RuntimeError('Target flask upright registration did not execute')
    original_init=load_jsonl(init_output/'target_flask_poses.jsonl')[0]
    init_manifest=json.loads((init_output/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('calibration') != init_manifest.get('calibration'):
        raise RuntimeError('Calibration provenance mismatch between original full pose and target reinitialization')
    mesh_file=Path(init_manifest['mesh_conversions']['target_flask']['output_mesh'])
    import trimesh
    mesh=trimesh.load(str(mesh_file),force='mesh',process=False)
    clean,qa,report=compute_target_records(src_target,original_init,np.asarray(mesh.vertices),decisions[-1]['mouth_sign'],args)
    with (output/'target_flask_poses.jsonl').open('w',encoding='utf8') as f:
        for row in clean:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    shutil.copy2(raw_funnel,output/'funnel_poses.jsonl')
    source_manifest=dict(manifest)
    source_manifest['flask_pivot_constraint']={
        'source_full_pose_run':str(source), 'target_initial_pose_run':str(init_output),
        'method':'causal fixed bottom pivot, small mouth axis tilt, no axial yaw drift',
        'target_pose_source_mixed':True, 'flask_pose_not_ground_truth':True,
        'params':{key:getattr(args,key) for key in
            ('flask_max_tilt_deg','flask_max_axis_jump_deg','flask_max_pivot_shift_m','flask_axis_alpha')},
    }
    (output/'manifest.json').write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report['init_candidate']=decisions[-1]
    report['source_pose_run']=str(source)
    report['episode']=args.episode
    report['params']={key:getattr(args,key) for key in
        ('flask_max_tilt_deg','flask_max_axis_jump_deg','flask_max_pivot_shift_m','flask_axis_alpha')}
    report['num_frames']=len(clean)
    (output/'flask_pivot_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (output/'flask_pivot_qa.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(qa[0]));w.writeheader();w.writerows(qa)
    if len(load_jsonl(output/'funnel_poses.jsonl')) != len(clean):
        raise ValueError('Input funnel/target frame counts differ')
    print('[FLASK PIVOT] original funnel pose bytes preserved:',
          raw_funnel.read_bytes()==(output/'funnel_poses.jsonl').read_bytes(),flush=True)
    print('[FLASK PIVOT] flask pose status counts:',dict(report['summary']),flush=True)
    print('[FLASK PIVOT] output:',output,flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--mask-root',type=Path,required=True)
    p.add_argument('--source-pose-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--funnel-glb',type=Path,required=True)
    p.add_argument('--flask-glb',type=Path,required=True)
    p.add_argument('--scale-manifest',type=Path,required=True)
    p.add_argument('--front-calibration-profile',type=Path,required=True)
    p.add_argument('--flask-mouth-axis',choices=('auto','+z','-z'),default='auto')
    p.add_argument('--flask-init-tilt-deg',type=float,default=20.)
    p.add_argument('--flask-upright-max-rank',type=int,default=180)
    p.add_argument('--flask-max-tilt-deg',type=float,default=20.)
    p.add_argument('--flask-max-axis-jump-deg',type=float,default=12.)
    p.add_argument('--flask-max-pivot-shift-m',type=float,default=0.025)
    p.add_argument('--flask-axis-alpha',type=float,default=0.25)
    args=p.parse_args()
    if not 0.0<args.flask_axis_alpha<=1.0:p.error('--flask-axis-alpha must be in (0,1]')
    if not 0.0<args.flask_init_tilt_deg<90:p.error('invalid init tilt')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
