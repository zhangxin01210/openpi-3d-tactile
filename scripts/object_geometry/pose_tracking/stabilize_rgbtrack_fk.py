#!/usr/bin/env python3
"""Create a NON-DESTRUCTIVE, auditable FK-gated RGBTrack pose proposal.

This is a candidate track for human review, NOT new ground truth. Uses previously
registered metric mesh poses and the matching dataset's measured robot joints.
The RGBTrack pose at --grasp-anchor (when the object is ALREADY stably held) is
calibrated into hand-local coordinates; this intentionally does NOT reuse the
f0-as-grasp-anchor assumption previously found unreliable.

After grasp: predict motion by FK; apply a small RGBTrack correction only if
its object-axis direction and translation agree with the prediction. Twist
around the funnel local z-axis is symmetry-normalized to the predicted yaw.

Output is a NEW RGBTrack-compatible pose-run directory: pass it unchanged to
visualize_rgbtrack_full_episode.py. Never edits input poses, robot data, or
training sidecars. The target can be locked to frame 0 as an uncertain prior.

Example:
    PYTHONPATH=src .venv/bin/python scripts/object_geometry/pose_tracking/stabilize_rgbtrack_fk.py \\
      --repo-root . --dataset data/1018_peg_all --episode 23 \\
      --pose-run output/pose_eval/hand_scene_e23_metric_full_XXX \\
      --output output/pose_eval/hand_scene_e23_fk_fused_XXX \\
      --grasp-anchor 141 --target-policy static
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def check_T(t, name):
    t = np.asarray(t, np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f'{name} not finite 4x4 SE(3)')
    R = t[:3, :3]
    if np.linalg.norm(R.T @ R - np.eye(3)) > 0.04 or abs(np.linalg.det(R) - 1) > 0.02:
        raise ValueError(f'{name} rotation not SO(3)')
    return t


def read_rows(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        f = int(row['frame_index'])
        if f in rows: raise ValueError(f'{path}: duplicate frame {f}')
        if row.get('valid') and row.get('status') == 'observed':
            row['_T'] = check_T(row['T_base_link_object'], f'{path} f{f}')
        elif row.get('valid'):
            # Predicted/fallback RGBTrack poses should not update FK state.
            row['_T'] = None
        else:
            row['_T'] = None
        rows[f] = row
    if not rows: raise ValueError(f'No poses in {path}')
    return rows


def z_spin(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def twist_to_reference(R_meas, R_ref):
    """Choose local-z symmetry rotation minimizing SO(3) distance to R_ref."""
    A = R_ref.T @ R_meas
    theta = np.arctan2(A[0,1]-A[1,0], A[0,0]+A[1,1])
    return R_meas @ z_spin(theta), float(np.rad2deg(theta))


def axis_deg(R1, R2):
    v1, v2 = R1[:,2], R2[:,2]
    return float(np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1, 1))))


def so3_deg(R1, R2):
    a = R1.T @ R2
    return float(np.rad2deg(np.arccos(np.clip((np.trace(a)-1)/2, -1, 1))))


def blend_transform(a, b, w):
    from scipy.spatial.transform import Rotation, Slerp
    t = np.eye(4)
    t[:3,3] = (1-w)*a[:3,3] + w*b[:3,3]
    rotations = Rotation.from_matrix(np.stack([a[:3,:3], b[:3,:3]]))
    t[:3,:3] = Slerp([0, 1], rotations)([w]).as_matrix()[0]
    return t


def dump_rows(path, rows):
    with Path(path).open('w', encoding='utf8') as f:
        for frame in sorted(rows):
            row = {k:v for k,v in rows[frame].items() if k != '_T'}
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--episode', type=int, default=23)
    p.add_argument('--pose-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--grasp-anchor', type=int, required=True,
                   help='Verified frame with rigid grasp AND reliable measured funnel pose. Do not use f0 by default.')
    p.add_argument('--release-frame', type=int, default=None,
                   help='After this frame, rigid-grasp model not valid; output raw observations only')
    p.add_argument('--hand-link', default='right_hand_ee_link')
    p.add_argument('--max-translation-residual-m', type=float, default=0.025)
    p.add_argument('--max-axis-residual-deg', type=float, default=22.0)
    p.add_argument('--observation-weight', type=float, default=0.20)
    p.add_argument('--target-policy', choices=('static','raw'), default='static',
                   help='Static prior is not a claim flask cannot move on collision.')
    a=p.parse_args()
    if not 0 <= a.observation_weight <= 1: p.error('observation-weight must be [0,1]')
    if a.max_translation_residual_m <= 0 or a.max_axis_residual_deg <= 0: p.error('Invalid gating thresholds')
    repo = a.repo_root.expanduser().resolve()
    data = a.dataset.expanduser().resolve()
    run = a.pose_run.expanduser().resolve()
    out = a.output.expanduser().resolve()
    if out.exists(): raise FileExistsError(f'Refusing existing output: {out}')
    source_manifest=json.loads((run/'manifest.json').read_text(encoding='utf8'))
    if int(source_manifest.get('episode',-1)) != a.episode: raise ValueError('Episode mismatch')
    if Path(source_manifest.get('dataset','/nonexistent')).resolve() != data: raise ValueError('Dataset mismatch')
    for name in ('funnel','target_flask'):
        conv=source_manifest['mesh_conversions'][name]
        if not np.isclose(float(conv['scale_m_per_unit']),1.0): raise ValueError(f'{name}: not 1.0 metric scale')
        if not Path(conv['source_glb']).is_file(): raise FileNotFoundError(conv['source_glb'])
    poses={name:read_rows(run/f'{name}_poses.jsonl') for name in ('funnel','target_flask')}
    frames=sorted(poses['funnel'])
    if frames != sorted(poses['target_flask']) or frames != list(range(len(frames))):
        raise ValueError('Incomplete or misaligned pose frame streams')
    if a.grasp_anchor not in poses['funnel']: raise ValueError('Grasp anchor outside pose frames')
    if poses['funnel'][a.grasp_anchor]['_T'] is None:
        raise ValueError('Grasp anchor is not an observed, valid RGBTrack pose')
    if poses['funnel'][0]['_T'] is None or poses['target_flask'][0]['_T'] is None:
        raise ValueError('Both f0 poses must be observed and valid')
    if a.release_frame is not None and a.release_frame <= a.grasp_anchor:
        raise ValueError('release-frame must be later than grasp-anchor')
    # Exact RGBTrack calibration (constant front camera) recovered from saved
    # synchronized base/color frames, avoiding silent calibration switching.
    original = poses['funnel'][0]
    T_base_color = check_T(original['T_base_link_object'], 'f0 T_base_obj') @ np.linalg.inv(check_T(original['T_color_object'], 'f0 T_color_obj'))
    from openpi.spatial_dataset.source import RawSpatialDataset
    from openpi.spatial.kinematics import RobotKinematics, load_state_mapping_csv, joint_positions_from_state
    fk=RobotKinematics(repo/'configs/ur7e_xhand/ur7e_xhand_verified.urdf')
    if a.hand_link not in fk.links: raise ValueError(f'Unknown hand link {a.hand_link}')
    mapping=load_state_mapping_csv(repo/'configs/ur7e_xhand/state_mapping.csv')
    ds=RawSpatialDataset(data, episode=a.episode)
    hands={}
    for row in ds.iter_rows(depth_roles=()):
        f=int(row['frame_index'])
        q=joint_positions_from_state(np.asarray(row['observation.state']).reshape(-1), mapping=mapping, kinematics=fk)
        hands[f]=check_T(fk.compute(q, root='base_link')[a.hand_link], f'hand f{f}')
    if sorted(hands) != frames: raise ValueError('FK frames differ from tracked pose frames')
    hand_object=np.linalg.inv(hands[a.grasp_anchor]) @ poses['funnel'][a.grasp_anchor]['_T']
    target_f0=poses['target_flask'][0]['_T']
    new={'funnel':{},'target_flask':{}}
    audit=[]
    verdicts={}
    for f in frames:
        raw = poses['funnel'][f]
        rawT=raw['_T']
        stat='raw_before_grasp'
        applied=None
        trans_res=None
        axis_res=None
        twist_deg=None
        Rjump = None
        if f < a.grasp_anchor:
            if rawT is not None:
                T=rawT
                stat='rgbtrack_observed_pregrasp'
            else:
                T=None
                stat='pregrasp_rgbtrack_invalid'
        elif a.release_frame is not None and f > a.release_frame:
            T=rawT
            stat='postrelease_rgbtrack_observed' if rawT is not None else 'postrelease_no_valid_observation'
        else:
            predicted=hands[f] @ hand_object
            T=predicted
            if f == a.grasp_anchor:
                stat='hand_funnel_anchor_observed'
                applied=True
            elif rawT is None:
                stat='fk_prediction_no_observation'
                applied=False
            else:
                # Object is approximately symmetric around its local Z axis:
                # choose the equivalent twist that is closest to FK prior.
                aligned_R,twist_deg=twist_to_reference(rawT[:3,:3], predicted[:3,:3])
                aligned=rawT.copy()
                aligned[:3,:3]=aligned_R
                trans_res=float(np.linalg.norm(rawT[:3,3]-predicted[:3,3]))
                axis_res=axis_deg(aligned_R,predicted[:3,:3])
                Rjump=so3_deg(aligned_R,predicted[:3,:3])
                applied=(trans_res <= a.max_translation_residual_m and
                         axis_res <= a.max_axis_residual_deg and
                         Rjump <= a.max_axis_residual_deg+7)
                if applied:
                    T=blend_transform(predicted,aligned,a.observation_weight)
                    stat='fk_visual_gated_fusion'
                else:
                    stat='fk_prediction_rejected_visual_jump'
            # Always use corrected pose for next FK relative offset; if visual
            # agrees, it can slowly track real grip pose drift, never from a jump.
            hand_object=np.linalg.inv(hands[f]) @ T
        row={k:v for k,v in raw.items() if k!='_T'}
        if T is None:
            row['valid']=False
        else:
            row['valid']=True
            row['T_base_link_object']=check_T(T,'fused T').tolist()
            row['T_color_object']=(np.linalg.inv(T_base_color)@T).tolist()
        row.update(status=stat,source='rgbtrack_fk_audit_postprocess',
                   rgbtrack_original_valid=bool(raw.get('valid')),
                   rgbtrack_original_status=str(raw.get('status')),
                   pose_is_physical_ground_truth=False)
        new['funnel'][f]=row
        audit.append({'frame':f,'status':stat,'raw_observed':rawT is not None,
                      'visual_accepted':applied,'trans_residual_m':trans_res,
                      'axis_residual_deg':axis_res,'full_rotation_residual_deg':Rjump,
                      'twist_aligned_deg':twist_deg})
        verdicts[stat]=verdicts.get(stat,0)+1
        flask=poses['target_flask'][f]
        newflask={k:v for k,v in flask.items() if k!='_T'}
        if a.target_policy == 'static':
            newflask['valid']=True
            newflask['T_base_link_object']=target_f0.tolist()
            newflask['T_color_object']=(np.linalg.inv(T_base_color)@target_f0).tolist()
            newflask['status']='static_f0_prior_uncertain_after_contact'
            newflask['source']='fixed_target_scene_prior_not_gt'
            newflask['rgbtrack_original_valid']=bool(flask.get('valid'))
            newflask['rgbtrack_original_status']=str(flask.get('status'))
            newflask['pose_is_physical_ground_truth']=False
        new['target_flask'][f]=newflask
    out.mkdir(parents=True)
    out_manifest=dict(source_manifest)
    out_manifest['postprocess']={
        'type':'fk_gated_visual_trajectory_PROPOSAL_NOT_GT',
        'source_run':str(run),'grasp_anchor':a.grasp_anchor,'hand_link':a.hand_link,
        'max_translation_residual_m':a.max_translation_residual_m,
        'max_axis_residual_deg':a.max_axis_residual_deg,
        'observation_weight':a.observation_weight,'release_frame':a.release_frame,
        'target_policy':a.target_policy,
        'caveats':['Anchor frame must be stable grasp with reliable measured pose.',
                   'Rigid hand-object assumption invalid during slip, release or hand-object regrasp.',
                   'Symmetry-aware twist assumes the funnel mesh axis is local +Z.',
                   'Target static is a coarse prior; collision can move the physical flask.',
                   'Must not be treated as supervised 6D ground truth.']
    }
    (out/'manifest.json').write_text(json.dumps(out_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name in new: dump_rows(out/f'{name}_poses.jsonl',new[name])
    with (out/'pose_fusion_audit.csv').open('w',newline='',encoding='utf8') as file:
        w=csv.DictWriter(file,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    z=np.array([0,0,1.])
    tilt=[]
    for f in [0,a.grasp_anchor]:
        v=poses['funnel'][f]['_T'][:3,:3]@z
        tilt.append({'frame':f, 'unsigned_axis_tilt_from_base_up_deg':float(np.rad2deg(np.arccos(np.clip(abs(v[2]),0,1))))})
    rep={'episode':a.episode,'frames':len(frames),'grasp_anchor':a.grasp_anchor,
         'counts':verdicts,'raw_orientation_tilt':tilt,
         'user_review_required':True,
         'note':'Pose stream is a physically constrained CANDIDATE, not a reliable object tracker under slip/occlusion.'}
    (out/'pose_fusion_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(rep,ensure_ascii=False,indent=2))
    print('[DONE]',out)
    print('[NEXT] Run the existing visualize_rgbtrack_full_episode.py with --pose-run',out)

if __name__=='__main__':main()
