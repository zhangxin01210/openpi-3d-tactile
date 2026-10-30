#!/usr/bin/env python3
"""Read-only Hand-Scene V0 geometry QA, using *metric* CAD + frame-0 observed RGBTrack pose.

Creates derived QA output under the formal OpenPI checkout. Does NOT create training
sidecars, assert fingertip/tool contact truth, propagate funnel after pickup, or edit
poses/masks/data. Requires the existing OpenPI/SAM3D Python environment.

  PYTHONPATH=src python scripts/object_geometry/validate_hand_scene_geometry_e23.py \
    --repo-root "$PWD" --dataset "$PWD/data/1018_peg_all" \
    --pose-run "$PWD/output/pose_eval/hand_scene_e23_metric_f0_20261025_145902" \
    --output "$PWD/output/hand_scene_qa/e23_geometry_v0"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


OBJECTS = {
    'funnel': 'assets/object_geometry/synthware_19_22/funnel_F181975.glb',
    'target_flask': 'assets/object_geometry/synthware_19_22/flask_F301950.glb',
}


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def _read_first_pose(path: Path) -> dict:
    with path.open(encoding='utf-8') as file:
        line = next((line for line in file if line.strip()), None)
    if line is None:
        raise ValueError(f'Empty pose file: {path}')
    rec = json.loads(line)
    if rec.get('frame_index') != 0 or not rec.get('valid') or rec.get('status') != 'observed':
        raise ValueError(f'Expected observed VALID frame 0 pose, got {rec.get("status")}, frame={rec.get("frame_index")}, valid={rec.get("valid")}')
    return rec


def _se3(T, label: str) -> np.ndarray:
    T = np.asarray(T, np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1], atol=1e-6):
        raise ValueError(f'{label} must be finite SE(3) 4x4 matrix')
    R = T[:3,:3]
    if not np.allclose(R.T @ R, np.eye(3), atol=.02) or abs(np.linalg.det(R)-1)>.02:
        raise ValueError(f'{label} has invalid rotation')
    return T


def _sample_mesh(path: Path, n: int, seed: int) -> tuple[np.ndarray, dict]:
    import trimesh
    scene = trimesh.load(str(path), force='scene', process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene,trimesh.Scene) else scene
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] < 10:
        raise ValueError(f'Not a valid surface mesh: {path}')
    # Deterministic area-weighted triangle sampling; GLB vertices are ALREADY METRIC.
    tris = np.asarray(mesh.triangles, np.float64)
    areas = np.linalg.norm(np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0]), axis=1) * .5
    if np.sum(areas) <= 0 or not np.isfinite(areas).all():
        raise ValueError(f'Degenerate mesh triangles: {path}')
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(tris), n, replace=True, p=areas/np.sum(areas))
    u = rng.random(n); v = rng.random(n)
    a = np.sqrt(u)
    points = (1-a)[:,None]*tris[ids,0] + (a*(1-v))[:,None]*tris[ids,1] + (a*v)[:,None]*tris[ids,2]
    if not np.isfinite(points).all(): raise ValueError('Nonfinite sampled vertices')
    bounds=np.asarray(mesh.bounds, dtype=np.float64)
    return np.asarray(points,np.float32), {
        'mesh_sha256':_hash(path),
        'mesh_bounds_m':bounds.tolist(),
        'mesh_extents_m':(bounds[1]-bounds[0]).tolist(),
        'faces':int(len(mesh.faces)), 'vertices':int(len(mesh.vertices)),
        'mesh_sample_seed':int(seed), 'mesh_sample_count':int(n),
    }


def _transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (np.asarray(points, np.float64) @ T[:3,:3].T + T[:3,3]).astype(np.float32)


def _save_ply(path: Path, xyz_rgb: list[tuple[np.ndarray, tuple[int,int,int]]]):
    points = np.concatenate([np.asarray(p, np.float64) for p,_ in xyz_rgb], axis=0)
    colors = np.concatenate([np.repeat(np.asarray(rgb, np.uint8)[None,:], len(p), axis=0) for p,rgb in xyz_rgb], axis=0)
    if not len(points) or not np.isfinite(points).all():raise ValueError('Invalid PLY vertices')
    with path.open('w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(points, colors):
            f.write(f'{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {c[0]} {c[1]} {c[2]}\n')


def _draw_overlay(repo: Path, dataset: Path, pose_run: Path, calib: Path, points: dict, output: Path, episode: int):
    import cv2
    sys.path.insert(0, str(repo / 'scripts/object_geometry/pose_tracking'))
    from rgbtrack_adapter import load_front_calibration
    from openpi.spatial_dataset.source import RawSpatialDataset
    camera=load_front_calibration(repo,calib)
    ds=RawSpatialDataset(dataset,episode=episode)
    imgs=ds.load_video_frames('front',selected={0})
    if 0 not in imgs:raise FileNotFoundError('No RGB frame 0')
    img=np.asarray(imgs[0],np.uint8).copy()
    T_cam_base=np.linalg.inv(_se3(camera.T_base_color,'T_base_color'))
    K=np.asarray(camera.K_color,np.float64)
    for name,color in [('funnel',(30,230,30)),('target_flask',(30,180,240))]:
        pts=_transform(points[name],T_cam_base)
        good=pts[:,2]>0.05
        uv=(pts[good,:2]/pts[good,2:3]) @ K[:2,:2].T + K[:2,2]
        for u,v in uv:
            if 0<=round(u)<img.shape[1] and 0<=round(v)<img.shape[0]:
                cv2.circle(img,(int(round(u)),int(round(v))),2,color,-1,cv2.LINE_AA)
    cv2.imwrite(str(output / 'mesh_surface_projection_f000000.png'),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))


def _distance_report(repo: Path,dataset:Path,episode:int,points:dict,output:Path,viz_frames:set[int]):
    from openpi.spatial_dataset.source import RawSpatialDataset
    from openpi.spatial.preprocess import SpatialPreprocessor
    from openpi.object_geometry.calibration import make_front_geometry_config
    cfg,_=make_front_geometry_config(repo,'configs/ur7e_xhand/front_calibration_candidate.json')
    prep=SpatialPreprocessor.from_repo_root(repo_root=repo,config=cfg)
    ds=RawSpatialDataset(dataset,episode=episode)
    try:
        from scipy.spatial import cKDTree
        trees={name:cKDTree(arr) for name,arr in points.items()}
        def near(name,xyz):return trees[name].query(xyz,k=1,workers=1)[0]
    except ImportError:
        def near(name,xyz):return np.linalg.norm(xyz[:,None,:]-points[name][None,:,:],axis=-1).min(axis=1)
    records=[]
    for row in ds.iter_rows(depth_roles=()):
        frame=int(row['frame_index'])
        state=np.asarray(row['observation.state']).reshape(-1)
        tactile=prep.preprocess_tactile(state)
        h_xyz=np.asarray(tactile.xyz_m,np.float32)
        ids=np.asarray(tactile.finger_id,np.int32)
        forces=np.asarray(tactile.force_norm,np.float32)
        if h_xyz.shape!=(600,3) or not np.isfinite(h_xyz).all():
            raise ValueError(f'f{frame}: invalid tactile geometry {h_xyz.shape}')
        record={'frame_index':frame}
        for obj in ('funnel','target_flask'):
            d=near(obj,h_xyz)
            for finger in range(5):
                sel=(ids==finger)
                if not np.any(sel):raise ValueError(f'f{frame}: no taxels finger {finger}')
                record[f'{obj}_finger{finger}_min_m']=float(d[sel].min())
                record[f'{obj}_finger{finger}_p10_m']=float(np.percentile(d[sel],10))
        for finger in range(5):
            sel=ids==finger
            record[f'finger{finger}_force_max_native']=float(forces[sel].max())
        records.append(record)
        if frame in viz_frames:
            _save_ply(output/f'hand_scene_f{frame:06d}.ply',[(points['funnel'],(30,205,60)),(points['target_flask'],(30,100,240)),(h_xyz,(240,80,60))])
        if len(records)%150==0:print(f'[QA] computed hand/funnel/goal distances for {len(records)} frames',flush=True)
    if not records:raise ValueError('Empty episode')
    import csv
    with (output/'distance_by_finger.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]))
        wr.writeheader();wr.writerows(records)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t=np.array([r['frame_index'] for r in records])
        for obj in ('funnel','target_flask'):
            fig,ax=plt.subplots(figsize=(11,4))
            for fid in range(5):
                d=np.array([r[f'{obj}_finger{fid}_min_m'] for r in records])*100
                ax.plot(t,d,label=f'finger {fid}',lw=1)
            ax.set(xlabel='Episode frame',ylabel='Nearest-taxel to static mesh (cm)',title=f'{obj}: STATIC f0 anchor (NOT dynamic object tracking)')
            ax.legend(ncol=5,fontsize=8)
            fig.tight_layout()
            fig.savefig(output/f'{obj}_distance_curves.png',dpi=160)
            plt.close(fig)
    except ImportError:
        print('[WARN] matplotlib missing, distance csv and PLY still available',flush=True)
    return len(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--pose-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,default=23)
    p.add_argument('--viz-frames',default='0,100,200,300,400')
    p.add_argument('--calibration-profile',type=Path,default=Path('configs/ur7e_xhand/front_calibration_candidate.json'))
    args=p.parse_args()
    repo=args.repo_root.resolve();ds=args.dataset.resolve();pose=args.pose_run.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Output already exists (refusing overwrite): {out}')
    if not (pose/'manifest.json').exists():raise FileNotFoundError(pose/'manifest.json')
    manifest=json.loads((pose/'manifest.json').read_text())
    if manifest.get('episode')!=args.episode:raise ValueError('Pose-run episode mismatch')
    if set(manifest.get('objects',[]))!={'funnel','target_flask'}:raise ValueError('Expected both objects registered')
    profile=args.calibration_profile
    if not profile.is_absolute():profile=repo/profile
    if not profile.is_file():raise FileNotFoundError(profile)
    from openpi.object_geometry.calibration import front_calibration_provenance
    expected=front_calibration_provenance(repo,profile)
    observed=manifest.get('calibration')
    from openpi.object_geometry.calibration import calibration_provenance_error
    err=calibration_provenance_error(observed,expected)
    if err:raise ValueError(f'Pose calibration mismatch, cannot merge with spatial sidecar: {err}')
    points={};meta={}
    for i,(name,rel) in enumerate(OBJECTS.items()):
        mesh_path=repo/rel
        pose_record=_read_first_pose(pose/f'{name}_poses.jsonl')
        conv=manifest.get('mesh_conversions',{}).get(name,{})
        if Path(conv.get('source_glb','')).resolve()!=mesh_path.resolve():
            raise ValueError(f'Pose run used different mesh for {name}: {conv.get("source_glb")}')
        if not np.isclose(float(conv.get('scale_m_per_unit', np.nan)),1.0,atol=1e-8):
            raise ValueError(f'{name}: pose registration did not use the current METRIC GLB at scale 1.0: {conv.get("scale_m_per_unit")}')
        if conv.get('source_sha256')!=_hash(mesh_path):
            raise ValueError(f'{name}: pose-run mesh SHA256 differs from current source GLB')
        local,stats=_sample_mesh(mesh_path,512,seed=20261025+i)
        T=_se3(pose_record['T_base_link_object'],f'f0 {name} pose')
        base=_transform(local,T)
        points[name]=base
        meta[name]={**stats,'mesh':str(mesh_path),'pose_confidence_NOT_calibrated_prob':pose_record.get('confidence'),'T_base_link_object':T.tolist()}
        print('[MESH]',name,'extents m=',np.round(stats['mesh_extents_m'],4),'base centroid=',np.round(base.mean(axis=0),4))
    out.mkdir(parents=True)
    np.savez_compressed(out/'static_mesh_points_base_e23.npz',
                        funnel_xyz_base_m=points['funnel'],target_flask_xyz_base_m=points['target_flask'])
    with (out/'geometry_manifest.json').open('w') as f:
        json.dump({'schema_version':1,'status':'qa_only_NOT_training_sidecar','episode':args.episode,
                   'pose_run':str(pose),'calibration':expected,'objects':meta,
                   'warning':'Funnel geometry is static at f0; its distances after pickup are NOT hand-to-moving-funnel distances. Flask geometry can change slightly after impacts. RGBTrack confidence is not a metric accuracy guarantee.'},f,indent=2)
    _draw_overlay(repo,ds,pose,profile,points,out,args.episode)
    n=_distance_report(repo,ds,args.episode,points,out,{int(x) for x in args.viz_frames.split(',') if x.strip()})
    print('[DONE]',n,'frames in',out)
    print('[OUTPUT] mesh_surface_projection_f000000.png')
    print('[OUTPUT] funnel_distance_curves.png, target_flask_distance_curves.png')
    print('[OUTPUT] static_mesh_points_base_e23.npz and hand_scene_f*.ply')
    print('[CAUTION] Diagnostic only. Static funnel distances after grasp are not physically valid hand-funnel relations.')

if __name__=='__main__':main()
