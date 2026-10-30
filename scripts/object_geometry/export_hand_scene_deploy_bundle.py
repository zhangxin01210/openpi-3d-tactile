#!/usr/bin/env python3
"""Computer-4 export: minimal real replay+teacher references+manual seed mask assets.

Run with PYTHONPATH=src in the pinned openpi-3d-tactile repository.
Does not modify raw data, pose runs or checkpoints. Copies only chosen episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] if 'scripts' in Path(__file__).parts else Path.cwd().resolve()
SEED_EPISODES = (0, 23, 38, 49, 58, 65)
ALIASES = {'funnel': ('funnel', '漏斗'), 'target_flask': ('target_flask', 'flask', 'target-flask', 'bottle')}


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            dig.update(chunk)
    return dig.hexdigest()


def find_mask(root: Path, obj: str, frame: int) -> Path | None:
    for alias in ALIASES[obj]:
        for d in (root / alias, root / 'masks' / alias):
            for stem in (f'{frame:05d}', f'{frame:06d}', f'frame_{frame:06d}', str(frame)):
                for ext in ('.png', '.PNG'):
                    p = d / (stem + ext)
                    if p.is_file():
                        return p
    return None


def find_mask_root(hoi_outputs: Path, episode: int) -> Path | None:
    preferred = (f'level1_sequence_e{episode}_maskfix_v1', f'level1_sequence_e{episode}')
    candidates = [hoi_outputs / p / 'masks' for p in preferred]
    # If the conventional folders do not contain both seed masks, examine variants.
    candidates += sorted(hoi_outputs.glob(f'level1_sequence_e{episode}_*/masks'))
    hits = []
    for p in candidates:
        if p.is_dir() and all(find_mask(p, obj, 0) for obj in ALIASES):
            if p not in hits:
                hits.append(p)
    return hits[0] if hits else None


def stage_masks(repo_root: Path, bundle_root: Path, hoi_outputs: Path, episode: int,
                *, install: bool) -> dict:
    target_rel = Path('assets/object_geometry/manual_seed_masks') / f'e{episode:03d}'
    src = find_mask_root(hoi_outputs, episode)
    info = {'episode': episode, 'source_root': str(src) if src else None, 'status': 'MISSING', 'objects': {}}
    if src is None:
        return info
    import cv2
    filenames = {}
    sizes = []
    for obj in ALIASES:
        source = find_mask(src, obj, 0)
        assert source is not None
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.ndim != 2 or np.count_nonzero(mask) == 0:
            raise RuntimeError(f'Mask invalid/empty: {source}')
        names = {'funnel': 'funnel.png', 'target_flask': 'target_flask.png'}
        dest = bundle_root / target_rel / names[obj]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        info['objects'][obj] = {'file': str(target_rel / names[obj]), 'source_path': str(source),
                                'sha256': sha256(dest), 'size_hw': list(mask.shape),
                                'foreground_pixels': int(np.count_nonzero(mask))}
        filenames[obj] = dest
        sizes.append(tuple(mask.shape))
    if len(set(sizes)) != 1:
        raise RuntimeError(f'Episode {episode} funnel and flask seed shapes disagree: {sizes}')
    info['status'] = 'OK'
    info['seed_frame'] = 0
    info['camera_role'] = 'front'
    manifest = bundle_root / target_rel / 'manifest.json'
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    if install:
        for f in (filenames.values()):
            rel = f.relative_to(bundle_root)
            installed = repo_root / rel
            if installed.exists() and sha256(installed) != sha256(f):
                raise FileExistsError(f'Refusing to overwrite existing seed asset: {installed}')
            installed.parent.mkdir(parents=True, exist_ok=True)
            if not installed.exists():
                shutil.copy2(f, installed)
        installed_manifest = repo_root / target_rel / 'manifest.json'
        installed_manifest.write_text(manifest.read_text(encoding='utf-8'), encoding='utf-8')
    return info


def align_depth_to_color(depth_m, camera, color_shape):
    """Same nearest Z-buffer projection as rgb_depth_joint_diagnostic.py.

    Local copy avoids importing visualization-only root_to_tip_mesh dependencies.
    Calibration comes from the repository's canonical calibration loader.
    """
    h_color, w_color = color_shape[:2]
    h_depth, w_depth = depth_m.shape
    kd, kc = camera.depth_intrinsics, camera.color_intrinsics
    ix = np.flatnonzero(np.isfinite(depth_m) & (depth_m > 0))
    yy, xx = np.divmod(ix, w_depth)
    z = depth_m.ravel()[ix].astype(np.float64)
    pts = np.column_stack(((xx-kd.cx)*z/kd.fx, (yy-kd.cy)*z/kd.fy, z))
    trans = camera.T_color_depth
    pts = pts @ trans[:3,:3].T + trans[:3,3]
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:,2] > 0)]
    xp = np.rint(kc.fx*pts[:,0]/pts[:,2]+kc.cx).astype(np.int64)
    yp = np.rint(kc.fy*pts[:,1]/pts[:,2]+kc.cy).astype(np.int64)
    keep = (xp>=0)&(xp<w_color)&(yp>=0)&(yp<h_color)
    out = np.full(h_color*w_color, np.inf, np.float32)
    np.minimum.at(out, yp[keep]*w_color+xp[keep], pts[keep,2].astype(np.float32))
    out = out.reshape(h_color,w_color)
    out[~np.isfinite(out)] = 0.0
    return out


def flatten_teacher(sample) -> dict:
    src = sample.as_dict()
    out = {}
    for k, val in src.items():
        if k == 'hand_scene':
            for hkey, hval in val.items():
                if hkey == 'causal_tactile':
                    for tkey, tval in hval.items():
                        out[f'object_geometry__hand_scene__causal_tactile__{tkey}'] = np.asarray(tval)
                else:
                    out[f'object_geometry__hand_scene__{hkey}'] = np.asarray(hval)
        else:
            out[f'object_geometry__{k}'] = np.asarray(val)
    return out


def export_episode(repo_root: Path, bundle_root: Path, dataset_root: Path, episode: int,
                   *, mask_info: dict, calib):
    from openpi.spatial_dataset.source import RawSpatialDataset
    from openpi.spatial_dataset.derived import SpatialDerivedDataset
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    import itertools
    raw = RawSpatialDataset(dataset_root, episode=episode)
    available = set(raw.camera_roles)
    if not {'front','left','right'}.issubset(available):
        raise RuntimeError(f'episode {episode} is missing Pi0 image slots, camera roles={available}')
    if mask_info['status'] != 'OK':
        raise RuntimeError(f'Episode {episode}: missing BOTH first-frame masks, cannot seed valid replay')
    spatial = SpatialDerivedDataset(dataset_root,version='v1_front_candidate')
    geo = ObjectGeometryDerivedDataset(dataset_root,version='v1_hand_scene_q2')
    video = {role: raw.iter_video_frames(role) for role in ('front','left','right')}
    rows = raw.iter_rows(depth_roles=('front',))
    count = 0
    replay = bundle_root / 'data/deploy/replay' / f'e{episode:03d}'
    reference = bundle_root / 'data/deploy/reference' / f'e{episode:03d}'
    replay.mkdir(parents=True,exist_ok=True)
    reference.mkdir(parents=True,exist_ok=True)
    import cv2
    seeds = {}
    for obj in ALIASES:
        p = bundle_root / mask_info['objects'][obj]['file']
        seeds[obj] = (cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)>0)
    names = ('front','left','right')
    for row in rows:
        f = int(row['frame_index'])
        imgs = {}
        for role in names:
            try: v = next(video[role])
            except StopIteration: raise RuntimeError(f'Episode {episode}: video {role} stops before parquet row frame {f}')
            if v.frame_index != f:
                raise RuntimeError(f'Episode {episode}: misaligned video {role} frame {v.frame_index} vs parquet {f}')
            imgs[role] = v.rgb
        depth = np.asarray(row['observation.depths.cam_front'])
        if depth.dtype != np.uint16:
            raise TypeError(f'front depth expected uint16, got {depth.dtype}')
        aligned_m = align_depth_to_color(depth.astype(np.float32)*calib.depth_scale_m_per_unit, calib, imgs['front'].shape)
        # Contract is raw depth-units; values now live on front COLOR pixels, not original depth pixels.
        aligned_raw = np.clip(np.rint(aligned_m/calib.depth_scale_m_per_unit),0,65535).astype(np.uint16)
        state = np.asarray(row['observation.state'],dtype=np.float32)
        if state.shape != (1972,): raise ValueError(f'Episode {episode} frame {f}: state shape={state.shape}')
        sample = spatial.get(episode_index=episode,frame_index=f)
        teacher = geo.get(episode_index=episode,frame_index=f)
        payload = {'front_rgb':imgs['front'], 'left_rgb':imgs['left'], 'right_rgb':imgs['right'],
                   'front_depth_raw':aligned_raw, 'robot_state':state,
                   'tactile_force_norm':np.asarray(sample.tactile_force_norm,dtype=np.float32),
                   'frame_index':np.int64(f), 'timestamp_s':np.float64(row['timestamp'])}
        if f == 0:
            for obj in ALIASES:
                if seeds[obj].shape != imgs['front'].shape[:2]:
                    raise RuntimeError(f'Episode {episode} {obj} seed shape {seeds[obj].shape} != front RGB {imgs["front"].shape[:2]}')
                payload[f'{obj}_seed_mask'] = seeds[obj]
        name = f'frame_{f:06d}.npz'
        np.savez_compressed(replay/name,**payload)
        ref = flatten_teacher(teacher)
        ref['episode_index']=np.asarray(episode,dtype=np.int64)
        ref['frame_index']=np.asarray(f,dtype=np.int64)
        ref['spatial__tactile_force_norm']=np.asarray(sample.tactile_force_norm,dtype=np.float32)
        np.savez_compressed(reference/name,**ref)
        count += 1
        if count % 100==0: print(f'e{episode:03d}: {count} frames exported',flush=True)
    if count == 0: raise RuntimeError(f'Episode {episode}: no rows')
    return {'episode':episode,'frames':count,'replay_dir':str(replay.relative_to(bundle_root)),
            'reference_dir':str(reference.relative_to(bundle_root)),
            'depth': 'front RGB-color aligned, nearest/Z-buffer, uint16 raw scale units',
            'depth_scale_m_per_unit': float(calib.depth_scale_m_per_unit)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--dataset-root',type=Path,default=Path('data/1018_peg_all'))
    p.add_argument('--hoi-outputs',type=Path,default=Path('/workspace/mnt/sqzhang26/hoi/outputs'))
    p.add_argument('--bundle-root',type=Path,default=Path('scratch/hand_scene_deploy_bundle'))
    p.add_argument('--front-calibration-profile',type=Path,default=Path('configs/ur7e_xhand/front_calibration_candidate.json'))
    p.add_argument('--episodes',type=int,nargs='*',default=[23],help='replay episodes, usually 23 or 23 38')
    p.add_argument('--seed-episodes',type=int,nargs='*',default=list(SEED_EPISODES))
    p.add_argument('--install-masks-to-repo',action='store_true')
    p.add_argument('--masks-only',action='store_true')
    p.add_argument('--no-tar',action='store_true')
    args=p.parse_args()
    repo_root=args.repo_root.resolve()
    dataset_root=args.dataset_root if args.dataset_root.is_absolute() else (repo_root/args.dataset_root)
    bundle_root=args.bundle_root if args.bundle_root.is_absolute() else (repo_root/args.bundle_root)
    if not (repo_root/'src/openpi').is_dir(): raise RuntimeError(f'Not openpi project root: {repo_root}')
    if not args.masks_only and not dataset_root.is_dir():raise FileNotFoundError(dataset_root)
    bundle_root.mkdir(parents=True,exist_ok=True)
    masks={}
    for ep in sorted(set(args.seed_episodes)|set(args.episodes)):
        info=stage_masks(repo_root,bundle_root,args.hoi_outputs,ep,install=args.install_masks_to_repo)
        masks[str(ep)]=info
        print(f'mask e{ep}: {info["status"]}; {info["source_root"]}',flush=True)
    episodes=[]
    if not args.masks_only:
        from openpi.object_geometry.calibration import make_front_geometry_config
        from openpi.spatial.calibration import load_camera_calibrations
        from openpi.spatial.geometry import apply_diagnostic_overrides
        profile = args.front_calibration_profile
        if not profile.is_absolute(): profile = repo_root / profile
        if not profile.is_file(): raise FileNotFoundError(f'Missing reviewed front calibration profile: {profile}')
        cfg,_=make_front_geometry_config(repo_root,profile)
        calibs=load_camera_calibrations(repo_root=repo_root,calibration_config=cfg.calibration,camera_roles=('front',))
        front=apply_diagnostic_overrides(calibs.cameras,cfg.diagnostics)['front']
        for ep in args.episodes:
            episodes.append(export_episode(repo_root,bundle_root,dataset_root,ep,mask_info=masks[str(ep)],calib=front))
    summary={
        'schema_version':1,'purpose':'selected-episode replay, reference teacher arrays, and reviewed manual initial masks',
        'source_repo':str(repo_root),'dataset_source':str(dataset_root),
        'front_calibration_profile':str(args.front_calibration_profile),
        'masks':masks,'episodes':episodes,
        'warnings':[
            'Masks are manually reviewed seed images; verify alignment against original FRONT RGB before use.',
            'Offline teacher reference is an imperfect RGBTrack estimate, not metric ground truth.',
            'Replay aligned front depth uses the explicit reviewed front calibration profile and nearest Z-buffer; verify it matches the q2 training run.',
            'Do not use stored mask as automatic seed for a newly moved camera/scene.',
            'Do not shuffle or skip video frames when evaluating online causal tracking.',
        ],
    }
    (bundle_root/'bundle_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not args.no_tar:
        archive=bundle_root.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for path in sorted(bundle_root.rglob('*')):
                if path.is_file():tar.add(path,arcname=str(path.relative_to(bundle_root)))
        print('ARCHIVE:',archive,'SIZE_MB:',round(archive.stat().st_size/1e6,1),flush=True)
    print('MASKS:',{k:v['status'] for k,v in masks.items()})
    print('EPISODES:',episodes)
    print('DONE')

if __name__=='__main__':main()
