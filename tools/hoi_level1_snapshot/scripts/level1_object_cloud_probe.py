#!/usr/bin/env python3
"""Level-1 object point cloud audit, Episode 0.

Commands
--------
track-funnel: propagate the EXISTING frame-212 SAM2 binary mask backward to
frame 200 and forward to frame 240; never prompts or regenerates a mesh.

qa: read the ORIGINAL front RGB video and ORIGINAL uint16 depth from the
LeRobot data, project depth measurements into the RGB camera via the project's
actual camera calibration, select projected points inside object masks, and
save candidate base_link clouds + verification pictures.  Never edits the
training sidecar, calibrations or original masks.

IMPORTANT: "candidate" points are depth measurements projected inside an RGB
segmentation. For glass, many of them can be the BACKGROUND BEHIND THE OBJECT;
this code intentionally NEVER labels such measurements as proven object surface.

Project version: user-maintained /workspace/mnt/sqzhang26/openpi-3d-tactile.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HOI = Path('/workspace/mnt/sqzhang26/hoi')
REPO = Path('/workspace/mnt/sqzhang26/openpi-3d-tactile')
DATASET = REPO / 'data/1018_peg_all'
DEFAULT_OUT = HOI / 'outputs/level1_object_cloud_audit_e0'


def _load_image(path: Path, *, grayscale: bool = False):
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return arr


def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')


def _manifest_map(clip: Path) -> dict[int, int]:
    raw = json.loads((clip / 'manifest.json').read_text(encoding='utf8'))
    out = {int(x['episode_frame_index']): int(x['clip_index']) for x in raw['mapping']}
    return out


def track_funnel(args):
    """Reuse an accepted frame-212 mask; only infer the two missing masks."""
    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    clip = Path(args.clip)
    original = Path(args.anchor_mask)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.mask_output)
    anchor = int(args.anchor)
    targets = tuple(sorted({int(x) for x in args.targets.split(',')} | {anchor}))
    mapping = _manifest_map(clip)
    missing = set(targets) - set(mapping)
    if missing:
        raise ValueError(f'Clip {clip} does not cover episode frames: {sorted(missing)}')
    frame_anchor = mapping[anchor]
    mask = _load_image(original, grayscale=True) > 127
    im = _load_image(clip / 'frames' / f'{frame_anchor:05d}.jpg')
    if mask.shape != im.shape[:2]:
        raise ValueError(f'Original mask {mask.shape} != SAM2 video frame {im.shape[:2]}')
    if not np.any(mask):
        raise ValueError('Anchor mask is empty')

    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(
        'configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint),
        device='cuda', apply_postprocessing=True,
    )
    reverse_targets = {mapping[t] for t in targets if t < anchor}
    forward_targets = {mapping[t] for t in targets if t > anchor}
    index_to_ep = {v: k for k, v in mapping.items()}
    count = {}

    def save(frame_idx, masks, obj_ids):
        if frame_idx not in index_to_ep:
            return
        eid = index_to_ep[frame_idx]
        if eid not in targets:
            return
        ids = [int(x) for x in obj_ids]
        if 1 not in ids:
            raise RuntimeError(f'Frame {eid}: tracking lost object id 1')
        j = ids.index(1)
        binary = (masks[j] > 0).detach().cpu().numpy().squeeze().astype(np.uint8)
        if binary.shape != mask.shape:
            raise RuntimeError(f'Frame {eid}: bad mask shape: {binary.shape}')
        path = out_dir / f'episode0_frame{eid:04d}.png'
        cv2.imwrite(str(path), binary * 255)
        count[str(eid)] = int(binary.sum())
        print(f'[SAM2] episode0 frame={eid}: mask pixels={count[str(eid)]}; {path}', flush=True)

    # Use the stored accepted mask, not a newly invented click/box.
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(str(clip / 'frames'), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        predictor.add_new_mask(inference_state=state, frame_idx=frame_anchor,
                               obj_id=1, mask=mask.astype(np.uint8))
        save(frame_anchor, torch.from_numpy(mask)[None], [1])
        if reverse_targets:
            reverse_length = frame_anchor - min(reverse_targets) + 1
            for idx, ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=frame_anchor,
                    max_frame_num_to_track=reverse_length, reverse=True):
                if idx in reverse_targets:
                    save(idx, logits, ids)
        if forward_targets:
            forward_length = max(forward_targets) - frame_anchor + 1
            for idx, ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=frame_anchor,
                    max_frame_num_to_track=forward_length, reverse=False):
                if idx in forward_targets:
                    save(idx, logits, ids)

    expected = {str(t) for t in targets}
    if set(count) != expected:
        raise RuntimeError(f'Missing SAM2 tracking output: {expected - set(count)}')
    _save_json(out_dir / 'masks_manifest.json', {
        'episode': 0, 'anchor_frame': anchor, 'anchor_mask': str(original),
        'target_frames': targets, 'mask_pixels': count,
        'warning': 'Mask propagation is unverified until overlays are inspected.'
    })
    print('[SAM2] Finished. Inspect masks/overlays during QA before interpreting depth.', flush=True)


def _fullmask_to_rgb(mask_path: Path, rgb_hw: tuple[int,int], *, crop_file: Path | None):
    """Only permit exact match or explicit crop metadata, never naive resize."""
    import cv2
    raw = _load_image(mask_path, grayscale=True)
    h, w = rgb_hw
    if raw.shape == (h, w):
        return (raw > 127), {'transformation':'identity', 'source_shape':list(raw.shape)}
    if crop_file is None or not crop_file.is_file():
        raise ValueError(f'{mask_path}: {raw.shape} != RGB {(h,w)} and no crop manifest was given')
    meta = json.loads(crop_file.read_text(encoding='utf8'))
    if 'crop_xyxy' not in meta:
        raise ValueError(f'{crop_file} missing crop_xyxy; refusing to guess mask mapping')
    x1,y1,x2,y2 = map(int,meta['crop_xyxy'])
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):
        raise ValueError(f'crop_xyxy outside image: {(x1,y1,x2,y2)}')
    # SAM2 Mask on enlarged crop --> ORIGINAL full RGB image coordinates.
    sub = cv2.resize((raw>127).astype(np.uint8), (x2-x1,y2-y1),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    full = np.zeros((h,w), dtype=bool)
    full[y1:y2,x1:x2] = sub
    return full, {'transformation':'resize-crop-back', 'source_shape':list(raw.shape),
                  'crop_xyxy':[x1,y1,x2,y2], 'crop_manifest':str(crop_file)}


def _make_calibration(repo: Path, profile: Path | None):
    sys.path.insert(0,str(repo/'src'))
    from openpi.object_geometry.calibration import make_front_geometry_config
    from openpi.spatial.calibration import load_camera_calibrations
    from openpi.spatial.geometry import apply_diagnostic_overrides
    cfg, _ = make_front_geometry_config(repo, profile)
    bundle = load_camera_calibrations(repo_root=repo,calibration_config=cfg.calibration,
                                       camera_roles=cfg.visual.camera_roles)
    # build_visual_geometry() applies diagnostic overrides, so follow it here.
    cam = apply_diagnostic_overrides(bundle.cameras,cfg.diagnostics)['front']
    return cfg,cam


def _flatten_depth_to_points(depth_raw: np.ndarray, cam):
    """Project ALL valid original depth pixels -> color optical -> base_link.

    Return arrays aligned by input depth pixel; avoids future ambiguity from
    premade 4096-point scene subsampling. No depth-to-color pixel identity assumption.
    """
    from openpi.spatial.geometry import build_camera_cache, transform_points
    H,W=depth_raw.shape
    cache = build_camera_cache(cam,image_height=H,image_width=W)
    z = depth_raw.reshape(-1).astype(np.float32)*np.float32(cam.depth_scale_m_per_unit)
    good = np.isfinite(z)&(z>0)
    xyz_base = cache.rays_base_per_meter[good]*z[good,None]+cache.base_origin_m[None,:]
    source_pixel = np.flatnonzero(good).astype(np.int32)
    xyz_color=transform_points(xyz_base, np.linalg.inv(cam.T_base_color))
    return xyz_base, xyz_color, source_pixel


def _project_mask_filter(xyz_base,xyz_color, depth_z_m, mask,cam, cfg):
    from openpi.spatial.geometry import project_pinhole
    H,W=mask.shape
    uv=project_pinhole(xyz_color,cam.color_intrinsics)
    uv_safe=np.nan_to_num(uv,nan=-1e9,posinf=-1e9,neginf=-1e9)
    pix=np.rint(uv_safe).astype(np.int64)
    valid=(np.isfinite(uv).all(axis=1)&(xyz_color[:,2]>0)&
           (pix[:,0]>=0)&(pix[:,0]<W)&(pix[:,1]>=0)&(pix[:,1]<H))
    inside=np.zeros(len(xyz_base),dtype=bool)
    inside[valid]=mask[pix[valid,1],pix[valid,0]]
    roi=cfg.roi
    in_range=(depth_z_m>=cfg.visual.min_depth_m)&(depth_z_m<=cfg.visual.max_depth_m)
    workspace=((xyz_base[:,0]>=roi.x_min_m)&(xyz_base[:,0]<=roi.x_max_m)&
               (xyz_base[:,1]>=roi.y_min_m)&(xyz_base[:,1]<=roi.y_max_m)&
               (xyz_base[:,2]>=roi.z_min_m)&(xyz_base[:,2]<=roi.z_max_m))
    candidate = valid & inside & in_range & workspace
    return pix, valid, inside, in_range, workspace,candidate


def _write_ply_ascii(path:Path, xyz:np.ndarray, rgb:np.ndarray):
    assert xyz.shape==(len(rgb),3)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        for a in ('x','y','z'):
            f.write(f'property float {a}\n')
        for a in ('red','green','blue'):
            f.write(f'property uchar {a}\n')
        f.write('end_header\n')
        for xyz_row,col in zip(xyz,rgb):
            f.write(f'{xyz_row[0]:.6f} {xyz_row[1]:.6f} {xyz_row[2]:.6f} '
                    f'{int(col[0])} {int(col[1])} {int(col[2])}\n')


def _plots(frame:int, kind:str, rgb:np.ndarray, mask:np.ndarray,
           uv:np.ndarray, inside:np.ndarray, candidates:np.ndarray,xyz_base:np.ndarray,
           tactile_xyz:np.ndarray|None,output:Path):
    import cv2
    # Images in RGB for display; write with RGB -> BGR conversion.
    overlay=rgb.copy()
    overlay[mask]=(0.45*overlay[mask]+0.55*np.asarray([255,40,40])).astype(np.uint8)
    rgba=cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)
    # Outline mask in WHITE. Points: yellow=projection inside mask+valid depth+ROI;
    # cyan=raw depth whose projection hit mask but fell outside analysis ROI.
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgba,contours,-1,(255,255,255),1)
    sel=np.flatnonzero(candidates)
    # all projected candidate points, display at most 12k, deterministic indices
    if len(sel)>12000:sel=sel[np.linspace(0,len(sel)-1,12000).astype(int)]
    for i in sel:
        u,v=uv[i]
        rgba[v,u]=(0,255,255) # yellow in BGR
    cv2.putText(rgba, f'e0 f{frame} {kind}: candidate={int(candidates.sum())} / mask={int(mask.sum())}px',
                (6,26),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(rgba, f'e0 f{frame} {kind}: candidate={int(candidates.sum())} / mask={int(mask.sum())}px',
                (6,26),cv2.FONT_HERSHEY_SIMPLEX,0.52,(250,250,250),1,cv2.LINE_AA)
    cv2.imwrite(str(output/'overlay.png'),rgba)

    # Three metric orthographic views, generated with OpenCV only (no Open3D,
    # matplotlib, or custom renderer needed on the OpenPI environment).
    rng = np.random.default_rng(0)
    scene_idx = np.flatnonzero(~candidates)
    if len(scene_idx) > 4500:
        scene_idx = rng.choice(scene_idx, 4500, replace=False)
    object_idx = np.flatnonzero(candidates)
    if len(object_idx) > 6500:
        object_idx = rng.choice(object_idx, 6500, replace=False)
    ref = xyz_base[object_idx]
    if tactile_xyz is not None:
        ref = np.vstack([ref, tactile_xyz]) if len(ref) else tactile_xyz
    if len(ref) == 0:
        ref = xyz_base[scene_idx] if len(scene_idx) else np.zeros((1,3))
    panels=[]
    views=[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]
    for a,b,title in views:
        canvas=np.full((460,480,3),245,dtype=np.uint8)
        center=np.median(ref[:,[a,b]],axis=0)
        distances=np.abs(ref[:,[a,b]]-center)
        halfspan=float(max(0.085, np.quantile(distances,0.95)*1.15))
        halfspan=min(halfspan,0.85)  # avoid outlier explosions
        px_per_m=380./(2*halfspan)
        def to_px(points):
            p=np.asarray(points)
            if not len(p):return np.zeros((0,2),dtype=np.int32)
            u=np.rint(240+(p[:,a]-center[0])*px_per_m)
            v=np.rint(240-(p[:,b]-center[1])*px_per_m)
            return np.stack([u,v],axis=1).astype(np.int32)
        cv2.rectangle(canvas,(40,40),(440,440),(150,150,150),1)
        for centerline in [0]:
            cv2.line(canvas,(240,40),(240,440),(220,220,220),1)
            cv2.line(canvas,(40,240),(440,240),(220,220,220),1)
        for u,v in to_px(xyz_base[scene_idx]):
            if 40<=u<440 and 40<=v<440:canvas[v,u]=(190,190,190)
        for u,v in to_px(xyz_base[object_idx]):
            if 40<=u<440 and 40<=v<440:cv2.circle(canvas,(int(u),int(v)),1,(40,125,245),-1)
        if tactile_xyz is not None:
            for u,v in to_px(tactile_xyz):
                if 40<=u<440 and 40<=v<440:cv2.circle(canvas,(int(u),int(v)),2,(40,155,40),-1)
        cv2.putText(canvas,title+' (base_link)',(28,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(canvas,f'{"XYZ"[a]} horizontal / {"XYZ"[b]} vertical | +/-{halfspan:.2f} m',
                    (32,453),cv2.FONT_HERSHEY_SIMPLEX,.42,(45,45,45),1,cv2.LINE_AA)
        panels.append(canvas)
    overview=np.concatenate(panels,axis=1)
    cv2.putText(overview,'gray:other depth; orange:mask-selected depth; green:XHand FK',
                (15,60),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(output/'base_xyz_views.png'),overview)


def _quantiles(x:np.ndarray):
    if len(x)==0:return None
    x=np.asarray(x,dtype=np.float64)
    return {str(p):float(np.percentile(x,p)) for p in (0,10,25,50,75,90,100)}


def qa(args):
    import cv2
    repo=Path(args.repo).resolve()
    hoi=Path(args.hoi).resolve()
    dataset_root=Path(args.dataset).resolve()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty, refusing to overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'src'))
    from openpi.spatial_dataset.source import RawSpatialDataset
    cfg,cam=_make_calibration(repo,Path(args.front_calibration_profile).resolve() if args.front_calibration_profile else None)
    ds=RawSpatialDataset(dataset_root,episode=0)
    ids=[0,200,212,240]
    image_map=ds.load_video_frames('front',selected=ids)
    rows={int(row['frame_index']):row for row in ds.iter_rows(selected=ids,depth_roles=('front',))}

    # Optional XHand FK. A failure here must not kill our depth+mask QA; make
    # it explicitly visible in the report and print a warning.
    prep=None
    tactile_error=None
    try:
        from openpi.spatial.preprocess import SpatialPreprocessor
        prep=SpatialPreprocessor.from_repo_root(repo_root=repo,config=cfg)
    except Exception as exc:
        tactile_error=f'{type(exc).__name__}: {exc}'
        print('[WARN] Cannot initialize tactile FK, continuing visual QA:',tactile_error,flush=True)

    labels={
        0:'target_flask',
        200:'funnel',
        212:'funnel',
        240:'funnel',
    }
    summaries=[]
    for frame in ids:
        rgb=image_map[frame]
        row=rows[frame]
        kind=labels[frame]
        candidate_path=None
        crop_file=None
        if frame==0 and args.flask_mask:
            candidate_path=Path(args.flask_mask)
            if args.flask_crop_manifest:
                crop_file=Path(args.flask_crop_manifest)
        elif frame==0:
            # Prefer the attempted full-glass segmentation (the zoom mask is
            # mapped back using explicit crop metadata); fall back to mouth.
            zoom=hoi/'outputs/sam2_target_flask_zoom_v1/masks/obj_01/00000.png'
            zoom_manifest=hoi/'data/sam2_target_flask_zoom_v1/manifest.json'
            mouth=hoi/'outputs/sam2_target_flask_mouth_v1/masks/obj_01/00000.png'
            if zoom.is_file() and zoom_manifest.is_file():
                candidate_path=zoom;crop_file=zoom_manifest
            elif mouth.is_file():
                candidate_path=mouth
                kind='target_flask_MOUTH_ONLY'
        elif frame==212:
            candidate_path=hoi/'outputs/sam2_funnel_212_v1/masks/obj_01/00000.png'
        else:
            candidate_path=Path(args.funnel_masks)/f'episode0_frame{frame:04d}.png'
        if candidate_path is None or not candidate_path.is_file():
            warning=f'[MISSING MASK] f{frame} {kind}: {candidate_path}. '
            if frame in (200,240):warning+='Run track-funnel first.'
            print(warning,flush=True)
            summaries.append({'episode':0,'frame':frame,'object':kind,'status':'missing_mask',
                              'mask_path':str(candidate_path)})
            continue
        mask,mask_meta=_fullmask_to_rgb(candidate_path,rgb.shape[:2],crop_file=crop_file)
        if mask.sum()==0:
            raise RuntimeError(f'Frame {frame} mask is empty: {candidate_path}')
        depth=np.asarray(row['observation.depths.cam_front'])
        xyz_base,xyz_color,depth_pixel=_flatten_depth_to_points(depth,cam)
        depth_z_m = depth.reshape(-1)[depth_pixel].astype(np.float32)*np.float32(cam.depth_scale_m_per_unit)
        uv,projected,inside,in_range,workspace,candidate=_project_mask_filter(
            xyz_base,xyz_color,depth_z_m,mask,cam,cfg)
        valid_idx=np.flatnonzero(projected)
        selected_idx=np.flatnonzero(candidate)
        # pixel hit rate, NOT real surface coverage
        unique_proj=np.unique(uv[selected_idx,1]*rgb.shape[1]+uv[selected_idx,0]).size if len(selected_idx) else 0
        candidate_colors=rgb[uv[selected_idx,1],uv[selected_idx,0]] if len(selected_idx) else np.zeros((0,3),dtype=np.uint8)
        tactile_xyz=None
        if prep is not None:
            try:
                tactile=prep.preprocess_tactile(np.asarray(row['observation.state'],dtype=np.float64))
                tactile_xyz=tactile.xyz_m
            except Exception as exc:
                msg=f'{type(exc).__name__}: {exc}'
                tactile_error=msg
                print(f'[WARN] Frame {frame} tactile FK failed: {msg}',flush=True)
        fdir=out/f'frame_{frame:04d}_{kind}'
        fdir.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(fdir/'mask_rgb.png'),mask.astype(np.uint8)*255)
        _write_ply_ascii(fdir/'candidate_depth_in_mask_base.ply',xyz_base[selected_idx],candidate_colors)
        np.savez_compressed(fdir/'candidates_base.npz',xyz_base_m=xyz_base[selected_idx],
                            rgb=candidate_colors,depth_pixel_index=depth_pixel[selected_idx],
                            color_pixel_uv=uv[selected_idx].astype(np.int32))
        _plots(frame,kind,rgb,mask,uv,inside,candidate,xyz_base,tactile_xyz,fdir)
        one={
          'episode':0,'frame':frame,'object':kind,'status':'completed',
          'rgb_resolution_hw':list(rgb.shape[:2]),'depth_resolution_hw':list(depth.shape),
          'mask_path':str(candidate_path),'mask_mapping':mask_meta,
          'mask_area_pixels':int(mask.sum()),
          'valid_depth_pixels_anywhere':int(len(xyz_base)),
          'depth_points_projected_to_rgb':int(projected.sum()),
          'depth_points_projected_inside_mask_before_roi':int(inside.sum()),
          'mask_selected_depth_points_in_workspace_roi':int(candidate.sum()),
          'distinct_mask_pixels_hit_by_candidates':int(unique_proj),
          'mask_pixel_hit_fraction':float(unique_proj/mask.sum()),
          'roi_rejection_after_mask':int((inside&(~workspace)).sum()),
          'candidate_color_depth_Z_m_quantiles':_quantiles(xyz_color[selected_idx,2]),
          'candidate_base_xyz_m_quantiles':{axis:_quantiles(xyz_base[selected_idx,j]) for j,axis in enumerate('xyz')},
          'tactile_fk_present':tactile_xyz is not None,
          'files':['mask_rgb.png','overlay.png','base_xyz_views.png','candidate_depth_in_mask_base.ply','candidates_base.npz'],
          'critical_warning':('These are ONLY depth samples whose RGB projection fell inside a SAM2 mask. '
                              'For transparent glass the cloud may mainly represent background/fixtures. '
                              'ROI filtering and mask membership do NOT establish real object ownership.')
        }
        _save_json(fdir/'summary.json',one)
        summaries.append(one)
        print(f'[QA] f{frame:03d} {kind}: mask_area={mask.sum()}, candidate_points={candidate.sum()}, '
              f'mask_pixel_hit={one["mask_pixel_hit_fraction"]:.3f}, tactile={tactile_xyz is not None}',flush=True)
        print(f'     {fdir}/overlay.png  |  base_xyz_views.png',flush=True)
    report={'dataset':str(dataset_root),'camera_role':'front','episode':0,
            'front_calibration_profile':str(args.front_calibration_profile) if args.front_calibration_profile else None,
            'tactile_fk_error':tactile_error,'frames':summaries,
            'caution':'Measured depth behind glass cannot be distinguished from glass surface using masks alone.'}
    _save_json(out/'report.json',report)
    print(f'[DONE] report: {out / "report.json"}',flush=True)
    skipped=[x['frame'] for x in summaries if x['status']!='completed']
    if skipped:
        print(f'[INCOMPLETE] Missing masks for frames {skipped}; run track-funnel and rerun QA to a NEW --output path.',flush=True)
        if args.require_all:
            sys.exit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sp=p.add_subparsers(dest='action',required=True)
    t=sp.add_parser('track-funnel',help='SAM2 propagate accepted frame 212 mask to 200/240')
    t.add_argument('--clip',default=str(HOI/'data/sam2_1018_e0_f170_289'))
    t.add_argument('--checkpoint',default=str(HOI/'weights/sam2/sam2.1_hiera_small.pt'))
    t.add_argument('--anchor-mask',default=str(HOI/'outputs/sam2_funnel_212_v1/masks/obj_01/00000.png'))
    t.add_argument('--anchor',type=int,default=212)
    t.add_argument('--targets',default='200,212,240')
    t.add_argument('--mask-output',default=str(HOI/'outputs/level1_funnel_propagated_masks'))
    t.set_defaults(fn=track_funnel)
    q=sp.add_parser('qa',help='Read actual RGB/raw depth/calibration and evaluate candidate object clouds')
    q.add_argument('--repo',default=str(REPO))
    q.add_argument('--hoi',default=str(HOI))
    q.add_argument('--dataset',default=str(DATASET))
    q.add_argument('--output',default=str(DEFAULT_OUT))
    q.add_argument('--funnel-masks',default=str(HOI/'outputs/level1_funnel_propagated_masks'))
    q.add_argument('--front-calibration-profile',default=None)
    q.add_argument('--flask-mask',default=None,help='Optional replacement flask mask, RGB-native or with crop manifest')
    q.add_argument('--flask-crop-manifest',default=None,help='Required if flask-mask differs from RGB dimensions')
    q.add_argument('--require-all',action='store_true')
    q.set_defaults(fn=qa)
    a=p.parse_args();a.fn(a)

if __name__=='__main__':
    main()
