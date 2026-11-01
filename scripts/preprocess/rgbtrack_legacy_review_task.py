#!/usr/bin/env python3
"""Task-generic entry point to the EXISTING E23 3D tracking viewer.

This deliberately reuses scripts/object_geometry/pose_tracking/
visualize_rgbtrack_full_episode.py for rendering, tactile features, geometry,
RGB projection, and scene points; it does not infer or alter any object pose.

The older viewer refers to its two objects as funnel and target_flask internally.
The adapter maps them to task manifest names and rewrites the generated HTML
references consistently. No new 3D renderer or tracker is introduced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ALIASES = ('funnel', 'target_flask')


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for piece in iter(lambda: f.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def verify_sources(repo: Path, task: str, ep: int, run: Path, calibrated_profile: Path):
    m = read_json(run / 'manifest.json')
    if m.get('episode') != ep or int(m.get('frames_written', -1)) < 1:
        raise ValueError('Pose run is not a complete tracking run for requested episode')
    data = (repo / 'data' / task).resolve()
    if Path(m.get('dataset', '/MISSING')).resolve() != data:
        raise ValueError('Tracking run belongs to another dataset')
    names = m.get('objects')
    if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 2:
        raise ValueError('Expected exactly two named objects from tracking manifest')
    base = repo / 'output/preprocess' / task
    approval = read_json(base / 'geometry/approved/manifest.json')
    if approval.get('status') != 'human_accepted' or approval.get('objects') != names:
        raise ValueError('Tracking objects do not match approved geometry')
    for name in names:
        src = (base / 'geometry/approved' / (name + '.glb')).resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        conv = m.get('mesh_conversions', {}).get(name, {})
        if Path(conv.get('source_glb', '/MISSING')).resolve() != src:
            raise ValueError(f'{name} pose run was made using a different mesh path')
        if conv.get('source_sha256') != hash_file(src):
            raise ValueError(f'{name} pose run mesh hash differs from approved')
        if float(conv.get('scale_m_per_unit', -1)) != 1:
            raise ValueError(f'{name} pose run not using metre-scale mesh')
    sys.path.insert(0, str(repo / 'scripts/object_geometry/pose_tracking'))
    from rgbtrack_adapter import load_front_calibration
    camera = load_front_calibration(repo, calibrated_profile)
    # Same contract as original E23 viewer: prevent mixing camera calibration.
    from openpi.object_geometry.calibration import calibration_provenance_error
    error = calibration_provenance_error(m.get('calibration'), camera.provenance)
    if error:
        raise ValueError(f'Pose run front calibration mismatch: {error}')
    return m, names, camera


def load_pose_streams(legacy, run: Path, names: list[str], n_frames: int):
    # Keep original pose matrix validation, invalid-frame behavior and statuses.
    actual = {name: legacy.read_poses(run / f'{name}_poses.jsonl', name) for name in names}
    expected = set(range(n_frames))
    for name, rows in actual.items():
        if set(rows) != expected:
            raise ValueError(f'{name}: expected pose IDs 0..{n_frames-1}')
    return {alias: actual[name] for alias, name in zip(ALIASES, names)}


def remap_original_html(html_text: str, names: list[str]) -> str:
    """Map old viewer's internal display keys + labels consistently.

    Keep the mapping order: replace target_flask first, then funnel.  The
    original HTML uses these canonical identifiers in both JS and JSON; since
    both are renamed in one pass, frame synchronisation stays intact.
    """
    if len(names) != 2 or any(not x.isidentifier() for x in names):
        raise ValueError('Unsupported object names for original viewer labels')
    if any(x in ALIASES for x in names):
        # Identical or swapped identifiers risk cross-renaming; use placeholders.
        html_text = html_text.replace('target_flask', '__ORIGINAL_TARGET_FLASK__').replace('funnel', '__ORIGINAL_FUNNEL__')
        html_text = html_text.replace('__ORIGINAL_TARGET_FLASK__', names[1]).replace('__ORIGINAL_FUNNEL__', names[0])
    else:
        html_text = html_text.replace('target_flask', names[1]).replace('funnel', names[0])
    return html_text.replace('漏斗 mesh', names[0] + ' mesh').replace('目标烧瓶 mesh', names[1] + ' mesh')


def mask_contours(rgb: np.ndarray, mask_root: Path, names: list[str], frame: int):
    import cv2
    image = rgb.copy()
    colors = ((20, 240, 90), (80, 160, 250))  # RGB image supplied to cv2 drawing
    for name, col in zip(names, colors):
        folder = mask_root / name
        for stem in (f'{frame:05d}', f'{frame:06d}'):
            candidate = folder / (stem + '.png')
            if candidate.is_file():
                break
        else:
            raise FileNotFoundError(f'Mask missing: {name} frame {frame}')
        mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f'Mask missing or shape mismatch: {candidate}')
        contours, _ = cv2.findContours(np.uint8(mask > 127), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, col, 2)
    return image


def generate(args):
    repo = args.repo.resolve()
    task = args.dataset
    ep = args.episode
    base = repo / 'output/preprocess' / task
    run = (args.pose_run or (base / 'rgbtrack/episodes' / f'e{ep:05d}')).resolve()
    out = (args.output or (base / 'rgbtrack/reviews' / f'e{ep:05d}_legacy_style')).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite previous review: {out}')
    profile = args.front_calibration
    if not profile.is_absolute():
        profile = repo / profile
    profile = profile.resolve()
    if not (1 <= args.viz_stride <= 100):
        raise ValueError('viz-stride must be 1..100')
    if args.scene_cameras not in ('none', 'front', 'front,left'):
        raise ValueError('Only calibrated scene cameras front / left supported')
    sys.path.insert(0, str(repo / 'scripts/object_geometry/pose_tracking'))
    import visualize_rgbtrack_full_episode as legacy
    manifest, names, cam = verify_sources(repo, task, ep, run, profile)
    streams = load_pose_streams(legacy, run, names, int(manifest['frames_written']))
    total = int(manifest['frames_written'])
    chosen = sorted(set(range(0, total, args.viz_stride)) | {total - 1})
    chosen_set = set(chosen)
    if args.scene_cameras != 'none':
        scene_roles = tuple(args.scene_cameras.split(','))
    else:
        scene_roles = ()

    from openpi.object_geometry.calibration import make_front_geometry_config
    from openpi.spatial.preprocess import SpatialPreprocessor
    from openpi.spatial_dataset.source import RawSpatialDataset
    cfg, _ = make_front_geometry_config(repo, profile)
    tactile_processor = SpatialPreprocessor.from_repo_root(repo_root=repo, config=cfg)
    ds = RawSpatialDataset(repo / 'data' / task, episode=ep)
    local = {
        alias: legacy.get_mesh(base / 'geometry/approved' / f'{name}.glb', args.mesh_points, seed=20261031+i)
        for i, (alias, name) in enumerate(zip(ALIASES, names))
    }
    history = {alias: [] for alias in ALIASES}
    payload, trace_records = [], []
    try:
        from scipy.spatial import cKDTree
        def nearest(points, hand):
            return cKDTree(points).query(hand, k=1, workers=1)[0]
    except ImportError:
        def nearest(points, hand):
            return np.min(np.linalg.norm(points[:, None, :] - hand[None, :, :], axis=-1), axis=1)

    for row in ds.iter_rows(depth_roles=()):
        frame = int(row['frame_index'])
        if frame >= total:
            raise ValueError('Dataset has more frames than pose stream')
        state = np.asarray(row['observation.state']).reshape(-1)
        tactile = tactile_processor.preprocess_tactile(state)
        hand = np.asarray(tactile.xyz_m, np.float32)
        fid = np.asarray(tactile.finger_id, np.int32)
        if hand.shape != (600, 3) or not np.isfinite(hand).all():
            raise ValueError(f'Bad tactile FK positions at frame {frame}: {hand.shape}')
        xyz, dist, row_rec = {}, {}, {'frame_index': frame}
        for alias in ALIASES:
            p = streams[alias][frame]
            good = bool(p.get('valid'))
            if good:
                xyz[alias] = legacy.transform(local[alias], p['_T'])
                d = nearest(xyz[alias], hand)
                dist[alias] = [float(np.min(d[fid == f])) for f in range(5)]
                if p.get('status') == 'observed':
                    history[alias].append(p['_T'][:3, 3].round(4).tolist())
            else:
                xyz[alias] = None
                dist[alias] = [None] * 5
            row_rec[f'{alias}_valid'] = good
            row_rec[f'{alias}_status'] = str(p.get('status', 'unknown'))
            for f, val in enumerate(dist[alias]):
                row_rec[f'{alias}_finger{f}_nearest_m'] = val if val is not None else ''
        trace_records.append(row_rec)
        if frame in chosen_set:
            payload.append({
                'frame': frame,
                'pose': {a: {'valid': bool(streams[a][frame].get('valid')),
                             'status': str(streams[a][frame].get('status', 'unknown')),
                             'confidence': streams[a][frame].get('confidence')} for a in ALIASES},
                'object_xyz': {a: xyz[a].round(4).tolist() if xyz[a] is not None else [] for a in ALIASES},
                'hand': legacy.downsample_taxels(hand, fid, np.asarray(tactile.force_norm), args.taxels_per_finger),
                'distance_cm': {a: [round(100 * v, 3) if v is not None else None for v in dist[a]] for a in ALIASES},
                'trajectory': {a: history[a][-100:] for a in ALIASES},
            })
    if len(trace_records) != total:
        raise ValueError(f'Dataset rows {len(trace_records)} != poses {total}')

    by_frame = {item['frame']: item for item in payload}
    T_cam_base = np.linalg.inv(legacy.check_pose(cam.T_base_color, 'front camera T_base_color'))
    if scene_roles:
        from openpi.spatial.config import make_baseline_config
        from openpi.spatial.calibration_profile import apply_front_calibration_profile
        scene_cfg = make_baseline_config().with_camera_roles(*scene_roles)
        scene_cfg = apply_front_calibration_profile(scene_cfg, profile)
        scene_processor = SpatialPreprocessor.from_repo_root(repo_root=repo, config=scene_cfg)
        streams_rgb = {role: iter(ds.iter_video_frames(role, selected=chosen_set)) for role in scene_roles}
        for row in ds.iter_rows(selected=chosen_set, depth_roles=scene_roles):
            frame = int(row['frame_index'])
            rgb_roles = {}
            for role in scene_roles:
                v = next(streams_rgb[role])
                if int(v.frame_index) != frame:
                    raise ValueError('Scene RGB/depth frame misalignment')
                rgb_roles[role] = np.asarray(v.rgb)
            depth_roles = {role: np.asarray(row[f'observation.depths.cam_{role}']) for role in scene_roles}
            geo = scene_processor.preprocess_visual(depth_by_role=depth_roles, rgb_by_role=rgb_roles)
            xyz = np.asarray(geo.xyz_m, np.float32)
            rgb = np.asarray(geo.rgb, np.uint8)
            valid = np.asarray(geo.rgb_valid, bool)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                raise ValueError(f'Invalid scene points f{frame}')
            ix = np.linspace(0, len(xyz)-1, min(args.scene_display_points, len(xyz))).round().astype(int)
            item = by_frame[frame]
            item['scene_xyz'] = xyz[ix].round(4).tolist()
            item['scene_rgb'] = ['rgb(%d,%d,%d)' % tuple(map(int, c)) if v else '#888888'
                                 for c, v in zip(rgb[ix], valid[ix])]
            if 'front' in rgb_roles:
                rgb_front = rgb_roles['front']
                if args.mask_contours:
                    rgb_front = mask_contours(rgb_front, base / 'sam2/episodes' / f'e{ep:05d}' / 'masks', names, frame)
                objects_xyz = {a: np.asarray(item['object_xyz'][a], np.float32)
                               if item['object_xyz'][a] else None for a in ALIASES}
                item['jpg'] = legacy._rgb_to_jpeg_data(rgb_front, objects_xyz, T_cam_base, cam.K_color)
            else:
                item['jpg'] = None
    else:
        for v in ds.iter_video_frames('front', selected=chosen_set):
            frame = int(v.frame_index)
            item = by_frame[frame]
            rgb_front = np.asarray(v.rgb)
            if args.mask_contours:
                rgb_front = mask_contours(rgb_front, base / 'sam2/episodes' / f'e{ep:05d}' / 'masks', names, frame)
            objects_xyz = {a: np.asarray(item['object_xyz'][a], np.float32)
                           if item['object_xyz'][a] else None for a in ALIASES}
            item['jpg'] = legacy._rgb_to_jpeg_data(rgb_front, objects_xyz, T_cam_base, cam.K_color)
    for item in payload:
        if 'jpg' not in item:
            raise ValueError(f'No corresponding RGB frame for f{item["frame"]}')

    report = {
        'schema_version': 1,
        'dataset': task, 'episode': ep,
        'pose_run': str(run), 'pose_manifest_sha256': hash_file(run / 'manifest.json'),
        'objects': names, 'frames_all': total, 'frames_html': len(payload),
        'scene_cameras': list(scene_roles),
        'mesh_source': {name: str(base / 'geometry/approved' / f'{name}.glb') for name in names},
        'warning': 'Review only. valid/score is not proven 6D accuracy. No pose values changed.',
        'status_counts': {name: dict(Counter(p.get('status', 'unknown') for p in streams[alias].values()))
                          for alias, name in zip(ALIASES, names)},
    }
    # The original HTML renderer stays authoritative: no reimplementation.
    document = legacy.html_document(payload, report, f'RGBTrack 3D full review: {task} E{ep}')
    document = remap_original_html(document, names)
    out.mkdir(parents=True)
    import csv
    with (out / 'finger_nearest_m.csv').open('w', newline='', encoding='utf-8') as fh:
        cols = sorted(set().union(*(r.keys() for r in trace_records)))
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(trace_records)
    (out / 'review.html').write_text(document, encoding='utf-8')
    (out / 'review_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[REVIEW]', out / 'review.html', flush=True)
    print('[STATS]', out / 'review_report.json', flush=True)
    print('[NOTE] Uses existing E23 3D HTML renderer and approved meshes; baseline poses unchanged.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--dataset', required=True, help='Task name (e.g. 1028_press_button)')
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--pose-run', type=Path, default=None, help='Optional alternative tracking output to compare')
    p.add_argument('--output', type=Path, default=None, help='New, non-existing destination directory')
    p.add_argument('--front-calibration', type=Path, default=Path('configs/ur7e_xhand/front_calibration_candidate.json'))
    p.add_argument('--viz-stride', type=int, default=1, help='1 renders every frame, same as original E23')
    p.add_argument('--scene-cameras', choices=('none', 'front', 'front,left'), default='front')
    p.add_argument('--scene-display-points', type=int, default=800)
    p.add_argument('--mesh-points', type=int, default=220)
    p.add_argument('--taxels-per-finger', type=int, default=30)
    p.add_argument('--mask-contours', action=argparse.BooleanOptionalAction, default=True)
    generate(p.parse_args())


if __name__ == '__main__':
    main()
