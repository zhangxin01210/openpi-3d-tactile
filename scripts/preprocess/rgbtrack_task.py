#!/usr/bin/env python3
"""Dataset-level, two-object RGBTrack front-end. Reuses rgbtrack_adapter.py.

Inputs (all task scoped):
  output/preprocess/<task>/sam2/episodes/eXXXXX/masks/{object}/{frame}.png
  output/preprocess/<task>/geometry/approved/{object}.glb
Outputs:
  output/preprocess/<task>/rgbtrack/episodes/eXXXXX/{object}_poses.jsonl

Formal two-object RGBTrack pipeline. Dynamic object stays native; optional upright+mask
initialization and freeze only applies to a configured box-like static object.
Uses shared factory-SDK front RGB-D registration (must already be installed).
Produces isolated, provenance-hashed runs and optional legacy full 3D review HTML.
Not validated for transparent / non-convex static meshes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def task_inputs(repo: Path, task: str) -> dict:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', task):
        raise ValueError('Invalid task name')
    dataset = repo / 'data' / task
    base = repo / 'output/preprocess' / task
    sam = read_json(base / 'sam2/manifest.json')
    geo = read_json(base / 'geometry/approved/manifest.json')
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if Path(sam['dataset']).resolve() != dataset.resolve():
        raise ValueError('SAM2 manifest refers to another LeRobot dataset')
    objects = list(sam.get('objects', []))
    if len(objects) != 2 or len(set(objects)) != 2 or objects != list(geo.get('objects', [])):
        raise ValueError('SAM2 and approved-geometry object names do not match exactly')
    if geo.get('status') != 'human_accepted' or geo.get('dataset_id') != task:
        raise ValueError('Geometry not approved for this dataset')
    if sam.get('camera') != 'front':
        raise ValueError('This version expects the calibrated front camera; do not silently use wrist camera')
    meshes = {}
    for name in objects:
        src = base / 'geometry/approved' / f'{name}.glb'
        cfg = geo['assets'][name]
        if not src.is_file() or cfg['sha256'] != digest(src):
            raise ValueError(f'{name}: missing or changed approved mesh')
        if float(cfg['scale_m_per_unit']) != 1.0:
            raise ValueError(f'{name}: approved mesh must already be metres (scale=1.0)')
        meshes[name] = src
    return {'repo': repo, 'task': task, 'dataset': dataset, 'base': base, 'sam': sam, 'geo': geo,
            'objects': objects, 'meshes': meshes, 'episode_ids': [int(x) for x in sam['episode_ids']]}


def select_episodes(spec: str | None, ids: list[int], *, trial: bool) -> list[int]:
    if spec is None:
        if not trial:
            return ids
        # Five evenly distributed *indices*, not a hard-coded episode count.
        n = min(len(ids), 5)
        return sorted({ids[round(i * (len(ids) - 1) / max(n - 1, 1))] for i in range(n)})
    if spec == 'all':
        return ids
    episodes = []
    for token in spec.split(','):
        if '-' in token:
            a, b = map(int, token.split('-', 1))
            if b < a:
                raise ValueError('Reversed episode range')
            episodes.extend(range(a, b + 1))
        else:
            episodes.append(int(token))
    result = sorted(set(episodes))
    if not result or not set(result).issubset(set(ids)):
        raise ValueError('Selected episodes are missing from SAM2 manifest')
    return result


def verify_episode(info: dict, ep: int) -> tuple[Path, int]:
    root = info['base'] / 'sam2/episodes' / f'e{ep:05d}'
    report = read_json(root / 'tracking_report.json')
    if report.get('status') != 'tracking_complete_unreviewed' or int(report.get('episode', -1)) != ep:
        raise ValueError(f'e{ep}: SAM2 video tracking has not completed')
    if set(report.get('objects', [])) != set(info['objects']):
        raise ValueError(f'e{ep}: SAM2 tracked objects differ')
    frames = int(report.get('frames', 0))
    if frames < 2:
        raise ValueError(f'e{ep}: too few video frames')
    mask_root = root / 'masks'
    expected = set(range(frames))
    expected_hashes = report.get('mask_stream_sha256')
    if not isinstance(expected_hashes, dict):
        raise ValueError(f'e{ep}: missing SAM2 mask-stream provenance')
    for name in info['objects']:
        folder = mask_root / name
        observed = set()
        for path in folder.glob('*.png'):
            if not path.stem.isdigit():
                raise ValueError(f'e{ep}: nonnumeric mask filename: {path.name}')
            observed.add(int(path.stem))
        if observed != expected or len(list(folder.glob('*.png'))) != frames:
            raise ValueError(f'e{ep}: {name} mask frame ids do not match 0..{frames-1}')
        stream = hashlib.sha256()
        for frame in range(frames):
            stream.update((folder / f'{frame:05d}.png').read_bytes())
        if stream.hexdigest() != expected_hashes.get(name):
            raise ValueError(f'e{ep}: {name} mask bytes changed since SAM2 tracking')
    return mask_root, frames


def scale_manifest(info: dict) -> Path:
    out = info['base'] / 'rgbtrack'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'scale_manifest.json'
    data = {'schema_version': 1, 'objects': {x: 1.0 for x in info['objects']}}
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f'Scale manifest conflicts with approved geometry: {path}')
    else:
        path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return path


def policy_settings(args: argparse.Namespace, info: dict) -> dict:
    static = getattr(args, 'static_object', None)
    dynamic = getattr(args, 'dynamic_object', None)
    if static is not None and static not in info['objects']:
        raise ValueError(f'Unknown static object {static!r}; expected one of {info["objects"]}')
    if dynamic is not None and dynamic not in info['objects']:
        raise ValueError(f'Unknown dynamic object {dynamic!r}; expected one of {info["objects"]}')
    if static is not None and dynamic is not None and static == dynamic:
        raise ValueError('Static and dynamic object cannot be the same')
    if dynamic is not None and static is None:
        raise ValueError('Provide --static-object as well; there are two objects in this pipeline')
    if dynamic is not None and set(info['objects']) != {static, dynamic}:
        raise ValueError('Static/dynamic roles must cover both objects')
    if getattr(args, 'mode', 'depth') != 'depth' and static:
        raise ValueError('Static mask registration has only been verified with --mode depth')
    max_tilt = float(getattr(args, 'init_tilt_deg', 25.))
    max_rank = int(getattr(args, 'init_rank', 150))
    min_iou = float(getattr(args, 'min_mask_iou', .35))
    motion_step = getattr(args, 'guard_max_step_cm', None)
    motion_turn = getattr(args, 'guard_max_turn_deg', None)
    recovery_cap = float(getattr(args, 'guard_recovery_max_turn_deg', 60.0))
    dyn_axis = getattr(args,'dynamic_init_up_axis',None)
    dyn_tilt = float(getattr(args,'dynamic_init_max_tilt_deg',25.))
    dyn_iou = float(getattr(args,'dynamic_init_min_iou',.20))
    if dyn_axis is not None and (not dynamic or not static or getattr(args,'mode','depth') != 'depth'):
        raise ValueError('Dynamic upright initialization requires static/dynamic roles and depth mode')
    if dyn_axis is not None and (not 0<dyn_tilt<90 or not 0<=dyn_iou<=1):
        raise ValueError('Invalid dynamic upright parameters')
    if (motion_step is None) != (motion_turn is None):
        raise ValueError('Provide both --guard-max-step-cm and --guard-max-turn-deg')
    if motion_step is not None:
        motion_step = float(motion_step) / 100.0
        motion_turn = float(motion_turn)
        if not static or not dynamic:
            raise ValueError('Motion guard requires explicit static/dynamic object roles')
        if not 0 < motion_step < 0.5 or not 0 < motion_turn <= recovery_cap <= 180:
            raise ValueError('Invalid motion guard bounds: recovery rotation cap must be >= per-frame bound and <= 180')
    if not 0 < max_tilt < 90 or max_rank < 1 or not 0 <= min_iou <= 1:
        raise ValueError('Invalid static initialization thresholds')
    return {'method': 'static_upright_convex_mask_v1' if static else 'native_rgbtrack',
            'static_object': static, 'dynamic_object': dynamic or next((x for x in info['objects'] if x != static), None),
            'init_tilt_deg': max_tilt, 'init_rank': max_rank, 'min_mask_iou': min_iou,
            'dynamic_init_up_axis':dyn_axis,'dynamic_init_max_tilt_deg':dyn_tilt if dyn_axis else None,
            'dynamic_init_min_iou':dyn_iou if dyn_axis else None,
            'mode': getattr(args, 'mode', 'depth'),
            'guard_max_step_m': motion_step, 'guard_max_turn_deg': motion_turn,
            'guard_recovery_max_turn_deg': recovery_cap if motion_turn is not None else None}


def source_provenance(info: dict, ep: int, calibration: Path, policy: dict) -> dict:
    """Pin both dataset inputs and the *actual common* RGB-D algorithm files."""
    sam_report = info['base'] / 'sam2/episodes' / f'e{ep:05d}' / 'tracking_report.json'
    files = {
        'camera_factory_calibration': info['repo'] / 'configs/ur7e_xhand/cameras_0903.json',
        'base_camera_extrinsic': info['repo'] / 'configs/ur7e_xhand/extrinsic_0909.yaml',
        'adapter': info['repo'] / 'scripts/object_geometry/pose_tracking/rgbtrack_adapter.py',
        'registered_depth': info['repo'] / 'src/openpi/spatial/rgbd_registration.py',
        'task': info['repo'] / 'scripts/preprocess/rgbtrack_task.py',
        'task_runner': info['repo'] / 'scripts/preprocess/rgbtrack_task_runner.py',
        'static_policy': info['repo'] / 'scripts/preprocess/rgbtrack_static_policy.py',
        'dynamic_init': info['repo'] / 'scripts/preprocess/rgbtrack_dynamic_init.py',
        'motion_guard': info['repo'] / 'scripts/preprocess/rgbtrack_motion_guard.py',
    }
    missing = [str(x) for x in files.values() if not x.is_file()]
    if missing:
        raise FileNotFoundError('Missing common RGB-D/tracking files: ' + ', '.join(missing))
    return {
        'schema_version': 2,
        'dataset': str(info['dataset'].resolve()), 'episode': int(ep),
        'approved_geometry_manifest_sha256': digest(info['base']/'geometry/approved/manifest.json'),
        'sam2_tracking_report_sha256': digest(sam_report),
        'front_calibration': str(calibration.resolve()),
        'front_calibration_sha256': digest(calibration),
        'shared_pipeline_sources_sha256': {k: digest(v) for k,v in files.items()},
        'objects': info['objects'], 'policy': policy,
    }


def completed(info: dict, ep: int, output: Path, mask_root: Path, frames: int,
              calibration: Path, policy: dict | None = None) -> bool:
    if not output.exists():
        return False
    manifest_path = output / 'manifest.json'
    source_path = output / 'task_input_provenance.json'
    if not manifest_path.is_file() or not source_path.is_file():
        raise FileExistsError(f'e{ep}: unfinished output exists; do not overwrite: {output}')
    actual = read_json(source_path)
    check_policy = policy if policy is not None else actual.get('policy', {})
    expected = source_provenance(info, ep, calibration, check_policy)
    if actual != expected:
        raise FileExistsError(f'e{ep}: source calibration/geometry/masks/code or policy changed: {output}')
    manifest = read_json(manifest_path)
    if int(manifest.get('frames_written', -1)) != frames or Path(manifest.get('mask_root', '/MISSING')).resolve() != mask_root.resolve():
        raise FileExistsError(f'e{ep}: incomplete output manifest: {output}')
    for name in info['objects']:
        path = output / f'{name}_poses.jsonl'
        if not path.is_file() or sum(bool(l.strip()) for l in path.open(encoding='utf-8')) != frames:
            raise FileExistsError(f'e{ep}: pose stream incomplete: {path}')
        if manifest.get('mesh_conversions', {}).get(name, {}).get('source_sha256') != digest(info['meshes'][name]):
            raise FileExistsError(f'e{ep}: approved mesh hash changed for {name}')
    return True

def build_command(info: dict, *, ep: int, mask_root: Path, output: Path, scale_path: Path,
                  calibration: Path, viz: bool, mode: str, max_frames: int | None) -> list[str]:
    repo = info['repo']
    cmd = [str(repo / 'scripts/object_geometry/pose_tracking/rgbtrack_adapter.py'),
           '--repo-root', str(repo), '--dataset', str(info['dataset']),
           '--episode', str(ep), '--mask-root', str(mask_root),
           '--scale-manifest', str(scale_path), '--front-calibration-profile', str(calibration),
           '--output', str(output), '--objects', *info['objects'], '--mode', mode,
           '--viz-frames', '0,50,100,150,200,300,400' if viz else '']
    for name in info['objects']:
        cmd.extend(['--mesh', f'{name}={info["meshes"][name]}'])
    if not viz:
        cmd.append('--skip-preview-html')
    if max_frames is not None:
        cmd.extend(['--max-frames', str(max_frames)])
    return cmd


def rgbtrack_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra = [str(repo / 'src'),
             str(repo / 'data/pose_eval/rgbtrack_pydeps'),
             str(repo / 'data/pose_eval/rgbtrack_sources/bop_toolkit-master')]
    env['PYTHONPATH'] = os.pathsep.join(extra + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env['PATH'] = os.pathsep.join([str(repo / 'data/pose_eval/rgbtrack_bin'), env.get('PATH', '')])
    env.setdefault('TORCH_EXTENSIONS_DIR', str(repo / 'data/pose_eval/torch_extensions_rgbtrack_native'))
    env.setdefault('CUDA_HOME', '/usr/local/cuda')
    env.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    env.setdefault('MAX_JOBS', '4')
    return env


def audit_pose_streams(info: dict, ep: int, output: Path, policy: dict) -> None:
    """Non-invasive postrun QC, not a replacement for visual human review."""
    from collections import Counter
    import numpy as np
    data = {'schema_version': 1, 'episode': ep,
            'note': 'valid/confidence are tracker flags, not 6D geometric accuracy',
            'objects': {}}
    for name in info['objects']:
        rows = [json.loads(x) for x in (output / f'{name}_poses.jsonl').read_text().splitlines() if x.strip()]
        valid = [r for r in rows if r.get('valid', False)]
        status = Counter(r.get('status','unknown') for r in rows)
        leaps = []
        previous = None
        frozen_poses = []
        for r in rows:
            if not r.get('valid',False):
                previous = None
                continue
            mat = np.asarray(r['T_base_link_object'],dtype=np.float64)
            if mat.shape != (4,4) or not np.isfinite(mat).all():
                raise ValueError(f'{name}: invalid finite 4x4 in valid record f={r["frame_index"]}')
            if policy.get('static_object') == name:
                frozen_poses.append(mat)
            if previous is not None:
                prev_index, prev_mat = previous
                d = float(np.linalg.norm(mat[:3,3] - prev_mat[:3,3]))
                if d > 0.05: # advisory only; never modify trajectory
                    leaps.append({'from':prev_index,'to':int(r['frame_index']), 'distance_m':round(d,5)})
            previous = (int(r['frame_index']),mat)
        if policy.get('static_object') == name and frozen_poses:
            max_d = max(float(np.max(np.abs(m - frozen_poses[0]))) for m in frozen_poses)
            if max_d > 1e-6:
                raise RuntimeError(f'{name}: static object changed pose by {max_d}, refusing to approve stream')
            selected_path = output / f'{name}_init_candidates.json'
            if not selected_path.is_file() or read_json(selected_path).get('selected') is None:
                raise RuntimeError(f'{name}: static object has no approved initialization audit')
        if name==policy.get('dynamic_object') and policy.get('dynamic_init_up_axis'):
            check=output / f'{name}_dynamic_init_candidates.json'
            if not check.is_file() or not read_json(check).get('selected'):
                raise RuntimeError(f'{name}: dynamic upright initialization has no selected candidate')
        data['objects'][name] = {'frames':len(rows),'valid_frames':len(valid),
                                 'valid_ratio':len(valid)/max(1,len(rows)),
                                 'statuses':dict(status),'valid_gt_5cm_adjacent_jumps':leaps[:100],
                                 'num_valid_gt_5cm_adjacent_jumps':len(leaps)}
    (output / 'quality_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for name, d in data['objects'].items():
        print(f'[QC] e{ep} {name}: valid={d["valid_frames"]}/{d["frames"]}, '
              f'large jumps={d["num_valid_gt_5cm_adjacent_jumps"]}; statuses={d["statuses"]}',flush=True)


def check_geometry_runtime(repo: Path, calibration: Path) -> None:
    """Fail closed if the verified front SDK factory RGB-D geometry was rolled back."""
    import numpy as np
    import sys as _sys
    path = str(repo / 'scripts/object_geometry/pose_tracking')
    if path not in _sys.path:
        _sys.path.insert(0, path)
    from rgbtrack_adapter import _load_front_camera_for_depth_registration
    cam = _load_front_camera_for_depth_registration(repo, calibration)
    sdk = np.array([[.999976634979248, -.00312922359444201, .006081835366785526],
                    [.003140633925795555, .9999933242797852, -.0018675047904253006],
                    [-.006075950805097818, .0018865619786083698, .9999797344207764]])
    K = cam.color_intrinsics
    if not np.allclose(cam.T_color_depth[:3, :3], sdk, atol=1e-7):
        raise RuntimeError('Front Depth->Color runtime rotation is not the verified SDK orientation')
    if not np.allclose([K.fx,K.fy,K.cx,K.cy],
                       [606.449951171875,606.171142578125,323.8404541015625,234.1532440185547], atol=1e-4):
        raise RuntimeError('Front Color runtime K is not the verified factory K')
    print('[RGB-D SDK VERIFIED] front serial 347622074420 profile 640x480 raw depth expected',flush=True)


def review_one(info: dict, ep: int, output: Path, tag: str, calibration: Path,
               stride: int, *, dry_run: bool=False) -> Path:
    script = info['repo'] / 'scripts/preprocess/rgbtrack_legacy_review_task.py'
    if not script.is_file():
        raise FileNotFoundError('Missing previously validated legacy HTML viewer: ' + str(script))
    review = info['base'] / 'rgbtrack/reviews' / tag / f'e{ep:05d}'
    cmd = [str(info['repo'] / '.venv/bin/python'), str(script),
           '--repo', str(info['repo']), '--dataset', info['task'], '--episode', str(ep),
           '--pose-run', str(output), '--output', str(review),
           '--front-calibration', str(calibration), '--scene-cameras', 'front',
           '--viz-stride', str(stride)]
    print('[REVIEW]', ' '.join(cmd), flush=True)
    if dry_run:
        return review
    if review.exists():
        # Do not regenerate an old HTML from potentially changed inputs.
        if (review / 'review.html').is_file():
            print('[REVIEW EXISTS]', review / 'review.html', flush=True)
            return review
        raise FileExistsError(f'Partial review directory exists: {review}')
    subprocess.run(cmd, cwd=info['repo'], env=rgbtrack_env(info['repo']), check=True)
    if not (review / 'review.html').is_file():
        raise RuntimeError(f'Legacy viewer did not produce review.html in {review}')
    print('[HTML]', review / 'review.html', flush=True)
    return review


def run_batch(args: argparse.Namespace, info: dict, episodes: list[int]) -> None:
    repo = info['repo']
    calibration = (args.front_calibration if args.front_calibration.is_absolute()
                   else repo / args.front_calibration).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    scale = scale_manifest(info)
    policy = policy_settings(args, info)
    tag = args.run_tag + (f'_debug{args.max_frames}' if args.max_frames is not None else '')
    runner = (args.runner or (Path(os.environ.get('HOI','/workspace/mnt/sqzhang26/hoi')) / 'envs/sam3d/bin/python')).expanduser()
    if not runner.is_absolute():
        runner = repo / runner
    runner = runner.absolute()
    adapter = repo / 'scripts/object_geometry/pose_tracking/rgbtrack_adapter.py'
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    adapter_src = adapter.read_text(encoding='utf-8')
    if '"--mesh", action="append"' not in adapter_src or '"--skip-preview-html"' not in adapter_src:
        raise RuntimeError('RGBTrack adapter generic --mesh support is missing')
    # Calib check uses CPU only; on A800 run from the repository's venv.
    if not args.dry_run:
        check_geometry_runtime(repo, calibration)
    print(f'[TASK] {info["task"]} objects={info["objects"]} static={policy["static_object"]} '
          f'episodes={episodes} tag={tag}', flush=True)
    for ep in episodes:
        mask_root, frames = verify_episode(info, ep)
        output = info['base'] / 'rgbtrack/runs' / tag / 'episodes' / f'e{ep:05d}'
        if completed(info, ep, output, mask_root, frames, calibration, policy):
            print(f'[SKIP VERIFIED] e{ep}', flush=True)
            if args.review:
                review_one(info, ep, output, tag, calibration, args.viz_stride, dry_run=args.dry_run)
            continue
        cmd = build_command(info, ep=ep, mask_root=mask_root, output=output,
                            scale_path=scale, calibration=calibration, viz=False,
                            mode=args.mode, max_frames=args.max_frames)
        if policy['static_object'] or policy['guard_max_step_m'] is not None or policy['dynamic_init_up_axis']:
            cmd[0] = str(repo / 'scripts/preprocess/rgbtrack_task_runner.py')
            if policy['static_object']:
                cmd.extend(['--static-object', policy['static_object'],
                            '--init-tilt-deg', str(policy['init_tilt_deg']),
                            '--init-rank', str(policy['init_rank']),
                            '--min-mask-iou', str(policy['min_mask_iou'])])
            if policy['dynamic_init_up_axis']:
                cmd.extend(['--dynamic-object',policy['dynamic_object'],
                            '--dynamic-init-up-axis',policy['dynamic_init_up_axis'],
                            '--dynamic-init-max-tilt-deg',str(policy['dynamic_init_max_tilt_deg']),
                            '--dynamic-init-min-iou',str(policy['dynamic_init_min_iou'])])
            if policy['guard_max_step_m'] is not None:
                cmd.extend(['--dynamic-object', policy['dynamic_object'],
                            '--guard-max-step-m', str(policy['guard_max_step_m']),
                            '--guard-max-turn-deg', str(policy['guard_max_turn_deg']),
                            '--guard-recovery-max-turn-deg', str(policy['guard_recovery_max_turn_deg'])])
        full_command = [str(runner), *cmd]
        print('[RUN]', ' '.join(full_command), flush=True)
        if args.dry_run:
            continue
        if not runner.is_file():
            raise FileNotFoundError(runner)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite previous tracking: {output}')
        subprocess.run(full_command, cwd=repo, env=rgbtrack_env(repo), check=True)
        if args.max_frames is None:
            manifest = read_json(output / 'manifest.json')
            if int(manifest.get('frames_written', -1)) != frames:
                raise RuntimeError(f'e{ep}: incomplete RGBTrack output, do not continue to review')
            audit_pose_streams(info, ep, output, policy)
            (output / 'task_input_provenance.json').write_text(
                json.dumps(source_provenance(info, ep, calibration, policy), indent=2) + '\n', encoding='utf-8')
            if not completed(info, ep, output, mask_root, frames, calibration, policy):
                raise RuntimeError(f'e{ep}: output verification failed')
            print(f'[TRACKED] e{ep}: {output}', flush=True)
            if args.review:
                review_one(info, ep, output, tag, calibration, args.viz_stride)
        else:
            print('[DEBUG ONLY] incomplete frames cannot be marked completed', flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(os.environ.get('REPO','/workspace/mnt/sqzhang26/openpi-3d-tactile')))
    p.add_argument('--dataset', required=True, help='Dataset name, e.g. 1028_press_button')
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('check', 'trial', 'batch', 'review', 'status'):
        s = sub.add_parser(command)
        if command not in ('status',):
            s.add_argument('--episodes', default=('0' if command == 'review' else None))
        if command != 'check':
            s.add_argument('--run-tag', default='sdk_static_mask_v1', help='Isolated run namespace; never overwrites prior experiments')
        if command in ('trial','batch'):
            s.add_argument('--front-calibration', type=Path,
                           default=Path('configs/ur7e_xhand/front_calibration_candidate.json'))
            s.add_argument('--runner', type=Path, default=None)
            s.add_argument('--mode', choices=('depth','no_depth','unknown_scale'), default='depth')
            s.add_argument('--static-object', help='Object fixed in world after upright+mask initialization')
            s.add_argument('--dynamic-object', help='Optional other object name; native RGBTrack plus optional motion guard')
            s.add_argument('--dynamic-init-up-axis', choices=['+x','-x','+y','-y','+z','-z'],
                           help='Opt-in upright mesh axis for dynamic object FIRST registration only')
            s.add_argument('--dynamic-init-max-tilt-deg',type=float,default=25.)
            s.add_argument('--dynamic-init-min-iou',type=float,default=.20)
            s.add_argument('--guard-max-step-cm', type=float, help='Opt-in max 3D translation per frame (cm); rejects bad pose, never clamps')
            s.add_argument('--guard-max-turn-deg', type=float, help='Opt-in max SO(3) angle per frame (degrees); must accompany --guard-max-step-cm')
            s.add_argument('--guard-recovery-max-turn-deg', type=float, default=60.,
                           help='Maximum total orientation change from last trusted pose during re-registration (default: 60 degrees)')
            s.add_argument('--init-tilt-deg', type=float, default=25.)
            s.add_argument('--init-rank', type=int, default=150)
            s.add_argument('--min-mask-iou', type=float, default=.35)
            s.add_argument('--max-frames', type=int, default=None, help='Debug only; uses separate _debugN namespace')
            s.add_argument('--dry-run', action='store_true')
            s.add_argument('--review', action='store_true', help='Produce familiar full 3D/RGB HTML for selected episodes')
            s.add_argument('--viz-stride', type=int, default=1)
        if command=='review':
            s.add_argument('--front-calibration', type=Path, default=Path('configs/ur7e_xhand/front_calibration_candidate.json'))
            s.add_argument('--viz-stride', type=int, default=1)
        if command=='status':
            s.add_argument('--front-calibration', type=Path, default=Path('configs/ur7e_xhand/front_calibration_candidate.json'))
    args = p.parse_args(argv)
    if args.command != 'check' and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run_tag):
        p.error('--run-tag must contain only letters, digits, underscores or hyphens')
    if args.command in ('trial','batch') and (args.max_frames is not None and args.max_frames < 1):
        p.error('--max-frames must be positive, or omit for the full episode')
    if args.command in ('trial','batch') and args.viz_stride < 1:
        p.error('--viz-stride must be >= 1')
    repo = args.repo.expanduser().resolve()
    info = task_inputs(repo, args.dataset)
    if args.command == 'status':
        root = info['base'] / 'rgbtrack/runs' / args.run_tag / 'episodes'
        calibration = (args.front_calibration if args.front_calibration.is_absolute() else repo / args.front_calibration).resolve()
        done = []
        problematic = []
        for ep in info['episode_ids']:
            path = root / f'e{ep:05d}'
            if not path.exists():
                continue
            try:
                mask, frames = verify_episode(info, ep)
                if completed(info, ep, path, mask, frames, calibration): done.append(ep)
            except (FileExistsError, FileNotFoundError, ValueError) as err:
                problematic.append((ep,str(err)))
        print(f'[STATUS] tag={args.run_tag} done={len(done)}/{len(info["episode_ids"])} episodes={done}')
        if problematic:
            print(f'[WARN] {len(problematic)} incomplete/changed episodes (showing up to 10)')
            for ep, msg in problematic[:10]: print(f'  e{ep}: {msg}')
        return
    episodes = select_episodes(args.episodes, info['episode_ids'], trial=args.command=='trial')
    errors = []
    for ep in episodes:
        try:
            _, frames = verify_episode(info, ep)
            print(f'[CHECK] e{ep} frames={frames}', flush=True)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f'e{ep}: {exc}')
    if errors:
        raise RuntimeError('Invalid episode inputs:\n' + '\n'.join(errors[:30]))
    if args.command == 'check':
        print(f'[CHECK OK] dataset={args.dataset} objects={info["objects"]}; {len(episodes)} episodes')
        return
    if args.command=='review':
        calibration = (args.front_calibration if args.front_calibration.is_absolute() else repo / args.front_calibration).resolve()
        for ep in episodes:
            mask, frames = verify_episode(info, ep)
            run = info['base'] / 'rgbtrack/runs' / args.run_tag / 'episodes' / f'e{ep:05d}'
            if not completed(info, ep, run, mask, frames, calibration):
                raise RuntimeError(f'e{ep}: no completed tracking in {args.run_tag}')
            review_one(info, ep, run, args.run_tag, calibration, args.viz_stride)
        return
    run_batch(args, info, episodes)

if __name__ == '__main__':
    main()
