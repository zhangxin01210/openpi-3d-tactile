#!/usr/bin/env python3
"""Build synchronized sensor/nearest-demonstration movies for all saved USB trials.

Only recorded pre-action policy snapshots are rendered. No simulator reruns or
interpolated point clouds. Training next-frame images are display-only and never
enter retrieval. Every training candidate must belong to the train split.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import html
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist

VERSION = 'cw-40x13-review-v1'
WIDTH, HEIGHT = 1536, 1080
ROOT = Path('/home/sai/zx/openpi-sim-runtime/runs')
DEFAULT_EVAL = ROOT / 'contactworld_pi0_v3_13models_40trials_20261003'
DEFAULT_NEIGHBOR = ROOT / 'contactworld_neighbor_night_20261003'


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def label(im, s, xy, size=.48, color=(225, 232, 240)):
    cv2.putText(im, str(s), xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def decode_video(path):
    cap = cv2.VideoCapture(str(path))
    images = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not images:
        raise ValueError(f'Cannot decode {path}')
    return np.stack(images)


def cloud_basis():
    az, el = np.deg2rad([48, 35])
    return np.array([[-np.sin(az), np.cos(az), 0],
                     [-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)]])


def cloud_panel(xyz, taxel, bounds, width=512, height=252):
    """Fixed metric orthographic camera, all 4096 points, no per-frame recentering."""
    im = np.full((height, width, 3), (15, 23, 34), np.uint8)
    basis = cloud_basis()
    low, high = np.asarray(bounds)
    factor = min((width-30)/(high[0]-low[0]), (height-32)/(high[1]-low[1]))
    center = (low+high)/2

    def project(p):
        uv = (np.asarray(p) @ basis.T-center)*factor
        return np.rint(np.c_[uv[:, 0]+width/2, height/2-uv[:, 1]]).astype(int)

    # Colour is height in base, not RGB supplied to the policy.
    level = np.rint(np.clip(np.asarray(xyz)[:, 2]/.35, 0, 1)*255).astype(np.uint8)
    colors = cv2.applyColorMap(level[:, None], cv2.COLORMAP_VIRIDIS)[:, 0, ::-1]
    order = np.argsort(np.asarray(xyz)[:, 2])
    uv, colors = project(xyz)[order], colors[order]
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = uv[:, 0]+dx, uv[:, 1]+dy
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        im[yy[valid], xx[valid]] = colors[valid]
    for i, color in enumerate(((242, 95, 218), (255, 160, 50))):
        for x, y in project(np.asarray(taxel)[i]):
            cv2.circle(im, (int(x), int(y)), 1, color, -1)
    axes = project(np.array([[0, 0, 0], [.12, 0, 0], [0, .12, 0], [0, 0, .12]]))
    for j, axis in enumerate('XYZ', 1):
        cv2.line(im, tuple(axes[0]), tuple(axes[j]), (190, 200, 212), 1)
        label(im, axis, tuple(axes[j]), .4)
    label(im, '4096 XYZ / fixed base view / colour = height', (10, 18), .40)
    label(im, 'pad sites: left pink / right orange', (10, height-8), .40)
    return im


def force_maps(panel, field, limits, y, title, channels):
    field = np.asarray(field).reshape(2, 10, 14, 3)
    label(panel, title, (9, y), .46)
    for side in range(2):
        for axis in range(3):
            x, top = axis*170+7, y+8+side*94
            values = field[side, :, :, axis]
            v = np.clip(values/limits[axis], -1, 1)
            red = np.where(v >= 0, 245, 245+190*v)
            blue = np.where(v <= 0, 245, 245-190*v)
            green = 245-160*np.abs(v)
            rgb = np.stack([red, green, blue], -1).astype(np.uint8)
            panel[top:top+65, x:x+156] = cv2.resize(rgb, (156, 65), interpolation=cv2.INTER_NEAREST)
            name = ('L ' if side == 0 else 'R ')+channels[axis]
            clip = np.mean(np.abs(values) > limits[axis])
            label(panel, f'{name} max {np.abs(values).max():.1e}', (x, top+80), .36)
            if clip:
                label(panel, f'clip {clip:.0%}', (x+95, top+12), .30, (10, 10, 10))


def sensor_panel(front, wrist, base, local, cloud, taxel, action, scales):
    im = np.full((1000, 512, 3), (22, 30, 42), np.uint8)
    im[:256, :256] = front
    im[:256, 256:] = wrist
    label(im, 'FRONT', (8, 20), .52, (255, 255, 255))
    label(im, 'WRIST', (265, 20), .52, (255, 255, 255))
    label(im, 'Action dXYZ mm: '+', '.join(f'{v:+.2f}' for v in np.asarray(action)[:3]*10), (9, 279), .45)
    force_maps(im, base, scales['base'], 307, 'BASE signed forces: blue (-), white (0), red (+)', ['Fx', 'Fy', 'Fz'])
    force_maps(im, local, scales['local'], 526, 'SENSOR LOCAL channels (not base Fx/Fy/Fz)', ['C0', 'C1', 'C2'])
    im[748:1000] = cloud_panel(cloud, taxel, scales['cloud_projection_bounds'])
    return im


def model_mode(number):
    return 'R' if number == 1 else 'RP' if number == 2 else 'RT' if number == 3 else 'RPT'


def input_description(number):
    if number in (1, 2):
        tactile = 'T: not input (shown for diagnosis)'
    elif number in (3, 4, 5, 13):
        tactile = 'T: local summary' if number == 5 else 'T: local map'
    else:
        tactile = 'T: base forces'
    return ('P: input | ' if number not in (1, 3) else 'P: not input | ')+tactile


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset/'spatial/manifest.json').read_text())
    entries = {int(e['source_episode']): e for e in manifest['episodes'] if e['split'] == 'train'}
    with np.load(args.neighbor/'features/train_features.npz') as z:
        train = {k: z[k] for k in ('rgb', 'source', 'episode', 'frame', 'action')}
    assert set(train['source']).issubset(entries)
    mean, std = train['rgb'].mean(0), train['rgb'].std(0)
    std = np.maximum(std, max(np.sqrt(np.mean(std**2))*.1, 1e-5))
    reference = ((train['rgb']-mean)/std).astype(np.float32)
    jobs, requested, samples = [], {}, {'base': [], 'local': []}
    cloud_low, cloud_high = np.full(2, np.inf), np.full(2, -np.inf)
    summaries = sorted(args.eval_root.glob('pi0_cw3_usb_*/summary.json'))
    if len(summaries) != 13:
        raise ValueError(f'Expected 13 models, found {len(summaries)}')
    seed_sets = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        config = summary_path.parent.name
        number = int(config.split('_usb_')[1][:2])
        seed_sets.append({int(r['seed']) for r in summary['results']})
        for result in summary['results']:
            seed = int(result['seed'])
            source_dir = summary_path.parent/f'seed_{seed}'
            with np.load(source_dir/'inputs.npz') as z:
                steps = z['steps']
                assert np.all(np.diff(steps) == summary['replan_every'])
                projected = z['cloud_xyz'].reshape(-1, 3) @ cloud_basis().T
                cloud_low = np.minimum(cloud_low, projected.min(0))
                cloud_high = np.maximum(cloud_high, projected.max(0))
                for key in samples:
                    values = z['force_'+key].reshape(-1, 3)
                    samples[key].append(np.abs(values[::13]))
            with np.load(args.neighbor/f'eval_features/{config}/seed_{seed}.npz') as z:
                query = (z['rgb']-mean)/std
                assert len(query) == len(steps)
            # Visual-only: both cameras, train-only frozen feature scaling.
            distance = cdist(query, reference, 'sqeuclidean')/reference.shape[1]
            ix = distance.argmin(1)
            nearest = {'visual': {'indices': ix, 'distance': np.sqrt(distance[np.arange(len(ix)), ix])}}
            mode = model_mode(number)
            with np.load(args.neighbor/f'rollout_predictions/{config}/seed_{seed}_{mode}.npz') as z:
                assert np.array_equal(z['step'], steps)
                assert np.array_equal(train['source'][z['neighbors'][:, 0]], z['neighbor_sources'][:, 0])
                nearest['modal'] = {'indices': z['neighbors'][:, 0], 'distance': np.sqrt(z['distances'][:, 0])}
            match = {}
            for kind, info in nearest.items():
                match[kind] = []
                for i, d in zip(info['indices'], info['distance']):
                    src, ep, frame = (int(train[k][i]) for k in ('source', 'episode', 'frame'))
                    assert entries[src]['episode_index'] == ep and frame+1 < entries[src]['length']
                    match[kind].append({'source': src, 'episode': ep, 'frame': frame, 'distance': float(d)})
                    for f in (frame, frame+1):
                        requested[(src, f)] = (ep, f)
            jobs.append({'config': config, 'number': number, 'seed': seed, 'steps': steps.tolist(),
                         'dt': float(summary['action_dt_s']), 'stride': int(summary['replan_every']),
                         'input': input_description(number), 'mode': mode, 'matches': match,
                         'ever': result['source_ever'], 'final': result['source_final'],
                         'first_success': result['first_success_step'], 'result': result})
        print(f'Index prepared {number:02d}/13', flush=True)
    if len(seed_sets[0]) != 40 or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError('Models do not cover the same 40 seeds')
    # Same signed scales across every movie, both fingers and all training frames.
    for src, e in entries.items():
        folder = args.dataset/f"spatial/episodes/episode_{e['episode_index']:06d}"
        projected = np.load(folder/'pointcloud_fused_xyz.npy', mmap_mode='r').reshape(-1, 3) @ cloud_basis().T
        cloud_low = np.minimum(cloud_low, projected.min(0))
        cloud_high = np.maximum(cloud_high, projected.max(0))
        for coord in samples:
            names = ([f'tactile_force_{s}_base' for s in ('left', 'right')] if coord == 'base'
                     else [f'force_grid_{s}_local' for s in ('left', 'right')])
            for name in names:
                v = np.load(folder/(name+'.npy'), mmap_mode='r').reshape(-1, 3)
                samples[coord].append(np.abs(v[::13]))
    scales = {k: np.maximum(np.quantile(np.concatenate(v), .995, axis=0), 1e-7).tolist() for k, v in samples.items()}
    pad = (cloud_high-cloud_low)*.02
    scales['cloud_projection_bounds'] = [(cloud_low-pad).tolist(), (cloud_high+pad).tolist()]
    scales['note'] = 'Global sampled absolute P99.5, symmetric signed limits; synthetic TacFF units, not calibrated Newtons.'
    # Training actions: current and next frame already available in four-step chunks.
    actions = {}
    for src, f, a in zip(train['source'], train['frame'], train['action']):
        for d in (0, 1):
            actions[(int(src), int(f)+d)] = a[d].tolist()
    plan = {'version': VERSION, 'eval_root': str(args.eval_root), 'neighbor': str(args.neighbor),
            'dataset': str(args.dataset), 'jobs': jobs, 'scales': scales,
            'retrieval': {'visual': 'Double-camera frozen ResNet18 features only; z-scored on 95 train demos; squared Euclidean.',
                          'modal': 'Existing R/RP/RT/RPT nearest frame using fixed diagnostic features, not policy encoder embeddings.'}}
    existing = out/'manifest.json'
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError('Output protocol changed; use a new output directory')
    write_json(existing, plan)
    return plan, entries, requested, actions


def prepare_panels(args, plan, entries, requested, actions):
    cache = args.output/'training_panels'
    cache.mkdir(exist_ok=True)
    for j, (source, entry) in enumerate(sorted(entries.items())):
        frames = sorted(f for s, f in requested if s == source)
        if all((cache/f's{source}_f{f}.jpg').is_file() for f in frames):
            continue
        ep = entry['episode_index']
        front, wrist = [decode_video(args.dataset/f'videos/chunk-000/observation.images.{v}/episode_{ep:06d}.mp4') for v in ('front', 'wrist')]
        if len(front) != entry['length'] or len(wrist) != entry['length']:
            raise ValueError(f'Training video length differs for {source}')
        folder = args.dataset/f'spatial/episodes/episode_{ep:06d}'
        load = lambda key: np.load(folder/(key+'.npy'), mmap_mode='r')
        cloud = load('pointcloud_fused_xyz')
        xyz = [load(f'tactile_xyz_{s}_base') for s in ('left', 'right')]
        base = [load(f'tactile_force_{s}_base') for s in ('left', 'right')]
        local = [load(f'force_grid_{s}_local') for s in ('left', 'right')]
        for f in frames:
            path = cache/f's{source}_f{f}.jpg'
            if path.exists():
                continue
            im = sensor_panel(front[f], wrist[f], np.stack([v[f] for v in base]),
                              np.stack([v[f] for v in local]), cloud[f], np.stack([v[f] for v in xyz]),
                              actions[(source, f)], plan['scales'])
            if not cv2.imwrite(str(path), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(path)
        print(f'Train panels {j+1}/{len(entries)} source={source} frames={len(frames)}', flush=True)


@lru_cache(maxsize=96)
def read_panel(path):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def render_job(payload):
    cv2.setNumThreads(1)
    out, eval_root, job, scales = payload
    out, eval_root = Path(out), Path(eval_root)
    folder = out/f"seed_{job['seed']}"/job['config']
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'complete.json'
    if marker.exists():
        return job['seed'], job['number'], 'cached'
    snap_path = eval_root/job['config']/f"seed_{job['seed']}"/'inputs.npz'
    with np.load(snap_path) as z:
        snap = {k: z[k] for k in ('front', 'wrist', 'force_base', 'force_local', 'cloud_xyz', 'taxel_xyz_base', 'action_chunks', 'steps')}
    trajectory = json.loads((snap_path.parent/'trajectory.json').read_text())
    panels = [sensor_panel(snap['front'][i], snap['wrist'][i], snap['force_base'][i], snap['force_local'][i],
                           snap['cloud_xyz'][i], snap['taxel_xyz_base'][i],
                           trajectory['rows'][int(snap['steps'][i])]['action_sim_clipped'], scales)
              for i in range(len(snap['steps']))]
    interval = job['dt']*job['stride']
    result = 'FINAL SUCCESS' if job['final'] else 'EVER ONLY' if job['ever'] else 'FAIL'
    for kind in ('visual', 'modal'):
        target = folder/f'{kind}.mp4'
        temporary = folder/f'{kind}.tmp.mp4'
        if target.exists():
            continue
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', f'{1/interval:.10f}',
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21', '-threads', '2',
            '-r', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        try:
            for i, panel in enumerate(panels):
                match = job['matches'][kind][i]
                src, f, ep = match['source'], match['frame'], match['episode']
                im = np.full((HEIGHT, WIDTH, 3), (12, 18, 28), np.uint8)
                im[80:, :512] = panel
                for column, frame in ((1, f), (2, f+1)):
                    im[80:, column*512:(column+1)*512] = read_panel(str(out/f'training_panels/s{src}_f{frame}.jpg'))
                step = job['steps'][i]
                label(im, f"TEST {job['seed']} M{job['number']:02d} | {result}", (10, 22), .58)
                label(im, f'step {step:03d}  t={step*job["dt"]:.3f}s  snapshot {i+1}/{len(panels)}', (10, 44), .47)
                label(im, job['input'], (10, 66), .40)
                label(im, f'TRAIN nearest | source {src} / episode {ep}', (524, 22), .53)
                label(im, f'frame {f} | '+('VISUAL ONLY' if kind == 'visual' else f"{job['mode']} diagnostic features"), (524, 44), .46)
                changed = i > 0 and job['matches'][kind][i-1]['source'] != src
                label(im, 'SOURCE SWITCH' if changed else 'Re-retrieved at each test observation', (524, 66), .43, (255, 192, 91))
                label(im, f'TRAIN next actual frame | source {src}', (1036, 22), .53)
                label(im, f'frame {f+1} | display-only lookahead', (1036, 44), .46)
                label(im, 'Montage: NOT one continuous demonstration', (1036, 66), .42, (255, 192, 91))
                if i == 0:
                    cv2.imwrite(str(folder/f'{kind}.jpg'), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
                encoder.stdin.write(im.tobytes())
        finally:
            encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError(f'ffmpeg failed for {target}')
        temporary.replace(target)
    write_json(folder/'matches.json', job)
    write_json(marker, {'version': VERSION, 'snapshots': len(panels), 'interval_s': interval,
                        'duration_s': len(panels)*interval, 'width': WIDTH, 'height': HEIGHT})
    return job['seed'], job['number'], 'rendered'


def write_page(args, plan):
    """No fetch/CDN: opens directly as file://, videos attach only near viewport."""
    phase_manifest = args.output/'fixed_initial_manifest.json'
    phase_by_trial = {}
    if phase_manifest.exists():
        for case in json.loads(phase_manifest.read_text())['cases']:
            for model in case['models']:
                phase_by_trial[(case['seed'], model['config'])] = model['trace']
    rows = []
    for job in plan['jobs']:
        row = {k: job[k] for k in ('seed', 'config', 'number', 'mode', 'input', 'ever', 'final', 'first_success', 'dt', 'stride', 'steps', 'matches')}
        if phase_by_trial:
            trace = phase_by_trial[(job['seed'], job['config'])]
            row['stage'] = [trace[frame] for frame in row['steps']]
        with np.load(args.neighbor/f"eval_features/{job['config']}/seed_{job['seed']}.npz") as z:
            row['phase'] = z['phase'].tolist()
        base = f"seed_{job['seed']}/{job['config']}"
        row.update(visual=base+'/visual.mp4', modal=base+'/modal.mp4', provenance=base+'/matches.json',
                   original=os.path.relpath(args.eval_root/job['config']/f"seed_{job['seed']}"/'front_wrist.mp4', args.output),
                   ready=(args.output/base/'complete.json').exists())
        rows.append(row)
    template = Path(__file__).with_name('contactworld_compare_all.html').read_text()
    body = template.replace('__DATA__', json.dumps(rows, ensure_ascii=False).replace('</', '<\\/'))
    body = body.replace('__SCALES__', html.escape(json.dumps(plan['scales'], ensure_ascii=False)))
    page_name = 'dynamic_nearest.html' if (args.output/'fixed_initial_manifest.json').exists() else 'index.html'
    (args.output/page_name).write_text(body)


def verify_output(args):
    plan = json.loads((args.output/'manifest.json').read_text())
    dataset = Path(plan['dataset'])
    manifest = json.loads((dataset/'spatial/manifest.json').read_text())
    entries = {int(e['source_episode']): e for e in manifest['episodes'] if e['split'] == 'train'}
    assert len(plan['jobs']) == 520
    by_seed = {}
    files, pairs = [], 0
    for job in plan['jobs']:
        by_seed.setdefault(job['seed'], []).append(job)
        folder = args.output/f"seed_{job['seed']}"/job['config']
        marker = json.loads((folder/'complete.json').read_text())
        assert marker['snapshots'] == len(job['steps']) == 45
        assert json.loads((folder/'matches.json').read_text()) == job
        for kind in ('visual', 'modal'):
            assert len(job['matches'][kind]) == len(job['steps'])
            for match in job['matches'][kind]:
                entry = entries[match['source']]
                assert entry['episode_index'] == match['episode']
                assert 0 <= match['frame'] < entry['length']-15
                assert match['frame']+1 < entry['length']
                assert np.isfinite(match['distance']) and match['distance'] >= 0
                pairs += 1
            assert (folder/f'{kind}.jpg').is_file()
            files.append((folder/f'{kind}.mp4', marker['duration_s']))
    assert len(by_seed) == 40
    for jobs in by_seed.values():
        assert sorted(j['number'] for j in jobs) == list(range(1, 14))
        assert len({j['result']['initial_state_sha256'] for j in jobs}) == 1

    def probe(item):
        path, duration = item
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,duration,r_frame_rate',
            '-of', 'json', str(path)], check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)['streams'][0]
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p', path
        assert (stream['width'], stream['height']) == (WIDTH, HEIGHT), path
        assert abs(float(stream['duration'])-duration) < 1/15+.001, path
        assert abs(int(stream['nb_frames'])-round(duration*15)) <= 1, path
        return str(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(probe, files))
    # Independently recompute visual-only matching on beginning/middle/end seeds.
    neighbor = Path(plan['neighbor'])
    with np.load(neighbor/'features/train_features.npz') as z:
        train = {k: z[k] for k in ('rgb', 'source', 'frame')}
    mean, std = train['rgb'].mean(0), train['rgb'].std(0)
    std = np.maximum(std, max(np.sqrt(np.mean(std**2))*.1, 1e-5))
    reference = (train['rgb']-mean)/std
    checked_queries, decoded = 0, 0
    for job in plan['jobs']:
        if job['seed'] not in (10000, 10020, 10039):
            continue
        with np.load(neighbor/f"eval_features/{job['config']}/seed_{job['seed']}.npz") as z:
            for t in (0, 22, 44):
                ix = np.square(((z['rgb'][t]-mean)/std)-reference).sum(1).argmin()
                match = job['matches']['visual'][t]
                assert (int(train['source'][ix]), int(train['frame'][ix])) == (match['source'], match['frame'])
                checked_queries += 1
        if job['number'] in (1, 4, 13):
            for kind in ('visual', 'modal'):
                path = args.output/f"seed_{job['seed']}"/job['config']/f'{kind}.mp4'
                cap = cv2.VideoCapture(str(path))
                assert cap.read()[0], path
                cap.set(cv2.CAP_PROP_POS_FRAMES, 179)
                ok, frame = cap.read()
                assert ok and frame.shape == (HEIGHT, WIDTH, 3), path
                cap.release()
                decoded += 1
    report = {'status': 'pass', 'cases': 40, 'models': 13, 'trials': 520,
              'verified_videos': len(verified), 'verified_training_frame_pairs': pairs,
              'independent_visual_nearest_checks': checked_queries,
              'sample_videos_decoded_first_and_last': decoded,
              'paired_initial_state_hashes': True,
              'real_browser_test': 'not covered by this data/video verifier'}
    write_json(args.output/'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root', type=Path, default=DEFAULT_EVAL)
    p.add_argument('--neighbor', type=Path, default=DEFAULT_NEIGHBOR)
    p.add_argument('--dataset', type=Path, default=Path('data/contactworld_usb_v3_bilateral'))
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='Render only this many trials for visual QA; index still covers all 520')
    p.add_argument('--page-only', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    args.eval_root, args.neighbor, args.dataset = [v.resolve() for v in (args.eval_root, args.neighbor, args.dataset)]
    args.output = (args.output or args.eval_root/'all_sensor_comparison').resolve()
    if args.verify_only:
        verify_output(args)
        return
    if args.page_only:
        write_page(args, json.loads((args.output/'manifest.json').read_text()))
        return
    cv2.setNumThreads(1)
    plan, entries, requested, actions = prepare(args)
    # Pilot renders use only their required train panels.
    jobs = sorted(plan['jobs'], key=lambda j: (j['seed'], j['number']))
    if args.limit:
        jobs = jobs[:args.limit]
        keys = {(m['source'], m['frame']+d) for j in jobs for mm in j['matches'].values() for m in mm for d in (0, 1)}
        requested = {k: v for k, v in requested.items() if k in keys}
    prepare_panels(args, plan, entries, requested, actions)
    write_page(args, plan)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(render_job, (str(args.output), str(args.eval_root), j, plan['scales'])) for j in jobs]
        for n, future in enumerate(as_completed(pending), 1):
            seed, number, state = future.result()
            print(f'[{n}/{len(jobs)}] seed={seed} M{number:02d} {state} elapsed={time.monotonic()-started:.1f}s', flush=True)
            if n % 26 == 0:
                write_page(args, plan)
    write_page(args, plan)
    print(f'COMPLETE {args.output / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
