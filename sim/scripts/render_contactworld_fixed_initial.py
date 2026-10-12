#!/usr/bin/env python3
"""Continuous test/reference comparison with a fixed initial-state neighbor.

Training videos use frame-index timing at 15 displayed steps/s. Historical
collection timing is unknown. Alignment never warps or interpolates trajectories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003')


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)['streams'][0]


def make_reference(task):
    dataset, output, entry = task
    ep, source, count = entry['episode_index'], entry['source_episode'], entry['length']
    target = output/f'source_{source:03d}.mp4'
    if not target.exists():
        caps = [cv2.VideoCapture(str(dataset/f'videos/chunk-000/observation.images.{view}/episode_{ep:06d}.mp4'))
                for view in ('front', 'wrist')]
        tmp = target.with_suffix('.tmp.mp4')
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '512x256', '-r', '15', '-i', '-',
            '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '1',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
        complete = False
        try:
            for f in range(count):
                pairs = [cap.read() for cap in caps]
                if not all(ok and im.shape == (256, 256, 3) for ok, im in pairs):
                    raise ValueError(f'Train source {source}: missing or invalid frame {f}')
                image = np.hstack([im for _, im in pairs])
                encoder.stdin.write(image.tobytes())
                if f == 0:
                    cv2.imwrite(str(target.with_suffix('.jpg')), image)
            if any(cap.read()[0] for cap in caps):
                raise ValueError(f'Train source {source}: unexpected extra frames')
            complete = True
        finally:
            for cap in caps:
                cap.release()
            encoder.stdin.close()
            code = encoder.wait()
        if code or not complete:
            raise RuntimeError(f'Reference rendering failed: {target}')
        tmp.replace(target)
    stream = probe(target)
    assert int(stream['nb_frames']) == count, target
    assert (stream['width'], stream['height']) == (512, 256), target
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p', target
    assert stream['r_frame_rate'] == '15/1', target
    return source


def contact_frame(rows, key):
    hits = np.flatnonzero(np.linalg.norm(np.asarray([r[key] for r in rows]), axis=1) > .1)
    return int(hits[0])+1 if len(hits) else None


def phase_trace(forces, xy_mm, z_mm, axis_deg, successes):
    """Observational labels for displayed frames; never policy inputs."""
    assert len(forces) == len(xy_mm) == len(z_mm) == len(axis_deg) == len(successes)
    trace = []
    ever_contact = False
    contact_start = 0
    first_contact_z = None
    last_contact = False
    for frame, (force, xy, z, axis, success) in enumerate(zip(forces, xy_mm, z_mm, axis_deg, successes)):
        contact = force > .1
        if contact and not last_contact:
            contact_start = frame
        if contact and first_contact_z is None:
            first_contact_z = z
        ever_contact |= contact
        z_drop = z_mm[max(contact_start, frame-3)] - z
        if success:
            phase = 5  # Environment's pose/keypoint success condition.
        elif not contact:
            phase = 1 if ever_contact else 0
        elif xy <= 2. and axis <= 5. and abs(z) <= 5.:
            phase = 7  # Near target height, regardless of current vertical speed.
        elif xy > 2. or axis > 5.:
            phase = 2  # Contact while outside a deliberately generous near-axis range.
        elif z_drop >= .5:
            phase = 4  # Near axis with measured axial descent over <= 3 frames.
        elif first_contact_z is not None and first_contact_z-z >= 3.:
            phase = 6  # Has descended since first socket contact, now without recent descent.
        else:
            phase = 3  # Near axis, before measurable descent.
        trace.append([phase, round(float(force), 2), round(float(xy), 1),
                      round(float(z), 1), round(float(axis), 1)])
        last_contact = contact
    return trace


def train_trace(folder, replay, count):
    with np.load(folder/'v2_capture.npz') as capture:
        plug, socket = capture['plug_pose'].copy(), capture['socket_pose'].copy()
    assert len(plug) == len(socket) == count == len(replay['rows'])
    delta = (plug[:, :3] - socket[:, :3])*1000
    plug_axis = Rotation.from_quat(plug[:, 3:]).apply([0, 0, 1])
    socket_axis = Rotation.from_quat(socket[:, 3:]).apply([0, 0, 1])
    axis = np.degrees(np.arccos(np.clip(np.abs(np.sum(plug_axis*socket_axis, axis=1)), 0, 1)))
    rows = replay['rows']
    # Capture frame f is a pre-action observation. Replay row f-1 describes
    # the action that led to it. Row count-1 has no displayed next frame.
    forces = [0.] + [float(np.linalg.norm(r['plug_socket_force_post'])) for r in rows[:-1]]
    successes = [False] + [bool(r['source_success']) for r in rows[:-1]]
    return phase_trace(forces, np.linalg.norm(delta[:, :2], axis=1), delta[:, 2], axis, successes)


def test_trace(record):
    frames = [record['result']['initial']] + record['rows']
    return phase_trace([float(np.linalg.norm(r['plug_socket_contact_force_xyz'])) for r in frames],
                       [r['plug_socket_xy_error_mm'] for r in frames],
                       [r['plug_socket_z_mm'] for r in frames],
                       [r['unsigned_axis_error_deg'] for r in frames],
                       [r['source_success'] for r in frames])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--dataset', type=Path, default=Path('data/contactworld_usb_v3_bilateral'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    args.eval_root, args.dataset = args.eval_root.resolve(), args.dataset.resolve()
    args.output = (args.output or args.eval_root/'all_sensor_comparison').resolve()
    assets = args.output/'fixed_initial/training'
    assets.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((args.dataset/'spatial/manifest.json').read_text())
    entries = {int(e['source_episode']): e for e in dataset_manifest['episodes'] if e['split'] == 'train'}
    nearest = json.loads((args.eval_root/'training_neighbors/analysis.json').read_text())['neighbors']
    assert len(nearest) == 40 and len({n['seed'] for n in nearest}) == 40
    selected = {int(n['nearest_source_episode']) for n in nearest}
    for n in nearest:
        assert entries[n['nearest_source_episode']]['episode_index'] == n['nearest_episode_index']
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(make_reference, [(args.dataset, assets, entries[s]) for s in sorted(selected)]))
    summaries = sorted(args.eval_root.glob('pi0_cw3_usb_*/summary.json'))
    assert len(summaries) == 13
    models = {s.parent.name: json.loads(s.read_text()) for s in summaries}
    data = []
    for n in sorted(nearest, key=lambda x: x['seed']):
        seed, source = n['seed'], n['nearest_source_episode']
        entry = entries[source]
        replay_path = Path(dataset_manifest['capture_root'])/f'insertion_usb_episode_{source:03d}/replay.json'
        replay = json.loads(replay_path.read_text())
        trace = train_trace(replay_path.parent, replay, entry['length'])
        train_contact = contact_frame(replay['rows'], 'plug_socket_force_post')
        if train_contact is not None and train_contact >= entry['length']:
            train_contact = None  # No saved image at this contact anchor.
        case = {'seed': seed, 'nearest': n,
                'train': {'source': source, 'episode': entry['episode_index'],
                          'frames': entry['length'], 'fps': 15, 'contact': train_contact,
                          'trace': trace,
                          'video': os.path.relpath(assets/f'source_{source:03d}.mp4', args.output),
                          'poster': os.path.relpath(assets/f'source_{source:03d}.jpg', args.output)}, 'models': []}
        hashes = set()
        for config, summary in models.items():
            result = next(r for r in summary['results'] if r['seed'] == seed)
            folder = args.eval_root/config/f'seed_{seed}'
            trajectory = json.loads((folder/'trajectory.json').read_text())
            model_trace = test_trace(trajectory)
            assert len(model_trace) == result['steps']+1
            hashes.add(result['initial_state_sha256'])
            video = folder/'front_wrist.mp4'
            assert video.is_file()
            case['models'].append({'number': int(config.split('_usb_')[1][:2]), 'config': config,
                'frames': result['steps']+1, 'fps': summary['video_fps'],
                'trace': model_trace,
                'contact': contact_frame(trajectory['rows'], 'plug_socket_contact_force_xyz'),
                'ever': result['source_ever'], 'final': result['source_final'],
                'first_success': result['first_success_step'],
                'video': os.path.relpath(video, args.output),
                'sensors': f'dynamic_nearest.html#seed{seed}'})
        assert len(hashes) == 1
        data.append(case)
    assert sum(len(c['models']) for c in data) == 520
    manifest = {'format': 'cw-fixed-initial-continuous-v1', 'cases': data,
        'retrieval': 'Existing train-only full initial geometry nearest neighbor: target, relative plug pose, grasp pose. One fixed reference per seed.',
        'time': 'Frame/control-step alignment only. Historical training collection FPS unknown. No trajectory warping.',
        'contact': 'First plug/socket contact-force norm > 0.1, action-post index + 1; absent contacts cannot align.',
        'phases': 'Diagnostic ground truth only: 0 no socket contact yet; 1 separated after contact; 2 plug/socket contact outside XY<=2 mm and axis<=5 deg; 3 near axis before 3 mm cumulative descent; 4 near axis with >=0.5 mm descent during the preceding three frames of the current contact run; 5 source keypoint success; 6 >=3 mm descent since first socket contact but no recent descent; 7 aligned and within 5 mm of target height but source criterion false. Sensor TacFF grip contact is unrelated to socket contact.',
        'verified': {'cases': 40, 'models_per_case': 13, 'test_videos': 520,
                     'unique_train_videos_encoded_and_probed': len(sources), 'paired_initial_hashes': True}}
    manifest_path = args.output/'fixed_initial_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    index, archive = args.output/'index.html', args.output/'dynamic_nearest.html'
    if index.exists() and not archive.exists():
        shutil.copy2(index, archive)
    template = Path(__file__).with_name('contactworld_fixed_initial.html').read_text()
    page = template.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    temp = args.output/'fixed_initial.tmp.html'
    temp.write_text(page)
    temp.replace(index)
    print(json.dumps(manifest['verified'], ensure_ascii=False))
    print(index)


if __name__ == '__main__':
    main()
