#!/usr/bin/env python3
"""V2 full-episode visual review: random held-out episodes, EVERY frame, offline one-file HTML.

Read-only: uses existing V2 checkpoint/cache and camera projection. Produces H.264
MP4 per episode, plus index_standalone.html with embedded video bytes (no broken
relative paths). Each video has two rows (manipulated/target), with RGB | GT | PRED.

Usage (from openpi-3d-tactile repository root, with companion scripts present):
  PYTHONPATH="$PWD:$PWD/src" .venv/bin/python -u v2_full_episode_review.py \\
    --checkpoint-json output/cad_pose_rgbd_v2/train_20261109_162628/best.json \\
    --split test --episodes-per-task 3 --seed 20261109 --device cuda:0 \\
    --output-dir output/cad_pose_rgbd_v2/full_episode_review

GT-relative error is NOT necessarily true physical error, particularly for Peg.
No CSV/GT is altered; video outputs are for manual review, NOT for π0 deployment.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'src'))

import cad_pose_rgbd_single_v2 as v2
import v2_geometry_bridge_pilot as bridge
import visualize_cad_pose_rgbd_v2 as viz

ROLES = ('manipulated', 'target')


def args_parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', type=Path, default=HERE)
    p.add_argument('--cache-dir', type=Path, default=HERE / 'output/cad_pose_rgbd_v2/cache')
    p.add_argument('--checkpoint-json', type=Path, default=HERE / 'output/cad_pose_rgbd_v2/train_20261109_162628/best.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    p.add_argument('--episodes-per-task', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--episodes', default='', help='Optional overrides: 1018_peg_all=23,65;1028_press_button=4,7')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--fps', type=int, default=12, help='Review playback speed (not raw acquisition frame rate)')
    p.add_argument('--crf', type=int, default=27, help='H264 video quality: 18 high, 27 default, 32 smaller')
    p.add_argument('--peg-intrinsics', choices=('legacy', 'current'), default='legacy')
    p.add_argument('--ref-width', type=int, default=640)
    p.add_argument('--ref-height', type=int, default=480)
    p.add_argument('--self-test', action='store_true', help='Test video + offline HTML packaging with synthetic frames')
    a = p.parse_args()
    if not 1 <= a.episodes_per_task <= 12:
        p.error('--episodes-per-task must be 1..12')
    if not 1 <= a.batch_size <= 256:
        p.error('--batch-size must be 1..256')
    if not 1 <= a.fps <= 60:
        p.error('--fps must be 1..60')
    if not 18 <= a.crf <= 36:
        p.error('--crf must be 18..36')
    a.repo = a.repo.resolve()
    a.cache_dir = a.cache_dir.resolve()
    a.output_dir = a.output_dir.resolve()
    return a


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def select_random_eps(cache_dir, split, count, seed, overrides):
    split_path = cache_dir / 'episode_splits.json'
    splits = json.loads(split_path.read_text())
    rng = np.random.default_rng(seed)
    selected = {}
    manual = {}
    if overrides:
        for item in overrides.split(';'):
            if not item.strip():
                continue
            name, eq, rest = item.partition('=')
            if not eq or name not in v2.TASK_NAMES:
                raise ValueError(f'Invalid --episodes {item!r}')
            manual[name] = list(dict.fromkeys(int(s) for s in rest.split(',') if s.strip()))
    for task in v2.TASK_NAMES:
        episodes = list(map(int, splits[task][split]))
        available = [e for e in episodes if all((cache_dir / task / f'e{e:05d}' / n).is_file()
                     for n in ('meta.json', 'rgb.npy', 'depth.npy', 'transform.npy', 'valid.npy'))]
        if task in manual:
            wrong = sorted(set(manual[task]) - set(available))
            if wrong:
                raise ValueError(f'{task}: episodes {wrong} missing cache or not in {split} split')
            chosen = manual[task]
        else:
            if len(available) < count:
                raise ValueError(f'{task}: only {len(available)} cached {split} episodes; requested {count}')
            chosen = list(map(int, rng.choice(available, size=count, replace=False)))
        if not chosen:
            raise ValueError(f'No episodes selected for {task}')
        selected[task] = chosen
    return selected


def open_h264_writer(path, width, height, fps, crf):
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('ffmpeg missing. Install ffmpeg or use a cloud environment with ffmpeg/H264.')
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-y',
           '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-g', str(fps), '-keyint_min', str(fps),
           '-movflags', '+faststart', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def finish_video(proc, dest, n_expected):
    if proc.stdin:
        proc.stdin.close()
    error = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr else ''
    ret = proc.wait()
    if ret != 0 or not dest.is_file() or dest.stat().st_size < 2048:
        raise RuntimeError(f'ffmpeg video encode failed rc={ret} for {dest}: {error[-1000:]}')
    probe = shutil.which('ffprobe')
    if probe:
        cmd = [probe, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=codec_name,nb_frames', '-of', 'json', str(dest)]
        pr = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream = json.loads(pr.stdout)['streams'][0]
        if stream.get('codec_name') != 'h264':
            raise RuntimeError(f'Video codec is not H264: {stream}')
        nf = stream.get('nb_frames')
        if nf and nf != 'N/A' and int(nf) != n_expected:
            raise RuntimeError(f'Video frame count {nf} != original episode {n_expected}: {dest}')
    return dest.stat().st_size


def labelled_panel(rgb, uv, title, color, *, invalid=False):
    if invalid:
        out = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(out, (0, 0), (out.shape[1]-1, 25), (23, 32, 46), -1)
        cv2.putText(out, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(out, 'GT LABEL INVALID', (5, out.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (50, 70, 245), 1, cv2.LINE_AA)
        return out
    out, _ = viz.draw_rgb_overlay(rgb, uv, title=title, color=color)
    return out


def video_frame(rgb, gtT, predT, valid, cad, cam, idx, object_names, ref_width, ref_height):
    h, w = rgb.shape[:2]
    raw = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    rows = []
    for role in range(2):
        role_name = ROLES[role]
        name = object_names[role]
        original = raw.copy()
        cv2.putText(original, f'RGB f{idx:04d} {name}', (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .44, (250, 250, 250), 1, cv2.LINE_AA)
        if valid[role]:
            g = viz.transformed(cad[role], gtT[role, :3, 3], gtT[role, :3, :3])
            g_uv = viz.project_xyz(g, cam, w, h, ref_width, ref_height)
        else:
            g_uv = None
        p = viz.transformed(cad[role], predT[role, :3, 3], predT[role, :3, :3])
        p_uv = viz.project_xyz(p, cam, w, h, ref_width, ref_height)
        gp = labelled_panel(rgb, g_uv, 'GT - GREEN', (60, 230, 60), invalid=not bool(valid[role]))
        pp = labelled_panel(rgb, p_uv, 'PRED - MAGENTA', (205, 40, 235))
        stripe = np.full((27, w * 3, 3), (25, 32, 43), np.uint8)
        if valid[role]:
            terr = np.linalg.norm(gtT[role, :3, 3] - predT[role, :3, 3]) * 1000
            tilt = viz.tilt_deg(predT[role, :3, :3], gtT[role, :3, :3])
            msg = f'{role_name}: GT-relative t={terr:.1f}mm  Z tilt={tilt:.1f}deg'
        else:
            msg = f'{role_name}: no valid GT (prediction is NOT confirmed detection)'
        cv2.putText(stripe, msg, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .44,
                    (225, 225, 230), 1, cv2.LINE_AA)
        rows.append(np.concatenate((np.concatenate((original, gp, pp), axis=1), stripe), axis=0))
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def make_video(task, ep, a, data, cam, object_names):
    cache_ep = a.cache_dir / task / f'e{ep:05d}'
    rgbs = np.load(cache_ep / 'rgb.npy', mmap_mode='r', allow_pickle=False)
    n = len(rgbs)
    if n != len(data['pred_T']):
        raise RuntimeError('Video RGB length != inference frame count')
    h, w = rgbs.shape[1:3]
    if h % 2 or w % 2:
        raise RuntimeError('H264 yuv420p requires even width and height')
    full_h, full_w = 2 * (h + 27), 3 * w
    video_path = a.output_dir / 'videos' / task / f'e{ep:05d}.mp4'
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        raise FileExistsError(video_path)
    proc = open_h264_writer(video_path, full_w, full_h, a.fps, a.crf)
    poster = None
    start = time.monotonic()
    try:
        for i in range(n):
            frame = video_frame(np.asarray(rgbs[i]), data['gt_T'][i],
                                data['pred_T'][i], data['gt_valid'][i],
                                data['cad'], cam, i, object_names,
                                a.ref_width, a.ref_height)
            if poster is None:
                ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if not ok:
                    raise RuntimeError('Poster encoding failed')
                poster = base64.b64encode(enc.tobytes()).decode('ascii')
            proc.stdin.write(frame.tobytes())
            if i == 0 or (i+1) % 125 == 0 or i == n-1:
                print(f'   [FRAME] {task} e{ep:05d} {i+1}/{n}', flush=True)
        size = finish_video(proc, video_path, n)
    except Exception:
        if proc.poll() is None:
            proc.kill()
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        video_path.unlink(missing_ok=True)
        raise
    valid = np.asarray(data['gt_valid'], dtype=bool)
    predt = data['pred_T'][:, :, :3, 3]
    gtt = data['gt_T'][:, :, :3, 3]
    errs = np.linalg.norm(predt - gtt, axis=-1) * 1000
    info = {
        'task': task, 'episode': ep, 'frames': n,
        'video': str(video_path.relative_to(a.output_dir)),
        'video_bytes': size, 'video_sha256': hash_file(video_path),
        'playback_fps': a.fps, 'duration_s_at_review_fps': round(n/a.fps, 3),
        'gt_valid_count_by_role': {ROLES[r]: int(valid[:,r].sum()) for r in range(2)},
        'gt_relative_mean_translation_mm_by_role': {
            ROLES[r]: round(float(errs[valid[:,r],r].mean()), 3) if valid[:,r].any() else None
            for r in range(2)
        },
        'seconds_to_generate': round(time.monotonic()-start, 2),
        'poster_b64': poster,
    }
    return info


STYLE = '''
body{font:15px/1.55 system-ui,-apple-system,sans-serif;background:#0e1420;color:#ecf1f9;margin:20px auto;max-width:1180px;padding:0 16px}
h1{font-size:24px;margin-bottom:7px}.sub{color:#b7c5d7}.warn{color:#ffcc88}
details{background:#192436;border:1px solid #324359;border-radius:12px;margin:16px 0;overflow:hidden}
summary{padding:15px;cursor:pointer;font-weight:650;list-style-position:inside}
article{padding:0 16px 16px}.meta{font-size:13px;color:#b7c5d7;margin:9px 0}
video{display:block;width:100%;height:auto;background:#0a1019;border-radius:6px}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;align-items:center}
button,select{border:1px solid #7890ad;border-radius:7px;padding:9px 12px;background:#26344a;color:white;font:inherit;cursor:pointer}
button:disabled{cursor:default;opacity:.5}.frame{font-variant-numeric:tabular-nums;color:#bdd3ef}
code{overflow-wrap:anywhere;color:#b8d6ff}h2{margin-top:26px}
.foot{font-size:12px;color:#abb9cc}button:focus-visible,summary:focus-visible{outline:2px solid #9cc4ff}
'''

JS = r'''
(() => {
  const rows = [...document.querySelectorAll('.episode')];
  const urls = [];
  function decodeBase64(b64) {
    const parts = [];
    const step = 4 * 65536;
    for (let i = 0; i < b64.length; i += step) {
      const str = atob(b64.slice(i, i + step));
      const bytes = new Uint8Array(str.length);
      for (let j = 0; j < str.length; j++) bytes[j] = str.charCodeAt(j);
      parts.push(bytes);
    }
    return new Blob(parts, {type: 'video/mp4'});
  }
  rows.forEach((box) => {
    const vid = box.querySelector('video');
    const load = box.querySelector('.load');
    const left = box.querySelector('.prev');
    const right = box.querySelector('.next');
    const speed = box.querySelector('.speed');
    const status = box.querySelector('.frame');
    const fps = Number(box.dataset.fps);
    const n = Number(box.dataset.n);
    let loaded = false;
    function showFrame() {
      const idx = Math.max(0, Math.min(n - 1, Math.floor(vid.currentTime * fps + 0.00001)));
      status.textContent = `第 ${idx + 1} / ${n} 帧`;
    }
    function ready() {
      if (loaded) return;
      const embedded = document.getElementById(box.dataset.encoded);
      if (!embedded) {status.textContent = '离线视频数据缺失';return;}
      const blob = decodeBase64(embedded.textContent.trim());
      const url = URL.createObjectURL(blob);
      urls.push(url);
      vid.src = url;
      vid.load();
      embedded.textContent = '';
      loaded = true;
      load.textContent = '视频已加载';
      load.disabled = true;
      left.disabled = false;
      right.disabled = false;
      speed.disabled = false;
      showFrame();
    }
    load.addEventListener('click', ready);
    function step(by) {
      if (!loaded) ready();
      vid.pause();
      const ix = Math.floor(vid.currentTime * fps + 0.00001);
      const target = Math.max(0, Math.min(n-1, ix + by));
      vid.currentTime = target / fps + 0.00001;
      showFrame();
    }
    left.addEventListener('click', () => step(-1));
    right.addEventListener('click', () => step(1));
    speed.addEventListener('change', () => {vid.playbackRate = Number(speed.value) || 1;});
    vid.addEventListener('timeupdate', showFrame);
    vid.addEventListener('seeked', showFrame);
    vid.addEventListener('error', () => {status.textContent = '视频解码失败；请检查浏览器是否支持 H.264 MP4';});
  });
  window.addEventListener('pagehide', () => urls.forEach(u => URL.revokeObjectURL(u)), {once:true});
})();
'''


def _write_video_b64(file_obj, vid_path):
    # Use chunk sizes divisible by 3 to avoid '=' midstream padding.
    with Path(vid_path).open('rb') as f:
        while True:
            chunk = f.read(3 * 65536)
            if not chunk:
                break
            file_obj.write(base64.b64encode(chunk).decode('ascii'))


def write_standalone(outdir, reports, metadata):
    target = outdir / 'index_standalone.html'
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix('.html.incomplete')
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        with tmp.open('w', encoding='utf-8') as f:
            f.write('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">')
            f.write('<meta name="viewport" content="width=device-width,initial-scale=1">')
            f.write('<title>V2 完整 Episode 追踪可视化</title><style>'+STYLE+'</style></head><body>')
            f.write('<h1>V2 — 完整 Episode 追踪预测复核</h1>')
            f.write('<p class="sub">每条视频包含 <b>所有原始帧</b>：上行操作物体，下行目标物体；每行从左到右为 RGB / 绿色旧 GT / 品红色 V2 预测。')
            f.write('可以播放、暂停、跳帧、调整播放速度。<b>视频和封面全部嵌入本 HTML，不需要外部图片。</b></p>')
            f.write('<p class="warn">重要：GT 可能有错；GT-relative mm 仅表示与现有标签差异，不代表真实误差。')
            f.write('没有 GT 的帧仍显示模型预测，但不等于物体已被检测或可见。')
            f.write('当前画面为 256×192 缓存 RGB；仅供离线视觉检查。</p>')
            f.write(f'<p class="foot">Checkpoint: <code>{html.escape(metadata["checkpoint"])}</code><br>')
            f.write(f'数据划分：{html.escape(metadata["split"])}；随机种子：{metadata["seed"]}；')
            f.write(f'播放帧率：{metadata["playback_fps"]} FPS（播放速度，不是采集速度）</p>')
            by_task = {}
            for j, r in enumerate(reports):
                by_task.setdefault(r['task'], []).append((j, r))
            for task in v2.TASK_NAMES:
                f.write(f'<h2>{html.escape(task)} — {len(by_task[task])} 条完整 Episode</h2>')
                for j, r in by_task[task]:
                    encid = f'embedded_video_{j}'
                    label = f'e{r["episode"]:05d}'
                    human = f'{r["frames"]} 帧 · {r["duration_s_at_review_fps"]} 秒（按播放帧率）'
                    f.write(f'<details class="episode" data-encoded="{encid}" data-n="{r["frames"]}" data-fps="{r["playback_fps"]}">')
                    f.write(f'<summary>{html.escape(task)} / {label}　{human}</summary><article>')
                    f.write(f'<video controls playsinline preload="none" poster="data:image/jpeg;base64,{r["poster_b64"]}"></video>')
                    f.write('<div class="actions"><button type="button" class="load">加载这条完整视频</button>')
                    f.write('<button type="button" class="prev" disabled>◀ 上一帧</button>')
                    f.write('<button type="button" class="next" disabled>下一帧 ▶</button>')
                    f.write('<label>速度 <select class="speed" disabled><option value="0.25">0.25×</option>')
                    f.write('<option value="0.5">0.5×</option><option value="1" selected>1×</option>')
                    f.write('<option value="2">2×</option></select></label>')
                    f.write(f'<span class="frame" aria-live="polite">第 1 / {r["frames"]} 帧</span></div>')
                    f.write('<p class="meta">')
                    for role in ROLES:
                        err = r['gt_relative_mean_translation_mm_by_role'][role]
                        shown = f'{err:.1f} mm' if err is not None else 'n/a'
                        f.write(f'{role}: 旧 GT 有效 {r["gt_valid_count_by_role"][role]}/{r["frames"]}，')
                        f.write(f'与旧 GT 的平均平移差 {shown}；')
                    f.write('</p><p class="foot">若 GT 偏离真实物体，请以原始 RGB 和连续预测轨迹进行人工判断。')
                    f.write('定位精度尚未经过独立外部真值验收。</p>')
                    f.write('</article></details>')
                    f.write(f'<script type="text/plain" id="{encid}">')
                    _write_video_b64(f, outdir / r['video'])
                    f.write('</script>')
                    print(f'[EMBED] {task} {label} H264={r["video_bytes"]/1048576:.2f} MiB', flush=True)
            f.write('<script>'+JS+'</script></body></html>')
        txt = tmp.stat().st_size
        if txt < 10000:
            raise RuntimeError('Standalone HTML unexpectedly small')
        tmp.rename(target)
        print(f'[DONE] Self-contained HTML: {target} ({txt/1048576:.1f} MiB)', flush=True)
        print('[VERIFIED] All videos are embedded, no relative external image/video paths.', flush=True)
        return target
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def self_test(a):
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f'Output directory not empty: {a.output_dir}')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = a.output_dir / 'videos' / '1018_peg_all' / 'e00000.mp4'
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_h264_writer(sample_path, 768, 438, 6, 28)
    try:
        for k in range(12):
            frame = np.full((438,768,3), 40, np.uint8)
            cv2.rectangle(frame, (20+k*15, 100), (120+k*15, 200), (32, 190, 220), -1)
            writer.stdin.write(frame.tobytes())
        size = finish_video(writer, sample_path, 12)
    except Exception:
        if writer.poll() is None: writer.kill()
        writer.wait()
        raise
    poster_ok, p = cv2.imencode('.jpg',np.full((438,768,3),55,np.uint8))
    if not poster_ok:
        raise RuntimeError('JPEG failed')
    item = {'task':'1018_peg_all','episode':0,'frames':12,'video':str(sample_path.relative_to(a.output_dir)),
            'playback_fps':6,'duration_s_at_review_fps':2,'video_bytes':size,
            'poster_b64':base64.b64encode(p.tobytes()).decode('ascii'),
            'gt_valid_count_by_role':{'manipulated':12,'target':12},
            'gt_relative_mean_translation_mm_by_role':{'manipulated':0.,'target':0.}}
    # For test-only HTML, simulate other tasks without embedding duplicates, by modifying task catalog locally is prohibited.
    # Create 4 separate self-test items, one per task to exercise all render paths.
    items=[]
    for task in v2.TASK_NAMES:
        vid = a.output_dir / 'videos' / task / 'e00000.mp4'
        vid.parent.mkdir(parents=True, exist_ok=True)
        if vid != sample_path: shutil.copy2(sample_path, vid)
        items.append({**item, 'task':task, 'video':str(vid.relative_to(a.output_dir))})
    index=write_standalone(a.output_dir,items,{'checkpoint':'self-test','split':'test','seed':42,'playback_fps':6})
    text=index.read_text()
    if text.count('<script type="text/plain" id="embedded_video_')!=4 or 'video src=' in text:
        raise RuntimeError('Standalone HTML video embedding sanity check failed')
    print('[PASS] Synthetic H264 encoding + frame count + single-file HTML packaging', flush=True)


def main():
    a = args_parse()
    if a.self_test:
        self_test(a)
        return
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f'Refusing nonempty --output-dir: {a.output_dir}')
    if not a.checkpoint_json.is_file():
        raise FileNotFoundError(a.checkpoint_json)
    ckpt_path = Path(json.loads(a.checkpoint_json.read_text())['checkpoint']).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path,map_location='cpu',weights_only=False) # trusted checkpoint only
    if ckpt.get('schema') != v2.SCHEMA or not {'width','height','depth_scale'} <= ckpt.keys():
        raise RuntimeError('Checkpoint is not compatible with RGB-D V2')
    cache_cfg = json.loads((a.cache_dir / 'cache_config.json').read_text())
    if (ckpt['width'],ckpt['height']) != (cache_cfg['width'],cache_cfg['height']):
        raise ValueError('RGB-D cached resolution does not match checkpoint')
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg required for H.264 offline video encoding')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable for requested device {device}')
    selected = select_random_eps(a.cache_dir, a.split,a.episodes_per_task,a.seed,a.episodes)
    model = v2.SingleCadRGBDPose()
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device).eval()
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    camera = viz.front_camera(a.repo)
    peg_cam = viz.peg_camera(a.repo, camera) if a.peg_intrinsics=='legacy' else camera
    meta={'schema':'v2_full_episode_video_review_v1',
          'status':'unreviewed; candidate predictions only; not for deployment',
          'checkpoint':str(ckpt_path), 'checkpoint_sha256': hash_file(ckpt_path),
          'cache_config_sha256':hash_file(a.cache_dir / 'cache_config.json'),
          'split':a.split,'seed':a.seed,'selected_episodes':selected,
          'frame_policy':'all frames, including frames with missing GT',
          'playback_fps':a.fps,'peg_intrinsics':a.peg_intrinsics,
          'projection':'existing color camera; image = RGB cache, not original resolution'}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'selection.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    reports=[]
    for task in v2.TASK_NAMES:
        for ep in selected[task]:
            print(f'[INFER] {task} e{ep:05d}',flush=True)
            data = bridge.infer_episode(model,ckpt,a.cache_dir,task,ep,device,a.batch_size)
            cam = peg_cam if task == '1018_peg_all' else camera
            info=make_video(task,ep,a,data,cam,bridge.OBJECTS[task])
            reports.append(info)
            print(f'[OK] {task} e{ep:05d}: {info["frames"]} frames, H264 {info["video_bytes"]/1048576:.1f} MiB, render {info["seconds_to_generate"]:.1f}s', flush=True)
    # Embedded data includes posters; leave posters out of machine-readable summary to keep it small.
    clean=[{k:v for k,v in r.items() if k!='poster_b64'} for r in reports]
    (a.output_dir/'summary.json').write_text(json.dumps({**meta,'episodes':clean},indent=2,ensure_ascii=False)+'\n')
    write_standalone(a.output_dir,reports,meta)
    print(f'[FINAL] {len(reports)} complete episodes, {sum(r["frames"] for r in reports)} frames; all embedded', flush=True)


if __name__ == '__main__':
    main()
