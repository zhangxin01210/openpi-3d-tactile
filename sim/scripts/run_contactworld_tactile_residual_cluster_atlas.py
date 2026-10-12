#!/usr/bin/env python3
"""Whole-trajectory clustering of bilateral TacFF after stable-grasp subtraction.

The pre-contact baseline is estimated in each finger's local sensor frame.
The residual is then rotated to the robot base at every frame. Only the six
signed residual maps enter clustering and K selection; motion and contact
labels are withheld until after assignment.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import run_contactworld_tactile_cluster_atlas as raw_atlas
from run_contactworld_matched_dynamics import atomic_json, encode_frames
from run_contactworld_sensor_discovery import DEFAULT_DATASET, image_paths
from run_contactworld_tactile_action_branches_v3 import stable_baseline

DEFAULT_OUTPUT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_residual_clusters_20261008')
VERSION = 'cw-tactile-residual-whole-trajectory-v1'
SPLITS = ('train', 'val', 'test')


def local_field_to_base(field, taxel_quat_world, base_quat_world):
    """Convert TacFF [normal,shear_x,shear_y] to robot-base XYZ."""
    field = np.asarray(field, np.float32)
    n = len(field)
    if field.shape != (n, 2, 140, 3) or taxel_quat_world.shape != (n, 2, 140, 4):
        raise ValueError('Unexpected local tactile/taxel dimensions')
    if base_quat_world.shape != (n, 4):
        raise ValueError('Unexpected robot-base orientation dimensions')
    local_xyz = np.stack((-field[..., 1], -field[..., 0], field[..., 2]), axis=-1)
    sensor_rot = Rotation.from_quat(taxel_quat_world.reshape(-1, 4)).as_matrix().reshape(n, 2, 140, 3, 3)
    world = np.einsum('ntkij,ntkj->ntki', sensor_rot, local_xyz, optimize=True)
    base_inv = Rotation.from_quat(base_quat_world).inv().as_matrix()
    return np.einsum('nij,ntkj->ntki', base_inv, world, optimize=True).astype(np.float32)


def load_episodes(args):
    manifest = json.loads((args.dataset / 'spatial/manifest.json').read_text())
    entries = manifest['episodes']
    if args.smoke:
        entries = [e for split, count in (('train', 6), ('val', 2), ('test', 2))
                   for e in [x for x in entries if x['split'] == split][:count]]
    capture_root = Path(manifest['capture_root'])
    episodes = {}
    parts = {split: [] for split in SPLITS}
    baselines = []
    for i, e in enumerate(entries, 1):
        ep = int(e['episode_index'])
        source = int(e['source_episode'])
        n = int(e['length'])
        folder = capture_root / f'insertion_usb_episode_{source:03d}'
        with np.load(folder / 'v2_capture.npz') as z:
            plug = z['plug_pose'].copy()
            socket = z['socket_pose'].copy()
            ee = z['ee_pose'].copy()
            action = z['action_executed'].copy()
        replay = json.loads((folder / 'replay.json').read_text())['rows']
        if any(len(x) != n for x in (plug, socket, ee, action, replay)):
            raise ValueError(f'Capture/replay length mismatch: source {source}')
        with np.load(folder / 'bilateral_tactile.npz') as z:
            local_grid = np.stack([z[f'force_grid_{hand}'] for hand in ('left', 'right')], axis=1).astype(np.float32)
            raw_base = np.stack([z[f'force_base_{hand}'] for hand in ('left', 'right')], axis=1).astype(np.float32)
            taxel_quat = np.stack([z[f'taxel_quat_world_{hand}'] for hand in ('left', 'right')], axis=1).astype(np.float32)
            base_quat = z['base_pose_world'][:, 3:].copy()
        if local_grid.shape != (n, 2, 10, 14, 3) or raw_base.shape != (n, 2, 140, 3):
            raise ValueError(f'Bad bilateral force shape: source {source}')
        contact = np.r_[False, [np.linalg.norm(r['plug_socket_force_post']) > .1 for r in replay[:-1]]]
        baseline = stable_baseline(local_grid, contact, action)
        if baseline is None:
            raise ValueError(f'No stable pre-contact grasp baseline for source {source}; cannot use frame 0 instead')
        local = local_grid.reshape(n, 2, 140, 3)
        background_local = baseline['local'].reshape(2, 140, 3)
        residual_base = local_field_to_base(local - background_local[None], taxel_quat, base_quat)
        reconstructed_raw_base = local_field_to_base(local, taxel_quat, base_quat)
        roundtrip_error = float(np.max(np.abs(reconstructed_raw_base - raw_base)))
        if roundtrip_error > 2e-5:
            raise ValueError(f'Local/base force conversion mismatch source {source}: {roundtrip_error}')
        baseline_base = raw_base - residual_base
        baselines.append(dict(split=e['split'], source=source,
                              window=[baseline['start'], baseline['stop']],
                              first_contact=int(np.flatnonzero(contact)[0]) if contact.any() else None,
                              baseline_local_abs=float(np.abs(background_local).mean()),
                              residual_base_abs=float(np.abs(residual_base).mean()),
                              raw_roundtrip_max_abs=roundtrip_error))
        sock_rot = Rotation.from_quat(socket[:, 3:])
        relative = (sock_rot.inv().apply(plug[:, :3] - socket[:, :3]) * 1000).astype(np.float32)
        action_xyz = sock_rot.inv().apply(action[:, :3] * .01) * 1000
        action_rot = action[:, 3:] * (.05 * 180 / np.pi)
        action_scaled = np.concatenate((action_xyz, action_rot), axis=1).astype(np.float32)
        next_motion = np.zeros((n, 3), np.float32)
        next_motion[:-1] = relative[1:] - relative[:-1]
        success = np.r_[False, [bool(r['source_success']) for r in replay[:-1]]]
        item = dict(entry=e, force=residual_base, raw_force=raw_base,
                    baseline_force=baseline_base, relative=relative,
                    action=action_scaled, next_motion=next_motion,
                    contact=contact, success=success,
                    grasp=(plug[:, :3] - ee[:, :3]).astype(np.float32))
        episodes[(e['split'], source)] = item
        parts[e['split']].append((e, item))
        print(f'load {i}/{len(entries)} {e["split"]} source={source} frames={n} '
              f'baseline={baseline["start"]}:{baseline["stop"]}', flush=True)
    return entries, episodes, parts, baselines


def choose_k(args, features):
    train, val = features['train'], features['val']
    rng = np.random.default_rng(20261008)
    sampled = rng.choice(len(val), min(args.silhouette_frames, len(val)), replace=False)
    candidates = list(range(args.k_min, args.k_max + 1, args.k_step))
    if args.smoke:
        candidates = [k for k in candidates if k <= min(10, len(train) // 20)]
    if args.k is not None:
        candidates.append(args.k)
    candidates = sorted(set(candidates))
    if not candidates or len(train) <= max(candidates):
        raise ValueError('Not enough training frames for candidate K values')
    rows, models = [], {}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4)
    for k in candidates:
        cv2.setRNGSeed(6100 + k)
        compactness, train_label, centers = cv2.kmeans(
            train, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.float32)
        val_label, val_distance = raw_atlas.assignment(val, centers)
        cv2.setRNGSeed(8100 + k)
        _, _, alternative = cv2.kmeans(train, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        alternative_label, _ = raw_atlas.assignment(val, alternative)
        train_count = np.bincount(train_label[:, 0], minlength=k)
        val_count = np.bincount(val_label, minlength=k)
        silhouette = raw_atlas.silhouette_sample(val[sampled], val_label[sampled])
        eligible_occupancy = (train_count.min() >= max(8, int(.005 * len(train)))
                              and val_count.min() >= max(3, int(.005 * len(val))))
        stable = raw_atlas.adjusted_rand(val_label, alternative_label)
        row = dict(k=k, val_silhouette=silhouette,
                   val_rms=float(np.sqrt(np.mean(val_distance ** 2))),
                   train_rms=float(np.sqrt(compactness / (len(train) * train.shape[1]))),
                   stability_ari=stable,
                   train_min_count=int(train_count.min()),
                   val_min_count=int(val_count.min()),
                   eligible=bool(eligible_occupancy and stable >= .6),
                   occupancy_ok=bool(eligible_occupancy))
        rows.append(row)
        models[k] = centers
        print(f'K sweep {k}: silhouette={silhouette:.3f} val_rms={row["val_rms"]:.3f} '
              f'ARI={stable:.3f} min_val={val_count.min()} eligible={row["eligible"]}', flush=True)
    if args.k is not None:
        chosen, reason = args.k, 'explicit --k override'
    else:
        pool = [r for r in rows if r['eligible']]
        if not pool:
            pool = [r for r in rows if r['occupancy_ok']]
        if not pool:
            pool = rows
        best = max(r['val_silhouette'] for r in pool)
        chosen = min(r['k'] for r in pool if r['val_silhouette'] >= best - .005)
        reason = ('best held-out silhouette within 0.005, preferring smaller K; '
                  'occupancy >=0.5% per train/val class and ARI >=0.6 when feasible')
    return chosen, models[chosen], rows, reason


def attach_frame_labels(parts, assigned):
    for split in SPLITS:
        pos = 0
        for _, episode in parts[split]:
            n = len(episode['force'])
            episode['frame_label'] = assigned[split]['label'][pos:pos+n].copy()
            pos += n
        if pos != len(assigned[split]['label']):
            raise ValueError(f'Assignment/episode mismatch: {split}')


def composite(front, wrist, left, right, episode, scale, case, shown):
    canvas = np.full((750, 1280, 3), 25, np.uint8)
    canvas[50:370, 0:320] = front
    canvas[50:370, 320:640] = wrist
    cv2.putText(canvas, 'FRONT', (8, 33), cv2.FONT_HERSHEY_SIMPLEX, .6, (245,245,245), 1)
    cv2.putText(canvas, 'WRIST', (327, 33), cv2.FONT_HERSHEY_SIMPLEX, .6, (245,245,245), 1)
    tile = raw_atlas.six_force_image(episode['force'][shown], scale)
    canvas[35:385, 645:1275] = tile
    canvas[425:745, 0:320] = left
    canvas[425:745, 320:640] = right
    cv2.putText(canvas, 'LEFT RAW TACTILE PREVIEW', (8, 408), cv2.FONT_HERSHEY_SIMPLEX, .52, (245,245,245), 1)
    cv2.putText(canvas, 'RIGHT RAW TACTILE PREVIEW', (327, 408), cv2.FONT_HERSHEY_SIMPLEX, .52, (245,245,245), 1)
    xyz = episode['relative'][shown]
    action = episode['action'][shown]
    motion = episode['next_motion'][shown]
    current_class = int(episode['frame_label'][shown])
    lines = [
        f'MARKED CLASS {case["cluster"]} | CURRENT CLASS {current_class} | source {case["source"]}',
        f'marked frame {case["frame"]} | shown {shown} ({shown-case["frame"]:+d}) | {case["split"]}',
        f'contact {int(episode["contact"][shown])} | success {int(episode["success"][shown])}',
        f'plug relative xyz mm: {xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f}',
        f'command xyz mm/step: {action[0]:+.1f}, {action[1]:+.1f}, {action[2]:+.1f}',
        f'next actual plug dz mm: {motion[2]:+.2f}',
        'Base-frame RESIDUAL TacFF: red +, blue -',
    ]
    for j, line in enumerate(lines):
        cv2.putText(canvas, line, (650, 427+j*43),
                    cv2.FONT_HERSHEY_SIMPLEX, .51, (235,235,235), 1, cv2.LINE_AA)
    return canvas


def render_examples(args, selected, episodes, residual_scale, raw_scale, centers):
    out = args.output / 'examples'
    out.mkdir(parents=True, exist_ok=True)
    for cluster, cases in selected.items():
        center = centers[cluster].reshape(2, 140, 3) * residual_scale[:, None, :]
        cv2.imwrite(str(out / f'cluster_{cluster:02d}_center.png'),
                    raw_atlas.six_force_image(center, residual_scale))
        for i, case in enumerate(cases):
            ep = episodes[(case['split'], case['source'])]
            n, t = len(ep['force']), case['frame']
            lo, hi = max(0, t-8), min(n, t+13)
            normalized = np.clip(ep['force'][t] / residual_scale[:, None, :], -3, 3)
            center_map = centers[cluster].reshape(2, 140, 3)
            case['six_map_rms'] = np.sqrt(np.mean((normalized-center_map)**2, axis=1)).round(3).tolist()
            paths = image_paths(args.dataset, case['episode'])
            stem = f'cluster_{cluster:02d}_{i:02d}_{case["split"]}_source{case["source"]:03d}_f{t:03d}'
            movie = out / f'{stem}.mp4'
            marker = out / f'{stem}_residual.png'
            raw_marker = out / f'{stem}_raw.png'
            baseline_marker = out / f'{stem}_baseline.png'
            cv2.imwrite(str(marker), raw_atlas.six_force_image(ep['force'][t], residual_scale))
            cv2.imwrite(str(raw_marker), raw_atlas.six_force_image(ep['raw_force'][t], raw_scale))
            cv2.imwrite(str(baseline_marker), raw_atlas.six_force_image(ep['baseline_force'][t], raw_scale))
            indices = list(range(lo, hi))
            if not movie.is_file() or not movie.with_suffix('.jpg').is_file():
                front = raw_atlas.read_video_frames(paths['front'], indices)
                wrist = raw_atlas.read_video_frames(paths['wrist'], indices)
                left = raw_atlas.read_video_frames(paths['tacff_left_preview'], indices)
                right = raw_atlas.read_video_frames(paths['tacff_right_preview'], indices)
                encode_frames(movie, 1280, 750,
                    (composite(a, b, c, d, ep, residual_scale, case, j)
                     for a,b,c,d,j in zip(front,wrist,left,right,indices)), fps=10)
                cv2.imwrite(str(movie.with_suffix('.jpg')),
                    composite(front[t-lo], wrist[t-lo], left[t-lo], right[t-lo],
                              ep, residual_scale, case, t))
            case.update(video=str(movie.relative_to(args.output)),
                        poster=str(movie.with_suffix('.jpg').relative_to(args.output)),
                        marker_force=str(marker.relative_to(args.output)),
                        marker_raw=str(raw_marker.relative_to(args.output)),
                        marker_baseline=str(baseline_marker.relative_to(args.output)),
                        marked_video_frame=t-lo)
            print(f'render class {cluster} example {i+1}/{len(cases)} {stem}', flush=True)
    atomic_json(args.output / 'representatives.json', selected)


def report(args, selected, summary, sweep, chosen, reason, scale, raw_scale):
    css = """<style>
body{font:16px/1.5 system-ui;background:#101827;color:#edf4fe;max-width:1900px;margin:auto;padding:22px}
nav{display:flex;gap:11px;flex-wrap:wrap;background:#243b59;padding:13px;position:sticky;top:0;z-index:5}a{color:#a9d8ff}
section{background:#1d2a3e;margin:20px 0;padding:20px;border-radius:12px;scroll-margin-top:70px}
article{background:#263750;padding:13px;border-radius:9px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}
.grid video,.grid img{max-width:100%;height:auto}
.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}
.maps figure{margin:0}.maps figcaption{font-size:14px}
button{background:#2d75af;color:white;border:0;border-radius:6px;padding:9px 13px;margin:5px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #53677f;padding:6px;text-align:left}th{background:#354a68}
.note{background:#4a3927;padding:10px;border-radius:7px}
@media(max-width:900px){.grid,.maps{grid-template-columns:1fr}}
</style>"""
    nav = '<nav><a href="#method">方法</a>' + ' '.join(
        f'<a href="#class-{i}">类 {i}</a>' for i in range(chosen)) + ' <a href="#selection">类数评估</a></nav>'
    sections = []
    for cluster in range(chosen):
        cases = selected[cluster]
        row = summary['test'][cluster]
        examples = []
        for case in cases:
            delta = ', '.join(f'{v:+.1f}' for v in case['action'][:3])
            map_rms = '；'.join(
                f'{side} Fx {values[0]:.2f} / Fy {values[1]:.2f} / Fz {values[2]:.2f}'
                for side, values in zip(('左', '右'), case['six_map_rms']))
            examples.append(f"""<article><h3>{html.escape(case['split'])} · source {case['source']} · 帧 {case['frame']}</h3>
<p>距类中心 {case['center_distance']:.3f}；接触 {int(case['contact'])}；
当前动作 XYZ {delta} mm/步；下一步 USB 实际 Z 位移 {case['next_motion'][2]:+.2f} mm。<br>
六张残差图各自到中心的归一化 RMS：{map_rms}。</p>
<div class="maps">
<figure><img loading="lazy" src="{html.escape(case['marker_raw'])}"><figcaption>原始 base 力图</figcaption></figure>
<figure><img loading="lazy" src="{html.escape(case['marker_baseline'])}"><figcaption>投到当前 base 的稳定夹持基线</figcaption></figure>
<figure><img loading="lazy" src="{html.escape(case['marker_force'])}"><figcaption>实际聚类的残差图</figcaption></figure>
</div>
<video controls preload="none" width="1280" height="750" poster="{html.escape(case['poster'])}"
src="{html.escape(case['video'])}" data-mark="{case['marked_video_frame']}"></video></article>""")
        sections.append(f"""<section id="class-{cluster}">
<h2>类 {cluster}：测试集 {row['frames']} 帧，{row['demonstrations']} 条示范</h2>
<p>插座接触 {row.get('contact_fraction',0):.0%}；下推指令 {row.get('down_command_fraction',0):.0%}；
横向指令 {row.get('lateral_command_fraction',0):.0%}。明确下推结果中：
推进 {row.get('down_progress',0)} 帧，受阻 {row.get('down_block',0)} 帧。</p>
<p>类中心六张残差图：</p><img width="630" height="350" src="examples/cluster_{cluster:02d}_center.png">
<p><button onclick="playAligned({cluster})">从各自代表帧同步播放</button>
<button onclick="playStart({cluster})">从片段开头同步播放</button>
<button onclick="pauseAll({cluster})">暂停</button>
<label>相对代表帧偏移 <input type="range" min="-8" max="12" value="0"
oninput="scrub({cluster},this.value)"> <span id="offset-{cluster}">0</span> 步</label></p>
<div class="grid" data-class="{cluster}">{''.join(examples) if examples else '<p>独立示范不足。</p>'}</div></section>""")
    ks = ''.join(f'<tr><td>{r["k"]}</td><td>{r["val_silhouette"]:.3f}</td><td>{r["val_rms"]:.3f}</td>'
                 f'<td>{r["stability_ari"]:.3f}</td><td>{r["val_min_count"]}</td>'
                 f'<td>{"✓" if r["eligible"] else "—"}</td></tr>' for r in sweep)
    scale_text = '；'.join(
        f'{side} Fx {scale[h,0]:.6f}, Fy {scale[h,1]:.6f}, Fz {scale[h,2]:.6f}'
        for h, side in enumerate(('左指', '右指')))
    script = """<script>
function videos(k){return [...document.querySelectorAll('[data-class="'+k+'"] video')]}
function ready(v){return new Promise(resolve=>{if(v.readyState>=1){resolve();return}v.addEventListener('loadedmetadata',resolve,{once:true});v.load()})}
async function playAligned(k){const a=videos(k);await Promise.all(a.map(ready));a.forEach(v=>{v.pause();v.currentTime=Number(v.dataset.mark)/10});await Promise.all(a.map(v=>v.play().catch(()=>{})))}
async function playStart(k){const a=videos(k);await Promise.all(a.map(ready));a.forEach(v=>{v.pause();v.currentTime=0});await Promise.all(a.map(v=>v.play().catch(()=>{})))}
function pauseAll(k){videos(k).forEach(v=>v.pause())}
async function scrub(k,offset){document.getElementById('offset-'+k).textContent=offset;const a=videos(k);await Promise.all(a.map(ready));a.forEach(v=>{v.pause();v.currentTime=Math.max(0,Math.min((Number(v.dataset.mark)+Number(offset))/10,v.duration-.05))})}
</script>"""
    content = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>ContactWorld 双指触觉残差聚类</title>{css}
<h1>ContactWorld USB 双指触觉残差聚类</h1>
<p>接近、接触、调整和插入的整段示范都参与聚类。每类包含多个来自不同示范的真实代表帧及连续片段。</p>
{nav}
<section id="method"><h2>什么是本次输入</h2>
<p>每条示范先在首次插座接触前寻找四帧稳定夹持窗口；在左右指各自的传感器局部坐标减去该窗口的中位力图，
再把每帧残差旋转到该帧机器人 base 坐标。已逐帧核对局部到 base 的转换与原始记录一致。
聚类输入为左、右指各 140 个 taxel 的 Fx/Fy/Fz 残差，共 840 个有符号数值。</p>
<p>各手指、各分量按训练集残差绝对值的 99.5% 分位固定缩放并截断到 ±3。
没有逐帧强度归一化或 PCA。欧氏距离比较六张完整图；代表帧还要求每张图均接近类中心。
接触、动作、USB 位姿及结果只用于聚类后说明，不参与距离或类数选择。</p>
<p class="note">类别更少只说明压缩得更多，不自动意味着类内更精确。请核对每类原始／基线／残差三组图、
不同示范的代表帧、跨类中心距离和视频中的逐帧类别。视频离开代表帧后可能进入其他类别。</p>
<p>残差图红色为正、蓝色为负；所有类别共用各分量标尺：{scale_text}。
原始与基线图彼此共用另一个标尺，以免残差过小看不清。</p></section>
{''.join(sections)}
<section id="selection"><h2>为什么选 {chosen} 类</h2>
<p>候选 K 在训练示范上拟合，整条示范隔离的验证集评价 silhouette、图像重建误差和重启稳定性。
在每类至少约占训练／验证帧 0.5%、ARI 至少 0.6 的候选中，选验证 silhouette 最高者；
差距不超过 0.005 时选较小 K。若没有候选满足条件，会放宽门槛并在 JSON 保留全部指标。
本次选择：{html.escape(reason)}。K 并非唯一的“真实类别数”。</p>
<img src="figures/k_selection.png" style="max-width:100%">
<table><tr><th>K</th><th>验证 silhouette ↑</th><th>验证 RMS ↓</th>
<th>重启稳定性 ARI ↑</th><th>验证最小类帧数</th><th>合格</th></tr>{ks}</table>
<h3>类间差异与完整轨迹</h3>
<img src="figures/class_separation.png" style="max-width:48%">
<img src="figures/action_associations.png" style="max-width:48%">
<img src="figures/test_timelines.png" style="max-width:100%"></section>
<section><h2>完整数据</h2>
<p><a href="baseline_audit.json">逐示范夹持基线</a> · <a href="k_selection.json">候选 K</a> ·
<a href="cluster_summary.json">逐类统计</a> · <a href="representatives.json">代表帧</a> ·
<a href="frame_assignments.csv">全部逐帧类别与动作</a></p>
<p>同类是力图相似的描述，不代表动作后果相同；力还可能反映前几步运动。
动作关联是观察性结果，不能单独证明触觉能选择纠偏方向。</p></section>{script}</html>"""
    (args.output / 'index.html').write_text(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--k', type=int, help='Explicit K; otherwise select from held-out validation')
    p.add_argument('--k-min', type=int, default=2)
    p.add_argument('--k-max', type=int, default=18)
    p.add_argument('--k-step', type=int, default=1)
    p.add_argument('--silhouette-frames', type=int, default=700)
    p.add_argument('--examples-per-cluster', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.k_min < 2 or args.k_step < 1 or (args.k is not None and args.k < 2):
        p.error('K must be >=2')
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    entries, episodes, parts, baselines = load_episodes(args)
    identity = dict(version=VERSION, smoke=args.smoke,
                    entries=[[e['split'], e['source_episode']] for e in entries],
                    examples_per_cluster=args.examples_per_cluster, k=args.k,
                    k_range=[args.k_min, args.k_max, args.k_step])
    identity_file = args.output / 'identity.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError('Output identity differs; use another --output')
    atomic_json(identity_file, identity)
    (args.output / 'complete.json').unlink(missing_ok=True)
    atomic_json(args.output / 'baseline_audit.json',
                dict(method='local median over stable four-frame pre-contact grasp, rotated to each current base pose',
                     rows=baselines, found=len(baselines), episodes=len(entries)))
    data = raw_atlas.concatenate(parts)
    features, scale = raw_atlas.normalized_force(data)
    train_raw = np.concatenate([d['raw_force'] for _,d in parts['train']])
    raw_scale = np.maximum(np.quantile(np.abs(train_raw), .995, axis=(0, 2)).astype(np.float32), 1e-6)
    chosen, centers, sweep, reason = choose_k(args, features)
    assigned = raw_atlas.annotate(data, features, centers)
    attach_frame_labels(parts, assigned)
    summary = raw_atlas.summarize(data, assigned, chosen)
    selected = raw_atlas.select_representatives(args, episodes, data, assigned, features, centers, chosen)
    atomic_json(args.output / 'k_selection.json',
                dict(chosen_k=chosen, reason=reason, candidates=sweep,
                     distance='RMS over 840 signed baseline-subtracted base-frame force values',
                     same_map_validation='representatives selected by maximum per-map normalized RMS'))
    atomic_json(args.output / 'cluster_summary.json', summary)
    np.savez_compressed(args.output / 'cluster_model.npz',
                        centers=centers, scale=scale, raw_scale=raw_scale)
    raw_atlas.save_assignments(args, data, assigned)
    raw_atlas.plot_diagnostics(args, sweep, chosen, centers, data, assigned, summary)
    render_examples(args, selected, episodes, scale, raw_scale, centers)
    report(args, selected, summary, sweep, chosen, reason, scale, raw_scale)
    from build_contactworld_residual_all_members import build as build_all_members
    browser = build_all_members(args.output, args.dataset)
    atomic_json(args.output / 'complete.json',
                dict(version=VERSION, elapsed_s=round(time.monotonic()-start, 1),
                     frames={split:len(data[split]['force']) for split in SPLITS},
                     chosen_k=chosen, example_videos=sum(map(len, selected.values())),
                     all_member_frames=browser['frames'], all_member_source_videos=browser['source_videos']))
    print(f'DONE k={chosen} elapsed_s={time.monotonic()-start:.1f} report={args.output/"index.html"}', flush=True)


if __name__ == '__main__':
    main()

