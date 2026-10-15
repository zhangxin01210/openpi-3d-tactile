#!/usr/bin/env python3
"""Add synchronized, per-control-step bilateral tactile videos to failure_audit.html.

The existing RGB video has one initial frame followed by one frame after every
control step. ``force_local_post_step`` has the corresponding post-step force
grids. The initial force grid is ``force_local[0]``. We encode a separate video
with exactly the RGB video's frame count and frame rate, then synchronize the
two in the report. The original evaluation videos and data are never changed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPORT = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_action_effect_20261009')
SCALE_MANIFEST = Path('/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/manifest.json')
WIDTH, HEIGHT = 512, 432
FOCUS_MODELS = {
    '01': ('RGB', 'pi0_cw3_usb_01_rgb'),
    '02': ('＋点云', 'pi0_cw3_usb_02_fused_pc'),
    '03': ('＋局部触觉图', 'pi0_cw3_usb_03_rgb_local_map'),
    '04': ('点云＋局部触觉图', 'pi0_cw3_usb_04_fused_pc_local_map'),
    '05': ('触觉统计量', 'pi0_cw3_usb_05_fused_pc_local_summary'),
    '06': ('base 三维触觉、左右分组', 'pi0_cw3_usb_06_fused_pc_base_structured'),
    '07': ('base 三维触觉、合池化', 'pi0_cw3_usb_07_fused_pc_base_pooled'),
    '08': ('base 耦合力图', 'pi0_cw3_usb_08_fused_pc_base_map'),
    '09': ('Fx/Fy/Fz 分开编码', 'pi0_cw3_usb_09_fused_pc_base_map_split'),
    '10': ('耦合力图容量对照', 'pi0_cw3_usb_10_fused_pc_base_map_matched'),
    '11': ('点云、触觉均接 prefix', 'pi0_cw3_usb_11_fused_pc_base_structured_prefix'),
    '12': ('点云 prefix、触觉 suffix', 'pi0_cw3_usb_12_fused_pc_base_structured_split_route'),
    '13': ('局部触觉图每指 3 token', 'pi0_cw3_usb_13_fused_pc_local_map_3tok'),
}
EVENTS = (
    ('final_success', '最终成功'),
    ('transient_success', '曾成功、末态失败'),
    ('contact_no_success', '从未成功：检测到插座接触'),
    ('no_detected_contact', '从未成功：未检测到插座接触'),
)
EVENT_NAMES = dict(EVENTS)


def video_info(path: Path) -> tuple[str, int]:
    proc = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=avg_frame_rate,nb_frames', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)['streams'][0]
    return stream['avg_frame_rate'], int(stream['nb_frames'])


def label(image: np.ndarray, text: str, x: int, y: int, size: float = .46,
          color: tuple[int, int, int] = (230, 236, 242)) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 1, cv2.LINE_AA)


def heatmap(values: np.ndarray, limit: float) -> np.ndarray:
    v = np.clip(values / limit, -1, 1)
    red = np.where(v >= 0, 245, 245 + 190 * v)
    blue = np.where(v <= 0, 245, 245 - 190 * v)
    green = 245 - 160 * np.abs(v)
    bgr = np.stack([blue, green, red], -1).astype(np.uint8)
    return cv2.resize(bgr, (156, 100), interpolation=cv2.INTER_NEAREST)


def curve_background(forces: np.ndarray, limits: np.ndarray,
                     first_contact_step: int | None, mode: str) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.full((HEIGHT, WIDTH, 3), (28, 32, 39), dtype=np.uint8)
    heading = ('ROBOT-BASE TACTILE | every 4 steps, held' if mode == 'base'
               else 'SENSOR-LOCAL TACTILE | per control step')
    label(canvas, heading, 8, 18, .46)
    label(canvas, 'blue: negative  /  white: zero  /  red: positive', 8, 306, .39)
    label(canvas, 'Mean normalized force: left (cyan), right (orange)', 8, 326, .38)
    cv2.rectangle(canvas, (32, 341), (500, 412), (73, 83, 91), 1)
    if first_contact_step is not None:
        x = 32 + round(468 * (first_contact_step + 1) / max(1, len(forces) - 1))
        cv2.line(canvas, (x, 342), (x, 411), (95, 100, 108), 1)
    strengths = np.linalg.norm(forces / limits[None, None, None, None, :], axis=-1)
    strengths = strengths.mean(axis=(2, 3))
    xs = 32 + np.rint(np.linspace(0, 468, len(forces))).astype(np.int32)
    colors = ((245, 215, 105), (80, 181, 250))  # BGR: cyan, orange.
    for hand in range(2):
        ys = 411 - np.rint(np.clip(strengths[:, hand], 0, 3) * 22).astype(np.int32)
        points = np.c_[xs, ys].astype(np.int32)
        cv2.polylines(canvas, [points], False, colors[hand], 2, cv2.LINE_AA)
    label(canvas, '0', 8, 410, .34)
    label(canvas, '3', 8, 349, .34)
    return canvas, xs


def draw_frame(base: np.ndarray, force: np.ndarray, limits: np.ndarray,
               step: int, xs: np.ndarray, mode: str) -> np.ndarray:
    frame = base.copy()
    for hand, side in enumerate(('L', 'R')):
        for axis, name in enumerate(('Fx', 'Fy', 'Fz')):
            x, y = axis * 170 + 7, hand * 133 + 44
            label(frame, f'{side} {name} {mode}', x, y - 8, .44)
            frame[y:y + 100, x:x + 156] = heatmap(force[hand, :, :, axis], limits[axis])
    cv2.line(frame, (int(xs[step]), 342), (int(xs[step]), 411), (120, 225, 215), 1)
    label(frame, f'frame {step}/{len(xs)-1}', 365, 18, .42)
    return frame


def render_one(job: tuple[dict, str, dict]) -> tuple[str, int, str]:
    row, output, scale_map = job
    cv2.setNumThreads(1)
    rgb_path = Path(row['video'])
    folder = rgb_path.parent
    fps, rgb_frames = video_info(rgb_path)
    with np.load(folder / 'inputs.npz') as inputs:
        initial = inputs['force_local'][0]
        post = inputs['force_local_post_step']
        steps = inputs['steps']
        base_snapshots = inputs['force_base']
    if initial.shape != (2, 10, 14, 3) or post.shape[1:] != initial.shape:
        raise ValueError(f'Unexpected tactile grid shape: {folder}')
    local = np.concatenate((initial[None], post), axis=0)
    if len(local) != rgb_frames:
        raise ValueError(f'RGB/tactile frame mismatch {folder}: {rgb_frames}/{len(local)}')
    if len(steps) != len(base_snapshots) or steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError(f'Invalid base force sampling times: {folder}')
    snapshot_indices = np.searchsorted(steps, np.arange(rgb_frames), side='right') - 1
    base_force = base_snapshots[snapshot_indices].reshape(-1, 2, 10, 14, 3)
    if not np.isfinite(local).all() or not np.isfinite(base_force).all():
        raise ValueError(f'Nonfinite tactile values: {folder}')
    statuses = []
    for mode, forces in (('base', base_force), ('local', local)):
        target = (Path(output) / 'failure_tactile' / mode / row['model'] /
                  f"seed_{row['seed']}.mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                tactile_fps, tactile_frames = video_info(target)
                if tactile_fps == fps and tactile_frames == rgb_frames:
                    statuses.append('cached')
                    continue
            except (ValueError, KeyError, subprocess.CalledProcessError):
                pass
        limits = np.asarray(scale_map[mode], np.float32)
        background, xs = curve_background(forces, limits, row['first_contact_step'], mode)
        temporary = target.with_name(target.stem + '.partial.mp4')
        encoder = subprocess.Popen(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}',
             '-r', fps, '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
             '-crf', '23', '-threads', '1', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        try:
            assert encoder.stdin is not None
            for step, force in enumerate(forces):
                image = draw_frame(background, force, limits, step, xs, mode)
                encoder.stdin.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes())
        finally:
            if encoder.stdin is not None:
                encoder.stdin.close()
            code = encoder.wait()
        if code != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'ffmpeg failed for {target}')
        encoded_fps, encoded_frames = video_info(temporary)
        if encoded_fps != fps or encoded_frames != rgb_frames:
            temporary.unlink(missing_ok=True)
            raise ValueError(f'Encoded tactile video differs from RGB: {target}')
        os.replace(temporary, target)
        statuses.append('rendered')
    return row['model'], row['seed'], '/'.join(statuses)


def page(output: Path, rows: list[dict], counts: dict, by_model: dict,
         focus: str | None = None) -> None:
    if focus is not None:
        rows = sorted(rows, key=lambda row: (
            list(EVENT_NAMES).index(row['event']), row['seed']))
    entries = []
    cases = []
    last_event = None
    for row in rows:
        if focus is not None and row['event'] != last_event:
            event = row['event']
            entries.append('<tr class="group" id="group-%s"><th colspan="9">%s · %d 条</th></tr>' %
                           (event, EVENT_NAMES[event], counts[event]))
            last_event = event
        visual = os.path.relpath(row['video'], output)
        tactile_base = f"failure_tactile/base/{row['model']}/seed_{row['seed']}.mp4"
        tactile_local = f"failure_tactile/local/{row['model']}/seed_{row['seed']}.mp4"
        cases.append(dict(model=row['model'], seed=row['seed'], event=EVENT_NAMES[row['event']],
                          visual=visual, tactile_base=tactile_base,
                          tactile_local=tactile_local,
                          first_contact=row['first_contact_step'],
                          first_success=row['first_success_step']))
        entries.append('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td>'
                       '<td>%.1f</td><td>%.1f</td><td>%.1f</td>'
                       '<td><button class="watch" data-case="%d">双视图＋触觉</button> '
                       '<a href="%s">原视频</a></td></tr>' %
                       (html.escape(row['model']), row['seed'], EVENT_NAMES[row['event']],
                        row['first_contact_step'] if row['first_contact_step'] is not None else '—',
                        row['observed_stall_proxy'], row['best_xy_error_mm'],
                        row['best_socket_z_mm'], row['best_axis_error_deg'],
                        len(cases) - 1, html.escape(visual, quote=True)))
    model_rows = ''.join('<tr><td>%s</td><td>%s</td></tr>' %
                         (html.escape(k), html.escape(str(v))) for k, v in by_model.items())
    if focus is None:
        heading = '原 13 模型 × 40 初态：失败现象'
        nav = '<p><a href="failure_audit_by_seed.html">按初态比较 13 个模型</a></p>'
        nav += '<p>按模型查看：' + ' · '.join(
            f'<a href="failure_audit_model{number}.html">{number} {html.escape(description)}</a>'
            for number, (description, _) in FOCUS_MODELS.items()) + '</p>'
        subtitle = '全部轨迹'
    else:
        description, _ = FOCUS_MODELS[focus]
        heading = f'模型 {focus}：{description} · 40 个初态'
        count = len(rows)
        ever = counts.get('final_success', 0) + counts.get('transient_success', 0)
        final = counts.get('final_success', 0)
        numbers = tuple(FOCUS_MODELS)
        position = numbers.index(focus)
        previous = numbers[(position - 1) % len(numbers)]
        following = numbers[(position + 1) % len(numbers)]
        nav = (f'<p>曾通过成功判据 {ever}/{count}（{ever / count:.1%}）；'
               f'最终通过 {final}/{count}（{final / count:.1%}）。 '
               f'<a href="failure_audit.html">返回 520 条总表</a> · '
               f'<a href="failure_audit_by_seed.html">按初态比较 13 个模型</a> · '
               f'<a href="failure_audit_model{previous}.html">上一模型 {previous}</a> · '
               f'<a href="failure_audit_model{following}.html">下一模型 {following}</a></p>')
        nav += '<details><summary>切换到其他模型</summary><p>' + ' · '.join(
            f'<a href="failure_audit_model{number}.html">{number} {html.escape(label)}</a>'
            for number, (label, _) in FOCUS_MODELS.items() if number != focus) + '</p></details>'
        nav += '<p>直接跳到：' + ' · '.join(
            f'<a href="#group-{event}">{title} {counts[event]}</a>'
            for event, title in EVENTS if counts.get(event)) + '</p>'
        subtitle = '按结果分组的 40 条轨迹'
    data = json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>__TITLE__</title>
<style>
body{font:15px/1.5 system-ui;background:#142032;color:#eaf2fc;max-width:1500px;margin:auto;padding:12px}
a{color:#9ad5ff}td,th{border-bottom:1px solid #466079;padding:5px;text-align:left}
button{cursor:pointer;background:#365a7d;color:#fff;border:1px solid #7694b4;border-radius:5px;padding:5px 9px}
dialog{background:#17283d;color:#eaf2fc;border:1px solid #7290ad;border-radius:9px;width:min(94vw,1100px);max-height:92vh;overflow:auto}
dialog::backdrop{background:rgba(0,0,0,.79)}
.media{display:grid;grid-template-columns:1fr 1fr;gap:12px}.media video{width:100%;display:block;background:#090e17}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
tr.group th{background:#294561;border-top:2px solid #81a2bf;padding:12px;font-size:1.08em}
@media(max-width:760px){.media{grid-template-columns:1fr}}
</style>
<h1>__HEADING__</h1>
<p>来源：原始 trajectory 和动作前力图；接触力是 USB—插座仿真记录。末态、曾成功与接触后未成功分开计数。
“夹爪下移而 USB 少动”仅是症状。__INTRO_NOTE__</p>
__NAV__
<p>点击「双视图＋触觉」逐帧同步观看。可切换机器人 base Fx/Fy/Fz（策略重规划时记录，每 4 步一次）与传感器局部 Fx/Fy/Fz（每个控制步之后记录）。
两种坐标各有跨全部案例固定的颜色标尺；视频第 0 帧使用初始观测。竖线标记 USB—插座首次检测到接触的时间；
触觉图反映指尖与 USB 的接触，不是插座接触力。所有模型都有诊断记录；是否作为策略输入取决于该模型配置。</p>
<p>类别计数：__COUNTS__</p><table><tr><th>模型</th><th>轨迹计数</th></tr>__MODELS__</table>
<h2>__SUBTITLE__</h2><table><tr><th>模型</th><th>seed</th><th>事件</th><th>首次接触步</th>
<th>停滞症状</th><th>最佳 XY mm</th><th>最佳 Z mm</th><th>最佳轴误差 °</th><th>视频</th></tr>__ROWS__</table>
<dialog id="viewer"><div class="bar"><strong id="caseTitle"></strong><button id="closeViewer">关闭</button>
<button id="firstContact">跳到首次接触</button><button id="firstSuccess">跳到首次成功</button>
<select id="forceMode"><option value="base">机器人 base Fx/Fy/Fz（每 4 步）</option><option value="local">传感器局部 Fx/Fy/Fz（每步）</option></select>
<span id="frameStatus"></span></div>
<div class="media"><section><b>前视｜腕视（播放与拖动这里）</b><video id="visual" controls preload="metadata" playsinline></video></section>
<section><b>双指 Fx/Fy/Fz 与强度变化</b><video id="tactile" muted preload="metadata" playsinline></video></section></div>
<p>红色为正、蓝色为负；六张图使用固定的通道色标。下方曲线是相对于固定色标的平均力强度。
网页以视觉视频为时间主轴，同步触觉视频。可暂停、单步拖动并在同一位置比较两个画面。</p></dialog>
<script>
const cases=__CASES__;
const dlg=document.querySelector('#viewer'),visual=document.querySelector('#visual'),tactile=document.querySelector('#tactile');
const title=document.querySelector('#caseTitle'),status=document.querySelector('#frameStatus');
const forceMode=document.querySelector('#forceMode');
let active=null;
function synchronize(force=false){
  if(!active)return;
  if(force || Math.abs(tactile.currentTime-visual.currentTime)>0.07)tactile.currentTime=visual.currentTime;
  status.textContent=`时间 ${visual.currentTime.toFixed(2)} s / ${visual.duration ? visual.duration.toFixed(2) : '…'} s`;
}
function setForceMode(){
  if(!active)return;
  tactile.pause();tactile.src=active['tactile_'+forceMode.value];tactile.load();
  tactile.addEventListener('loadedmetadata',()=>{
    synchronize(true);if(!visual.paused)tactile.play().catch(()=>{});
  },{once:true});
}
document.querySelectorAll('.watch').forEach(button=>button.addEventListener('click',()=>{
  visual.pause();tactile.pause();active=cases[Number(button.dataset.case)];
  title.textContent=`${active.model} · seed ${active.seed} · ${active.event}`;
  visual.src=active.visual;forceMode.value='base';setForceMode();
  document.querySelector('#firstContact').disabled=active.first_contact===null;
  document.querySelector('#firstSuccess').disabled=active.first_success===null;
  dlg.showModal();visual.load();
}));
document.querySelector('#closeViewer').onclick=()=>dlg.close();
dlg.addEventListener('close',()=>{visual.pause();tactile.pause();visual.removeAttribute('src');tactile.removeAttribute('src');
  visual.load();tactile.load();active=null;});
visual.addEventListener('play',()=>{synchronize(true);tactile.play().catch(()=>{});});
visual.addEventListener('pause',()=>tactile.pause());
visual.addEventListener('seeking',()=>synchronize(true));
visual.addEventListener('seeked',()=>synchronize(true));
visual.addEventListener('timeupdate',()=>synchronize());
visual.addEventListener('ratechange',()=>{tactile.playbackRate=visual.playbackRate;});
visual.addEventListener('ended',()=>tactile.pause());
forceMode.addEventListener('change',setForceMode);
function jump(step){if(active&&step!==null&&visual.duration){
  const fps=(visual.videoWidth ? (window.currentFrameRate||15) : 15);
  visual.currentTime=Math.min(visual.duration,Math.max(0,(step+1)/fps));synchronize(true);
}}
document.querySelector('#firstContact').onclick=()=>jump(active?.first_contact);
document.querySelector('#firstSuccess').onclick=()=>jump(active?.first_success);
</script></html>'''
    intro_note = ('下表每行是一次模型评测，不是独立环境初态。' if focus is None
                  else '下表每行对应一个不同的测试初态。')
    content = (content.replace('__TITLE__', html.escape(heading))
              .replace('__INTRO_NOTE__', intro_note)
              .replace('__HEADING__', html.escape(heading))
              .replace('__NAV__', nav).replace('__SUBTITLE__', html.escape(subtitle))
              .replace('__COUNTS__', html.escape(str(counts)))
              .replace('__MODELS__', model_rows).replace('__ROWS__', ''.join(entries))
              .replace('__CASES__', data))
    stem = 'failure_audit' if focus is None else f'failure_audit_model{focus}'
    target = output / f'{stem}.html'
    temporary = output / f'{stem}.partial.html'
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, target)


def seed_page(output: Path, rows: list[dict]) -> None:
    """Compare all policies from the same initial state on one synchronized page."""
    model_ids = {name: number for number, (_, name) in FOCUS_MODELS.items()}
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row['model'] not in model_ids:
            raise ValueError(f"Unknown model: {row['model']}")
        grouped.setdefault(int(row['seed']), []).append(row)
    sections = []
    cases = {}
    for seed, group in sorted(grouped.items()):
        group.sort(key=lambda row: int(model_ids[row['model']]))
        if len(group) != len(FOCUS_MODELS) or len({row['model'] for row in group}) != len(group):
            raise ValueError(f'Seed {seed} must contain one row for each of the 13 models')
        counts = Counter(row['event'] for row in group)
        cases[str(seed)] = [dict(
            id=model_ids[row['model']], label=FOCUS_MODELS[model_ids[row['model']]][0],
            model=row['model'], event=row['event'],
            visual=os.path.relpath(row['video'], output),
            tactile_base=f"failure_tactile/base/{row['model']}/seed_{seed}.mp4",
            tactile_local=f"failure_tactile/local/{row['model']}/seed_{seed}.mp4",
            first_contact=row['first_contact_step'], first_success=row['first_success_step'],
            best_xy=row['best_xy_error_mm'], best_z=row['best_socket_z_mm'],
            best_axis=row['best_axis_error_deg'], stall=row['observed_stall_proxy'],
        ) for row in group]
        strip = ''.join(
            f'<span class="chip {html.escape(row["event"])}" title="{html.escape(EVENT_NAMES[row["event"]], quote=True)}">'
            f'{model_ids[row["model"]]}</span>' for row in group)
        sections.append(
            f'<details class="seed" data-seed="{seed}" id="seed-{seed}">'
            f'<summary><b>测试初态 {seed}</b> · 最终成功 {counts["final_success"]} / 13 · '
            f'曾成功后失败 {counts["transient_success"]} · '
            f'从未成功 {counts["contact_no_success"] + counts["no_detected_contact"]}'
            f'<span class="strip">{strip}</span></summary>'
            '<div class="seed-content"></div></details>')
    data = json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')
    content = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>40 个初态 × 13 个模型：视频与触觉对比</title>
<style>
body{font:15px/1.5 system-ui;background:#142032;color:#eaf2fc;max-width:1700px;margin:auto;padding:12px}
a{color:#9ad5ff}button,select{cursor:pointer;background:#365a7d;color:#fff;border:1px solid #7694b4;border-radius:5px;padding:5px 9px}
button:disabled{opacity:.4;cursor:default}summary{cursor:pointer;padding:12px;background:#243b55;position:sticky;top:0;z-index:2}
details.seed{border:1px solid #496581;border-radius:8px;margin:12px 0;background:#182a40}details.seed[open]{border-color:#92b6d6}
.strip{display:inline-flex;gap:3px;margin-left:15px;vertical-align:middle}.chip{font-weight:700;font-size:12px;min-width:21px;text-align:center;border-radius:3px;padding:1px 3px}
.final_success{background:#176b53}.transient_success{background:#805c22}.contact_no_success,.no_detected_contact{background:#763e4a}
.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:10px;border-bottom:1px solid #496581;background:#20344c;position:sticky;top:50px;z-index:1}
.controls input[type=range]{flex:1;min-width:180px}.row{border-bottom:1px solid #496581;padding:13px}.row:last-child{border-bottom:0}
.heading{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.heading h3{margin:0}.badge{border-radius:5px;padding:2px 7px}
.metrics{font-size:13px;color:#b6c9dc;margin:5px 0 9px}.media{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:10px}
.media video{width:100%;display:block;background:#080d15}.media small{display:block;margin-bottom:4px;color:#b6c9dc}
.row-controls{display:flex;gap:6px;flex-wrap:wrap;margin:7px 0}.status{color:#bbd9ee;font-size:13px}
@media(max-width:850px){.media{grid-template-columns:1fr}.controls{top:40px}.strip{display:flex;margin:4px 0 0}}
</style>
<h1>同一初态：13 个模型逐例对比</h1>
<p>共 40 个测试初态，每个初态包含 13 条策略轨迹。绿色表示最终成功，黄色表示曾通过成功判据但末态失败，红色表示从未成功。
打开一个初态即可同时查看 13 个模型的前视／腕视、双指六张 Fx/Fy/Fz 图和结果。只加载当前打开的初态，以减少浏览器负担。</p>
<p><a href="failure_audit.html">返回 520 条总表</a> · <a href="index.html">动作效果实验首页</a></p>
<p>触觉视频是所有模型的诊断记录，是否作为策略输入取决于配置。机器人 base 力图每 4 步记录一次并保持；传感器局部力图每控制步记录。
红色为正、蓝色为负，各通道使用固定色标。触觉来自 USB 与指尖之间的接触；首次接触标记来自 USB 与插座的独立仿真记录。
“曾成功”与“最终成功”是两种不同结果。播放统一以 RGB 视频为时间基准；单条轨迹结束后保持末帧。</p>
__SECTIONS__
<script>
const cases=__CASES__;
const names={final_success:'最终成功',transient_success:'曾成功、末态失败',contact_no_success:'从未成功：检测到插座接触',no_detected_contact:'从未成功：未检测到插座接触'};
const FPS=15, MAX_FRAME=180;
let active=null;
function el(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n}
function button(label,fn){const b=el('button','',label);b.type='button';b.onclick=fn;return b}
function seekPair(pair,frame){const t=Math.max(0,Math.min(MAX_FRAME,frame))/FPS;
  for(const v of [pair.visual,pair.tactile]){
    const seek=()=>{v.currentTime=Math.min(t,Math.max(0,(v.duration||t)-.001));};
    if(v.readyState>=1)seek();else v.addEventListener('loadedmetadata',seek,{once:true});}}
function pausePair(pair){pair.visual.pause();pair.tactile.pause()}
function syncPair(pair,force=false){const a=pair.visual,b=pair.tactile;
  if(b.readyState>=1&&a.readyState>=1&&(force||Math.abs(a.currentTime-b.currentTime)>.075))b.currentTime=a.currentTime;
  pair.status.textContent=`帧 ${Math.min(MAX_FRAME,Math.round(a.currentTime*FPS))} / ${MAX_FRAME}`;}
function changeMode(ctx){for(const pair of ctx.pairs){const t=pair.visual.currentTime,playing=!pair.visual.paused;
  pair.tactile.pause();pair.tactile.src=pair.rec['tactile_'+ctx.mode.value];pair.tactile.load();
  pair.tactile.addEventListener('loadedmetadata',()=>{pair.tactile.currentTime=Math.min(t,pair.tactile.duration-.001);
    if(playing)pair.tactile.play().catch(()=>{});},{once:true});}}
function unload(d){if(!d._ctx)return;for(const pair of d._ctx.pairs){pausePair(pair);
  pair.visual.removeAttribute('src');pair.tactile.removeAttribute('src');pair.visual.load();pair.tactile.load();}
  d.querySelector('.seed-content').replaceChildren();d._ctx=null;if(active===d)active=null;}
function render(d){if(d._ctx)return;const host=d.querySelector('.seed-content');const ctx={pairs:[],mode:null,slider:null,allPlaying:false};d._ctx=ctx;active=d;
  const controls=el('div','controls');host.append(controls);
  controls.append(button('同步播放',()=>{ctx.allPlaying=true;for(const p of ctx.pairs)p.visual.play().catch(()=>{});}));
  controls.append(button('全部暂停',()=>{ctx.allPlaying=false;for(const p of ctx.pairs)pausePair(p);}));
  controls.append(button('◀ 上一帧',()=>stepAll(-1)));controls.append(button('下一帧 ▶',()=>stepAll(1)));
  const slider=el('input');slider.type='range';slider.min='0';slider.max=String(MAX_FRAME);slider.value='0';ctx.slider=slider;
  slider.oninput=()=>{ctx.allPlaying=false;for(const p of ctx.pairs){pausePair(p);seekPair(p,Number(slider.value));}};controls.append(slider);
  const frameText=el('span','status','共同帧 0 / 180');controls.append(frameText);
  const mode=el('select');mode.innerHTML='<option value="base">机器人 base 力图（每 4 步）</option><option value="local">传感器局部力图（每步）</option>';
  mode.onchange=()=>changeMode(ctx);ctx.mode=mode;controls.append(mode);
  function stepAll(delta){ctx.allPlaying=false;const frame=Math.max(0,Math.min(MAX_FRAME,Number(slider.value)+delta));
    slider.value=String(frame);frameText.textContent=`共同帧 ${frame} / ${MAX_FRAME}`;
    for(const p of ctx.pairs){pausePair(p);seekPair(p,frame);}}
  for(const rec of cases[d.dataset.seed]){const row=el('article','row');host.append(row);
    const head=el('div','heading');row.append(head);head.append(el('h3','',`${rec.id} · ${rec.label}`));
    head.append(el('span','badge '+rec.event,names[rec.event]));
    const metrics=el('div','metrics',`首次插座接触 ${rec.first_contact??'—'} 步 · 首次成功 ${rec.first_success??'—'} 步 · `+
      `停滞症状 ${rec.stall?'有':'无'} · 最佳 XY ${Number(rec.best_xy).toFixed(1)} mm · 最佳 Z ${Number(rec.best_z).toFixed(1)} mm · 最佳轴误差 ${Number(rec.best_axis).toFixed(1)}°`);
    row.append(metrics);
    const pair={rec,status:el('span','status','帧 0 / 180')};ctx.pairs.push(pair);
    const rowControls=el('div','row-controls');row.append(rowControls);
    rowControls.append(button('播放本模型',()=>{ctx.allPlaying=false;for(const p of ctx.pairs)if(p!==pair)pausePair(p);pair.visual.play().catch(()=>{});}));
    rowControls.append(button('暂停',()=>pausePair(pair)));
    const contact=button('跳到首次接触',()=>seekPair(pair,rec.first_contact+1));contact.disabled=rec.first_contact===null;rowControls.append(contact);
    const success=button('跳到首次成功',()=>seekPair(pair,rec.first_success+1));success.disabled=rec.first_success===null;rowControls.append(success);
    rowControls.append(pair.status);
    const media=el('div','media');row.append(media);
    for(const [title,key] of [['前视｜腕视','visual'],['双指 Fx/Fy/Fz 与力强度','tactile']]){
      const box=el('div');box.append(el('small','',title));const v=el('video');v.preload='metadata';v.playsInline=true;
      v.controls=key==='visual';v.muted=key==='tactile';v.src=key==='visual'?rec.visual:rec.tactile_base;
      box.append(v);media.append(box);pair[key]=v;}
    const v=pair.visual;v.addEventListener('play',()=>{syncPair(pair,true);pair.tactile.play().catch(()=>{});});
    v.addEventListener('pause',()=>pair.tactile.pause());v.addEventListener('seeking',()=>syncPair(pair,true));
    v.addEventListener('seeked',()=>syncPair(pair,true));v.addEventListener('ratechange',()=>{pair.tactile.playbackRate=v.playbackRate;});
    v.addEventListener('timeupdate',()=>{syncPair(pair);if(ctx.allPlaying){const master=ctx.pairs[0].visual;
      if(pair===ctx.pairs[0]){const frame=Math.min(MAX_FRAME,Math.round(master.currentTime*FPS));slider.value=String(frame);frameText.textContent=`共同帧 ${frame} / ${MAX_FRAME}`;}
      else if(master.readyState>=1&&Math.abs(v.currentTime-master.currentTime)>.15&&v.readyState>=1)v.currentTime=Math.min(master.currentTime,v.duration-.001);}});
    v.addEventListener('ended',()=>pair.tactile.pause());
  }
}
for(const d of document.querySelectorAll('details.seed'))d.addEventListener('toggle',()=>{
  if(d.open){for(const other of document.querySelectorAll('details.seed'))if(other!==d&&other.open)other.open=false;render(d);}
  else unload(d);});
</script></html>'''
    content = content.replace('__SECTIONS__', '\n'.join(sections)).replace('__CASES__', data)
    target = output / 'failure_audit_by_seed.html'
    temporary = output / 'failure_audit_by_seed.partial.html'
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--output', type=Path, default=REPORT)
    parser.add_argument('--scales', type=Path, default=SCALE_MANIFEST)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Smoke check only; writes a partial report')
    parser.add_argument('--pages-only', action='store_true',
                        help='Reuse existing tactile videos and regenerate the report pages')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.report / 'failure_audit.json').read_text())
    rows = report['rows'][:args.limit or None]
    if not rows:
        raise ValueError('No audited trajectories')
    if args.pages_only:
        missing = [str(args.output / 'failure_tactile' / mode / row['model'] /
                       f"seed_{row['seed']}.mp4")
                   for row in rows for mode in ('base', 'local')
                   if not (args.output / 'failure_tactile' / mode / row['model'] /
                           f"seed_{row['seed']}.mp4").is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} tactile videos missing, first: {missing[0]}')
    else:
        saved_scales = json.loads(args.scales.read_text())['scales']
        scale_map = {mode: saved_scales[mode] for mode in ('base', 'local')}
        jobs = [(row, str(args.output), scale_map) for row in rows]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, (model, seed, status) in enumerate(pool.map(render_one, jobs), 1):
                if index == 1 or index % 20 == 0 or index == len(jobs):
                    print(f'TACTILE {index}/{len(jobs)} {model} seed={seed} {status}', flush=True)
    by_model = {model: dict(Counter(row['event'] for row in rows if row['model'] == model))
                for model in sorted({row['model'] for row in rows})}
    page(args.output, rows, dict(Counter(row['event'] for row in rows)), by_model)
    print(f'REPORT {args.output / "failure_audit.html"}', flush=True)
    for focus, (_, model) in FOCUS_MODELS.items():
        selected = [row for row in rows if row['model'] == model]
        if not selected:
            continue
        counts = dict(Counter(row['event'] for row in selected))
        page(args.output, selected, counts, {model: counts}, focus=focus)
        print(f'FOCUS {focus} {len(selected)} {counts} '
              f'{args.output / f"failure_audit_model{focus}.html"}', flush=True)
    if not args.limit:
        seed_page(args.output, rows)
        print(f'SEEDS {len({row["seed"] for row in rows})} '
              f'{args.output / "failure_audit_by_seed.html"}', flush=True)


if __name__ == '__main__':
    main()
