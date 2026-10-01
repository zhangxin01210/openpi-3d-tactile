#!/usr/bin/env python3
"""Audit released ContactWorld Zarrs and render deterministic, complete demo samples.

Only reads data. No Isaac Gym, torch, learned model, or dataset modifications.
Frame indices are authoritative; playback at 10 fps follows the paper, not timestamps.
Contact/recovery candidates are deliberately not labeled ground-truth events.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import zarr

W, H = 384, 320
COLORS = [(90, 160, 255), (120, 230, 130), (255, 170, 110)]


def text(image, value, x=10, y=22, color=(235, 235, 235), scale=.48):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def blank(title):
    out = np.full((H, W, 3), 22, np.uint8)
    text(out, title)
    return out


def rgb_panel(a, title):
    if a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.dtype != np.uint8:
        a = np.uint8(np.clip(a * 255, 0, 255))
    a = a[..., :3][..., ::-1]  # raw Zarr stores RGB; OpenCV needs BGR
    scale = min(W / a.shape[1], (H - 35) / a.shape[0])
    a = cv2.resize(a, (round(a.shape[1]*scale), round(a.shape[0]*scale)))
    out = blank(title)
    x, y = (W-a.shape[1])//2, 33
    out[y:y+a.shape[0], x:x+a.shape[1]] = a
    return out


def chart(values, i, title, labels, limits=None, markers=()):
    out = blank(title)
    values = np.asarray(values).reshape(len(values), -1)
    lo, hi = limits if limits is not None else (float(values.min()), float(values.max()))
    if hi-lo < 1e-8:
        lo, hi = lo-1, hi+1
    pad = .06*(hi-lo)
    lo, hi = lo-pad, hi+pad
    x = np.linspace(48, W-12, len(values))
    for v in np.linspace(lo, hi, 5):
        yy = int(265-(v-lo)/(hi-lo)*203)
        cv2.line(out, (48, yy), (W-12, yy), (55, 55, 55), 1)
        text(out, f"{v:.2g}", 2, yy+3, scale=.36)
    for c in range(values.shape[1]):
        y = 265-(values[:, c]-lo)/(hi-lo)*203
        points = np.stack([x, y], -1).astype(np.int32)
        cv2.polylines(out, [points], False, COLORS[c % 3], 1, cv2.LINE_AA)
        text(out, labels[c], 50+c*105, 45, COLORS[c % 3], .38)
    for m in markers:
        cv2.line(out, (int(x[m]), 60), (int(x[m]), 267), (90, 90, 130), 1)
    cv2.line(out, (int(x[i]), 60), (int(x[i]), 267), (245, 245, 245), 1)
    text(out, f"frame 0                             {len(values)-1}", 48, 285, scale=.38)
    text(out, "now " + ", ".join(f"{v:.3g}" for v in values[i]), 48, 309, scale=.40)
    return out


def force_panel(ff, normal_scale, shear_scale):
    out = blank("Right TacFF: signed normal + shear")
    a = cv2.applyColorMap(np.uint8(np.clip((ff[..., 0]/normal_scale+1)*127.5, 0, 255)), cv2.COLORMAP_TURBO)
    out[48:248, 52:332] = cv2.resize(a, (280, 200), interpolation=cv2.INTER_NEAREST)
    for row in range(ff.shape[0]):
        for col in range(ff.shape[1]):
            start = (62+20*col, 58+20*row)
            # Raw components along field columns/rows; physical tangent basis unverified.
            vec = np.clip(ff[row, col, 1:]/shear_scale, -1, 1)*12
            end = (round(start[0]+vec[0]), round(start[1]+vec[1]))
            cv2.arrowedLine(out, start, end, (255, 255, 255), 1, tipLength=.3)
    text(out, f"color +/-{normal_scale:.3g}; arrows / {shear_scale:.3g}", 15, 278, scale=.42)
    text(out, "Fixed task scale; sensor field axes", 15, 302, scale=.42)
    return out


def pc_panel(pc, bounds):
    out = blank("Stored point cloud (frame unverified)")
    # Fixed orthographic view, no fitted transformation or auto-centering per frame.
    basis = np.array([[.7071, -.7071, 0], [-.35, -.35, .866]])
    xy = pc[:, :3] @ basis.T
    low, high = bounds
    center = (low+high)/2
    factor = min((W-35)/max(high[0]-low[0], 1e-6), 235/max(high[1]-low[1], 1e-6))
    uv = (xy-center)*factor*np.array([1, -1]) + [W/2, 158]
    colors = np.uint8(np.clip(pc[:, 3:6]*255, 0, 255))[:, ::-1]
    for index in np.argsort(pc[:, 0]+pc[:, 1]):
        u, v = np.round(uv[index]).astype(int)
        if 0 <= u < W and 33 <= v < 283:
            cv2.circle(out, (u, v), 1, colors[index].tolist(), -1)
    text(out, f"{len(pc)} XYZRGB points; fixed view", 10, 305, scale=.44)
    return out


def analyze_episode(d, episode_id, start, end):
    a = d["action"][start:end].reshape(end-start, -1)
    plug = d["plug_pos"][start:end]
    socket = d["socket_pos_gt"][start:end]
    ff = d["tactile_force_field_right"][start:end]
    xy = np.linalg.norm(plug[:, :2]-socket[:, :2], axis=-1)*1000
    height = (plug[:, 2]-socket[:, 2])*1000
    force = np.stack([ff[..., 0].sum((1, 2)), np.linalg.norm(ff[..., 1:], axis=-1).sum((1, 2))], -1)
    # Motion-only candidates: positive z retreat >0.5 mm followed by >0.5 mm descent
    # in the next 12 frames; require this is in second half of episode.
    dz = np.diff(plug[:, 2])*1000
    retreats = [int(i+1) for i in range(len(dz)) if i >= len(dz)//2 and dz[i] > .5
                and np.min(plug[i+1:min(i+14, len(plug)), 2]) < plug[i+1, 2]-.0005]
    metric = {
        "episode": int(episode_id), "start": int(start), "end": int(end), "frames": int(end-start),
        "initial_socket_xyz": socket[0].tolist(), "initial_xy_error_mm": float(xy[0]),
        "final_xy_error_mm": float(xy[-1]), "initial_tip_minus_socket_base_z_mm": float(height[0]),
        "final_tip_minus_socket_base_z_mm": float(height[-1]),
        "peak_normal_frame": int(force[:, 0].argmax()), "retreat_candidates": retreats,
        "first_plug_socket_contact_frame": None, "recovery_confirmed": None,
        "success_label": None,
        "event_caveat": "TacFF is finger/plug contact; retreats are motion candidates, not verified collision/recovery.",
    }
    return metric, a, xy, height, force, ff


def video_task(root_path, output, count, fps, encoder):
    root = zarr.open_group(str(root_path), mode="r")
    d = root["data"]
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.r_[0, ends[:-1]]
    assert len(ends) and np.all(np.diff(np.r_[0, ends]) > 0)
    required = {"front", "wrist", "pointcloud", "tactile_force_field_right", "plug_pos", "socket_pos_gt", "action"}
    assert required <= set(d.keys()), sorted(required-set(d.keys()))
    assert all(x.shape[0] == ends[-1] for _, x in d.arrays())
    chosen = np.linspace(0, len(ends)-1, min(count, len(ends)), dtype=int)
    schema = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "chunks": list(v.chunks),
                  "attrs": dict(v.attrs)} for k, v in d.arrays()}
    task = root_path.name
    output.mkdir(parents=True, exist_ok=True)
    metrics = [analyze_episode(d, i, s, e)[0] for i, (s, e) in enumerate(zip(starts, ends))]
    socket_initial = np.asarray([m["initial_socket_xyz"] for m in metrics])
    cells = np.clip(np.floor((socket_initial[:, :2]-[.45, -.05])/.1*3), 0, 2).astype(int)
    outside_grid = np.any((socket_initial[:, :2] < [.45, -.05]) | (socket_initial[:, :2] > [.55, .05]), axis=1)
    grid = np.zeros((3, 3), int)
    for x, y in cells:
        grid[y, x] += 1
    ffields = np.concatenate([d["tactile_force_field_right"][starts[i]:ends[i]] for i in chosen])
    normal_scale = max(float(np.quantile(np.abs(ffields[..., 0]), .995)), 1e-9)
    shear_scale = max(float(np.quantile(np.linalg.norm(ffields[..., 1:], axis=-1), .995)), 1e-9)
    pc_samples = np.concatenate([d["pointcloud"][int(s)] for s in starts])
    finite = np.isfinite(pc_samples).all(-1)
    if not finite.all():
        raise ValueError("Nonfinite point cloud at episode starts")
    basis = np.array([[.7071, -.7071, 0], [-.35, -.35, .866]])
    pc_xy = pc_samples[:, :3] @ basis.T
    bounds = (pc_xy.min(0), pc_xy.max(0))
    report = {
        "task": task, "source": str(root_path), "episodes": len(ends), "frames": int(ends[-1]),
        "schema": schema, "root_attrs": dict(root.attrs), "meta_keys": list(root["meta"].keys()),
        "selected_episodes": chosen.tolist(), "selection": "linspace over full episode index range; no success/cherry-pick filter",
        "fps": fps, "fps_caveat": "Playback follows paper (10 Hz); no per-frame timestamps verified.",
        "initial_socket_xyz_min": socket_initial.min(0).tolist(), "initial_socket_xyz_max": socket_initial.max(0).tolist(),
        "grid_counts_yx_nominal_10cm_square": grid.tolist(),
        "initial_sockets_outside_nominal_square": int(outside_grid.sum()),
        "pc_initial_xyz_min": pc_samples[:, :3].min(0).tolist(), "pc_initial_xyz_max": pc_samples[:, :3].max(0).tolist(),
        "tacff_selected_min": ffields.min((0, 1, 2)).tolist(), "tacff_selected_max": ffields.max((0, 1, 2)).tolist(),
        "normal_display_scale": normal_scale, "shear_display_scale": shear_scale,
        "all_episode_metrics": metrics, "videos": [],
    }
    all_actions = np.asarray(d["action"][:])
    report["action_min"] = all_actions.min(0).tolist()
    report["action_max"] = all_actions.max(0).tolist()
    report["zarr_group_and_array_attributes"] = {
        str(p.relative_to(root_path)): json.loads(p.read_text()) for p in root_path.rglob(".zattrs")
    }
    action_limit = max(1., float(np.abs(all_actions).max()))
    tactile_key = next((k for k in ("tactile_rgb_right", "right_tactile_camera_taxim", "tactile_img_right") if k in d), None)
    for ep in chosen:
        start, end = int(starts[ep]), int(ends[ep])
        metric, a, xy, height, force, ff = analyze_episode(d, ep, start, end)
        # Load only this episode's RGB; retain complete trajectory.
        front, wrist, pc = (np.asarray(d[k][start:end]) for k in ("front", "wrist", "pointcloud"))
        tactile = np.asarray(d[tactile_key][start:end]) if tactile_key else None
        assert all(np.isfinite(x).all() for x in (a, xy, height, force, ff, pc))
        video = output / f"episode_{ep:04d}.mp4"
        temp = video.with_suffix(".part.mp4")
        options = {"av1_nvenc": ["-c:v", "av1_nvenc", "-preset", "p4", "-cq", "27"],
                   "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]}[encoder]
        command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{3*W}x{3*H}",
                   "-r", str(fps), "-i", "pipe:0", "-an", *options, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        story_ids = np.unique(np.r_[np.linspace(0, end-start-1, 7, dtype=int), metric["peak_normal_frame"]])
        story = []
        try:
            for i in range(end-start):
                parts = [rgb_panel(front[i], f"{task} ep{ep} f{i}/{end-start-1}: front"),
                         rgb_panel(wrist[i], "Wrist RGB"), pc_panel(pc[i], bounds),
                         rgb_panel(tactile[i], "Right tactile RGB") if tactile is not None else blank("No tactile RGB field"),
                         force_panel(ff[i], normal_scale, shear_scale),
                         chart(np.stack([xy, height], -1), i, "Tip vs socket base (mm)", ["XY error", "z difference"]),
                         chart(a[:, :3], i, "Position command (stored units)", ["dx", "dy", "dz"], (-action_limit, action_limit)),
                         chart(a[:, 3:6], i, "Rotation command (stored units)", ["rx", "ry", "rz"], (-action_limit, action_limit)),
                         chart(force, i, "Finger/plug TacFF sums (raw)", ["normal sum", "shear norms"], markers=metric["retreat_candidates"])]
                frame = np.vstack([np.hstack(parts[j:j+3]) for j in (0, 3, 6)])
                proc.stdin.write(frame.tobytes())
                if i == 0:
                    cv2.imwrite(str(output / f"episode_{ep:04d}.jpg"), frame)
                if i in story_ids:
                    story.append(np.vstack([parts[0], parts[4]]))
            proc.stdin.close()
            error = proc.stderr.read().decode(errors="replace")
            if proc.wait():
                raise RuntimeError(error)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stderr.close()
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(temp)]))["streams"][0]
        assert int(probe["nb_read_frames"]) == end-start and probe["pix_fmt"] == "yuv420p"
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(temp), "-f", "null", "-"], check=True)
        temp.replace(video)
        while len(story) < 8:
            story.append(np.zeros_like(story[0]))
        cv2.imwrite(str(output / f"episode_{ep:04d}_story.jpg"), np.vstack([np.hstack(story[:4]), np.hstack(story[4:8])]))
        with (output / f"episode_{ep:04d}.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "global_row", "action_dx", "action_dy", "action_dz", "action_rx", "action_ry", "action_rz",
                             "tip_xy_error_mm", "tip_minus_socket_base_z_mm", "tac_normal_sum", "tac_shear_norm_sum"])
            for i in range(end-start):
                writer.writerow([i, start+i, *a[i], xy[i], height[i], *force[i]])
        report["videos"].append({**metric, "file": video.name, "probe": probe, "full_decode_passed": True})
        print(f"{task} ep {ep}: {end-start} frames encoded and fully decoded", flush=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--encoder", choices=["av1_nvenc", "libx264"], default="av1_nvenc")
    parser.add_argument("--gallery-only", action="store_true", help="Rebuild HTML from completed per-task audit.json")
    parser.add_argument("--notes", type=Path, help="Optional manually reviewed frame ranges")
    args = parser.parse_args()
    if args.count < 1 or args.fps <= 0:
        parser.error("count and fps must be positive")
    if args.gallery_only:
        reports = [json.loads((args.output / t / "audit.json").read_text()) for t in ("insertion_usb", "insertion_peg")]
    else:
        reports = [video_task(args.data / t, args.output / t, args.count, args.fps, args.encoder)
                   for t in ("insertion_usb", "insertion_peg")]
    notes = json.loads(args.notes.read_text()) if args.notes else {"events": []}
    page = ['<!doctype html><meta charset="utf-8"><title>ContactWorld 数据审查</title>',
            '<style>body{background:#15191f;color:#e8edf4;font:17px sans-serif;margin:30px auto;max-width:1500px}video{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{background:#242b34;padding:16px}a{color:#8fc9ff}p{line-height:1.6}</style>',
            '<h1>ContactWorld USB / Peg：完整示范审查</h1><p>每任务按编号均匀抽取 10 条，完整帧序列。视频 AV1 / yuv420p。时间按论文 10 Hz 播放，帧编号为准。',
            '点云展示原始存储值，基座坐标系尚未验证；触觉是手指与插件的接触，不能标记首次插孔接触。白线为当前帧；力曲线竖线仅标记后半段抬起再下降的候选，不代表已确认恢复。',
            '相机内外参、深度、成功标签是否具备请查看审查报告。<a href="manifest.json">完整数值报告</a></p>']
    if notes["events"]:
        page.append('<h2>建议先看的片段</h2><p>下列判断来自分镜和轨迹，首次碰孔帧仍未知；不能用 TacFF 非零代替碰孔标签。</p><ul>')
        for event in notes["events"]:
            target = f'{event["task"]}_{event["episode"]:04d}'
            page.append(f'<li><a href="#{target}">{html.escape(event["title"])}：{event["task"]} / {event["episode"]}，帧 {event["frames"]}</a> — {html.escape(event["note"])}</li>')
        page.append('</ul>')
    if (args.output / 'geometry/geometry.json').exists():
        page.append('<p><a href="geometry/geometry.json">点云重投影检查</a> · <a href="geometry/insertion_usb_0111_middle.jpg">原始坐标与候选校正的对比图</a>。视频中的点云保留原始存储坐标，当前未直接当作已验证的基座坐标使用。</p>')
    for report in reports:
        task = report["task"]
        page.append(f'<h2>{task}：{report["episodes"]} 条 / {report["frames"]} 帧，抽取 {len(report["videos"])} 条</h2><div class="grid">')
        for v in report["videos"]:
            stem = Path(v["file"]).stem
            target = f'{task}_{v["episode"]:04d}'
            relevant = [e for e in notes["events"] if e["task"] == task and e["episode"] == v["episode"]]
            buttons = ''.join(f'<button onclick="this.closest(\'article\').querySelector(\'video\').currentTime={e["frames"][0]/report["fps"]}">跳到帧 {e["frames"][0]}</button>' for e in relevant)
            page.append(f'<article id="{target}"><h3>{stem} · {v["frames"]} 帧</h3><video controls preload="none" poster="{task}/{stem}.jpg" src="{task}/{v["file"]}"></video>{buttons}'
                        f'<p>初始横向误差 {v["initial_xy_error_mm"]:.1f} mm；末帧 {v["final_xy_error_mm"]:.1f} mm。'
                        f'法向力峰值帧 {v["peak_normal_frame"]}；抬起候选帧 {html.escape(str(v["retreat_candidates"]))}。</p>'
                        f'<a href="{task}/{stem}_story.jpg">分镜</a> · <a href="{task}/{stem}.csv">逐帧动作及指标</a></article>')
        page.append('</div>')
    (args.output / "index.html").write_text('\n'.join(page))
    (args.output / "manifest.json").write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == "__main__":
    main()
