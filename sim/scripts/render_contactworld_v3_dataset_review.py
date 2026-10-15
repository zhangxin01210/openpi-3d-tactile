#!/usr/bin/env python3
"""Review exact LeRobot V3 videos and spatial arrays, not raw replay files."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np
import plotly.graph_objects as go


def video_path(root: Path, role: str) -> Path:
    return root / f"videos/chunk-000/observation.images.{role}/episode_000000.mp4"


def load_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {path}")
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    spatial = root / "spatial/episodes/episode_000000"
    roles = ("front", "wrist", "tacff_left_preview", "tacff_right_preview")
    videos = {role: load_video(video_path(root, role)) for role in roles}
    n = len(videos["front"])
    if any(len(video) != n for video in videos.values()):
        raise ValueError("Unpaired LeRobot video lengths")
    with np.load(spatial / "sensor_geometry.npz") as geometry:
        front_uv = geometry["cloud_front_uv"]
        fused_uv = geometry["cloud_fused_uv"]
        fused_camera = geometry["cloud_fused_camera"]
    front_xyz = np.load(spatial / "pointcloud_front_xyz.npy", mmap_mode="r")
    fused_xyz = np.load(spatial / "pointcloud_fused_xyz.npy", mmap_mode="r")
    taxels = {side: np.load(spatial / f"tactile_xyz_{side}_base.npy", mmap_mode="r")
              for side in ("left", "right")}
    forces = {side: np.load(spatial / f"tactile_force_{side}_base.npy", mmap_mode="r")
              for side in ("left", "right")}
    frames = sorted({0, min(20, n - 1), min(63, n - 1), n - 1})
    links = []
    contact_links = []
    for frame in frames:
        front_rgb = videos["front"][frame]
        wrist_rgb = videos["wrist"][frame]
        front_colors = front_rgb[front_uv[frame, :, 1], front_uv[frame, :, 0]]
        fused_colors = np.empty((4096, 3), dtype=np.uint8)
        for camera, image in ((0, front_rgb), (1, wrist_rgb)):
            selected = fused_camera[frame] == camera
            uv = fused_uv[frame, selected]
            fused_colors[selected] = image[uv[:, 1], uv[:, 0]]
        plot = go.Figure()
        for name, xyz, colors, visible in (
            ("4096 fused front+wrist (model input 02–07)", fused_xyz[frame], fused_colors, True),
            ("4096 front only (model input 08)", front_xyz[frame], front_colors, "legendonly"),
        ):
            plot.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                name=name, visible=visible,
                marker={"size": 2, "opacity": 0.95,
                        "color": [f"rgb({r},{g},{b})" for r, g, b in colors]},
            ))
        plot.update_layout(
            title=f"LeRobot episode 0, pre-action frame {frame}: robot-base cloud",
            scene={"xaxis_title": "base X (m)", "yaxis_title": "base Y (m)",
                   "zaxis_title": "base Z (m)", "aspectmode": "data",
                   "camera": {"eye": {"x": 1.35, "y": 1.6, "z": 0.85}}},
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )
        filename = f"pointcloud_frame_{frame:03d}.html"
        plot.write_html(args.output / filename, include_plotlyjs=True)
        links.append(f'<li><a href="{filename}">第 {frame} 帧：点云 3D（图例可切前视／融合）</a></li>')
        contact = go.Figure()
        centers = np.stack([taxels[side][frame].mean(axis=0) for side in ("left", "right")])
        center = centers.mean(axis=0)
        local = np.all(np.abs(fused_xyz[frame] - center) < np.array([0.09, 0.09, 0.09]), axis=1)
        xyz = fused_xyz[frame, local]
        colors = fused_colors[local]
        contact.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            name=f"fused cloud near gripper ({len(xyz)} points)",
            marker={"size": 2.1, "opacity": 0.55,
                    "color": [f"rgb({r},{g},{b})" for r, g, b in colors]},
        ))
        details = []
        for side, color in (("left", "#e54e91"), ("right", "#168ac7")):
            position = taxels[side][frame]
            force = forces[side][frame]
            magnitude = np.linalg.norm(force, axis=-1)
            contact.add_trace(go.Scatter3d(
                x=position[:, 0], y=position[:, 1], z=position[:, 2],
                mode="markers", name=f"{side}: all 140 taxel positions",
                marker={"size": 3.3, "color": color, "opacity": 0.92},
            ))
            selected = np.argsort(magnitude)[-24:]
            line_x, line_y, line_z = [], [], []
            for point, vector in zip(position[selected], force[selected], strict=True):
                end = point + vector * 10.0  # synthetic force 0.002 -> 2 cm diagram arrow
                line_x.extend((point[0], end[0], None))
                line_y.extend((point[1], end[1], None))
                line_z.extend((point[2], end[2], None))
            contact.add_trace(go.Scatter3d(
                x=line_x, y=line_y, z=line_z, mode="lines",
                name=f"{side}: strongest 24 base XYZ force vectors ×10",
                line={"color": color, "width": 5},
            ))
            nearest = np.sqrt(np.min(np.sum(
                (position[:, None, :] - fused_xyz[frame][None, :, :]) ** 2,
                axis=-1), axis=-1))
            details.append(
                f"{side}: center={np.round(position.mean(axis=0), 4).tolist()} m; "
                f"max |force|={magnitude.max():.5f} (synthetic); "
                f"mean |Fx,Fy,Fz|={np.round(np.abs(force).mean(axis=0), 6).tolist()}; "
                f"nearest sampled cloud median={np.median(nearest)*1000:.1f} mm")
        contact.update_layout(
            title=f"LeRobot frame {frame}: fused cloud + both 3D tactile pads",
            scene={"xaxis_title": "base X (m)", "yaxis_title": "base Y (m)",
                   "zaxis_title": "base Z (m)", "aspectmode": "data",
                   "camera": {"eye": {"x": 1.35, "y": 1.6, "z": 0.85}}},
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )
        contact_name = f"contact_cloud_tactile_frame_{frame:03d}.html"
        contact.write_html(args.output / contact_name, include_plotlyjs=True)
        contact_links.append(f'<li><a href="{contact_name}">第 {frame} 帧：局部点云＋左右采样点＋三维力</a>'
                             f'<br>{"; ".join(details)}</li>')
    process = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "512x512", "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output / "lerobot_four_streams.mp4"),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in range(n):
        panels = []
        for role, label in (("front", "front RGB"),
                            ("tacff_left_preview", "LEFT TacFF N/Sx/Sy"),
                            ("wrist", "wrist RGB"),
                            ("tacff_right_preview", "RIGHT TacFF N/Sx/Sy")):
            panel = videos[role][frame].copy()
            cv2.rectangle(panel, (0, 0), (255, 25), (20, 20, 20), -1)
            cv2.putText(panel, f"{label}  t={frame:03d}", (7, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        panel = np.concatenate((
            np.concatenate((panels[0], panels[1]), axis=1),
            np.concatenate((panels[2], panels[3]), axis=1),
        ), axis=0)
        process.stdin.write(panel.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError("ffmpeg montage failed")
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ContactWorld V3 LeRobot 数据审核</title>
<body style="font:16px sans-serif;max-width:1150px;margin:auto">
<h1>ContactWorld USB：转好的 LeRobot 单条审核</h1>
<p>以下内容直接读取 <code>{root}</code>；第 0 条，共 {n} 帧。</p>
<p>视频上排：前视 RGB／左指力场编码；下排：腕视 RGB／右指力场编码。
左右视频均为局部法向、切向 X、切向 Y 三通道数值的诊断编码，128 表示零、固定比例为 ±0.003。
它们不是触觉相机 RGB，也不进入当前 OpenPI 配置；模型读取左右各 140 个 taxel 的 base 坐标和三维力。</p>
<video controls preload="metadata" src="lerobot_four_streams.mp4" style="width:100%"></video>
<h2>从 LeRobot sidecar 直接读取的点云</h2><ul>{''.join(links)}</ul>
<p>3D 点的坐标是实际模型输入；颜色只用于审查，来自相同帧的 RGB 视频及保留的像素索引。
融合云默认为显示；单击图例可查看前视云。</p>
<h2>点云与左右触觉在同一个机械臂 base 坐标系</h2>
<p>下列局部图保留夹指附近的融合点云、左右各 140 个真实采样点及每侧幅值最大的 24 根三维力向量。
为看清方向，箭头位移按“仿真力值 ×10 m/单位”绘制；这是显示比例，不是物理位移或牛顿单位。
采样点到点云的距离只作可视审查，点云本身稀疏且可能遮挡胶垫内侧。</p><ul>{''.join(contact_links)}</ul>
<h2>验证记录</h2><p><a href="model_input_audit.json">8 个训练配置的实际输入和编码器检查</a>。
左右 140 点已与原始采集归档逐值核对，诊断视频帧数及图像内容已核对。</p>
</body></html>'''
    (args.output / "index.html").write_text(html)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
