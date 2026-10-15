#!/usr/bin/env python3
"""Review a workcell ROI and 4096-point cloud from saved replay RGB-D.

This is a review artifact, not a training-data exporter. It uses only the
action-pre front depth and its camera matrices; object ground truth is never
used to crop or sample points.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from openpi.spatial.config import WorkspaceROI
from openpi.spatial.geometry import sample_indices, voxel_representatives
from contactworld_dense_capture import camera_xyz_from_matrices
from render_contactworld_dense_pilot import selected_frames


EPISODES = (1, 4, 67, 101, 132)
ROI = WorkspaceROI(x_min_m=-0.20, x_max_m=0.65,
                   y_min_m=-0.35, y_max_m=0.35,
                   z_min_m=-0.005, z_max_m=0.35)
POINTS = 4096
VOXEL_M = 0.005
TABLE_CUTOFF_M = 0.003


def select_frame(depth, rgb, view, projection):
    # In the captured simulator the Franka base pose was checked on every
    # frame to be world origin / identity. Thus world XYZ == base XYZ here.
    xyz, uv = camera_xyz_from_matrices(depth, view, projection)
    inside = ((xyz[:, 0] >= ROI.x_min_m) & (xyz[:, 0] <= ROI.x_max_m) &
              (xyz[:, 1] >= ROI.y_min_m) & (xyz[:, 1] <= ROI.y_max_m) &
              (xyz[:, 2] >= ROI.z_min_m) & (xyz[:, 2] <= ROI.z_max_m))
    xyz, uv = xyz[inside], uv[inside]
    voxel = voxel_representatives(xyz, np.zeros(len(xyz), np.int16),
                                  np.arange(len(xyz), dtype=np.int32),
                                  roi=ROI, voxel_size_m=VOXEL_M)
    vx, _, source, ix, iy, iz = voxel
    if len(vx) < POINTS:
        raise ValueError(f"Only {len(vx)} voxels in ROI; need {POINTS}")
    uniform = sample_indices(ix, iy, iz, num_points=POINTS,
                             sampler="morton_stride")
    # A second candidate keeps every observed object/arm voxel above the
    # table, then fills remaining slots with table voxels in Morton order.
    # It introduces no synthetic geometry and never uses object labels.
    above = np.flatnonzero(vx[:, 2] > TABLE_CUTOFF_M)
    table = np.flatnonzero(vx[:, 2] <= TABLE_CUTOFF_M)
    if len(above) >= POINTS:
        balance = above[sample_indices(ix[above], iy[above], iz[above],
                                       num_points=POINTS, sampler="morton_stride")]
    else:
        remainder = POINTS - len(above)
        if len(table) < remainder:
            raise ValueError("Insufficient table voxels")
        balance = np.r_[above, table[sample_indices(
            ix[table], iy[table], iz[table],
            num_points=remainder, sampler="morton_stride")]]
        # Fixed permutation avoids ordering the model input by surface class.
        balance = balance[np.random.default_rng(2602).permutation(POINTS)]
    selected = {}
    for name, indices in (("uniform", uniform), ("surface", balance)):
        suv = uv[source[indices]]
        selected[name] = (vx[indices].astype(np.float32), suv.astype(np.int16),
                          rgb[suv[:, 1], suv[:, 0]])
    return selected, {"roi_depth_points": int(len(xyz)),
                      "voxel_points": int(len(vx)),
                      "above_table_voxels": int(len(above))}


def select_fused_frame(data, frame):
    xyz_parts, uv_parts, camera_parts = [], [], []
    for camera, role in enumerate(("front", "wrist")):
        view = data["camera_view"] if camera == 0 else data["wrist_view"][frame]
        projection = data["camera_projection"] if camera == 0 else data["wrist_projection"][frame]
        xyz, uv = camera_xyz_from_matrices(data[role + "_depth"][frame], view, projection)
        inside = ((xyz[:, 0] >= ROI.x_min_m) & (xyz[:, 0] <= ROI.x_max_m) &
                  (xyz[:, 1] >= ROI.y_min_m) & (xyz[:, 1] <= ROI.y_max_m) &
                  (xyz[:, 2] >= ROI.z_min_m) & (xyz[:, 2] <= ROI.z_max_m))
        xyz_parts.append(xyz[inside]); uv_parts.append(uv[inside])
        camera_parts.append(np.full(int(inside.sum()), camera, np.int16))
    xyz = np.concatenate(xyz_parts)
    uv = np.concatenate(uv_parts)
    camera_id = np.concatenate(camera_parts)
    voxel = voxel_representatives(xyz, camera_id, np.arange(len(xyz), dtype=np.int32),
                                  roi=ROI, voxel_size_m=VOXEL_M)
    vx, vc, source, ix, iy, iz = voxel
    above = np.flatnonzero(vx[:, 2] > TABLE_CUTOFF_M)
    table = np.flatnonzero(vx[:, 2] <= TABLE_CUTOFF_M)
    if len(above) >= POINTS:
        chosen = above[sample_indices(ix[above], iy[above], iz[above],
                                      num_points=POINTS, sampler="morton_stride")]
    else:
        chosen = np.r_[above, table[sample_indices(ix[table], iy[table], iz[table],
            num_points=POINTS - len(above), sampler="morton_stride")]]
        chosen = chosen[np.random.default_rng(2603).permutation(POINTS)]
    suv = uv[source[chosen]]
    cid = vc[chosen]
    colors = np.empty((POINTS, 3), np.uint8)
    for camera, role in enumerate(("front", "wrist")):
        mask = cid == camera
        colors[mask] = data[role][frame][suv[mask, 1], suv[mask, 0]]
    return (vx[chosen].astype(np.float32), suv.astype(np.int16),
            colors, cid.astype(np.uint8)), {"voxel_points": int(len(vx)),
                                      "above_table_voxels": int(len(above)),
                                      "selected_wrist_points": int((cid == 1).sum())}


def overlay(rgb, uv):
    out = cv2.addWeighted(rgb, .68, np.full_like(rgb, 50), .32, 0)
    for u, v in uv:
        cv2.circle(out, (int(u), int(v)), 1, (45, 245, 20), -1)
    return out


def write_video(path, data, uniform_uv, surface_uv):
    process = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "768x256", "-r", "10", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    try:
        for frame, rgb in enumerate(data["front"]):
            process.stdin.write(np.concatenate(
                (rgb, overlay(rgb, uniform_uv[frame]),
                 overlay(rgb, surface_uv[frame])), axis=1).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def write_wrist_video(path, data, fused_uv, fused_camera):
    process = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "512x256", "-r", "10", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    try:
        for frame, wrist in enumerate(data["wrist"]):
            uv = fused_uv[frame][fused_camera[frame] == 1]
            process.stdin.write(np.concatenate((wrist, overlay(wrist, uv)), axis=1).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def write_figure(path, rgb, wrist, old_uv, narrow_uv, uniform, surface, fused, title):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    images = (rgb, overlay(rgb, old_uv), overlay(rgb, narrow_uv),
              wrist, overlay(rgb, uniform[1]), overlay(rgb, surface[1]),
              overlay(rgb, fused[1][fused[3] == 0]),
              overlay(wrist, fused[1][fused[3] == 1]))
    names = ("Same-frame front RGB", "Released sparse 1024",
             "Earlier narrow ROI 1024", "Same-frame wrist RGB",
             "Workcell ROI / 4096 / uniform", "Workcell ROI / 4096 / surface-balanced",
             "Fused 4096: front contribution", "Fused 4096: wrist contribution")
    for ax, image, name in zip(axes.flat, images, names, strict=False):
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_3d(path, old_xyz, uniform, surface, fused, title):
    fig = go.Figure()
    traces = ((uniform[0], uniform[2], "4096 uniform", True),
              (surface[0], surface[2], "4096 surface-balanced", "legendonly"),
              (fused[0], fused[2], "4096 fused surface-balanced", "legendonly"),
              (old_xyz, None, "Released sparse 1024", "legendonly"))
    for xyz, rgb, name, visibility in traces:
        colors = ([f"rgb({r},{g},{b})" for r, g, b in rgb]
                  if rgb is not None else "#777777")
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers", name=name, visible=visibility,
            marker={"size": 2.0, "color": colors, "opacity": .95}))
    fig.update_layout(title=title, scene={"xaxis_title": "base X (m)",
        "yaxis_title": "base Y (m)", "zaxis_title": "base Z (m)",
        "aspectmode": "data", "camera": {"eye": {"x": 1.35, "y": 1.6, "z": .85}}},
        margin={"l": 0, "r": 0, "b": 0, "t": 45})
    fig.write_html(path, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_dense_dual_pilot_20261002"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_base_cloud_4096_20261002"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cards, summary = [], []
    for episode in EPISODES:
        name = f"insertion_usb_episode_{episode:03d}"
        folder = args.runs / name
        out = args.output / name
        out.mkdir(exist_ok=True)
        replay = json.loads((folder / "replay.json").read_text())
        with np.load(folder / "dense_capture.npz") as data:
            uniform_xyz, uniform_uv = [], []
            surface_xyz, surface_uv = [], []
            fused_xyz, fused_uv, fused_camera = [], [], []
            metrics = []
            frames = selected_frames(replay, len(data["front"]))
            stills = []
            for frame in range(len(data["front"])):
                selected, metric = select_frame(data["front_depth"][frame],
                    data["front"][frame], data["camera_view"], data["camera_projection"])
                uniform, surface = selected["uniform"], selected["surface"]
                fused, fused_metric = select_fused_frame(data, frame)
                uniform_xyz.append(uniform[0]); uniform_uv.append(uniform[1])
                surface_xyz.append(surface[0]); surface_uv.append(surface[1])
                fused_xyz.append(fused[0]); fused_uv.append(fused[1]); fused_camera.append(fused[3])
                metrics.append({**metric, **{"fused_" + k: v for k, v in fused_metric.items()}})
                if frame in frames:
                    stem = f"frame_{frame:03d}"
                    title = f"USB {episode}, pre-action frame {frame}; front RGB-D to base-frame XYZ"
                    write_figure(out / f"{stem}.png", data["front"][frame], data["wrist"][frame],
                                 data["source_cloud_uv"][frame], data["cloud_uv"][frame],
                                 uniform, surface, fused, title)
                    write_3d(out / f"{stem}.html", data["source_cloud_xyz"][frame],
                             uniform, surface, fused, title)
                    stills.append(f'<details><summary>Frame {frame}</summary>'
                                  f'<a href="{name}/{stem}.html">Rotate 3D, switch candidate in legend</a>'
                                  f'<br><img src="{name}/{stem}.png"></details>')
            write_video(out / "rgb_uniform_surface.mp4", data, uniform_uv, surface_uv)
            write_wrist_video(out / "wrist_fused.mp4", data, fused_uv, fused_camera)
            np.savez_compressed(out / "preview_clouds.npz",
                uniform_xyz=np.stack(uniform_xyz), uniform_uv=np.stack(uniform_uv),
                surface_xyz=np.stack(surface_xyz), surface_uv=np.stack(surface_uv),
                fused_xyz=np.stack(fused_xyz), fused_uv=np.stack(fused_uv),
                fused_camera=np.stack(fused_camera))
            summary.append({"episode": episode, "frames": len(data["front"]),
                            "roi_depth_points_min": min(x["roi_depth_points"] for x in metrics),
                            "voxel_points_min": min(x["voxel_points"] for x in metrics),
                            "above_table_voxels_min": min(x["above_table_voxels"] for x in metrics),
                            "fused_voxel_points_min": min(x["fused_voxel_points"] for x in metrics),
                            "fused_wrist_points_median": float(np.median(
                                [x["fused_selected_wrist_points"] for x in metrics])),
                            "uniform_above_table_median": float(np.median(
                                [(x[:, 2] > TABLE_CUTOFF_M).sum() for x in uniform_xyz])),
                            "surface_above_table_median": float(np.median(
                                [(x[:, 2] > TABLE_CUTOFF_M).sum() for x in surface_xyz]))})
            cards.append(f'<section><h2>USB {episode} · {len(data["front"])} frames</h2>'
                         f'<video controls preload="metadata" src="{name}/rgb_uniform_surface.mp4"></video>'
                         f'<p>Wrist view and its contribution to fused 4096:</p>'
                         f'<video controls preload="metadata" src="{name}/wrist_fused.mp4"></video>'
                         + "".join(stills) + "</section>")
    (args.output / "summary.json").write_text(json.dumps({
        "status": "visual_review_required_before_training", "point_count": POINTS,
        "voxel_m": VOXEL_M, "roi_base_m": [ROI.x_min_m, ROI.x_max_m,
            ROI.y_min_m, ROI.y_max_m, ROI.z_min_m, ROI.z_max_m],
        "table_cutoff_m": TABLE_CUTOFF_M, "episodes": summary}, indent=2) + "\n")
    (args.output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>ContactWorld 4096-point base cloud</title>'
        '<style>body{font:16px sans-serif;max-width:1250px;margin:2rem auto;background:#15191c;color:#eee}'
        'a{color:#9bd}video{width:100%;max-width:960px}img{width:100%}'
        'section{border-top:1px solid #555;padding:1rem 0}details{margin:1rem 0}</style></head><body>'
        '<h1>ContactWorld: front RGB-D → base ROI → 4096 points</h1>'
        '<p>Video: RGB / uniform 4096 projection / surface-balanced 4096 projection. '
        'The wrist video shows its contribution to a third candidate that fuses both camera depths in base XYZ. '
        'Each 3D figure defaults to diagnostic RGB colors; click the legend to compare all candidates and old cloud. '
        'The model would receive XYZ only. Camera visibility limits which robot surfaces appear. '
        'These are review candidates, not approved training data.</p>'
        f'<p>ROI base XYZ: {escape(str([ROI.x_min_m, ROI.x_max_m, ROI.y_min_m, ROI.y_max_m, ROI.z_min_m, ROI.z_max_m]))} m; '
        f'{POINTS} points, {VOXEL_M*1000:g} mm voxel.</p>' + "".join(cards) +
        '</body></html>', encoding="utf-8")


if __name__ == "__main__":
    main()
