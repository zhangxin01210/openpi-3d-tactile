#!/usr/bin/env python3
"""Render a five-episode RGB-D / sparse-cloud / cropped-cloud review gallery."""

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

from contactworld_dense_capture import CROP, GEOMETRY, camera_xyz, camera_xyz_from_matrices


EPISODES = (1, 4, 67, 101, 132)


def overlay(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    canvas = cv2.addWeighted(image, 0.72, np.full_like(image, 50), 0.28, 0)
    for u, v in uv:
        cv2.circle(canvas, (int(u), int(v)), 1, (40, 255, 40), -1)
    return canvas


def selected_frames(replay: dict, length: int) -> list[int]:
    steps = {0, length // 4, length // 2, 3 * length // 4, length - 1}
    contact = next((row["frame"] for row in replay["rows"]
                    if np.linalg.norm(row["plug_socket_force_post"]) > 0.1), None)
    if contact is not None:
        steps.add(int(contact))
    return sorted(steps)


def review_png(path: Path, recorded: np.ndarray, data, frame: int, episode: int) -> None:
    rgb = data["front"][frame]
    old = overlay(rgb, data["source_cloud_uv"][frame])
    new = overlay(rgb, data["cloud_uv"][frame])
    depth = data["front_depth"][frame]
    dual = "cloud_fused_xyz" in data
    if dual:
        wrist = data["wrist"][frame]
        wrist_points = data["cloud_fused_uv"][frame][data["cloud_fused_camera_id"][frame] == 1]
        wrist_overlay = overlay(wrist, wrist_points)
        pictures = (recorded, rgb, depth, wrist, old, new, wrist_overlay)
        titles = ("Released front RGB", "Replayed front RGB", "Same-frame front depth (m)",
                  "Same-frame wrist RGB", "Old whole-scene front 1024",
                  "New cropped front 1024", "Fused cloud's 512 wrist points")
        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        axes = [fig.add_subplot(2, 4, i + 1, projection="3d" if i == 7 else None)
                for i in range(8)]
    else:
        pictures = (recorded, rgb, depth, old, new)
        titles = ("Released RGB", "Replayed pre-action RGB", "Same-frame front depth (m)",
                  "Old whole-scene 1024 points", "New cropped 1024 points")
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        axes = [fig.add_subplot(2, 3, i + 1, projection="3d" if i == 5 else None)
                for i in range(6)]
    for ax, picture, title in zip(axes[:len(pictures)], pictures, titles, strict=True):
        ax.imshow(picture, cmap="magma_r" if picture is depth else None,
                  vmin=0 if picture is depth else None, vmax=2 if picture is depth else None)
        ax.set_title(title)
        ax.axis("off")
    ax = axes[-1]
    xyz = data["cloud_fused_xyz"][frame] if dual else data["cloud_xyz"][frame]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap="viridis",
               vmin=0, vmax=0.3, s=3, depthshade=False)
    ax.set(xlabel="base X (m)", ylabel="base Y (m)", zlabel="base Z (m)",
           title="Fused XYZ (512+512)" if dual else "New front XYZ, height coloring")
    ax.view_init(elev=22, azim=135)
    fig.suptitle(f"USB episode {episode}, pre-action frame {frame}. "
                 "RGB colors and pose labels are diagnostic only; model would receive XYZ.")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def review_3d(path: Path, data, frame: int, episode: int) -> None:
    dual = "cloud_fused_xyz" in data
    xyz = data["cloud_fused_xyz"][frame] if dual else data["cloud_xyz"][frame]
    rgb = data["cloud_fused_rgb"][frame] if dual else data["cloud_rgb"][frame]
    old = data["source_cloud_xyz"][frame]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=old[:, 0], y=old[:, 1], z=old[:, 2],
        mode="markers", marker={"size": 1.8, "color": "#777"},
        name="Old whole-scene 1024", visible="legendonly"))
    if dual:
        front = data["cloud_xyz"][frame]
        fig.add_trace(go.Scatter3d(x=front[:, 0], y=front[:, 1], z=front[:, 2],
            mode="markers", marker={"size": 2.1, "color": front[:, 2], "colorscale": "Viridis",
                                    "cmin": 0, "cmax": 0.3},
            name="New front-only XYZ", visible="legendonly"))
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers", marker={"size": 2.4, "color": xyz[:, 2], "colorscale": "Viridis",
                                "cmin": 0, "cmax": 0.3, "colorbar": {"title": "Z (m)"}},
        name="New fused XYZ, height color" if dual else "New front XYZ, height color"))
    colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb]
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers", marker={"size": 2.4, "color": colors},
        name="New fused XYZ, diagnostic RGB" if dual else "New front XYZ, diagnostic RGB",
        visible="legendonly"))
    fig.update_layout(title=f"Episode {episode} · frame {frame}: new cloud in base frame",
        scene={"xaxis_title": "base X (m)", "yaxis_title": "base Y (m)",
               "zaxis_title": "base Z (m)", "aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 45})
    fig.write_html(path, include_plotlyjs=True)


def make_video(path: Path, recorded: np.ndarray, data) -> None:
    process = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "1024x256", "-r", "10", "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)
    try:
        for i in range(len(recorded)):
            panels = (recorded[i], data["front"][i],
                      overlay(data["front"][i], data["source_cloud_uv"][i]),
                      overlay(data["front"][i], data["cloud_uv"][i]))
            process.stdin.write(np.concatenate(panels, axis=1).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def make_wrist_video(path: Path, data) -> None:
    process = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "512x256", "-r", "10", "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)
    try:
        for i in range(len(data["wrist"])):
            wrist = data["wrist"][i]
            uv = data["cloud_fused_uv"][i][data["cloud_fused_camera_id"][i] == 1]
            process.stdin.write(np.concatenate((wrist, overlay(wrist, uv)), axis=1).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def coverage(data, rotation, translation) -> dict:
    red_fraction, blue_fraction = [], []
    for frame in range(len(data["front"])):
        xyz, uv = camera_xyz(data["front_depth"][frame], data["camera_view"],
                             data["camera_projection"], rotation, translation)
        colors = data["front"][frame][uv[:, 1], uv[:, 0]].astype(np.int16)
        inside = np.ones(len(xyz), bool)
        for axis, (lo, hi) in enumerate(CROP):
            inside &= (xyz[:, axis] >= lo) & (xyz[:, axis] <= hi)
        red = (colors[:, 0] > 150) & (colors[:, 1] < 120) & (colors[:, 2] < 120)
        blue = (colors[:, 2] > colors[:, 0] + 20) & (colors[:, 2] > colors[:, 1] + 20)
        if red.sum():
            red_fraction.append(float((red & inside).sum() / red.sum()))
        if blue.sum():
            blue_fraction.append(float((blue & inside).sum() / blue.sum()))
    return {"min_red_pixel_coverage": min(red_fraction),
            "min_blue_pixel_coverage": min(blue_fraction),
            "note": "Color thresholds are diagnostic, not object segmentation ground truth"}


def reprojection(data) -> dict:
    pixel_errors = []
    depth_errors = []
    for frame in range(len(data["front"])):
        xyz = data["cloud_fused_xyz"][frame] if "cloud_fused_xyz" in data else data["cloud_xyz"][frame]
        uv_expected = data["cloud_fused_uv"][frame] if "cloud_fused_xyz" in data else data["cloud_uv"][frame]
        camera_id = data["cloud_fused_camera_id"][frame] if "cloud_fused_xyz" in data else np.zeros(len(xyz), np.uint8)
        for camera in (0, 1) if "cloud_fused_xyz" in data else (0,):
            which = camera_id == camera
            if not which.any():
                continue
            view = data["camera_view"] if camera == 0 else data["wrist_view"][frame]
            projection = data["camera_projection"] if camera == 0 else data["wrist_projection"][frame]
            depth = data["front_depth"][frame] if camera == 0 else data["wrist_depth"][frame]
            height, width = depth.shape
            inverse = np.linalg.inv(view)
            rotation = inverse[:3, :3].T
            position = inverse[3, :3]
            optical = (xyz[which] - position) @ rotation @ np.diag([1., -1., -1.])
            focal = np.array([projection[0, 0] * width / 2, projection[1, 1] * height / 2])
            uv = optical[:, :2] / optical[:, 2:3] * focal + [width / 2, height / 2] - 0.5
            expected = uv_expected[which]
            pixel_errors.append(float(np.abs(uv - expected).max()))
            depth_errors.append(float(np.abs(optical[:, 2] -
                depth[expected[:, 1], expected[:, 0]]).max()))
    if max(pixel_errors) > 1e-3 or max(depth_errors) > 1e-5:
        raise ValueError("Camera reprojection/depth mismatch")
    return {"reprojection_max_pixel_error": max(pixel_errors),
            "reprojection_max_depth_error_m": max(depth_errors),
            "wrist_view_matrix_max_change": (float(np.abs(data["wrist_view"] -
                data["wrist_view"][0]).max()) if "wrist_view" in data else None)}


def table_plane_agreement(data) -> dict | None:
    if "wrist_view" not in data:
        return None
    medians = []
    for frame in sorted({0, len(data["front"]) // 2, len(data["front"]) - 1}):
        front_xyz, _ = camera_xyz_from_matrices(data["front_depth"][frame],
                                                 data["camera_view"], data["camera_projection"])
        wrist_xyz, _ = camera_xyz_from_matrices(data["wrist_depth"][frame],
                                                 data["wrist_view"][frame],
                                                 data["wrist_projection"][frame])
        front_table = front_xyz[np.abs(front_xyz[:, 2]) < 0.005, 2]
        wrist_table = wrist_xyz[np.abs(wrist_xyz[:, 2]) < 0.005, 2]
        if len(front_table) and len(wrist_table):
            medians.append(float(abs(np.median(front_table) - np.median(wrist_table))))
    return {"shared_table_z_median_disagreement_max_m": max(medians) if medians else None,
            "checked_frames": len(medians)}


def quantiles(rows: list[dict], name: str) -> dict:
    values = np.asarray([row[name] for row in rows])
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_dense_pilot_20261002"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_dense_pilot_20261002"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(GEOMETRY.read_text())
    rotation = np.asarray(geometry["A"], np.float32)
    translation = np.asarray(geometry["t"], np.float32)
    cards = []
    summary = {"status": "visual_review_required_before_training", "crop_base_m": CROP,
               "episodes": []}
    for episode in EPISODES:
        name = f"insertion_usb_episode_{episode:03d}"
        path = args.runs / name
        out = args.output / name
        out.mkdir(exist_ok=True)
        report = json.loads((path / "dense_capture.json").read_text())
        replay = json.loads((path / "replay.json").read_text())
        with np.load(path / "dense_capture.npz") as data, np.load(report["source_demo"]) as demo:
            recorded = np.rint(np.clip(demo["front"], 0, 1) * 255).astype(np.uint8)
            if len(recorded) != len(data["front"]) or len(replay["rows"]) != len(recorded):
                raise ValueError("Frame count mismatch")
            if not replay["final_source_success"] or not replay["source_pointcloud_alignment_passed"]:
                raise ValueError(f"Replay or source cloud failed: {episode}")
            make_video(out / "four_views.mp4", recorded, data)
            dual = "cloud_fused_xyz" in data
            if dual:
                make_wrist_video(out / "wrist_views.mp4", data)
            frames = selected_frames(replay, len(recorded))
            stills = []
            for frame in frames:
                stem = f"frame_{frame:03d}"
                review_png(out / f"{stem}.png", recorded[frame], data, frame, episode)
                if frame in {0, len(recorded) // 2, len(recorded) - 1}:
                    review_3d(out / f"{stem}.html", data, frame, episode)
                three_d = f' · <a href="{name}/{stem}.html">rotate 3D</a>' if (
                    out / f"{stem}.html").exists() else ""
                stills.append(f'<details><summary>Frame {frame}</summary><p>'
                              f'<a href="{name}/{stem}.png">full image</a>{three_d}</p>'
                              f'<img src="{name}/{stem}.png"></details>')
            row = {"episode": episode, "frames": len(recorded),
                   "final_source_success": replay["final_source_success"],
                   "crop_candidates": quantiles(report["rows"], "fixed_crop_candidates"),
                   "new_red_pixel_points": quantiles(report["rows"], "red_pixel_points"),
                   "new_blue_pixel_points": quantiles(report["rows"], "blue_pixel_points"),
                   "old_red_pixel_points_median": float(np.median([
                       x["source_cloud_color_counts"]["red_pixel_points"] for x in report["rows"]])),
                   "old_blue_pixel_points_median": float(np.median([
                       x["source_cloud_color_counts"]["blue_pixel_points"] for x in report["rows"]])),
                   "replay_rgb_mae": quantiles(report["rows"], "released_rgb_mae"),
                   "plug_pos_error_mm": quantiles(report["rows"], "plug_pos_error_mm"),
                   "action_correction_l2": quantiles(report["rows"], "action_correction_l2"),
                   "action_raw_clipped_fraction": float((np.abs(data["action_executed"] -
                       data["action_sim_clipped"]) > 1e-6).mean()),
                   **coverage(data, rotation, translation), **reprojection(data)}
            if dual:
                row["table_plane_agreement"] = table_plane_agreement(data)
                row["wrist_crop_candidates"] = quantiles(report["rows"], "wrist_crop_candidates")
                row["fused_red_pixel_points_median"] = float(np.median([
                    x["fused_cloud_color_counts"]["red_pixel_points"] for x in report["rows"]]))
                row["fused_blue_pixel_points_median"] = float(np.median([
                    x["fused_cloud_color_counts"]["blue_pixel_points"] for x in report["rows"]]))
                row["first_frame_fused_blue_pixel_points"] = report["rows"][0][
                    "fused_cloud_color_counts"]["blue_pixel_points"]
            summary["episodes"].append(row)
            cards.append(f'<section><h2>USB episode {episode} · {len(recorded)} frames</h2>'
                         f'<p>Crop candidates min {int(row["crop_candidates"]["min"])}; '
                         f'new red/blue median {row["new_red_pixel_points"]["median"]:.0f}/'
                         f'{row["new_blue_pixel_points"]["median"]:.0f}; '
                         f'old {row["old_red_pixel_points_median"]:.0f}/'
                         f'{row["old_blue_pixel_points_median"]:.0f}. '
                         f'Min red/blue crop coverage {row["min_red_pixel_coverage"]:.1%}/'
                         f'{row["min_blue_pixel_coverage"]:.1%}.'
                         + (f' Fused red/blue median {row["fused_red_pixel_points_median"]:.0f}/'
                            f'{row["fused_blue_pixel_points_median"]:.0f}.' if dual else "") + '</p>'
                         '<p>Video panels: released RGB · replayed RGB · old sampled points · new sampled points.</p>'
                         f'<video controls preload="metadata" src="{name}/four_views.mp4"></video>'
                         + (f'<p>Wrist RGB · sampled wrist points (512).</p>'
                            f'<video controls preload="metadata" src="{name}/wrist_views.mp4"></video>'
                            if dual else "")
                         + "".join(stills) + "</section>")
        print(name, "rendered", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld dense point cloud pilot</title>"
        "<style>body{font:16px sans-serif;max-width:1500px;margin:2rem auto;background:#171a1e;color:#eee}"
        "a{color:#9ce}video,img{width:100%}section{border-top:1px solid #555;padding:1rem 0}"
        "details{margin:1rem 0;border:1px solid #555;padding:.6rem}</style></head><body>"
        "<h1>USB dense depth → fixed-workspace 1024-point cloud: five replayed demonstrations</h1>"
        "<p>Each saved RGB, depth, XYZ and executed action is from the same pre-action frame. "
        "The new cloud is sampled after a fixed base-frame crop (x 0.25–0.62 m, y ±0.20 m, "
        "z 0.01–0.30 m). The dual-view candidate mixes 512 front and 512 wrist depth points "
        "using each frame's camera matrices. The crop never reads per-frame plug/socket poses. Colors shown in 3D "
        "are only for diagnosis; the proposed model input remains XYZ. These five trajectories "
        "are a pilot requiring visual review and are not approved training data.</p>"
        '<p><a href="summary.json">Numeric audit</a></p>' + "".join(cards) + "</body></html>",
        encoding="utf-8")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
