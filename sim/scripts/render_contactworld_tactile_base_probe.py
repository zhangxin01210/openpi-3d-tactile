#!/usr/bin/env python3
"""Visual check of replay-captured ContactWorld right taxels in robot base XYZ."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from preview_contactworld_base_cloud_4096 import select_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_base_probe_20261002/insertion_usb_episode_001/dense_capture.npz"))
    p.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_tactile_base_probe_20261002"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cards = []
    with np.load(args.capture) as data:
        view = data["camera_view"]
        projection = data["camera_projection"]
        inverse = np.linalg.inv(view)
        rotation = inverse[:3, :3].T
        position = inverse[3, :3]
        for frame in sorted({0, len(data["front"]) // 2, len(data["front"]) - 1}):
            rgb = data["front"][frame]
            tactile_xyz = data["tactile_xyz_base"][frame]
            tactile_force = data["tactile_force_base"][frame]
            (selected, _) = select_frame(data["front_depth"][frame], rgb,
                                          view, projection)
            xyz, uv, color = selected["surface"]
            optical = (tactile_xyz - position) @ rotation @ np.diag([1., -1., -1.])
            focal = np.array([projection[0, 0] * 128, projection[1, 1] * 128])
            tactile_uv = np.rint(optical[:, :2] / optical[:, 2:3] * focal + 128 - .5).astype(int)
            inside = ((tactile_uv >= 0) & (tactile_uv < 256)).all(axis=1)
            canvas = cv2.addWeighted(rgb, .75, np.full_like(rgb, 40), .25, 0)
            for u, v in uv:
                cv2.circle(canvas, (int(u), int(v)), 1, (40, 235, 30), -1)
            for u, v in tactile_uv[inside]:
                cv2.circle(canvas, (int(u), int(v)), 2, (250, 40, 245), -1)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            axes[0].imshow(rgb); axes[0].set_title("Front RGB")
            axes[1].imshow(canvas); axes[1].set_title("4096 cloud green / 140 taxels magenta")
            for ax in axes: ax.axis("off")
            fig.savefig(args.output / f"frame_{frame:03d}.png", dpi=150)
            plt.close(fig)
            colors = [f"rgb({r},{g},{b})" for r, g, b in color]
            three = go.Figure()
            three.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers", name="RGB-D XYZ, base", marker={"size": 2, "color": colors}))
            magnitude = np.linalg.norm(tactile_force, axis=-1)
            three.add_trace(go.Scatter3d(x=tactile_xyz[:, 0], y=tactile_xyz[:, 1], z=tactile_xyz[:, 2],
                mode="markers", name="Right taxel XYZ, base",
                marker={"size": 3, "color": magnitude, "colorscale": "Plasma",
                        "cmin": 0, "cmax": .003,
                        "colorbar": {"title": "Source force norm"}}))
            active = magnitude > np.quantile(magnitude, .8)
            starts = tactile_xyz[active]
            ends = starts + tactile_force[active] * 10  # visualization scale only
            for a, b in zip(starts, ends):
                three.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                    mode="lines", line={"color": "#ff2fd1", "width": 2},
                    showlegend=False, hoverinfo="skip"))
            three.update_layout(title=f"Frame {frame}: true simulated right taxel poses and base force",
                scene={"xaxis_title": "base X (m)", "yaxis_title": "base Y (m)",
                       "zaxis_title": "base Z (m)", "aspectmode": "data"},
                margin={"l": 0, "r": 0, "b": 0, "t": 45})
            three.write_html(args.output / f"frame_{frame:03d}.html", include_plotlyjs=True)
            cards.append(f'<section><h2>Frame {frame}</h2><a href="frame_{frame:03d}.html">Rotate 3D</a>'
                         f'<br><img src="frame_{frame:03d}.png"></section>')
    (args.output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>ContactWorld tactile base probe</title>'
        '<style>body{font:16px sans-serif;max-width:1050px;margin:2rem auto;background:#15191c;color:#eee}'
        'a{color:#9bd}img{width:100%}section{border-top:1px solid #555;padding:1rem 0}</style>'
        '</head><body><h1>ContactWorld right TacFF: measured taxel geometry in base frame</h1>'
        '<p>One deterministic replay, three frames. The 3D vectors use source simulator force units; '
        'line lengths are multiplied by 10 solely for viewing. 2D taxel projection can include occluded taxels. '
        f'Source capture: {escape(str(args.capture))}.</p>' + "".join(cards) + '</body></html>',
        encoding="utf-8")


if __name__ == "__main__":
    main()
