#!/usr/bin/env python3
"""Audit ContactWorld right-finger TacFF normal, shear, and base vectors."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


EPISODES = (1, 4, 67, 101, 132)


def describe(field):
    normal = np.abs(field[..., 0])
    shear = np.linalg.norm(field[..., 1:3], axis=-1)
    active = normal > 1e-4
    ratio = shear[active] / np.maximum(normal[active], 1e-9)
    return {"frames": len(field), "active_taxel_fraction": float(active.mean()),
            "active_shear_over_normal_median": float(np.median(ratio)),
            "active_shear_over_normal_p90": float(np.quantile(ratio, .9)),
            "mean_abs_normal_range": [float(x) for x in
                (normal.mean((1, 2)).min(), normal.mean((1, 2)).max())],
            "mean_shear_magnitude_range": [float(x) for x in
                (shear.mean((1, 2)).min(), shear.mean((1, 2)).max())],
            "mean_shear_x_signed_range": [float(x) for x in
                (field[..., 1].mean((1, 2)).min(), field[..., 1].mean((1, 2)).max())],
            "mean_shear_y_signed_range": [float(x) for x in
                (field[..., 2].mean((1, 2)).min(), field[..., 2].mean((1, 2)).max())]}


def components_in_base(field, quaternion, base_quaternion):
    raw = field.reshape(140, 3).astype(np.float64)
    rotation = Rotation.from_quat(quaternion)
    base_inverse = Rotation.from_quat(base_quaternion).inv()
    # Simulator stores [normal, shear_x, shear_y] = [-Fy, -Fx, Fz]
    normal_local = np.stack((np.zeros(140), -raw[:, 0], np.zeros(140)), axis=-1)
    shear_local = np.stack((-raw[:, 1], np.zeros(140), raw[:, 2]), axis=-1)
    return (base_inverse.apply(rotation.apply(normal_local)),
            base_inverse.apply(rotation.apply(shear_local)))


def line_trace(points, vectors, indices, color, name, scale=10.0):
    x, y, z = [], [], []
    for index in indices:
        a = points[index]
        b = a + vectors[index] * scale
        x.extend((a[0], b[0], None))
        y.extend((a[1], b[1], None))
        z.extend((a[2], b[2], None))
    return go.Scatter3d(x=x, y=y, z=z, mode="lines",
                        name=name, line={"color": color, "width": 4})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_dense_dual_pilot_20261002"))
    parser.add_argument("--base-probe", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_3d_audit_20261002/insertion_usb_episode_001/dense_capture.npz"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_tactile_3d_audit_20261002"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for episode in EPISODES:
        with np.load(args.runs / f"insertion_usb_episode_{episode:03d}/dense_capture.npz") as data:
            stats[str(episode)] = describe(data["force_grid"])
    with np.load(args.base_probe) as data:
        field = data["force_grid"]
        normal = np.abs(field[..., 0]).mean((1, 2))
        shear = np.linalg.norm(field[..., 1:3], axis=-1).mean((1, 2))
        sx = field[..., 1].mean((1, 2))
        sy = field[..., 2].mean((1, 2))
        strongest = int(np.argmax(shear))
        frames = sorted({0, len(field) // 2, strongest, len(field) - 1})
        all_errors = []
        for frame in range(len(field)):
            nbase, tbase = components_in_base(field[frame], data["tactile_quat_world"][frame],
                                              data["base_pose_world"][frame, 3:])
            all_errors.append(float(np.max(np.abs(nbase + tbase - data["tactile_force_base"][frame]))))
        if max(all_errors) > 1e-6:
            raise ValueError("Base force does not equal rotated normal + shear")
        norm_error = float(np.max(np.abs(
            np.linalg.norm(data["tactile_force_base"], axis=-1) -
            np.linalg.norm(field.reshape(len(field), 140, 3), axis=-1))))
        if norm_error > 1e-6:
            raise ValueError("Force norm changed during frame conversion")
        stats["base_conversion"] = {"frames": len(field),
            "normal_plus_shear_max_error": max(all_errors),
            "rotation_norm_max_error": norm_error,
            "strongest_mean_shear_frame": strongest,
            "selected_frames": frames,
            "right_sensor_only": True,
            "source_force_unit": "simulator internal scale; Newton calibration not established"}
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
        axes[0].plot(normal, label="mean |normal|", color="#2875bc")
        axes[0].plot(shear, label="mean tangent magnitude", color="#d47816")
        axes[0].set_ylabel("source force scale")
        axes[0].legend()
        axes[1].plot(sx, label="mean signed shear x", color="#b2325a")
        axes[1].plot(sy, label="mean signed shear y", color="#7154a0")
        axes[1].axhline(0, color="gray", linewidth=.7)
        axes[1].set_ylabel("source force scale")
        axes[1].legend()
        axes[2].plot((np.linalg.norm(field[..., 1:3], axis=-1) > 1e-4).mean((1, 2)),
                     label="fraction with tangent magnitude > 1e-4", color="#b2325a")
        axes[2].plot((np.abs(field[..., 0]) > 1e-4).mean((1, 2)),
                     label="fraction with |normal| > 1e-4", color="#2875bc")
        axes[2].set(xlabel="pre-action frame", ylabel="taxel fraction", ylim=(0, 1))
        axes[2].legend()
        for ax in axes:
            for frame in frames: ax.axvline(frame, color="#999", alpha=.45)
        fig.suptitle("USB episode 1: right sensor normal and shear over time")
        fig.savefig(args.output / "time_series.png", dpi=160)
        plt.close(fig)
        cards = []
        for frame in frames:
            grid = field[frame]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
            for channel, (ax, title) in enumerate(zip(axes,
                    ("normal", "shear x", "shear y"))):
                image = ax.imshow(grid[..., channel], cmap="coolwarm", vmin=-.0025,
                                  vmax=.0025, interpolation="nearest")
                ax.set_title(title); ax.set(xlabel="column", ylabel="row")
                fig.colorbar(image, ax=ax, shrink=.8)
            fig.suptitle(f"Right sensor 10×14 field, frame {frame}; shared signed scale")
            fig.savefig(args.output / f"field_{frame:03d}.png", dpi=160)
            plt.close(fig)
            points = data["tactile_xyz_base"][frame]
            nbase, tbase = components_in_base(grid, data["tactile_quat_world"][frame],
                                              data["base_pose_world"][frame, 3:])
            mag = np.linalg.norm(nbase + tbase, axis=-1)
            selected = np.argsort(mag)[-25:]
            three = go.Figure()
            three.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers", name="right tactile positions, base",
                marker={"size": 3, "color": mag, "colorscale": "Viridis", "cmin": 0, "cmax": .003}))
            three.add_trace(line_trace(points, nbase, selected, "#2875bc", "normal component ×10"))
            three.add_trace(line_trace(points, tbase, selected, "#e87a25", "shear component ×10"))
            three.add_trace(line_trace(points, nbase+tbase, selected, "#b2268c", "total base force ×10"))
            three.update_layout(title=f"Episode 1 frame {frame}: normal + shear = base 3D force",
                scene={"xaxis_title": "base X (m)", "yaxis_title": "base Y (m)",
                       "zaxis_title": "base Z (m)", "aspectmode": "data"},
                margin={"l": 0, "r": 0, "b": 0, "t": 45})
            three.write_html(args.output / f"base_vectors_{frame:03d}.html", include_plotlyjs=True)
            cards.append(f'<section><h2>Pre-action frame {frame}</h2>'
                f'<a href="base_vectors_{frame:03d}.html">Rotate and toggle normal/shear/total vectors</a>'
                f'<br><img src="field_{frame:03d}.png"></section>')
    (args.output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>ContactWorld 3D tactile audit</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:2rem auto;background:#15191c;color:#eee}'
        'a{color:#9bd}img{width:100%}section{border-top:1px solid #555;padding:1rem 0}</style></head><body>'
        '<h1>Right-finger TacFF: normal and two shear channels</h1>'
        '<p>Only the right tactile force field is in the released observation. These are simulated '
        'source units, not calibrated Newtons. 3D lines are multiplied by 10 for legibility; '
        'component direction and relative magnitude are retained.</p>'
        '<img src="time_series.png">' + "".join(cards) +
        f'<p><a href="summary.json">Five-demo channel statistics</a> · '
        f'Base conversion error checked on {escape(str(len(field)))} frames.</p>'
        '</body></html>', encoding="utf-8")


if __name__ == "__main__":
    main()
