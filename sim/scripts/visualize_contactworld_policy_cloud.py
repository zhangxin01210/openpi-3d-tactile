#!/usr/bin/env python3
"""Show the exact xyz arrays sent to a ContactWorld point-cloud policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def state_at(record: dict, step: int) -> dict:
    if step == 0:
        return record["result"]["initial_physical_state"]
    return record["rows"][step - 1]


def summarize(xyz: np.ndarray, plug: np.ndarray, socket: np.ndarray) -> dict:
    return {
        "points": len(xyz), "min_xyz_m": xyz.min(0).tolist(), "max_xyz_m": xyz.max(0).tolist(),
        "table_like_z_below_2cm": int((xyz[:, 2] < 0.02).sum()),
        "within_5cm_plug": int((np.linalg.norm(xyz - plug, axis=1) < 0.05).sum()),
        "within_5cm_socket": int((np.linalg.norm(xyz - socket, axis=1) < 0.05).sum()),
        "illustrative_work_region": int((
            (xyz[:, 0] >= 0.25) & (xyz[:, 0] <= 0.65) &
            (np.abs(xyz[:, 1]) <= 0.2) & (xyz[:, 2] <= 0.3)).sum()),
    }


def plot_png(path: Path, front: np.ndarray, xyz: np.ndarray, plug: np.ndarray,
             socket: np.ndarray, seed: int, step: int, counts: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.imshow(front)
    ax.set_title("Front RGB (same query)")
    ax.axis("off")
    ax = axes[0, 1]
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=8,
                         cmap="viridis", vmin=0, vmax=0.25, alpha=0.75)
    ax.scatter(*plug[:2], marker="x", s=150, c="red", label="plug pose")
    ax.scatter(*socket[:2], marker="+", s=150, c="magenta", label="socket pose")
    ax.set(xlabel="base X (m)", ylabel="base Y (m)", title="Top view: all 1024 model-input points",
           xlim=(-0.3, 0.7), ylim=(-0.5, 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=ax, label="height Z (m)")
    ax = axes[1, 0]
    ax.scatter(xyz[:, 0], xyz[:, 2], c=xyz[:, 2], s=8,
               cmap="viridis", vmin=0, vmax=0.25, alpha=0.75)
    ax.scatter(plug[0], plug[2], marker="x", s=150, c="red")
    ax.scatter(socket[0], socket[2], marker="+", s=150, c="magenta")
    ax.set(xlabel="base X (m)", ylabel="base Z (m)", title="Side view",
           xlim=(-0.3, 0.7), ylim=(-0.01, 0.3))
    ax = axes[1, 1]
    near = (np.linalg.norm(xyz[:, :2] - socket[:2], axis=1) < 0.16)
    ax.scatter(xyz[near, 0], xyz[near, 1], c=xyz[near, 2], s=16,
               cmap="viridis", vmin=0, vmax=0.25, alpha=0.8)
    ax.scatter(*plug[:2], marker="x", s=150, c="red")
    ax.scatter(*socket[:2], marker="+", s=150, c="magenta")
    ax.set(xlabel="base X (m)", ylabel="base Y (m)", title="Near socket: XY radius 16 cm",
           xlim=(socket[0] - 0.16, socket[0] + 0.16),
           ylim=(socket[1] - 0.16, socket[1] + 0.16))
    ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"Seed {seed}, policy query at step {step} | "
                 f"z<2cm: {counts['table_like_z_below_2cm']}/1024 | "
                 f"within 5cm of socket: {counts['within_5cm_socket']}/1024", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_html(path: Path, xyz: np.ndarray, plug: np.ndarray,
              socket: np.ndarray, seed: int, step: int) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker={"size": 2.6, "color": xyz[:, 2], "colorscale": "Viridis", "cmin": 0,
                "cmax": 0.25, "colorbar": {"title": "Z (m)"}}, name="1024 policy input points"))
    for name, point, color, symbol in (("plug pose", plug, "red", "x"),
                                        ("socket pose", socket, "magenta", "diamond")):
        fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                                  mode="markers", marker={"size": 8, "color": color,
                                                           "symbol": symbol}, name=name))
    fig.update_layout(title=f"Actual policy point cloud · seed {seed}, step {step}",
                      scene={"xaxis_title": "base X (m)", "yaxis_title": "base Y (m)",
                             "zaxis_title": "base Z (m)", "aspectmode": "data"},
                      margin={"l": 0, "r": 0, "b": 0, "t": 45})
    fig.write_html(path, include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_paired_20261002/pi0_cw_usb_02_rgb_pc"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_policy_cloud_20261002"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10000, 10001, 10002, 10003, 10004])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    cards = []
    for seed in args.seeds:
        root = args.rollouts / f"seed_{seed}"
        record = json.loads((root / "trajectory.json").read_text())
        with np.load(root / "inputs.npz") as data:
            query_steps = data["steps"]
            contact = next((r["step"] for r in record["rows"]
                            if np.linalg.norm(r["plug_socket_contact_force_xyz"]) > 0.1), None)
            selected = [0]
            if seed == args.seeds[0] and contact is not None:
                selected.append(int(np.argmin(np.abs(query_steps - contact))))
            for index in selected:
                step = int(query_steps[index])
                xyz = data["cloud_xyz"][index]
                front = data["front"][index]
                state = state_at(record, step)
                plug = np.asarray(state["plug_pos"])
                socket = np.asarray(state["socket_pos_gt"])
                counts = summarize(xyz, plug, socket)
                counts.update(seed=seed, policy_query_step=step,
                              source_inputs=str(root / "inputs.npz"))
                reports.append(counts)
                stem = f"seed_{seed}_step_{step}"
                plot_png(args.output / f"{stem}.png", front, xyz, plug, socket, seed, step, counts)
                plot_html(args.output / f"{stem}.html", xyz, plug, socket, seed, step)
                cards.append(f'<section><h2>Seed {seed}, step {step}</h2>'
                             f'<p>Points with z&lt;2 cm: {counts["table_like_z_below_2cm"]}/1024; '
                             f'within 5 cm of socket: {counts["within_5cm_socket"]}/1024.</p>'
                             f'<a href="{stem}.html">Rotate 3D cloud</a><br>'
                             f'<img src="{stem}.png" alt="Point cloud with RGB and XY/XZ views"></section>')
    (args.output / "counts.json").write_text(json.dumps(reports, indent=2) + "\n")
    (args.output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ContactWorld model input cloud</title>"
        "<style>body{font:16px sans-serif;max-width:1400px;margin:2rem auto;background:#171a1e;color:#eee}"
        "a{color:#9ce}img{width:100%}section{border-top:1px solid #555;margin:1rem 0;padding:1rem 0}</style>"
        "</head><body><h1>Point cloud actually sent to RGB+PC policy</h1>"
        "<p>Exactly 1024 corrected xyz points per query. Color shows Z; red X and magenta diamond "
        "show simulator plug/socket poses for diagnosis and are not policy inputs. "
        "The policy receives xyz only, with all point masks true and RGB point color set to zero.</p>"
        + "".join(cards) + "</body></html>", encoding="utf-8")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
