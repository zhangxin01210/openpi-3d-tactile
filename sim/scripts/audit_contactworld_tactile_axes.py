"""Inspect the released right-finger TacFF axes against replay contact loads.

This is a diagnostic plot, not a force calibration: TacFF values and PhysX
socket contact loads are displayed on separate axes and must not be compared
numerically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=59)
    args = parser.parse_args()

    path = args.capture / "v2_capture.npz"
    if not path.exists():
        path = args.capture / "dense_capture.npz"
    with np.load(path) as data:
        force = data["tactile_force_base"].astype(np.float64)
        local = data["force_grid_local" if "force_grid_local" in data else "force_grid"]
    replay = json.loads((args.capture / "replay.json").read_text())
    socket_force = np.array([row["plug_socket_force_post"] for row in replay["rows"]])
    t = args.frame
    if t < 0 or t >= len(force):
        raise ValueError(f"frame must be in [0, {len(force)})")
    active = np.linalg.norm(force, axis=-1) > 1e-7
    mean_abs = np.array([
        np.mean(np.abs(force[i, active[i]]), axis=0) if np.any(active[i]) else np.zeros(3)
        for i in range(len(force))
    ])
    names = ("base Fx", "base Fy", "base Fz")
    colors = ("#b04a58", "#247da4", "#22824c")
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :])
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        ax.plot(mean_abs[:, i], label=f"mean |{name}|, active taxels", color=color)
    ax.axvline(t, color="black", linestyle="--", linewidth=1, label=f"frame {t}")
    ax.set_xlabel("Pre-action observation frame")
    ax.set_ylabel("Synthetic TacFF units")
    ax.legend(loc="upper left", fontsize=9)
    other = ax.twinx()
    other.plot(np.arange(1, len(socket_force) + 1), socket_force[:, 2],
               color="#888888", alpha=0.6, label="socket load Z after prior action")
    other.set_ylabel("PhysX socket contact load Z (different units)")
    other.legend(loc="upper right", fontsize=9)
    maxima = max(np.max(np.abs(force[t])), 1e-9)
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        grid = force[t, :, i].reshape(10, 14)
        panel = fig.add_subplot(gs[1, i])
        im = panel.imshow(grid, cmap="RdBu_r", vmin=-maxima, vmax=maxima,
                          aspect="equal", interpolation="nearest")
        panel.set_title(f"frame {t}: {name}")
        panel.set_xlabel("taxel column")
        panel.set_ylabel("taxel row")
        fig.colorbar(im, ax=panel, shrink=0.8)
    fig.suptitle("ContactWorld USB: right-finger force components in robot base", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    selected = force[t]
    mask = active[t]
    report = {
        "capture": str(args.capture), "frame": t,
        "scope": "right finger only", "sample_timing": "TacFF pre-action; socket force post-action",
        "active_taxels": int(mask.sum()),
        "mean_abs_base_xyz_active": np.mean(np.abs(selected[mask]), axis=0).tolist(),
        "sum_base_xyz": selected.sum(axis=0).tolist(),
        "local_abs_normal_shearx_sheary": np.mean(np.abs(local[t]), axis=(0, 1)).tolist(),
        "socket_force_post_previous_frame_xyz": socket_force[max(0, t - 1)].tolist(),
        "plot": str(args.output),
        "limitation": "TacFF uses synthetic penetration/velocity model; its units are not PhysX contact-load units.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
