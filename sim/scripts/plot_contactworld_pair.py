#!/usr/bin/env python3
"""Plot aligned trajectories, contact and actions for a paired USB seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    for config in args.configs:
        path = args.root / config / f"seed_{args.seed}" / "trajectory.json"
        rows = json.loads(path.read_text())["rows"]
        time = np.array([r["time_s"] for r in rows])
        error = np.array([r["keypoint_error_mm"] for r in rows])
        contact = np.array([np.linalg.norm(r["plug_socket_contact_force_xyz"]) for r in rows])
        force = np.array([r["tacff_mean_norm"] for r in rows])
        action = np.array([r["action_raw"] for r in rows])
        label = config.removeprefix("pi0_cw_usb_")
        axes[0].plot(time, error, label=label)
        axes[1].plot(time, contact, label=label)
        axes[2].plot(time, force, label=label)
        axes[3].plot(time, np.linalg.norm(action[:, :3], axis=1), label=label)
        hits = np.flatnonzero([r["source_success"] for r in rows])
        if len(hits):
            axes[0].scatter(time[hits[0]], error[hits[0]], s=35)
    axes[0].axhline(7.9916, color="black", linestyle="--", linewidth=1, label="source threshold")
    for ax, title in zip(axes, ("Mean plug/socket keypoint error (mm)",
                                "Plug/socket contact-force norm (sim units)",
                                "Mean TacFF norm (raw units)",
                                "Translation action norm (source units)"), strict=True):
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Simulated time (s)")
    fig.suptitle(f"ContactWorld USB paired seed {args.seed}")
    output = args.output or args.root / f"seed_{args.seed}_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
