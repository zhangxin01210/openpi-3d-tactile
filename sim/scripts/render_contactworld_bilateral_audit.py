"""Render orthogonal, equal-scale views of a bilateral USB TacFF replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=63)
    args = parser.parse_args()
    data = np.load(args.capture)
    t = args.frame
    if not 0 <= t < len(data["frame"]):
        raise ValueError("Frame outside capture")
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for column, label in enumerate(("normal", "shear X", "shear Y")):
        l = data["force_grid_left"][t, :, :, column]
        r = data["force_grid_right"][t, :, :, column]
        scale = max(np.max(np.abs(l)), np.max(np.abs(r)), 1e-8)
        for row, (side, field) in enumerate((("left", l), ("right", r))):
            ax = axs[row, column]
            im = ax.imshow(field, cmap="RdBu_r", vmin=-scale, vmax=scale,
                           interpolation="nearest", aspect="equal")
            ax.set_title(f"{side} {label}")
            ax.set_xlabel("taxel column")
            ax.set_ylabel("taxel row")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"ContactWorld USB frame {t}: separate left and right TacFF pads")
    fig.savefig(args.output / "two_pad_fields.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for side, color in (("left", "#bf5879"), ("right", "#267fba")):
        xyz = data["taxel_xyz_base_" + side][t]
        axs[0].scatter(xyz[:, 0], xyz[:, 1], s=14, label=side, color=color, alpha=0.8)
        axs[1].scatter(xyz[:, 0], xyz[:, 2], s=14, label=side, color=color, alpha=0.8)
    for ax, vertical in zip(axs, ("base Y (m)", "base Z (m)"), strict=True):
        ax.set_xlabel("base X (m)")
        ax.set_ylabel(vertical)
        ax.axis("equal")
        ax.legend()
    fig.suptitle(f"Frame {t}: actual 140 taxel positions per pad, robot base")
    fig.savefig(args.output / "two_pad_geometry.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for axis, name in enumerate(("X", "Y", "Z")):
        for side, color in (("left", "#bf5879"), ("right", "#267fba")):
            force = data["force_base_" + side]
            mean_abs = np.mean(np.abs(force[:, :, axis]), axis=1)
            axs[axis].plot(mean_abs, label=side, color=color)
        axs[axis].axvline(t, color="black", linestyle="--", linewidth=1)
        axs[axis].set_ylabel(f"mean |base F{name}|")
        axs[axis].legend()
    axs[-1].set_xlabel("pre-action replay frame")
    fig.suptitle("Both pads: per-taxel base-frame component magnitudes (synthetic units)")
    fig.savefig(args.output / "two_pad_timeline.png", dpi=160)
    plt.close(fig)
    (args.output / "index.html").write_text("""<!doctype html><html lang=\"zh\"><meta charset=\"utf-8\">
<title>ContactWorld 双指触觉单条审查</title><body style=\"font:16px sans-serif;max-width:1200px;margin:auto\">
<h1>ContactWorld USB：单条双指触觉审查</h1>
<p>源示范 001，共 82 帧；仿真重放最终插入成功。这里只审核这一条，不代表整个数据集已通过。</p>
<p>独立仿真副本；左右各 10×14 个采样点，法向加双切向。力值为仿真合成单位，未校准为 N。
左指来自新增传感器；同初态下原有右指、RGB、点云、动作及插件轨迹逐值不变。</p>
<h2>同帧 RGB 与左右力场视频</h2>
<p>上排为前视及左右法向；下排为腕视及左右局部 shear Y。力图使用全条视频固定色标，蓝/红表示相反符号。</p>
<video controls preload=\"metadata\" src=\"bilateral_rgb_tacff_episode_001.mp4\" style=\"width:100%\"></video>
<h2>原始示范与仿真重放</h2>
<p>左半边是发布示范，右半边是本次重放；供核对初态及插入过程。</p>
<video controls preload=\"metadata\" src=\"../../runs/contactworld_bilateral_probe_20261002/episode_001_v2/recorded_left_replay_right.mp4\" style=\"width:100%\"></video>
<h2>左右采样点位置</h2><img src=\"two_pad_geometry.png\" style=\"width:100%\">
<h2>第 63 帧左右三通道力场</h2><img src=\"two_pad_fields.png\" style=\"width:100%\">
<h2>整个示范的 base X/Y/Z 分量</h2><img src=\"two_pad_timeline.png\" style=\"width:100%\">
<p>限制：触觉测量的是 U 盘与夹指胶垫接触，不能把 Z 分量解释为插座轴向力。
大批采集需在这条审核获得确认后再启动。</p>
</body></html>""")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
