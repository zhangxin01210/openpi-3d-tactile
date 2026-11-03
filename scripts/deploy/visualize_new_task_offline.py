"""Render recorded final-v2 inputs and first-observation action comparisons.

Uses saved predictions only: no model resampling, no robot transport.
"""
# ruff: noqa: RUF001 - Chinese report text uses Chinese punctuation.

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess
from scripts.deploy.run_policy_dry_run import _observation_from_npz
from scripts.object_geometry.pose_tracking.rgbtrack_adapter import load_front_calibration

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("pi0_rgb", "pi05_rgb", "pi0_st", "pi0_full")
LABELS = ("Pi0 RGB", "Pi0.5 RGB", "Pi0 ST", "Pi0 Full")
COLORS = ("#008c75", "#db6900", "#6554b4", "#cf3483")
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11), (0, 12, 13, 14), (0, 15, 16, 17))


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def scatter3d(xyz, **kwargs):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], **kwargs)


def draw_spatial(spatial, title, dest):
    visual, tactile, hand = (spatial[key] for key in ("visual", "tactile", "hand_keypoints"))
    xyz = visual["xyz_m"]
    rgb = visual["rgb"].copy()
    rgb[~visual["rgb_valid"]] = [150, 150, 150]
    rgb_strings = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    fig = go.Figure()
    fig.add_trace(
        scatter3d(
            xyz, mode="markers", name="Scene: 4096 points", marker={"size": 2, "color": rgb_strings, "opacity": 0.55}
        )
    )
    fig.add_trace(
        scatter3d(
            tactile["xyz_m"],
            mode="markers",
            name="Tactile: 600 taxels",
            customdata=np.column_stack([tactile["finger_id"], tactile["taxel_id"], tactile["force_norm"]]),
            hovertemplate="finger=%{customdata[0]} taxel=%{customdata[1]}<br>force norm=%{customdata[2]:.4f}<br>xyz=(%{x:.3f},%{y:.3f},%{z:.3f}) m<extra></extra>",
            marker={
                "size": 4,
                "color": tactile["force_norm"],
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": max(float(tactile["force_norm"].max()), 1e-6),
                "colorbar": {"title": "Force norm<br>(raw units)"},
            },
        )
    )
    hand_xyz = hand["xyz_m"]
    fig.add_trace(
        scatter3d(
            hand_xyz,
            mode="markers+text",
            text=list(map(str, range(18))),
            textposition="top center",
            name="Hand FK: 18 anchors",
            marker={"size": 5, "color": "#111111"},
        )
    )
    for i, chain in enumerate(CHAINS):
        fig.add_trace(
            scatter3d(
                hand_xyz[list(chain)],
                mode="lines",
                name="Hand FK bones",
                legendgroup="bones",
                showlegend=i == 0,
                line={"width": 5, "color": "#111111"},
            )
        )
    if "object_geometry" in spatial:
        geom = spatial["object_geometry"]
        for role, color in (("manipulated", "#e84333"), ("target", "#e9af00")):
            mask = geom[f"{role}_mask"]
            fig.add_trace(
                scatter3d(
                    geom[f"{role}_xyz_m"][mask],
                    mode="markers",
                    name=f"{role}: {int(mask.sum())}/512 valid",
                    marker={"size": 3, "color": color, "opacity": 0.85},
                )
            )
    fig.update_layout(
        title=title,
        height=760,
        scene={
            "aspectmode": "data",
            "xaxis_title": "base_link X (m)",
            "yaxis_title": "base_link Y (m)",
            "zaxis_title": "base_link Z (m)",
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.0}},
        },
        legend={"x": 0, "y": 1},
        margin={"l": 0, "r": 0, "t": 65, "b": 0},
    )
    fig.write_html(str(dest.with_suffix(".html")), include_plotlyjs="directory")

    fig_static = plt.figure(figsize=(14, 6))
    for index, close in enumerate((False, True), start=1):
        ax = fig_static.add_subplot(1, 2, index, projection="3d")
        center = hand_xyz.mean(axis=0)
        half = max(float(np.ptp(hand_xyz, axis=0).max()) * 0.7, 0.08)
        visible = np.all(np.abs(xyz - center) <= half, axis=1) if close else np.ones(len(xyz), bool)
        ax.scatter(*xyz[visible].T, s=1, c=rgb[visible] / 255, alpha=0.25)
        ax.scatter(
            *tactile["xyz_m"].T,
            c=tactile["force_norm"],
            s=7,
            cmap="viridis",
            vmin=0,
            vmax=max(float(tactile["force_norm"].max()), 1e-6),
        )
        for chain in CHAINS:
            ax.plot(*hand_xyz[list(chain)].T, color="black", linewidth=1.2)
        ax.scatter(*hand_xyz.T, s=15, c="black")
        if "object_geometry" in spatial:
            for role, color in (("manipulated", "#e84333"), ("target", "#e9af00")):
                geom = spatial["object_geometry"]
                mask = geom[f"{role}_mask"]
                if close:
                    mask = mask & np.all(np.abs(geom[f"{role}_xyz_m"] - center) <= half, axis=1)
                ax.scatter(*geom[f"{role}_xyz_m"][mask].T, s=4, c=color)
        if close:
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Hand close-up" if close else "Scene overview")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-65)
    fig_static.suptitle(title + "\nBlack: FK | red: manipulated | yellow: target | taxels: force colormap", fontsize=11)
    fig_static.tight_layout()
    fig_static.savefig(dest.with_suffix(".png"), dpi=170)
    plt.close(fig_static)


def input_images(packet, spatial, title, path):
    calibration = load_front_calibration(ROOT, None)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), layout="constrained")
    for ax, camera in zip(axs[0], ("front", "left", "right"), strict=True):
        ax.imshow(packet[f"{camera}_rgb"])
        ax.set_title(camera + " RGB (raw)")
    for ax, key, label in zip(
        axs[1, :2],
        ("front_depth_native", "front_depth_color_aligned"),
        ("Native depth optical grid", "Depth registered to RGB"),
        strict=True,
    ):
        depth_m = packet[key] * calibration.depth_scale_m_per_unit
        im = ax.imshow(
            np.ma.masked_where(depth_m <= 0, depth_m),
            cmap=plt.colormaps["turbo"].with_extremes(bad="black"),
            vmin=0.1,
            vmax=1.8,
        )
        ax.set_title(label + " (m; black = missing)")
        ax.set_facecolor("black")
        fig.colorbar(im, ax=ax, shrink=0.7)
    ax = axs[1, 2]
    ax.imshow(packet["front_rgb"])
    geom = spatial["object_geometry"]
    t_color_base = np.linalg.inv(calibration.T_base_color)
    for role, color in (("manipulated", "#ff2525"), ("target", "#ffe000")):
        xyz = geom[f"{role}_xyz_m"][geom[f"{role}_mask"]]
        camera_xyz = xyz @ t_color_base[:3, :3].T + t_color_base[:3, 3]
        camera_xyz = camera_xyz[camera_xyz[:, 2] > 0.01]
        projected = camera_xyz @ calibration.K_color.T
        uv = projected[:, :2] / projected[:, 2:3]
        ax.scatter(uv[:, 0], uv[:, 1], s=2, c=color, label=role, alpha=0.65)
    ax.set_xlim(0, packet["front_rgb"].shape[1])
    ax.set_ylim(packet["front_rgb"].shape[0], 0)
    ax.set_title("Estimated mesh surface points\nprojected to front RGB")
    ax.legend(loc="lower left", fontsize=8)
    for ax in axs.flat:
        ax.set_axis_off()
    fig.suptitle(title + " | baseline_unprofiled; object poses are estimates", fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def action_figures(predictions, gt, measured, names, title, directory):
    seconds = np.arange(len(gt)) / 15
    fig, axs = plt.subplots(6, 3, figsize=(15, 19), sharex=True)
    for j, ax in enumerate(axs.flat):
        ax.axhline(0, color="#999999", linestyle=":", linewidth=1, label="Hold current")
        ax.plot(seconds, np.rad2deg(gt[:, j] - measured[j]), color="black", linewidth=2.2, label="Dataset GT")
        for method, label, color in zip(METHODS, LABELS, COLORS, strict=True):
            ax.plot(
                seconds, np.rad2deg(predictions[method][:, j] - measured[j]), color=color, linewidth=1.4, label=label
            )
        ax.set_title(names[j], fontsize=10)
        ax.grid(alpha=0.2)
        if j % 3 == 0:
            ax.set_ylabel("Target - measured q (deg)")
        if j >= 15:
            ax.set_xlabel("Future action time (s)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.97), fontsize=10)
    fig.suptitle(
        title
        + " | first observation: complete 50-step action chunk\nAll curves use absolute targets minus the SAME measured joint state",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(directory / "action_chunk_first_observation.png", dpi=155)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    for ax, ids, group in ((axs[0], np.arange(6), "Arm 6"), (axs[1], np.arange(6, 18), "Hand 12")):
        ax.axhline(0, color="black", linewidth=1)
        x = np.arange(len(ids))
        ax.plot(
            x,
            np.rad2deg(measured[ids] - gt[0, ids]),
            linestyle=":",
            color="#888888",
            marker="x",
            label="Hold-current error",
        )
        for k, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS, strict=True)):
            ax.bar(
                x + (k - 1.5) * 0.18,
                np.rad2deg(predictions[method][0, ids] - gt[0, ids]),
                width=0.18,
                color=color,
                label=label,
            )
        ax.set_xticks(x, [names[i] for i in ids], rotation=25, ha="right")
        ax.set(ylabel="Predicted first target - dataset first target (deg)", title=group)
        ax.grid(axis="y", alpha=0.2)
    axs[0].legend(ncol=5, fontsize=9)
    fig.suptitle(title + " | signed error of the FIRST predicted action (step 0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(directory / "action_first_step_error.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "output/deploy/new_tasks_latest_20261102")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = run / "visualizations"
    output.mkdir(exist_ok=True)
    report = {}
    for task in ("1028_press_button", "1030_yellow_socket"):
        short = task[:4]
        folder = output / task
        folder.mkdir(exist_ok=True)
        title = "Press button" if short == "1028" else "Insert into yellow socket"
        with np.load(run / f"perception_{short}/perception/frames/frame_000000.npz") as data:
            packet = dict(data)
            prompt = json.loads((ROOT / "data" / task / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
            obs = _observation_from_npz(data, prompt=prompt)
        spatial = XHandSpatialOnlinePreprocess(
            camera_roles=("front",), require_object_geometry=True, require_hand_keypoints=True
        )(obs)["spatial"]
        input_images(packet, spatial, title + ": episode 0, frame 0", folder / "inputs_first_frame.png")
        draw_spatial(spatial, title + ": episode 0, frame 0", folder / "spatial_first_frame")
        np.savez_compressed(
            folder / "spatial_first_frame.npz",
            **{f"{branch}__{key}": value for branch, values in spatial.items() for key, value in values.items()},
        )
        raw_path = run / "packets" / task / "ep000_f000000.npz"
        with np.load(raw_path) as raw:
            gt = raw["target_actions"].copy()
            state = raw["observation.state"].copy()
        measured = np.r_[state[:6], state[28:52:2]]
        names = json.loads((ROOT / "data" / task / "meta/info.json").read_text())["features"]["action"]["names"]
        predictions = {}
        for method in METHODS:
            path = (
                run / f"full_{short}/actions_000000.npz"
                if method == "pi0_full"
                else run / f"{task}_{method}/actions.npz"
            )
            with np.load(path) as saved:
                predictions[method] = saved["actions" if method == "pi0_full" else "ep000_f000000"].copy()
        action_figures(predictions, gt, measured, names, title, folder)
        metrics = []
        for method, label in zip(METHODS, LABELS, strict=True):
            error = np.abs(predictions[method] - gt)
            metrics.append(
                {
                    "method": method,
                    "label": label,
                    "first_mae_rad": float(error[0].mean()),
                    "first_arm_mae_rad": float(error[0, :6].mean()),
                    "first_hand_mae_rad": float(error[0, 6:].mean()),
                    "first_max_error_rad": float(error[0].max()),
                    "chunk_mae_rad": float(error.mean()),
                    "chunk_arm_mae_rad": float(error[:, :6].mean()),
                    "chunk_hand_mae_rad": float(error[:, 6:].mean()),
                }
            )
        hold_error = np.abs(measured - gt)
        baseline = {
            "first_mae_rad": float(hold_error[0].mean()),
            "first_arm_mae_rad": float(hold_error[0, :6].mean()),
            "first_hand_mae_rad": float(hold_error[0, 6:].mean()),
            "chunk_mae_rad": float(hold_error.mean()),
        }
        contact = max(
            json.loads((run / f"{task}_pi0_st/result.json").read_text())["spatial_preprocessing"],
            key=lambda row: row["force_norm_max"],
        )
        with np.load(run / "packets" / task / contact["packet"]) as raw:
            contact_obs = {key: raw[key] for key in raw.files if key.startswith("observation.")}
            contact_obs.update(
                frame_index=int(raw["frame_index"]),
                timestamp_s=float(raw["timestamp_s"]),
                episode_index=int(raw["episode_index"]),
            )
        contact_spatial = XHandSpatialOnlinePreprocess(camera_roles=("front",), require_hand_keypoints=True)(
            contact_obs
        )["spatial"]
        draw_spatial(
            contact_spatial,
            title + ": contact sample " + contact["packet"] + " (no object geometry replay)",
            folder / "spatial_contact",
        )
        plt.imsave(folder / "contact_front_rgb.png", contact_obs["observation.images.cam_front"])
        data = {
            "task": task,
            "episode": 0,
            "frame": 0,
            "prompt": prompt,
            "metrics": metrics,
            "hold_baseline": baseline,
            "joint_names": names,
            "measured": measured.tolist(),
            "ground_truth": gt.tolist(),
            "predictions": {key: value.tolist() for key, value in predictions.items()},
            "first_frame_force_norm_max": float(spatial["tactile"]["force_norm"].max()),
            "contact_example": contact,
            "calibration": "baseline_unprofiled; training parity not established",
        }
        save_json(folder / "comparison.json", data)
        report[task] = data
        rows = "".join(
            f"<tr><td>{r['label']}</td><td>{r['first_mae_rad']:.5f}</td><td>{r['first_arm_mae_rad']:.5f}</td><td>{r['first_hand_mae_rad']:.5f}</td><td>{r['chunk_mae_rad']:.5f}</td></tr>"
            for r in metrics
        )
        joint_rows = "".join(
            "<tr><td>"
            + html.escape(name)
            + "</td>"
            + "".join(
                f"<td>{value:.5f}</td>" for value in [measured[j], gt[0, j], *[predictions[m][0, j] for m in METHODS]]
            )
            + "</tr>"
            for j, name in enumerate(names)
        )
        page = f"""<!doctype html><meta charset='utf-8'><title>{title} offline review</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#1e293b}}img{{width:100%;height:auto}}iframe{{width:100%;height:790px;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#fff5d8;padding:16px;border-radius:8px}}details{{margin:24px 0}}</style>
<a href='../index.html'>All tasks</a><h1>{title}: 输入与第一帧动作检查</h1>
<p>Episode 0 / frame 0. Prompt: {html.escape(prompt)}</p>
<p class='note'>真实记录输入 + 已保存的真实 checkpoint 输出。物体 pose 来自实际 SAM2/RGBTrack，属于估计值。标定为 baseline_unprofiled，未验证训练标定一致性。当前帧触觉力最大值为 {data["first_frame_force_norm_max"]:.4f}（原始单位），因此另附有接触信号的帧。动作均为绝对关节目标；第一步接近静止不能证明任务成功。</p>
<h2>原始输入、深度注册和物体投影</h2><img src='inputs_first_frame.png'>
<h2>模型空间输入：可旋转，点击图例开关各层</h2><iframe src='spatial_first_frame.html'></iframe><img src='spatial_first_frame.png'>
<h2>有触觉接触的补充帧：{contact["packet"]}</h2><p>只展示这个时刻由原始 state/depth/RGB 重建的场景、触觉和 FK。此处未运行物体跟踪，不显示物体几何。</p><img style='max-width:640px' src='contact_front_rgb.png'><iframe src='spatial_contact.html'></iframe>
<h2>第一帧预测 vs 数据集动作</h2><p>所有误差为 rad。First = 输入 frame 0 后输出动作 chunk 的 step 0；Chunk = 同一输出的 50 步 vs 数据集 frame 0…49。相同测量状态的 hold 基线：first MAE={baseline["first_mae_rad"]:.5f} rad；chunk MAE={baseline["chunk_mae_rad"]:.5f} rad。每个模型只看一次已保存的随机采样。</p>
<table><tr><th>Model</th><th>First MAE</th><th>First arm MAE</th><th>First hand MAE</th><th>Chunk MAE</th></tr>{rows}</table>
<img src='action_first_step_error.png'><details><summary>第一步逐关节绝对目标（rad）</summary><table><tr><th>Joint</th><th>Measured</th><th>Dataset GT</th><th>Pi0 RGB</th><th>Pi0.5 RGB</th><th>Pi0 ST</th><th>Pi0 Full</th></tr>{joint_rows}</table></details>
<h2>第一帧输出的完整 50 步轨迹</h2><p>曲线减去同一份当前实测关节状态，单位为度；零线是保持当前姿态。没有执行这些动作。不是闭环成功率测试。</p><img src='action_chunk_first_observation.png'>
<p><a href='comparison.json'>Exact comparisons (JSON)</a> · <a href='spatial_first_frame.npz'>Reconstructed spatial inputs (NPZ)</a></p>"""
        (folder / "index.html").write_text(page)
        print(task, json.dumps({"metrics": metrics, "hold": baseline}), flush=True)
    save_json(
        output / "first_frame_summary.json",
        {
            task: {key: value for key, value in data.items() if key in ("metrics", "hold_baseline", "contact_example")}
            for task, data in report.items()
        },
    )
    (output / "index.html").write_text(
        """<!doctype html><meta charset='utf-8'><title>Offline checkpoint input review</title><style>body{font:18px system-ui;max-width:1100px;margin:50px auto;line-height:1.6}a{color:#086f85}img{width:100%}</style><h1>新任务部署：中间输入与第一帧动作</h1><p>实际数据、实际 SAM2/RGBTrack 和 8 个实际 checkpoint。无机器人动作。打开任务可查看 RGB-D、可旋转点云/触觉/FK/物体几何、第一步误差与 50 步预测轨迹。</p><h2><a href='1028_press_button/index.html'>1028 按按钮、拾取并放入盒子</a></h2><a href='1028_press_button/index.html'><img src='1028_press_button/inputs_first_frame.png'></a><h2><a href='1030_yellow_socket/index.html'>1030 红色组件插入黄色插座</a></h2><a href='1030_yellow_socket/index.html'><img src='1030_yellow_socket/inputs_first_frame.png'></a><p>标定采用任务默认 baseline_unprofiled，训练标定一致性未知。离线动作拟合不等于真实任务成功。</p>"""
    )


if __name__ == "__main__":
    main()
