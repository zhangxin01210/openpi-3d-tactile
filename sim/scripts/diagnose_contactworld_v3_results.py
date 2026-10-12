#!/usr/bin/env python3
"""Offline pose-decomposed diagnosis of an existing V3 evaluation matrix."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw


LABELS = ["RGB", "+P", "+local T", "+P/local T", "summary T", "base structured",
          "base pooled", "base map", "split axes", "matched map", "prefix", "split route", "local 3 queries"]
CASES = {10014: (4, 13), 10017: (2, 13), 10019: (6, 12), 10034: (1, 12), 10029: (4, 5)}
CASE_NOTES = {
    10014: "04 在约 18 mm 的高度受阻，夹爪继续下移；13 在接触后继续推进并通过。两者在接触前的姿态已经不同，尚不能把差异归因于更多 token 带来的触觉纠偏。",
    10017: "02 和 13 都曾通过，最终根部距离均约 0.9 mm；02 最终倾角约 3.9°，13 约 1.0°。这个差异主要落在姿态与成功阈值上，不能直接称为 02 退出或掉落。",
    10019: "06 是本 seed 唯一最终通过的配置；12 的轨迹在接触前就明显偏离目标。此案例体现接近路径差异，不能用来证明某种接入位置更擅长接触纠偏。",
    10034: "全部 13 个模型未通过，最终仍存在明显横向偏差。这是接近目标失败的共同困难初态，应单独诊断视觉定位与动作分布覆盖。",
    10029: "05 粗略触觉统计是本 seed 唯一最终通过的配置；04 完整局部触觉图停在孔口附近。它反对“更丰富的触觉表示必然更好”的简单推断，但不能证明统计量对所有接触恢复都足够。",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    out = args.root / "diagnosis"
    out.mkdir(exist_ok=True)
    paths = sorted(args.root.glob("pi0_cw3_usb_*/summary.json"))
    config = json.loads((paths[0].parent / "task_config.json").read_text())
    threshold = config["rl"]["close_error_thresh"] * 1000
    offsets = np.linspace(-.5, .5, config["rl"]["num_keypoints"]) * config["rl"]["keypoint_scale"]
    assert config["rl"]["insertion_frac"] == 1
    angle_cut = np.degrees(2 * np.arcsin(threshold / (2000 * np.abs(offsets).mean())))
    records, trajectories, names, overview = [], {}, {}, []
    max_reconstruction_error = 0.
    for path in paths:
        summary = json.loads(path.read_text())
        number = int(path.parent.name.split("_usb_")[1][:2])
        names[number] = path.parent.name
        for result in summary["results"]:
            seed = result["seed"]
            rows = json.loads((path.parent / f"seed_{seed}/trajectory.json").read_text())["rows"]
            trajectories[number, seed] = rows
            translation = np.asarray([r["plug_pos"] for r in rows]) - np.asarray([r["socket_pos_gt"] for r in rows])
            axes = np.tile([0., 0., 1.], (len(rows), 1))
            plug_axis = Rotation.from_quat([r["plug_quat"] for r in rows]).apply(axes)
            socket_axis = Rotation.from_quat([r["socket_quat"] for r in rows]).apply(axes)
            angular_offsets = offsets[None, :, None] * (plug_axis - socket_axis)[:, None, :]
            rebuilt = np.linalg.norm(translation[:, None, :] + angular_offsets, axis=-1).mean(1) * 1000
            measured = np.asarray([r["keypoint_error_mm"] for r in rows])
            max_reconstruction_error = max(max_reconstruction_error, float(np.max(np.abs(rebuilt - measured))))
            angular_error = np.linalg.norm(angular_offsets, axis=-1).mean(1) * 1000
            root_error = np.linalg.norm(translation, axis=-1) * 1000
            success = np.asarray([r["source_success"] for r in rows])
            assert np.array_equal(success, measured < threshold)
            assert bool(success.any()) == result["source_ever"]
            contact = np.linalg.norm([r["plug_socket_contact_force_xyz"] for r in rows], axis=-1)
            contact_ix = np.flatnonzero(contact > .1)
            first_contact = int(contact_ix[0]) if len(contact_ix) else None
            angular_fail, translation_fail = angular_error[-1] >= threshold, root_error[-1] >= threshold
            failure_component = ("success" if success[-1] else "both" if angular_fail and translation_fail else
                                 "orientation_only" if angular_fail else "translation_only" if translation_fail else
                                 "combined")
            row = {
                "model": number, "seed": seed, "ever": bool(success.any()), "final": bool(success[-1]),
                "min_error_mm": float(measured.min()), "final_error_mm": float(measured[-1]),
                "final_root_mm": float(root_error[-1]), "final_angular_only_mm": float(angular_error[-1]),
                "final_axis_deg": rows[-1]["unsigned_axis_error_deg"],
                "final_z_mm": rows[-1]["plug_socket_z_mm"], "final_xy_mm": rows[-1]["plug_socket_xy_error_mm"],
                "failure_component": failure_component, "longest_success_streak": result["max_consecutive_source_steps"],
                "first_contact_step": first_contact, "first_success_step": result["first_success_step"],
                "ee_down_after_contact_mm": (rows[first_contact]["ee_pos"][2] - rows[-1]["ee_pos"][2]) * 1000
                if first_contact is not None else None,
                "plug_down_after_contact_mm": (rows[first_contact]["plug_pos"][2] - rows[-1]["plug_pos"][2]) * 1000
                if first_contact is not None else None,
            }
            records.append(row)
        group = [r for r in records if r["model"] == number]
        overview.append({"model": number, "label": LABELS[number - 1], "n": len(group),
                         "ever": sum(r["ever"] for r in group), "final": sum(r["final"] for r in group),
                         "median_best_mm": float(np.median([r["min_error_mm"] for r in group])),
                         "median_final_root_mm": float(np.median([r["final_root_mm"] for r in group])),
                         "median_final_axis_deg": float(np.median([r["final_axis_deg"] for r in group])),
                         "contact_trials": sum(r["first_contact_step"] is not None for r in group)})
    assert max_reconstruction_error < .001
    transient = [r for r in records if r["ever"] and not r["final"]]
    lip = [r for r in records if not r["ever"] and 15 <= r["final_z_mm"] <= 23
           and r["first_contact_step"] is not None]
    report = {
        "overview": overview, "source_threshold_mm": threshold, "keypoint_span_m": float(offsets[-1] - offsets[0]),
        "perfect_translation_axis_threshold_deg": float(angle_cut),
        "max_reconstructed_keypoint_error_difference_mm": max_reconstruction_error,
        "final_failure_components": {k: sum(r["failure_component"] == k for r in records)
                                     for k in ("orientation_only", "translation_only", "both", "combined")},
        "transient_count": len(transient),
        "transient_final_root_lt2mm": sum(r["final_root_mm"] < 2 for r in transient),
        "transient_final_error_range_mm": [min(r["final_error_mm"] for r in transient),
                                           max(r["final_error_mm"] for r in transient)],
        "never_success": sum(not r["ever"] for r in records),
        "never_success_final_z15to23mm": len(lip),
        "lip_ee_down_median_mm": float(np.median([r["ee_down_after_contact_mm"] for r in lip])),
        "lip_plug_down_median_mm": float(np.median([r["plug_down_after_contact_mm"] for r in lip])),
        "note": "Height band is an exploratory symptom tag, not proof of collision cause. The original success score is unchanged.",
    }
    (out / "deep_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "per_trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(overview))
    ax.bar(x - .18, [100 * r["ever"] / r["n"] for r in overview], .36, label="Ever source success")
    ax.bar(x + .18, [100 * r["final"] / r["n"] for r in overview], .36, label="Final source success")
    ax.set_xticks(x, [f"{r['model']:02d}\n{r['label']}" for r in overview], rotation=30, ha="right")
    ax.set(ylabel="Success (%)", ylim=(0, 75), title="40 paired resets per model; exploratory results")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "success_rates.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color in (("success", "#31865e"), ("orientation_only", "#d49526"),
                       ("translation_only", "#3378af"), ("both", "#c9494b"), ("combined", "#8563ad")):
        group = [r for r in records if r["failure_component"] == key]
        ax.scatter([r["final_root_mm"] for r in group], [r["final_angular_only_mm"] for r in group],
                   s=14, alpha=.55, color=color, label=f"{key} ({len(group)})")
    ax.axvline(threshold, color="gray", ls="--"); ax.axhline(threshold, color="gray", ls="--")
    ax.set(xscale="symlog", yscale="symlog", xlabel="Translation-only error (mm)",
           ylabel="Orientation-only keypoint error (mm)", title="Final failures: position and axis tilt contributions")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out / "pose_decomposition.png", dpi=160);plt.close(fig)
    for seed, models in CASES.items():
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        for number, color in zip(models, ("#277db5", "#d17427")):
            rows = trajectories[number, seed];t = np.arange(len(rows));label = f"{number:02d} {LABELS[number-1]}"
            axes[0, 0].plot(t, [r["keypoint_error_mm"] for r in rows], color=color, label=label)
            axes[0, 1].plot(t, [r["plug_socket_z_mm"] for r in rows], color=color)
            axes[1, 0].plot(t, [r["unsigned_axis_error_deg"] for r in rows], color=color)
            axes[1, 1].plot(t, [(r["ee_pos"][2] - rows[0]["ee_pos"][2]) * 1000 for r in rows],
                            color=color, label=f"{number:02d} EE")
            axes[1, 1].plot(t, [(r["plug_pos"][2] - rows[0]["plug_pos"][2]) * 1000 for r in rows],
                            color=color, ls="--", label=f"{number:02d} plug")
            axes[2, 0].plot(t, np.linalg.norm([r["plug_socket_contact_force_xyz"] for r in rows], axis=1), color=color)
            q = [r for r in rows if r["tactile_pre_action"] is not None]
            for side, linestyle in (("left", "-"), ("right", "--")):
                axes[2, 1].plot([r["step"] for r in q],
                                [r["tactile_pre_action"][side]["base_abs_mean"][2] for r in q],
                                color=color, ls=linestyle, label=f"{number:02d} {side}")
        axes[0, 0].axhline(threshold, color="gray", ls=":")
        axes[1, 0].axhline(angle_cut, color="gray", ls=":")
        for ax, title in zip(axes.flat, ("Source keypoint error (mm)", "Plug - socket root Z (mm)",
                                        "Axis tilt (degrees)", "Vertical motion since step 0 (mm)",
                                        "Plug/socket pair force magnitude", "Pad mean |base Fz| (synthetic units)")):
            ax.set_title(title, fontsize=10);ax.grid(alpha=.2)
        axes[2, 0].set_yscale("symlog");axes[0, 0].legend(fontsize=8)
        axes[1, 1].legend(fontsize=8);axes[2, 1].legend(fontsize=8)
        axes[2, 0].set_xlabel("Post-action step (0 based)");axes[2, 1].set_xlabel("Step; tactile measured pre-action")
        fig.suptitle(f"Seed {seed}: observed trajectories, not matched-state causal interventions")
        fig.tight_layout();fig.savefig(out / f"case_{seed}_traces.png", dpi=150);plt.close(fig)
        canvas = Image.new("RGB", (2048, 286 * len(models)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_number, number in enumerate(models):
            video = args.root / names[number] / f"seed_{seed}/front_wrist.mp4"
            raw = subprocess.check_output([
                "ffmpeg", "-loglevel", "error", "-i", str(video), "-vf",
                r"select=eq(n\,0)+eq(n\,61)+eq(n\,101)+eq(n\,180)",
                "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ])
            frames = np.frombuffer(raw, np.uint8).reshape(-1, 256, 512, 3)
            for column, frame in enumerate(frames):
                canvas.paste(Image.fromarray(frame), (512 * column, 286 * row_number + 30))
                caption = ["initial", "post step60", "post step100", "post step179"][column]
                draw.text((512 * column + 10, 286 * row_number + 8),
                          f"Model {number:02d} seed {seed}, {caption}", fill="black")
        canvas.save(out / f"case_{seed}.png")
    table = "".join(
        f"<tr><td>{r['model']:02d} {r['label']}</td><td>{r['ever']}/40 ({r['ever']*2.5:g}%)</td>"
        f"<td>{r['final']}/40 ({r['final']*2.5:g}%)</td><td>{r['median_best_mm']:.1f}</td>"
        f"<td>{r['median_final_root_mm']:.1f}</td><td>{r['median_final_axis_deg']:.1f}</td></tr>"
        for r in overview)
    cases_html = ""
    for seed, models in CASES.items():
        links = " · ".join(f'<a href="../{names[n]}/seed_{seed}/front_wrist.mp4">{n:02d} 全程视频</a>'
                           for n in models)
        cases_html += (f"<section><h2>Seed {seed}</h2><p>{CASE_NOTES[seed]}</p><p>{links}</p>"
                       f'<a href="case_{seed}.png"><img src="case_{seed}.png"></a>'
                       f'<a href="case_{seed}_traces.png"><img src="case_{seed}_traces.png"></a></section>')
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ContactWorld V3：40 初态 × 13 模型诊断</title>
<style>body{{font:17px/1.7 sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#243344}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}
td,th{{padding:7px 10px;text-align:left;border-bottom:1px solid #ccd4db}}section{{margin:40px 0}}
a{{color:#12649b}}.note{{background:#eef4f8;padding:16px;border-radius:8px}}</style>
<h1>ContactWorld USB V3：第一轮诊断</h1>
<p>13 个模型、40 个配对初态、520 条完整 180 步轨迹。原始成功评分保持不变。
<a href="integrity.json">完整性核对</a> · <a href="deep_analysis.json">诊断统计</a> ·
<a href="per_trajectory.csv">逐轨迹指标</a> · <a href="../analysis.json">配对差异与置信区间</a></p>
<p class="note">当前没有一个预设配对成功率差异的 bootstrap 95% 区间排除零。
这一轮最有信息量的发现是区分接近失败、孔口受阻和成功阈值附近的姿态漂移。
40 个初态用于筛选机制；每个配置只有一个训练结果，尚未估计训练随机性的影响。</p>
<h2>成功率与连续几何误差</h2><img src="success_rates.png">
<table><tr><th>配置</th><th>曾通过</th><th>最终通过</th><th>最小关键点误差中位数 mm</th>
<th>最终根部距离中位数 mm</th><th>最终倾角中位数 °</th></tr>{table}</table>
<h2>先读懂成功判据</h2>
<p>源码使用沿插件轴线、跨度 {report['keypoint_span_m']:.1f} m 的四个虚拟关键点，平均距离阈值
{threshold:.4f} mm。根部位置完全重合时，轴线倾角超过约 {angle_cut:.2f}° 就会失败。
用每帧保存的插件/插座位姿重建全部关键点误差，与原记录的最大差异只有
{max_reconstruction_error:.6f} mm。该判据没有独立验证接触拓扑，也不辨认绕轴旋转。</p>
<p>曾通过但最终未通过共 {len(transient)} 条，其中 {report['transient_final_root_lt2mm']} 条最终根部距离仍小于
2 mm；这组最终关键点误差范围 {report['transient_final_error_range_mm'][0]:.2f}–
{report['transient_final_error_range_mm'][1]:.2f} mm。应把姿态漂移与真正退出/掉落区分开，不能把最终失败数都解释成失去插入。</p>
<img src="pose_decomposition.png">
<h2>最常见的接触后症状</h2>
<p>{report['never_success']} 条从未通过的轨迹中，{len(lip)} 条最终插件相对根部高度在 15–23 mm。
这组从首次接触到结束的夹爪下移中位数为 {report['lip_ee_down_median_mm']:.1f} mm，插件下移中位数仅
{report['lip_plug_down_median_mm']:.1f} mm。它支持“持续下压却没有推进插件”的症状判断；
高度分组是事后诊断，不能直接证明是摩擦不足还是特定方向卡阻。</p>
<h2>对原有假说的判断</h2>
<ul><li>01/02/03/04：单加点云或局部触觉图未显示清楚的成功率收益。04 对 02 的最终成功净增 5 条；
两者都曾通过的 10 个 seed 中，04 最终通过 10 条、02 为 6 条。值得研究姿态维持，但这还是事后筛选子集。</li>
<li>04/05：统计量 20/13、完整局部图 17/14（曾/最终）。当前不支持接触分布对这批普通插入不可替代；
需要固定状态下打乱 taxel 空间顺序，验证模型是否真正依赖分布。</li>
<li>06/07：合池化 21/15 高于左右分组 16/11，但配对变化双向且区间跨零。结构化分组的优势尚未得到支持。</li>
<li>04/08：局部坐标与 base 坐标力图最终均为 14。统一坐标系在本轮没有显著领先的证据。</li>
<li>08/09/10：拆分 Fx/Fy/Fz 没有领先；09 和容量对照 10 最终都为 13，曾通过分别为 18、22。
目前不能支持“通道耦合不利于学习”。</li>
<li>04/13：增加输出 token 后曾通过增加 3 条、最终减少 2 条。可能改善部分推进过程，但未显示更好的终态控制。</li>
<li>06/11/12：分路 12 最终 15，另外两组均为 11。可作为后续候选；seed 10019 说明其部分退步发生在接触前，
不能把总体差异全部解释为触觉接入动作分支的收益。</li></ul>
<h2>建议的下一轮</h2>
<ol><li>保留本轮原评分，同时查看根部位置、插入高度、轴线倾角及视频。先固定物理完成的诊断协议，避免事后放宽阈值选冠军。</li>
<li>在少量已保存的成功、受阻、姿态漂移状态上固定 RGB、状态、点云和扩散噪声，对触觉做完整/归零/指内 taxel 联合置乱。
05 的统计输入对指内置乱应保持不变，是对照。离线动作变化只证明敏感性；之后需在可重复接触历史下做小规模闭环分支。</li>
<li>由结果决定改法：确实使用触觉但持续压而不进，优先试触觉历史/滑动信息与恢复示范；成功后姿态漂移，试终止或末端稳定训练；
若根本不利用细节，再考虑训练目标或融合方式。暂不依据这 40 条重训全部架构。</li></ol>
{cases_html}</html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
