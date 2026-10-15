#!/usr/bin/env python3
"""Render the fixed-initial and rolling-observation neighbor ensemble audit."""

import argparse
import csv
import gzip
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw


def video_frame(path, frame):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode {path} frame {frame}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def n(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--perturb", type=Path, default=Path("/home/sai/zx/openpi-sim-runtime/runs/contactworld_target_perturb_20261003"))
    p.add_argument("--dataset", type=Path, default=Path("data/contactworld_usb_v3_bilateral"))
    a = p.parse_args()
    out = a.root / "training_neighbors/ensemble_audit"
    fixed = json.loads((out / "analysis.json").read_text())
    dynamic = json.loads((out / "dynamic_analysis.json").read_text())
    perturb = json.loads((a.perturb / "analysis/dynamic_neighbor_switches.json").read_text())
    with (out / "dynamic_per_trajectory.csv").open(newline="") as stream:
        trajectories = list(csv.DictReader(stream))
    lookup = {(int(r["seed"]), int(r["model"]), r["retrieval"], r["phase"], r["setting"]): r
              for r in trajectories}
    model_paths = sorted(a.root.glob("pi0_cw3_usb_*/summary.json"))
    outcome = {}
    for model, path in enumerate(model_paths, 1):
        for result in json.loads(path.read_text())["results"]:
            outcome[(int(result["seed"]), model)] = bool(result["source_ever"])
    rng = np.random.default_rng(20261003)
    contrasts = []
    for retrieval in ("geometry", "pixel"):
        for phase in ("pre", "post"):
            for left, right in (("uniform10", "random10"), ("uniform10", "fixed_source10"),
                                ("distance10", "uniform10")):
                for metric in ("cos3", "rms_mm"):
                    seed_deltas = []
                    for seed in range(10000, 10040):
                        model_deltas = []
                        for model in range(1, 14):
                            x = lookup.get((seed, model, retrieval, phase, left))
                            y = lookup.get((seed, model, retrieval, phase, right))
                            if x and y and x[metric] and y[metric]:
                                model_deltas.append(float(x[metric]) - float(y[metric]))
                        if model_deltas:
                            seed_deltas.append(float(np.mean(model_deltas)))
                    v = np.asarray(seed_deltas)
                    b = rng.integers(len(v), size=(5000, len(v)))
                    contrasts.append({"retrieval": retrieval, "phase": phase, "contrast": f"{left} minus {right}",
                                      "metric": metric, "seeds": len(v), "mean": float(v.mean()),
                                      "bootstrap95": np.quantile(v[b].mean(axis=1), [.025, .975]).tolist()})
    (out / "pairwise_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    case_rows = []
    for seed in range(10000, 10040):
        for model in range(1, 14):
            values = {}
            for retrieval in ("geometry", "pixel"):
                for phase in ("pre", "post"):
                    x = lookup.get((seed, model, retrieval, phase, "uniform10"))
                    y = lookup.get((seed, model, retrieval, phase, "nearest1"))
                    values[f"{retrieval}_{phase}"] = (float(x["cos3"]) - float(y["cos3"])) if x and y and x["cos3"] and y["cos3"] else None
                    values[f"{retrieval}_{phase}_switches"] = int(x["top1_switches"]) if x else None
            video = f"../../{model_paths[model-1].parent.name}/seed_{seed}/front_wrist.mp4"
            case_rows.append(f"<tr><td>{seed}</td><td>{model:02d}</td><td>{'yes' if outcome[(seed, model)] else 'no'}</td>"
                             f"<td>{n(values['geometry_pre'])}</td><td>{n(values['geometry_post'])}</td>"
                             f"<td>{n(values['pixel_pre'])}</td><td>{n(values['pixel_post'])}</td>"
                             f"<td>{n(values['geometry_pre_switches'],0)}/{n(values['geometry_post_switches'],0)}</td>"
                             f"<td>{n(values['pixel_pre_switches'],0)}/{n(values['pixel_post_switches'],0)}</td>"
                             f"<td><a href='{video}'>video</a></td></tr>")
    cases_page = """<!doctype html><html lang='zh'><meta charset='utf-8'><title>520 cases: neighbor ensemble</title>
    <style>body{font:15px/1.5 system-ui;margin:25px;color:#243348}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#eef3f8}input{padding:8px;min-width:300px}</style>
    <h1>520 条轨迹的动态近邻集成增益</h1><p><a href='index.html'>返回总览</a>。数值为 10 条平均减去单近邻的 3D 平移动作余弦；正值说明更接近策略。来源切换数按接触前／后分别统计。缺失接触则后段为 —。</p>
    <input id='q' placeholder='筛选种子、模型、成功与否' oninput='filter()'><table><thead><tr><th>seed</th><th>model</th><th>ever</th><th>几何前增益</th><th>几何后增益</th><th>图像前增益</th><th>图像后增益</th><th>几何切换前/后</th><th>图像切换前/后</th><th>视频</th></tr></thead><tbody>""" + "".join(case_rows) + """</tbody></table><script>function filter(){let q=document.getElementById('q').value.trim().toLowerCase();for(let r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(q)}</script></html>"""
    (out / "case_table.html").write_text(cases_page)
    plt.rcParams.update({"font.size": 10})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = ["nearest1", "uniform3", "uniform5", "uniform10", "random10"]
    colors = {"nearest1": "#102f5e", "uniform3": "#3174af", "uniform5": "#32a6a1",
              "uniform10": "#d98e2b", "random10": "#999999"}
    for col, retrieval in enumerate(("geometry", "pixel")):
        for row, phase in enumerate(("pre", "post")):
            ax = axs[row, col]
            vals = [dynamic["summary"][f"{retrieval}/{phase}/{label}"]["cos3"] for label in labels]
            ax.bar(range(len(labels)), vals, color=[colors[x] for x in labels])
            ax.set_xticks(range(len(labels)), ["1", "3", "5", "10", "random 10"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Mean action direction cosine")
            ax.set_title(f"{retrieval} retrieval · {phase} contact")
            ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "dynamic_cosine.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, retrieval in enumerate(("geometry", "pixel")):
        for row, phase in enumerate(("pre", "post")):
            ax = axs[row, col]
            vals = [dynamic["summary"][f"{retrieval}/{phase}/{label}"]["rms_mm"] for label in labels]
            ax.bar(range(len(labels)), vals, color=[colors[x] for x in labels])
            ax.set_xticks(range(len(labels)), ["1", "3", "5", "10", "random 10"])
            ax.set_ylabel("4-step translation action RMS error (mm)")
            ax.set_title(f"{retrieval} retrieval · {phase} contact")
            ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "dynamic_rms.png", dpi=160)
    plt.close(fig)
    with gzip.open(out / "dynamic_observations.csv.gz", "rt", newline="") as stream:
        observations = [r for r in csv.DictReader(stream)
                        if r["setting"] == "nearest1" and int(r["seed"]) in (10005, 10017, 10034)
                        and int(r["model"]) in (1, 13)]
    with gzip.open(out / "dynamic_observations.csv.gz", "rt", newline="") as stream:
        montage_rows = [r for r in csv.DictReader(stream)
                        if r["setting"] == "uniform3" and int(r["seed"]) in (10005, 10017)
                        and int(r["model"]) in (1, 13)]
    manifest = json.loads((a.dataset / "spatial/manifest.json").read_text())
    source_to_ep = {int(e["source_episode"]): int(e["episode_index"]) for e in manifest["episodes"]}
    gallery = []
    for seed in (10005, 10017):
        for model in (1, 13):
            model_dir = sorted(a.root.glob("pi0_cw3_usb_*/summary.json"))[model-1].parent
            trial_dir = model_dir / f"seed_{seed}"
            with np.load(trial_dir / "inputs.npz") as z:
                steps = z["steps"].copy()
                test_front = z["front"].copy()
                test_wrist = z["wrist"].copy()
            group = [r for r in montage_rows if int(r["seed"]) == seed and int(r["model"]) == model]
            if not group:
                continue
            contact = int(group[0]["contact_step"]) if group[0]["contact_step"] else None
            choices = [0]
            if contact is not None:
                choices.append(next((i for i, step in enumerate(steps) if int(step) >= contact), len(steps)-1))
            for oi in sorted(set(choices)):
                step = int(steps[oi])
                canvas = Image.new("RGB", (875, 7*264), "white")
                draw = ImageDraw.Draw(canvas)
                records = [("test", test_front[oi], test_wrist[oi])]
                for retrieval in ("geometry", "pixel"):
                    rr = next(r for r in group if r["retrieval"] == retrieval and int(r["step"]) == step)
                    for rank, (source, frame) in enumerate(zip(map(int, rr["sources"].split(";")),
                                                               map(int, rr["frames"].split(";"))), 1):
                        ep = source_to_ep[source]
                        pfx = a.dataset / "videos/chunk-000"
                        name = f"episode_{ep:06d}.mp4"
                        front = video_frame(pfx / "observation.images.front" / name, frame)
                        wrist = video_frame(pfx / "observation.images.wrist" / name, frame)
                        records.append((f"{retrieval} #{rank} source {source} frame {frame}", front, wrist))
                for i, (label, front, wrist) in enumerate(records):
                    y = i*264
                    draw.text((8, y+10), label, fill="#172a44")
                    canvas.paste(Image.fromarray(front).resize((256, 256)), (350, y+4))
                    canvas.paste(Image.fromarray(wrist).resize((256, 256)), (612, y+4))
                name = f"seed_{seed}_model_{model:02d}_step_{step:03d}_neighbors.jpg"
                canvas.save(out / name, quality=88)
                gallery.append((seed, model, step, name))
    fig, axs = plt.subplots(3, 2, figsize=(14, 9), layout="constrained")
    for row, seed in enumerate((10005, 10017, 10034)):
        for col, model in enumerate((1, 13)):
            ax = axs[row, col]
            for retrieval, color in (("geometry", "#2673a5"), ("pixel", "#d3842d")):
                rr = [r for r in observations if int(r["seed"]) == seed and int(r["model"]) == model
                      and r["retrieval"] == retrieval]
                ax.scatter([int(r["step"]) for r in rr], [int(r["top1_source"]) for r in rr],
                           label=retrieval, s=15, color=color)
                if rr and rr[0]["contact_step"]:
                    ax.axvline(int(rr[0]["contact_step"]), color="black", ls="--", lw=1)
            ax.set(xlabel="Policy observation step", ylabel="Nearest train source ID", title=f"Seed {seed}, model {model:02d}")
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
    fig.savefig(out / "neighbor_switch_timelines.png", dpi=160)
    plt.close(fig)
    fixed_rows = []
    for label in labels:
        x = fixed["configurations"][label]
        fixed_rows.append(f"<tr><td>{label}</td><td>{n(x['pre']['cos3']['mean_by_seed'])}</td>"
                          f"<td>{n(x['post']['cos3']['mean_by_seed'])}</td>"
                          f"<td>{n(x['pre']['rms_mm']['mean_by_seed'], 2)}</td>"
                          f"<td>{n(x['post']['rms_mm']['mean_by_seed'], 2)}</td>"
                          f"<td>{n(x['pre']['ee_shape_rmse_mm']['mean_by_seed'], 1)}</td>"
                          f"<td>{n(x['post']['ee_shape_rmse_mm']['mean_by_seed'], 1)}</td></tr>")
    dynamic_rows = []
    for retrieval in ("geometry", "pixel"):
        for phase in ("pre", "post"):
            for label in labels + ["distance3", "distance5", "distance10", "fixed_source1", "fixed_source10"]:
                x = dynamic["summary"][f"{retrieval}/{phase}/{label}"]
                delta = x.get("cos3_minus_nearest1")
                ci = x.get("cos3_delta_bootstrap95")
                dynamic_rows.append(f"<tr><td>{retrieval}</td><td>{phase}</td><td>{label}</td>"
                                    f"<td>{n(x.get('cos3'))}</td><td>{n(delta)}"
                                    f"{(' ['+n(ci[0])+', '+n(ci[1])+']') if ci else ''}</td>"
                                    f"<td>{n(x.get('rms_mm'), 2)}</td>"
                                    f"<td>{n(x.get('rms_mm_minus_nearest1'), 2)}</td>"
                                    f"<td>{n(x.get('top1_switches_per_trajectory'), 1)}</td></tr>")
    perturb_rows = []
    for x in perturb:
        perturb_rows.append(f"<tr><td>{html.escape(x['model'].replace('pi0_cw3_usb_', ''))}</td>"
                            f"<td>{x['seed']}</td><td>{x['condition']}</td>"
                            f"<td>{n(x['first_action_delta_over_0.1mm_step'],0)}</td>"
                            f"<td>{n(x['geometry_first_top1_switch_step'],0)}</td>"
                            f"<td>{n(x['pixel_first_top1_switch_step'],0)}</td>"
                            f"<td>{n(x['geometry_mean_top5_overlap'],2)}</td>"
                            f"<td>{n(x['pixel_mean_top5_overlap'],2)}</td></tr>")
    contrast_rows = []
    for x in contrasts:
        contrast_rows.append(f"<tr><td>{x['retrieval']}</td><td>{x['phase']}</td>"
                             f"<td>{x['contrast']}</td><td>{x['metric']}</td>"
                             f"<td>{n(x['mean'])}</td><td>[{n(x['bootstrap95'][0])}, {n(x['bootstrap95'][1])}]</td></tr>")
    gallery_html = "".join(f"<details><summary>seed {seed}, model {model:02d}, step {step}</summary><img loading='lazy' src='{name}'></details>"
                           for seed, model, step, name in gallery)
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>动态近邻与轨迹集成审计</title>
<style>body{{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#243348}}h1,h2{{color:#17385d}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:7px;border-bottom:1px solid #dce4ed;text-align:left}}th{{background:#edf3f9;position:sticky;top:0}}.scroll{{max-height:650px;overflow:auto}}img{{max-width:100%}}.notice{{background:#eff6fc;padding:18px;border-left:4px solid #3276bb}}a{{color:#1c65a3}}</style>
<h1>测试轨迹是否接近多个训练示范的集成？</h1>
<p><a href="../index.html">返回训练近邻总览</a> · <a href="../phase_audit/index.html">接触阶段审计</a> · <a href="../../pi0_cw3_usb_01_rgb/index.html">模型 01 视频</a></p>
<div class="notice"><b>解释边界：</b>动态几何检索使用仿真真值位姿。动态视觉检索使用双相机 16×16 LAB 像素代理：前视权重为腕视的 2 倍，两侧 3 格和底部 2 格权重为 0.2，以免静态背景主导。两者均是事后诊断，不等同于 π0 内部显式检索。接触时间对齐也使用了结果信息。动作平均后的“合成轨迹”不保证物理可执行。</div>
<h2>固定初态近邻，按首次接触重采样</h2>
<p>95 条训练示范，40 个初态 × 13 个模型。初态距离包括目标位置、USB 对插座的相对位姿、USB 对夹爪的相对位姿。每一行用同一批近邻，方向余弦越高越像；RMS 和末端轨迹形状误差越低越像。随机 10 条是检索特异性对照。</p>
<table><tr><th>集成</th><th>接触前余弦</th><th>接触后余弦</th><th>前动作 RMS mm</th><th>后动作 RMS mm</th><th>前末端形状误差 mm</th><th>后末端形状误差 mm</th></tr>{''.join(fixed_rows)}</table>
<h2>动态检索：每次策略观测重选训练帧</h2>
<p>{dynamic['train_frames']} 个训练帧；每 4 步策略观测重新检索，选不同训练示范的最相似帧，比较对应的 4 步动作。对跨越接触边界的动作窗不作接触前后归类。误差统计先在每条轨迹内平均，再按 40 个初态汇总；区间为按初态 bootstrap 的相对单近邻差值 95% 区间。最后一列仅显示该检索法动态单近邻来源的切换次数，与行内集成设置无关。</p>
<img src="dynamic_cosine.png" alt="动态检索动作方向余弦"><img src="dynamic_rms.png" alt="动态检索动作误差">
<div class="scroll"><table><tr><th>检索</th><th>阶段</th><th>集成</th><th>余弦</th><th>相对单近邻变化 [95%]</th><th>动作 RMS mm</th><th>相对单近邻变化</th><th>每轨迹近邻切换</th></tr>{''.join(dynamic_rows)}</table></div>
<h3>关键配对对照</h3><p>正的余弦差和负的动作误差差表示左侧方法更接近策略。区间按 40 个初态 bootstrap。</p>
<table><tr><th>检索</th><th>阶段</th><th>对照</th><th>指标</th><th>平均差</th><th>95% 区间</th></tr>{''.join(contrast_rows)}</table>
<h2>最近训练示范是否随运动切换？</h2><p>虚线是首次 USB 与插座接触。散点是各策略观测的最近训练示范来源 ID；切换只是相关现象，不代表策略真的在切换示范。</p>
<img src="neighbor_switch_timelines.png" alt="近邻随轨迹变化">
<h3>实际图像与检索图像</h3><p>左为训练来源说明，右边依次是前视、腕视。每张图按当前帧选取不同示范中的前三名；初态和接触附近各取一次。</p>{gallery_html}
<h2>±3 mm 配对试验</h2><p>16 组模型 × 偏移。表中“动作首次变”阈值为 4 步平移动作 RMS 差异 &gt;0.1 mm。“近邻首次变”指相对该模型未偏移轨迹的最近训练来源 ID 不同；后续切换可能由策略早期动作差异引起。</p>
<div class="scroll"><table><tr><th>模型</th><th>种子</th><th>偏移</th><th>动作首次变</th><th>几何近邻首次变</th><th>图像近邻首次变</th><th>几何前5重合均值</th><th>图像前5重合均值</th></tr>{''.join(perturb_rows)}</table></div>
<p><a href="case_table.html">逐条查看 520 个模型轨迹的集成增益、近邻切换与视频</a></p>
<p>下载：<a href="analysis.json">固定初态汇总 JSON</a> · <a href="ensemble_metrics.csv">固定初态逐轨迹 CSV</a> · <a href="dynamic_analysis.json">动态汇总 JSON</a> · <a href="dynamic_per_trajectory.csv">动态逐轨迹 CSV</a> · <a href="dynamic_observations.csv.gz">逐观测及匹配帧 CSV.gz</a> · <a href="pairwise_contrasts.json">配对对照 JSON</a> · <a href="perturb_neighbor_switches.json">初态扰动近邻 JSON</a></p>
</html>"""
    (out / "index.html").write_text(page)
    print(out / "index.html")


if __name__ == "__main__":
    main()
