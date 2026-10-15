#!/usr/bin/env python3
"""One-command resumable overnight action-predictability and pi0 pairing audit.

Run with the existing repository .venv. No downloads, simulator, or pi0 training.
Each major stage/model runs in a fresh process so Torch and JAX release memory.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNTIME = Path("/home/sai/zx/openpi-sim-runtime/runs")
EVAL = RUNTIME / "contactworld_pi0_v3_13models_40trials_20261003"
OUTPUT = RUNTIME / "contactworld_neighbor_night_20261003"


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def snapshots(path):
    keys = ("front", "wrist", "state", "cloud_xyz", "taxel_xyz_base", "force_base", "force_local", "steps")
    with np.load(path) as z:
        return {k: z[k].copy() for k in keys}


def observed_features(extractor, snap):
    return extractor.observation(**{k: snap[k] for k in
        ("front", "wrist", "state", "cloud_xyz", "taxel_xyz_base", "force_base", "force_local")})


def evaluation_trials(args):
    paths = sorted(args.eval_root.glob("pi0_cw3_usb_*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"No evaluation summaries in {args.eval_root}")
    if args.smoke:
        paths = paths[:1]
    trials = []
    for path in paths:
        results = json.loads(path.read_text())["results"]
        if args.smoke:
            results = results[:1]
        for result in results:
            trials.append((path.parent.name, int(result["seed"]), path.parent / f"seed_{result['seed']}"))
    return trials


def stage_rollouts(args):
    from contactworld_neighbor_features import Features, save_npz, choose_device
    from contactworld_neighbor_weights import predict_observations
    train = load_npz(args.output / "features/train_features.npz")
    extractor = None
    device = choose_device(args.device)
    trial_list = evaluation_trials(args)
    for number, (config, seed, folder) in enumerate(trial_list, 1):
        cache = args.output / "eval_features" / config / f"seed_{seed}.npz"
        if not cache.exists():
            if extractor is None:
                extractor = Features(args.vision_weights, args.device, args.batch)
            snap = snapshots(folder / "inputs.npz")
            if args.smoke:
                snap = {k: v[:2] for k, v in snap.items()}
            features = observed_features(extractor, snap)
            trajectory = json.loads((folder / "trajectory.json").read_text())
            rows = trajectory["rows"]
            act = np.array([r["action_sim_clipped"] for r in rows], np.float32)
            steps = snap["steps"]
            target = np.stack([act[int(s):int(s)+4] for s in steps])
            if target.shape != (len(steps), 4, 6):
                raise ValueError(f"Incomplete four-step window: {folder}")
            hit = np.flatnonzero(np.linalg.norm([r["plug_socket_contact_force_xyz"] for r in rows], axis=1) > .1)
            contact = int(hit[0])+1 if len(hit) else len(rows)+1
            save_npz(cache, **features, target=target, step=steps, phase=(steps >= contact).astype(np.int8),
                     seed=np.full(len(steps), seed), ever=np.full(len(steps), bool(trajectory["result"]["source_ever"])))
        obs = load_npz(cache)
        for mode in args.modes:
            target = args.output / "rollout_predictions" / config / f"seed_{seed}_{mode}.npz"
            if target.exists():
                continue
            prediction = predict_observations(train, obs, args.output / f"cv/{mode}/final/model.pt", device)
            save_npz(target, **prediction, target=obs["target"], phase=obs["phase"], step=obs["step"],
                     seed=obs["seed"], ever=obs["ever"])
        print(f"rollout diagnostic {number}/{len(trial_list)} {config} {seed}", flush=True)
    build_anchors(args)


def build_anchors(args):
    from contactworld_neighbor_features import save_npz
    config_paths = sorted(args.eval_root.glob("pi0_cw3_usb_*/summary.json"))
    reference = {config_paths[0].parent.name, config_paths[-1].parent.name}
    anchors, feature_rows = [], []
    for config, seed, folder in evaluation_trials(args):
        if config not in reference:
            continue
        obs = load_npz(args.output / "eval_features" / config / f"seed_{seed}.npz")
        available_steps = snapshots(folder / "inputs.npz")["steps"]
        indices = np.unique(np.rint(np.linspace(0, len(obs["step"])-1,
                                                min(args.anchors_per_trial, len(obs["step"])))).astype(int))
        for i in indices:
            step = int(obs["step"][i])
            snapshot_index = int(np.flatnonzero(available_steps == step)[0])
            anchors.append({"id": len(anchors), "origin_model": config, "seed": seed, "step": step,
                            "snapshot_index": snapshot_index, "inputs": str((folder / "inputs.npz").resolve()),
                            "phase": int(obs["phase"][i]), "origin_ever": bool(obs["ever"][i])})
            feature_rows.append({k: obs[k][i] for k in ("rgb", "state", "cloud", "tactile", "target")})
    (args.output / "anchors.json").write_text(json.dumps(anchors, indent=2)+"\n")
    save_npz(args.output / "anchor_features.npz", **{k: np.stack([r[k] for r in feature_rows]) for k in feature_rows[0]})
    # Predict all input variants on exactly the same anchor observations.
    from contactworld_neighbor_weights import predict_observations
    train = load_npz(args.output / "features/train_features.npz")
    obs = load_npz(args.output / "anchor_features.npz")
    for mode in args.modes:
        file = args.output / f"shared/neighbor_{mode}.npz"
        if not file.exists():
            save_npz(file, **predict_observations(train, obs, args.output / f"cv/{mode}/final/model.pt"))


def stage_shared(args):
    # This process uses the existing JAX pi0 environment, not Torch CUDA.
    from contactworld_neighbor_features import save_npz
    from contactworld_v3_eval_inputs import policy_input
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    anchors = json.loads((args.output / "anchors.json").read_text())
    destination = args.output / "shared" / args.config
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoints / args.config / "cw3_round1_20k/19999"
    config = get_config(args.config)
    policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": 10})
    loaded_path, snap = None, None
    for a in anchors:
        result_path = destination / f"anchor_{a['id']:05d}.npz"
        if result_path.exists():
            continue
        if loaded_path != a["inputs"]:
            snap = snapshots(a["inputs"])
            loaded_path = a["inputs"]
        obs = {k: v[a["snapshot_index"]] for k, v in snap.items() if k != "steps"}
        request = policy_input(obs, args.config)
        noise_seed = args.seed + 100000 + a["id"]
        noise = np.random.default_rng(noise_seed).normal(size=(config.model.action_horizon, config.model.action_dim)).astype(np.float32)
        begin = time.monotonic()
        result = policy.infer(request, noise=noise)
        actions = np.asarray(result["actions"], np.float32)
        if actions.shape != (16, 6) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid inference: {actions.shape}")
        save_npz(result_path, actions=actions, noise_seed=np.array(noise_seed),
                 latency_s=np.array(time.monotonic()-begin), anchor_id=np.array(a["id"]))
        if a["id"] % 20 == 0:
            print(f"{args.config} shared {a['id']+1}/{len(anchors)}", flush=True)
    actions = np.stack([load_npz(destination / f"anchor_{a['id']:05d}.npz")["actions"] for a in anchors])
    save_npz(destination / "complete.npz", actions=actions, anchor_id=np.arange(len(anchors)))


def stage_report(args):
    from contactworld_neighbor_weights import metrics
    rows, case_rows = [], []
    for file in sorted((args.output / "cv").glob("*/fold_*/predictions.npz")):
        z = load_npz(file)
        mode, fold = file.parent.parent.name, int(file.parent.name.split("_")[-1])
        for method in ("mean", "nearest", "fixed", "initialized", "learned"):
            mm = metrics(z[method], z["target"])
            for source in np.unique(z["source"]):
                for phase in (-2, 0, 1):
                    mask = (z["source"] == source) & ((z["phase"] == phase) if phase != -2 else True)
                    if not mask.any():
                        continue
                    r = {"mode": mode, "fold": fold, "source": int(source), "phase": phase, "method": method, "frames": int(mask.sum())}
                    r.update({k: float(np.nanmean(v[mask])) if np.isfinite(v[mask]).any() else None for k, v in mm.items()})
                    rows.append(r)
        mm = metrics(z["learned"], z["target"])
        fm = metrics(z["fixed"], z["target"])
        for i in range(len(z["source"])):
            case_rows.append({"mode": mode, "fold": fold, "source": int(z["source"][i]), "frame": int(z["frame"][i]),
                "phase": int(z["phase"][i]), "translation_mm": float(mm["translation_mm"][i]),
                "fixed_translation_mm": float(fm["translation_mm"][i]), "rotation_deg": float(mm["rotation_deg"][i]),
                "effective_k": float(z["effective_k"][i]), "spread_mm": float(z["spread_translation_mm"][i]),
                "opposing_xy_mass": float(z["opposing_xy_mass"][i])})
    def csv_write(name, records):
        if not records:
            (args.output / name).write_text("status\nno completed rows\n")
            return
        with (args.output / name).open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    csv_write("cv_episode_metrics.csv", rows)
    csv_write("cv_frame_diagnostics.csv", case_rows)
    uncertainty_rows = []
    for mode in args.modes:
        for phase in (-2, 0, 1):
            rr = [r for r in case_rows if r["mode"] == mode and (phase == -2 or r["phase"] == phase)]
            if len(rr) < 4:
                continue
            # Descriptive bins only, not a test-set-fitted decision threshold.
            order = np.argsort([r["spread_mm"] for r in rr])
            for group, indexes in enumerate(np.array_split(order, 4)):
                group_rows = [rr[i] for i in indexes]
                uncertainty_rows.append({"mode": mode, "phase": phase, "spread_quartile": group+1,
                    "frames": len(group_rows), "spread_mm": float(np.mean([r["spread_mm"] for r in group_rows])),
                    "error_mm": float(np.mean([r["translation_mm"] for r in group_rows])),
                    "effective_k": float(np.mean([r["effective_k"] for r in group_rows]))})
    csv_write("uncertainty_bins.csv", uncertainty_rows)
    summary, paired = [], []
    for mode in args.modes:
        for phase in (-2, 0, 1):
            for method in ("mean", "nearest", "fixed", "initialized", "learned"):
                rr = [r for r in rows if (r["mode"], r["phase"], r["method"]) == (mode, phase, method)]
                if not rr:
                    continue
                r = {"mode": mode, "phase": phase, "method": method, "episodes": len(rr)}
                for k in ("translation_mm", "xy_mm", "rotation_deg", "chunk_translation_mm", "cos3"):
                    v = [x[k] for x in rr if x[k] is not None]
                    r[k] = float(np.mean(v)) if v else None
                summary.append(r)
            for baseline in ("fixed", "initialized"):
                rr = {r["source"]: r for r in rows if (r["mode"], r["phase"], r["method"]) == (mode, phase, baseline)}
                learned = [r for r in rows if (r["mode"], r["phase"], r["method"]) == (mode, phase, "learned") and r["source"] in rr]
                for metric in ("translation_mm", "rotation_deg"):
                    d = np.array([r[metric]-rr[r["source"]][metric] for r in learned])
                    if not len(d):
                        continue
                    boot = np.random.default_rng(args.seed).integers(len(d), size=(3000, len(d)))
                    paired.append({"mode": mode, "phase": phase, "baseline": baseline, "metric": metric,
                                   "delta": float(d.mean()), "bootstrap95_episode": np.quantile(d[boot].mean(1), [.025, .975]).tolist()})
    modality_pairs = []
    for richer, base in (("R", "S"), ("RP", "R"), ("RT", "R"), ("RPT", "RP"), ("RPT", "RT")):
        for phase in (-2, 0, 1):
            baseline = {r["source"]: r for r in rows if (r["mode"], r["phase"], r["method"]) == (base, phase, "learned")}
            rich_rows = [r for r in rows if (r["mode"], r["phase"], r["method"]) == (richer, phase, "learned") and r["source"] in baseline]
            for metric in ("translation_mm", "xy_mm", "rotation_deg"):
                d = np.array([r[metric]-baseline[r["source"]][metric] for r in rich_rows])
                if not len(d):
                    continue
                boot = np.random.default_rng(args.seed).integers(len(d), size=(3000, len(d)))
                lo, hi = np.quantile(d[boot].mean(1), [.025, .975])
                modality_pairs.append({"richer": richer, "base": base, "phase": phase, "metric": metric,
                                       "episodes": len(d), "delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)})
    csv_write("modality_pairs.csv", modality_pairs)
    # Own-rollout correspondence, kept separate from held-out human targets.
    rollout_rows = []
    for file in sorted((args.output / "rollout_predictions").glob("*/*.npz")):
        z = load_npz(file)
        for method in ("learned", "initialized", "nearest"):
            m = metrics(z[method], z["target"])
            rollout_rows.append({"policy": file.parent.name, "seed": int(z["seed"][0]), "mode": file.stem.split("_")[-1],
                                 "method": method, **{k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in m.items()}})
    csv_write("rollout_correspondence.csv", rollout_rows)
    shared_rows = []
    anchor_path = args.output / "anchors.json"
    anchors = json.loads(anchor_path.read_text()) if anchor_path.exists() else []
    for file in sorted((args.output / "shared").glob("pi0*/complete.npz")):
        policy_actions = load_npz(file)["actions"][:, :4]
        for mode in args.modes:
            neighbor = load_npz(args.output / f"shared/neighbor_{mode}.npz")
            m = metrics(neighbor["learned"], policy_actions)
            for i, a in enumerate(anchors):
                shared_rows.append({"policy": file.parent.name, "neighbor_mode": mode, "anchor_id": a["id"],
                                   "origin": a["origin_model"], "seed": a["seed"], "step": a["step"], "phase": a["phase"],
                                   **{k: float(v[i]) if np.isfinite(v[i]) else None for k, v in m.items()}})
    csv_write("shared_observation_correspondence.csv", shared_rows)
    shared_files = sorted((args.output / "shared").glob("pi0*/complete.npz"))
    model_pairs = []
    for i, left in enumerate(shared_files):
        a = load_npz(left)["actions"][:, :4]
        for right in shared_files[i+1:]:
            b = load_npz(right)["actions"][:, :4]
            mm = metrics(a, b)
            for metric in ("translation_mm", "xy_mm", "rotation_deg"):
                seed_values = [float(np.mean([mm[metric][j] for j, anchor in enumerate(anchors) if anchor["seed"] == seed]))
                               for seed in sorted({x["seed"] for x in anchors})]
                model_pairs.append({"left": left.parent.name, "right": right.parent.name, "metric": metric,
                                    "mean_by_seed": float(np.mean(seed_values)), "seeds": len(seed_values)})
    csv_write("shared_policy_pairs.csv", model_pairs)
    failed_stages = [p.stem for p in (args.output/"stages").glob("*.json") if json.loads(p.read_text()).get("returncode") != 0]
    report = {"smoke": args.smoke, "cv": summary, "paired": paired, "modality_pairs": modality_pairs,
              "failed_stages": failed_stages, "cv_frames_by_mode": len(case_rows),
              "rollout_rows": len(rollout_rows), "shared_correspondence_rows": len(shared_rows),
              "notes": ["Outer test episodes never enter fitting, scaling or retrieval.",
                        "CV intervals resample episodes, not adjacent frames; exploratory, overlapping CV training sets.",
                        "Held-out demonstration error, own-rollout correspondence and shared-input correspondence are different targets.",
                        "No prediction is a success-rate estimate. Projected feature limits and candidate coverage bound the conclusions."]}
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    table = "".join("<tr>"+"".join(f"<td>{html.escape(str(r.get(k))) if not isinstance(r.get(k), float) else format(r[k], '.3f')}</td>" for k in
        ("mode", "phase", "method", "episodes", "translation_mm", "xy_mm", "rotation_deg", "chunk_translation_mm", "cos3"))+"</tr>" for r in summary)
    page = f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>ContactWorld 动作可预测性</title>
<style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#23344a}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef4fa}}a{{color:#2367a8}}</style>
<h1>观测条件权重网络与示范动作组合 {'— SMOKE ONLY' if args.smoke else ''}</h1>
<p>失败阶段：{html.escape(str(failed_stages))}；阶段日志在 logs/，同一命令可以继续未完成部分。</p>
<p>每条示范的外层留出预测；查询所属整条示范不进入检索库。R 是冻结 ResNet18 双相机特征＋状态；P 是 base 点云三投影；T 是双指坐标＋局部及 base 力。权重初始为相似度，随后学习有界 logit 修正。动作没有送入权重网络。</p>
<p>phase -2：全部，0：接触前，1：接触后。误差先按示范平均；平移/横向单位 mm，旋转为小角度控制向量差的度数。initialized 与 learned 共用十个候选；fixed 在内层验证选择 k 和温度。</p>
<table><tr><th>输入</th><th>阶段</th><th>方法</th><th>示范数</th><th>平移误差</th><th>XY误差</th><th>旋转误差</th><th>四步平移误差</th><th>方向余弦</th></tr>{table}</table>
<p>CV 帧记录 {len(case_rows)}；原轨迹对应表 {len(rollout_rows)} 行；共同观测对应表 {len(shared_rows)} 行。模态比较应同时查看预测误差、有效候选数和不确定性；更低邻域方差本身不是充分证据。</p>
<p><a href='analysis.json'>汇总及配对区间</a> · <a href='cv_episode_metrics.csv'>逐示范指标</a> · <a href='cv_frame_diagnostics.csv'>逐帧误差与分歧</a> · <a href='modality_pairs.csv'>模态配对差异</a> · <a href='uncertainty_bins.csv'>分歧与误差关系</a> · <a href='rollout_correspondence.csv'>原520轨迹对应</a> · <a href='shared_observation_correspondence.csv'>同观测13模型对应</a> · <a href='shared_policy_pairs.csv'>同观测π0模型间差异</a> · <a href='anchors.json'>共同观测来源</a> · <a href='plan.json'>运行协议</a></p></html>"""
    (args.output / "index.html").write_text(page)
    print(args.output / "index.html", flush=True)


def parser():
    from contactworld_neighbor_features import DEFAULT_WEIGHTS, MODES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--dataset", type=Path, default=REPO / "data/contactworld_usb_v3_bilateral")
    p.add_argument("--eval-root", type=Path, default=EVAL)
    p.add_argument("--checkpoints", type=Path, default=Path("/home/sai/zsq/FactileLDM/1002_ckpt"))
    p.add_argument("--vision-weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--anchors-per-trial", type=int, default=6)
    p.add_argument("--skip-shared", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--stage", choices=("all", "prepare", "train", "rollouts", "shared", "report"), default="all")
    p.add_argument("--config")
    return p


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.batch < 1 or args.threads < 1 or args.anchors_per_trial < 1:
        raise ValueError("epochs, batch, threads and anchors must be positive")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.modes = [m for m in args.modes if m in ("R", "RPT")]
        if not args.modes:
            args.modes = ["R"]
        if args.output == OUTPUT:
            args.output = OUTPUT.with_name(OUTPUT.name+"_smoke")
    args.output = args.output.resolve()
    if args.stage == "all":
        import fcntl
        from contactworld_neighbor_features import digest, VERSION
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / "run.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("stage", "config", "check_only")}
        plan["features_version"] = VERSION
        plan["manifest_sha256"] = digest(args.dataset / "spatial/manifest.json")
        plan["vision_sha256"] = digest(args.vision_weights)
        plan["eval_summary_sha256"] = {p.parent.name: digest(p) for p in sorted(args.eval_root.glob("pi0_cw3_usb_*/summary.json"))}
        plan["script_sha256"] = {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("contactworld_neighbor_features.py"), Path(__file__).with_name("contactworld_neighbor_weights.py"))}
        configs = [p.parent.name for p in sorted(args.eval_root.glob("pi0_cw3_usb_*/summary.json"))]
        if not configs:
            raise FileNotFoundError("No recorded evaluations")
        if args.smoke:
            configs = configs[:1]
        if not args.skip_shared:
            for config in configs:
                path = args.checkpoints / config / "cw3_round1_20k/19999/params/_METADATA"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing checkpoint: {path}")
            plan["checkpoint_metadata"] = {c: digest(args.checkpoints/c/"cw3_round1_20k/19999/params/_METADATA") for c in configs}
            plan["checkpoint_manifests"] = {c: digest(args.checkpoints/c/"cw3_round1_20k/19999/params/manifest.ocdbt") for c in configs}
        existing = args.output / "plan.json"
        if existing.exists() and json.loads(existing.read_text()) != plan:
            raise ValueError("Existing output uses a different protocol/code. Choose a new --output; do not mix runs.")
        existing.write_text(json.dumps(plan, indent=2)+"\n")
        print(f"PRECHECK OK: modes={args.modes}, outer folds=5, shared models={0 if args.skip_shared else len(configs)}", flush=True)
        if args.check_only:
            return
        base = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(args.output),
                "--dataset", str(args.dataset.resolve()), "--eval-root", str(args.eval_root.resolve()),
                "--checkpoints", str(args.checkpoints.resolve()), "--vision-weights", str(args.vision_weights.resolve()),
                "--modes", *args.modes, "--epochs", str(args.epochs), "--device", args.device,
                "--threads", str(args.threads), "--batch", str(args.batch), "--seed", str(args.seed),
                "--anchors-per-trial", str(args.anchors_per_trial)]
        if args.smoke:
            base += ["--smoke"]
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        jobs = [(s, None) for s in ("prepare", "train", "rollouts")]
        if not args.skip_shared:
            jobs += [("shared", c) for c in configs]
        jobs.append(("report", None))
        failures = []
        for stage, config in jobs:
            label = stage+("_"+config if config else "")
            stamp = args.output / "stages" / (label+".json")
            if stamp.exists() and json.loads(stamp.read_text()).get("returncode") == 0 and stage != "report":
                print(f"SKIP completed {label}", flush=True)
                continue
            stamp.parent.mkdir(exist_ok=True)
            logfile = args.output / "logs" / (label+".log")
            logfile.parent.mkdir(exist_ok=True)
            cmd = base+["--stage", stage]+(["--config", config] if config else [])
            begin = time.time()
            print(f"START {label}: {logfile}", flush=True)
            with logfile.open("a") as log:
                log.write(f"\nSTART {time.ctime()}\n"); log.flush()
                result = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            stamp.write_text(json.dumps({"returncode": result.returncode, "seconds": time.time()-begin})+"\n")
            print(f"DONE {label}: code={result.returncode} elapsed={time.time()-begin:.0f}s", flush=True)
            if result.returncode:
                failures.append(label)
                if stage in ("prepare", "train", "rollouts"):
                    raise RuntimeError(f"Required stage failed: {logfile}")
        if failures:
            raise RuntimeError(f"Some stages failed; successes retained. Re-run identical command: {failures}")
        print(f"ALL COMPLETE: {args.output/'index.html'}", flush=True)
        return
    import torch
    torch.set_num_threads(args.threads)
    if args.stage == "prepare":
        from contactworld_neighbor_features import prepare
        prepare(args.dataset, args.output/"features", args.vision_weights, args.device, args.batch,
                limit=20 if args.smoke else None)
    elif args.stage == "train":
        from contactworld_neighbor_weights import train_cv
        train_cv(load_npz(args.output/"features/train_features.npz"), args.output/"cv", args.modes,
                 args.epochs, seed=args.seed, device=args.device, outer_limit=1 if args.smoke else None)
    elif args.stage == "rollouts":
        stage_rollouts(args)
    elif args.stage == "shared":
        if not args.config:
            raise ValueError("--config required for shared stage")
        stage_shared(args)
    else:
        stage_report(args)


if __name__ == "__main__":
    main()
