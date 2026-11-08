#!/usr/bin/env python3
"""Render fresh checkpoint predictions for unseen episodes and worst frames."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from openpi.cad_pose.pipeline import PROFILE
from openpi.cad_pose.pipeline import load_checkpoint
from openpi.cad_pose.pipeline import open_tasks
from openpi.cad_pose.pipeline import predict
from openpi.cad_pose.pipeline import verify_checkpoint_sources
from openpi.cad_pose.visualization import render_comparison


def camera(task, synthetic):
    if synthetic:
        return np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]]), np.eye(4)
    cam = task.preprocess.color_camera()
    k = cam.color_intrinsics
    return np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]]), cam.T_base_color


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data-root", type=Path)
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes-per-task", type=int, default=2)
    p.add_argument("--frames-per-episode", type=int, default=16)
    p.add_argument("--worst-per-task", type=int, default=5)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    if a.episodes_per_task < 2 or a.frames_per_episode < 16 or a.worst_per_task < 1:
        raise ValueError("require at least 2 episodes, 16 frames per episode and 1 worst frame")
    report = json.loads((a.evaluation_dir / "evaluation.json").read_text())
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if report["checkpoint_sha256"] != digest or report["split"] != "test" or bool(report["synthetic"]) != a.synthetic:
        raise ValueError("evaluation must be test split from the same checkpoint and data mode")
    rows = [json.loads(line) for line in (a.evaluation_dir / "rows.jsonl").read_text().splitlines()]
    model, ckpt = load_checkpoint(a.checkpoint, a.device)
    repo = a.repo.resolve()
    tasks, audits = open_tasks(
        repo, (a.data_root or repo / "data").resolve(), list(ckpt["splits"]), a.profile, synthetic=a.synthetic
    )
    if set(tasks) != set(ckpt["splits"]):
        raise ValueError(f"pose source audit failed: {audits}")
    verify_checkpoint_sources(ckpt, tasks, repo, a.profile, synthetic=a.synthetic)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError("visualization output must be empty")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for name, task in tasks.items():
        records = [r for r in rows if r["dataset"] == name]
        episodes = sorted({r["episode"] for r in records})
        if len(episodes) < a.episodes_per_task:
            raise ValueError(f"{name}: test evaluation has fewer than {a.episodes_per_task} episodes")
        selected = {}
        for episode in episodes[: a.episodes_per_task]:
            frames = sorted({r["frame"] for r in records if r["episode"] == episode})
            if len(frames) < a.frames_per_episode:
                raise ValueError(f"{name} episode {episode}: fewer than {a.frames_per_episode} evaluated frames")
            selected[episode] = set(
                np.asarray(frames)[np.linspace(0, len(frames) - 1, a.frames_per_episode, dtype=int)].tolist()
            )
        for row in sorted(records, key=lambda r: r["add_mm"], reverse=True)[: a.worst_per_task]:
            selected.setdefault(row["episode"], set()).add(row["frame"])
        intrinsic, base_color = camera(task, a.synthetic)
        for episode, frames in selected.items():
            missing = set(frames)
            for sample in task.iter_episode(episode, include_rgb=True):
                if sample["frame"] not in frames:
                    continue
                missing.discard(sample["frame"])
                poses = predict(model, sample, a.device)
                for i, role in enumerate(("manipulated", "target")):
                    filename = f"{name}/episode_{episode:05d}_frame_{sample['frame']:06d}_{role}.png"
                    label = f"{'SYNTHETIC ' if a.synthetic else ''}{name} ep={episode} frame={sample['frame']} {task.roles[role]}"
                    render_comparison(
                        rgb=sample["rgb"],
                        intrinsic=intrinsic,
                        t_base_color=base_color,
                        cad_points=sample["cad"][i],
                        predicted_pose=poses[i],
                        target_pose=sample["target_transform"][i] if sample["valid"][i] else None,
                        label=label,
                        output=a.output_dir / filename,
                        symmetry_rotations=ckpt.get("symmetry", {}).get(name + "/" + task.roles[role]),
                    )
                    links.append((label, filename))
            if missing:
                raise ValueError(f"{name} episode {episode}: missing frames {sorted(missing)}")
    page = '<!doctype html><meta charset="utf-8"><title>Pose review</title><h1>Pose review</h1>'
    page += "".join(
        f'<section><h3>{html.escape(label)}</h3><img loading="lazy" width="1200" src="{html.escape(path)}"></section>'
        for label, path in links
    )
    (a.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {len(links)} panels to {a.output_dir}/index.html")


if __name__ == "__main__":
    main()
