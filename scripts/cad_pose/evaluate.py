#!/usr/bin/env python3
"""Evaluate actual model predictions, grouped by dataset and object."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from openpi.cad_pose.geometry import predicted_geometry_packet
from openpi.cad_pose.pipeline import PROFILE
from openpi.cad_pose.pipeline import checkpoint_open_options
from openpi.cad_pose.pipeline import load_checkpoint
from openpi.cad_pose.pipeline import make_metric_rows
from openpi.cad_pose.pipeline import open_tasks
from openpi.cad_pose.pipeline import predict
from openpi.cad_pose.pipeline import report_metrics
from openpi.cad_pose.pipeline import verify_checkpoint_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--max-frames", type=int, default=0, help="per task; 0 means all")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    repo = a.repo.resolve()
    model, ckpt = load_checkpoint(a.checkpoint, a.device)
    if bool(ckpt["synthetic"]) != a.synthetic:
        raise ValueError("synthetic flag differs from checkpoint")
    tasks, audits = open_tasks(
        repo,
        (a.data_root or repo / "data").resolve(),
        list(ckpt["splits"]),
        a.profile,
        synthetic=a.synthetic,
        **checkpoint_open_options(ckpt),
    )
    if set(tasks) != set(ckpt["splits"]):
        raise ValueError(f"some task pose sources failed audit: {audits}")
    verify_checkpoint_sources(ckpt, tasks, repo, a.profile, synthetic=a.synthetic)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError("evaluation output directory must be new or empty")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows, latency = [], {"preprocess_ms": [], "forward_ms": [], "cad_transform_ms": [], "total_ms": []}
    if torch.device(a.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(a.device)
    for name, task in tasks.items():
        count = 0
        for episode in ckpt["splits"][name][a.split]:
            for sample in task.iter_episode(episode):
                if torch.device(a.device).type == "cuda":
                    torch.cuda.synchronize(a.device)
                start = time.perf_counter()
                predicted = predict(model, sample, a.device)
                if torch.device(a.device).type == "cuda":
                    torch.cuda.synchronize(a.device)
                forward = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                predicted_geometry_packet(sample["cad"], predicted, np.ones(2, bool))
                conversion = (time.perf_counter() - start) * 1000
                preprocess = sample.get("preprocessing_seconds")
                pre_ms = None if preprocess is None else preprocess * 1000
                for key, value in (("forward_ms", forward), ("cad_transform_ms", conversion)):
                    latency[key].append(value)
                if pre_ms is not None:
                    latency["preprocess_ms"].append(pre_ms)
                    latency["total_ms"].append(pre_ms + forward + conversion)
                rows.extend(make_metric_rows(sample, predicted, task.roles, ckpt.get("symmetry", {})))
                count += 1
                if a.max_frames and count >= a.max_frames:
                    break
            if a.max_frames and count >= a.max_frames:
                break
        if count == 0:
            raise ValueError(f"{name}: no valid evaluation frames")
    if not rows:
        raise ValueError("no valid object labels")
    with (a.output_dir / "rows.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    timing = {
        key: (
            {"p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)), "count": len(vals)}
            if vals
            else None
        )
        for key, vals in latency.items()
    }
    report = {
        "synthetic": a.synthetic,
        "split": a.split,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "metrics": report_metrics(rows),
        "latency_ms": timing,
        "latency_note": "forward includes host-to-device copies; total excludes file/video decoding; first-call warmup included",
        "params": sum(x.numel() for x in model.parameters()),
        "device": a.device,
        "host": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(a.device) if torch.device(a.device).type == "cuda" else None,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(a.device) if torch.device(a.device).type == "cuda" else None,
        "rotation_symmetry": {"explicit_keys": list(ckpt.get("symmetry", {})), "default": "identity only"},
        "adds_note": "nearest-neighbour ADD-S is reported as a diagnostic; not an assertion of object symmetry",
    }
    (a.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
