#!/usr/bin/env python3
"""Fit a small q/dq/elapsed-step behavior-cloning baseline on pose-positive demos."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import zarr

from contactworld_proprio_policy import ProprioPolicy


TASKS = ("insertion_usb", "insertion_peg")


def load_task(data_root, audit, task):
    group = zarr.open_group(str(data_root / task), mode="r")
    data = group["data"]
    ends = np.asarray(group["meta/episode_ends"][:], dtype=int)
    starts = np.r_[0, ends[:-1]]
    q = np.asarray(data["dof_pos"][:], dtype=np.float32)
    dq = np.asarray(data["dof_vel"][:], dtype=np.float32)
    actions = np.asarray(data["action"][:], dtype=np.float32)
    if q.shape != dq.shape or q.shape[1] != 9 or actions.shape != (len(q), 6):
        raise ValueError(f"Unexpected state/action shape: {task}")
    selections = {}
    for split in ("train", "val", "test"):
        indices = []
        episodes = []
        for ep, (start, end) in enumerate(zip(starts, ends)):
            row = audit[(task, ep)]
            if row["split"] != split or row["source_endpoint_pass"] != "True":
                continue
            frames = np.arange(end-start, dtype=np.float32)
            x = np.column_stack((q[start:end], dq[start:end], frames / 100))
            indices.append((x, actions[start:end]))
            episodes.append(ep)
        if not indices:
            raise ValueError(f"No pose-positive demos for {task}:{split}")
        selections[split] = (np.concatenate([x for x, _ in indices]),
                             np.concatenate([y for _, y in indices]), episodes)
    return selections


def fit_one(task, selections, output, epochs, patience, seed, device):
    train_x, train_y, train_eps = selections["train"]
    val_x, val_y, val_eps = selections["val"]
    input_mean = train_x.mean(0)
    input_std = np.maximum(train_x.std(0), 1e-4)
    action_mean = train_y.mean(0)
    action_std = np.maximum(train_y.std(0), 1e-3)
    action_min = train_y.min(0)
    action_max = train_y.max(0)
    tx = torch.as_tensor((train_x-input_mean)/input_std, device=device)
    ty = torch.as_tensor((train_y-action_mean)/action_std, device=device)
    vx = torch.as_tensor((val_x-input_mean)/input_std, device=device)
    vy = torch.as_tensor((val_y-action_mean)/action_std, device=device)
    model = ProprioPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    best = float("inf")
    wait = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tx), generator=generator).to(device)
        losses = []
        for indices in order.split(512):
            predicted = model(tx[indices])
            loss = nn.functional.mse_loss(predicted, ty[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_mse = float(nn.functional.mse_loss(model(vx), vy).item())
        history.append({"epoch": epoch+1, "train_normalized_mse": float(np.mean(losses)),
                        "val_normalized_mse": val_mse})
        if val_mse < best - 1e-5:
            best = val_mse
            wait = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    bundle = {"model": model.state_dict(), "input_mean": torch.as_tensor(input_mean, device="cpu"),
              "input_std": torch.as_tensor(input_std, device="cpu"),
              "action_mean": torch.as_tensor(action_mean, device="cpu"),
              "action_std": torch.as_tensor(action_std, device="cpu"),
              "action_min": torch.as_tensor(action_min, device="cpu"),
              "action_max": torch.as_tensor(action_max, device="cpu"),
              "task": task, "seed": seed, "best_epoch": best_epoch}
    path = output / (task + "_proprio.pt")
    torch.save(bundle, path)
    result = {"checkpoint": str(path), "best_epoch": best_epoch,
              "train_episodes": len(train_eps), "val_episodes": len(val_eps),
              "train_frames": len(train_x), "val_frames": len(val_x),
              "action_mean": action_mean.tolist(), "action_std": action_std.tolist(),
              "action_min": action_min.tolist(), "action_max": action_max.tolist(),
              "history": history, "splits": {}}
    for split, (x, y, episodes) in selections.items():
        normalized = torch.as_tensor((x-input_mean)/input_std, device=device)
        with torch.no_grad():
            pred = model(normalized).cpu().numpy() * action_std + action_mean
        result["splits"][split] = {
            "episodes": len(episodes), "frames": len(x),
            "action_mse": float(np.mean((pred-y)**2)),
            "constant_train_mean_action_mse": float(np.mean((action_mean-y)**2)),
            "zero_action_mse": float(np.mean(y**2)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with args.audit.open(newline="") as stream:
        audit = {(r["task"], int(r["episode"])): r for r in csv.DictReader(stream)}
    report = {"scope": "Two independent q/dq/time MLPs; pose-positive train demos only; no visual, tactile, pointcloud, goal, or demo reference inputs",
              "device": str(device), "seed": args.seed, "tasks": {}}
    for task in TASKS:
        selections = load_task(args.data, audit, task)
        report["tasks"][task] = fit_one(task, selections, args.output, args.epochs,
                                         args.patience, args.seed, device)
        print(task, report["tasks"][task]["best_epoch"],
              report["tasks"][task]["splits"], flush=True)
        (args.output / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
