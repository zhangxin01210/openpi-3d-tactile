#!/usr/bin/env python3
"""Per-task data/model smoke; never writes a training checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openpi.cad_pose.model import XHandCADPoseTransformerV1
from openpi.cad_pose.model import pose_loss
from openpi.cad_pose.pipeline import DEFAULT_TASKS
from openpi.cad_pose.pipeline import PROFILE
from openpi.cad_pose.pipeline import batchify
from openpi.cad_pose.pipeline import open_tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data-root", type=Path)
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    repo = a.repo.resolve()
    tasks, reports = open_tasks(
        repo, (a.data_root or repo / "data").resolve(), a.tasks, a.profile, synthetic=a.synthetic
    )
    for name, task in tasks.items():
        try:
            sample = next(iter(task.iter_episode(task.episodes[0])))
            b = batchify([sample], a.device)
            model = XHandCADPoseTransformerV1().to(a.device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            prediction = model(b["scene"], b["scene_mask"], b["cad"])
            loss = pose_loss(prediction, b["target_transform"], b["cad"], b["valid"])
            loss.backward()
            if not torch.isfinite(loss) or any(
                x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()
            ):
                raise ValueError("non-finite loss or gradients")
            opt.step()
            reports[name]["smoke"] = {
                "status": "passed",
                "loss": float(loss.detach()),
                "model_state_shapes": {
                    "scene": list(b["scene"].shape),
                    "cad": list(b["cad"].shape),
                    "poses": list(prediction["rotation"].shape),
                },
                "episode": int(sample["episode"]),
                "frame": int(sample["frame"]),
                "synthetic": a.synthetic,
                "valid_objects": np.asarray(sample["valid"]).tolist(),
            }
        except (ValueError, OSError, KeyError, RuntimeError, StopIteration) as exc:
            reports[name]["smoke"] = {"status": "failed", "error": str(exc)}
    for name in set(a.tasks) - set(tasks):
        reports[name]["smoke"] = {"status": "blocked"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    return int(any(r["smoke"]["status"] != "passed" for r in reports.values()))


if __name__ == "__main__":
    raise SystemExit(main())
