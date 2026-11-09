#!/usr/bin/env python3
"""Publish verified V2 candidate predictions into independent native sidecars."""

import argparse
from pathlib import Path

from openpi.cad_pose.v2_native import TASKS
from openpi.cad_pose.v2_native import publish_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", type=Path, default=Path("output/cad_pose_rgbd_v2/geometry_full_candidate"))
    p.add_argument("--cache-root", type=Path, default=Path("output/cad_pose_rgbd_v2/cache"))
    p.add_argument("--checkpoint", type=Path, default=Path("output/cad_pose_rgbd_v2/train_20261109_162628/best.pt"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    for task in a.tasks:
        path = publish_task(
            candidate_root=a.candidate_root,
            cache_root=a.cache_root,
            checkpoint=a.checkpoint,
            dataset_root=a.data_root / task,
            task=task,
            device=a.device,
        )
        print(f"[PUBLISHED] {task}: {path}", flush=True)


if __name__ == "__main__":
    main()
