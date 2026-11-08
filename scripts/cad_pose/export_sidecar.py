#!/usr/bin/env python3
"""Export raw-RGB-D predictions after explicit checkpoint acceptance."""

import argparse
import hashlib
import json
from pathlib import Path

from openpi.cad_pose.pipeline import DEFAULT_TASKS
from openpi.cad_pose.pipeline import PROFILE
from openpi.cad_pose.pipeline import checkpoint_open_options
from openpi.cad_pose.pipeline import open_tasks
from openpi.cad_pose.runtime import PoseRuntime
from openpi.cad_pose.sidecar import export_sidecar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--acceptance", type=Path, required=True, help="human acceptance bound to checkpoint SHA256")
    p.add_argument("--dataset", choices=DEFAULT_TASKS, required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--data-root", type=Path)
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--version", default="pretrained_pose_v1")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    import torch

    checkpoint_hash = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    acceptance = json.loads(a.acceptance.read_text())
    if (
        acceptance.get("status") != "accepted"
        or acceptance.get("checkpoint_sha256") != checkpoint_hash
        or not acceptance.get("reviewer")
    ):
        raise ValueError("acceptance must explicitly identify reviewer and the exact accepted checkpoint")
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("synthetic"):
        raise ValueError("synthetic checkpoints cannot generate real geometry")
    repo = a.repo.resolve()
    tasks, reports = open_tasks(
        repo, (a.data_root or repo / "data").resolve(), [a.dataset], a.profile, **checkpoint_open_options(saved)
    )
    if a.dataset not in tasks:
        raise ValueError(f"task asset/provenance audit failed: {reports}")
    task = tasks[a.dataset]
    runtime = PoseRuntime.from_checkpoint(
        a.checkpoint, repo, a.profile, task.ordered_cad, dataset=a.dataset, device=a.device
    )
    result = export_sidecar(
        task=task,
        runtime=runtime,
        repo=repo,
        profile=a.profile,
        version=a.version,
        checkpoint_sha256=checkpoint_hash,
        camera_hashes=saved["camera_hashes"],
        acceptance=acceptance,
    )
    print(result)


if __name__ == "__main__":
    main()
