#!/usr/bin/env python3
"""Unpack/inspect a ContactWorld task and export a fixed replay input manifest.

This is read-only with respect to the original Zarr data.  The manifest is
created before Isaac Gym replay, so episode selection cannot be changed after
seeing a replay outcome.  ``--limit 1`` is the mandatory review smoke path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from contactworld_cross_task_common import TASKS, get_task

FIELDS = ("action", "dof_pos", "dof_vel", "ee_pos", "ee_quat", "plug_pos",
          "plug_quat", "socket_pos_gt", "socket_quat", "front", "wrist",
          "tactile_force_field_right", "tactile_rgb_right", "tactile_depth_right", "step_idx")


def maybe_unpack(spec, archive: Path, source: Path) -> Path:
    group = source / spec.source_group
    if group.exists():
        return group
    if not archive.exists():
        raise FileNotFoundError("Missing source group and archive: %s" % archive)
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        roots = {Path(member.name).parts[0] for member in members if member.name and not member.name.startswith("/")}
        if any(".." in Path(member.name).parts for member in members):
            raise ValueError("Unsafe archive member")
        tar.extractall(source)
    if group.exists():
        return group
    # Releases sometimes wrap exactly one source directory in a parent folder.
    candidates = list(source.rglob(spec.source_group))
    if len(candidates) == 1 and candidates[0].is_dir():
        shutil.move(str(candidates[0]), str(group))
        return group
    raise FileNotFoundError("Archive did not produce %s; roots=%s" % (group, sorted(roots)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--source", type=Path, default=Path("/home/sai/zx/openpi-sim-runtime/data/contactworld"))
    parser.add_argument("--archive", type=Path, help="Required only when --source lacks this task group")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--episodes", type=int, nargs="*", help="Precommitted source episode IDs; defaults to evenly spaced IDs")
    args = parser.parse_args()
    spec = get_task(args.task)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    group_path = maybe_unpack(spec, args.archive or Path("__missing__"), args.source)
    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("Source preparation needs zarr==2.16.1 in the local conversion environment") from error
    group = zarr.open_group(str(group_path), mode="r")
    data = group["data"]
    missing = [field for field in FIELDS if field not in data]
    if missing:
        raise ValueError("Released task lacks replay fields: %s" % missing)
    if data["action"].shape[1] != spec.action_dim:
        raise ValueError("%s action shape is %s, expected %d" % (spec.key, data["action"].shape, spec.action_dim))
    ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    count = len(ends)
    chosen = args.episodes if args.episodes is not None else np.linspace(0, count - 1, min(args.limit, count), dtype=int).tolist()
    if len(chosen) > args.limit:
        raise ValueError("--episodes contains more IDs than --limit")
    if len(set(chosen)) != len(chosen) or any(ep < 0 or ep >= count for ep in chosen):
        raise ValueError("Invalid or repeated source episode")
    args.output.mkdir(parents=True)
    episodes = []
    for output_episode, source_episode in enumerate(chosen):
        start, end = int(starts[source_episode]), int(ends[source_episode])
        values = {field: np.asarray(data[field][start:end]) for field in FIELDS}
        if any(len(value) != end - start or not np.isfinite(value).all() for value in values.values()):
            raise ValueError("Invalid source rows for episode %d" % source_episode)
        filename = "%s_episode_%03d.npz" % (spec.source_group, source_episode)
        path = args.output / filename
        np.savez_compressed(path, **values)
        episodes.append({"task": spec.key, "source_group": spec.source_group,
                         "source_episode": int(source_episode), "episode_index": output_episode,
                         "global_rows": [start, end], "frames": end - start,
                         "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "action_dim": spec.action_dim,
                         "columns": {key: list(value.shape) for key, value in values.items()}})
        print("Prepared source=%d frames=%d" % (source_episode, end - start), flush=True)
    manifest = {"format": "contactworld-cross-task-replay-input-v1", "task": spec.key,
                "source_group": spec.source_group, "source": str(group_path),
                "action_dim": spec.action_dim, "selection": "precommitted source episode IDs before replay",
                "fields": FIELDS, "episodes": episodes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
