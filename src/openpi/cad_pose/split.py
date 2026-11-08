"""Episode-only reproducible splits and equal-task epoch selection."""
from __future__ import annotations

import numpy as np


def split_episodes(episode_ids: list[int], seed: int = 42) -> dict[str, tuple[int, ...]]:
    values = np.asarray(episode_ids, dtype=np.int64)
    if values.ndim != 1 or len(values) < 3 or len(np.unique(values)) != len(values):
        raise ValueError("need at least three distinct episode IDs")
    ordered = np.sort(values)
    np.random.default_rng(seed).shuffle(ordered)
    n_test = max(1, round(len(values) * .1))
    n_val = max(1, round(len(values) * .1))
    if n_val + n_test >= len(values):
        raise ValueError("episode split has no training episodes")
    return {"test": tuple(sorted(map(int, ordered[:n_test]))),
            "val": tuple(sorted(map(int, ordered[n_test:n_test+n_val]))),
            "train": tuple(sorted(map(int, ordered[n_test+n_val:])))}


def balanced_tasks(task_names: tuple[str, ...], samples_per_task: int, seed: int = 42) -> tuple[str, ...]:
    if not task_names or len(set(task_names)) != len(task_names) or samples_per_task < 1:
        raise ValueError("task names must be unique and samples_per_task positive")
    values = np.repeat(np.asarray(task_names), samples_per_task)
    np.random.default_rng(seed).shuffle(values)
    return tuple(map(str, values))
