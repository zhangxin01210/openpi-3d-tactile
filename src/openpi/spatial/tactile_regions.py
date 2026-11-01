"""Deterministic XHand taxel regions derived from fixed link-local geometry.

The returned table is indexed by canonical finger ID and taxel ID minus one;
it can be serialized with a model config and reused for online inference.
"""
from __future__ import annotations

import numpy as np


def partition_taxel_geometry(xyz_link_m: np.ndarray) -> tuple[tuple[int, ...], ...]:
    xyz = np.asarray(xyz_link_m, dtype=np.float64)
    if xyz.shape != (5, 120, 3) or not np.isfinite(xyz).all():
        raise ValueError("expected finite link-local taxel geometry with shape (5, 120, 3)")
    result = []
    for finger in xyz:
        groups = [np.arange(120)]
        # Repeated longest-axis bisection creates four spatially bounded regions.
        for _ in range(2):
            next_groups = []
            for ids in groups:
                points = finger[ids]
                axis = int(np.argmax(np.ptp(points, axis=0)))
                if np.ptp(points[:, axis]) <= 0:
                    raise ValueError("degenerate taxel geometry cannot define spatial regions")
                order = np.lexsort((ids, points[:, axis]))
                ordered = ids[order]
                next_groups.extend((ordered[: len(ids) // 2], ordered[len(ids) // 2 :]))
            groups = next_groups
        labels = np.empty(120, dtype=np.int32)
        for region, ids in enumerate(groups):
            labels[ids] = region
        result.append(tuple(int(x) for x in labels))
    return tuple(result)
