"""Strictly causal object geometry history."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from openpi.object_geometry.cloud import as_points, transform_points


def _pose(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"pose must have finite shape (4, 4), got {pose.shape}")
    return pose


@dataclass(frozen=True)
class HistoryObservation:
    frame_index: int
    points: np.ndarray
    pose: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "points", as_points(self.points))
        object.__setattr__(self, "pose", _pose(self.pose))


@dataclass(frozen=True)
class HistoryResult:
    points: np.ndarray
    support_count: np.ndarray
    last_seen_frame: np.ndarray
    eligible_frames: tuple[int, ...]
    supported_cells: int

    def __post_init__(self) -> None:
        points = as_points(self.points)
        support = np.asarray(self.support_count, dtype=np.int16).reshape(-1)
        last_seen = np.asarray(self.last_seen_frame, dtype=np.int32).reshape(-1)
        if support.shape != (len(points),) or last_seen.shape != (len(points),):
            raise ValueError("history metadata must contain one value per point")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "support_count", support)
        object.__setattr__(self, "last_seen_frame", last_seen)
        object.__setattr__(self, "eligible_frames", tuple(int(x) for x in self.eligible_frames))


def motion_informative(
    first_pose: Any,
    second_pose: Any,
    *,
    min_translation_m: float,
    min_rotation_deg: float,
) -> bool:
    first = _pose(first_pose)
    second = _pose(second_pose)
    translation = np.linalg.norm(first[:3, 3] - second[:3, 3])
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    angle = math.degrees(math.acos(float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))))
    return bool(translation >= min_translation_m or angle >= min_rotation_deg)


def build_causal_history(
    observations: list[HistoryObservation],
    *,
    current_frame: int,
    window_frames: int,
    voxel_size_m: float,
    min_observations: int = 2,
    require_motion: bool = False,
    min_motion_m: float = 0.025,
    min_motion_deg: float = 6.0,
    max_points: int = 800,
) -> HistoryResult:
    """Voxel-support history using only observations at or before current_frame.

    For a manipulated object, observations are expected in the moving hand's
    local frame and ``require_motion`` demands at least one informative pose
    change per accepted voxel. For a stable target, points are expected in the
    base frame and repeated observations alone provide visibility support.
    """

    current_frame = int(current_frame)
    if window_frames < 0 or voxel_size_m <= 0 or min_observations < 1 or max_points < 0:
        raise ValueError("history window, voxel size, observation count, and point limit are invalid")
    eligible = sorted(
        (observation for observation in observations if current_frame - window_frames <= observation.frame_index <= current_frame),
        key=lambda observation: observation.frame_index,
    )
    cells: dict[tuple[int, int, int], dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for observation in eligible:
        per_frame: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
        for point in observation.points:
            key = tuple(np.floor(point / voxel_size_m).astype(np.int32).tolist())
            per_frame[key].append(point)
        for key, values in per_frame.items():
            cells[key][observation.frame_index] = (
                np.median(np.asarray(values, dtype=np.float32), axis=0).astype(np.float32),
                observation.pose,
            )

    accepted: list[np.ndarray] = []
    support: list[int] = []
    last_seen: list[int] = []
    for observations_by_frame in cells.values():
        values = list(observations_by_frame.items())
        if len(values) < min_observations:
            continue
        informative = True
        if require_motion:
            informative = any(
                motion_informative(
                    first[1][1],
                    second[1][1],
                    min_translation_m=min_motion_m,
                    min_rotation_deg=min_motion_deg,
                )
                for index, first in enumerate(values)
                for second in values[index + 1 :]
            )
        if not informative:
            continue
        accepted.append(np.median([value[1][0] for value in values], axis=0).astype(np.float32))
        support.append(len(values))
        last_seen.append(max(frame for frame, _ in values))

    if accepted:
        points = np.asarray(accepted, dtype=np.float32).reshape(-1, 3)
        support_array = np.asarray(support, dtype=np.int16)
        last_seen_array = np.asarray(last_seen, dtype=np.int32)
        if len(points) > max_points:
            order = np.lexsort((np.arange(len(points)), -support_array))[:max_points]
            points = points[order]
            support_array = support_array[order]
            last_seen_array = last_seen_array[order]
    else:
        points = np.empty((0, 3), dtype=np.float32)
        support_array = np.empty((0,), dtype=np.int16)
        last_seen_array = np.empty((0,), dtype=np.int32)
    return HistoryResult(
        points=points,
        support_count=support_array,
        last_seen_frame=last_seen_array,
        eligible_frames=tuple(observation.frame_index for observation in eligible),
        supported_cells=len(points),
    )


def history_to_base(result: HistoryResult, current_pose: Any) -> np.ndarray:
    """Transform moving-frame history into the current base frame."""

    return transform_points(result.points, _pose(current_pose))
