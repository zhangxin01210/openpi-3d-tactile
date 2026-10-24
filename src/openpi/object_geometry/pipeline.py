"""Small composition layer for current geometry A and causal history B."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from openpi.object_geometry.cloud import mask_depth_to_cloud
from openpi.object_geometry.history import HistoryObservation, build_causal_history, history_to_base
from openpi.object_geometry.types import ObjectGeometry, ObjectGeometryFrame, PointCloud


def build_object_geometry_frame(
    *,
    frame_index: int,
    depth: Any,
    rays_base_per_meter: Any,
    transform_base_color: Any,
    color_intrinsics: Any,
    depth_scale_m_per_unit: float,
    object_masks: Mapping[str, Any],
    histories: Mapping[str, Sequence[HistoryObservation]] | None = None,
    object_poses: Mapping[str, Any] | None = None,
    manipulated_objects: Iterable[str] = (),
    manipulated_valid: Mapping[str, bool] | None = None,
    workspace_bounds: tuple[float, float, float, float, float, float] | None = None,
    rgb: Any | None = None,
    history_window_frames: int = 75,
    history_voxel_size_m: float = 0.012,
    min_history_observations: int = 2,
    min_motion_m: float = 0.025,
    min_motion_deg: float = 6.0,
    max_history_points: int = 800,
) -> ObjectGeometryFrame:
    """Build one frame without importing SAM2 or robot-specific OpenPI modules.

    ``histories`` contain already-projected object points. Stable target points
    must be in the base frame. Manipulated points must be in the moving local
    frame and have a current base transform in ``object_poses``.
    """

    histories = {} if histories is None else histories
    object_poses = {} if object_poses is None else object_poses
    moving = set(manipulated_objects)
    valid = {} if manipulated_valid is None else manipulated_valid
    result: dict[str, ObjectGeometry] = {}
    for name, mask in object_masks.items():
        current_raw = mask_depth_to_cloud(
            depth,
            mask,
            rays_base_per_meter,
            transform_base_color,
            color_intrinsics,
            depth_scale_m_per_unit=depth_scale_m_per_unit,
            rgb=rgb,
            workspace_bounds=workspace_bounds,
        )
        current = PointCloud(
            xyz=current_raw.xyz,
            rgb=current_raw.rgb,
            source_pixel=current_raw.source_pixel,
            source=np.zeros(len(current_raw.xyz), dtype=np.int8),
            confidence=current_raw.confidence,
        )
        observations = list(histories.get(name, ()))
        if name in moving and not valid.get(name, True):
            observations = []
        if observations:
            history_result = build_causal_history(
                observations,
                current_frame=frame_index,
                window_frames=history_window_frames,
                voxel_size_m=history_voxel_size_m,
                min_observations=min_history_observations,
                require_motion=name in moving,
                min_motion_m=min_motion_m,
                min_motion_deg=min_motion_deg,
                max_points=max_history_points,
            )
            history_xyz = history_to_base(history_result, object_poses[name]) if name in moving else history_result.points
            age = np.asarray(frame_index - history_result.last_seen_frame, dtype=np.int32)
            history = PointCloud(
                xyz=history_xyz,
                source=np.ones(len(history_xyz), dtype=np.int8),
                age=age,
            )
            result[name] = ObjectGeometry(
                current=current,
                history=history,
                history_support=history_result.support_count,
                history_space="base",
            )
        else:
            result[name] = ObjectGeometry(current=current)
    return ObjectGeometryFrame(frame_index=int(frame_index), objects=result)
