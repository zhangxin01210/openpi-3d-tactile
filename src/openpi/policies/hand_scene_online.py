"""Online adapter around the *training* mesh/pose/hand-scene producer.

No separate point sampler, mesh transform, confidence heuristic, or tactile
reference implementation is permitted here.  This module only adapts the live
current-frame arrays to the exact training-side schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from openpi.object_geometry.production import (
    CausalHandSceneState,
    PoseObservation,
    _geometry_record,
    as_transform,
    load_mesh_surface_points,
)
from openpi.object_geometry_dataset.derived import (
    CausalTactileSample,
    HandSceneSample,
    ObjectGeometrySample,
    POINT_COUNT,
    TACTILE_POINT_COUNT,
    FUNNEL_MASK_SIZE,
)
from openpi.spatial.tactile_ema import CausalTactileEMA


@dataclass(frozen=True)
class OnlinePose:
    T_base_link_object: np.ndarray
    valid: bool
    confidence: float
    status: str
    pose_source: str
    observed_pose_valid: bool

    def as_training_observation(self, frame_index: int) -> PoseObservation:
        transform = as_transform(self.T_base_link_object, "T_base_link_object")
        # No causal extrapolation is allowed to masquerade as observed geometry.
        observed = bool(self.valid and self.observed_pose_valid)
        status = str(self.status).lower()
        if any(word in status for word in ("predict", "occlud", "invalid", "lost")):
            observed = False
        return PoseObservation(
            frame_index=int(frame_index),
            T_base_link_object=transform,
            valid=observed,
            confidence=float(self.confidence) if observed else 0.0,
            status=self.status,
            pose_source=self.pose_source,
            observed_pose_valid=observed,
            raw_heuristic_score=None,
        )


class _CurrentTactileProvider:
    """One-frame bridge to the training CausalHandSceneState; not another filter."""
    def __init__(self) -> None:
        self.force: np.ndarray | None = None

    def get_force_norm(self, episode: int, frame: int) -> np.ndarray | None:
        return self.force


class OnlineObjectGeometryBuilder:
    """Build canonical 512+512 mesh clouds and Hand-Scene v1 state."""

    def __init__(
        self,
        *,
        funnel_surface_points_m: np.ndarray,
        flask_surface_points_m: np.ndarray,
        static_target_anchor: np.ndarray | None = None,
        static_target_anchor_provenance: str | None = None,
        static_target_anchor_start_frame: int = 1,
        tactile_ema_tau_s: float = 1.0,
        contact_force_threshold: float = 0.05,
    ):
        self._funnel_points = self._validate_points(funnel_surface_points_m, "funnel")
        self._flask_points = self._validate_points(flask_surface_points_m, "target_flask")
        if static_target_anchor is not None and static_target_anchor_provenance not in (
            "human_reviewed", "reference_geometry"
        ):
            raise ValueError("Static target anchor requires explicit reviewed provenance")
        if static_target_anchor is None and static_target_anchor_provenance is not None:
            raise ValueError("Static target anchor provenance supplied without anchor")
        if int(static_target_anchor_start_frame) < 1:
            raise ValueError("Static target prior cannot start at frame zero")
        self._static_anchor = (
            as_transform(static_target_anchor, "static_target_anchor")
            if static_target_anchor is not None else None
        )
        self._static_anchor_start_frame = int(static_target_anchor_start_frame)
        self._tactile = _CurrentTactileProvider()
        self._hand_scene = CausalHandSceneState(
            spatial_tactile=self._tactile,
            contact_force_threshold=contact_force_threshold,
        )
        self._ema = CausalTactileEMA(tau_s=tactile_ema_tau_s, taxel_count=TACTILE_POINT_COUNT)
        self._previous_frame: int | None = None
        self._last_timestamp: float | None = None

    def reset_episode(self) -> None:
        """Reset all tactile history and frame-order checks."""
        self._hand_scene.reset()
        self._ema.reset()
        self._previous_frame = None
        self._last_timestamp = None

    @staticmethod
    def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(points, dtype=np.float32)
        if value.shape != (POINT_COUNT, 3) or not np.isfinite(value).all():
            raise ValueError(f"{name} surface points must be finite ({POINT_COUNT},3)")
        return value.copy()

    @staticmethod
    def load_scale_manifest(path: str | Path) -> dict[str, float]:
        """Read explicit mesh-to-metre scale; never guess a normalized SAM3D unit."""
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        def find(name: str) -> float:
            candidates = [data.get(name)]
            for outer in ("scales", "objects", "meshes", "assets"):
                mapping = data.get(outer)
                if isinstance(mapping, dict):
                    candidates.append(mapping.get(name))
            for entry in candidates:
                if isinstance(entry, (int, float)):
                    return float(entry)
                if isinstance(entry, dict):
                    for key in ("scale_m_per_unit", "meters_per_unit", "scale"):
                        if key in entry:
                            return float(entry[key])
            raise ValueError(f"No explicit scale for {name}: {path}")
        scales = {name: find(name) for name in ("funnel", "target_flask")}
        if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
            raise ValueError("Mesh scale values must be positive and finite")
        return scales

    @classmethod
    def from_meshes(
        cls, *, funnel_mesh: str | Path, flask_mesh: str | Path,
        scales: Mapping[str, float], sampling_seeds: Mapping[str, int],
        **kwargs: Any,
    ) -> "OnlineObjectGeometryBuilder":
        # Same sampler and same seeding used by the training sidecar producer.
        funnel, _ = load_mesh_surface_points(
            funnel_mesh, scale_m_per_unit=scales["funnel"],
            count=POINT_COUNT, seed=sampling_seeds["funnel"],
        )
        flask, _ = load_mesh_surface_points(
            flask_mesh, scale_m_per_unit=scales["target_flask"],
            count=POINT_COUNT, seed=sampling_seeds["target_flask"],
        )
        return cls(funnel_surface_points_m=funnel, flask_surface_points_m=flask, **kwargs)

    @staticmethod
    def _pose(value: OnlinePose | Mapping[str, Any] | PoseObservation | None, frame: int) -> PoseObservation:
        if value is None:
            return OnlinePose(np.eye(4), False, 0.0, "invalid", "unknown", False).as_training_observation(frame)
        if isinstance(value, PoseObservation):
            if int(value.frame_index) != frame:
                raise ValueError("Pose frame_index mismatch")
            return value
        if isinstance(value, Mapping):
            value = OnlinePose(
                T_base_link_object=value["T_base_link_object"],
                valid=bool(value.get("valid", False)),
                confidence=float(value.get("confidence", 0.0)),
                status=str(value.get("status", "invalid")),
                pose_source=str(value.get("pose_source", "unknown")),
                observed_pose_valid=bool(value.get("observed_pose_valid", False)),
            )
        return value.as_training_observation(frame)

    def build_frame(
        self, *, frame_index: int, timestamp_s: float,
        funnel_pose: OnlinePose | Mapping[str, Any] | PoseObservation | None,
        target_flask_pose: OnlinePose | Mapping[str, Any] | PoseObservation | None,
        tactile_force_norm: np.ndarray | None,
        funnel_mask: np.ndarray | None,
        target_flask_mask: np.ndarray | None = None,
        funnel_mask_hand_offset_xy: np.ndarray | None = None,
    ) -> ObjectGeometrySample:
        frame = int(frame_index)
        timestamp = float(timestamp_s)
        if frame < 0 or not np.isfinite(timestamp):
            raise ValueError("Frame/timestamp must be nonnegative and finite")
        if self._previous_frame is not None and frame <= self._previous_frame:
            raise ValueError("Hand-scene requires strictly increasing frame IDs")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Non-monotonic timestamp")
        contiguous = self._previous_frame is None or frame == self._previous_frame + 1

        funnel = self._pose(funnel_pose, frame)
        target = self._pose(target_flask_pose, frame)
        target_source = 0
        if self._static_anchor is not None and frame >= self._static_anchor_start_frame:
            target = PoseObservation(
                frame_index=frame, T_base_link_object=self._static_anchor,
                valid=True, confidence=1.0, status="static_geometry_prior",
                pose_source="static_geometry_prior", observed_pose_valid=False,
                raw_heuristic_score=None,
            )
            target_source = 1

        force = None if tactile_force_norm is None else np.asarray(tactile_force_norm, dtype=np.float32)
        if force is not None:
            if force.shape != (TACTILE_POINT_COUNT,):
                raise ValueError("tactile_force_norm must contain 600 taxels")
            if not np.isfinite(force).all():
                force = None
            else:
                force = np.maximum(force, 0.0)
        # Share the offline state machine, including nearest-neighbour mask handling.
        state = self._hand_scene.next_from_arrays(
            frame=frame,
            force_norm=force,
            funnel_mask=funnel_mask,
            funnel_mask_hand_offset_xy=funnel_mask_hand_offset_xy,
        )
        ema = self._ema.update(
            timestamp_s=timestamp, force_norm=force if contiguous else None
        )
        state["manipulated_relation_valid"] = np.asarray(bool(funnel.valid))
        state["target_relation_valid"] = np.asarray(bool(target.valid))
        state["manipulated_geometry_quality"] = np.asarray(float(funnel.valid), dtype=np.float32)
        state["target_geometry_quality"] = np.asarray(float(target.valid), dtype=np.float32)

        state["causal_tactile"] = CausalTactileSample(
            ema_before_force_norm=ema.ema_before_force_norm,
            ema_after_force_norm=ema.ema_after_force_norm,
            current_minus_ema_force_norm=ema.current_minus_ema_force_norm,
            current_minus_previous_force_norm=ema.current_minus_previous_force_norm,
            reference_valid=ema.reference_valid,
            previous_valid=ema.previous_valid,
            sample_valid=ema.sample_valid,
            alpha=np.asarray(ema.alpha, dtype=np.float32),
        )

        # Exact training-side function: structural validity, point age, source
        # labels and confidence are not separately implemented here.
        manipulated = _geometry_record(self._funnel_points, funnel, source=0)
        goal = _geometry_record(self._flask_points, target, source=target_source)
        self._previous_frame = frame
        self._last_timestamp = timestamp
        return ObjectGeometrySample(
            episode_index=0, frame_index=frame,
            **{f"manipulated_{name}": value for name, value in manipulated.items()},
            **{f"target_{name}": value for name, value in goal.items()},
            hand_scene=HandSceneSample(**state),
        )

    def build_spatial_object_geometry(self, **kwargs: Any) -> dict[str, Any]:
        return self.build_frame(**kwargs).as_dict()
