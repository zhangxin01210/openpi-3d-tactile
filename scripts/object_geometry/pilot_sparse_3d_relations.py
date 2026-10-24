#!/usr/bin/env python3
"""Causal, lightweight 3-D keypoints from reviewed RGB-D masks.

This is a geometry prototype, not an object detector.  It consumes existing
SAM2 PNG masks and an explicit 2-D initialization file, then estimates a small
set of keypoints in ``base_link``.  Depth samples inside a mask are labelled as
candidate observations only: transparent objects can expose background depth.

Example (computer 4)::

    python scripts/object_geometry/pilot_sparse_3d_relations.py \
      --repo-root /workspace/mnt/sqzhang26/openpi-3d-tactile \
      --dataset /workspace/mnt/sqzhang26/openpi-3d-tactile/data/1018_peg_all \
      --episode 23 \
      --mask-root /workspace/mnt/sqzhang26/hoi/outputs/level1_sequence_e23/masks \
      --init-json /workspace/mnt/sqzhang26/hoi/outputs/level1_sequence_e23/keypoint_init_e23.json \
      --output /workspace/mnt/sqzhang26/hoi/outputs/level1_sequence_e23/sparse_3d_relations_front_candidate

The initialization format is intentionally explicit::

    {
      "schema_version": 1,
      "objects": {
        "funnel": {"initial_frame": 212,
          "keypoints": {"shaft_end": [u, v], "axis_reference": [u, v]}},
        "target_flask": {"initial_frame": 0,
          "keypoints": {"mouth_center": [u, v], "axis_reference": [u, v]}}
      }
    }

The two points per object are local RGB anchors.  The mask centroid and PCA
axis transport those anchors through translation and small image-plane turns;
an anchor outside a mask is not silently snapped to a plausible point.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# This script is intentionally directly executable from a clean checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


OBJECT_KEYPOINTS: dict[str, tuple[str, str]] = {
    "funnel": ("shaft_end", "axis_reference"),
    "target_flask": ("mouth_center", "axis_reference"),
}
OBJECT_COLORS = {"funnel": (35, 210, 255), "target_flask": (220, 90, 245)}  # BGR


def _finite_uv(value: Any, *, label: str) -> np.ndarray:
    uv = np.asarray(value, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError(f"{label} must be two finite image coordinates")
    return uv


@dataclass(frozen=True)
class Initialization:
    """Explicit human-provided 2-D anchors for one episode."""

    initial_frame: int
    keypoints: dict[str, np.ndarray]


def load_initialization(path: Path) -> dict[str, Initialization]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported keypoint initialization schema: {path}")
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("initialization must contain an objects object")
    result: dict[str, Initialization] = {}
    for object_name, names in OBJECT_KEYPOINTS.items():
        raw = objects.get(object_name)
        if not isinstance(raw, dict):
            raise ValueError(f"initialization is missing object {object_name!r}")
        frame = int(raw.get("initial_frame", -1))
        if frame < 0:
            raise ValueError(f"{object_name}.initial_frame must be >= 0")
        raw_points = raw.get("keypoints")
        if not isinstance(raw_points, dict):
            raise ValueError(f"{object_name}.keypoints must be an object")
        points: dict[str, np.ndarray] = {}
        for keypoint_name in names:
            if keypoint_name not in raw_points:
                raise ValueError(f"{object_name}.keypoints is missing {keypoint_name!r}")
            points[keypoint_name] = _finite_uv(
                raw_points[keypoint_name], label=f"{object_name}.{keypoint_name}"
            )
        result[object_name] = Initialization(frame, points)
    return result


@dataclass(frozen=True)
class MaskGeometry:
    centroid_uv: np.ndarray
    basis_uv: np.ndarray  # columns: principal and perpendicular image axes
    area_pixels: int
    bbox_xyxy: tuple[int, int, int, int]


def mask_geometry(mask: np.ndarray) -> MaskGeometry | None:
    """Compute local RGB-mask geometry without filling or resizing the mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {binary.shape}")
    yy, xx = np.nonzero(binary)
    if len(xx) < 3:
        return None
    points = np.column_stack((xx, yy)).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(1, len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = np.linalg.norm(principal)
    if norm <= 1e-9:
        return None
    principal = principal / norm
    perpendicular = np.array([-principal[1], principal[0]], dtype=np.float64)
    basis = np.column_stack((principal, perpendicular))
    return MaskGeometry(
        centroid_uv=centroid,
        basis_uv=basis,
        area_pixels=len(points),
        bbox_xyxy=(int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())),
    )


def _align_basis(current: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return current
    candidates = [
        current * np.array([sign0, sign1], dtype=np.float64)
        for sign0 in (-1.0, 1.0)
        for sign1 in (-1.0, 1.0)
    ]
    return min(candidates, key=lambda value: float(np.linalg.norm(value - previous)))


@dataclass
class TemporalUVTrack:
    position_uv: np.ndarray | None = None
    velocity_uv_per_frame: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    last_frame: int | None = None
    was_predicted: bool = False

    def update(self, frame: int, observation_uv: np.ndarray | None) -> dict[str, Any]:
        frame = int(frame)
        dt = 1 if self.last_frame is None else max(1, frame - self.last_frame)
        predicted = None
        if self.position_uv is not None:
            predicted = self.position_uv + self.velocity_uv_per_frame * dt
        reobserved = bool(observation_uv is not None and self.was_predicted)
        if observation_uv is not None:
            measurement = _finite_uv(observation_uv, label="observation_uv")
            if predicted is None:
                filtered = measurement
                self.velocity_uv_per_frame = np.zeros(2, dtype=np.float64)
            else:
                residual = measurement - predicted
                filtered = predicted + 0.72 * residual
                self.velocity_uv_per_frame = self.velocity_uv_per_frame + 0.22 * residual / dt
            self.position_uv = filtered
            self.was_predicted = False
            source = "observed"
        elif predicted is not None:
            self.position_uv = predicted
            self.velocity_uv_per_frame *= 0.82
            self.was_predicted = True
            source = "predicted"
        else:
            source = "invalid"
        self.last_frame = frame
        return {
            "uv": None if self.position_uv is None else [float(x) for x in self.position_uv],
            "source": source,
            "observed": source == "observed",
            "reobserved": reobserved,
        }


@dataclass
class TemporalPointTrack:
    """Alpha-beta-like causal 3-D state with conservative prediction validity."""

    max_prediction_frames: int = 8
    max_jump_m: float = 0.20
    position_m: np.ndarray | None = None
    velocity_m_per_frame: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    confidence: float = 0.0
    last_frame: int | None = None
    prediction_age: int = 0
    was_predicted: bool = False

    def update(
        self,
        frame: int,
        observation_m: np.ndarray | None,
        observation_confidence: float = 0.0,
    ) -> dict[str, Any]:
        frame = int(frame)
        dt = 1 if self.last_frame is None else max(1, frame - self.last_frame)
        predicted = None
        if self.position_m is not None:
            predicted = self.position_m + self.velocity_m_per_frame * dt
        reobserved = bool(observation_m is not None and self.was_predicted)
        if observation_m is not None:
            measurement = np.asarray(observation_m, dtype=np.float64)
            if measurement.shape != (3,) or not np.isfinite(measurement).all():
                raise ValueError("observation_m must be three finite coordinates")
            confidence = float(np.clip(observation_confidence, 0.0, 1.0))
            if predicted is None:
                filtered = measurement
                self.velocity_m_per_frame = np.zeros(3, dtype=np.float64)
                jump = 0.0
            else:
                residual = measurement - predicted
                jump = float(np.linalg.norm(residual))
                filtered = predicted + 0.72 * residual
                self.velocity_m_per_frame = self.velocity_m_per_frame + 0.22 * residual / dt
                if self.max_jump_m > 0:
                    confidence *= float(np.exp(-max(0.0, jump - self.max_jump_m) / self.max_jump_m))
            self.position_m = filtered
            self.confidence = confidence
            self.prediction_age = 0
            self.was_predicted = False
            source = "observed"
            valid = confidence >= 0.20
        elif predicted is not None and self.prediction_age < self.max_prediction_frames:
            self.position_m = predicted
            self.velocity_m_per_frame *= 0.82
            self.prediction_age += dt
            self.confidence *= 0.72 ** dt
            self.was_predicted = True
            source = "predicted"
            valid = False
            jump = None
        else:
            self.position_m = None
            self.velocity_m_per_frame[:] = 0.0
            self.prediction_age += dt
            self.confidence = 0.0
            self.was_predicted = False
            source = "invalid"
            valid = False
            jump = None
        self.last_frame = frame
        return {
            "position_m": None if self.position_m is None else [float(x) for x in self.position_m],
            "velocity_m_per_frame": [float(x) for x in self.velocity_m_per_frame],
            "valid": bool(valid),
            "confidence": float(np.clip(self.confidence, 0.0, 1.0)),
            "source": source,
            "prediction_age_frames": int(self.prediction_age),
            "reobserved": reobserved,
            "jump_m": jump,
        }


@dataclass
class ObjectTracker:
    """Mask-geometry transport plus independent causal UV tracks."""

    initialization: Initialization
    uv_tracks: dict[str, TemporalUVTrack] = field(default_factory=dict)
    initial_geometry: MaskGeometry | None = None
    previous_basis: np.ndarray | None = None
    local_coordinates: dict[str, np.ndarray] = field(default_factory=dict)
    last_geometry: MaskGeometry | None = None
    last_mask_area: int = 0
    initialization_frame_seen: bool = False

    def update(self, frame: int, mask: np.ndarray | None) -> tuple[dict[str, Any], list[str]]:
        if frame < self.initialization.initial_frame:
            return {
                "visible": False,
                "area_pixels": 0,
                "keypoints": {name: self.uv_tracks.setdefault(name, TemporalUVTrack()).update(frame, None)
                               for name in self.initialization.keypoints},
            }, ["before_initialization"]

        geometry = None if mask is None else mask_geometry(mask)
        anomalies: list[str] = []
        if geometry is None:
            anomalies.append("missing_or_too_small_mask")
        elif self.initial_geometry is None:
            self.initialization_frame_seen = True
            self.initial_geometry = geometry if frame == self.initialization.initial_frame else None
            if self.initial_geometry is not None:
                self.previous_basis = geometry.basis_uv.copy()
                for name, uv in self.initialization.keypoints.items():
                    self.local_coordinates[name] = geometry.basis_uv.T @ (uv - geometry.centroid_uv)
            else:
                anomalies.append("missing_initialization_mask")
        if geometry is not None and self.initial_geometry is not None:
            basis = _align_basis(geometry.basis_uv, self.previous_basis)
            self.previous_basis = basis
            self.last_geometry = geometry
            self.last_mask_area = geometry.area_pixels
            if self.initial_geometry.area_pixels:
                area_ratio = geometry.area_pixels / self.initial_geometry.area_pixels
                if area_ratio < 0.35 or area_ratio > 3.0:
                    anomalies.append("mask_area_changed")
        else:
            basis = self.previous_basis

        keypoint_rows: dict[str, Any] = {}
        for name in self.initialization.keypoints:
            observation = None
            if geometry is not None and basis is not None:
                observation = geometry.centroid_uv + basis @ self.local_coordinates[name]
                x = int(np.rint(observation[0]))
                y = int(np.rint(observation[1]))
                if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])):
                    anomalies.append(f"{name}_outside_current_mask")
                    observation = None
            row = self.uv_tracks.setdefault(name, TemporalUVTrack()).update(frame, observation)
            keypoint_rows[name] = row
        return {
            "visible": geometry is not None,
            "area_pixels": 0 if geometry is None else geometry.area_pixels,
            "bbox_xyxy": None if geometry is None else list(geometry.bbox_xyxy),
            "keypoints": keypoint_rows,
        }, sorted(set(anomalies))


@dataclass(frozen=True)
class DepthProjection:
    xyz_base_m: np.ndarray
    xyz_color_m: np.ndarray
    uv: np.ndarray
    depth_m: np.ndarray
    valid_projection: np.ndarray


def project_depth_to_base(depth: np.ndarray, camera: Any) -> DepthProjection:
    """Use the repository camera cache and return base_link points plus RGB UV."""
    from openpi.spatial.geometry import build_camera_cache
    from openpi.spatial.geometry import project_pinhole
    from openpi.spatial.geometry import transform_points

    raw = np.asarray(depth)
    if raw.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {raw.shape}")
    cache = build_camera_cache(camera, image_height=raw.shape[0], image_width=raw.shape[1])
    depth_m_all = raw.reshape(-1).astype(np.float32) * np.float32(camera.depth_scale_m_per_unit)
    valid_depth = np.isfinite(depth_m_all) & (depth_m_all > 0)
    xyz_base = cache.rays_base_per_meter[valid_depth] * depth_m_all[valid_depth, None]
    xyz_base += cache.base_origin_m[None, :]
    xyz_color = transform_points(xyz_base, np.linalg.inv(camera.T_base_color))
    uv = project_pinhole(xyz_color, camera.color_intrinsics)
    valid_projection = np.isfinite(uv).all(axis=1) & (xyz_color[:, 2] > 0)
    return DepthProjection(
        xyz_base_m=xyz_base.astype(np.float32),
        xyz_color_m=xyz_color.astype(np.float32),
        uv=uv.astype(np.float32),
        depth_m=depth_m_all[valid_depth],
        valid_projection=valid_projection,
    )


@dataclass(frozen=True)
class DepthObservation:
    position_m: np.ndarray | None
    confidence: float
    candidate_count: int
    spread_m: float | None
    reason: str | None


def sample_depth_observation(
    projection: DepthProjection,
    mask: np.ndarray,
    uv_target: np.ndarray,
    *,
    radius_px: float = 14.0,
    min_samples: int = 3,
    max_samples: int = 32,
) -> DepthObservation:
    """Robustly sample local candidate depth; never treats the whole mask as surface."""
    binary = np.asarray(mask, dtype=bool)
    target = _finite_uv(uv_target, label="uv_target")
    if binary.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {binary.shape}")
    pixels = np.rint(projection.uv).astype(np.int64)
    h, w = binary.shape
    in_image = (
        projection.valid_projection
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < h)
    )
    inside_mask = np.zeros(len(pixels), dtype=bool)
    inside_mask[in_image] = binary[pixels[in_image, 1], pixels[in_image, 0]]
    distance = np.linalg.norm(projection.uv - target[None, :], axis=1)
    selected = inside_mask & np.isfinite(distance) & (distance <= float(radius_px))
    indices = np.flatnonzero(selected)
    if len(indices) < min_samples:
        reason = "missing_depth" if len(indices) == 0 else "insufficient_local_depth"
        return DepthObservation(None, 0.0, len(indices), None, reason)
    indices = indices[np.argsort(distance[indices])[:max_samples]]
    xyz = projection.xyz_base_m[indices]
    median = np.median(xyz, axis=0)
    residual = np.linalg.norm(xyz - median[None, :], axis=1)
    inliers = residual <= max(0.012, float(np.percentile(residual, 75)) * 1.5)
    if int(inliers.sum()) < min_samples:
        return DepthObservation(None, 0.0, len(indices), None, "inconsistent_local_depth")
    xyz_inlier = xyz[inliers]
    position = np.median(xyz_inlier, axis=0).astype(np.float32)
    spread = float(np.median(np.linalg.norm(xyz_inlier - position[None, :], axis=1)))
    count_score = min(1.0, len(xyz_inlier) / 12.0)
    distance_score = float(np.exp(-float(np.median(distance[indices[inliers]])) / max(1.0, radius_px)))
    spread_score = float(np.exp(-spread / 0.025))
    confidence = float(np.clip(0.45 * count_score + 0.30 * distance_score + 0.25 * spread_score, 0.0, 1.0))
    return DepthObservation(position, confidence, len(indices), spread, None)


def _axis_row(origin: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    origin_xyz = origin.get("position_m")
    reference_xyz = reference.get("position_m")
    if not origin.get("valid") or not reference.get("valid") or origin_xyz is None or reference_xyz is None:
        return {"valid": False, "confidence": 0.0, "origin_m": origin_xyz, "reference_m": reference_xyz}
    vector = np.asarray(reference_xyz, dtype=np.float64) - np.asarray(origin_xyz, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-4 or not np.isfinite(length):
        return {"valid": False, "confidence": 0.0, "origin_m": origin_xyz, "reference_m": reference_xyz}
    return {
        "valid": True,
        "confidence": float(min(origin["confidence"], reference["confidence"])),
        "origin_m": origin_xyz,
        "reference_m": reference_xyz,
        "unit_vector_base": (vector / length).tolist(),
        "length_m": length,
    }


def _mask_path(mask_root: Path, object_name: str, frame: int) -> Path:
    return Path(mask_root) / object_name / f"{int(frame):05d}.png"


def _read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    import cv2

    if not path.is_file():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Cannot decode mask: {path}")
    if raw.shape != shape_hw:
        raise ValueError(f"Mask {path} has shape {raw.shape}, expected RGB {shape_hw}")
    return raw > 127


def _overlay(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray | None],
    rows: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    import cv2

    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    for object_name, mask in masks.items():
        if mask is not None:
            color_rgb = np.asarray(OBJECT_COLORS[object_name][::-1], dtype=np.float32)
            canvas[mask] = (0.55 * canvas[mask] + 0.45 * color_rgb).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for object_name, object_row in rows.items():
        color = OBJECT_COLORS[object_name]
        keypoints = object_row["keypoints"]
        for keypoint in keypoints.values():
            uv = keypoint.get("uv")
            if uv is None:
                continue
            point = tuple(round(float(x)) for x in uv)
            marker_color = (60, 220, 60) if keypoint["source"] == "observed" else (0, 165, 255)
            if keypoint["source"] == "invalid":
                marker_color = (0, 0, 255)
            cv2.circle(canvas, point, 5, marker_color, -1)
            cv2.circle(canvas, point, 7, color, 1)
        names = list(keypoints)
        if len(names) == 2:
            first, second = keypoints[names[0]], keypoints[names[1]]
            if first.get("uv") is not None and second.get("uv") is not None:
                p0 = tuple(round(float(x)) for x in first["uv"])
                p1 = tuple(round(float(x)) for x in second["uv"])
                cv2.line(canvas, p0, p1, color, 2)
        cv2.putText(canvas, object_name, (10, 24 + 22 * list(rows).index(object_name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _build_3d_html(frames: list[dict[str, Any]], output: Path) -> None:
    data = json.dumps(frames, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    try:
        from plotly.offline import get_plotlyjs

        plotly_js = get_plotlyjs()
    except Exception as error:  # pragma: no cover - fallback depends on environment packaging.
        output.write_text(
            "<html><body><h3>Plotly unavailable</h3><pre>" + json.dumps(
                {"error": str(error), "frames": frames}, ensure_ascii=False, indent=2
            ) + "</pre></body></html>",
            encoding="utf-8",
        )
        return
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Sparse 3D relations</title>
<script>{plotly_js}</script></head><body><h3>Causal sparse 3D keypoints (base_link, m)</h3>
<div id='plot' style='width:100%;height:88vh'></div><script>
const frames={data}; const traces=[];
const colors={{funnel:'#23d2ff',target_flask:'#dc5af5'}};
for (const [obj,color] of Object.entries(colors)) {{
  for (const kp of (obj==='funnel'?['shaft_end','axis_reference']:['mouth_center','axis_reference'])) {{
    const rows=frames.map(f=>f.objects[obj].keypoints[kp]);
    traces.push({{x:rows.map(r=>r.position_m?r.position_m[0]:null),y:rows.map(r=>r.position_m?r.position_m[1]:null),z:rows.map(r=>r.position_m?r.position_m[2]:null),mode:'lines+markers',name:obj+'.'+kp,line:{{color}},marker:{{size:4}}}});
  }}
}}
Plotly.newPlot('plot',traces,{{scene:{{xaxis:{{title:'base x (m)'}},yaxis:{{title:'base y (m)'}},zaxis:{{title:'base z (m)'}}}},legend:{{orientation:'h'}}}},{{responsive:true}});
</script></body></html>"""
    output.write_text(html, encoding="utf-8")


def _process_frames(
    rows: Iterable[tuple[int, np.ndarray, np.ndarray]],
    *,
    camera: Any,
    mask_root: Path,
    output: Path,
    initializations: dict[str, Initialization],
    sample_radius_px: float,
    min_depth_samples: int,
    max_prediction_frames: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, int]]:
    trackers = {
        name: ObjectTracker(initialization=initialization)
        for name, initialization in initializations.items()
    }
    point_tracks = {
        f"{object_name}.{keypoint}": TemporalPointTrack(max_prediction_frames=max_prediction_frames)
        for object_name, keypoints in OBJECT_KEYPOINTS.items()
        for keypoint in keypoints
    }
    frame_rows: list[dict[str, Any]] = []
    timing_ms: list[float] = []
    anomaly_counts: dict[str, int] = {}
    for frame, rgb, depth in rows:
        start = time.perf_counter()
        shape_hw = tuple(int(x) for x in rgb.shape[:2])
        masks = {
            object_name: _read_mask(_mask_path(mask_root, object_name, frame), shape_hw)
            for object_name in OBJECT_KEYPOINTS
        }
        object_uv_rows: dict[str, dict[str, Any]] = {}
        all_anomalies: list[str] = []
        for object_name, tracker in trackers.items():
            object_row, anomalies = tracker.update(frame, masks[object_name])
            object_uv_rows[object_name] = object_row
            all_anomalies.extend(f"{object_name}:{item}" for item in anomalies)

        projection = project_depth_to_base(depth, camera)
        object_rows: dict[str, dict[str, Any]] = {}
        for object_name, keypoint_names in OBJECT_KEYPOINTS.items():
            keypoint_rows: dict[str, Any] = {}
            object_anomalies = [item for item in all_anomalies if item.startswith(f"{object_name}:")]
            for keypoint_name in keypoint_names:
                uv_row = object_uv_rows[object_name]["keypoints"][keypoint_name]
                observation = DepthObservation(None, 0.0, 0, None, "no_2d_observation")
                if uv_row.get("source") == "observed" and uv_row.get("uv") is not None:
                    observation = sample_depth_observation(
                        projection,
                        masks[object_name] if masks[object_name] is not None else np.zeros(shape_hw, dtype=bool),
                        np.asarray(uv_row["uv"], dtype=np.float64),
                        radius_px=sample_radius_px,
                        min_samples=min_depth_samples,
                    )
                point_key = f"{object_name}.{keypoint_name}"
                point_row = point_tracks[point_key].update(
                    frame,
                    observation.position_m,
                    observation.confidence,
                )
                point_row.update({
                    "uv": uv_row.get("uv"),
                    "candidate_depth_count": observation.candidate_count,
                    "candidate_spread_m": observation.spread_m,
                    "observation_reason": observation.reason,
                })
                if observation.reason is not None:
                    object_anomalies.append(f"{keypoint_name}:{observation.reason}")
                keypoint_rows[keypoint_name] = point_row
            axis = _axis_row(keypoint_rows[keypoint_names[0]], keypoint_rows[keypoint_names[1]])
            object_rows[object_name] = {
                "mask": object_uv_rows[object_name],
                "keypoints": keypoint_rows,
                "axis": axis,
                "anomalies": sorted(set(object_anomalies)),
            }
        all_anomalies = sorted(set(all_anomalies + [item for row in object_rows.values() for item in row["anomalies"]]))
        for anomaly in all_anomalies:
            anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        elapsed = (time.perf_counter() - start) * 1000.0
        timing_ms.append(elapsed)
        frame_row = {
            "frame_index": int(frame),
            "objects": object_rows,
            "anomalies": all_anomalies,
            "timing_ms": elapsed,
        }
        frame_rows.append(frame_row)
        _overlay(rgb, masks, object_rows, output / "overlays" / f"frame_{int(frame):05d}.png")
    return frame_rows, timing_ms, anomaly_counts


def run(args: argparse.Namespace) -> None:
    from openpi.object_geometry.calibration import front_calibration_provenance
    from openpi.object_geometry.calibration import make_front_geometry_config
    from openpi.spatial.calibration import load_camera_calibrations
    from openpi.spatial.geometry import apply_diagnostic_overrides
    from openpi.spatial_dataset.source import RawSpatialDataset

    repo_root = Path(args.repo_root).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")
    output.mkdir(parents=True, exist_ok=True)
    initializations = load_initialization(Path(args.init_json).expanduser().resolve())
    config, profile = make_front_geometry_config(repo_root, args.front_calibration_profile)
    bundle = load_camera_calibrations(
        repo_root=repo_root,
        calibration_config=config.calibration,
        camera_roles=("front",),
    )
    camera = apply_diagnostic_overrides(bundle.cameras, config.diagnostics)["front"]
    calibration = front_calibration_provenance(repo_root, profile, config=config)
    dataset_reader = RawSpatialDataset(dataset, episode=args.episode)

    def source_rows() -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
        rows = dataset_reader.iter_rows(depth_roles=("front",))
        images = dataset_reader.iter_video_frames("front")
        for row, video in zip(rows, images, strict=True):
            frame = int(row["frame_index"])
            if int(video.frame_index) != frame:
                raise ValueError(f"RGB/depth frame mismatch: row={frame}, video={video.frame_index}")
            yield frame, np.asarray(video.rgb), np.asarray(row["observation.depths.cam_front"])

    frames, timings, anomaly_counts = _process_frames(
        source_rows(),
        camera=camera,
        mask_root=mask_root,
        output=output,
        initializations=initializations,
        sample_radius_px=args.sample_radius_px,
        min_depth_samples=args.min_depth_samples,
        max_prediction_frames=args.max_prediction_frames,
    )
    with (output / "keypoints.jsonl").open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
    _build_3d_html(frames, output / "keypoints_3d.html")
    timing_array = np.asarray(timings, dtype=np.float64)
    report = {
        "schema_version": 1,
        "episode": int(args.episode),
        "dataset": str(dataset),
        "mask_root": str(mask_root),
        "initialization": str(Path(args.init_json).expanduser().resolve()),
        "coordinate_frame": "base_link",
        "units": "meters",
        "calibration": calibration,
        "calibration_profile": None if profile is None else str(profile),
        "method": {
            "mask_transport": "mask centroid + PCA local-anchor transport",
            "depth_observation": "nearest robust local candidate depth under RGB mask",
            "temporal_filter": "causal alpha-beta style prediction; predicted states are valid=false",
            "surface_warning": "transparent-mask depth is candidate geometry, not asserted object surface",
            "robot_following": "no robot state or hand pose is used",
        },
        "parameters": {
            "sample_radius_px": args.sample_radius_px,
            "min_depth_samples": args.min_depth_samples,
            "max_prediction_frames": args.max_prediction_frames,
        },
        "frame_count": len(frames),
        "timing_ms": {
            "mean": float(timing_array.mean()) if len(timing_array) else None,
            "p50": float(np.percentile(timing_array, 50)) if len(timing_array) else None,
            "p95": float(np.percentile(timing_array, 95)) if len(timing_array) else None,
            "max": float(timing_array.max()) if len(timing_array) else None,
        },
        "anomaly_counts": anomaly_counts,
        "files": ["keypoints.jsonl", "keypoints_3d.html", "overlays/"],
        "warning": "This prototype does not establish transparent object surfaces or physical contact truth.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(frames), "timing_ms": report["timing_ms"]}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--mask-root", type=Path, required=True,
                        help="Existing SAM2 mask directory containing funnel/ and target_flask/")
    parser.add_argument("--init-json", type=Path, required=True,
                        help="Explicit 2-D keypoint initialization; no automatic semantic guessing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--front-calibration-profile", type=Path,
                        default=Path("configs/ur7e_xhand/front_calibration_candidate.json"))
    parser.add_argument("--sample-radius-px", type=float, default=14.0)
    parser.add_argument("--min-depth-samples", type=int, default=3)
    parser.add_argument("--max-prediction-frames", type=int, default=8)
    args = parser.parse_args()
    if args.episode < 0 or args.sample_radius_px <= 0 or args.min_depth_samples < 1 or args.max_prediction_frames < 0:
        parser.error("invalid episode or geometry parameters")
    run(args)


if __name__ == "__main__":
    main()
