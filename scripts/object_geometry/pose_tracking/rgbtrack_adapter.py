#!/usr/bin/env python3
"""Headless RGBTrack input, calibration, and output adapter.

This module deliberately keeps the RGBTrack import lazy.  The upstream
submodule imports CUDA, nvdiffrast, Open3D, PyTorch3D and model wrappers at
module import time, while preparation and the synthetic tests only need NumPy
and OpenCV.

Coordinate contract
-------------------
RGBTrack returns ``ob_in_cam``: a transform from the object mesh frame to the
front colour optical frame.  This adapter composes it with the active front
calibration as ``T_base_color @ T_color_object`` and writes both matrices.
All translations are metres and all poses are 4x4 homogeneous matrices.

The two upstream initialization modes remain explicit:

``depth``
    Uses RGBTrack ``register``/``track_one_new`` with real depth.
``no_depth``
    Uses ``register_without_depth``/``track_one_new_without_depth``.  The
    upstream implementation searches synthetic depths; it does not recover a
    metric scale from the scene.
``unknown_scale``
    Uses upstream ``binary_search_scale`` and therefore requires real depth.
    This is never selected implicitly.

No future frame is read by this module.  A missing/occluded mask produces a
causal short-term constant-velocity prediction marked invalid, rather than a
fabricated valid measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import sys
import time
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Sequence

import cv2
import numpy as np


VIZ_FRAMES = (120, 200, 300, 400, 440, 460)
OBJECT_ALIASES = {
    "funnel": ("funnel", "漏斗"),
    "target_flask": ("target_flask", "flask", "target-flask", "bottle"),
}
POSE_MODES = ("depth", "no_depth", "unknown_scale")


def _as_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix, got {transform.shape}")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    return transform


def _as_intrinsics(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"camera intrinsics must be a finite 3x3 matrix, got {matrix.shape}")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    return matrix


def pose_color_to_base(T_base_color: Any, T_color_object: Any) -> np.ndarray:
    """Compose an RGBTrack camera-frame pose into ``base_link``."""
    return _as_transform(T_base_color, "T_base_color") @ _as_transform(
        T_color_object, "T_color_object"
    )


def project_points(points_camera_m: Any, K: Any) -> np.ndarray:
    """Project camera-frame points; points behind the camera become NaN."""
    points = np.asarray(points_camera_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera_m must have shape [N,3], got {points.shape}")
    intrinsic = _as_intrinsics(K)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    uv[valid, 0] = intrinsic[0, 0] * points[valid, 0] / points[valid, 2] + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * points[valid, 1] / points[valid, 2] + intrinsic[1, 2]
    return uv


def transform_points(points: Any, transform: Any) -> np.ndarray:
    """Apply a homogeneous transform to an ``[N,3]`` array."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {xyz.shape}")
    matrix = _as_transform(transform, "transform")
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def _frame_stems(frame_index: int) -> tuple[str, ...]:
    return (
        f"{frame_index:06d}",
        f"{frame_index:05d}",
        f"frame_{frame_index:06d}",
        f"frame_{frame_index:05d}",
        str(frame_index),
    )


def resolve_mask_path(mask_root: str | Path, object_name: str, frame_index: int) -> Path:
    """Resolve a Level-1 SAM2 PNG without recursively scanning the mask tree."""
    root = Path(mask_root).expanduser()
    canonical = str(object_name).strip()
    aliases = OBJECT_ALIASES.get(canonical)
    if aliases is None:
        for key, candidates in OBJECT_ALIASES.items():
            if canonical in candidates:
                aliases = OBJECT_ALIASES[key]
                break
    if aliases is None:
        aliases = (canonical,)
    directories: list[Path] = []
    for alias in aliases:
        directories.extend((root / alias, root / "masks" / alias))
    directories.append(root)
    suffixes = (".png", ".PNG")
    for directory in directories:
        for stem in _frame_stems(int(frame_index)):
            for suffix in suffixes:
                candidate = directory / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
    searched = ", ".join(str(directory) for directory in directories)
    raise FileNotFoundError(
        f"No SAM2 mask for object={canonical!r}, frame={frame_index}; searched {searched}"
    )


def load_mask(mask_root: str | Path, object_name: str, frame_index: int, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    path = resolve_mask_path(mask_root, object_name, frame_index)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode mask PNG: {path}")
    if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"Mask shape mismatch for {path}: got {mask.shape}, expected {expected_shape}"
        )
    return mask > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on computer 4 env
        raise RuntimeError(
            "GLB conversion requires the RGBTrack/trimesh environment; "
            "install the existing project environment on computer 4, do not "
            "install it as part of this adapter."
        ) from exc
    return trimesh


def _mesh_visual_metadata(mesh: Any) -> dict[str, Any]:
    visual = getattr(mesh, "visual", None)
    kind = str(getattr(visual, "kind", "none"))
    vertex_colors = getattr(visual, "vertex_colors", None)
    has_vertex_colors = bool(vertex_colors is not None and np.asarray(vertex_colors).ndim == 2)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) if material is not None else None
    has_texture = image is not None
    return {
        "visual_kind": kind,
        "has_vertex_colors": has_vertex_colors,
        "has_texture": bool(has_texture),
        "texture_shape": list(np.asarray(image).shape) if has_texture else None,
    }


def _safe_scale_tag(scale_m_per_unit: float) -> str:
    return f"{scale_m_per_unit:.12g}".replace("-", "m").replace(".", "p")


def convert_glb_to_rgbtrack_mesh(
    source_glb: str | Path,
    output_root: str | Path,
    object_name: str,
    scale_m_per_unit: float,
) -> dict[str, Any]:
    """Convert a GLB into a co-located RGBTrack mesh and write provenance.

    ``scale_m_per_unit`` is mandatory because SAM3D GLB coordinates may be
    normalized.  No unit inference is attempted.  A new output directory is
    used for every source/scale pair, so existing prototype outputs remain
    untouched.
    """
    source = Path(source_glb).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    scale = float(scale_m_per_unit)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale_m_per_unit must be a finite positive number in metres per GLB unit")
    trimesh = _require_trimesh()

    scene = trimesh.load(str(source), force="scene", process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"GLB has no mesh vertices: {source}")
    mesh = mesh.copy()
    source_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    source_extents = np.asarray(mesh.extents, dtype=np.float64)
    visual = _mesh_visual_metadata(mesh)
    mesh.apply_scale(scale)

    root = Path(output_root).expanduser().resolve()
    run_dir = root / f"{object_name}__rgbtrack__{_sha256(source)[:12]}__scale_{_safe_scale_tag(scale)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_format = "obj" if visual["has_texture"] else "ply"
    mesh_path = run_dir / f"{object_name}.{output_format}"
    manifest_path = run_dir / "mesh_conversion.json"
    manifest = {
        "schema_version": 1,
        "object_name": str(object_name),
        "source_glb": str(source),
        "source_sha256": _sha256(source),
        "source_units": getattr(scene, "units", None),
        "source_bounds": source_bounds.tolist(),
        "source_extents": source_extents.tolist(),
        "scale_m_per_unit": scale,
        "output_mesh": str(mesh_path),
        "output_format": output_format,
        "scaled_bounds_m": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "scaled_extents_m": np.asarray(mesh.extents, dtype=np.float64).tolist(),
        "visual": visual,
        "conversion": "trimesh.load(force='scene', process=False), concatenate Scene, apply_scale(explicit), export",
        "scale_inference": "never; explicit CLI/manifest value required",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_sha256") != manifest["source_sha256"] or existing.get("scale_m_per_unit") != scale:
            raise FileExistsError(f"Refusing to overwrite incompatible mesh conversion: {manifest_path}")
    else:
        # Export only the derived mesh.  The source GLB is never modified.
        mesh.export(str(mesh_path), file_type=output_format)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class FrontCalibration:
    K_color: np.ndarray
    T_base_color: np.ndarray
    depth_scale_m_per_unit: float
    provenance: dict[str, Any]


def load_front_calibration(repo_root: str | Path, profile: str | Path | None) -> FrontCalibration:
    """Load the same profiled front camera used by SpatialPreprocessor."""
    from openpi.object_geometry.calibration import front_calibration_provenance
    from openpi.object_geometry.calibration import make_front_geometry_config
    from openpi.spatial.calibration import load_camera_calibrations
    from openpi.spatial.geometry import apply_diagnostic_overrides

    root = Path(repo_root).expanduser().resolve()
    config, resolved_profile = make_front_geometry_config(root, profile)
    bundle = load_camera_calibrations(
        repo_root=root,
        calibration_config=config.calibration,
        camera_roles=("front",),
    )
    cameras = apply_diagnostic_overrides(bundle.cameras, config.diagnostics)
    camera = cameras["front"]
    intrinsics = camera.color_intrinsics
    K = np.array(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return FrontCalibration(
        K_color=K,
        T_base_color=_as_transform(camera.T_base_color, "T_base_color"),
        depth_scale_m_per_unit=float(camera.depth_scale_m_per_unit),
        provenance=front_calibration_provenance(root, resolved_profile, config=config),
    )


@dataclass(frozen=True)
class EpisodeFrame:
    frame_index: int
    rgb: np.ndarray
    depth_raw: np.ndarray
    state: Any = None
    timestamp: float | None = None


def iter_episode_frames(
    dataset_root: str | Path,
    episode: int,
    *,
    selected_frames: Iterable[int] | None = None,
    camera_role: str = "front",
) -> Iterator[EpisodeFrame]:
    """Join the front RGB video and front depth rows by frame index."""
    from openpi.spatial_dataset.source import RawSpatialDataset

    selected = None if selected_frames is None else {int(value) for value in selected_frames}
    dataset = RawSpatialDataset(dataset_root, episode=int(episode))
    rows = dataset.iter_rows(
        selected=selected,
        depth_roles=(camera_role,),
    )
    frames = dataset.iter_video_frames(camera_role, selected=selected)
    rows_iter = iter(rows)
    frames_iter = iter(frames)
    while True:
        try:
            row = next(rows_iter)
        except StopIteration:
            try:
                extra = next(frames_iter)
            except StopIteration:
                return
            raise ValueError(f"RGB video has an extra frame {extra.frame_index} after parquet rows ended")
        try:
            video_frame = next(frames_iter)
        except StopIteration as exc:
            raise ValueError(f"Parquet has an extra frame {row['frame_index']} after RGB video ended") from exc
        row_index = int(row["frame_index"])
        if row_index != int(video_frame.frame_index):
            raise ValueError(
                f"RGB/depth frame-index mismatch: parquet={row_index}, video={video_frame.frame_index}"
            )
        yield EpisodeFrame(
            frame_index=row_index,
            rgb=np.asarray(video_frame.rgb),
            depth_raw=np.asarray(row[f"observation.depths.cam_{camera_role}"]),
            state=row.get("observation.state"),
            timestamp=float(row["timestamp"]) if row.get("timestamp") is not None else None,
        )


def _rgbtrack_commit(repo_root: Path) -> str | None:
    try:
        import subprocess

        value = subprocess.check_output(
            ["git", "-C", str(repo_root / "third_party" / "RGBTrack"), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _import_rgbtrack(repo_root: Path):  # pragma: no cover - unavailable on computer 1
    """Import RGBTrack only when actual CUDA inference was requested."""
    source = (repo_root / "third_party" / "RGBTrack").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"RGBTrack submodule is missing: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from estimater import FoundationPose
    except Exception as exc:
        raise RuntimeError(
            "RGBTrack import failed. Its upstream import chain requires the "
            "CUDA/nvdiffrast/Open3D/PyTorch3D environment (and imports the "
            "TensorRT model wrappers even when USE_TRT=False)."
        ) from exc
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("RGBTrack inference requires trimesh in the computer-4 environment") from exc
    return FoundationPose, trimesh


@dataclass
class PoseRecord:
    frame_index: int
    object_name: str
    T_color_object: np.ndarray
    T_base_object: np.ndarray
    valid: bool
    confidence: float
    status: str
    elapsed_ms: float
    mask_area_px: int
    mode: str
    scale_recovery: float | None = None

    def as_dict(self, calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result = {
            "frame_index": int(self.frame_index),
            "object_name": self.object_name,
            "T_color_object": np.asarray(self.T_color_object, dtype=np.float64).tolist(),
            "T_base_link_object": np.asarray(self.T_base_object, dtype=np.float64).tolist(),
            "valid": bool(self.valid),
            "confidence": float(self.confidence),
            "status": self.status,
            "elapsed_ms": float(self.elapsed_ms),
            "mask_area_px": int(self.mask_area_px),
            "mode": self.mode,
            "scale_recovery": self.scale_recovery,
        }
        if calibration is not None:
            result["calibration"] = dict(calibration)
        return result


@dataclass
class _CausalState:
    last_pose: np.ndarray | None = None
    last_frame: int | None = None
    velocity_m_per_frame: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def observe(self, pose: np.ndarray, frame_index: int) -> None:
        pose = _as_transform(pose, "pose")
        if self.last_pose is not None and self.last_frame is not None and frame_index > self.last_frame:
            self.velocity_m_per_frame = (pose[:3, 3] - self.last_pose[:3, 3]) / float(frame_index - self.last_frame)
        self.last_pose = pose.copy()
        self.last_frame = int(frame_index)

    def predict(self, frame_index: int) -> np.ndarray | None:
        if self.last_pose is None or self.last_frame is None or frame_index <= self.last_frame:
            return None
        pose = self.last_pose.copy()
        steps = float(frame_index - self.last_frame)
        # Keep the fallback deliberately short-horizon and causal.
        steps = min(steps, 15.0)
        pose[:3, 3] += self.velocity_m_per_frame * steps
        return pose


class RGBTrackObject:
    """One independently tracked object; no robot-hand motion is injected."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        object_name: str,
        mesh_path: str | Path,
        mode: str,
        debug_dir: str | Path,
        init_iterations: int = 5,
        track_iterations: int = 2,
        min_mask_area_px: int = 40,
    ) -> None:
        if mode not in POSE_MODES:
            raise ValueError(f"mode must be one of {POSE_MODES}, got {mode!r}")
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.object_name = object_name
        self.mesh_path = Path(mesh_path).expanduser().resolve()
        self.mode = mode
        self.debug_dir = Path(debug_dir).expanduser().resolve()
        self.init_iterations = int(init_iterations)
        self.track_iterations = int(track_iterations)
        self.min_mask_area_px = int(min_mask_area_px)
        self._estimator = None
        self._mesh = None
        self._initialized = False
        self._state = _CausalState()
        self._scale_recovery: float | None = None

    def _ensure_estimator(self) -> None:
        if self._estimator is not None:
            return
        FoundationPose, trimesh = _import_rgbtrack(self.repo_root)
        mesh = trimesh.load(str(self.mesh_path), force="mesh", process=False)
        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            raise ValueError(f"RGBTrack mesh has no vertices: {self.mesh_path}")
        self._mesh = mesh
        self._estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            debug=0,
            debug_dir=str(self.debug_dir / self.object_name),
        )

    @property
    def mesh_vertices(self) -> np.ndarray | None:
        if self._mesh is None:
            return None
        return np.asarray(self._mesh.vertices, dtype=np.float64)

    def _prediction_record(self, frame_index: int, mask_area: int, start: float, T_base_color: np.ndarray) -> PoseRecord:
        predicted = self._state.predict(frame_index)
        elapsed = (time.perf_counter() - start) * 1000.0
        if predicted is None:
            predicted = np.eye(4, dtype=np.float64)
            status = "invalid_no_observation"
        else:
            status = "occluded_causal_prediction"
        return PoseRecord(
            frame_index=int(frame_index),
            object_name=self.object_name,
            T_color_object=predicted,
            T_base_object=pose_color_to_base(T_base_color, predicted),
            valid=False,
            confidence=0.0,
            status=status,
            elapsed_ms=elapsed,
            mask_area_px=int(mask_area),
            mode=self.mode,
            scale_recovery=self._scale_recovery,
        )

    def estimate(
        self,
        *,
        frame_index: int,
        rgb: np.ndarray,
        depth_m: np.ndarray | None,
        mask: np.ndarray,
        K: np.ndarray,
        T_base_color: np.ndarray,
    ) -> PoseRecord:
        start = time.perf_counter()
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim != 2 or rgb.shape[:2] != mask_bool.shape:
            raise ValueError(f"RGB/mask shape mismatch: rgb={rgb.shape}, mask={mask_bool.shape}")
        mask_area = int(mask_bool.sum())
        if mask_area < self.min_mask_area_px:
            return self._prediction_record(frame_index, mask_area, start, T_base_color)
        if self.mode != "no_depth" and depth_m is None:
            raise ValueError(f"mode={self.mode} requires depth")
        self._ensure_estimator()
        assert self._estimator is not None
        try:
            if not self._initialized:
                if self.mode == "no_depth":
                    pose = self._estimator.register_without_depth(
                        K=K, rgb=rgb, ob_mask=mask_bool, iteration=self.init_iterations
                    )
                elif self.mode == "unknown_scale":
                    # Import only this optional path; it is intentionally not
                    # used by the no-depth mode.
                    source = str((self.repo_root / "third_party" / "RGBTrack").resolve())
                    if source not in sys.path:
                        sys.path.insert(0, source)
                    from tools import binary_search_scale  # type: ignore

                    pose, self._scale_recovery = binary_search_scale(
                        self._estimator,
                        self._mesh,
                        rgb,
                        depth_m,
                        mask_bool,
                        K,
                        debug=False,
                    )
                    if self._scale_recovery is not None and self._mesh is not None:
                        self._mesh = self._mesh.copy()
                        self._mesh.apply_scale(float(self._scale_recovery))
                else:
                    pose = self._estimator.register(
                        K=K, rgb=rgb, depth=depth_m, ob_mask=mask_bool, iteration=self.init_iterations
                    )
                self._initialized = True
            elif self.mode == "no_depth":
                pose = self._estimator.track_one_new_without_depth(
                    rgb=rgb, K=K, iteration=self.track_iterations, mask=mask_bool
                )
            else:
                pose = self._estimator.track_one_new(
                    rgb=rgb, depth=depth_m, K=K, iteration=self.track_iterations, mask=mask_bool
                )
            pose = _as_transform(pose, "RGBTrack pose")
        except Exception as exc:
            # Preserve the causal state but never turn a failed model call into
            # a valid pose.  The exception is re-raised with context so a real
            # computer-4 run cannot silently produce a partial result.
            raise RuntimeError(f"RGBTrack failed for {self.object_name} frame {frame_index}") from exc

        track_good = bool(getattr(self._estimator, "track_good", True))
        self._state.observe(pose, int(frame_index))
        confidence = min(1.0, max(0.0, mask_area / 5000.0))
        if not track_good:
            confidence *= 0.5
        elapsed = (time.perf_counter() - start) * 1000.0
        return PoseRecord(
            frame_index=int(frame_index),
            object_name=self.object_name,
            T_color_object=pose,
            T_base_object=pose_color_to_base(T_base_color, pose),
            valid=bool(np.all(np.isfinite(pose)) and track_good),
            confidence=float(confidence),
            status="observed" if track_good else "observed_low_quality",
            elapsed_ms=elapsed,
            mask_area_px=mask_area,
            mode=self.mode,
            scale_recovery=self._scale_recovery,
        )


def _bbox_corners(vertices: np.ndarray) -> np.ndarray:
    low = np.min(vertices, axis=0)
    high = np.max(vertices, axis=0)
    return np.array(
        [
            [low[0], low[1], low[2]], [high[0], low[1], low[2]],
            [high[0], high[1], low[2]], [low[0], high[1], low[2]],
            [low[0], low[1], high[2]], [high[0], low[1], high[2]],
            [high[0], high[1], high[2]], [low[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )


_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def draw_pose_overlay(
    rgb: np.ndarray,
    records: Sequence[PoseRecord],
    object_vertices: Mapping[str, np.ndarray],
    K: np.ndarray,
) -> np.ndarray:
    """Draw projected 3-D AABBs and axes without any GUI calls."""
    image = np.asarray(rgb).copy()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    colors = ((0, 220, 0), (0, 140, 255), (255, 120, 0), (255, 0, 255))
    for index, record in enumerate(records):
        if not record.valid or record.object_name not in object_vertices:
            continue
        corners = _bbox_corners(np.asarray(object_vertices[record.object_name], dtype=np.float64))
        uv = project_points(transform_points(corners, record.T_color_object), K)
        color = colors[index % len(colors)]
        for a, b in _BOX_EDGES:
            if np.isfinite(uv[[a, b]]).all():
                cv2.line(image, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), color, 2)
        origin = np.zeros((1, 3), dtype=np.float64)
        axes = np.vstack((origin, np.eye(3, dtype=np.float64) * max(np.ptp(corners, axis=0).max() * 0.35, 0.01)))
        axis_uv = project_points(transform_points(axes, record.T_color_object), K)
        if np.isfinite(axis_uv).all():
            for axis, axis_color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), start=1):
                cv2.line(image, tuple(np.round(axis_uv[0]).astype(int)), tuple(np.round(axis_uv[axis]).astype(int)), axis_color, 2)
        label = f"{record.object_name} {record.status} c={record.confidence:.2f}"
        cv2.putText(image, label, (10, 24 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return image


def write_bbox_ply(path: str | Path, vertices: np.ndarray, pose_base_object: np.ndarray) -> None:
    """Write an ASCII PLY containing transformed object AABB vertices/edges."""
    corners = transform_points(_bbox_corners(np.asarray(vertices, dtype=np.float64)), pose_base_object)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(corners)}", "property float x", "property float y", "property float z",
        f"element edge {len(_BOX_EDGES)}", "property int vertex1", "property int vertex2", "end_header",
    ]
    lines.extend("{:.9f} {:.9f} {:.9f}".format(*point) for point in corners)
    lines.extend(f"{a} {b}" for a, b in _BOX_EDGES)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preview_html(path: str | Path, records: Sequence[PoseRecord], mesh_manifests: Mapping[str, Mapping[str, Any]]) -> None:
    """Write a dependency-free inspection page with pose/bbox provenance."""
    payload = {
        "records": [record.as_dict() for record in records],
        "meshes": {name: dict(manifest) for name, manifest in mesh_manifests.items()},
        "note": "Use the converted OBJ/PLY in a 3D viewer and the per-frame PLY files for base_link bbox inspection.",
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    output.write_text(
        "<!doctype html><meta charset='utf-8'><title>RGBTrack pose preview</title>"
        "<h1>RGBTrack pose preview</h1><p>3D mesh paths and base_link bbox records are embedded below.</p>"
        "<pre id='data'></pre><script>const data=" + text + ";document.getElementById('data').textContent=JSON.stringify(data,null,2);</script>\n",
        encoding="utf-8",
    )


def _load_scale_manifest(path: Path, objects: Sequence[str]) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported scale manifest schema: {path}")
    values = raw.get("objects")
    if not isinstance(values, dict):
        raise ValueError("scale manifest must contain objects mapping")
    result: dict[str, float] = {}
    for object_name in objects:
        value = values.get(object_name)
        if value is None:
            raise ValueError(f"scale manifest has no explicit scale for {object_name}")
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid explicit scale for {object_name}: {value!r}")
        result[object_name] = scale
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--funnel-glb", type=Path, required=True)
    parser.add_argument("--flask-glb", type=Path, required=True)
    parser.add_argument("--scale-manifest", type=Path, required=True, help="JSON with explicit metres-per-GLB-unit values")
    parser.add_argument("--front-calibration-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", choices=("funnel", "target_flask"), default=("funnel",))
    parser.add_argument("--mode", choices=POSE_MODES, default="depth")
    parser.add_argument("--camera-role", default="front")
    parser.add_argument("--viz-frames", default=",".join(str(value) for value in VIZ_FRAMES))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true", help="Convert meshes and calibration only; do not import RGBTrack")
    return parser


def run(args: argparse.Namespace) -> int:  # pragma: no cover - exercised on computer 4
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "rgb_overlays").mkdir(parents=True, exist_ok=True)
    objects = tuple(dict.fromkeys(args.objects))
    mesh_sources = {"funnel": args.funnel_glb, "target_flask": args.flask_glb}
    scales = _load_scale_manifest(args.scale_manifest.expanduser().resolve(), objects)
    calibration = load_front_calibration(repo_root, args.front_calibration_profile)

    mesh_manifests = {
        name: convert_glb_to_rgbtrack_mesh(mesh_sources[name], output / "meshes", name, scales[name])
        for name in objects
    }
    manifest = {
        "schema_version": 1,
        "episode": int(args.episode),
        "objects": list(objects),
        "mode": args.mode,
        "dataset": str(args.dataset.expanduser().resolve()),
        "mask_root": str(args.mask_root.expanduser().resolve()),
        "calibration": calibration.provenance,
        "rgbtrack_commit": _rgbtrack_commit(repo_root),
        "scale_manifest": str(args.scale_manifest.expanduser().resolve()),
        "mesh_conversions": mesh_manifests,
        "causal_online": True,
        "future_frames_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.prepare_only:
        return 0

    trackers = {
        name: RGBTrackObject(
            repo_root=repo_root,
            object_name=name,
            mesh_path=mesh_manifests[name]["output_mesh"],
            mode=args.mode,
            debug_dir=output / "rgbtrack_debug",
        )
        for name in objects
    }
    viz_frames = {int(value.strip()) for value in str(args.viz_frames).split(",") if value.strip()}
    vertices: dict[str, np.ndarray] = {}
    for name, tracker in trackers.items():
        # Loading a mesh for bbox visualization is lazy and does not alter the
        # source GLB.  RGBTrack has already loaded it on first estimate.
        tracker._ensure_estimator()
        vertices[name] = np.asarray(tracker.mesh_vertices)

    all_records: list[PoseRecord] = []
    per_object_lines = {name: (output / f"{name}_poses.jsonl").open("w", encoding="utf-8") for name in objects}
    try:
        frame_count = 0
        for frame in iter_episode_frames(args.dataset, args.episode, camera_role=args.camera_role):
            if args.max_frames is not None and frame_count >= int(args.max_frames):
                break
            frame_count += 1
            if frame.depth_raw.ndim != 2 or tuple(frame.depth_raw.shape) != tuple(frame.rgb.shape[:2]):
                raise ValueError(
                    f"front RGB/depth resolution mismatch at frame {frame.frame_index}: "
                    f"rgb={frame.rgb.shape[:2]}, depth={frame.depth_raw.shape}"
                )
            depth_m = frame.depth_raw.astype(np.float32) * calibration.depth_scale_m_per_unit
            records: list[PoseRecord] = []
            for name in objects:
                mask = load_mask(args.mask_root, name, frame.frame_index, expected_shape=frame.rgb.shape[:2])
                record = trackers[name].estimate(
                    frame_index=frame.frame_index,
                    rgb=frame.rgb,
                    depth_m=depth_m if args.mode != "no_depth" else None,
                    mask=mask,
                    K=calibration.K_color,
                    T_base_color=calibration.T_base_color,
                )
                records.append(record)
                all_records.append(record)
                per_object_lines[name].write(json.dumps(record.as_dict(calibration.provenance), ensure_ascii=False) + "\n")
                if frame.frame_index in viz_frames and record.valid:
                    write_bbox_ply(output / "bbox_base_link" / f"f{frame.frame_index:06d}_{name}.ply", vertices[name], record.T_base_object)
            if frame.frame_index in viz_frames:
                overlay = draw_pose_overlay(frame.rgb, records, vertices, calibration.K_color)
                cv2.imwrite(str(output / "rgb_overlays" / f"f{frame.frame_index:06d}.png"), overlay[..., ::-1])
    finally:
        for handle in per_object_lines.values():
            handle.close()
    write_preview_html(output / "preview.html", all_records, mesh_manifests)
    timing_summary: dict[str, Any] = {}
    for name in objects:
        values = np.asarray([record.elapsed_ms for record in all_records if record.object_name == name], dtype=np.float64)
        timing_summary[name] = {
            "frames": int(values.size),
            "mean_ms": float(values.mean()) if values.size else None,
            "p95_ms": float(np.percentile(values, 95)) if values.size else None,
            "max_ms": float(values.max()) if values.size else None,
        }
    (output / "timing_summary.json").write_text(json.dumps(timing_summary, indent=2) + "\n", encoding="utf-8")
    report = {
        name: {
            "frames": sum(record.object_name == name for record in all_records),
            "valid": sum(record.object_name == name and record.valid for record in all_records),
            "invalid": sum(record.object_name == name and not record.valid for record in all_records),
            "statuses": {
                status: sum(record.object_name == name and record.status == status for record in all_records)
                for status in sorted({record.status for record in all_records if record.object_name == name})
            },
        }
        for name in objects
    }
    (output / "tracking_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest["frames_written"] = frame_count
    manifest["visualization_frames"] = sorted(viz_frames)
    manifest["timing_summary"] = timing_summary
    manifest["tracking_report"] = report
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main() -> int:  # pragma: no cover - CLI wrapper
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
