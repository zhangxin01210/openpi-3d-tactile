"""Mesh + causal 6-D pose production for ``v1_hand_scene``.

This module is intentionally small and independent of RGBTrack/SAM2.  Those
systems produce reviewed pose JSONL and masks; this boundary turns that
reviewed output into the fixed-size training sidecar.  The same deterministic
mesh sampling and transform functions can therefore be used by training and
deployment without importing a GPU tracker.

The producer is deliberately conservative:

* scale is an explicit metres-per-mesh-unit value;
* an invalid dynamic pose produces an invalid zero cloud, never a copied last
  pose;
* a static flask is accepted only when its pose run declares the reviewed
  static-anchor prior;
* tactile and mask features are invalid when their source is absent; no label
  or confidence is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from openpi.object_geometry_dataset.derived import FUNNEL_MASK_SIZE
from openpi.object_geometry_dataset.derived import HAND_SCENE_FIELDS
from openpi.object_geometry_dataset.derived import POINT_COUNT
from openpi.object_geometry_dataset.derived import TACTILE_POINT_COUNT


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_transform(value: Any, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix, got {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must have homogeneous bottom row [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) < 0.98:
        raise ValueError(f"{name} has a non-rigid rotation")
    return matrix


def transform_points_local_to_base(points_local_m: Any, T_base_object: Any) -> np.ndarray:
    points = np.asarray(points_local_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_local_m must have shape [N,3], got {points.shape}")
    matrix = as_transform(T_base_object, "T_base_link_object")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def deterministic_surface_points(
    vertices_m: Any,
    faces: Any,
    *,
    count: int = POINT_COUNT,
    seed: int = 0,
) -> np.ndarray:
    """Sample a fixed number of surface points without global RNG state.

    Triangle selection is area weighted and barycentric coordinates are drawn
    from a local, seeded generator.  The output is stable for one mesh asset,
    face ordering and ``seed``; callers record the asset hash and sampler
    version in the sidecar manifest.
    """

    vertices = np.asarray(vertices_m, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"vertices must have shape [N,3], got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError(f"faces must have shape [M,3], got {triangles.shape}")
    if not np.isfinite(vertices).all() or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("mesh contains invalid vertices or face indices")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    usable = np.isfinite(areas) & (areas > 1e-14)
    if not np.any(usable):
        # Degenerate synthetic meshes are useful for contract tests.  They do
        # not pretend to be a surface: use a deterministic vertex cycle.
        ids = np.arange(count, dtype=np.int64) % len(vertices)
        return vertices[ids].astype(np.float32)
    triangles = triangles[usable]
    areas = areas[usable]
    probabilities = np.cumsum(areas / np.sum(areas))
    rng = np.random.default_rng(int(seed))
    choices = np.searchsorted(probabilities, rng.random(count), side="right")
    choices = np.minimum(choices, len(triangles) - 1)
    u = rng.random(count)
    v = rng.random(count)
    root_u = np.sqrt(u)
    w0 = 1.0 - root_u
    w1 = root_u * (1.0 - v)
    w2 = root_u * v
    sampled = (
        w0[:, None] * vertices[triangles[choices, 0]]
        + w1[:, None] * vertices[triangles[choices, 1]]
        + w2[:, None] * vertices[triangles[choices, 2]]
    )
    return sampled.astype(np.float32)


def load_mesh_surface_points(
    mesh_path: str | Path,
    *,
    scale_m_per_unit: float,
    count: int = POINT_COUNT,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load GLB/mesh using the existing computer-4 ``trimesh`` dependency."""

    scale = float(scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale_m_per_unit must be a finite positive value")
    source = Path(mesh_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        import trimesh  # type: ignore
    except ImportError as exc:  # pragma: no cover - computer-4 dependency
        raise RuntimeError(
            "Mesh sampling requires the existing computer-4 trimesh dependency; "
            "do not substitute a normalized SAM3D scale."
        ) from exc
    scene = trimesh.load(str(source), force="scene", process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError(f"Mesh has no triangular surface: {source}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    faces = np.asarray(mesh.faces, dtype=np.int64)
    points = deterministic_surface_points(vertices, faces, count=count, seed=seed)
    return points, {
        "source_mesh": str(source),
        "source_sha256": sha256_file(source),
        "scale_m_per_unit": scale,
        "sampler": "deterministic_surface_points.v1",
        "seed": int(seed),
        "point_count": int(count),
        "source_vertex_count": int(len(vertices)),
        "source_face_count": int(len(faces)),
        "scaled_bounds_m": (np.asarray(mesh.bounds, dtype=np.float64) * scale).tolist()
        if np.asarray(mesh.bounds).shape == (2, 3)
        else None,
    }


@dataclass(frozen=True)
class PoseObservation:
    frame_index: int
    T_base_link_object: np.ndarray
    valid: bool
    confidence: float
    status: str
    pose_source: str
    observed_pose_valid: bool
    raw_heuristic_score: float | None


def _pose_matrix(row: Mapping[str, Any]) -> np.ndarray:
    for name in ("T_base_link_object", "T_base_object"):
        if name in row:
            return as_transform(row[name], name)
    raise ValueError(f"Pose row {row.get('frame_index')} has no T_base_link_object")


def read_pose_jsonl(path: str | Path, *, object_name: str) -> list[PoseObservation]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty pose stream: {source}")
    result: list[PoseObservation] = []
    previous = -1
    for row in rows:
        frame = int(row["frame_index"])
        if frame <= previous:
            raise ValueError(f"Non-increasing frame index in {source}: {frame}")
        previous = frame
        matrix = _pose_matrix(row)
        confidence = float(row.get("confidence", 0.0))
        if not np.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Invalid pose confidence at {source}:{frame}: {confidence}")
        valid = bool(row.get("valid", False))
        raw_score_value = row.get("raw_heuristic_score", row.get("heuristic_score", row.get("confidence")))
        raw_score = None if raw_score_value is None else float(raw_score_value)
        if raw_score is not None and not np.isfinite(raw_score):
            raw_score = None
        if not valid:
            # Invalid dynamic observations are not allowed to carry a useful
            # cloud.  Keep the matrix only for audit/debugging.
            confidence = 0.0
        result.append(
            PoseObservation(
                frame_index=frame,
                T_base_link_object=matrix,
                valid=valid,
                confidence=confidence,
                status=str(row.get("status", "unknown")),
                pose_source=str(row.get("pose_source", "unknown")),
                observed_pose_valid=bool(row.get("observed_pose_valid", valid)),
                raw_heuristic_score=raw_score,
            )
        )
    if [row.frame_index for row in result] != list(range(result[-1].frame_index + 1)):
        raise ValueError(f"Pose stream must cover contiguous frame indices from zero: {source}")
    return result


def read_pose_run(run_root: str | Path, *, episode: int) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if int(manifest.get("episode", -1)) != int(episode):
        raise ValueError(f"Pose run episode mismatch: expected {episode}, got {manifest.get('episode')}")
    if bool(manifest.get("future_frames_used", False)):
        raise ValueError(f"Pose run explicitly used future frames: {root}")
    funnel = read_pose_jsonl(root / "funnel_poses.jsonl", object_name="funnel")
    flask = read_pose_jsonl(root / "target_flask_poses.jsonl", object_name="target_flask")
    if [x.frame_index for x in funnel] != [x.frame_index for x in flask]:
        raise ValueError("Funnel and target flask pose streams have different frames")
    return {"root": root, "manifest": manifest, "funnel": funnel, "target_flask": flask}


def validate_static_flask_prior(observations: Iterable[PoseObservation]) -> None:
    rows = list(observations)
    if not rows or not rows[0].valid:
        raise ValueError("Static flask prior requires a valid frame-0 anchor")
    anchor = rows[0].T_base_link_object
    for row in rows:
        if not row.valid:
            raise ValueError(f"Static flask prior has invalid frame {row.frame_index}")
        if not np.allclose(row.T_base_link_object, anchor, atol=1e-6):
            raise ValueError(f"Static flask prior drifts at frame {row.frame_index}")
    if any(row.frame_index > 0 and "static" not in row.status.lower() and "prior" not in row.pose_source.lower() for row in rows):
        raise ValueError("Static flask stream must explicitly tag frames after frame 0 as a prior")


def _mask_path(mask_root: Path, object_name: str, frame_index: int) -> Path | None:
    aliases = {
        "funnel": ("funnel",),
        "target_flask": ("target_flask", "flask", "target-flask", "bottle"),
    }.get(object_name, (object_name,))
    stems = (f"{frame_index:06d}", f"{frame_index:05d}", str(frame_index))
    for alias in aliases:
        for directory in (mask_root / alias, mask_root / "masks" / alias, mask_root):
            for stem in stems:
                for suffix in (".png", ".PNG"):
                    path = directory / f"{stem}{suffix}"
                    if path.is_file():
                        return path
    return None


def load_downsampled_mask(
    mask_root: str | Path | None,
    *,
    object_name: str,
    frame_index: int,
) -> np.ndarray | None:
    if mask_root is None:
        return None
    path = _mask_path(Path(mask_root).expanduser().resolve(), object_name, frame_index)
    if path is None:
        return None
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenPI already depends on OpenCV
        raise RuntimeError("Mask downsampling requires OpenCV") from exc
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode mask: {path}")
    resized = cv2.resize(image, (FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _load_tactile_force(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as raw:
        candidate = None
        for name in ("force_norm", "tactile_force_norm", "force", "tactile_force"):
            if name in raw.files:
                candidate = np.asarray(raw[name], dtype=np.float32)
                break
        if candidate is None:
            return None
    if candidate.shape == (TACTILE_POINT_COUNT,):
        values = candidate
    elif candidate.shape[0] == TACTILE_POINT_COUNT:
        values = np.linalg.norm(candidate.reshape(TACTILE_POINT_COUNT, -1), axis=1)
    elif candidate.size == TACTILE_POINT_COUNT:
        values = candidate.reshape(TACTILE_POINT_COUNT)
    else:
        raise ValueError(f"Tactile force source must contain 600 taxels: {path} has {candidate.shape}")
    return values.astype(np.float32)


class CausalHandSceneState:
    """Generate only states supported by current-frame/previous-frame inputs."""

    def __init__(
        self,
        *,
        tactile_root: str | Path | None = None,
        spatial_tactile: "SpatialTactileProvider | None" = None,
        mask_root: str | Path | None = None,
        contact_force_threshold: float = 0.05,
        hand_offset_root: str | Path | None = None,
    ) -> None:
        self.tactile_root = None if tactile_root is None else Path(tactile_root).expanduser().resolve()
        self.spatial_tactile = spatial_tactile
        if self.tactile_root is not None and spatial_tactile is not None:
            raise ValueError("Choose either tactile_root or spatial_tactile, not both")
        self.mask_root = None if mask_root is None else Path(mask_root).expanduser().resolve()
        self.hand_offset_root = (
            None if hand_offset_root is None else Path(hand_offset_root).expanduser().resolve()
        )
        self.contact_force_threshold = float(contact_force_threshold)
        if not np.isfinite(self.contact_force_threshold) or self.contact_force_threshold < 0:
            raise ValueError("contact_force_threshold must be finite and non-negative")
        self._reference: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._previous_frame: int | None = None
        self.tactile_frames_total = 0
        self.tactile_frames_valid = 0
        self.tactile_missing_frames = 0
        self.tactile_discontinuous_frames = 0

    def reset(self) -> None:
        """Reset all per-episode causal state."""
        self._reference = None
        self._previous = None
        self._previous_frame = None
        self.tactile_frames_total = 0
        self.tactile_frames_valid = 0
        self.tactile_missing_frames = 0
        self.tactile_discontinuous_frames = 0

    def _tactile_path(self, episode: int, frame: int) -> Path | None:
        if self.tactile_root is None:
            return None
        candidates = (
            self.tactile_root / f"e{episode:03d}" / f"f{frame:06d}.npz",
            self.tactile_root / f"e{episode:03d}" / f"{frame:06d}.npz",
            self.tactile_root / f"e{episode:03d}" / f"{frame:05d}.npz",
        )
        return next((path for path in candidates if path.is_file()), None)

    def _offset(self, episode: int, frame: int) -> np.ndarray | None:
        if self.hand_offset_root is None:
            return None
        path = self.hand_offset_root / f"e{episode:03d}" / f"f{frame:06d}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as raw:
            if "funnel_mask_hand_offset_xy" not in raw.files:
                return None
            value = np.asarray(raw["funnel_mask_hand_offset_xy"], dtype=np.float32)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid funnel hand offset: {path}")
        return value

    def next(self, *, episode: int, frame: int) -> dict[str, np.ndarray]:
        if self.spatial_tactile is not None:
            force = self.spatial_tactile.get_force_norm(episode, frame)
        else:
            tactile_path = self._tactile_path(episode, frame)
            force = _load_tactile_force(tactile_path) if tactile_path else None
        return self.next_from_arrays(
            frame=frame,
            force_norm=force,
            funnel_mask=load_downsampled_mask(self.mask_root, object_name="funnel", frame_index=frame),
            funnel_mask_hand_offset_xy=self._offset(episode, frame),
        )

    def next_from_arrays(
        self, *, frame: int, force_norm: np.ndarray | None,
        funnel_mask: np.ndarray | None = None,
        funnel_mask_hand_offset_xy: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the identical causal state machine for a live or offline frame."""
        self.tactile_frames_total += 1
        force = force_norm
        frame_contiguous = self._previous_frame is None or int(frame) == self._previous_frame + 1
        tactile_valid = force is not None and np.asarray(force).shape == (TACTILE_POINT_COUNT,) and np.isfinite(force).all()
        if not frame_contiguous:
            self.tactile_discontinuous_frames += 1
            tactile_valid = False
        if not tactile_valid:
            current = np.zeros((TACTILE_POINT_COUNT,), dtype=np.float32)
            contact = np.zeros((TACTILE_POINT_COUNT,), dtype=bool)
            reference = np.zeros((TACTILE_POINT_COUNT,), dtype=np.float32) if self._reference is None else self._reference.copy()
            delta = np.zeros((TACTILE_POINT_COUNT,), dtype=np.float32)
            quality = np.asarray(0.0, dtype=np.float32)
            self._previous = None
            self.tactile_missing_frames += 1
        else:
            current = np.maximum(force.astype(np.float32), 0.0)
            if self._reference is None:
                self._reference = current.copy()
            reference = self._reference.copy()
            delta = np.zeros_like(current) if self._previous is None else current - self._previous
            contact = current >= self.contact_force_threshold
            # This field is a grasp-quality descriptor, not a data-integrity
            # score.  Force availability is reported separately by the batch
            # QA; no grasp stability label is inferred from finite values.
            quality = np.asarray(0.0, dtype=np.float32)
            self._previous = current.copy()
            self.tactile_frames_valid += 1
        self._previous_frame = int(frame)
        mask = None if funnel_mask is None else np.asarray(funnel_mask, dtype=bool)
        offset = None if funnel_mask_hand_offset_xy is None else np.asarray(funnel_mask_hand_offset_xy, dtype=np.float32)
        if mask is not None and mask.ndim != 2:
            raise ValueError("funnel_mask must be 2D")
        if offset is not None and (offset.shape != (2,) or not np.isfinite(offset).all()):
            raise ValueError("hand offset must be finite [2]")
        mask_valid = mask is not None and offset is not None
        if mask_valid:
            if mask.shape != (FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE):
                import cv2
                mask_value = cv2.resize(mask.astype(np.uint8), (FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask_value = mask
            mask_quality = np.asarray(1.0, dtype=np.float32)
            offset_value = offset
        else:
            mask_value = np.zeros((FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE), dtype=bool)
            mask_quality = np.asarray(0.0, dtype=np.float32)
            offset_value = np.zeros((2,), dtype=np.float32)
        return {
            "manipulated_relation_valid": np.asarray(False, dtype=bool),
            "target_relation_valid": np.asarray(False, dtype=bool),
            "manipulated_geometry_quality": np.asarray(0.0, dtype=np.float32),
            "target_geometry_quality": np.asarray(0.0, dtype=np.float32),
            "grasp_reference_force_norm": reference,
            "grasp_short_delta_force_norm": delta.astype(np.float32),
            "grasp_contact_valid": contact,
            "grasp_quality": quality,
            "funnel_mask": mask_value,
            "funnel_mask_valid": np.asarray(mask_valid, dtype=bool),
            "funnel_mask_quality": mask_quality,
            "funnel_mask_hand_offset_xy": offset_value,
        }


class SpatialTactileProvider:
    """Read the existing 600-taxel Spatial sidecar by episode/frame key.

    It deliberately uses ``SpatialDerivedDataset`` rather than copying its
    arrays into a second tactile cache.  Missing frame keys and non-finite
    values are returned as invalid observations to the causal state machine.
    """

    def __init__(self, dataset_root: str | Path, *, version: str = "v1_front_candidate") -> None:
        from openpi.spatial_dataset.derived import SpatialDerivedDataset

        self.dataset = SpatialDerivedDataset(dataset_root, version=version)
        if self.dataset.tactile_points_per_frame != TACTILE_POINT_COUNT:
            raise ValueError(
                f"Spatial sidecar tactile count is {self.dataset.tactile_points_per_frame}, expected {TACTILE_POINT_COUNT}"
            )
        self.version = str(version)
        self._frame_keys: dict[int, set[int]] = {}
        # The existing reader intentionally keeps shard arrays lazy.  Indexing
        # frame IDs here only touches the small frame_index arrays, not RGB-D
        # or tactile feature payloads.
        for episode in self.dataset.episode_ids:
            keys: set[int] = set()
            for record in self.dataset._episode_shards[int(episode)]:  # noqa: SLF001 - provider boundary
                arrays = self.dataset._load_shard_arrays(record)  # noqa: SLF001
                keys.update(int(value) for value in arrays["frame_index"])
            self._frame_keys[int(episode)] = keys

    def frame_indices(self, episode: int) -> set[int]:
        return set(self._frame_keys.get(int(episode), set()))

    def get_force_norm(self, episode: int, frame: int) -> np.ndarray | None:
        if int(frame) not in self._frame_keys.get(int(episode), set()):
            return None
        try:
            sample = self.dataset.get(episode_index=int(episode), frame_index=int(frame))
        except (KeyError, ValueError):
            return None
        force = np.asarray(sample.tactile_force_norm, dtype=np.float32)
        if force.shape != (TACTILE_POINT_COUNT,) or not np.isfinite(force).all():
            return None
        return force


def _geometry_record(
    points_local: np.ndarray,
    observation: PoseObservation,
    *,
    source: int,
) -> dict[str, np.ndarray]:
    xyz = np.zeros((POINT_COUNT, 3), dtype=np.float32)
    valid = bool(observation.valid)
    if valid:
        transformed = transform_points_local_to_base(points_local, observation.T_base_link_object)
        if not np.isfinite(transformed).all():
            valid = False
        else:
            xyz[:] = transformed.astype(np.float32)
    mask = np.full((POINT_COUNT,), valid, dtype=bool)
    # Per-point confidence is structural validity only.  RGBTrack's mask-area
    # heuristic is retained in pose provenance/QA and is never exposed as a
    # calibrated geometric accuracy score.
    confidence = np.full((POINT_COUNT,), 1.0 if valid else 0.0, dtype=np.float32)
    return {
        "xyz_m": xyz,
        "mask": mask,
        "source": np.full((POINT_COUNT,), int(source if valid else 0), dtype=np.int8),
        "age": np.zeros((POINT_COUNT,), dtype=np.int16),
        "confidence": confidence,
    }


def build_episode_records(
    *,
    episode: int,
    pose_run: Mapping[str, Any],
    funnel_points: np.ndarray,
    flask_points: np.ndarray,
    flask_motion: str = "static",
    state: CausalHandSceneState | None = None,
) -> list[dict[str, np.ndarray]]:
    funnel = list(pose_run["funnel"])
    flask = list(pose_run["target_flask"])
    if flask_motion == "static":
        validate_static_flask_prior(flask)
    if len(funnel) != len(flask):
        raise ValueError("funnel and target pose streams have different lengths")
    state = state or CausalHandSceneState()
    records: list[dict[str, np.ndarray]] = []
    for funnel_obs, flask_obs in zip(funnel, flask, strict=True):
        if funnel_obs.frame_index != flask_obs.frame_index:
            raise ValueError("funnel and target frame indices differ")
        frame = funnel_obs.frame_index
        causal = state.next(episode=episode, frame=frame)
        funnel_geo = _geometry_record(funnel_points, funnel_obs, source=0)
        flask_source = 1 if flask_motion == "static" and flask_obs.pose_source == "static_geometry_prior" else 0
        flask_geo = _geometry_record(flask_points, flask_obs, source=flask_source)
        causal.update(
            manipulated_relation_valid=np.asarray(bool(funnel_obs.valid), dtype=bool),
            target_relation_valid=np.asarray(bool(flask_obs.valid), dtype=bool),
            # Geometry quality means that a finite approved asset was
            # transformed by a tracker state accepted by the pose pipeline.
            # It is not a probability of metric pose accuracy.
            manipulated_geometry_quality=np.asarray(1.0 if funnel_obs.valid else 0.0, dtype=np.float32),
            target_geometry_quality=np.asarray(1.0 if flask_obs.valid else 0.0, dtype=np.float32),
        )
        records.append(
            {
                "episode_index": np.asarray(episode, dtype=np.int64),
                "frame_index": np.asarray(frame, dtype=np.int64),
                **{f"manipulated_{key}": value for key, value in funnel_geo.items()},
                **{f"target_{key}": value for key, value in flask_geo.items()},
                **causal,
            }
        )
    return records


def sidecar_fields() -> tuple[str, ...]:
    return (
        "episode_index",
        "frame_index",
        *(field for name in ("manipulated", "target") for field in (
            f"{name}_xyz_m", f"{name}_mask", f"{name}_source", f"{name}_age", f"{name}_confidence"
        )),
        *HAND_SCENE_FIELDS,
    )


def write_episode_sidecar(
    *,
    dataset_root: str | Path,
    episode: int,
    records: list[dict[str, np.ndarray]],
    manifest_context: Mapping[str, Any],
    requested_episodes: Iterable[int],
    version: str = "v1_hand_scene",
    resume: bool = True,
) -> dict[str, Any]:
    """Append one episode shard without changing an existing episode.

    Shards are written in a temporary directory and renamed before the small
    manifest is replaced.  This makes the command safe to resume after an
    interrupted 70-episode run.  Existing shards are never silently replaced.
    """

    if not records:
        raise ValueError(f"Episode {episode} has no records")
    if not version or "/" in version or "\\" in version or version in {".", ".."}:
        raise ValueError(f"Invalid sidecar version: {version!r}")
    root = Path(dataset_root).expanduser().resolve() / "object_geometry" / version
    manifest_path = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    requested = sorted({int(value) for value in requested_episodes})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest.get("producer") != "mesh_6d_hand_scene_v1" or manifest.get("dataset_version") != version:
            raise ValueError(
                f"Existing sidecar {root} was not produced by mesh_6d_hand_scene_v1; "
                "refusing to mix it with the legacy exporter."
            )
    else:
        if any(root.iterdir()):
            raise FileExistsError(
                f"Object geometry directory contains data but no producer manifest: {root}; "
                "inspect/archive it before starting a new production run."
            )
        manifest = {
            "status": "complete",
            "schema_version": 1,
            "producer": "mesh_6d_hand_scene_v1",
            "dataset_version": version,
            "generation": dict(manifest_context),
            # Keep the same top-level provenance key consumed by the existing
            # ObjectGeometry training join/loader.
            "calibration": manifest_context.get("calibration"),
            "object_geometry_contract": {
                "version": version,
                "point_count": POINT_COUNT,
                "units": "meters",
                "coordinate_frame": "base_link",
                "source_semantics": {
                    "0": "current validated Mesh+6D pose",
                    "1": "reviewed static target-flask geometry prior",
                },
            },
            "hand_scene_contract": {
                "version": "v1_hand_scene",
                "tactile_point_count": TACTILE_POINT_COUNT,
                "funnel_mask_size": FUNNEL_MASK_SIZE,
                "relation_valid_semantics": "causal current-frame relation validity; invalid funnel pose disables Hand-Funnel",
                "geometry_quality_semantics": "finite approved mesh transformed by an accepted tracker/prior state; this is structural validity, not metric pose accuracy",
                "geometry_quality_semantics_version": "structural_validity_v2",
                "grasp_semantics": "force magnitudes, causal first-valid reference and previous-frame delta; grasp_quality remains unknown unless an explicit stability label exists",
                "mask_semantics": "current-frame SAM2 mask; valid only when a current hand offset source is present",
                "future_frame_policy": "all state inputs are current or previous frames only",
            },
            "requested_episodes": requested,
            "episode_provenance": {},
            "records": [],
        }
    if manifest.get("requested_episodes") != requested:
        # ``requested_episodes`` is the user's coverage contract, not the
        # subset that happened to pass QA in this invocation.  Preserve the
        # historical request range and safely add newly requested episodes;
        # never narrow it during a resume.
        old_requested = sorted({int(value) for value in manifest.get("requested_episodes", [])})
        manifest["requested_episodes"] = sorted(set(old_requested) | set(requested))
        requested = manifest["requested_episodes"]
    existing_episode_records = [row for row in manifest.get("records", []) if int(row["episode_index"]) == int(episode)]
    shard_name = f"shard_e{int(episode):03d}"
    shard = root / shard_name
    if existing_episode_records or shard.exists():
        if resume:
            previous_context = manifest.get("episode_provenance", {}).get(str(int(episode)))
            if previous_context is not None and dict(previous_context) != dict(manifest_context):
                raise ValueError(
                    f"Episode {episode} already exists with different provenance; "
                    f"use a new sidecar version instead of overwriting {root}"
                )
            return {"episode": int(episode), "status": "skipped_existing", "records": len(existing_episode_records)}
        raise FileExistsError(f"Refusing to overwrite existing episode shard: {shard}")
    fields = sidecar_fields()
    temp = root / f".{shard_name}.tmp-{os.getpid()}"
    if temp.exists():
        raise FileExistsError(f"Stale temporary shard exists; inspect before removing: {temp}")
    temp.mkdir()
    try:
        for field in fields:
            if any(field not in record for record in records):
                raise ValueError(f"Episode {episode} record is missing field {field}")
            np.save(temp / f"{field}.npy", np.asarray([record[field] for record in records]))
        os.replace(temp, shard)
    except Exception:
        # Do not recursively delete a directory containing user data.  The
        # uniquely named temporary path is left for explicit inspection.
        raise
    manifest["records"] = [row for row in manifest.get("records", []) if int(row["episode_index"]) != int(episode)]
    manifest["records"].extend(
        {"episode_index": int(episode), "frame_index": int(record["frame_index"]), "shard": shard_name, "row": row}
        for row, record in enumerate(records)
    )
    manifest["records"].sort(key=lambda row: (int(row["episode_index"]), int(row["frame_index"])))
    manifest.setdefault("episode_provenance", {})[str(int(episode))] = dict(manifest_context)
    complete_episodes = sorted({int(row["episode_index"]) for row in manifest["records"]})
    manifest["coverage"] = {
        "requested_episodes": requested,
        "complete_episodes": complete_episodes,
        "missing_episodes": sorted(set(requested) - set(complete_episodes)),
    }
    manifest["status"] = "complete"
    temp_manifest = manifest_path.with_suffix(f".json.tmp-{os.getpid()}")
    temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    os.replace(temp_manifest, manifest_path)
    return {"episode": int(episode), "status": "written", "records": len(records), "root": str(root)}


def merge_requested_episodes(
    *,
    dataset_root: str | Path,
    version: str,
    requested_episodes: Iterable[int],
) -> dict[str, Any]:
    """Merge a user's requested coverage range without touching shards.

    A batch can finish with no newly QA-passed episode (all requested shards
    may already be present or all new episodes may be pending).  In that case
    the manifest still needs to remember the complete user request.  This
    helper updates only manifest coverage metadata and never creates,
    replaces, or deletes an episode shard.
    """

    root = Path(dataset_root).expanduser().resolve() / "object_geometry" / version
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "missing_manifest", "root": str(root)}
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("producer") != "mesh_6d_hand_scene_v1" or manifest.get("dataset_version") != version:
        raise ValueError(f"Existing sidecar {root} has an incompatible producer or dataset version")
    old_requested = {int(value) for value in manifest.get("requested_episodes", [])}
    merged = sorted(old_requested | {int(value) for value in requested_episodes})
    complete = sorted({int(row["episode_index"]) for row in manifest.get("records", [])})
    changed = manifest.get("requested_episodes") != merged or manifest.get("coverage", {}).get("requested_episodes") != merged
    manifest["requested_episodes"] = merged
    manifest["coverage"] = {
        "requested_episodes": merged,
        "complete_episodes": complete,
        "missing_episodes": sorted(set(merged) - set(complete)),
    }
    if changed:
        temp_manifest = manifest_path.with_suffix(f".json.tmp-{os.getpid()}")
        temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        os.replace(temp_manifest, manifest_path)
    return {"status": "updated" if changed else "unchanged", "requested_episodes": merged, "root": str(root)}
