"""Memory-mapped reader for independent object-geometry sidecars.

``v1_hand_scene`` extends the reviewed fixed-cloud contract with a small,
causal state record.  It remains an independent version, so existing v1 and
v1_front_candidate data need neither rewriting nor reinterpretation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OBJECT_NAMES = ("manipulated", "target")
POINT_COUNT = 512
TACTILE_POINT_COUNT = 600
FUNNEL_MASK_SIZE = 32
DTCR_TACTILE_FIELDS = (
    "ema_before_force_norm",
    "ema_after_force_norm",
    "current_minus_ema_force_norm",
    "current_minus_previous_force_norm",
    "reference_valid",
    "previous_valid",
    "sample_valid",
    "alpha",
)


@dataclass(frozen=True, slots=True)
class CausalTactileSample:
    """Per-frame causal tactile EMA state joined by episode/frame."""

    ema_before_force_norm: np.ndarray
    ema_after_force_norm: np.ndarray
    current_minus_ema_force_norm: np.ndarray
    current_minus_previous_force_norm: np.ndarray
    reference_valid: np.ndarray
    previous_valid: np.ndarray
    sample_valid: np.ndarray
    alpha: np.ndarray

    def __post_init__(self) -> None:
        for name in DTCR_TACTILE_FIELDS:
            value = np.asarray(getattr(self, name), dtype=np.bool_ if name.endswith("valid") else np.float32)
            expected = () if name == "alpha" else (TACTILE_POINT_COUNT,)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in DTCR_TACTILE_FIELDS}


@dataclass(frozen=True, slots=True)
class HandSceneSample:
    """One causal Hand-Scene V0 state record joined with a geometry frame."""

    manipulated_relation_valid: np.ndarray
    target_relation_valid: np.ndarray
    manipulated_geometry_quality: np.ndarray
    target_geometry_quality: np.ndarray
    grasp_reference_force_norm: np.ndarray
    grasp_short_delta_force_norm: np.ndarray
    grasp_contact_valid: np.ndarray
    grasp_quality: np.ndarray
    funnel_mask: np.ndarray
    funnel_mask_valid: np.ndarray
    funnel_mask_quality: np.ndarray
    funnel_mask_hand_offset_xy: np.ndarray
    causal_tactile: CausalTactileSample | None = None

    def __post_init__(self) -> None:
        scalar_bool = ("manipulated_relation_valid", "target_relation_valid", "funnel_mask_valid")
        scalar_float = (
            "manipulated_geometry_quality",
            "target_geometry_quality",
            "grasp_quality",
            "funnel_mask_quality",
        )
        for name in scalar_bool:
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != ():
                raise ValueError(f"{name} must be scalar, got {value.shape}")
            object.__setattr__(self, name, value)
        for name in scalar_float:
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != () or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        for name, dtype in (
            ("grasp_reference_force_norm", np.float32),
            ("grasp_short_delta_force_norm", np.float32),
            ("grasp_contact_valid", bool),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != (TACTILE_POINT_COUNT,):
                raise ValueError(f"{name} must have shape ({TACTILE_POINT_COUNT},), got {value.shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
            object.__setattr__(self, name, value)
        mask = np.asarray(self.funnel_mask, dtype=bool)
        if mask.shape != (FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE):
            raise ValueError(
                f"funnel_mask must have shape ({FUNNEL_MASK_SIZE}, {FUNNEL_MASK_SIZE}), got {mask.shape}"
            )
        object.__setattr__(self, "funnel_mask", mask)
        hand_offset = np.asarray(self.funnel_mask_hand_offset_xy, dtype=np.float32)
        if hand_offset.shape != (2,) or not np.isfinite(hand_offset).all():
            raise ValueError("funnel_mask_hand_offset_xy must be finite with shape (2,)")
        object.__setattr__(self, "funnel_mask_hand_offset_xy", hand_offset)
        if self.causal_tactile is not None and not isinstance(self.causal_tactile, CausalTactileSample):
            object.__setattr__(self, "causal_tactile", CausalTactileSample(**self.causal_tactile))

    def as_dict(self) -> dict[str, np.ndarray]:
        result = {
            name: getattr(self, name)
            for name in HAND_SCENE_FIELDS
        }
        if self.causal_tactile is not None:
            result["causal_tactile"] = self.causal_tactile.as_dict()
        return result


HAND_SCENE_FIELDS = (
    "manipulated_relation_valid",
    "target_relation_valid",
    "manipulated_geometry_quality",
    "target_geometry_quality",
    "grasp_reference_force_norm",
    "grasp_short_delta_force_norm",
    "grasp_contact_valid",
    "grasp_quality",
    "funnel_mask",
    "funnel_mask_valid",
    "funnel_mask_quality",
    "funnel_mask_hand_offset_xy",
)


@dataclass(frozen=True, slots=True)
class ObjectGeometrySample:
    """One fixed-size object-geometry record joined by episode/frame identity."""

    episode_index: int
    frame_index: int
    manipulated_xyz_m: np.ndarray
    manipulated_mask: np.ndarray
    manipulated_source: np.ndarray
    manipulated_age: np.ndarray
    manipulated_confidence: np.ndarray
    target_xyz_m: np.ndarray
    target_mask: np.ndarray
    target_source: np.ndarray
    target_age: np.ndarray
    target_confidence: np.ndarray
    hand_scene: HandSceneSample | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_index", int(self.episode_index))
        object.__setattr__(self, "frame_index", int(self.frame_index))
        for name in OBJECT_NAMES:
            xyz = np.asarray(getattr(self, f"{name}_xyz_m"), dtype=np.float32)
            mask = np.asarray(getattr(self, f"{name}_mask"), dtype=bool)
            source = np.asarray(getattr(self, f"{name}_source"), dtype=np.int8)
            age = np.asarray(getattr(self, f"{name}_age"), dtype=np.int16)
            confidence = np.asarray(getattr(self, f"{name}_confidence"), dtype=np.float32)
            if xyz.shape != (POINT_COUNT, 3):
                raise ValueError(f"{name}_xyz_m must have shape ({POINT_COUNT}, 3), got {xyz.shape}")
            for field_name, value in (
                ("mask", mask),
                ("source", source),
                ("age", age),
                ("confidence", confidence),
            ):
                if value.shape != (POINT_COUNT,):
                    raise ValueError(f"{name}_{field_name} must have shape ({POINT_COUNT},), got {value.shape}")
            if not np.isfinite(xyz).all() or not np.isfinite(confidence).all():
                raise ValueError(f"{name} geometry contains NaN or Inf")
            if not np.isin(source, (0, 1)).all():
                raise ValueError(f"{name}_source must contain only 0 or 1")
            if np.any(mask & (confidence < 0)):
                raise ValueError(f"{name}_confidence must be non-negative for valid points")
            object.__setattr__(self, f"{name}_xyz_m", xyz)
            object.__setattr__(self, f"{name}_mask", mask)
            object.__setattr__(self, f"{name}_source", source)
            object.__setattr__(self, f"{name}_age", age)
            object.__setattr__(self, f"{name}_confidence", confidence)

    def as_dict(self) -> dict[str, Any]:
        values = {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            **{
                field: getattr(self, field)
                for name in OBJECT_NAMES
                for field in (
                    f"{name}_xyz_m",
                    f"{name}_mask",
                    f"{name}_source",
                    f"{name}_age",
                    f"{name}_confidence",
                )
            },
        }
        if self.hand_scene is not None:
            values["hand_scene"] = self.hand_scene.as_dict()
        return values


class ObjectGeometryDerivedDataset:
    """Read fixed-size sidecar arrays with a small mmap shard cache."""

    _GEOMETRY_FIELDS = tuple(
        field
        for name in OBJECT_NAMES
        for field in (
            f"{name}_xyz_m",
            f"{name}_mask",
            f"{name}_source",
            f"{name}_age",
            f"{name}_confidence",
        )
    )

    def __init__(self, dataset_root: str | Path, *, version: str = "v1", max_cached_shards: int = 8) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.root = self.dataset_root / "object_geometry" / version
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = __import__("json").loads(manifest_path.read_text(encoding="utf8"))
        if self.manifest.get("status") != "complete":
            raise ValueError(f"Object geometry sidecar is not complete: {self.manifest.get('status')!r}")
        contract = self.manifest.get("object_geometry_contract", {})
        if int(contract.get("point_count", -1)) != POINT_COUNT:
            raise ValueError("Object geometry sidecar point_count must be 512")
        hand_scene_contract = self.manifest.get("hand_scene_contract")
        self.has_hand_scene = hand_scene_contract is not None
        self.has_dtcr_tactile = False
        if self.has_hand_scene:
            if not isinstance(hand_scene_contract, dict) or hand_scene_contract.get("version") != "v1_hand_scene":
                raise ValueError("Hand-Scene sidecar must declare hand_scene_contract version v1_hand_scene")
            if int(hand_scene_contract.get("tactile_point_count", -1)) != TACTILE_POINT_COUNT:
                raise ValueError("Hand-Scene sidecar tactile_point_count must be 600")
            if int(hand_scene_contract.get("funnel_mask_size", -1)) != FUNNEL_MASK_SIZE:
                raise ValueError("Hand-Scene sidecar funnel_mask_size must be 32")
            self.has_dtcr_tactile = bool(hand_scene_contract.get("dtcr_tactile", False))
        self._fields = self._GEOMETRY_FIELDS + (HAND_SCENE_FIELDS if self.has_hand_scene else ()) + (
            tuple(f"dtcr_{field}" for field in DTCR_TACTILE_FIELDS) if self.has_dtcr_tactile else ()
        )
        self._records = []
        self._lookup: dict[tuple[int, int], tuple[str, int]] = {}
        episode_frame_ids: dict[int, list[int]] = {}
        for record in self.manifest.get("records", []):
            key = (int(record["episode_index"]), int(record["frame_index"]))
            if key in self._lookup:
                raise ValueError(f"Duplicate object geometry key {key}")
            shard = str(record["shard"])
            row = int(record["row"])
            self._lookup[key] = (shard, row)
            self._records.append(key)
            episode_frame_ids.setdefault(key[0], []).append(key[1])
        # The manifest index is already built above.  Expose the distinct
        # episode ids without rescanning shards or materialising any arrays;
        # diagnostics and batch tooling should not reach into ``_records``.
        self._episode_ids = tuple(sorted({episode for episode, _ in self._records}))
        self._episode_frame_ids = {
            episode: tuple(sorted(frame_ids))
            for episode, frame_ids in episode_frame_ids.items()
        }
        self._max_cached_shards = max(1, int(max_cached_shards))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """Distinct manifest episode ids in immutable public form."""

        return self._episode_ids

    def frame_ids(self, episode_index: int) -> tuple[int, ...]:
        """Return manifest frame ids for one episode without reading shards."""

        try:
            return self._episode_frame_ids[int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"Unknown object geometry episode {episode_index}") from exc

    def __getitem__(self, index: int | tuple[int, int]) -> ObjectGeometrySample:
        if isinstance(index, tuple):
            return self.get(episode_index=index[0], frame_index=index[1])
        index = int(index)
        if index < 0:
            index += len(self._records)
        if not 0 <= index < len(self._records):
            raise IndexError(index)
        episode_index, frame_index = self._records[index]
        return self.get(episode_index=episode_index, frame_index=frame_index)

    def get(self, *, episode_index: int, frame_index: int) -> ObjectGeometrySample:
        key = (int(episode_index), int(frame_index))
        try:
            shard_name, row = self._lookup[key]
        except KeyError as exc:
            raise KeyError(f"No object geometry record for episode/frame {key}") from exc
        arrays = self._load_shard(shard_name)
        if not 0 <= row < len(arrays["frame_index"]):
            raise ValueError(f"Row {row} outside shard {shard_name}")
        if int(arrays["episode_index"][row]) != key[0] or int(arrays["frame_index"][row]) != key[1]:
            raise ValueError(f"Manifest/index mismatch for {key} in {shard_name}[{row}]")
        values = {field: arrays[field][row] for field in self._GEOMETRY_FIELDS}
        hand_scene = None
        if self.has_hand_scene:
            causal_tactile = None
            if self.has_dtcr_tactile:
                causal_tactile = CausalTactileSample(
                    **{field: arrays[f"dtcr_{field}"][row] for field in DTCR_TACTILE_FIELDS}
                )
            hand_scene = HandSceneSample(
                **{field: arrays[field][row] for field in HAND_SCENE_FIELDS},
                causal_tactile=causal_tactile,
            )
        return ObjectGeometrySample(episode_index=key[0], frame_index=key[1], hand_scene=hand_scene, **values)

    def _load_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        cached = self._cache.get(shard_name)
        if cached is not None:
            self._cache.move_to_end(shard_name)
            return cached
        shard_root = self.root / shard_name
        if not shard_root.is_dir():
            raise FileNotFoundError(shard_root)
        arrays = {
            field: np.load(shard_root / f"{field}.npy", mmap_mode="r", allow_pickle=False)
            for field in (*self._fields, "episode_index", "frame_index")
        }
        self._cache[shard_name] = arrays
        self._cache.move_to_end(shard_name)
        while len(self._cache) > self._max_cached_shards:
            self._cache.popitem(last=False)
        return arrays
