"""Memory-mapped reader for the independent object_geometry/v1 sidecar."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OBJECT_NAMES = ("manipulated", "target")
POINT_COUNT = 512


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
        return {
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


class ObjectGeometryDerivedDataset:
    """Read fixed-size sidecar arrays with a small mmap shard cache."""

    _FIELDS = tuple(
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
        self._records = []
        self._lookup: dict[tuple[int, int], tuple[str, int]] = {}
        for record in self.manifest.get("records", []):
            key = (int(record["episode_index"]), int(record["frame_index"]))
            if key in self._lookup:
                raise ValueError(f"Duplicate object geometry key {key}")
            shard = str(record["shard"])
            row = int(record["row"])
            self._lookup[key] = (shard, row)
            self._records.append(key)
        self._max_cached_shards = max(1, int(max_cached_shards))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._records)

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
        values = {field: arrays[field][row] for field in self._FIELDS}
        return ObjectGeometrySample(episode_index=key[0], frame_index=key[1], **values)

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
            for field in (*self._FIELDS, "episode_index", "frame_index")
        }
        self._cache[shard_name] = arrays
        self._cache.move_to_end(shard_name)
        while len(self._cache) > self._max_cached_shards:
            self._cache.popitem(last=False)
        return arrays
