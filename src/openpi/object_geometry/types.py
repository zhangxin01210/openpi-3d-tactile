"""Typed containers for mask-derived object geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _points(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{field_name} must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _per_point(value: Any | None, *, count: int, field_name: str, dtype: np.dtype) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (count,):
        raise ValueError(f"{field_name} must have shape ({count},), got {array.shape}")
    return array


@dataclass(frozen=True)
class PointCloud:
    """A compact point cloud with optional provenance attached per point."""

    xyz: np.ndarray
    rgb: np.ndarray | None = None
    source_pixel: np.ndarray | None = None
    source: np.ndarray | None = None
    confidence: np.ndarray | None = None
    age: np.ndarray | None = None

    def __post_init__(self) -> None:
        xyz = _points(self.xyz, field_name="xyz")
        object.__setattr__(self, "xyz", xyz)
        count = len(xyz)

        if self.rgb is not None:
            rgb = np.asarray(self.rgb, dtype=np.uint8)
            if rgb.shape != (count, 3):
                raise ValueError(f"rgb must have shape ({count}, 3), got {rgb.shape}")
            object.__setattr__(self, "rgb", rgb)
        object.__setattr__(
            self,
            "source_pixel",
            _per_point(self.source_pixel, count=count, field_name="source_pixel", dtype=np.int32),
        )
        source = _per_point(self.source, count=count, field_name="source", dtype=np.int8)
        if source is not None and not np.isin(source, (0, 1)).all():
            raise ValueError("source values must be 0 (current) or 1 (history)")
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "confidence",
            _per_point(self.confidence, count=count, field_name="confidence", dtype=np.float32),
        )
        object.__setattr__(
            self,
            "age",
            _per_point(self.age, count=count, field_name="age", dtype=np.int32),
        )

    @classmethod
    def empty(cls) -> "PointCloud":
        return cls(np.empty((0, 3), dtype=np.float32))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for small manifests."""

        return {
            "xyz": self.xyz.tolist(),
            "rgb": None if self.rgb is None else self.rgb.tolist(),
            "source_pixel": None if self.source_pixel is None else self.source_pixel.tolist(),
            "source": None if self.source is None else self.source.tolist(),
            "confidence": None if self.confidence is None else self.confidence.tolist(),
            "age": None if self.age is None else self.age.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PointCloud":
        return cls(
            xyz=value["xyz"],
            rgb=value.get("rgb"),
            source_pixel=value.get("source_pixel"),
            source=value.get("source"),
            confidence=value.get("confidence"),
            age=value.get("age"),
        )


@dataclass(frozen=True)
class ObjectGeometry:
    """Current geometry A and optional causal history geometry B for one object."""

    current: PointCloud
    history: PointCloud = field(default_factory=PointCloud.empty)
    history_support: np.ndarray | None = None
    history_space: str = "base"

    def __post_init__(self) -> None:
        if self.history_space not in {"base", "local"}:
            raise ValueError("history_space must be 'base' or 'local'")
        if self.history_support is not None:
            support = np.asarray(self.history_support, dtype=np.int16).reshape(-1)
            if support.shape != (len(self.history.xyz),):
                raise ValueError(
                    f"history_support must have shape ({len(self.history.xyz)},), got {support.shape}"
                )
            object.__setattr__(self, "history_support", support)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict(),
            "history": self.history.to_dict(),
            "history_support": None
            if self.history_support is None
            else self.history_support.tolist(),
            "history_space": self.history_space,
        }


@dataclass(frozen=True)
class ObjectGeometryFrame:
    """All named object geometries for one causal frame."""

    frame_index: int
    objects: dict[str, ObjectGeometry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "objects": {name: geometry.to_dict() for name, geometry in self.objects.items()},
        }
