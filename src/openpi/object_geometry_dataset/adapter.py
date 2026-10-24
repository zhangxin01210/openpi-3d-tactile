"""Optional join of object_geometry/v1 after the existing spatial adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, SupportsIndex

from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset


class RandomAccessDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: SupportsIndex) -> Mapping[str, Any]: ...


class ObjectGeometryAugmentedDataset:
    """Join object geometry by identity while preserving all existing fields."""

    def __init__(
        self,
        base_dataset: RandomAccessDataset,
        object_geometry_dataset: ObjectGeometryDerivedDataset,
        *,
        copy_arrays: bool = True,
        missing: str = "error",
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
        output_key: str = "object_geometry",
    ) -> None:
        if missing not in {"error", "skip"}:
            raise ValueError("missing must be 'error' or 'skip'")
        self.base_dataset = base_dataset
        self.object_geometry_dataset = object_geometry_dataset
        self.copy_arrays = bool(copy_arrays)
        self.missing = missing
        self.episode_key = episode_key
        self.frame_key = frame_key
        self.output_key = output_key

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        base = self.base_dataset[index]
        if not isinstance(base, Mapping):
            raise TypeError(f"Base dataset item must be a mapping, got {type(base)!r}")
        try:
            sample = self.object_geometry_dataset.get(
                episode_index=int(_scalar(base[self.episode_key])),
                frame_index=int(_scalar(base[self.frame_key])),
            )
        except KeyError:
            if self.missing == "skip":
                return dict(base)
            raise
        geometry = sample.as_dict()
        if self.copy_arrays:
            geometry = {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in geometry.items()
            }
        output = dict(base)
        spatial = dict(output.get("spatial", {}))
        if self.output_key in spatial:
            raise KeyError(f"spatial already contains {self.output_key!r}")
        spatial[self.output_key] = geometry
        output["spatial"] = spatial
        return output


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
