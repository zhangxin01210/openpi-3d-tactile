"""Optional join of object_geometry/v1 after the existing spatial adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, SupportsIndex

import numpy as np

from openpi.object_geometry_dataset.derived import FUNNEL_MASK_SIZE
from openpi.object_geometry_dataset.derived import TACTILE_POINT_COUNT
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
        m2_state_from_rgbtrack_v1: bool = False,
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
        self.m2_state_from_rgbtrack_v1 = m2_state_from_rgbtrack_v1
        if m2_state_from_rgbtrack_v1:
            manifest = object_geometry_dataset.manifest
            expected_roles = {
                "1028_press_button": {"manipulated": "button", "target": "box"},
                "1030_yellow_socket": {"manipulated": "red", "target": "yellow"},
            }
            if (object_geometry_dataset.has_hand_scene
                    or manifest.get("producer") != "scripts/preprocess/export_object_geometry_rgbtrack.py"
                    or manifest.get("object_geometry_contract", {}).get("version") != "v1"
                    or manifest.get("role_to_object") != expected_roles.get(manifest.get("dataset"))
                    or manifest.get("dataset") not in expected_roles
                    or manifest.get("dataset") != object_geometry_dataset.dataset_root.name
                    or manifest.get("coordinate_frame") != "base_link"
                    or manifest.get("length_unit") != "m"
                    or not manifest.get("approved_geometry_sha256")):
                raise ValueError("M2 state derivation requires the accepted task RGBTrack v1 geometry contract")

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
        if self.m2_state_from_rgbtrack_v1:
            geometry["hand_scene"] = _m2_state_from_rgbtrack_v1(geometry)
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


def _m2_state_from_rgbtrack_v1(geometry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Match historical production's pose-valid / structural-quality gates.

    The v1 exporter writes all 512 mask entries from the same pose-valid flag.
    M2 reads only the four gates; the remaining zero arrays are unused ABI
    placeholders, not grasp or segmentation observations or labels.
    """
    state = {}
    for role in ("manipulated", "target"):
        mask = np.asarray(geometry[f"{role}_mask"], dtype=bool)
        if mask.any() != mask.all():
            raise ValueError(f"{role} RGBTrack v1 pose mask must be uniformly valid or invalid")
        valid = bool(mask.all())
        if valid and (np.any(geometry[f"{role}_source"] != 0) or np.any(geometry[f"{role}_age"] != 0)):
            raise ValueError(f"{role} valid RGBTrack v1 pose must be current/frozen, not inferred history")
        state[f"{role}_relation_valid"] = np.asarray(valid, dtype=bool)
        state[f"{role}_geometry_quality"] = np.asarray(float(valid), dtype=np.float32)
    state.update(
        grasp_reference_force_norm=np.zeros(TACTILE_POINT_COUNT, dtype=np.float32),
        grasp_short_delta_force_norm=np.zeros(TACTILE_POINT_COUNT, dtype=np.float32),
        grasp_contact_valid=np.zeros(TACTILE_POINT_COUNT, dtype=bool),
        grasp_quality=np.asarray(0.0, dtype=np.float32),
        funnel_mask=np.zeros((FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE), dtype=bool),
        funnel_mask_valid=np.zeros((), dtype=bool),
        funnel_mask_quality=np.asarray(0.0, dtype=np.float32),
        funnel_mask_hand_offset_xy=np.zeros(2, dtype=np.float32),
    )
    return state
