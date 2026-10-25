"""Focused object-geometry sidecar path tests without a LeRobot dataset."""

from __future__ import annotations

import numpy as np

from openpi.models.model import Observation
from openpi.models.pi0 import Pi0
from openpi.training.config import DataConfig
from openpi.training.config import ObjectGeometryDataConfig
from openpi.training.data_loader import transform_dataset
from openpi.transforms import Group
from openpi.transforms import RepackTransform


def _geometry(batch_size: int = 1) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name in ("manipulated", "target"):
        fields[f"{name}_xyz_m"] = np.zeros((batch_size, 512, 3), dtype=np.float32)
        fields[f"{name}_mask"] = np.zeros((batch_size, 512), dtype=bool)
        fields[f"{name}_source"] = np.zeros((batch_size, 512), dtype=np.int8)
        fields[f"{name}_age"] = np.zeros((batch_size, 512), dtype=np.int16)
        fields[f"{name}_confidence"] = np.zeros((batch_size, 512), dtype=np.float32)
    fields["manipulated_mask"][:, 0] = True
    fields["target_mask"][:, 0] = True
    return fields


def _hand_scene(batch_size: int = 1) -> dict[str, np.ndarray]:
    return {
        "manipulated_relation_valid": np.ones((batch_size,), dtype=bool),
        "target_relation_valid": np.ones((batch_size,), dtype=bool),
        "manipulated_geometry_quality": np.ones((batch_size,), dtype=np.float32),
        "target_geometry_quality": np.full((batch_size,), 0.5, dtype=np.float32),
        "grasp_reference_force_norm": np.zeros((batch_size, 600), dtype=np.float32),
        "grasp_short_delta_force_norm": np.zeros((batch_size, 600), dtype=np.float32),
        "grasp_contact_valid": np.zeros((batch_size, 600), dtype=bool),
        "grasp_quality": np.zeros((batch_size,), dtype=np.float32),
        "funnel_mask": np.zeros((batch_size, 32, 32), dtype=bool),
        "funnel_mask_valid": np.zeros((batch_size,), dtype=bool),
        "funnel_mask_quality": np.zeros((batch_size,), dtype=np.float32),
        "funnel_mask_hand_offset_xy": np.zeros((batch_size, 2), dtype=np.float32),
    }


def test_object_geometry_only_sidecar_survives_transforms() -> None:
    raw = [{"state": np.array([1.0], dtype=np.float32), "spatial": {"object_geometry": _geometry(1)}}]
    config = DataConfig(
        repo_id="fake",
        object_geometry=ObjectGeometryDataConfig(dataset_root="unused"),
        # Mimic the repack stage that would otherwise discard unknown keys.
        repack_transforms=Group(inputs=[RepackTransform({"state": "state"})]),
    )
    transformed = transform_dataset(raw, config)
    item = transformed[0]
    assert item["state"].tolist() == [1.0]
    assert item["spatial"]["object_geometry"]["manipulated_mask"][0, 0]


def test_observation_and_pi0_handoff_keep_object_geometry() -> None:
    geometry = _geometry()
    data = {
        "image": {"base_0_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((1,), dtype=bool)},
        "state": np.zeros((1, 18), dtype=np.float32),
        "spatial": {"object_geometry": geometry},
    }
    observation = Observation.from_dict(data)
    assert observation.spatial is not None
    assert observation.spatial.object_geometry is not None
    np.testing.assert_array_equal(observation.spatial.object_geometry.manipulated_mask, geometry["manipulated_mask"])

    class Router:
        def __call__(self, spatial):
            self.last_input = spatial
            return spatial

    class Pi0HandoffHarness:
        spatial_use_visual = True
        spatial_use_tactile = True

        def __init__(self):
            self.spatial_router = Router()

    harness = Pi0HandoffHarness()
    encoded = Pi0._encode_spatial_conditioning(harness, observation)  # noqa: SLF001 - exercise the integration seam.
    assert encoded is harness.spatial_router.last_input
    assert encoded.object_geometry is observation.spatial.object_geometry


def test_observation_constructs_optional_hand_scene_contract() -> None:
    geometry = _geometry()
    geometry["hand_scene"] = _hand_scene()
    data = {
        "image": {"base_0_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((1,), dtype=bool)},
        "state": np.zeros((1, 18), dtype=np.float32),
        "spatial": {"object_geometry": geometry},
    }
    observation = Observation.from_dict(data)
    assert observation.spatial is not None
    assert observation.spatial.object_geometry is not None
    hand_scene = observation.spatial.object_geometry.hand_scene
    assert hand_scene is not None
    assert hand_scene.grasp_reference_force_norm.shape == (1, 600)
    assert hand_scene.funnel_mask_hand_offset_xy.shape == (1, 2)
