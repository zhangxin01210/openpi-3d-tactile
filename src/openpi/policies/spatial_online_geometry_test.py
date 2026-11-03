"""Fail-closed geometry forwarding tests; no hardware/model weights required."""
import numpy as np
import pytest
from openpi.policies.spatial_online import _validate_object_geometry
from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess


def sample():
    out = {}
    for name in ("manipulated", "target"):
        out[name + "_xyz_m"] = np.zeros((512, 3), np.float32)
        out[name + "_mask"] = np.zeros(512, bool)
        out[name + "_source"] = np.zeros(512, np.int8)
        out[name + "_age"] = np.zeros(512, np.int16)
        out[name + "_confidence"] = np.zeros(512, np.float32)
    return out


def test_schema_rejects_missing_cloud_fields():
    with pytest.raises(KeyError, match="target_confidence"):
        _validate_object_geometry({k: v for k, v in sample().items() if k != "target_confidence"})


def test_schema_rejects_wrong_shape():
    value = sample()
    value["manipulated_xyz_m"] = np.zeros((511, 3), np.float32)
    with pytest.raises(ValueError, match="512"):
        _validate_object_geometry(value)


def test_existing_spatial_packet_attaches_geometry_without_fk():
    processor = XHandSpatialOnlinePreprocess.__new__(XHandSpatialOnlinePreprocess)
    object.__setattr__(processor, "require_object_geometry", True)
    object.__setattr__(processor, "require_hand_scene", False)
    packet = {"spatial": {"visual": {"sentinel": True}}, "object_geometry": sample()}
    converted = processor(packet)
    assert converted["spatial"]["visual"]["sentinel"]
    assert converted["spatial"]["object_geometry"]["target_xyz_m"].shape == (512, 3)
    assert packet["spatial"].get("object_geometry") is None


def test_missing_geometry_fails_closed_with_existing_spatial():
    processor = XHandSpatialOnlinePreprocess.__new__(XHandSpatialOnlinePreprocess)
    object.__setattr__(processor, "require_object_geometry", True)
    object.__setattr__(processor, "require_hand_scene", True)
    with pytest.raises(ValueError, match="Required online object_geometry missing"):
        processor({"spatial": {"visual": {}}})


def test_hand_scene_must_be_present():
    processor = XHandSpatialOnlinePreprocess.__new__(XHandSpatialOnlinePreprocess)
    object.__setattr__(processor, "require_object_geometry", True)
    object.__setattr__(processor, "require_hand_scene", True)
    with pytest.raises(ValueError, match="hand_scene"):
        processor({"spatial": {}, "object_geometry": sample()})


def test_v2_requires_and_forwards_hand_keypoints():
    processor = XHandSpatialOnlinePreprocess.__new__(XHandSpatialOnlinePreprocess)
    object.__setattr__(processor, "require_object_geometry", True)
    object.__setattr__(processor, "require_hand_scene", False)
    object.__setattr__(processor, "require_hand_keypoints", True)
    packet = {"spatial": {}, "object_geometry": sample()}
    with pytest.raises(ValueError, match="hand_keypoints missing"):
        processor(packet)
    packet["hand_keypoints"] = {"xyz_m": np.zeros((18, 3), np.float32), "point_mask": np.ones(18, bool)}
    assert processor(packet)["spatial"]["hand_keypoints"]["xyz_m"].shape == (18, 3)


def test_m2_online_geometry_uses_training_gate_derivation():
    from openpi.object_geometry_dataset.adapter import _m2_state_from_rgbtrack_v1

    processor = XHandSpatialOnlinePreprocess.__new__(XHandSpatialOnlinePreprocess)
    object.__setattr__(processor, "require_object_geometry", True)
    object.__setattr__(processor, "require_hand_scene", True)
    object.__setattr__(processor, "m2_state_from_rgbtrack_v1", True)
    geometry = sample()
    geometry["manipulated_mask"][:] = True
    expected = _m2_state_from_rgbtrack_v1(geometry)
    actual = processor({"spatial": {}, "object_geometry": geometry})["spatial"]["object_geometry"]["hand_scene"]
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    assert bool(actual["manipulated_relation_valid"])
    assert not bool(actual["target_relation_valid"])

    geometry["manipulated_mask"][0] = False
    with pytest.raises(ValueError, match="uniformly valid or invalid"):
        processor({"spatial": {}, "object_geometry": geometry})
