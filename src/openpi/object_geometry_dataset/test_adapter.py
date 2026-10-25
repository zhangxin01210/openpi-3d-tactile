"""CPU-only sidecar reader/adapter tests."""

from __future__ import annotations

import json

import numpy as np

from openpi.object_geometry_dataset.adapter import ObjectGeometryAugmentedDataset
from openpi.object_geometry_dataset.derived import FUNNEL_MASK_SIZE
from openpi.object_geometry_dataset.derived import HAND_SCENE_FIELDS
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset, POINT_COUNT, TACTILE_POINT_COUNT


def _write_sidecar(root):
    sidecar = root / "object_geometry" / "v1" / "shard_e0"
    sidecar.mkdir(parents=True)
    values = {
        "episode_index": np.array([0, 0], dtype=np.int64),
        "frame_index": np.array([3, 4], dtype=np.int64),
    }
    for name in ("manipulated", "target"):
        values[f"{name}_xyz_m"] = np.zeros((2, POINT_COUNT, 3), dtype=np.float32)
        values[f"{name}_mask"] = np.zeros((2, POINT_COUNT), dtype=bool)
        values[f"{name}_source"] = np.zeros((2, POINT_COUNT), dtype=np.int8)
        values[f"{name}_age"] = np.zeros((2, POINT_COUNT), dtype=np.int16)
        values[f"{name}_confidence"] = np.zeros((2, POINT_COUNT), dtype=np.float32)
    values["manipulated_mask"][0, 0] = True
    values["manipulated_confidence"][0, 0] = 1.0
    for field, value in values.items():
        np.save(sidecar / f"{field}.npy", value)
    (sidecar.parent / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "object_geometry_contract": {"version": "v1", "point_count": POINT_COUNT},
                "records": [
                    {"episode_index": 0, "frame_index": 3, "shard": "shard_e0", "row": 0},
                    {"episode_index": 0, "frame_index": 4, "shard": "shard_e0", "row": 1},
                ],
            }
        )
    )


def test_reader_join_and_shapes(tmp_path):
    _write_sidecar(tmp_path)
    dataset = ObjectGeometryDerivedDataset(tmp_path)
    sample = dataset.get(episode_index=0, frame_index=3)
    assert sample.manipulated_xyz_m.shape == (POINT_COUNT, 3)
    assert sample.manipulated_mask[0]
    assert sample.manipulated_source.dtype == np.int8
    try:
        dataset.get(episode_index=1, frame_index=3)
    except KeyError:
        pass
    else:
        raise AssertionError("missing sidecar record did not fail")


def test_adapter_preserves_old_sample_and_skip_behavior(tmp_path):
    _write_sidecar(tmp_path)
    sidecar = ObjectGeometryDerivedDataset(tmp_path)
    base = [{"episode_index": 0, "frame_index": 3, "spatial": {"visual": "kept"}}, {"episode_index": 1, "frame_index": 3}]
    dataset = ObjectGeometryAugmentedDataset(base, sidecar, missing="skip")
    joined = dataset[0]
    assert joined["spatial"]["visual"] == "kept"
    assert joined["spatial"]["object_geometry"]["manipulated_xyz_m"].shape == (POINT_COUNT, 3)
    assert dataset[1] == base[1]


def test_hand_scene_reader_exposes_causal_state_without_changing_old_sidecar_contract(tmp_path):
    sidecar = tmp_path / "object_geometry" / "v1_hand_scene" / "shard_e0"
    sidecar.mkdir(parents=True)
    values = {
        "episode_index": np.array([0], dtype=np.int64),
        "frame_index": np.array([3], dtype=np.int64),
    }
    for name in ("manipulated", "target"):
        values[f"{name}_xyz_m"] = np.zeros((1, POINT_COUNT, 3), dtype=np.float32)
        values[f"{name}_mask"] = np.ones((1, POINT_COUNT), dtype=bool)
        values[f"{name}_source"] = np.zeros((1, POINT_COUNT), dtype=np.int8)
        values[f"{name}_age"] = np.zeros((1, POINT_COUNT), dtype=np.int16)
        values[f"{name}_confidence"] = np.ones((1, POINT_COUNT), dtype=np.float32)
    values.update(
        manipulated_relation_valid=np.array([True]),
        target_relation_valid=np.array([True]),
        manipulated_geometry_quality=np.array([0.8], dtype=np.float32),
        target_geometry_quality=np.array([0.5], dtype=np.float32),
        grasp_reference_force_norm=np.zeros((1, TACTILE_POINT_COUNT), dtype=np.float32),
        grasp_short_delta_force_norm=np.zeros((1, TACTILE_POINT_COUNT), dtype=np.float32),
        grasp_contact_valid=np.ones((1, TACTILE_POINT_COUNT), dtype=bool),
        grasp_quality=np.array([0.9], dtype=np.float32),
        funnel_mask=np.zeros((1, FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE), dtype=bool),
        funnel_mask_valid=np.array([False]),
        funnel_mask_quality=np.array([0.0], dtype=np.float32),
        funnel_mask_hand_offset_xy=np.zeros((1, 2), dtype=np.float32),
    )
    assert set(HAND_SCENE_FIELDS) <= set(values)
    for field, value in values.items():
        np.save(sidecar / f"{field}.npy", value)
    (sidecar.parent / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "object_geometry_contract": {"version": "v1_hand_scene", "point_count": POINT_COUNT},
                "hand_scene_contract": {
                    "version": "v1_hand_scene",
                    "tactile_point_count": TACTILE_POINT_COUNT,
                    "funnel_mask_size": FUNNEL_MASK_SIZE,
                },
                "records": [{"episode_index": 0, "frame_index": 3, "shard": "shard_e0", "row": 0}],
            }
        )
    )
    sample = ObjectGeometryDerivedDataset(tmp_path, version="v1_hand_scene").get(episode_index=0, frame_index=3)
    assert sample.hand_scene is not None
    assert sample.hand_scene.grasp_contact_valid.shape == (TACTILE_POINT_COUNT,)
    assert not bool(sample.hand_scene.funnel_mask_valid)
    assert sample.as_dict()["hand_scene"]["target_geometry_quality"] == np.float32(0.5)
