"""CPU-only contract tests for the Hand-Scene sidecar exporter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from openpi.object_geometry_dataset.derived import DTCR_TACTILE_FIELDS
from openpi.object_geometry_dataset.derived import FUNNEL_MASK_SIZE
from openpi.object_geometry_dataset.derived import POINT_COUNT
from openpi.object_geometry_dataset.derived import TACTILE_POINT_COUNT
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _exporter_module():
    name = "object_geometry_export_hand_scene_sidecar_test_module"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts/object_geometry/export_hand_scene_sidecar.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(root: Path) -> None:
    shard = root / "object_geometry" / "v1_front_candidate" / "shard_e023"
    shard.mkdir(parents=True)
    values = {"episode_index": np.array([23]), "frame_index": np.array([7])}
    for name in ("manipulated", "target"):
        values[f"{name}_xyz_m"] = np.zeros((1, POINT_COUNT, 3), dtype=np.float32)
        values[f"{name}_mask"] = np.ones((1, POINT_COUNT), dtype=bool)
        values[f"{name}_source"] = np.zeros((1, POINT_COUNT), dtype=np.int8)
        values[f"{name}_age"] = np.zeros((1, POINT_COUNT), dtype=np.int16)
        values[f"{name}_confidence"] = np.ones((1, POINT_COUNT), dtype=np.float32)
    for field, value in values.items():
        np.save(shard / f"{field}.npy", value)
    (shard.parent / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "calibration": {"profile_sha256": "tested", "coordinate_frame": "base_link"},
                "object_geometry_contract": {"version": "v1_front_candidate", "point_count": POINT_COUNT},
                "records": [{"episode_index": 23, "frame_index": 7, "shard": "shard_e023", "row": 0}],
            }
        )
    )


def _write_state(root: Path, *, max_source_frame: int, dtcr: bool = False) -> None:
    target = root / "e023" / "f000007.npz"
    target.parent.mkdir(parents=True)
    np.savez_compressed(
        target,
        causal_max_source_frame=np.asarray(max_source_frame),
        manipulated_relation_valid=np.zeros((), dtype=bool),
        target_relation_valid=np.ones((), dtype=bool),
        manipulated_geometry_quality=np.asarray(0.0, dtype=np.float32),
        target_geometry_quality=np.asarray(0.6, dtype=np.float32),
        grasp_reference_force_norm=np.zeros((TACTILE_POINT_COUNT,), dtype=np.float32),
        grasp_short_delta_force_norm=np.zeros((TACTILE_POINT_COUNT,), dtype=np.float32),
        grasp_contact_valid=np.zeros((TACTILE_POINT_COUNT,), dtype=bool),
        grasp_quality=np.asarray(0.5, dtype=np.float32),
        funnel_mask=np.zeros((64, 48), dtype=bool),
        funnel_mask_valid=np.zeros((), dtype=bool),
        funnel_mask_quality=np.asarray(0.0, dtype=np.float32),
        funnel_mask_hand_offset_xy=np.zeros((2,), dtype=np.float32),
    )
    if dtcr:
        with np.load(target, allow_pickle=False) as raw:
            values = {name: raw[name] for name in raw.files}
        values.update(
            {
                "dtcr_ema_before_force_norm": np.ones((TACTILE_POINT_COUNT,), dtype=np.float32),
                "dtcr_ema_after_force_norm": np.ones((TACTILE_POINT_COUNT,), dtype=np.float32),
                "dtcr_current_minus_ema_force_norm": np.full((TACTILE_POINT_COUNT,), 0.2, dtype=np.float32),
                "dtcr_current_minus_previous_force_norm": np.full((TACTILE_POINT_COUNT,), 0.1, dtype=np.float32),
                "dtcr_reference_valid": np.ones((TACTILE_POINT_COUNT,), dtype=bool),
                "dtcr_previous_valid": np.ones((TACTILE_POINT_COUNT,), dtype=bool),
                "dtcr_sample_valid": np.ones((TACTILE_POINT_COUNT,), dtype=bool),
                "dtcr_alpha": np.asarray(0.1, dtype=np.float32),
            }
        )
        np.savez_compressed(target, **values)


def test_hand_scene_export_preserves_metric_clouds_and_causal_invalid_state(tmp_path: Path) -> None:
    exporter = _exporter_module()
    source_root = tmp_path / "source"
    state_root = tmp_path / "states"
    output_root = tmp_path / "output"
    _write_source(source_root)
    _write_state(state_root, max_source_frame=7)
    manifest = exporter.build_sidecar(
        source_dataset_root=source_root,
        source_version="v1_front_candidate",
        state_root=state_root,
        output_dataset_root=output_root,
        version="v1_hand_scene",
        episodes="23",
        generation_commit="test",
    )
    assert manifest["status"] == "complete"
    assert manifest["hand_scene_contract"]["future_frame_policy"].startswith("causal_max_source_frame")
    sample = ObjectGeometryDerivedDataset(output_root, version="v1_hand_scene").get(episode_index=23, frame_index=7)
    assert sample.manipulated_xyz_m.shape == (POINT_COUNT, 3)
    assert sample.hand_scene is not None
    assert not bool(sample.hand_scene.manipulated_relation_valid)
    assert sample.hand_scene.funnel_mask.shape == (FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE)


def test_hand_scene_export_rejects_future_state_record(tmp_path: Path) -> None:
    exporter = _exporter_module()
    source_root = tmp_path / "source"
    state_root = tmp_path / "states"
    _write_source(source_root)
    _write_state(state_root, max_source_frame=8)
    with pytest.raises(ValueError, match="after labelled frame"):
        exporter.build_sidecar(
            source_dataset_root=source_root,
            source_version="v1_front_candidate",
            state_root=state_root,
            output_dataset_root=tmp_path / "output",
            version="v1_hand_scene",
            episodes="23",
            generation_commit="test",
        )


def test_dtcr_sidecar_joins_causal_tactile_fields_without_touching_v0(tmp_path: Path) -> None:
    exporter = _exporter_module()
    source_root = tmp_path / "source"
    state_root = tmp_path / "states"
    output_root = tmp_path / "output"
    _write_source(source_root)
    _write_state(state_root, max_source_frame=7, dtcr=True)
    manifest = exporter.build_sidecar(
        source_dataset_root=source_root,
        source_version="v1_front_candidate",
        state_root=state_root,
        output_dataset_root=output_root,
        version="v1_hand_scene_dtcr_v1",
        episodes="23",
        generation_commit="test",
    )
    assert manifest["hand_scene_contract"]["dtcr_tactile"] is True
    sample = ObjectGeometryDerivedDataset(output_root, version="v1_hand_scene_dtcr_v1").get(
        episode_index=23, frame_index=7
    )
    assert sample.hand_scene is not None
    assert sample.hand_scene.causal_tactile is not None
    np.testing.assert_allclose(sample.hand_scene.causal_tactile.current_minus_ema_force_norm, 0.2)
    assert set(DTCR_TACTILE_FIELDS) == set(sample.hand_scene.causal_tactile.as_dict())


def test_dtcr_export_can_branch_from_existing_q2_without_rewriting_it(tmp_path: Path) -> None:
    exporter = _exporter_module()
    source_root = tmp_path / "source"
    state_root = tmp_path / "states"
    output_root = tmp_path / "output"
    _write_source(source_root)
    source_shard = source_root / "object_geometry" / "v1_front_candidate" / "shard_e023"
    for field, value in {
        "manipulated_relation_valid": np.asarray([False]),
        "target_relation_valid": np.asarray([True]),
        "manipulated_geometry_quality": np.asarray([0.0], dtype=np.float32),
        "target_geometry_quality": np.asarray([0.6], dtype=np.float32),
        "grasp_reference_force_norm": np.zeros((1, TACTILE_POINT_COUNT), dtype=np.float32),
        "grasp_short_delta_force_norm": np.zeros((1, TACTILE_POINT_COUNT), dtype=np.float32),
        "grasp_contact_valid": np.zeros((1, TACTILE_POINT_COUNT), dtype=bool),
        "grasp_quality": np.asarray([0.0], dtype=np.float32),
        "funnel_mask": np.zeros((1, FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE), dtype=bool),
        "funnel_mask_valid": np.asarray([False]),
        "funnel_mask_quality": np.asarray([0.0], dtype=np.float32),
        "funnel_mask_hand_offset_xy": np.zeros((1, 2), dtype=np.float32),
    }.items():
        np.save(source_shard / f"{field}.npy", value)
    source_manifest_path = source_shard.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["hand_scene_contract"] = {
        "version": "v1_hand_scene",
        "tactile_point_count": TACTILE_POINT_COUNT,
        "funnel_mask_size": FUNNEL_MASK_SIZE,
    }
    source_manifest_path.write_text(json.dumps(source_manifest))
    _write_state(state_root, max_source_frame=7, dtcr=True)
    manifest = exporter.build_sidecar(
        source_dataset_root=source_root,
        source_version="v1_front_candidate",
        state_root=state_root,
        output_dataset_root=output_root,
        version="v1_hand_scene_dtcr_v1",
        episodes="23",
        generation_commit="test",
    )
    assert manifest["hand_scene_contract"]["dtcr_tactile"] is True
