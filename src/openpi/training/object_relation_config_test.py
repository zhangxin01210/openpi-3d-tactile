"""Regression coverage for the front-only HO/OO training config."""

from __future__ import annotations

from dataclasses import replace

from openpi.training.config import get_config


def test_relational_config_derives_only_from_front_baseline() -> None:
    baseline = get_config("pi0_xhand_spatial_structured_suffix_front")
    relational = get_config("pi0_xhand_spatial_structured_suffix_front_relational")

    assert baseline.data.spatial is not None
    assert relational.data.spatial is not None
    assert relational.data.spatial.front_calibration_profile == baseline.data.spatial.front_calibration_profile
    assert relational.data.spatial.version == baseline.data.spatial.version
    assert relational.data.spatial.camera_roles == baseline.data.spatial.camera_roles
    assert relational.policy_metadata == baseline.policy_metadata
    assert relational.weight_loader == baseline.weight_loader

    # The front baseline object itself remains the reviewed non-relational
    # configuration; the relational config differs only by its encoder switch
    # and optional sidecar join.
    assert not baseline.model.encoder.enable_object_relations
    assert relational.model.encoder.enable_object_relations
    assert replace(relational.model.encoder, enable_object_relations=False) == baseline.model.encoder
    assert relational.data.object_geometry is not None
    assert relational.data.object_geometry.version == "v1"
    assert replace(relational.data, object_geometry=None) == baseline.data

    observation_spec, _ = relational.model.inputs_spec(batch_size=2)
    assert observation_spec.spatial is not None
    assert observation_spec.spatial.object_geometry is not None
    assert observation_spec.spatial.object_geometry.manipulated_xyz_m.shape == (2, 512, 3)
