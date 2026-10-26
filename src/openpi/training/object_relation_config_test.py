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
    assert relational.data.object_geometry.version == baseline.data.spatial.version
    assert replace(relational.data, object_geometry=None) == baseline.data

    observation_spec, _ = relational.model.inputs_spec(batch_size=2)
    assert observation_spec.spatial is not None
    assert observation_spec.spatial.object_geometry is not None
    assert observation_spec.spatial.object_geometry.manipulated_xyz_m.shape == (2, 512, 3)


def test_hand_scene_v0_has_independent_sidecar_and_preserves_xhand_action_contract() -> None:
    baseline = get_config("pi0_xhand_spatial_structured_suffix_front")
    hand_scene = get_config("pi0_xhand_spatial_structured_suffix_hand_scene_v0")

    assert hand_scene.model.encoder.enable_object_relations
    assert hand_scene.model.encoder.object_relation_mode == "hand_scene_v0"
    assert hand_scene.model.encoder.default_token_count == 68
    assert not hand_scene.model.encoder.hand_scene_enable_funnel_mask_encoder
    assert hand_scene.data.object_geometry is not None
    assert hand_scene.data.object_geometry.version == "v1_hand_scene"
    assert hand_scene.data.spatial is not None
    assert hand_scene.data.spatial.version == baseline.data.spatial.version
    assert hand_scene.data.spatial.front_calibration_profile == baseline.data.spatial.front_calibration_profile
    assert type(hand_scene.weight_loader) is type(baseline.weight_loader)
    assert hand_scene.data.use_delta_arm_actions is False
    assert hand_scene.data.delta_hand_actions is False
    assert hand_scene.model.action_dim == 32

    observation_spec, _ = hand_scene.model.inputs_spec(batch_size=2)
    assert observation_spec.spatial is not None
    assert observation_spec.spatial.object_geometry is not None
    sidecar = observation_spec.spatial.object_geometry.hand_scene
    assert sidecar is not None
    assert sidecar.grasp_reference_force_norm.shape == (2, 600)
    assert sidecar.funnel_mask.shape == (2, 32, 32)


def test_m2_and_dtcr_configs_keep_68_tokens_and_use_separate_sidecars() -> None:
    m2 = get_config("pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1")
    dtcr = get_config("pi0_xhand_spatial_dtcr_v1")
    independent = get_config("pi0_xhand_spatial_dtcr_independent_v1")
    assert m2.model.encoder.default_token_count == 68
    assert dtcr.model.encoder.default_token_count == 68
    assert independent.model.encoder.default_token_count == 68
    assert m2.model.encoder.object_relation_mode == "hand_scene_oo_v1"
    assert dtcr.model.encoder.object_relation_mode == "dtcr_v1"
    assert independent.model.encoder.object_relation_mode == "dtcr_independent_v1"
    assert m2.data.object_geometry.version == "v1_hand_scene_q2"
    assert dtcr.data.object_geometry.version == "v1_hand_scene_dtcr_v1"
    assert independent.data.object_geometry.version == "v1_hand_scene_dtcr_v1"
    spec, _ = dtcr.model.inputs_spec(batch_size=2)
    causal = spec.spatial.object_geometry.hand_scene.causal_tactile
    assert causal is not None
    assert causal.current_minus_ema_force_norm.shape == (2, 600)
