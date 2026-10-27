from __future__ import annotations

import numpy as np
import pytest

from openpi.object_geometry.production import PoseObservation
from openpi.object_geometry.production import build_episode_records
from openpi.object_geometry.production import deterministic_surface_points
from openpi.policies.hand_scene_online import OnlineObjectGeometryBuilder
from openpi.policies.hand_scene_online import OnlinePose
from openpi.policies.spatial_online import _validate_object_geometry


def _surface_points(seed: int) -> np.ndarray:
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return deterministic_surface_points(vertices, faces, count=512, seed=seed)


def _pose(x: float, *, valid: bool = True, status: str = "observed", source: str = "rgbtrack_observed") -> OnlinePose:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return OnlinePose(matrix, valid, 0.8 if valid else 0.0, status, source, valid)


def _builder(**kwargs) -> OnlineObjectGeometryBuilder:
    return OnlineObjectGeometryBuilder(
        funnel_surface_points_m=_surface_points(0),
        flask_surface_points_m=_surface_points(1),
        **kwargs,
    )


def test_online_mesh_cloud_matches_training_producer_and_reuses_sampled_points() -> None:
    builder = _builder()
    funnel_pose = _pose(0.25)
    flask_pose = _pose(0.75)
    first = builder.build_frame(
        frame_index=0,
        timestamp_s=0.0,
        funnel_pose=funnel_pose,
        target_flask_pose=flask_pose,
        tactile_force_norm=None,
        funnel_mask=None,
    )
    second = builder.build_frame(
        frame_index=1,
        timestamp_s=0.1,
        funnel_pose=funnel_pose,
        target_flask_pose=flask_pose,
        tactile_force_norm=None,
        funnel_mask=None,
    )

    train_pose_run = {
        "funnel": [funnel_pose.as_training_observation(0)],
        "target_flask": [flask_pose.as_training_observation(0)],
    }
    offline = build_episode_records(
        episode=0,
        pose_run=train_pose_run,
        funnel_points=builder._funnel_points,
        flask_points=builder._flask_points,
    )[0]
    np.testing.assert_array_equal(first.manipulated_xyz_m, offline["manipulated_xyz_m"])
    np.testing.assert_array_equal(first.target_xyz_m, offline["target_xyz_m"])
    np.testing.assert_array_equal(first.manipulated_mask, offline["manipulated_mask"])
    np.testing.assert_array_equal(first.target_confidence, offline["target_confidence"])
    np.testing.assert_allclose(second.manipulated_xyz_m, first.manipulated_xyz_m)
    assert first.manipulated_xyz_m.shape == (512, 3)
    assert first.hand_scene is not None
    assert bool(first.hand_scene.manipulated_relation_valid)


def test_invalid_or_predicted_funnel_pose_uses_zero_cloud_and_invalid_relations() -> None:
    builder = _builder()
    sample = builder.build_frame(
        frame_index=0,
        timestamp_s=0.0,
        funnel_pose=_pose(0.25, valid=True),
        target_flask_pose=_pose(0.75, valid=True),
        tactile_force_norm=np.ones(600, dtype=np.float32),
        funnel_mask=None,
    )
    lost = builder.build_frame(
        frame_index=1,
        timestamp_s=0.1,
        funnel_pose=_pose(5.0, valid=True, status="occluded_causal_prediction"),
        target_flask_pose=_pose(0.75, valid=True),
        tactile_force_norm=np.ones(600, dtype=np.float32) * 1.1,
        funnel_mask=None,
    )
    assert sample.hand_scene is not None and lost.hand_scene is not None
    assert not lost.manipulated_mask.any()
    assert not lost.manipulated_xyz_m.any()
    assert not bool(lost.hand_scene.manipulated_relation_valid)
    assert float(lost.hand_scene.manipulated_geometry_quality) == 0.0
    np.testing.assert_allclose(
        lost.hand_scene.grasp_short_delta_force_norm,
        np.full(600, 0.1, np.float32),
        atol=3e-8,
        rtol=0,
    )


def test_static_target_anchor_requires_provenance_and_matches_sidecar_frame_semantics() -> None:
    anchor = np.eye(4, dtype=np.float64)
    anchor[:3, 3] = [0.4, 0.2, 0.1]
    with pytest.raises(ValueError, match="provenance"):
        _builder(static_target_anchor=anchor)

    builder = _builder(static_target_anchor=anchor, static_target_anchor_provenance="human_reviewed")
    observed = builder.build_frame(
        frame_index=0,
        timestamp_s=0.0,
        funnel_pose=None,
        target_flask_pose=_pose(0.4),
        tactile_force_norm=np.zeros(600, dtype=np.float32),
        funnel_mask=None,
    )
    assert np.all(observed.target_source == 0)
    prior = builder.build_frame(
        frame_index=1,
        timestamp_s=1.0,
        funnel_pose=None,
        target_flask_pose=None,
        tactile_force_norm=np.zeros(600, dtype=np.float32),
        funnel_mask=None,
    )
    assert prior.hand_scene is not None
    assert bool(prior.hand_scene.target_relation_valid)
    assert np.all(prior.target_source == 1)
    np.testing.assert_allclose(prior.target_xyz_m.mean(axis=0), anchor[:3, 3], atol=0.8)


def test_dtcr_tactile_is_causal_and_gap_invalidates_short_delta() -> None:
    builder = _builder(tactile_ema_tau_s=1.0)
    first_force = np.linspace(0.1, 0.7, 600, dtype=np.float32)
    second_force = first_force + 0.25
    kwargs = dict(funnel_pose=None, target_flask_pose=None, funnel_mask=None)
    first = builder.build_frame(frame_index=0, timestamp_s=0.0, tactile_force_norm=first_force, **kwargs)
    second = builder.build_frame(frame_index=1, timestamp_s=0.2, tactile_force_norm=second_force, **kwargs)
    gap = builder.build_frame(frame_index=3, timestamp_s=0.6, tactile_force_norm=second_force + 0.1, **kwargs)
    after_gap = builder.build_frame(
        frame_index=4, timestamp_s=0.8, tactile_force_norm=second_force + 0.2, **kwargs
    )
    assert first.hand_scene and second.hand_scene and gap.hand_scene and after_gap.hand_scene
    assert first.hand_scene.causal_tactile is not None
    assert second.hand_scene.causal_tactile is not None
    assert gap.hand_scene.causal_tactile is not None
    np.testing.assert_array_equal(first.hand_scene.causal_tactile.current_minus_ema_force_norm, np.zeros(600))
    np.testing.assert_allclose(
        second.hand_scene.causal_tactile.current_minus_previous_force_norm, 0.25, atol=3e-8, rtol=0
    )
    assert not gap.hand_scene.causal_tactile.sample_valid.any()
    assert not gap.hand_scene.grasp_contact_valid.any()
    assert not after_gap.hand_scene.causal_tactile.previous_valid.any()
    np.testing.assert_array_equal(
        after_gap.hand_scene.causal_tactile.current_minus_previous_force_norm, np.zeros(600)
    )


def test_object_geometry_dictionary_roundtrips_the_encoder_input_contract() -> None:
    builder = _builder()
    sample = builder.build_frame(
        frame_index=4,
        timestamp_s=0.4,
        funnel_pose=_pose(0.0),
        target_flask_pose=_pose(1.0),
        tactile_force_norm=np.zeros(600, dtype=np.float32),
        funnel_mask=np.ones((480, 640), dtype=bool),
        funnel_mask_hand_offset_xy=np.zeros(2, dtype=np.float32),
    )
    data = sample.as_dict()
    data.pop("episode_index")
    data.pop("frame_index")
    normalized = _validate_object_geometry(data)
    assert normalized["manipulated_xyz_m"].shape == (512, 3)
    assert normalized["target_xyz_m"].shape == (512, 3)
    assert normalized["hand_scene"]["funnel_mask"].shape == (32, 32)
    assert normalized["hand_scene"]["causal_tactile"]["ema_after_force_norm"].shape == (600,)


def test_missing_hand_projection_keeps_funnel_mask_invalid() -> None:
    builder = _builder()
    sample = builder.build_frame(
        frame_index=0,
        timestamp_s=0.0,
        funnel_pose=None,
        target_flask_pose=None,
        tactile_force_norm=None,
        funnel_mask=np.ones((480, 640), dtype=bool),
        funnel_mask_hand_offset_xy=None,
    )
    assert sample.hand_scene is not None
    assert not bool(sample.hand_scene.funnel_mask_valid)
    assert not sample.hand_scene.funnel_mask.any()
