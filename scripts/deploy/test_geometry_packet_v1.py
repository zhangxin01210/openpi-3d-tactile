"""Parity of live packet construction with the accepted training exporter."""
import numpy as np
import pytest
from scripts.preprocess.export_object_geometry_rgbtrack import geometry_arrays, convert_chunk
from scripts.deploy.run_policy_dry_run import _geometry_from_npz


def test_generic_geometry_uses_training_exporter_and_preserves_invalidity(tmp_path):
    points = {name: np.zeros((512, 3), np.float32) for name in ('part', 'fixture')}
    rows = {'part': [{'valid': False}, {'valid': True, 'T_base_link_object': np.eye(4), 'confidence': .8}],
            'fixture': [{'valid': True, 'T_base_link_object': np.eye(4), 'confidence': 1.}] * 2}
    roles = {'manipulated': 'part', 'target': 'fixture'}
    live = geometry_arrays(0, 0, rows, points, roles, {r: None for r in roles})
    convert_chunk(0, 0, rows, points, roles, {r: None for r in roles}, tmp_path / 'shard', [])
    for field, values in live.items():
        np.testing.assert_array_equal(values, np.load(tmp_path / 'shard' / (field + '.npy')))
    assert not live['manipulated_mask'][0].any()
    assert (live['manipulated_age'][0] == -1).all()
    path = tmp_path / 'frame.npz'
    np.savez(path, **{'object_geometry__' + k: v[0] for k, v in live.items() if k not in ('frame_index', 'episode_index')})
    with np.load(path) as packet:
        result = _geometry_from_npz(packet)
    assert not result['manipulated_mask'].any() and result['target_mask'].all()
    assert 'hand_scene' not in result


def test_incomplete_hand_scene_rejected(tmp_path):
    p = tmp_path / 'bad.npz'
    from scripts.deploy.run_policy_dry_run import OBJECT_FIELDS
    np.savez(p, **{'object_geometry__' + k: np.zeros(1) for k in OBJECT_FIELDS},
             object_geometry__hand_scene__target_relation_valid=np.array(True))
    with np.load(p) as packet, pytest.raises(KeyError, match='incomplete hand_scene'):
        _geometry_from_npz(packet)


def test_button_full_packet_uses_training_fk_and_relation_encoder(tmp_path):
    """Set OPENPI_BUTTON_PERCEPTION_FRAME to check a recorded perception NPZ."""
    import os
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models.spatial_encoders.final_v2_relations import FinalV2RelationEncoder
    from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess
    from openpi.training import config
    from scripts.deploy.run_policy_dry_run import _observation_from_npz

    path = os.environ.get("OPENPI_BUTTON_PERCEPTION_FRAME")
    if path is None:
        path = tmp_path / "frame.npz"
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        geometry = {}
        for i, role in enumerate(("manipulated", "target")):
            geometry.update({
                f"object_geometry__{role}_xyz_m": np.full((512, 3), .1 + i * .05, np.float32),
                f"object_geometry__{role}_mask": np.ones(512, bool),
                f"object_geometry__{role}_source": np.zeros(512, np.int8),
                f"object_geometry__{role}_age": np.zeros(512, np.int16),
                f"object_geometry__{role}_confidence": np.ones(512, np.float32),
            })
        np.savez(path, front_rgb=image, left_rgb=image, right_rgb=image,
                 front_depth_native=np.full((480, 640), 1000, np.uint16),
                 front_depth_color_aligned=np.full((480, 640), 1000, np.uint16),
                 robot_state=np.zeros(1972, np.float32), frame_index=np.array(0), timestamp_s=np.array(0.), **geometry)
    assert Path(path).is_file(), f"Perception NPZ not found: {path}"
    with np.load(path, allow_pickle=False) as source:
        observation = _observation_from_npz(source, prompt="press the button")
    cfg = config.get_config("pi0_xhand_1028_press_button_full_v2")
    assert cfg.model.encoder.object_relation_mode == "final_v2"
    assert cfg.data.object_geometry is not None and cfg.data.spatial is not None
    transform = XHandSpatialOnlinePreprocess(
        camera_roles=cfg.data.spatial.camera_roles, use_visual=cfg.model.use_visual,
        use_tactile=cfg.model.use_tactile, front_calibration_profile=cfg.data.spatial.front_calibration_profile,
        require_object_geometry=True, require_hand_keypoints=True,
    )
    spatial = transform(observation)["spatial"]
    objects, hand = spatial["object_geometry"], spatial["hand_keypoints"]
    assert "hand_scene" not in objects
    assert hand["xyz_m"].shape == (18, 3) and hand["point_mask"].all()
    assert spatial["tactile"]["xyz_m"].shape == (600, 3)
    assert spatial["visual"]["xyz_m"].shape[-1] == 3
    for role in ("manipulated", "target"):
        assert objects[f"{role}_xyz_m"].shape == (512, 3)
        assert np.isfinite(objects[f"{role}_xyz_m"]).all()
        assert objects[f"{role}_mask"].all()
    # The real module computes HO/OO; this test uses random test parameters,
    # never fabricated relation inputs or trained-checkpoint claims.
    encoder = FinalV2RelationEncoder(rngs=nnx.Rngs(0), dim=cfg.model.encoder.token_dim)
    groups = jnp.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    membership = jnp.arange(5)[:, None] == groups[None, :]
    batch = lambda a: jnp.asarray(a)[None, ...]
    ho, ho_mask, oo, oo_mask = encoder(
        batch(objects["manipulated_xyz_m"]), batch(objects["manipulated_mask"]),
        batch(objects["target_xyz_m"]), batch(objects["target_mask"]),
        batch(hand["xyz_m"]), batch(hand["point_mask"]), membership,
    )
    assert ho.shape[1] == oo.shape[1] == 5
    assert np.asarray(ho_mask).all() and np.asarray(oo_mask).all()
    assert np.isfinite(np.asarray(jax.device_get(ho))).all()
    assert np.isfinite(np.asarray(jax.device_get(oo))).all()


def test_peg_q2_hand_scene_fields_are_preserved(tmp_path):
    from scripts.deploy.run_policy_dry_run import OBJECT_FIELDS, HAND_FIELDS, CAUSAL_FIELDS
    arrays = {f'object_geometry__{key}': np.array([i]) for i, key in enumerate(OBJECT_FIELDS)}
    arrays.update({f'object_geometry__hand_scene__{key}': np.array([i])
                   for i, key in enumerate(HAND_FIELDS)})
    arrays.update({f'object_geometry__hand_scene__causal_tactile__{key}': np.array([i])
                   for i, key in enumerate(CAUSAL_FIELDS)})
    path = tmp_path / 'peg.npz'
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as packet:
        parsed = _geometry_from_npz(packet)
    assert set(parsed['hand_scene']) == set(HAND_FIELDS) | {'causal_tactile'}
    for key in HAND_FIELDS:
        np.testing.assert_array_equal(parsed['hand_scene'][key], arrays[f'object_geometry__hand_scene__{key}'])
    assert set(parsed['hand_scene']['causal_tactile']) == set(CAUSAL_FIELDS)
