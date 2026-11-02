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
