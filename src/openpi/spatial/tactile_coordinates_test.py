# ruff: noqa: SLF001
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess
from openpi.spatial.tactile import extract_raw_tactile_force
from openpi.spatial.tactile import map_force_axes_to_link
from openpi.spatial_dataset.adapter import SpatialAugmentedDataset


def test_offline_online_local_coordinates_and_rigid_knn():
    root = Path(__file__).resolve().parents[3]
    online = XHandSpatialOnlinePreprocess(repo_root=root, camera_roles=("front",), use_visual=False,
                                         tactile_coordinate_frame="finger_link2")
    rng = np.random.default_rng(17)
    state = np.zeros(1972, np.float32)
    state[52:] = rng.normal(size=1920).astype(np.float32)
    base = online._preprocessor.preprocess_tactile(state)
    result = online({"observation.state": state})["spatial"]["tactile"]
    raw, _ = extract_raw_tactile_force(state, config=online._preprocessor.config.tactile)
    np.testing.assert_array_equal(result["xyz_m"], online._preprocessor.taxel_geometry.xyz_link_m.reshape(600, 3))
    np.testing.assert_array_equal(result["force"], map_force_axes_to_link(raw, config=online._preprocessor.config.tactile).reshape(600, 3))
    np.testing.assert_allclose(result["force_norm"], np.linalg.norm(result["force"], axis=-1), atol=1e-4, rtol=1e-5)
    np.testing.assert_array_equal(result["finger_id"], base.finger_id)
    np.testing.assert_array_equal(result["taxel_id"], base.taxel_id)
    for finger in range(5):
        sl = slice(120 * finger, 120 * (finger + 1))
        def distances(points):
            return np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        local_dist = distances(result["xyz_m"][sl])
        base_dist = distances(base.xyz_m[sl])
        np.testing.assert_allclose(base_dist, local_dist, atol=1e-6)
        # Symmetric layouts can have ties; compare selected neighbor distances.
        base_order = np.argsort(base_dist, axis=-1)
        local_order = np.argsort(local_dist, axis=-1)
        np.testing.assert_allclose(np.sort(base_dist, axis=-1)[:, :8], np.sort(local_dist, axis=-1)[:, :8], atol=1e-6)
        # Compare the exact K=8 adjacency where the eighth neighbor has no tie.
        sorted_local = np.sort(local_dist, axis=-1)
        unambiguous = sorted_local[:, 8] - sorted_local[:, 7] > 1e-6
        assert unambiguous.any()
        for row in np.flatnonzero(unambiguous):
            assert set(base_order[row, :8]) == set(local_order[row, :8])
    moved = state.copy()
    moved[0] = 0.5
    base_moved = online._preprocessor.preprocess_tactile(moved)
    local_moved = online({"observation.state": moved})["spatial"]["tactile"]
    assert not np.allclose(base.xyz_m, base_moved.xyz_m)
    np.testing.assert_array_equal(result["xyz_m"], local_moved["xyz_m"])
    sample = SimpleNamespace(visual_xyz_m=np.zeros((4, 3), np.float32), visual_rgb=np.zeros((4, 3), np.uint8),
        visual_rgb_valid=np.ones(4, bool), tactile_xyz_m=base.xyz_m, tactile_force_base=base.force_base,
        tactile_force_norm=base.force_norm, finger_id=base.finger_id, taxel_id=base.taxel_id)
    class Sidecar:
        def get(self, *, episode_index, frame_index):
            assert (episode_index, frame_index) == (2, 3)
            return sample
    dataset = SpatialAugmentedDataset([{"observation.state": state, "episode_index": 2, "frame_index": 3}],
        Sidecar(), use_visual=False, tactile_coordinate_frame="finger_link2")
    offline = dataset[0]["spatial"]["tactile"]
    for key in result:
        np.testing.assert_array_equal(offline[key], result[key])
    bad = dict(offline, force_norm=np.zeros(600, np.float32))
    with pytest.raises(ValueError, match="force"):
        online._tactile_adapter(state, bad)
