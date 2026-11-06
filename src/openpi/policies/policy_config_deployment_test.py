import pytest
from openpi.training import config
from openpi.policies.policy_config import _policy_metadata


@pytest.mark.parametrize('name,visual,tactile,roles', [
    ('pi0_xhand_1028_press_button_rgb_v2', False, False, []),
    ('pi05_xhand_1030_yellow_socket_rgb_v2', False, False, []),
    ('pi0_xhand_1018_peg_all_st_v2', True, True, ['front']),
    ('pi0_xhand_1028_press_button_full_v2', True, True, ['front']),
])
def test_xhand_metadata_follows_selected_training_config(name, visual, tactile, roles):
    cfg = config.get_config(name)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    meta = _policy_metadata(cfg, data)
    assert meta['spatial_camera_roles'] == roles
    assert meta['spatial_use_visual'] is visual
    assert meta['spatial_use_tactile'] is tactile
    assert meta['rgb_only_baseline'] is (not visual and not tactile)
    assert meta['dataset'] == cfg.policy_metadata['dataset']


def test_hand_object_only_policy_is_not_advertised_as_rgb_baseline():
    cfg = config.get_config("pi0_xhand_ho_v3/press_the_button")
    meta = _policy_metadata(cfg, cfg.data.create(cfg.assets_dirs, cfg.model))

    assert not meta["spatial_use_visual"]
    assert not meta["spatial_use_tactile"]
    assert meta["object_geometry_required"]
    assert not meta["rgb_only_baseline"]
