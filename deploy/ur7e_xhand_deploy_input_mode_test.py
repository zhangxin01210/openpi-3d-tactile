import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _deployment_client():
    path = Path(__file__).with_name("ur7e_xhand_deploy_3d_tactile_multi.py")
    spec = importlib.util.spec_from_file_location("ur7e_xhand_deploy_client_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hand_object_only_metadata_selects_spatial_mode():
    client = _deployment_client()
    args = SimpleNamespace(
        spatial_cameras=None,
        scene_position=None,
        realsense_width=640,
        realsense_height=480,
    )

    metadata = {
        "spatial_camera_roles": [],
        "spatial_use_visual": False,
        "spatial_use_tactile": False,
        "rgb_only_baseline": False,
        "object_geometry_required": True,
        "hand_scene_history_required": True,
        "hand_scene_position_required": False,
        "hand_scene_camera_roles": ["front"],
    }
    client.configure_input_mode_from_metadata(args, metadata)
    client.validate_geometry_mode(args, metadata)

    assert args.policy_input_mode == "spatial"
    assert args.spatial_cameras == ("front",)
    assert args.object_geometry_required
    assert not args.send_native_input_aliases


def test_rgb_baseline_metadata_keeps_native_aliases():
    client = _deployment_client()
    args = SimpleNamespace(spatial_cameras=None)

    client.configure_input_mode_from_metadata(
        args,
        {
            "spatial_camera_roles": [],
            "spatial_use_visual": False,
            "spatial_use_tactile": False,
            "rgb_only_baseline": True,
            "object_geometry_required": False,
        },
    )

    assert args.policy_input_mode == "rgb"
    assert args.send_native_input_aliases


def test_older_geometry_metadata_still_enables_front_depth():
    client = _deployment_client()
    args = SimpleNamespace(spatial_cameras=None)

    client.configure_input_mode_from_metadata(
        args,
        {
            "spatial_camera_roles": [],
            "spatial_use_visual": False,
            "spatial_use_tactile": False,
            "object_geometry_required": True,
            "hand_scene_history_required": True,
        },
    )

    assert args.spatial_cameras == ("front",)
