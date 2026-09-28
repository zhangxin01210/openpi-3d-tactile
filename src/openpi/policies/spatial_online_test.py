import numpy as np
from types import SimpleNamespace

from openpi.models import model
from openpi.policies import spatial_online
from openpi.policies import xhand_policy


def test_online_repack_matches_xhand_inference_input():
    state = np.arange(1972, dtype=np.float32)
    images = {
        camera_name: np.full((8, 12, 3), index, dtype=np.uint8)
        for index, camera_name in enumerate(("cam_front", "cam_left", "cam_right"), start=1)
    }
    raw_observation = {
        "observation.state": state,
        "prompt": "press the button",
        **{f"observation.images.{name}": image for name, image in images.items()},
    }

    repacked = spatial_online.XHandSpatialOnlineRepack()(raw_observation)
    inputs = xhand_policy.XHandInputs(model_type=model.ModelType.PI0)(repacked)

    np.testing.assert_array_equal(inputs["state"], np.concatenate((state[:6], state[28:52:2])))
    np.testing.assert_array_equal(inputs["image"]["base_0_rgb"], images["cam_front"])
    np.testing.assert_array_equal(inputs["image"]["left_wrist_0_rgb"], images["cam_left"])
    np.testing.assert_array_equal(inputs["image"]["right_wrist_0_rgb"], images["cam_right"])
    assert inputs["prompt"] == "press the button"
    assert "actions" not in inputs


def test_front_only_preprocess_reads_only_front_depth(monkeypatch):
    calls = []

    class FakePreprocessor:
        def preprocess_visual(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                xyz_m=np.zeros((4, 3), np.float32),
                rgb=np.zeros((4, 3), np.uint8),
                rgb_valid=np.ones(4, bool),
            )

    monkeypatch.setattr(spatial_online.SpatialPreprocessor, "from_repo_root", lambda **kwargs: FakePreprocessor())
    transform = spatial_online.XHandSpatialOnlinePreprocess(camera_roles=("front",), use_tactile=False)
    output = transform({
        "observation.state": np.zeros(1972, np.float32),
        "observation.images.cam_front": np.zeros((8, 12, 3), np.uint8),
        "observation.depths.cam_front": np.zeros((8, 12), np.uint16),
    })
    assert tuple(calls[0]["depth_by_role"]) == ("front",)
    assert "visual" in output["spatial"]
    assert "tactile" not in output["spatial"]


def test_tactile_only_preprocess_needs_no_depth(monkeypatch):
    class FakePreprocessor:
        def preprocess_tactile(self, state):
            return SimpleNamespace(
                xyz_m=np.zeros((2, 3), np.float32),
                force_base=np.zeros((2, 3), np.float32),
                force_norm=np.zeros(2, np.float32),
                finger_id=np.zeros(2, np.int8),
                taxel_id=np.arange(2, dtype=np.int16),
            )

    monkeypatch.setattr(spatial_online.SpatialPreprocessor, "from_repo_root", lambda **kwargs: FakePreprocessor())
    transform = spatial_online.XHandSpatialOnlinePreprocess(camera_roles=("front",), use_visual=False)
    output = transform({"observation.state": np.zeros(1972, np.float32)})
    assert tuple(output["spatial"]) == ("tactile",)
    assert output["spatial"]["tactile"]["xyz_m"].shape == (2, 3)
