import numpy as np

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
