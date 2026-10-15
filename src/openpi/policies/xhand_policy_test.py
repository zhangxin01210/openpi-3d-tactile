import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.policies import xhand_policy


def test_xhand_inputs_accepts_deployment_flat_keys():
    images = {
        "cam_front": np.full((8, 10, 3), 11, dtype=np.uint8),
        "cam_left": np.full((8, 10, 3), 22, dtype=np.uint8),
        "cam_right": np.full((8, 10, 3), 33, dtype=np.uint8),
    }
    state = np.arange(1972, dtype=np.float32)
    prompt = "press the button"
    transform = xhand_policy.XHandInputs(model_type=model_lib.ModelType.PI0)

    canonical = transform({"images": images, "state": state, "prompt": prompt})
    deployment = transform(
        {
            "observation.images.cam_front": images["cam_front"],
            "observation.images.cam_left": images["cam_left"],
            "observation.images.cam_right": images["cam_right"],
            "observation.state": state,
            "prompt": prompt,
        }
    )

    jax.tree.map(np.testing.assert_array_equal, deployment, canonical)
