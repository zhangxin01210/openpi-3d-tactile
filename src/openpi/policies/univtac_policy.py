"""RGB and joint contract for UniVTAC Franka insert_hole clean baselines.

The tactile RGB streams and world point clouds require separate modality
adapters. This module deliberately consumes only head/wrist RGB and joints.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image, got {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(np.clip(image * 255.0, 0, 255))
    return image


def _tactile_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack both GelSight RGB views in the available third image slot."""
    import cv2

    tiles = []
    for image in (_rgb(left), _rgb(right)):
        height, width = image.shape[:2]
        scale = min(224 / width, 112 / height)
        resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        tile = np.zeros((112, 224, 3), dtype=np.uint8)
        y = (112 - resized.shape[0]) // 2
        x = (224 - resized.shape[1]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        tiles.append(tile)
    return np.concatenate(tiles, axis=0)


@dataclasses.dataclass(frozen=True)
class UniVTACInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    use_tactile_rgb: bool = False

    def __call__(self, data: dict) -> dict:
        head = _rgb(data["observation/head_rgb"])
        wrist = _rgb(data["observation/wrist_rgb"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (9,):
            raise ValueError(f"Expected 9D Franka joint state, got {state.shape}")
        tactile_pair = (_tactile_pair(data["observation/left_tactile_rgb"],
                                      data["observation/right_tactile_rgb"])
                        if self.use_tactile_rgb else np.zeros_like(head))
        result = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": wrist,
                # This Pi0 image slot carries both GelSight images when enabled.
                "right_wrist_0_rgb": tactile_pair,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.use_tactile_rgb or self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "prompt": data.get("prompt", "insert the peg into the hole"),
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return result


@dataclasses.dataclass(frozen=True)
class UniVTACOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :9]}
