"""ContactWorld USB observation/action mapping, isolated from real-robot policies."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image


def depth_rgb(depth: np.ndarray) -> np.ndarray:
    """Encode source [0,1] tactile depth into the ordinary image slot."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (320, 240):
        raise ValueError(f"Expected 320x240 tactile depth, got {depth.shape}")
    gray = np.rint(np.clip(depth, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


@dataclasses.dataclass(frozen=True)
class ContactWorldInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    tactile_image: str = "none"  # none/rgb/depth

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (18,):
            raise ValueError(f"Expected 18D joint position/velocity, got {state.shape}")
        front = rgb(data["observation/front_rgb"])
        wrist = rgb(data["observation/wrist_rgb"])
        if self.tactile_image == "rgb":
            third = rgb(data["observation/tactile_rgb"])
        elif self.tactile_image == "depth":
            third = depth_rgb(data["observation/tactile_depth"])
        elif self.tactile_image == "none":
            third = np.zeros_like(front)
        else:
            raise ValueError(self.tactile_image)
        result = {
            "state": state,
            "image": {"base_0_rgb": front, "left_wrist_0_rgb": wrist,
                      "right_wrist_0_rgb": third},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.bool_(self.tactile_image != "none" or
                                                        self.model_type == _model.ModelType.PI0_FAST)},
            "prompt": data.get("prompt", "Insert the USB plug into the socket."),
        }
        if "actions" in data:
            action = np.asarray(data["actions"], dtype=np.float32)
            if action.shape[-1] != 6:
                raise ValueError(f"Expected 6D source action, got {action.shape}")
            result["actions"] = action
        return result


@dataclasses.dataclass(frozen=True)
class ContactWorldOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :6]}
