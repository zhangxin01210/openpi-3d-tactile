"""Online spatial preprocessing transforms for deployment policies."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import openpi.transforms as transforms
from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial.schema import SpatialObservation


def _default_repo_root() -> Path:
    env_root = os.environ.get("OPENPI_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _camera_name(role: str) -> str:
    return role if role.startswith("cam_") else f"cam_{role}"


def _get_any(data: transforms.DataDict, keys: Sequence[str], *, name: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing {name}; tried keys: {', '.join(keys)}")


def _as_scalar(value: Any, *, dtype: type) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return dtype(array.item())
    return dtype(array.reshape(-1)[0])


def _spatial_observation_to_model_dict(observation: SpatialObservation) -> dict[str, dict[str, np.ndarray]]:
    return {
        "visual": {
            "xyz_m": np.asarray(observation.visual_xyz_m, dtype=np.float32),
            "rgb": np.asarray(observation.visual_rgb, dtype=np.uint8),
            "rgb_valid": np.asarray(observation.visual_rgb_valid, dtype=np.bool_),
            "point_mask": np.ones((observation.visual_count,), dtype=np.bool_),
        },
        "tactile": {
            "xyz_m": np.asarray(observation.tactile_xyz_m, dtype=np.float32),
            "force": np.asarray(observation.tactile_force_base, dtype=np.float32),
            "force_norm": np.asarray(observation.tactile_force_norm, dtype=np.float32),
            "finger_id": np.asarray(observation.finger_id, dtype=np.int32),
            "taxel_id": np.asarray(observation.taxel_id, dtype=np.int32),
            "point_mask": np.ones((observation.tactile_count,), dtype=np.bool_),
        },
    }


@dataclasses.dataclass(frozen=True)
class XHandSpatialOnlinePreprocess(transforms.DataTransformFn):
    """Build the model's spatial input from raw online UR7e + XHand observations.

    The deploy client should send raw observation fields only. This transform runs
    on the policy server and uses the same SpatialPreprocessor as the offline
    derived-dataset pipeline.
    """

    repo_root: str | Path | None = None
    camera_roles: tuple[str, ...] = ("front", "left")

    def __post_init__(self) -> None:
        repo_root = Path(self.repo_root).expanduser().resolve() if self.repo_root is not None else _default_repo_root()
        config = make_baseline_config().with_camera_roles(*self.camera_roles)
        preprocessor = SpatialPreprocessor.from_repo_root(repo_root=repo_root, config=config)
        object.__setattr__(self, "_preprocessor", preprocessor)

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if "spatial" in data:
            return data

        state = np.asarray(
            _get_any(
                data,
                ("observation.state", "observation/state", "state"),
                name="observation.state",
            ),
            dtype=np.float32,
        )
        if state.ndim != 1:
            raise ValueError(f"Online spatial preprocessing expects 1-D observation.state, got {state.shape}")

        rgb_by_role: dict[str, np.ndarray] = {}
        depth_by_role: dict[str, np.ndarray] = {}
        for role in self.camera_roles:
            camera_name = _camera_name(role)
            rgb_by_role[role] = np.asarray(
                _get_any(
                    data,
                    (
                        f"observation.images.{camera_name}",
                        f"observation/{camera_name}_image",
                        camera_name,
                    ),
                    name=f"{camera_name} RGB image",
                ),
                dtype=np.uint8,
            )
            depth_by_role[role] = np.asarray(
                _get_any(
                    data,
                    (
                        f"observation.depths.{camera_name}",
                        f"observation/{camera_name}_depth",
                        f"{camera_name}_depth",
                        f"depth_{camera_name}",
                    ),
                    name=f"{camera_name} depth image",
                )
            )

        frame_index = _as_scalar(data.get("frame_index", data.get("current_action_step", 0)), dtype=int)
        timestamp_s = _as_scalar(data.get("timestamp_s", frame_index), dtype=float)

        spatial_observation = self._preprocessor.preprocess(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            state=state,
            depth_by_role=depth_by_role,
            rgb_by_role=rgb_by_role,
        )

        output = dict(data)
        output["spatial"] = _spatial_observation_to_model_dict(spatial_observation)
        return output
