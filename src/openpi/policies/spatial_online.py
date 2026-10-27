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
from openpi.spatial.tactile_ema import CausalTactileEMA


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


def _tactile_to_model_dict(tactile: Any) -> dict[str, np.ndarray]:
    return {
        "xyz_m": np.asarray(tactile.xyz_m, dtype=np.float32),
        "force": np.asarray(tactile.force_base, dtype=np.float32),
        "force_norm": np.asarray(tactile.force_norm, dtype=np.float32),
        "finger_id": np.asarray(tactile.finger_id, dtype=np.int32),
        "taxel_id": np.asarray(tactile.taxel_id, dtype=np.int32),
        "point_mask": np.ones((tactile.xyz_m.shape[0],), dtype=np.bool_),
    }


def _visual_to_model_dict(visual: Any) -> dict[str, np.ndarray]:
    return {
        "xyz_m": np.asarray(visual.xyz_m, dtype=np.float32),
        "rgb": np.asarray(visual.rgb, dtype=np.uint8),
        "rgb_valid": np.asarray(visual.rgb_valid, dtype=np.bool_),
        "point_mask": np.ones((visual.xyz_m.shape[0],), dtype=np.bool_),
    }


def _validate_object_geometry(value: Any) -> dict[str, Any]:
    """Canonical, strict training-side object-geometry schema validation.

    Do NOT silently drop the top-level object_geometry packet. The input must
    be sent to the spatial encoder as spatial.object_geometry.
    """
    from openpi.object_geometry_dataset.derived import (
        CausalTactileSample,
        HAND_SCENE_FIELDS,
        HandSceneSample,
        ObjectGeometrySample,
    )

    if not isinstance(value, dict):
        raise TypeError("object_geometry must be a dictionary")
    objects = {}
    for name in ("manipulated", "target"):
        for suffix in ("xyz_m", "mask", "source", "age", "confidence"):
            key = f"{name}_{suffix}"
            if key not in value:
                raise KeyError(f"Missing object_geometry.{key}")
            objects[key] = value[key]
    scene = value.get("hand_scene")
    typed_scene = None
    if scene is not None:
        if not isinstance(scene, dict):
            raise TypeError("object_geometry.hand_scene must be a dictionary")
        missing = set(HAND_SCENE_FIELDS) - set(scene)
        if missing:
            raise KeyError(f"Missing hand_scene fields: {sorted(missing)}")
        causal = scene.get("causal_tactile")
        if causal is not None:
            if not isinstance(causal, dict):
                raise TypeError("hand_scene.causal_tactile must be a dictionary")
            causal = CausalTactileSample(**causal)
        typed_scene = HandSceneSample(
            **{name: scene[name] for name in HAND_SCENE_FIELDS},
            causal_tactile=causal,
        )
    typed = ObjectGeometrySample(
        episode_index=0, frame_index=0, hand_scene=typed_scene, **objects
    )
    result = typed.as_dict()
    result.pop("episode_index")
    result.pop("frame_index")
    return result


@dataclasses.dataclass(frozen=True)
class XHandSpatialOnlineRepack(transforms.DataTransformFn):
    """Repack the raw deployment observation for XHandInputs at inference time."""

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        images = {
            camera_name: _get_any(
                data,
                (f"observation.images.{camera_name}", f"observation/{camera_name}_image"),
                name=f"{camera_name} RGB image",
            )
            for camera_name in ("cam_front", "cam_left", "cam_right")
        }
        output = {
            "images": images,
            "state": _get_any(data, ("observation.state", "observation/state"), name="observation.state"),
        }
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class XHandSpatialOnlinePreprocess(transforms.DataTransformFn):
    """Build the model's spatial input from raw online UR7e + XHand observations.

    The deploy client should send raw observation fields only. This transform runs
    on the policy server and uses the same SpatialPreprocessor as the offline
    derived-dataset pipeline.
    """

    repo_root: str | Path | None = None
    camera_roles: tuple[str, ...] = ("front", "left")
    use_visual: bool = True
    use_tactile: bool = True
    front_calibration_profile: str | None = None
    tactile_ema_tau_s: float = 1.0
    require_object_geometry: bool = False
    require_hand_scene: bool = False

    def __post_init__(self) -> None:
        if not (self.use_visual or self.use_tactile):
            raise ValueError("At least one spatial modality must be enabled")
        repo_root = Path(self.repo_root).expanduser().resolve() if self.repo_root is not None else _default_repo_root()
        config = make_baseline_config().with_camera_roles(*self.camera_roles)
        if self.front_calibration_profile is not None:
            from openpi.spatial.calibration_profile import apply_front_calibration_profile

            config = apply_front_calibration_profile(config, repo_root / self.front_calibration_profile)
        preprocessor = SpatialPreprocessor.from_repo_root(repo_root=repo_root, config=config)
        object.__setattr__(self, "_preprocessor", preprocessor)
        object.__setattr__(self, "_tactile_ema", None)
        object.__setattr__(self, "_ema_episode", None)

    def reset_tactile_ema(self) -> None:
        """Reset causal tactile state between episodes when no episode key is sent."""

        if self._tactile_ema is not None:
            self._tactile_ema.reset()
        object.__setattr__(self, "_ema_episode", None)

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        incoming_geometry = data.get("object_geometry")
        if "spatial" in data:
            existing = data["spatial"]
            if not isinstance(existing, dict):
                raise TypeError("spatial must be a mapping")
            if incoming_geometry is None:
                incoming_geometry = existing.get("object_geometry")
            if self.require_object_geometry and incoming_geometry is None:
                raise ValueError("Required online object_geometry missing; refusing geometry-free Pi0 inference")
            if incoming_geometry is not None:
                geom = _validate_object_geometry(incoming_geometry)
                if self.require_hand_scene and geom.get("hand_scene") is None:
                    raise ValueError("Hand-scene model requires object_geometry.hand_scene")
                result = dict(data)
                result["spatial"] = {**existing, "object_geometry": geom}
                return result
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
        for role in self.camera_roles if self.use_visual else ():
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
        episode_index = data.get("episode_index")
        if episode_index is not None:
            episode_index = _as_scalar(episode_index, dtype=int)
            if self._ema_episode is not None and episode_index != self._ema_episode:
                self._tactile_ema.reset()
            object.__setattr__(self, "_ema_episode", episode_index)

        output = dict(data)
        if self.use_visual and self.use_tactile:
            spatial_observation = self._preprocessor.preprocess(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                state=state,
                depth_by_role=depth_by_role,
                rgb_by_role=rgb_by_role,
            )
            spatial = _spatial_observation_to_model_dict(spatial_observation)
            ema_force_norm = spatial_observation.tactile_force_norm
        elif self.use_visual:
            visual = self._preprocessor.preprocess_visual(
                depth_by_role=depth_by_role, rgb_by_role=rgb_by_role
            )
            spatial = {"visual": _visual_to_model_dict(visual)}
        else:
            tactile = self._preprocessor.preprocess_tactile(state)
            spatial = {"tactile": _tactile_to_model_dict(tactile)}
            ema_force_norm = tactile.force_norm
        if self.use_tactile:
            if self._tactile_ema is None or self._tactile_ema.taxel_count != len(ema_force_norm):
                object.__setattr__(
                    self,
                    "_tactile_ema",
                    CausalTactileEMA(tau_s=self.tactile_ema_tau_s, taxel_count=len(ema_force_norm)),
                )
            ema = self._tactile_ema.update(timestamp_s=timestamp_s, force_norm=ema_force_norm)
            spatial["causal_tactile"] = {
                "ema_before_force_norm": ema.ema_before_force_norm,
                "ema_after_force_norm": ema.ema_after_force_norm,
                "current_minus_ema_force_norm": ema.current_minus_ema_force_norm,
                "current_minus_previous_force_norm": ema.current_minus_previous_force_norm,
                "reference_valid": ema.reference_valid,
                "previous_valid": ema.previous_valid,
                "sample_valid": ema.sample_valid,
                "alpha": np.asarray(ema.alpha, dtype=np.float32),
            }
        if self.require_object_geometry and incoming_geometry is None:
            raise ValueError("Required online object_geometry missing; refusing geometry-free Pi0 inference")
        if incoming_geometry is not None:
            geometry = _validate_object_geometry(incoming_geometry)
            if self.require_hand_scene and geometry.get("hand_scene") is None:
                raise ValueError("Hand-scene model requires object_geometry.hand_scene")
            spatial["object_geometry"] = geometry
        output["spatial"] = spatial
        return output
