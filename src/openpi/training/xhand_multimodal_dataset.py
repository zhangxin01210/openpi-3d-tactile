"""Read-only UR7e/XHand training-data join for the approved v1 sidecars.

This is DATA ONLY: no relation features, no Pi0 changes and no normalization
of metric spatial/object/hand coordinates.

Given an existing OpenPI `DataConfig` with:
  - repo_id pointing to one LeRobot dataset,
  - spatial pointing to the SAME dataset's spatial/v1 (front only), and
  - norm_stats loaded from that task's independent norm_stats.json,

call `create_xhand_multimodal_dataset(data_config, model_config, action_horizon=50)`.
The returned torch-style dataset yields normal OpenPI model-transformed
state/actions/images plus preserved:
  spatial, object_geometry, hand_keypoints.

`object_geometry`/`hand_keypoints` are current-frame observations, never future
or ground truth action hints. Validity masks are preserved, not discarded.
For now models must explicitly consume these keys; `Observation.from_dict`
does not automatically turn them into tokens.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np

from openpi import transforms
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
from openpi.training import data_loader


def _scalar_int(v: Any, *, name: str) -> int:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    a = np.asarray(v)
    if a.size != 1 or not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{name}: expected numeric scalar, got {a.shape} {a.dtype}")
    i = int(a.reshape(()))
    if not np.isfinite(a).all() or float(a.reshape(())) != float(i):
        raise ValueError(f"{name}: not an integral finite value: {a}")
    return i


class XHandGeometryHandJoin:
    """Attach two immutable v1 sidecars to an existing Spatial-augmented sample.

    Low-memory lazy access: geometry through the repository's mmap reader,
    hand FK through a per-episode bounded LRU cache. No images are re-decoded.
    """

    def __init__(self, base_dataset, dataset_root: str | Path, *,
                 object_version: str = "v1", hand_version: str = "v1",
                 max_hand_episodes: int = 2):
        self.base_dataset = base_dataset
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if max_hand_episodes < 1:
            raise ValueError("max_hand_episodes must be positive")
        self.max_hand_episodes = int(max_hand_episodes)
        self.objects = ObjectGeometryDerivedDataset(self.dataset_root, version=object_version)
        self.hand_root = self.dataset_root / "hand_keypoints" / hand_version
        self.hand_manifest = json.loads((self.hand_root / "manifest.json").read_text(encoding="utf-8"))
        h = self.hand_manifest
        if (h.get("status") != "complete" or h.get("keypoint_count") != 18 or
            h.get("coordinate_frame") != "base_link" or h.get("unit") != "m" or
            h.get("spatial_version") != "v1" or h.get("object_geometry_version") != object_version):
            raise ValueError(f"{self.hand_root}: wrong hand FK manifest contract")
        records = h["episodes"]
        self.hand_records = {int(r["episode_index"]): r for r in records}
        if len(self.hand_records) != len(records):
            raise ValueError("Duplicate episode in hand manifest")
        if set(self.hand_records) != set(self.objects.episode_ids):
            raise ValueError("Hand and object geometry episode sets differ")
        if sum(int(r["frames"]) for r in records) != len(self.objects):
            raise ValueError("Hand and object geometry frame counts differ")
        self._hand_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _hand_xyz(self, episode: int, frame: int) -> np.ndarray:
        if episode not in self._hand_cache:
            rec = self.hand_records[episode]
            path = self.hand_root / rec["file"]
            with np.load(path, allow_pickle=False) as z:
                frame_ids = np.array(z["frame_index"], dtype=np.int64, copy=True)
                xyz = np.array(z["xyz_m"], dtype=np.float32, copy=True)
            expected = int(rec["frames"])
            if (xyz.shape != (expected, 18, 3) or
                    frame_ids.shape != (expected,) or
                    not np.array_equal(frame_ids, np.arange(expected)) or
                    not np.isfinite(xyz).all()):
                raise ValueError(f"Invalid FK episode file: {path}")
            self._hand_cache[episode] = (frame_ids, xyz)
            while len(self._hand_cache) > self.max_hand_episodes:
                self._hand_cache.popitem(last=False)
        self._hand_cache.move_to_end(episode)
        frames, xyz = self._hand_cache[episode]
        if frame < 0 or frame >= len(frames) or int(frames[frame]) != frame:
            raise KeyError(f"No hand keypoints for episode={episode}, frame={frame}")
        return np.array(xyz[frame], copy=True)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        if "spatial" not in item:
            raise KeyError("Base dataset must be spatial-augmented")
        if "object_geometry" in item or "hand_keypoints" in item:
            raise KeyError("Refusing to overwrite existing sidecar fields")
        episode = _scalar_int(item["episode_index"], name="episode_index")
        frame = _scalar_int(item["frame_index"], name="frame_index")
        record = self.objects.get(episode_index=episode, frame_index=frame)
        objects = {}
        for role in ("manipulated", "target"):
            objects[role] = {
                "xyz_m": np.array(getattr(record, f"{role}_xyz_m"), dtype=np.float32, copy=True),
                "mask": np.array(getattr(record, f"{role}_mask"), dtype=np.bool_, copy=True),
                "source": np.array(getattr(record, f"{role}_source"), dtype=np.int8, copy=True),
                "age": np.array(getattr(record, f"{role}_age"), dtype=np.int16, copy=True),
                "confidence": np.array(getattr(record, f"{role}_confidence"), dtype=np.float32, copy=True),
            }
        item["object_geometry"] = objects
        item["hand_keypoints"] = {
            "xyz_m": self._hand_xyz(episode, frame),
            "point_mask": np.ones((18,), dtype=np.bool_),
        }
        return item


def create_xhand_multimodal_dataset(data_config, model_config, *,
                                    action_horizon: int = 50,
                                    object_version: str = "v1",
                                    hand_version: str = "v1",
                                    max_hand_episodes: int = 2):
    """Reusable data-only factory, compatible with OpenPI TorchDataLoader.

    `data_config` must be configured per task, including correct norm_stats.
    This does not edit the standard data loader or model; Codex can call this
    factory from a training integration once it chooses how to use the fields.
    """
    if data_config.spatial is None:
        raise ValueError("SpatialDataConfig is required for multimodal v1")
    if (Path(data_config.repo_id).expanduser().resolve() !=
            Path(data_config.spatial.dataset_root).expanduser().resolve()):
        raise ValueError("LeRobot root and Spatial sidecar root must be identical")
    if data_config.norm_stats is None or set(data_config.norm_stats) != {"state", "actions"}:
        raise ValueError("Expected task-specific state/actions norm_stats")
    if data_config.use_quantile_norm:
        raise ValueError("Expected PI0 mean/std normalization (not quantiles)")
    if not data_config.asset_id:
        raise ValueError("Missing task-specific asset_id")
    for field in ("state", "actions"):
        if np.asarray(data_config.norm_stats[field].mean).shape != (18,):
            raise ValueError(f"{field} normalization must be 18-D")
    base = data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    if len(base) < 1:
        raise ValueError("Empty LeRobot dataset")
    joined = XHandGeometryHandJoin(base, data_config.repo_id,
                                   object_version=object_version,
                                   hand_version=hand_version,
                                   max_hand_episodes=max_hand_episodes)
    if len(base) != len(joined.objects):
        raise ValueError("LeRobot frame count differs from Object Geometry")
    return data_loader.TransformedDataset(
        joined,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=False),
            *data_config.model_transforms.inputs,
        ],
        preserve_keys=("spatial", "object_geometry", "hand_keypoints"),
    )
