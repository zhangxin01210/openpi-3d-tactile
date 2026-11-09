from collections.abc import Iterator, Sequence
import dataclasses
import hashlib
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
try:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.spatial_dataset.adapter import SpatialAugmentedDataset
from openpi.spatial_dataset.derived import SpatialDerivedDataset
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

_V2_LOCAL_DATASETS = frozenset({"data/1018_peg_all", "data/1028_press_button", "data/1030_yellow_socket"})


def resolve_local_lerobot_root(repo_id: str) -> pathlib.Path | None:
    """Resolve checkout data/ paths and explicit local paths without Hub fallback."""
    path = pathlib.Path(repo_id)
    if path.parts and path.parts[0] == "data":
        return (pathlib.Path(__file__).resolve().parents[3] / repo_id).resolve()
    if path.is_absolute() or repo_id.startswith(("./", "../", "~/")):
        return path.expanduser().resolve()
    return None


def validate_spatial_source_identity(spatial_dataset, dataset_root: pathlib.Path, dataset_meta) -> None:
    """Reject stale or partial sidecars before joining a reindexed dataset."""
    source = spatial_dataset.manifest.get("source_dataset")
    provenance = source.get("source_meta_sha256") if isinstance(source, dict) else None
    if not isinstance(provenance, dict):
        raise ValueError(f"Spatial sidecar {spatial_dataset.root} lacks source metadata provenance")
    for name in ("info.json", "episodes.jsonl"):
        path = dataset_root / "meta" / name
        if not path.is_file():
            raise FileNotFoundError(f"Local LeRobot source metadata missing: {path}")
        entry = provenance.get(name)
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"Spatial sidecar {spatial_dataset.root} lacks source provenance for {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Spatial sidecar {spatial_dataset.root} does not match current {path}")

    for episode in range(dataset_meta.total_episodes):
        try:
            length = int(dataset_meta.episodes[episode]["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Local LeRobot metadata lacks a valid length for episode {episode}") from exc
        shards = spatial_dataset._episode_shards.get(episode, ())  # noqa: SLF001 - validate join coverage
        next_frame = 0
        for shard in shards:
            if shard.first_frame != next_frame or shard.count != shard.last_frame - shard.first_frame + 1:
                raise ValueError(f"Spatial sidecar {spatial_dataset.root} has a frame gap in episode {episode}")
            next_frame = shard.last_frame + 1
        if length <= 0 or next_frame != length:
            raise ValueError(
                f"Spatial sidecar {spatial_dataset.root} covers {next_frame} frames of episode {episode}; "
                f"the LeRobot dataset has {length}"
            )


def validate_object_geometry_calibration(data_config, model_config, manifest) -> bool:
    """Validate recorded calibration; report the approved historical omission.

    Returns True for explicit unverified legacy/V2 provenance exceptions.
    This does not establish calibration parity.
    """
    from openpi.object_geometry.calibration import calibration_provenance_error
    from openpi.object_geometry.calibration import front_calibration_provenance

    settings = data_config.object_geometry
    if settings.require_pose_prediction and getattr(settings, "pose_prediction_schema", "v1") == "v2":
        # The V2 model consumes unaligned raw RGB-D and predicts base_link; its
        # original camera calibration is not recorded. Spatial v1's own profile
        # is validated separately. Never manufacture a V2 calibration record.
        if manifest.get("calibration_source") != "V2 predicts base_link from unaligned raw RGB-D; model camera calibration provenance not recorded":
            raise ValueError("V2 prediction calibration provenance is missing or changed")
        logging.warning("V2 model camera calibration provenance is unrecorded; base_link compatibility requires review")
        return True
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    dataset_root = pathlib.Path(settings.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    accepted_roots = {(repo_root / name).resolve() for name in _V2_LOCAL_DATASETS}
    base_root = resolve_local_lerobot_root(data_config.repo_id) if data_config.repo_id else None
    accepted = (settings.version == "v1" and dataset_root.resolve() in accepted_roots and
                base_root == dataset_root.resolve())
    if accepted and "calibration" not in manifest:
        logging.warning(
            "LEGACY_UNRECORDED: Object geometry %s/%s has no recorded calibration provenance; "
            "its historical calibration cannot be considered verified.", settings.dataset_root, settings.version,
        )
        return True

    final_v2 = getattr(getattr(model_config, "encoder", None), "object_relation_mode", None) == "final_v2"
    if final_v2 and "calibration" not in manifest:
        raise ValueError("Object geometry sidecar missing calibration provenance")
    profile = data_config.spatial.front_calibration_profile if data_config.spatial is not None else None
    if profile is not None:
        expected = front_calibration_provenance(repo_root, profile)
        mismatch = calibration_provenance_error(manifest.get("calibration"), expected)
        if mismatch:
            raise ValueError(f"Object geometry sidecar {mismatch}")
    elif (accepted or final_v2) and "calibration" in manifest:
        actual = manifest["calibration"]
        expected = front_calibration_provenance(repo_root, None)
        if not isinstance(actual, dict):
            raise ValueError("Object geometry sidecar has malformed calibration provenance")
        mismatch = next((f"calibration {key} differs from requested front profile" for key, value in expected.items()
                         if actual.get(key) != value), None)
        if mismatch or actual.get("profile"):
            raise ValueError(f"Object geometry sidecar {mismatch or 'unexpected calibration profile'}")
    return False


class _LocalLeRobotMetadata(lerobot_dataset.LeRobotDatasetMetadata):
    """Prevent LeRobot's metadata-missing fallback from contacting the Hub."""

    def load_metadata(self):
        try:
            return super().load_metadata()
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise RuntimeError(f"Incomplete local LeRobot metadata under {self.root}: {exc}") from exc


class _LocalLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Keep local episode-loading failures out of the Hub retry path."""

    def get_episodes_file_paths(self):
        try:
            paths = super().get_episodes_file_paths()
        except (FileNotFoundError, NotADirectoryError, AssertionError) as exc:
            raise RuntimeError(f"Cannot enumerate local LeRobot data under {self.root}: {exc}") from exc
        missing = [path for path in paths if not (self.root / path).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete local LeRobot dataset: {self.root / missing[0]}")
        return paths

    def load_hf_dataset(self):
        try:
            return super().load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError, AssertionError) as exc:
            raise RuntimeError(f"Cannot load local LeRobot data under {self.root}: {exc}") from exc


def _lerobot_tasks_to_dict(tasks) -> dict[int, str]:
    if isinstance(tasks, dict):
        return tasks

    import pandas as pd

    if not isinstance(tasks, pd.DataFrame):
        return dict(tasks)

    if "task" in tasks.columns:
        return dict(zip(tasks["task_index"], tasks["task"]))

    if "task_index" in tasks.columns:
        return {
            int(row["task_index"]): str(index)
            for index, row in tasks.iterrows()
        }

    raise ValueError(
        "Unsupported LeRobot tasks table; expected columns "
        f"'task_index' and either 'task' or task text in the index, got {list(tasks.columns)!r}."
    )


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        preserve_keys: Sequence[str] = (),
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._preserve_keys = tuple(preserve_keys)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        item = self._dataset[index]

        if not self._preserve_keys:
            return self._transform(item)

        # spatial sidecar 有意绕开原有 OpenPI 的 repack / normalization /
        # model transforms。原有 transform 面向 image/state/action，且可能
        # 丢弃未知 key，因此这里显式保存并在 transform 后恢复。
        working = dict(item)
        preserved = {}

        for key in self._preserve_keys:
            if key not in working:
                raise KeyError(f"需要保留的 key 不存在： {key!r} in dataset item.")
            preserved[key] = working.pop(key)

        transformed = self._transform(working)
        output = dict(transformed)

        for key, value in preserved.items():
            if key in output:
                raise KeyError(f"transform 输出意外包含保留 key： {key!r}.")
            output[key] = value

        return typing.cast(T_co, output)

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def validate_object_geometry_roles(manifest, expected_roles):
    """Accept current RGBTrack or migrated-v1 provenance and reject swapped roles."""
    if expected_roles is None:
        return
    actual_roles = manifest.get("role_to_object") or manifest.get("source", {}).get("roles")
    if actual_roles != dict(expected_roles):
        raise ValueError(f"Object geometry roles differ: expected {dict(expected_roles)}, got {actual_roles}")


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    local_root = resolve_local_lerobot_root(repo_id)
    if local_root is not None:
        info_path = local_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Local LeRobot dataset {repo_id} is missing {info_path}; no Hub lookup is attempted")

        # This LeRobot version retries Hub downloads on missing metadata. Turn
        # those failures into local errors before the retry path is entered.
        dataset_meta = _LocalLeRobotMetadata(repo_id, root=local_root)
        missing = []
        for episode in range(dataset_meta.total_episodes):
            paths = [dataset_meta.get_data_file_path(episode)]
            paths.extend(dataset_meta.get_video_file_path(episode, key) for key in dataset_meta.video_keys)
            missing.extend(str(local_root / path) for path in paths if not (local_root / path).is_file())
            if missing:
                break
        if missing:
            raise FileNotFoundError(f"Local LeRobot dataset {repo_id} is incomplete: {missing[0]}; no Hub lookup is attempted")
        for field in ("spatial", "object_geometry"):
            settings = getattr(data_config, field)
            if settings is not None:
                settings_root = resolve_local_lerobot_root(settings.dataset_root)
                if settings_root is not None:
                    data_config = dataclasses.replace(
                        data_config, **{field: dataclasses.replace(settings, dataset_root=str(settings_root))}
                    )
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    # Some datasets can have tiny timestamp drift around frame decode boundaries.
    # Allow overriding tolerance via env to avoid hard failures in dataloader workers.
    tolerance_s = float(os.environ.get("OPENPI_LEROBOT_TOLERANCE_S", "0.0005"))
    dataset_class = _LocalLeRobotDataset if local_root is not None else lerobot_dataset.LeRobotDataset
    dataset = dataset_class(
        repo_id,
        **({"root": local_root} if local_root is not None else {}),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        tolerance_s=tolerance_s,
    )

    if data_config.prompt_from_task:
        tasks = _lerobot_tasks_to_dict(dataset_meta.tasks)
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])
    if data_config.contactworld is not None:
        settings = data_config.contactworld
        if settings.visual != bool(getattr(model_config, "use_visual", False)):
            raise ValueError("ContactWorld visual sidecar and model spec disagree")
        if settings.force != bool(getattr(model_config, "use_tactile", False)):
            raise ValueError("ContactWorld force sidecar and model spec disagree")
        if settings.dataset_version in {"v2", "v3"}:
            from openpi.spatial_dataset.contactworld_v2_lerobot import ContactWorldV2LeRobotSpatial

            if settings.visual != (settings.cloud_mode != "none") or settings.force != (settings.force_mode != "none"):
                raise ValueError("ContactWorld replayed cloud/force mode differs from model modality switches")
            repr_mode = settings.contact_repr_mode
            expected_centered = getattr(model_config, "tactile_representation", "") in {
                "contact_centered_point", "contact_centered_map"}
            expected_relation = bool(getattr(model_config, "use_tactile_scene_relation", False))
            if ((repr_mode == "centered") != expected_centered
                    or (repr_mode == "relation") != expected_relation):
                raise ValueError("Contact representation sidecar and model spec disagree")
            dataset = ContactWorldV2LeRobotSpatial(
                dataset, settings.root, cloud_mode=settings.cloud_mode,
                force_mode=settings.force_mode, depth=settings.depth,
                action_horizon=settings.action_horizon,
                contact_repr_mode=repr_mode, contact_repr_root=settings.contact_repr_root,
            )
        elif settings.dataset_version == "v1":
            from openpi.spatial_dataset.contactworld_lerobot import ContactWorldLeRobotSpatial

            dataset = ContactWorldLeRobotSpatial(
                dataset, settings.root, visual=settings.visual, force=settings.force,
                depth=settings.depth, cloud_noise_m=settings.cloud_noise_m,
                force_ee3d_proxy=settings.force_ee3d_proxy, action_horizon=settings.action_horizon,
            )
        else:
            raise ValueError("Unknown ContactWorld dataset version: " + settings.dataset_version)
    if data_config.univtac_pointcloud is not None:
        from openpi.spatial_dataset.univtac import UniVTACVisualDataset

        if data_config.spatial is not None:
            raise ValueError("UniVTAC pointcloud and spatial sidecar cannot both be configured")
        if not getattr(model_config, "use_visual", False) or getattr(model_config, "use_tactile", False):
            raise ValueError("UniVTAC pointcloud adapter requires a visual-only spatial model")
        settings = data_config.univtac_pointcloud
        if getattr(model_config, "visual_points", None) != 2 * settings.points_per_camera:
            raise ValueError("UniVTAC point count must match the model visual_points spec")
        dataset = UniVTACVisualDataset(
            dataset, settings.dataset_root,
            points_per_camera=settings.points_per_camera, stride=settings.stride,
        )
    if data_config.spatial is not None:
        spatial_dataset = SpatialDerivedDataset(
            data_config.spatial.dataset_root,
            version=data_config.spatial.version,
        )
        if data_config.spatial.require_source_metadata_match:
            if local_root is None:
                raise ValueError("Spatial source verification requires a local LeRobot dataset")
            validate_spatial_source_identity(spatial_dataset, local_root, dataset_meta)
        actual_roles = tuple(spatial_dataset.manifest["spatial_contract"]["camera_roles"])
        if actual_roles != data_config.spatial.camera_roles:
            raise ValueError(
                f"Spatial sidecar camera roles {actual_roles} do not match config "
                f"{data_config.spatial.camera_roles}"
            )
        manifest_diagnostics = spatial_dataset.manifest.get("preprocess_config", {}).get("diagnostics", {})
        if data_config.spatial.front_calibration_profile is None:
            expected_extrinsic = expected_intrinsic = False
        else:
            from openpi.spatial.calibration_profile import apply_front_calibration_profile
            from openpi.spatial.config import make_baseline_config

            profile_path = pathlib.Path(data_config.spatial.front_calibration_profile)
            if not profile_path.is_absolute():
                profile_path = pathlib.Path(__file__).resolve().parents[3] / profile_path
            expected = apply_front_calibration_profile(make_baseline_config(), profile_path).diagnostics
            expected_extrinsic = expected.enable_front_extrinsic_correction
            expected_intrinsic = expected.enable_front_intrinsic_k1
            if expected_extrinsic:
                actual_translation = np.asarray(
                    manifest_diagnostics.get("front_extrinsic_translation_base_m", ()), dtype=float
                )
                if actual_translation.shape != (3,) or not np.allclose(
                    actual_translation, expected.front_extrinsic_translation_base_m
                ):
                    raise ValueError("Spatial sidecar front extrinsic does not match the configured profile")
            if expected_intrinsic:
                actual_k = manifest_diagnostics.get("front_color_k1", {})
                if any(not np.isclose(actual_k.get(key, np.nan), getattr(expected.front_color_k1, key))
                       for key in ("fx", "fy", "cx", "cy")):
                    raise ValueError("Spatial sidecar front intrinsic does not match the configured profile")
        if (bool(manifest_diagnostics.get("enable_front_extrinsic_correction")) != expected_extrinsic
                or bool(manifest_diagnostics.get("enable_front_intrinsic_k1")) != expected_intrinsic):
            raise ValueError("Spatial sidecar diagnostic calibration flags do not match the train config")
        # Relation-only models still verify sidecar source and calibration metadata,
        # but never join visual or tactile values into their input.
        encoder = getattr(model_config, "encoder", None)
        hand_mode = getattr(encoder, "hand_object_hand_point_mode", "fk_18")
        if getattr(model_config, "use_visual", True) or getattr(model_config, "use_tactile", True) or hand_mode == "tactile_center_5":
            dataset = SpatialAugmentedDataset(
                dataset,
                spatial_dataset,
                copy_arrays=data_config.spatial.copy_arrays,
                use_visual=getattr(model_config, "use_visual", True),
                use_tactile=getattr(model_config, "use_tactile", True),
                tactile_coordinate_frame=data_config.spatial.tactile_coordinate_frame,
                hand_point_mode=hand_mode,
                validate_tactile_order=getattr(encoder, "tactile_encoder_type", "structured_v1") == "simple_mlp",
            )
    if data_config.object_geometry is not None:
        from openpi.object_geometry_dataset.adapter import ObjectGeometryAugmentedDataset
        from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

        settings = data_config.object_geometry
        if settings.version == "pretrained_pose_v2" and (
            not settings.require_pose_prediction or getattr(settings, "pose_prediction_schema", "v1") != "v2"
        ):
            raise ValueError("pretrained_pose_v2 requires the V2 provenance validation")
        object_geometry_dataset = ObjectGeometryDerivedDataset(
            settings.dataset_root,
            version=settings.version,
        )
        if settings.require_pose_prediction and getattr(settings, "pose_prediction_schema", "v1") == "v2":
            from openpi.cad_pose.v2_native import validate_native_manifest
            if not settings.pose_checkpoint_path or not settings.pose_v2_cache_root:
                raise ValueError("V2 requires an explicit pose checkpoint and RGB-D cache")
            if local_root is None or pathlib.Path(settings.dataset_root).resolve() != pathlib.Path(local_root).resolve():
                raise ValueError("V2 geometry must belong to the same local LeRobot dataset")
            validate_native_manifest(object_geometry_dataset.manifest, settings.dataset_root,
                                     settings.pose_checkpoint_path, settings.pose_v2_cache_root,
                                     reader=object_geometry_dataset)
        elif settings.require_pose_prediction and getattr(settings, "pose_prediction_schema", "v1") == "v1":
            import hashlib
            import torch
            checkpoint = pathlib.Path(settings.pose_checkpoint_path or "")
            if not settings.pose_checkpoint_path or not checkpoint.is_file():
                raise FileNotFoundError("Set OPENPI_CAD_POSE_CHECKPOINT to the accepted pose checkpoint")
            manifest = object_geometry_dataset.manifest
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            acceptance = manifest.get("acceptance") or {}
            if (manifest.get("producer") != "openpi.cad_pose.sidecar"
                    or manifest.get("pose_checkpoint_sha256") != digest
                    or manifest.get("model_version") != "xhand_cad_pose_transformer_v1"
                    or manifest.get("preprocess_version") != "front_rgbd_spatial_v1"
                    or state.get("synthetic") is not False
                    or state.get("camera_hashes") != manifest.get("camera_hashes")
                    or state.get("ordered_cad_sha256_by_dataset", {}).get(pathlib.Path(settings.dataset_root).name)
                       != manifest.get("ordered_cad_sha256")
                    or acceptance.get("status") != "accepted" or not acceptance.get("reviewer")
                    or acceptance.get("checkpoint_sha256") != digest):
                raise ValueError("predicted geometry sidecar and pose checkpoint provenance mismatch")
        elif settings.require_pose_prediction:
            raise ValueError(f"unknown pose prediction schema: {settings.pose_prediction_schema}")
        if settings.m2_state_from_rgbtrack_v1:
            encoder = getattr(model_config, "encoder", None)
            if (getattr(encoder, "object_relation_mode", None) != "hand_scene_oo_v1"
                    or not getattr(encoder, "enable_object_relations", False)
                    or getattr(encoder, "hand_scene_enable_funnel_mask_encoder", False)
                    or not getattr(model_config, "use_tactile", False)
                    or settings.version != "v1" or settings.missing != "error"):
                raise ValueError("RGBTrack v1 gate derivation is only supported for tactile M2 without a mask encoder")
        if (settings.require_hand_scene and not object_geometry_dataset.has_hand_scene
                and not settings.m2_state_from_rgbtrack_v1):
            raise ValueError(
                f"Object geometry {settings.dataset_root}/{settings.version} must include HandScene state"
            )
        # Object clouds are in base_link and must use the same profiled front
        # projection as the spatial sidecar they condition.  This check is at
        # the train boundary, not merely an exporter convention, so a copied
        # legacy cloud directory cannot silently pair with v1_front_candidate.
        validate_object_geometry_calibration(data_config, model_config, object_geometry_dataset.manifest)
        validate_object_geometry_roles(object_geometry_dataset.manifest, settings.expected_roles)
        fixed_anchor_assets = None
        if settings.anchor_asset_root is not None:
            if (getattr(getattr(model_config, "encoder", None), "object_relation_mode", None) != "hand_object_v4"
                    or settings.anchor_count != 16 or settings.version != "v1" or settings.missing != "error"):
                raise ValueError("HO V4 requires 16 fixed anchors and object_geometry/v1 with missing=error")
            from openpi.object_geometry.ho_v4_anchors import load_assets

            dataset_root = pathlib.Path(settings.dataset_root).resolve()
            assets = load_assets(dataset_root.parent.parent, dataset_root.name, settings.anchor_asset_root)
            fixed_anchor_assets = assets
        elif getattr(getattr(model_config, "encoder", None), "object_relation_mode", None) == "hand_object_v4":
            raise ValueError("hand_object_v4 requires fixed anchor assets")
        dataset = ObjectGeometryAugmentedDataset(
            dataset,
            object_geometry_dataset,
            copy_arrays=settings.copy_arrays,
            missing=settings.missing,
            m2_state_from_rgbtrack_v1=settings.m2_state_from_rgbtrack_v1,
            fixed_anchor_assets=fixed_anchor_assets,
        )
        if (getattr(getattr(model_config, "encoder", None), "object_relation_mode", None) in {"final_v2", "hand_object_v3", "hand_object_v4", "relation_set_v1"}
                and getattr(getattr(model_config, "encoder", None), "hand_object_hand_point_mode", "fk_18") == "fk_18"):
            from openpi.training.xhand_multimodal_dataset import XHandKeypointsSpatialJoin

            if data_config.spatial is None:
                raise ValueError("final_v2 relations require a spatial sidecar")
            dataset = XHandKeypointsSpatialJoin(
                dataset, settings.dataset_root, object_version=settings.hand_keypoint_object_version or settings.version,
                spatial_version=data_config.spatial.version,
            )
    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        preserve_keys=("spatial",) if data_config.spatial is not None or data_config.object_geometry is not None
        or data_config.univtac_pointcloud is not None
        or (data_config.contactworld is not None and
            (data_config.contactworld.visual or data_config.contactworld.force)) else (),
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        if data_config.spatial is not None:
            raise NotImplementedError("Spatial sidecar 目前只支持 LeRobot dataset。")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
