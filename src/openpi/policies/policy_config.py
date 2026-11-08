import json
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _policy_metadata(train_config: _config.TrainConfig, data_config: _config.DataConfig) -> dict[str, Any]:
    metadata = dict(train_config.policy_metadata or {})
    if isinstance(train_config.data, _config.LeRobotXHandDataConfig):
        spatial = data_config.spatial
        visual = spatial is not None and bool(getattr(train_config.model, "use_visual", True))
        tactile = spatial is not None and bool(getattr(train_config.model, "use_tactile", True))
        metadata.update(
            spatial_camera_roles=list(spatial.camera_roles) if visual else [],
            spatial_use_visual=visual,
            spatial_use_tactile=tactile,
            rgb_only_baseline=not visual and not tactile and data_config.object_geometry is None,
            training_config=train_config.name,
            object_geometry_required=data_config.object_geometry is not None,
        )
        if spatial is not None:
            metadata["front_calibration_profile"] = spatial.front_calibration_profile
        # Use the dataset's actual language conditioning, not the robot
        # client's historical Peg default. Do not guess for multi-task data.
        dataset_root = pathlib.Path(data_config.repo_id or "").expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (
                pathlib.Path(os.environ.get("OPENPI_REPO_ROOT", pathlib.Path(__file__).resolve().parents[3]))
                / dataset_root
            )
        tasks_path = dataset_root / "meta/tasks.jsonl"
        if tasks_path.is_file():
            prompts = {json.loads(line)["task"] for line in tasks_path.read_text().splitlines() if line.strip()}
            if len(prompts) == 1:
                metadata["training_prompt"] = next(iter(prompts))
                metadata["training_prompt_source"] = str(tasks_path)
    return metadata


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Support ordinary OpenPI layouts and explicitly identified Orbax manager exports.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    pose_runtime = None
    if (
        isinstance(train_config.data, _config.LeRobotXHandDataConfig)
        and data_config.object_geometry is not None
        and data_config.spatial is None
    ):
        raise ValueError("XHand online object geometry requires a spatial preprocessing configuration")
    metadata = _policy_metadata(train_config, data_config)
    if default_prompt is None:
        default_prompt = metadata.get("training_prompt")
    if default_prompt is not None:
        metadata["default_prompt"] = default_prompt
    params_dir = checkpoint_dir / "params"
    assets_dir = checkpoint_dir / "assets"
    if not params_dir.is_dir() and (checkpoint_dir / "_METADATA").is_file():
        try:
            import orbax.checkpoint as ocp

            manager_metadata = ocp.PyTreeCheckpointer().metadata(checkpoint_dir)
            if "params" in manager_metadata:
                params_dir = checkpoint_dir
        except Exception:
            pass  # Preserve standard loader diagnostics for malformed checkpoints.
    if (
        not assets_dir.is_dir()
        and data_config.asset_id is not None
        and (checkpoint_dir / data_config.asset_id / "norm_stats.json").is_file()
    ):
        assets_dir = checkpoint_dir

    if data_config.object_geometry is not None and getattr(
        data_config.object_geometry, "require_pose_prediction", False
    ):
        from openpi.cad_pose.policy import make_pose_runtime

        if not data_config.asset_id:
            raise ValueError("pose prediction policy requires a checkpoint asset id")
        root = pathlib.Path(os.environ.get("OPENPI_REPO_ROOT", pathlib.Path(__file__).resolve().parents[3]))
        pose_runtime = make_pose_runtime(
            data_config.object_geometry,
            data_config.spatial.front_calibration_profile,
            repo=root,
            provenance_path=assets_dir / data_config.asset_id / "pose_provenance.json",
            device=os.environ.get("OPENPI_CAD_POSE_DEVICE", "cpu"),
        )

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # XHand deployment must not silently discard trained spatial/relation
        # parameters when a checkpoint is paired with a different config.
        strict_xhand = isinstance(train_config.data, _config.LeRobotXHandDataConfig)
        params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
        try:
            model = train_config.model.load(params, remove_extra_params=not strict_xhand)
        except ValueError as exc:
            if not strict_xhand:
                raise
            raise ValueError(
                f"Checkpoint parameters do not match training config {train_config.name!r}; "
                "use the config and code matching this checkpoint"
            ) from exc
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        ),
        *data_config.model_transforms.inputs,
    ]

    if data_config.spatial is not None:
        import openpi.policies.spatial_online as spatial_online

        object_settings = data_config.object_geometry
        relation_mode = getattr(getattr(train_config.model, "encoder", None), "object_relation_mode", None)
        scene_modes = {"hand_scene_v0", "hand_scene_oo_v1", "dtcr_v1", "dtcr_independent_v1"}
        keypoint_modes = {"final_v2", "hand_object_v3", "hand_object_v4", "relation_set_v1"}
        if object_settings is not None and relation_mode not in scene_modes | keypoint_modes | {"legacy_ho_oo"}:
            raise ValueError(
                f"Unsupported online object relation mode: {relation_mode!r}; "
                "implement and validate its input contract before deployment"
            )
        input_transforms = [
            spatial_online.XHandSpatialOnlinePreprocess(
                camera_roles=data_config.spatial.camera_roles,
                use_visual=getattr(train_config.model, "use_visual", True),
                use_tactile=getattr(train_config.model, "use_tactile", True),
                hand_point_mode=getattr(
                    getattr(train_config.model, "encoder", None), "hand_object_hand_point_mode", "fk_18"
                ),
                validate_tactile_order=getattr(
                    getattr(train_config.model, "encoder", None), "tactile_encoder_type", "structured_v1"
                )
                == "simple_mlp",
                front_calibration_profile=data_config.spatial.front_calibration_profile,
                tactile_coordinate_frame=data_config.spatial.tactile_coordinate_frame,
                require_object_geometry=data_config.object_geometry is not None,
                allow_object_set=relation_mode == "relation_set_v1",
                anchor_asset_root=(object_settings.anchor_asset_root if object_settings is not None else None),
                anchor_dataset_root=(object_settings.dataset_root if object_settings is not None else None),
                require_hand_scene=(
                    object_settings is not None and (object_settings.require_hand_scene or relation_mode in scene_modes)
                ),
                require_hand_keypoints=(object_settings is not None and relation_mode in keypoint_modes),
                m2_state_from_rgbtrack_v1=(object_settings is not None and object_settings.m2_state_from_rgbtrack_v1),
            ),
            transforms.PreserveKeysTransform(
                transform=transforms.compose(
                    [
                        spatial_online.XHandSpatialOnlineRepack(),
                        *input_transforms,
                    ]
                ),
                keys=("spatial",),
            ),
        ]
    elif data_config.univtac_pointcloud is not None:
        # The live UniVTAC client supplies the same world visual point schema as
        # the HDF5-backed training dataset. Repacking images/state drops unknown
        # fields, so carry spatial through the standard policy transforms.
        input_transforms = [
            transforms.PreserveKeysTransform(
                transform=transforms.compose(input_transforms),
                keys=("spatial",),
            )
        ]
    elif data_config.contactworld is not None and (data_config.contactworld.visual or data_config.contactworld.force):
        # The ContactWorld simulator client supplies the same spatial schema as
        # the converted LeRobot sidecar; ordinary transforms repack image/state.
        input_transforms = [
            transforms.PreserveKeysTransform(
                transform=transforms.compose(input_transforms),
                keys=("spatial",),
            )
        ]

    policy = _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
    if pose_runtime is not None:
        from openpi.cad_pose.policy import PoseConditionedPolicy

        return PoseConditionedPolicy(policy, pose_runtime)
    return policy
