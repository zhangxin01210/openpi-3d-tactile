import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
import openpi.models.spatial_encoders.types as _spatial_types
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#     # 可选 derived spatial sidecar；shape contract 不绑定具体 encoder。
#     "spatial": {
#         "visual": {
#             "xyz_m": float32[*b, nv, 3],
#             "rgb": (uint8|float32)[*b, nv, 3],
#             "rgb_valid": bool[*b, nv],
#             "point_mask": bool[*b, nv],
#         },
#         "tactile": {
#             "xyz_m": float32[*b, nt, 3],
#             "force": float32[*b, nt, 3],
#             "force_norm": float32[*b, nt],
#             "finger_id": int32[*b, nt],
#             "taxel_id": int32[*b, nt],
#             "point_mask": bool[*b, nt],
#         },
#     },
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # 可选、与具体 encoder 无关的 spatial sidecar。
    spatial: _spatial_types.SpatialEncoderInput[ArrayT] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        spatial_data = data.get("spatial")
        spatial = None
        if spatial_data is not None:
            visual_data = spatial_data.get("visual")
            tactile_data = spatial_data.get("tactile")

            visual = None
            if visual_data is not None:
                visual = _spatial_types.VisualSpatialInput(
                    xyz_m=visual_data["xyz_m"],
                    rgb=visual_data["rgb"],
                    rgb_valid=visual_data["rgb_valid"],
                    point_mask=visual_data["point_mask"],
                )

            tactile = None
            if tactile_data is not None:
                tactile = _spatial_types.TactileSpatialInput(
                    xyz_m=tactile_data["xyz_m"],
                    force=tactile_data["force"],
                    force_norm=tactile_data["force_norm"],
                    finger_id=tactile_data["finger_id"],
                    taxel_id=tactile_data["taxel_id"],
                    point_mask=tactile_data["point_mask"],
                )

            centered_data = spatial_data.get("contact_centered_tactile")
            contact_centered_tactile = None
            if centered_data is not None:
                contact_centered_tactile = _spatial_types.ContactCenteredTactileInput(
                    patch_xyz_local=centered_data["patch_xyz_local"],
                    patch_force=centered_data["patch_force"],
                    patch_mask=centered_data["patch_mask"],
                    anchor_xyz_m=centered_data["anchor_xyz_m"],
                    anchor_stats=centered_data["anchor_stats"],
                    anchor_mask=centered_data["anchor_mask"],
                    finger_summary=centered_data["finger_summary"],
                    finger_center_xyz_m=centered_data["finger_center_xyz_m"],
                    finger_valid=centered_data["finger_valid"],
                )

            relation_data = spatial_data.get("tactile_scene_relation")
            tactile_scene_relation = None
            if relation_data is not None:
                tactile_scene_relation = _spatial_types.TactileSceneRelationInput(
                    query_xyz_m=relation_data["query_xyz_m"],
                    query_features=relation_data["query_features"],
                    query_valid=relation_data["query_valid"],
                    candidate_xyz_m=relation_data["candidate_xyz_m"],
                    candidate_mask=relation_data["candidate_mask"],
                )

            object_data = spatial_data.get("object_geometry")
            object_geometry = None
            if object_data is not None:
                hand_scene_data = object_data.get("hand_scene")
                hand_scene = None
                if hand_scene_data is not None:
                    hand_scene = _spatial_types.HandSceneInput(
                        manipulated_relation_valid=hand_scene_data["manipulated_relation_valid"],
                        target_relation_valid=hand_scene_data["target_relation_valid"],
                        manipulated_geometry_quality=hand_scene_data["manipulated_geometry_quality"],
                        target_geometry_quality=hand_scene_data["target_geometry_quality"],
                        grasp_reference_force_norm=hand_scene_data["grasp_reference_force_norm"],
                        grasp_short_delta_force_norm=hand_scene_data["grasp_short_delta_force_norm"],
                        grasp_contact_valid=hand_scene_data["grasp_contact_valid"],
                        grasp_quality=hand_scene_data["grasp_quality"],
                        funnel_mask=hand_scene_data["funnel_mask"],
                        funnel_mask_valid=hand_scene_data["funnel_mask_valid"],
                        funnel_mask_quality=hand_scene_data["funnel_mask_quality"],
                        funnel_mask_hand_offset_xy=hand_scene_data["funnel_mask_hand_offset_xy"],
                    )
                object_geometry = _spatial_types.ObjectGeometryInput(
                    manipulated_xyz_m=object_data["manipulated_xyz_m"],
                    manipulated_mask=object_data["manipulated_mask"],
                    manipulated_source=object_data["manipulated_source"],
                    manipulated_age=object_data["manipulated_age"],
                    manipulated_confidence=object_data["manipulated_confidence"],
                    target_xyz_m=object_data["target_xyz_m"],
                    target_mask=object_data["target_mask"],
                    target_source=object_data["target_source"],
                    target_age=object_data["target_age"],
                    target_confidence=object_data["target_confidence"],
                    hand_scene=hand_scene,
                )

            spatial = _spatial_types.SpatialEncoderInput(
                visual=visual,
                tactile=tactile,
                contact_centered_tactile=contact_centered_tactile,
                tactile_scene_relation=tactile_scene_relation,
                object_geometry=object_geometry,
            )

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            spatial=spatial,
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        spatial=observation.spatial,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        expected_params = state.to_pure_dict()
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(expected_params, params)
        # Orbax restores dictionary keys as strings (and intersect_trees also stringifies them), while NNX uses
        # integer keys for list members. Validate the canonical paths before rebuilding the model's key types.
        # Validate first so missing or extra weights still raise a descriptive error before reconstruction.
        flat_params = ocp.tree.to_flat_dict(params)
        at.check_pytree_equality(
            expected=ocp.tree.to_flat_dict(expected_params), got=flat_params, check_shapes=True, check_dtypes=False
        )
        params = ocp.tree.from_flat_dict(flat_params, target=expected_params)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
