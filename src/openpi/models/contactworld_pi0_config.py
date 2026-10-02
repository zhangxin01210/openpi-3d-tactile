"""ContactWorld-specific spatial tokens for USB modality experiments."""

from __future__ import annotations

import dataclasses
from typing import Literal

from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from openpi.models.spatial_encoders import structured_spatial_encoder
from openpi.models.spatial_encoders import transforms as spatial_transforms
from openpi.models.spatial_encoders.types import SpatialEncoderInput
from openpi.models.spatial_encoders.types import SpatialEncoderOutput
from openpi.models.spatial_pi0_config import SpatialPi0Config

TactileRepresentation = Literal[
    "none", "grid", "summary", "ee3d", "base3d",
    "map_coupled", "map_split", "map_local", "map_split_local", "map_coupled_matched",
    "map_local_multiquery",
]


@dataclasses.dataclass(frozen=True)
class ContactWorldPi0Config(SpatialPi0Config):
    tactile_representation: TactileRepresentation = "grid"
    token_dim: int = 128
    split_route: bool = False
    visual_points: int = 1024
    tactile_points: int = 140
    tactile_pool_per_finger: bool = False
    structured: bool = False
    visual_num_centers: int = 16
    visual_knn: int = 32
    tactile_local_tokens_per_finger: int = 2
    tactile_knn: int = 8
    spatial_route: Literal["shared", "visual_prefix_tactile_suffix"] = "shared"

    def __post_init__(self):
        super().__post_init__()
        if self.split_route and (not self.use_visual or not self.use_tactile or
                                 not self.conditioning.use_prefix or not self.conditioning.use_suffix):
            raise ValueError("Split route needs both modalities and both conditioning targets")
        if self.use_tactile and self.tactile_representation == "none":
            raise ValueError("Force branch enabled without a representation")
        if self.tactile_pool_per_finger and self.tactile_points != 280 and not (
                self.structured and self.tactile_points == 140):
            raise ValueError("Per-finger ContactWorld pooling requires 140 or 280 taxels")
        if self.tactile_representation in {"map_coupled", "map_split", "map_local", "map_split_local",
                                           "map_coupled_matched", "map_local_multiquery"} and (
                not self.use_tactile or not self.tactile_pool_per_finger or self.tactile_points != 280):
            raise ValueError("Bilateral tactile maps require separate 140-taxel finger grids")
        if self.spatial_route == "visual_prefix_tactile_suffix" and (
                not self.structured or not self.use_visual or not self.use_tactile or
                not self.conditioning.use_prefix or not self.conditioning.use_suffix):
            raise ValueError("Split modality route needs structured visual/tactile and both targets")

    @property
    def spatial_prefix_token_indices(self) -> tuple[int, ...] | None:
        if self.spatial_route == "visual_prefix_tactile_suffix":
            return tuple(range(self.visual_num_centers + 1))
        return None

    @property
    def spatial_suffix_token_indices(self) -> tuple[int, ...] | None:
        if self.spatial_route == "visual_prefix_tactile_suffix":
            visual_tokens = self.visual_num_centers + 1
            tactile_tokens = (2 if self.tactile_representation == "summary" else
                              2 * (self.tactile_local_tokens_per_finger + 1))
            return tuple(range(visual_tokens, visual_tokens + tactile_tokens))
        return None

    @property
    def spatial_encoder_token_dim(self) -> int:
        return self.token_dim

    def create_spatial_encoder(self, *, rngs: nnx.Rngs) -> nnx.Module:
        if self.structured:
            return ContactWorldStructuredEncoder(self, rngs=rngs)
        return ContactWorldEncoder(self, rngs=rngs)


class _TactileMapBlock(nnx.Module):
    """Small pre-norm ViT block for signed 10x14 force maps."""

    def __init__(self, width: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, width * 3, rngs=rngs)
        self.attn_out = nnx.Linear(width, width, rngs=rngs)
        self.norm2 = nnx.LayerNorm(width, rngs=rngs)
        self.mlp1 = nnx.Linear(width, width * 2, rngs=rngs)
        self.mlp2 = nnx.Linear(width * 2, width, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, length, width = x.shape
        q, k, v = jnp.split(self.qkv(self.norm1(x)), 3, axis=-1)
        q = q.reshape(batch, length, 4, width // 4)
        k = k.reshape(batch, length, 4, width // 4)
        v = v.reshape(batch, length, 4, width // 4)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) * (width // 4) ** -0.5
        weights = jnn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhts,bshd->bthd", weights, v).reshape(batch, length, width)
        x = x + self.attn_out(attended)
        return x + self.mlp2(jnn.gelu(self.mlp1(self.norm2(x))))


class _TactileMapViT(nnx.Module):
    """2x2 patches, 35 spatial patches and one or three learned output queries."""

    def __init__(self, channels: int, width: int, token_dim: int, num_outputs: int, *, rngs: nnx.Rngs):
        self.num_outputs = num_outputs
        self.patch = nnx.Linear(4 * channels, width, rngs=rngs)
        self.cls = nnx.Param(jax.random.normal(rngs.params(), (1, num_outputs, width)) * 0.02)
        self.pos = nnx.Param(jax.random.normal(rngs.params(), (1, num_outputs + 35, width)) * 0.02)
        self.blocks = [_TactileMapBlock(width, rngs=rngs) for _ in range(2)]
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        self.output = nnx.Linear(width, token_dim, rngs=rngs) if width != token_dim else None

    def __call__(self, grid: jax.Array) -> jax.Array:
        if grid.shape[1:3] != (10, 14):
            raise ValueError("TacFF map must be 10x14")
        batch, _, _, channels = grid.shape
        patches = grid.reshape(batch, 5, 2, 7, 2, channels)
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5)).reshape(batch, 35, 4 * channels)
        x = self.patch(patches)
        x = jnp.concatenate((jnp.broadcast_to(self.cls.value, (batch, self.num_outputs, x.shape[-1])), x), axis=1)
        x = x + self.pos.value
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, :self.num_outputs])
        return self.output(x) if self.output is not None else x


class ContactWorldStructuredEncoder(nnx.Module):
    """XHand structured XYZ/TacFF encoder plus independent tactile-map ViTs."""

    def __init__(self, config: ContactWorldPi0Config, *, rngs: nnx.Rngs):
        self.config = config
        self.map_mode = config.tactile_representation in {
            "map_coupled", "map_split", "map_local", "map_split_local", "map_coupled_matched",
            "map_local_multiquery",
        }
        self.summary_mode = config.tactile_representation == "summary"
        tactile_fingers = 2 if config.tactile_pool_per_finger else 1
        pooled = (config.use_tactile and not config.tactile_pool_per_finger and
                  config.tactile_points == 280)
        local_tokens = (4 if pooled else config.tactile_local_tokens_per_finger)
        structured_config = structured_spatial_encoder.StructuredSpatialEncoderConfig(
            token_dim=config.token_dim,
            visual_num_centers=config.visual_num_centers,
            visual_knn=config.visual_knn,
            num_fingers=tactile_fingers,
            tactile_knn=config.tactile_knn,
            tactile_local_tokens_per_finger=local_tokens,
            tactile_summary_tokens_per_finger=2 if pooled else 1,
            transforms=spatial_transforms.SpatialFeatureTransforms(
                force=spatial_transforms.LinearScaleForceTransform(scale=0.002)
            ),
        )
        self.structured_encoder = structured_config.create(rngs=rngs)
        if self.map_mode:
            split = config.tactile_representation in {"map_split", "map_split_local"}
            matched = config.tactile_representation == "map_coupled_matched"
            multiquery = config.tactile_representation == "map_local_multiquery"
            self.map_encoders = [
                _TactileMapViT(1 if split else 3, 224 if matched else config.token_dim,
                               config.token_dim, 3 if matched or multiquery else 1, rngs=rngs)
                for _ in range(3 if split else 1)
            ]
            self.map_finger_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (2, config.token_dim)) * 0.02
            )
            self.map_channel_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (3 if split or matched else 1,
                                                   config.token_dim)) * 0.02
            )
            self.map_position_proj = nnx.Linear(3, config.token_dim, rngs=rngs)
        if self.summary_mode:
            self.summary_in = nnx.Linear(13, config.token_dim, rngs=rngs)
            self.summary_out = nnx.Linear(config.token_dim, config.token_dim, rngs=rngs)
            self.summary_position_proj = nnx.Linear(3, config.token_dim, rngs=rngs)
            self.summary_finger_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (tactile_fingers, config.token_dim)) * 0.02
            )

    def __call__(self, spatial: SpatialEncoderInput) -> SpatialEncoderOutput:
        outputs = []
        if self.config.use_visual:
            if spatial.visual is None:
                raise ValueError("Missing ContactWorld point cloud")
            outputs.append(self.structured_encoder(SpatialEncoderInput(visual=spatial.visual)))
        if self.config.use_tactile:
            tactile = spatial.tactile
            if tactile is None:
                raise ValueError("Missing ContactWorld tactile force")
            if self.map_mode:
                outputs.append(self._encode_maps(tactile))
            elif self.summary_mode:
                outputs.append(self._encode_summary(tactile))
            else:
                # The pooled ablation has the same six-token budget and the
                # same 4-local/2-summary query types as two separate pads.
                if not self.config.tactile_pool_per_finger:
                    tactile = tactile.replace(finger_id=jnp.zeros_like(tactile.finger_id))
                outputs.append(self.structured_encoder(SpatialEncoderInput(tactile=tactile)))
        tokens = jnp.concatenate([output.tokens for output in outputs], axis=1)
        masks = jnp.concatenate([output.token_mask for output in outputs], axis=1)
        xyz = jnp.concatenate([output.token_xyz_m for output in outputs], axis=1)
        visual_tokens = self.config.visual_num_centers + 1 if self.config.use_visual else 0
        return SpatialEncoderOutput(tokens=tokens, token_mask=masks, token_xyz_m=xyz,
                                    aux={"visual_token_count": jnp.sum(masks[:, :visual_tokens], axis=1),
                                         "tactile_token_count": jnp.sum(masks[:, visual_tokens:], axis=1),
                                         "total_token_count": jnp.sum(masks, axis=1)})

    def _encode_maps(self, tactile) -> SpatialEncoderOutput:
        force = jnp.asarray(tactile.force, dtype=jnp.float32)
        if force.shape[1:] != (280, 3):
            raise ValueError("Bilateral force maps require two ordered 140-taxel grids")
        # Local channels are [normal, shear_x, shear_y]; base channels are Fx/Fy/Fz.
        grids = force.reshape(force.shape[0], 2, 10, 14, 3) / 0.002
        tokens, masks, xyz = [], [], []
        split = len(self.map_encoders) == 3
        for finger in range(2):
            point_mask = tactile.point_mask[:, finger * 140:(finger + 1) * 140]
            finger_mask = jnp.any(point_mask, axis=1)
            point_xyz = tactile.xyz_m[:, finger * 140:(finger + 1) * 140]
            center = jnp.sum(jnp.where(point_mask[..., None], point_xyz, 0), axis=1) / jnp.maximum(
                jnp.sum(point_mask, axis=1, keepdims=True), 1)
            position = self.map_position_proj(
                (center - jnp.asarray([0.5, 0.0, 0.1], dtype=jnp.float32)) / 0.2
            )
            for channel, encoder in enumerate(self.map_encoders):
                grid = grids[:, finger, :, :, channel:channel + 1] if split else grids[:, finger]
                outputs = encoder(grid)
                for query in range(outputs.shape[1]):
                    token_index = channel if split else query if self.config.tactile_representation == "map_coupled_matched" else 0
                    token = (outputs[:, query] + position + self.map_finger_embedding.value[finger]
                             + self.map_channel_embedding.value[token_index])
                    tokens.append(token)
                    masks.append(finger_mask)
                    xyz.append(center)
        return SpatialEncoderOutput(tokens=jnp.stack(tokens, axis=1),
                                    token_mask=jnp.stack(masks, axis=1),
                                    token_xyz_m=jnp.stack(xyz, axis=1))

    def _encode_summary(self, tactile) -> SpatialEncoderOutput:
        force = jnp.asarray(tactile.force, dtype=jnp.float32)
        ids = jnp.asarray(tactile.finger_id)
        tokens, masks, xyz = [], [], []
        for finger in range(2 if self.config.tactile_pool_per_finger else 1):
            valid = tactile.point_mask & (ids == finger)
            count = jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1)
            masked = jnp.where(valid[..., None], force, 0)
            mean = jnp.sum(masked, axis=1) / count
            absolute = jnp.sum(jnp.abs(masked), axis=1) / count
            rms = jnp.sqrt(jnp.sum(masked ** 2, axis=1) / count + 1e-12)
            max_abs = jnp.max(jnp.where(valid[..., None], jnp.abs(force), 0), axis=1)
            active = jnp.sum(valid & (jnp.linalg.norm(force, axis=-1) > 1e-4),
                             axis=1, keepdims=True) / count
            summary = jnp.concatenate((mean / 0.002, absolute / 0.002, rms / 0.002,
                                       max_abs / 0.002, active), axis=-1)
            center = jnp.sum(jnp.where(valid[..., None], tactile.xyz_m, 0), axis=1) / count
            position = self.summary_position_proj(
                (center - jnp.asarray([0.5, 0.0, 0.1], dtype=jnp.float32)) / 0.2
            )
            tokens.append(self.summary_out(jnn.gelu(self.summary_in(summary))) + position +
                          self.summary_finger_embedding.value[finger])
            masks.append(jnp.any(valid, axis=1))
            xyz.append(center)
        return SpatialEncoderOutput(tokens=jnp.stack(tokens, axis=1),
                                    token_mask=jnp.stack(masks, axis=1),
                                    token_xyz_m=jnp.stack(xyz, axis=1))


class ContactWorldEncoder(nnx.Module):
    """One geometry token and one force token; modality order is stable."""

    def __init__(self, config: ContactWorldPi0Config, *, rngs: nnx.Rngs):
        self.config = config
        d = config.token_dim
        if config.use_visual:
            self.visual_in = nnx.Linear(3, d, rngs=rngs)
            self.visual_out = nnx.Linear(2 * d, d, rngs=rngs)
        if config.use_tactile:
            if config.tactile_representation in {"map_coupled", "map_local"}:
                self.map_conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_out = nnx.Linear(128, d, rngs=rngs)
            elif config.tactile_representation in {"map_split", "map_split_local"}:
                # Each base-force component has its own filters until the
                # final token projection. All three maps keep their signed values.
                self.map_fx1 = nnx.Conv(1, 26, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_fx2 = nnx.Conv(26, 26, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_fy1 = nnx.Conv(1, 26, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_fy2 = nnx.Conv(26, 26, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_fz1 = nnx.Conv(1, 26, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_fz2 = nnx.Conv(26, 26, kernel_size=(3, 3), padding="SAME", rngs=rngs)
                self.map_fx_out = nnx.Linear(52, d, rngs=rngs)
                self.map_fy_out = nnx.Linear(52, d, rngs=rngs)
                self.map_fz_out = nnx.Linear(52, d, rngs=rngs)
            else:
                feature_dim = 10 if config.tactile_representation == "summary" else 6
                self.tactile_in = nnx.Linear(feature_dim, d, rngs=rngs)
                self.tactile_out = nnx.Linear(d if feature_dim == 10 else 2 * d, d, rngs=rngs)

    def _map_tokens(self, force_grid: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        if self.config.tactile_representation in {"map_split", "map_split_local"}:
            branches = ((self.map_fx1, self.map_fx2, self.map_fx_out),
                        (self.map_fy1, self.map_fy2, self.map_fy_out),
                        (self.map_fz1, self.map_fz2, self.map_fz_out))
            outputs = []
            for channel, (first, second, output) in enumerate(branches):
                h = jnn.gelu(first(force_grid[..., channel:channel + 1]))
                h = jnn.gelu(second(h))
                pooled = jnp.concatenate((jnp.mean(h, axis=(1, 2)),
                                          jnp.max(h, axis=(1, 2))), axis=-1)
                outputs.append(output(pooled))
            return tuple(outputs)
        h = jnn.gelu(self.map_conv1(force_grid))
        h = jnn.gelu(self.map_conv2(h))
        token = self.map_out(jnp.concatenate((jnp.mean(h, axis=(1, 2)),
                                              jnp.max(h, axis=(1, 2))), axis=-1))
        return (token,)

    def __call__(self, spatial: SpatialEncoderInput) -> SpatialEncoderOutput:
        tokens, masks = [], []
        if self.config.use_visual:
            if spatial.visual is None:
                raise ValueError("Missing ContactWorld cloud")
            visual = spatial.visual
            xyz = (jnp.asarray(visual.xyz_m, dtype=jnp.float32) -
                   jnp.array([0.5, 0.0, 0.1], dtype=jnp.float32)) / 0.2
            h = jnn.gelu(self.visual_in(xyz))
            mask = jnp.asarray(visual.point_mask, dtype=jnp.bool_)
            mean = jnp.sum(jnp.where(mask[..., None], h, 0), axis=1) / jnp.maximum(jnp.sum(mask, axis=1)[:, None], 1)
            maximum = jnp.max(jnp.where(mask[..., None], h, -1e6), axis=1)
            maximum = jnp.where(jnp.any(mask, axis=1)[:, None], maximum, 0)
            tokens.append(self.visual_out(jnp.concatenate([mean, maximum], axis=-1)))
            masks.append(jnp.any(mask, axis=1))
        if self.config.use_tactile:
            if spatial.tactile is None:
                raise ValueError("Missing ContactWorld force field")
            tactile = spatial.tactile
            force = jnp.asarray(tactile.force, dtype=jnp.float32)
            mask = jnp.asarray(tactile.point_mask, dtype=jnp.bool_)
            if self.config.tactile_representation in {"map_coupled", "map_split", "map_local", "map_split_local"}:
                if force.shape[1:] != (280, 3):
                    raise ValueError("Bilateral tactile map input must have 280 ordered 3D vectors")
                grids = force.reshape((force.shape[0], 2, 10, 14, 3)) / 0.002
                for finger in range(2):
                    finger_mask = jnp.any(mask[:, finger * 140:(finger + 1) * 140], axis=1)
                    for token in self._map_tokens(grids[:, finger]):
                        tokens.append(token)
                        masks.append(finger_mask)
                return SpatialEncoderOutput(tokens=jnp.stack(tokens, axis=1),
                                            token_mask=jnp.stack(masks, axis=1))
            if self.config.tactile_pool_per_finger:
                finger_id = jnp.asarray(tactile.finger_id, dtype=jnp.int32)
                pooling_masks = (mask & (finger_id == 0), mask & (finger_id == 1))
            else:
                pooling_masks = (mask,)
            for pool_mask in pooling_masks:
                if self.config.tactile_representation == "summary":
                    denominator = jnp.maximum(jnp.sum(pool_mask, axis=1, keepdims=True), 1)
                    mean = jnp.sum(jnp.where(pool_mask[..., None], force, 0), axis=1) / denominator
                    absolute = jnp.sum(jnp.where(pool_mask[..., None], jnp.abs(force), 0), axis=1) / denominator
                    rms = jnp.sqrt(jnp.sum(jnp.where(pool_mask[..., None], force ** 2, 0), axis=1) / denominator + 1e-12)
                    fraction = jnp.sum(jnp.logical_and(pool_mask, jnp.linalg.norm(force, axis=-1) > 1e-4), axis=1,
                                       keepdims=True) / denominator
                    summary = jnp.concatenate([mean / 0.002, absolute / 0.002,
                                               rms / 0.002, fraction], axis=-1)
                    token = self.tactile_out(jnn.gelu(self.tactile_in(summary)))
                else:
                    xyz = jnp.asarray(tactile.xyz_m, dtype=jnp.float32)
                    if self.config.tactile_representation in ("ee3d", "base3d"):
                        xyz = (xyz - jnp.array([0.5, 0.0, 0.1], dtype=jnp.float32)) / 0.2
                    feature = jnp.concatenate([xyz, force / 0.002], axis=-1)
                    h = jnn.gelu(self.tactile_in(feature))
                    count = jnp.maximum(jnp.sum(pool_mask, axis=1)[:, None], 1)
                    mean = jnp.sum(jnp.where(pool_mask[..., None], h, 0), axis=1) / count
                    maximum = jnp.max(jnp.where(pool_mask[..., None], h, -1e6), axis=1)
                    maximum = jnp.where(jnp.any(pool_mask, axis=1)[:, None], maximum, 0)
                    token = self.tactile_out(jnp.concatenate([mean, maximum], axis=-1))
                tokens.append(token)
                masks.append(jnp.any(pool_mask, axis=1))
        return SpatialEncoderOutput(tokens=jnp.stack(tokens, axis=1),
                                    token_mask=jnp.stack(masks, axis=1))
