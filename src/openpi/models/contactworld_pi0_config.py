"""ContactWorld-specific spatial tokens for USB modality experiments."""

from __future__ import annotations

import dataclasses
from typing import Literal

from flax import nnx
import jax.nn as jnn
import jax.numpy as jnp

from openpi.models.spatial_encoders.types import SpatialEncoderInput
from openpi.models.spatial_encoders.types import SpatialEncoderOutput
from openpi.models.spatial_pi0_config import SpatialPi0Config

TactileRepresentation = Literal["none", "grid", "summary", "ee3d"]


@dataclasses.dataclass(frozen=True)
class ContactWorldPi0Config(SpatialPi0Config):
    tactile_representation: TactileRepresentation = "grid"
    token_dim: int = 128
    split_route: bool = False
    visual_points: int = 1024
    tactile_points: int = 140

    def __post_init__(self):
        super().__post_init__()
        if self.split_route and (not self.use_visual or not self.use_tactile or
                                 not self.conditioning.use_prefix or not self.conditioning.use_suffix):
            raise ValueError("Split route needs both modalities and both conditioning targets")
        if self.use_tactile and self.tactile_representation == "none":
            raise ValueError("Force branch enabled without a representation")

    @property
    def spatial_encoder_token_dim(self) -> int:
        return self.token_dim

    def create_spatial_encoder(self, *, rngs: nnx.Rngs) -> nnx.Module:
        return ContactWorldEncoder(self, rngs=rngs)


class ContactWorldEncoder(nnx.Module):
    """One geometry token and one force token; modality order is stable."""

    def __init__(self, config: ContactWorldPi0Config, *, rngs: nnx.Rngs):
        self.config = config
        d = config.token_dim
        if config.use_visual:
            self.visual_in = nnx.Linear(3, d, rngs=rngs)
            self.visual_out = nnx.Linear(2 * d, d, rngs=rngs)
        if config.use_tactile:
            feature_dim = 10 if config.tactile_representation == "summary" else 6
            self.tactile_in = nnx.Linear(feature_dim, d, rngs=rngs)
            self.tactile_out = nnx.Linear(d if feature_dim == 10 else 2 * d, d, rngs=rngs)

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
            if self.config.tactile_representation == "summary":
                denominator = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1)
                mean = jnp.sum(jnp.where(mask[..., None], force, 0), axis=1) / denominator
                absolute = jnp.sum(jnp.where(mask[..., None], jnp.abs(force), 0), axis=1) / denominator
                rms = jnp.sqrt(jnp.sum(jnp.where(mask[..., None], force ** 2, 0), axis=1) / denominator + 1e-12)
                fraction = jnp.sum(jnp.logical_and(mask, jnp.linalg.norm(force, axis=-1) > 1e-4), axis=1,
                                   keepdims=True) / denominator
                summary = jnp.concatenate([mean / 0.002, absolute / 0.002,
                                           rms / 0.002, fraction], axis=-1)
                token = self.tactile_out(jnn.gelu(self.tactile_in(summary)))
            else:
                xyz = jnp.asarray(tactile.xyz_m, dtype=jnp.float32)
                if self.config.tactile_representation == "ee3d":
                    xyz = (xyz - jnp.array([0.5, 0.0, 0.1], dtype=jnp.float32)) / 0.2
                feature = jnp.concatenate([xyz, force / 0.002], axis=-1)
                h = jnn.gelu(self.tactile_in(feature))
                count = jnp.maximum(jnp.sum(mask, axis=1)[:, None], 1)
                mean = jnp.sum(jnp.where(mask[..., None], h, 0), axis=1) / count
                maximum = jnp.max(jnp.where(mask[..., None], h, -1e6), axis=1)
                maximum = jnp.where(jnp.any(mask, axis=1)[:, None], maximum, 0)
                token = self.tactile_out(jnp.concatenate([mean, maximum], axis=-1))
            tokens.append(token)
            masks.append(jnp.any(mask, axis=1))
        return SpatialEncoderOutput(tokens=jnp.stack(tokens, axis=1),
                                    token_mask=jnp.stack(masks, axis=1))
