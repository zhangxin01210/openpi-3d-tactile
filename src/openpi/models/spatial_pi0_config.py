"""
OpenPI 3D + tactile：可插拔 Spatial π0 配置
（openpi.models.spatial_pi0_config）

作用
----
把两个实验维度明确分开：

    1. spatial 怎么编码
       - JointPointNet
       - DualBranchPointNet
       - PointNet++
       - Point Transformer
       - ...

    2. 编码后的 spatial token 注入哪里
       - PREFIX：VLM / PaliGemma prefix
       - SUFFIX：Action Expert suffix
       - BOTH：两边同时注入

典型实验：

    JointPointNetPi0Config(
        conditioning=SpatialConditioningConfig(
            target="prefix"
        )
    )

    JointPointNetPi0Config(
        conditioning=SpatialConditioningConfig(
            target="suffix"
        )
    )

    JointPointNetPi0Config(
        conditioning=SpatialConditioningConfig(
            target="both"
        )
    )

这样 PREFIX / SUFFIX / BOTH 是不同 config 实例，
不需要复制三套 π0 代码。

模块边界
--------
SpatialPi0Config 只负责：

    - spatial input spec
    - 创建具体 spatial encoder
    - 声明 encoder token dimension
    - 声明 conditioning target
    - 声明 visual / tactile modality 是否启用

真正的 prefix / suffix 注入逻辑在 pi0.py，
真正的 token projection / routing 在：

    spatial_encoders/router.py

为什么保留 use_visual / use_tactile
---------------------------------
derived spatial/v1 永远保存完整 visual + tactile 数据。

但实验可以只改变模型 config：

    visual-only
    tactile-only
    visual+tactile

而不重新生成数据。

因此 modality ablation 属于 model config，
不属于 dataset contract。

当前 baseline
-------------
JointPointNetPi0Config：

    visual + tactile
        ↓
    single-branch early fusion
        ↓
    JointPointNet
        ↓
    global spatial token
        ↓
    SpatialConditioningRouter

注意
----
- 普通 upstream Pi0Config 完全不受影响。
- conditioning.enabled=False 时，不创建 spatial branch。
- visual_points / tactile_points 只描述 input spec，
  不限制未来 encoder 内部再次采样或聚合。
"""

from __future__ import annotations

import abc
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from openpi.models.spatial_encoders.conditioning import (
    SpatialConditioningConfig,
)
from openpi.models.spatial_encoders.joint_pointnet import (
    JointPointNetEncoder,
    JointPointNetEncoderConfig,
)
from openpi.models.spatial_encoders.structured_spatial_encoder import (
    StructuredSpatialEncoder,
    StructuredSpatialEncoderConfig,
)
from openpi.models.spatial_encoders.types import (
    SpatialEncoderInput,
    TactileSpatialInput,
    VisualSpatialInput,
)


# =============================================================================
# 1. Generic spatial π0 config
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class SpatialPi0Config(
    pi0_config.Pi0Config,
    abc.ABC,
):
    """
    所有 spatial π0 variant 的公共 config。
    """

    conditioning: SpatialConditioningConfig = (
        dataclasses.field(
            default_factory=lambda: (
                SpatialConditioningConfig(
                    target="suffix"
                )
            )
        )
    )

    # Modality ablation 由模型配置控制，不修改 derived dataset。
    use_visual: bool = True
    use_tactile: bool = True

    # 当前 spatial/v1 的默认 point counts。
    visual_points: int = 4096
    tactile_points: int = 600

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()

        if self.conditioning.enabled:
            if not (
                self.use_visual
                or self.use_tactile
            ):
                raise ValueError(
                    "Spatial conditioning is enabled, "
                    "but both use_visual and use_tactile "
                    "are False."
                )

        if self.visual_points <= 0:
            raise ValueError(
                "visual_points must be > 0"
            )

        if self.tactile_points <= 0:
            raise ValueError(
                "tactile_points must be > 0"
            )

    @property
    @abc.abstractmethod
    def spatial_encoder_token_dim(
        self,
    ) -> int:
        """
        SpatialEncoderOutput.tokens 的最后一维。
        """

    @abc.abstractmethod
    def create_spatial_encoder(
        self,
        *,
        rngs: nnx.Rngs,
    ) -> nnx.Module:
        """
        创建具体 spatial encoder。
        """

    @override
    def inputs_spec(
        self,
        *,
        batch_size: int = 1,
    ) -> tuple[
        _model.Observation,
        _model.Actions,
    ]:
        """
        在 upstream Pi0Config input spec 上加入 spatial。

        conditioning.enabled=False 时，
        完全返回 upstream spec。
        """
        observation, actions = (
            super().inputs_spec(
                batch_size=batch_size
            )
        )

        if not self.conditioning.enabled:
            return (
                observation,
                actions,
            )

        visual = None

        if self.use_visual:
            visual = VisualSpatialInput(
                xyz_m=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.visual_points,
                        3,
                    ],
                    jnp.float32,
                ),
                rgb=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.visual_points,
                        3,
                    ],
                    jnp.uint8,
                ),
                rgb_valid=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.visual_points,
                    ],
                    jnp.bool_,
                ),
                point_mask=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.visual_points,
                    ],
                    jnp.bool_,
                ),
            )

        tactile = None

        if self.use_tactile:
            tactile = TactileSpatialInput(
                xyz_m=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.tactile_points,
                        3,
                    ],
                    jnp.float32,
                ),
                force=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.tactile_points,
                        3,
                    ],
                    jnp.float32,
                ),
                force_norm=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.tactile_points,
                    ],
                    jnp.float32,
                ),
                finger_id=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.tactile_points,
                    ],
                    jnp.int32,
                ),
                taxel_id=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.tactile_points,
                    ],
                    jnp.int32,
                ),
                point_mask=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.tactile_points,
                    ],
                    jnp.bool_,
                ),
            )

        with at.disable_typechecking():
            observation = dataclasses.replace(
                observation,
                spatial=SpatialEncoderInput(
                    visual=visual,
                    tactile=tactile,
                ),
            )

        return (
            observation,
            actions,
        )


# =============================================================================
# 2. First encoder baseline
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class JointPointNetPi0Config(
    SpatialPi0Config
):
    """
    Single-branch early-fusion baseline。

    默认仍保持轻量：

        13-D joint point feature
            ↓
        32 -> 64 -> 128
            ↓
        mean + max
            ↓
        128-D global token

    conditioning 默认是 suffix，
    但 PREFIX / BOTH 只需要修改 config 实例。
    """

    encoder: JointPointNetEncoderConfig = (
        dataclasses.field(
            default_factory=lambda: (
                JointPointNetEncoderConfig(
                    hidden_dims=(
                        32,
                        64,
                        128,
                    ),
                    token_dim=128,
                )
            )
        )
    )

    @property
    @override
    def spatial_encoder_token_dim(
        self,
    ) -> int:
        return int(
            self.encoder.token_dim
        )

    @override
    def create_spatial_encoder(
        self,
        *,
        rngs: nnx.Rngs,
    ) -> JointPointNetEncoder:
        return self.encoder.create(
            rngs=rngs
        )


# =============================================================================
# 3. Structured spatial encoder
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class StructuredSpatialPi0Config(
    SpatialPi0Config
):
    """
    Structured RGB-D + tactile encoder variant.

    This keeps the same spatial conditioning interfaces as JointPointNet, but
    produces local visual tokens and per-finger tactile tokens.  Defaults:

        visual: 32 local SAT-style tokens + 1 global token
        tactile: 5 fingers * (4 local tokens + 1 summary token)
        total: 58 tokens, each 128-D
    """

    encoder: StructuredSpatialEncoderConfig = (
        dataclasses.field(
            default_factory=StructuredSpatialEncoderConfig
        )
    )

    @property
    @override
    def spatial_encoder_token_dim(
        self,
    ) -> int:
        return int(
            self.encoder.token_dim
        )

    @override
    def create_spatial_encoder(
        self,
        *,
        rngs: nnx.Rngs,
    ) -> StructuredSpatialEncoder:
        return self.encoder.create(
            rngs=rngs
        )
