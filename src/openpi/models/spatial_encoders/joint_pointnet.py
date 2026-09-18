"""
OpenPI 3D + tactile：Joint PointNet Spatial Encoder
（openpi.models.spatial_encoders.joint_pointnet）

作用
----
提供第一版最轻量的 spatial encoder baseline：

    visual points
         +
    tactile points
         ↓
    early fusion
         ↓
    shared point MLP
         ↓
    masked mean + max pooling
         ↓
    one global spatial token

这是一条 deliberately simple 的 baseline。

它不是为了提前追求最强性能，而是为了验证：

    “将 visual 3D + tactile 3D 合并为统一 point set，
     是否能稳定产生可供 π0 使用的 spatial representation？”

后续更强的方法，例如：

    DualBranchPointNet
    PointNet++
    Joint PointNet++
    Point Transformer
    Cross Attention
    Spatial Attention

都应该实现相同的：

    SpatialEncoderInput
        ->
    SpatialEncoderOutput

contract，而不修改 dataset / derived data / Observation。

统一 point feature
------------------
第一版每个 joint point 使用 13 维 feature：

    xyz                     3
    rgb                     3
    rgb_valid               1
    force_xyz               3
    force_norm              1
    modality_one_hot        2
    --------------------------------
                            13

visual point：

    [xyz, rgb, rgb_valid, 0,0,0, 0, 1,0]

tactile point：

    [xyz, 0,0,0, 0, force_xyz, force_norm, 0,1]

为什么必须保留 modality identity
-------------------------------
visual 与 tactile point 的 feature semantics 不同：

    visual:
        RGB

    tactile:
        force

如果只用补零而不告诉网络 modality，
“真实零值”与“该字段不存在”会发生语义混淆。

因此第一版使用最简单、无 learnable parameter 的：

    visual  -> [1,0]
    tactile -> [0,1]

为什么暂时不用 finger_id / taxel_id
----------------------------------
它们已经保存在 SpatialEncoderInput 中，但第一版 baseline 不使用。

原因：

    - 直接把 integer ID 当连续数值没有合理几何意义；
    - 使用 embedding 会增加额外结构假设；
    - 当前目标是得到最小 early-fusion baseline。

以后可以独立增加：

    finger embedding
    taxel embedding

作为明确消融项。

为什么只输出一个 token
----------------------
第一版使用 PointNet-style symmetric global pooling：

    point features
        ↓
    shared MLP
        ↓
    masked max pooling
      +
    masked mean pooling
        ↓
    projection
        ↓
    [B,1,D]

这样不引入：

    FPS
    neighborhood grouping
    learned queries
    cross attention
    voxel tokenization

因此非常适合作为最简单 baseline。

一个 global token 会损失局部结构，
这不是 bug，而是该 baseline 的明确能力边界。

后续 PointNet++ / spatial attention 的意义之一，
就是验证保留局部 spatial tokens 是否进一步提升效果。

输入
----
SpatialEncoderInput：

    visual:
        xyz_m          [B,Nv,3]
        rgb            [B,Nv,3]
        rgb_valid      [B,Nv]
        point_mask     [B,Nv]

    tactile:
        xyz_m          [B,Nt,3]
        force          [B,Nt,3]
        force_norm     [B,Nt]
        finger_id      [B,Nt]   # 第一版不使用
        taxel_id       [B,Nt]   # 第一版不使用
        point_mask     [B,Nt]

允许：

    visual-only
    tactile-only
    multimodal

输出
----
SpatialEncoderOutput：

    tokens:
        [B,1,token_dim]

    token_mask:
        [B,1]

    token_xyz_m:
        None

因为 global token 不对应一个明确的局部 3D anchor。

    aux:
        visual_point_count [B]
        tactile_point_count [B]
        total_point_count [B]

基本配置
--------
默认：

    hidden_dims = (64, 128, 256)
    token_dim = 256

参数量非常小，适合先验证整体链路。

Feature transform
-----------------
encoder 不修改 raw input，而是内部调用：

    SpatialFeatureTransforms

因此同一个 derived dataset 可以配置不同：

    XYZ normalization
    RGB scaling
    force scaling

当前默认：

    XYZ   identity
    RGB   uint8 -> [0,1]
    force identity

注意
----
- 本文件依赖 JAX / Flax NNX。
- 用户本地如果没有 JAX，只需要做 py_compile；
  真正 forward smoke test 留到服务器环境。
- 该 baseline 不代表最终推荐架构。
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.spatial_encoders.transforms import SpatialFeatureTransforms
from openpi.models.spatial_encoders.types import SpatialEncoderInput
from openpi.models.spatial_encoders.types import SpatialEncoderOutput


# =============================================================================
# 1. Config
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class JointPointNetEncoderConfig:
    """
    Joint PointNet baseline 配置。

    hidden_dims:
        shared point MLP 的隐藏维度。

    token_dim:
        最终 SpatialEncoderOutput token dimension。

    transforms:
        字段级 feature transform。
        它不包含 learnable parameter。
    """

    hidden_dims: tuple[
        int,
        ...,
    ] = (
        64,
        128,
        256,
    )

    token_dim: int = 256

    transforms: SpatialFeatureTransforms = (
        dataclasses.field(
            default_factory=(
                SpatialFeatureTransforms
            )
        )
    )

    def __post_init__(
        self,
    ) -> None:
        if not self.hidden_dims:
            raise ValueError(
                "hidden_dims cannot be empty"
            )

        if any(
            dimension <= 0
            for dimension in self.hidden_dims
        ):
            raise ValueError(
                "All hidden_dims must be > 0"
            )

        if self.token_dim <= 0:
            raise ValueError(
                "token_dim must be > 0"
            )

    def create(
        self,
        *,
        rngs: nnx.Rngs,
    ) -> "JointPointNetEncoder":
        """
        创建 encoder module。

        所有 spatial encoder config 后续都应遵循相同风格：

            config.create(rngs=...)
        """
        return JointPointNetEncoder(
            config=self,
            rngs=rngs,
        )


# =============================================================================
# 2. Joint feature packing
# =============================================================================

def _visual_features(
    inputs,
    *,
    transforms: SpatialFeatureTransforms,
) -> tuple[
    jax.Array,
    jax.Array,
]:
    """
    visual branch -> unified 13-D point feature。

    返回：
        features   [B,Nv,13]
        point_mask [B,Nv]
    """
    xyz = transforms.visual_xyz(
        inputs.xyz_m
    )

    rgb = transforms.rgb(
        inputs.rgb,
        inputs.rgb_valid,
    )

    rgb_valid = (
        inputs.rgb_valid.astype(
            jnp.float32
        )[
            ...,
            None,
        ]
    )

    batch_size = xyz.shape[
        0
    ]

    point_count = xyz.shape[
        1
    ]

    zeros_force = jnp.zeros(
        (
            batch_size,
            point_count,
            3,
        ),
        dtype=jnp.float32,
    )

    zeros_force_norm = jnp.zeros(
        (
            batch_size,
            point_count,
            1,
        ),
        dtype=jnp.float32,
    )

    modality = jnp.broadcast_to(
        jnp.asarray(
            [
                1.0,
                0.0,
            ],
            dtype=jnp.float32,
        ),
        (
            batch_size,
            point_count,
            2,
        ),
    )

    features = jnp.concatenate(
        [
            xyz.astype(
                jnp.float32
            ),
            rgb.astype(
                jnp.float32
            ),
            rgb_valid,
            zeros_force,
            zeros_force_norm,
            modality,
        ],
        axis=-1,
    )

    return (
        features,
        inputs.point_mask.astype(
            jnp.bool_
        ),
    )


def _tactile_features(
    inputs,
    *,
    transforms: SpatialFeatureTransforms,
) -> tuple[
    jax.Array,
    jax.Array,
]:
    """
    tactile branch -> unified 13-D point feature。

    finger_id / taxel_id 在第一版 baseline 中故意不使用。

    返回：
        features   [B,Nt,13]
        point_mask [B,Nt]
    """
    xyz = transforms.tactile_xyz(
        inputs.xyz_m
    )

    force_output = transforms.force(
        inputs.force,
        inputs.force_norm,
    )

    force = force_output.force

    force_norm = (
        force_output.force_norm[
            ...,
            None,
        ]
    )

    batch_size = xyz.shape[
        0
    ]

    point_count = xyz.shape[
        1
    ]

    zeros_rgb = jnp.zeros(
        (
            batch_size,
            point_count,
            3,
        ),
        dtype=jnp.float32,
    )

    zeros_rgb_valid = jnp.zeros(
        (
            batch_size,
            point_count,
            1,
        ),
        dtype=jnp.float32,
    )

    modality = jnp.broadcast_to(
        jnp.asarray(
            [
                0.0,
                1.0,
            ],
            dtype=jnp.float32,
        ),
        (
            batch_size,
            point_count,
            2,
        ),
    )

    features = jnp.concatenate(
        [
            xyz.astype(
                jnp.float32
            ),
            zeros_rgb,
            zeros_rgb_valid,
            force.astype(
                jnp.float32
            ),
            force_norm.astype(
                jnp.float32
            ),
            modality,
        ],
        axis=-1,
    )

    return (
        features,
        inputs.point_mask.astype(
            jnp.bool_
        ),
    )


def build_joint_point_features(
    inputs: SpatialEncoderInput,
    *,
    transforms: SpatialFeatureTransforms,
) -> tuple[
    jax.Array,
    jax.Array,
    dict[
        str,
        jax.Array,
    ],
]:
    """
    将 visual / tactile point 合并为统一 point set。

    输出：
        features:
            [B,N_joint,13]

        mask:
            [B,N_joint]

        aux:
            modality point counts
    """
    feature_parts = []
    mask_parts = []

    visual_count = None
    tactile_count = None

    batch_size = None

    if inputs.visual is not None:
        (
            visual_features,
            visual_mask,
        ) = _visual_features(
            inputs.visual,
            transforms=transforms,
        )

        feature_parts.append(
            visual_features
        )

        mask_parts.append(
            visual_mask
        )

        batch_size = (
            visual_features.shape[
                0
            ]
        )

        visual_count = jnp.sum(
            visual_mask,
            axis=1,
            dtype=jnp.int32,
        )

    if inputs.tactile is not None:
        (
            tactile_features,
            tactile_mask,
        ) = _tactile_features(
            inputs.tactile,
            transforms=transforms,
        )

        if (
            batch_size
            is not None
            and tactile_features.shape[
                0
            ]
            != batch_size
        ):
            raise ValueError(
                "visual and tactile batch sizes "
                "must match"
            )

        feature_parts.append(
            tactile_features
        )

        mask_parts.append(
            tactile_mask
        )

        batch_size = (
            tactile_features.shape[
                0
            ]
        )

        tactile_count = jnp.sum(
            tactile_mask,
            axis=1,
            dtype=jnp.int32,
        )

    if not feature_parts:
        raise ValueError(
            "SpatialEncoderInput must contain "
            "visual and/or tactile input"
        )

    features = jnp.concatenate(
        feature_parts,
        axis=1,
    )

    mask = jnp.concatenate(
        mask_parts,
        axis=1,
    )

    assert batch_size is not None

    if visual_count is None:
        visual_count = jnp.zeros(
            (
                batch_size,
            ),
            dtype=jnp.int32,
        )

    if tactile_count is None:
        tactile_count = jnp.zeros(
            (
                batch_size,
            ),
            dtype=jnp.int32,
        )

    aux = {
        "visual_point_count": (
            visual_count
        ),
        "tactile_point_count": (
            tactile_count
        ),
        "total_point_count": (
            visual_count
            + tactile_count
        ),
    }

    return (
        features,
        mask,
        aux,
    )


# =============================================================================
# 3. Shared point MLP
# =============================================================================

class JointPointNetEncoder(
    nnx.Module
):
    """
    最轻量 early-fusion PointNet-style encoder。
    """

    # 当前 unified point feature contract 是固定 13 维。
    _INPUT_DIM = 13

    def __init__(
        self,
        *,
        config: JointPointNetEncoderConfig,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config

        dimensions = (
            self._INPUT_DIM,
            *config.hidden_dims,
        )

        self.point_layers = nnx.List(
            [
                nnx.Linear(
                    dimensions[
                        index
                    ],
                    dimensions[
                        index
                        + 1
                    ],
                    rngs=rngs,
                )
                for index
                in range(
                    len(
                        dimensions
                    )
                    - 1
                )
            ]
        )

        pooled_dim = (
            config.hidden_dims[
                -1
            ]
            * 2
        )

        self.token_projection = nnx.Linear(
            pooled_dim,
            config.token_dim,
            rngs=rngs,
        )

    # =========================================================================
    # 4. Forward
    # =========================================================================

    def __call__(
        self,
        inputs: SpatialEncoderInput,
    ) -> SpatialEncoderOutput:
        (
            point_features,
            point_mask,
            aux,
        ) = build_joint_point_features(
            inputs,
            transforms=(
                self.config.transforms
            ),
        )

        hidden = point_features

        for layer in self.point_layers:
            hidden = layer(
                hidden
            )

            hidden = jax.nn.gelu(
                hidden
            )

        (
            pooled_mean,
            pooled_max,
            valid_sample,
        ) = _masked_global_pool(
            hidden,
            point_mask,
        )

        pooled = jnp.concatenate(
            [
                pooled_mean,
                pooled_max,
            ],
            axis=-1,
        )

        token = self.token_projection(
            pooled
        )

        token = jax.nn.gelu(
            token
        )

        tokens = token[
            :,
            None,
            :,
        ]

        token_mask = valid_sample[
            :,
            None,
        ]

        return SpatialEncoderOutput(
            tokens=tokens,
            token_mask=token_mask,
            token_xyz_m=None,
            aux=aux,
        )


# =============================================================================
# 5. Masked symmetric pooling
# =============================================================================

def _masked_global_pool(
    features: jax.Array,
    mask: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """
    PointNet-style symmetric pooling。

    输入：
        features [B,N,D]
        mask     [B,N]

    输出：
        mean_pool    [B,D]
        max_pool     [B,D]
        valid_sample [B]

    对极端的“整帧 point 全无效”情况：
        pooled feature 置零
        token_mask=False
    """
    mask = mask.astype(
        jnp.bool_
    )

    valid_sample = jnp.any(
        mask,
        axis=1,
    )

    mask_float = mask.astype(
        features.dtype
    )[
        ...,
        None,
    ]

    point_count = jnp.sum(
        mask_float,
        axis=1,
    )

    safe_count = jnp.maximum(
        point_count,
        jnp.asarray(
            1.0,
            dtype=features.dtype,
        ),
    )

    pooled_mean = jnp.sum(
        features
        * mask_float,
        axis=1,
    ) / safe_count

    negative_infinity = jnp.asarray(
        -jnp.inf,
        dtype=features.dtype,
    )

    masked_features = jnp.where(
        mask[
            ...,
            None,
        ],
        features,
        negative_infinity,
    )

    pooled_max = jnp.max(
        masked_features,
        axis=1,
    )

    pooled_mean = jnp.where(
        valid_sample[
            :,
            None,
        ],
        pooled_mean,
        jnp.zeros_like(
            pooled_mean
        ),
    )

    pooled_max = jnp.where(
        valid_sample[
            :,
            None,
        ],
        pooled_max,
        jnp.zeros_like(
            pooled_max
        ),
    )

    return (
        pooled_mean,
        pooled_max,
        valid_sample,
    )
