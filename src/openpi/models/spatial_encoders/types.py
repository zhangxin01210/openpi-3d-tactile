"""
OpenPI 3D + tactile：Spatial Encoder 插槽类型协议
（openpi.models.spatial_encoders.types）

作用
----
定义所有 spatial encoder 共享的输入 / 输出数据契约。

本文件不实现任何网络层，也不决定：

    - PointNet / PointNet++
    - Point Transformer
    - visual / tactile early fusion
    - dual branch
    - cross attention
    - Perceiver resampler
    - token 数量
    - force normalization
    - contact threshold

它只回答：

    “不管以后换什么 spatial encoder，
     数据应该以什么结构进入，
     encoder 最终应该返回什么？”

整体边界
--------
    SpatialDerivedDataset
            ↓
    model-side transform
            ↓
    SpatialEncoderInput
            ↓
      任意 SpatialEncoder
            ↓
    SpatialEncoderOutput
            ↓
    SpatialTokenAdapter
            ↓
           π0

为什么 visual / tactile 不直接拼接
---------------------------------
如果这里直接固定：

    [visual points ; tactile points]

就等于提前假设：

    “visual 与 tactile 应该 early fusion”

这会阻碍后续实验，例如：

    visual -> PointNet++ --------┐
                                ├-> cross attention
    tactile -> tactile encoder --┘

因此输入契约保留两个 modality 的原始结构。

为什么同时保留 rgb_valid 和 point_mask
--------------------------------------
二者语义不同：

rgb_valid:
    该 visual 3D point 存在，
    但是否成功关联到了可靠 RGB。

point_mask:
    该 point 本身是否是真实输入点。

当前 V1 固定：
    Nv = 4096
    Nt = 600

所以 point_mask 通常全 True。

但未来如果：
    - variable-N
    - padding
    - point dropout
    - 某个 modality 缺失

point_mask 可以继续使用，而不用修改 encoder API。

输入 contract
-------------
VisualSpatialInput：

    xyz_m:
        [B, Nv, 3] float
        base_link 下的 metric XYZ，单位 m。

    rgb:
        [B, Nv, 3]
        颜色值。
        具体是 uint8 [0,255] 还是 normalized float，
        由 model-side transform contract 决定。

    rgb_valid:
        [B, Nv] bool

    point_mask:
        [B, Nv] bool

TactileSpatialInput：

    xyz_m:
        [B, Nt, 3] float
        taxel 位置，base_link，单位 m。

    force:
        [B, Nt, 3] float
        当前仍为 dataset_native。

    force_norm:
        [B, Nt] float

    finger_id:
        [B, Nt] integer

    taxel_id:
        [B, Nt] integer

    point_mask:
        [B, Nt] bool

SpatialEncoderInput：

    visual:
        VisualSpatialInput | None

    tactile:
        TactileSpatialInput | None

允许 None 的原因：
    visual-only / tactile-only 消融不应该改接口。

输出 contract
-------------
SpatialEncoderOutput：

    tokens:
        [B, K, D_encoder]

    token_mask:
        [B, K] bool

    token_xyz_m:
        Optional [B, K, 3]

        如果 encoder 输出 token 有明确空间 anchor，
        可以提供 base_link metric position。

        如果是：
            global token
            learned latent
            Perceiver queries

        则可以为 None。

    aux:
        dict[str, Array]

        可选诊断信息，例如：
            attention weights
            contact score
            branch tokens
            pooling weights

        π0 integration 不应该依赖 aux。

重要原则
--------
1. Nv / Nt / K / D 都不在本文件写死。
2. input 保存 modality structure，不保存某个 encoder 的 feature layout。
3. output 保存通用 token contract，不暴露 encoder 内部实现。
4. normalization / augmentation 属于 transform，不属于 types。
5. modality embedding / finger embedding / taxel embedding 属于模型假设，
   不属于 raw data contract。
"""

from __future__ import annotations

from typing import Generic
from typing import TypeVar

from flax import struct


# =============================================================================
# 1. Generic array type
# =============================================================================

# 这里故意不把 ArrayT 绑定到 jax.Array。
#
# 原因：
# - data transform / smoke test 可能使用 NumPy；
# - JAX model 使用 jax.Array；
# - 将来若需要 PyTorch encoder，也不必修改这个数据契约。
#
# 真正的 backend-specific typing 可以在具体 encoder 文件中收紧。
ArrayT = TypeVar(
    "ArrayT"
)


# =============================================================================
# 2. Visual modality
# =============================================================================

@struct.dataclass
class VisualSpatialInput(
    Generic[
        ArrayT
    ]
):
    """
    Visual point branch 的结构化输入。

    Shape contract
    --------------
    xyz_m:
        [B, Nv, 3]

    rgb:
        [B, Nv, 3]

    rgb_valid:
        [B, Nv]

    point_mask:
        [B, Nv]
    """

    xyz_m: ArrayT
    rgb: ArrayT
    rgb_valid: ArrayT
    point_mask: ArrayT


# =============================================================================
# 3. Tactile modality
# =============================================================================

@struct.dataclass
class TactileSpatialInput(
    Generic[
        ArrayT
    ]
):
    """
    Tactile point branch 的结构化输入。

    Shape contract
    --------------
    xyz_m:
        [B, Nt, 3]

    force:
        [B, Nt, 3]

    force_norm:
        [B, Nt]

    finger_id:
        [B, Nt]

    taxel_id:
        [B, Nt]

    point_mask:
        [B, Nt]
    """

    xyz_m: ArrayT
    force: ArrayT
    force_norm: ArrayT
    finger_id: ArrayT
    taxel_id: ArrayT
    point_mask: ArrayT


# =============================================================================
# 4. Encoder input
# =============================================================================

@struct.dataclass
class SpatialEncoderInput(
    Generic[
        ArrayT
    ]
):
    """
    所有 spatial encoder 的统一输入。

    visual / tactile 可以独立为 None。

    典型配置：

    visual-only:
        visual != None
        tactile == None

    tactile-only:
        visual == None
        tactile != None

    multimodal:
        visual != None
        tactile != None

    注意：
        “两个 modality 如何融合”
        不在此处决定。
    """

    visual: VisualSpatialInput[
        ArrayT
    ] | None = None

    tactile: TactileSpatialInput[
        ArrayT
    ] | None = None


# =============================================================================
# 5. Encoder output
# =============================================================================

@struct.dataclass
class SpatialEncoderOutput(
    Generic[
        ArrayT
    ]
):
    """
    任意 spatial encoder 对外统一输出。

    Shape contract
    --------------
    tokens:
        [B, K, D_encoder]

    token_mask:
        [B, K]

    token_xyz_m:
        Optional [B, K, 3]

    aux:
        encoder-specific diagnostic arrays。

    K 可以因 encoder 不同而不同，例如：

        global PointNet:
            K = 1

        per-finger tactile:
            K = 5

        PointNet++:
            K = 64 / 128 / 256

        learned resampler:
            K = 32 / 64 / 128
    """

    tokens: ArrayT
    token_mask: ArrayT

    token_xyz_m: ArrayT | None = None

    aux: dict[
        str,
        ArrayT,
    ] = struct.field(
        default_factory=dict
    )
