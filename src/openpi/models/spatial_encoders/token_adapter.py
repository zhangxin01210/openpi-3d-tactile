"""
OpenPI 3D + tactile：Spatial Token Adapter
（openpi.models.spatial_encoders.token_adapter）

作用
----
将任意 spatial encoder 输出的 token dimension：

    [B, K, D_encoder]

映射到下游模型需要的 embedding dimension：

    [B, K, D_model]

典型数据流：

    SpatialEncoderInput
        ↓
    任意 SpatialEncoder
        ↓
    SpatialEncoderOutput
        tokens [B,K,D_encoder]
        ↓
    LinearSpatialTokenAdapter
        ↓
    SpatialEncoderOutput
        tokens [B,K,D_model]
        ↓
    π0 prefix

为什么需要单独 adapter
----------------------
不同 spatial encoder 的内部 hidden dimension 很可能不同：

    JointPointNet:
        256

    PointNet++:
        256 / 512

    Point Transformer:
        384 / 512 / 768

    Dual Branch:
        可能更大

如果要求所有 encoder 直接输出 π0 hidden size，
就会把：

    encoder architecture

和：

    downstream model dimension

耦合在一起。

单独 adapter 后：

    encoder 只负责表示学习；
    adapter 只负责接口维度对齐。

这使得后续替换 encoder 时，不需要修改 π0 integration。

第一版为什么只有 Linear
-----------------------
第一版只实现：

    y = Linear(x)

不加入：

    LayerNorm
    GELU
    residual MLP
    resampler
    attention

原因：
这些都会额外改变表示能力，
不应该偷偷混进“维度对齐”层。

如果后续实验需要：

    MLPSpatialTokenAdapter
    ResamplerSpatialTokenAdapter

应作为新的独立模块增加，而不是修改本实现。

输入
----
SpatialEncoderOutput：

    tokens:
        [B,K,D_encoder]

    token_mask:
        [B,K]

    token_xyz_m:
        Optional [B,K,3]

    aux:
        encoder-specific diagnostics

输出
----
新的 SpatialEncoderOutput：

    tokens:
        [B,K,D_model]

其余字段原样保留：

    token_mask
    token_xyz_m
    aux

注意
----
- 本文件依赖 JAX / Flax NNX。
- 本地没有 JAX 时，只需 py_compile。
- adapter 不负责 token 数量 K 的压缩。
- adapter 不负责 token position embedding。
"""

from __future__ import annotations

import dataclasses

from flax import nnx

from openpi.models.spatial_encoders.types import SpatialEncoderOutput


# =============================================================================
# 1. Config
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class LinearSpatialTokenAdapterConfig:
    """
    线性 spatial token adapter 配置。

    input_dim:
        spatial encoder 输出维度。

    output_dim:
        下游模型 embedding dimension。
    """

    input_dim: int
    output_dim: int

    def __post_init__(
        self,
    ) -> None:
        if self.input_dim <= 0:
            raise ValueError(
                "input_dim must be > 0"
            )

        if self.output_dim <= 0:
            raise ValueError(
                "output_dim must be > 0"
            )

    def create(
        self,
        *,
        rngs: nnx.Rngs,
    ) -> "LinearSpatialTokenAdapter":
        return LinearSpatialTokenAdapter(
            config=self,
            rngs=rngs,
        )


# =============================================================================
# 2. Adapter
# =============================================================================

class LinearSpatialTokenAdapter(
    nnx.Module
):
    """
    将 SpatialEncoderOutput.tokens 做线性维度对齐。

    不改变 token 数量 K。
    """

    def __init__(
        self,
        *,
        config: LinearSpatialTokenAdapterConfig,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config

        self.projection = nnx.Linear(
            config.input_dim,
            config.output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        output: SpatialEncoderOutput,
    ) -> SpatialEncoderOutput:
        """
        投影 tokens，并保留其余 spatial token metadata。
        """
        tokens = output.tokens

        if tokens.shape[
            -1
        ] != self.config.input_dim:
            raise ValueError(
                "Spatial token dimension mismatch: "
                f"got {tokens.shape[-1]}, "
                f"expected {self.config.input_dim}"
            )

        projected = self.projection(
            tokens
        )

        return SpatialEncoderOutput(
            tokens=projected,
            token_mask=(
                output.token_mask
            ),
            token_xyz_m=(
                output.token_xyz_m
            ),
            aux=(
                output.aux
            ),
        )
