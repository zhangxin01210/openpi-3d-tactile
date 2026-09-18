"""
OpenPI 3D + tactile：Spatial Conditioning Router
（openpi.models.spatial_encoders.router）

作用
----
把：

    SpatialEncoderInput
        ↓
    任意 SpatialEncoder
        ↓
    SpatialEncoderOutput

路由到：

    PREFIX
    SUFFIX
    BOTH

三个 conditioning target。

核心原则
--------
Spatial encoder 每次只执行一次。

例如 BOTH：

    SpatialEncoderInput
            ↓
       encoder()
            ↓
    SpatialEncoderOutput
        ├── prefix adapter
        │       ↓
        │   [B,K,D_vlm]
        │
        └── suffix adapter
                ↓
            [B,K,D_action]

而不是：

    encoder() for prefix
    encoder() for suffix

这样对未来较重的：

    PointNet++
    Point Transformer
    spatial attention encoder

尤其重要。

模块边界
--------
本模块负责：

    - 调用 spatial encoder 一次
    - 按 conditioning config 创建需要的 adapter
    - 将 encoder tokens 投影到 prefix / suffix width
    - 返回 SpatialConditionedTokens

本模块不负责：

    - spatial 数据读取
    - normalization 的具体策略
    - prefix 中 token 的具体插入顺序
    - suffix 中 token 的具体插入顺序
    - attention mask
    - KV cache
    - diffusion denoise loop

这些由 π0 integration 决定。

为什么 router 不知道具体 encoder 类型
-----------------------------------
构造时直接接收：

    encoder: nnx.Module

因此它可以包：

    JointPointNetEncoder
    DualBranchPointNetEncoder
    PointNet2Encoder
    PointTransformerEncoder
    ...

唯一要求是 encoder forward 满足：

    SpatialEncoderInput
        ->
    SpatialEncoderOutput

为什么 prefix / suffix 使用两个独立 adapter
------------------------------------------
VLM width 与 Action Expert width 可能不同。

因此 BOTH 模式必须是：

    same encoder output
        ├── LinearSpatialTokenAdapter(Denc -> Dvlm)
        └── LinearSpatialTokenAdapter(Denc -> Daction)

而不是共享一个 downstream projection。

推理缓存
--------
本 router 自身不做 cache。

正确的推理方式是：

    conditioned = router(observation.spatial)

只调用一次。

随后：

    embed_prefix(..., conditioned)
    denoise step 1 -> embed_suffix(..., conditioned)
    denoise step 2 -> embed_suffix(..., conditioned)
    ...

这样 spatial encoder 不会随 denoise step 重复计算。

输入
----
    spatial:
        SpatialEncoderInput

输出
----
    SpatialConditionedTokens

PREFIX：
    prefix_tokens [B,K,D_prefix]
    prefix_mask   [B,K]
    suffix_*      None

SUFFIX：
    prefix_*      None
    suffix_tokens [B,K,D_suffix]
    suffix_mask   [B,K]

BOTH：
    两组同时存在

注意
----
- 本文件依赖 JAX / Flax NNX。
- token 数 K 不在 router 中改变。
- adapter 当前仅为 linear projection。
"""

from __future__ import annotations

from flax import nnx

from openpi.models.spatial_encoders.conditioning import (
    SpatialConditionedTokens,
    SpatialConditioningConfig,
    validate_conditioned_tokens,
)
from openpi.models.spatial_encoders.token_adapter import (
    LinearSpatialTokenAdapter,
    LinearSpatialTokenAdapterConfig,
)
from openpi.models.spatial_encoders.types import (
    SpatialEncoderInput,
    SpatialEncoderOutput,
)


# =============================================================================
# 1. Router
# =============================================================================

class SpatialConditioningRouter(
    nnx.Module
):
    """
    任意 SpatialEncoder -> prefix / suffix conditioning tokens。
    """

    def __init__(
        self,
        *,
        encoder: nnx.Module,
        encoder_token_dim: int,
        conditioning: SpatialConditioningConfig,
        prefix_dim: int | None,
        suffix_dim: int | None,
        rngs: nnx.Rngs,
    ) -> None:
        if encoder_token_dim <= 0:
            raise ValueError(
                "encoder_token_dim must be > 0"
            )

        if (
            conditioning.use_prefix
            and (
                prefix_dim is None
                or prefix_dim <= 0
            )
        ):
            raise ValueError(
                "prefix_dim must be > 0 when "
                "prefix conditioning is enabled"
            )

        if (
            conditioning.use_suffix
            and (
                suffix_dim is None
                or suffix_dim <= 0
            )
        ):
            raise ValueError(
                "suffix_dim must be > 0 when "
                "suffix conditioning is enabled"
            )

        self.encoder = encoder

        self.encoder_token_dim = int(
            encoder_token_dim
        )

        self.conditioning = conditioning

        self.prefix_adapter: (
            LinearSpatialTokenAdapter
            | None
        ) = None

        self.suffix_adapter: (
            LinearSpatialTokenAdapter
            | None
        ) = None

        if conditioning.use_prefix:
            assert prefix_dim is not None

            self.prefix_adapter = (
                LinearSpatialTokenAdapterConfig(
                    input_dim=(
                        self.encoder_token_dim
                    ),
                    output_dim=int(
                        prefix_dim
                    ),
                ).create(
                    rngs=rngs
                )
            )

        if conditioning.use_suffix:
            assert suffix_dim is not None

            self.suffix_adapter = (
                LinearSpatialTokenAdapterConfig(
                    input_dim=(
                        self.encoder_token_dim
                    ),
                    output_dim=int(
                        suffix_dim
                    ),
                ).create(
                    rngs=rngs
                )
            )

    # =========================================================================
    # 2. Encode once, route many
    # =========================================================================

    def encode(
        self,
        spatial: SpatialEncoderInput,
    ) -> SpatialEncoderOutput:
        """
        只执行 spatial encoder，不做 downstream projection。

        单独暴露这个方法，方便未来：
            - profiling
            - encoder ablation
            - representation visualization
        """
        output = self.encoder(
            spatial
        )

        if (
            output.tokens.shape[
                -1
            ]
            != self.encoder_token_dim
        ):
            raise ValueError(
                "Spatial encoder output dimension "
                "does not match router config: "
                f"got {output.tokens.shape[-1]}, "
                f"expected {self.encoder_token_dim}"
            )

        return output

    def route(
        self,
        encoded: SpatialEncoderOutput,
    ) -> SpatialConditionedTokens:
        """
        将已经编码好的 spatial tokens 路由到 prefix / suffix。

        这个方法不会再次调用 encoder。
        """
        prefix_tokens = None
        prefix_mask = None

        suffix_tokens = None
        suffix_mask = None

        if self.conditioning.use_prefix:
            if self.prefix_adapter is None:
                raise RuntimeError(
                    "Prefix conditioning is enabled "
                    "but prefix_adapter is missing"
                )

            prefix_output = (
                self.prefix_adapter(
                    encoded
                )
            )

            prefix_tokens = (
                prefix_output.tokens
            )

            prefix_mask = (
                prefix_output.token_mask
            )

        if self.conditioning.use_suffix:
            if self.suffix_adapter is None:
                raise RuntimeError(
                    "Suffix conditioning is enabled "
                    "but suffix_adapter is missing"
                )

            suffix_output = (
                self.suffix_adapter(
                    encoded
                )
            )

            suffix_tokens = (
                suffix_output.tokens
            )

            suffix_mask = (
                suffix_output.token_mask
            )

        conditioned = SpatialConditionedTokens(
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            suffix_tokens=suffix_tokens,
            suffix_mask=suffix_mask,
        )

        validate_conditioned_tokens(
            config=self.conditioning,
            tokens=conditioned,
        )

        return conditioned

    def __call__(
        self,
        spatial: SpatialEncoderInput,
    ) -> SpatialConditionedTokens:
        """
        标准路径：

            encode once
                ↓
            route to prefix/suffix
        """
        encoded = self.encode(
            spatial
        )

        return self.route(
            encoded
        )
