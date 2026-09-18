"""
OpenPI 3D + tactile：Spatial Conditioning 路由协议
（openpi.models.spatial_encoders.conditioning）

作用
----
定义 spatial encoder 输出应该被注入到 π0 哪个位置。

本文件只定义“路由 contract”，不实现具体网络，也不修改 π0。

两个实验轴必须保持正交：

    轴 1：Spatial Encoder
        JointPointNet
        DualBranchPointNet
        PointNet++
        Point Transformer
        ...

    轴 2：Spatial Conditioning Target
        PREFIX
        SUFFIX
        BOTH

这样可以形成干净实验矩阵：

    JointPointNet + PREFIX
    JointPointNet + SUFFIX
    JointPointNet + BOTH

    PointNet++ + PREFIX
    PointNet++ + SUFFIX
    PointNet++ + BOTH

而不需要为每个组合复制模型代码。

三个 target 的语义
------------------
PREFIX
    spatial token 进入 VLM / PaliGemma prefix。

    概念上：
        image + spatial + language
            ↓
        VLM context
            ↓
        action expert

SUFFIX
    spatial token 直接进入 action expert suffix。

    概念上：
        VLM context
            ↓
        spatial + state + noisy action
            ↓
        action expert

BOTH
    同一个 spatial encoder output
    经过两个独立 adapter：

        encoder token
            ├── prefix adapter -> VLM width
            └── suffix adapter -> action expert width

    然后同时注入两处。

为什么 BOTH 必须有两个 adapter
------------------------------
PaliGemma/VLM hidden width 与 action expert hidden width
不一定相同。

因此不能假设：

    one adapter fits both

正确设计是：

    SpatialEncoderOutput.tokens
        ├── adapter_prefix
        └── adapter_suffix

这也使 PREFIX / SUFFIX / BOTH 的比较更清晰。

为什么这里不决定具体插入顺序
----------------------------
例如 PREFIX 内部可能是：

    image -> spatial -> language

或者：

    image -> language -> spatial

SUFFIX 内部也可能是：

    spatial -> state -> action

或：

    state -> spatial -> action

这些属于 π0 integration 的具体设计，
需要结合当前 embed_prefix / embed_suffix 结构决定。

因此本文件只表达：

    “是否进入 prefix”
    “是否进入 suffix”

不在这里写死 token 顺序。

为什么不在这里实现 encoder
--------------------------
conditioning 只负责“where”，encoder 负责“how”。

二者分离后：

    换 encoder
        不改 conditioning

    换 injection target
        不改 encoder

这正是后续消融需要的结构。

基本使用
--------
    config = SpatialConditioningConfig(
        target=SpatialInjectionTarget.SUFFIX,
    )

    if config.use_prefix:
        ...

    if config.use_suffix:
        ...

也支持字符串构造：

    SpatialConditioningConfig(
        target="both",
    )

注意
----
- 本文件不依赖 JAX / Flax / Torch。
- 不负责 token dimension projection。
- 不负责 token caching。
- 不负责 prefix/suffix attention mask。
- 不负责 encoder forward。
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Generic
from typing import TypeVar


# =============================================================================
# 1. Injection target
# =============================================================================

class SpatialInjectionTarget(
    str,
    Enum,
):
    """
    Spatial token 注入位置。
    """

    PREFIX = "prefix"
    SUFFIX = "suffix"
    BOTH = "both"


# =============================================================================
# 2. Conditioning config
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class SpatialConditioningConfig:
    """
    Spatial conditioning 路由配置。

    target:
        prefix / suffix / both

    enabled:
        总开关。

        enabled=False 时，即使 target 有值，
        也视为完全关闭 spatial conditioning。

        这个字段主要用于：
            - 快速 no-spatial baseline
            - config ablation
            - 保持训练配置结构一致
    """

    target: SpatialInjectionTarget | str = (
        SpatialInjectionTarget.SUFFIX
    )

    enabled: bool = True

    def __post_init__(
        self,
    ) -> None:
        target = self.target

        if isinstance(
            target,
            str,
        ):
            try:
                target = SpatialInjectionTarget(
                    target
                )

            except ValueError as exc:
                valid = [
                    item.value
                    for item
                    in SpatialInjectionTarget
                ]

                raise ValueError(
                    "Unknown spatial injection target "
                    f"{self.target!r}. "
                    f"Valid values: {valid}"
                ) from exc

            object.__setattr__(
                self,
                "target",
                target,
            )

    @property
    def use_prefix(
        self,
    ) -> bool:
        """
        是否需要向 VLM prefix 注入 spatial token。
        """
        if not self.enabled:
            return False

        return self.target in (
            SpatialInjectionTarget.PREFIX,
            SpatialInjectionTarget.BOTH,
        )

    @property
    def use_suffix(
        self,
    ) -> bool:
        """
        是否需要向 action expert suffix 注入 spatial token。
        """
        if not self.enabled:
            return False

        return self.target in (
            SpatialInjectionTarget.SUFFIX,
            SpatialInjectionTarget.BOTH,
        )


# =============================================================================
# 3. Routed token container
# =============================================================================

ArrayT = TypeVar(
    "ArrayT"
)


@dataclasses.dataclass(
    frozen=True
)
class SpatialConditionedTokens(
    Generic[
        ArrayT
    ]
):
    """
    spatial encoder + adapter 之后的路由结果。

    prefix_tokens:
        Optional [B,K,D_prefix]

    prefix_mask:
        Optional [B,K]

    suffix_tokens:
        Optional [B,K,D_suffix]

    suffix_mask:
        Optional [B,K]

    注意：
        PREFIX 模式：
            prefix_* != None
            suffix_* == None

        SUFFIX 模式：
            prefix_* == None
            suffix_* != None

        BOTH 模式：
            两组都存在

    这里不保存 token_xyz_m。

    原因：
        token_xyz_m 属于 SpatialEncoderOutput 的 encoder metadata，
        而本结构只负责送入 downstream transformer 的 conditioning token。
    """

    prefix_tokens: ArrayT | None = None
    prefix_mask: ArrayT | None = None

    suffix_tokens: ArrayT | None = None
    suffix_mask: ArrayT | None = None

    def __post_init__(
        self,
    ) -> None:
        prefix_pair = (
            self.prefix_tokens is None,
            self.prefix_mask is None,
        )

        suffix_pair = (
            self.suffix_tokens is None,
            self.suffix_mask is None,
        )

        if prefix_pair[
            0
        ] != prefix_pair[
            1
        ]:
            raise ValueError(
                "prefix_tokens and prefix_mask "
                "must either both be None or both be present"
            )

        if suffix_pair[
            0
        ] != suffix_pair[
            1
        ]:
            raise ValueError(
                "suffix_tokens and suffix_mask "
                "must either both be None or both be present"
            )

    @property
    def has_prefix(
        self,
    ) -> bool:
        return (
            self.prefix_tokens
            is not None
        )

    @property
    def has_suffix(
        self,
    ) -> bool:
        return (
            self.suffix_tokens
            is not None
        )


# =============================================================================
# 4. Small validation helper
# =============================================================================

def validate_conditioned_tokens(
    *,
    config: SpatialConditioningConfig,
    tokens: SpatialConditionedTokens,
) -> None:
    """
    检查 routed token 与 conditioning config 是否一致。

    这个 helper 不检查 shape；
    shape 属于具体 π0 / adapter integration 的职责。
    """
    if config.use_prefix != tokens.has_prefix:
        raise ValueError(
            "Spatial prefix routing mismatch: "
            f"config.use_prefix={config.use_prefix}, "
            f"tokens.has_prefix={tokens.has_prefix}"
        )

    if config.use_suffix != tokens.has_suffix:
        raise ValueError(
            "Spatial suffix routing mismatch: "
            f"config.use_suffix={config.use_suffix}, "
            f"tokens.has_suffix={tokens.has_suffix}"
        )
