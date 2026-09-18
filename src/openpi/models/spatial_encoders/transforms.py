"""
OpenPI 3D + tactile：Spatial Encoder 特征变换组件
（openpi.models.spatial_encoders.transforms）

作用
----
提供 spatial encoder 可组合、可替换的“字段级 feature transform”。

数据边界：

    Observation.spatial
        ↓
    raw physical fields
        xyz_m
        rgb
        force
        force_norm
        ids
        masks
        ↓
    encoder 内部按配置调用本模块
        ↓
    encoder-local features

本模块不会修改 Observation，也不会覆盖 raw spatial fields。

这样可以保证：

    同一份 spatial/v1
    同一个 SpatialEncoderInput

能够被不同 encoder 用不同方式解释。

例如：

    PointNet++:
        xyz_m -> workspace normalize
        rgb   -> [0,1]

    Metric spatial attention:
        xyz_m -> identity

    tactile baseline:
        force -> linear scale

而无需重新生成数据或修改 dataset reader。

为什么不在这里实现 contact threshold
-----------------------------------
当前 tactile force 非常稀疏，但：

    hard threshold
    soft contact weight
    top-k contact
    contact-aware attention

都属于模型假设，不是通用输入事实。

因此这一版不把 contact handling 写进公共 transform。
当我们真正实现第一个 contact-aware encoder 时，再增加对应组件。

为什么不用统一 mode="..."
-------------------------
这里不写：

    if mode == "identity":
    elif mode == "workspace":
    elif ...

因为以后新增方法会不断修改同一个中央函数。

相反，每一种变换都是独立 callable class。
新增方法只需要新增 class，不影响已有实现。

后端
----
本模块刻意只使用：

    - Python 运算
    - NumPy dtype / 常量
    - array.astype()
    - array.clip()

这些操作同时适用于：

    NumPy ndarray
    JAX Array

因此本地没有 JAX 环境时，也可以用 NumPy 做 contract smoke test。

注意
----
- 这里的 transform 返回 encoder-local feature。
- 输入 ``xyz_m`` 本身仍然保持 meter，不会被原地修改。
- force 仍然来自 dataset_native；scale 的物理意义由实验配置决定。
- 本文件没有任何 learnable parameter。
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Protocol

import numpy as np


# =============================================================================
# 1. Transform protocols
# =============================================================================

class XYZTransform(Protocol):
    """
    XYZ feature transform。

    输入：
        xyz_m [..., N, 3]

    输出：
        [..., N, 3]

    输出不一定仍然是 meter；
    例如 workspace normalization 会输出无量纲坐标。
    """

    def __call__(
        self,
        xyz_m: Any,
    ) -> Any:
        ...


class RGBTransform(Protocol):
    """
    RGB feature transform。

    输入：
        rgb       [..., N, 3]
        rgb_valid [..., N]

    输出：
        [..., N, 3]
    """

    def __call__(
        self,
        rgb: Any,
        rgb_valid: Any,
    ) -> Any:
        ...


@dataclasses.dataclass(
    frozen=True
)
class ForceTransformOutput:
    """
    force transform 的统一返回结构。

    force:
        [..., N, 3]

    force_norm:
        [..., N]
    """

    force: Any
    force_norm: Any


class ForceTransform(Protocol):
    """
    Tactile force feature transform。
    """

    def __call__(
        self,
        force: Any,
        force_norm: Any,
    ) -> ForceTransformOutput:
        ...


# =============================================================================
# 2. XYZ transforms
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class IdentityXYZTransform:
    """
    保留 metric XYZ，仅统一为 float32。

    适合：
        - 想直接保留物理尺度的 encoder
        - metric spatial attention
        - 作为消融 baseline
    """

    def __call__(
        self,
        xyz_m: Any,
    ) -> Any:
        return xyz_m.astype(
            np.float32
        )


@dataclasses.dataclass(
    frozen=True
)
class WorkspaceNormalizeXYZTransform:
    """
    使用固定 workspace 将 XYZ 映射到约 [-1, 1]。

    公式：

        center = (min + max) / 2
        half_extent = (max - min) / 2

        xyz_normalized =
            (xyz_m - center) / half_extent

    默认不 clip。

    原因：
        如果未来某些真实点轻微越过 frozen workspace，
        clip 会静默丢失越界信息。

    如果某个实验明确希望限制到 [-1,1]，
    可以设置：

        clip=True

    注意：
        min_xyz_m / max_xyz_m 是模型配置的一部分，
        不应从当前 batch 动态估计。
    """

    min_xyz_m: tuple[
        float,
        float,
        float,
    ]

    max_xyz_m: tuple[
        float,
        float,
        float,
    ]

    clip: bool = False

    def __post_init__(
        self,
    ) -> None:
        minimum = np.asarray(
            self.min_xyz_m,
            dtype=np.float32,
        )

        maximum = np.asarray(
            self.max_xyz_m,
            dtype=np.float32,
        )

        if minimum.shape != (
            3,
        ):
            raise ValueError(
                "min_xyz_m must have shape (3,)"
            )

        if maximum.shape != (
            3,
        ):
            raise ValueError(
                "max_xyz_m must have shape (3,)"
            )

        if np.any(
            maximum
            <= minimum
        ):
            raise ValueError(
                "Each max_xyz_m value must "
                "be greater than min_xyz_m"
            )

    def __call__(
        self,
        xyz_m: Any,
    ) -> Any:
        xyz = xyz_m.astype(
            np.float32
        )

        minimum = np.asarray(
            self.min_xyz_m,
            dtype=np.float32,
        )

        maximum = np.asarray(
            self.max_xyz_m,
            dtype=np.float32,
        )

        center = (
            minimum
            + maximum
        ) * np.float32(
            0.5
        )

        half_extent = (
            maximum
            - minimum
        ) * np.float32(
            0.5
        )

        output = (
            xyz
            - center
        ) / half_extent

        if self.clip:
            output = output.clip(
                np.float32(
                    -1.0
                ),
                np.float32(
                    1.0
                ),
            )

        return output.astype(
            np.float32
        )


# =============================================================================
# 3. RGB transforms
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class RGBZeroOneTransform:
    """
    raw uint8 RGB -> float32 [0,1]。

    默认将 rgb_valid=False 的 RGB feature 置零。

    这是为了避免：

        无效 RGB 的占位值

    被 encoder 当成真实颜色。

    rgb_valid 本身仍然独立保留在 SpatialEncoderInput，
    encoder 如果需要可以额外把 valid mask 当作 feature。
    """

    zero_invalid: bool = True

    def __call__(
        self,
        rgb: Any,
        rgb_valid: Any,
    ) -> Any:
        output = (
            rgb.astype(
                np.float32
            )
            / np.float32(
                255.0
            )
        )

        if self.zero_invalid:
            valid = rgb_valid.astype(
                np.float32
            )

            output = (
                output
                * valid[
                    ...,
                    None,
                ]
            )

        return output.astype(
            np.float32
        )


# =============================================================================
# 4. Force transforms
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class IdentityForceTransform:
    """
    保留 dataset_native force，只统一为 float32。

    这是最重要的无假设 baseline。
    """

    def __call__(
        self,
        force: Any,
        force_norm: Any,
    ) -> ForceTransformOutput:
        return ForceTransformOutput(
            force=force.astype(
                np.float32
            ),
            force_norm=(
                force_norm.astype(
                    np.float32
                )
            ),
        )


@dataclasses.dataclass(
    frozen=True
)
class LinearScaleForceTransform:
    """
    tactile force 的线性缩放变换。

    公式：

        force_feature = force / scale
        norm_feature  = force_norm / scale

    可选 clip_abs：

        component:
            clip(force, -clip_abs, clip_abs)

        norm:
            clip(force_norm, 0, clip_abs)

    注意：
        scale / clip_abs 都是实验配置，
        本类不根据当前 batch 自动估计。

    这保证 train / eval / deployment 使用完全相同的变换。

    当前 force 单位仍然是 dataset_native，
    因此 scale 也以 dataset_native 表示。
    """

    scale: float
    clip_abs: float | None = None

    def __post_init__(
        self,
    ) -> None:
        if not np.isfinite(
            self.scale
        ):
            raise ValueError(
                "scale must be finite"
            )

        if self.scale <= 0:
            raise ValueError(
                "scale must be > 0"
            )

        if (
            self.clip_abs
            is not None
        ):
            if not np.isfinite(
                self.clip_abs
            ):
                raise ValueError(
                    "clip_abs must be finite"
                )

            if self.clip_abs <= 0:
                raise ValueError(
                    "clip_abs must be > 0"
                )

    def __call__(
        self,
        force: Any,
        force_norm: Any,
    ) -> ForceTransformOutput:
        force_value = force.astype(
            np.float32
        )

        norm_value = force_norm.astype(
            np.float32
        )

        if (
            self.clip_abs
            is not None
        ):
            clip_value = np.float32(
                self.clip_abs
            )

            force_value = (
                force_value.clip(
                    -clip_value,
                    clip_value,
                )
            )

            norm_value = (
                norm_value.clip(
                    np.float32(
                        0.0
                    ),
                    clip_value,
                )
            )

        scale = np.float32(
            self.scale
        )

        return ForceTransformOutput(
            force=(
                force_value
                / scale
            ).astype(
                np.float32
            ),
            force_norm=(
                norm_value
                / scale
            ).astype(
                np.float32
            ),
        )


# =============================================================================
# 5. Composable transform bundle
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class SpatialFeatureTransforms:
    """
    一个 encoder 可以持有的 feature transform bundle。

    它只组合独立 transform，不规定 encoder 架构。

    例如：

        transforms = SpatialFeatureTransforms(
            visual_xyz=WorkspaceNormalizeXYZTransform(...),
            tactile_xyz=WorkspaceNormalizeXYZTransform(...),
            rgb=RGBZeroOneTransform(),
            force=LinearScaleForceTransform(
                scale=20.0,
            ),
        )

    然后 encoder 内部：

        visual_xyz = transforms.visual_xyz(
            inputs.visual.xyz_m
        )

        rgb = transforms.rgb(
            inputs.visual.rgb,
            inputs.visual.rgb_valid,
        )

        tactile_xyz = transforms.tactile_xyz(
            inputs.tactile.xyz_m
        )

        force = transforms.force(
            inputs.tactile.force,
            inputs.tactile.force_norm,
        )

    注意：
        上面的 scale=20.0 只是文档示例，
        不是当前 baseline 的冻结值。
    """

    visual_xyz: XYZTransform = dataclasses.field(
        default_factory=(
            IdentityXYZTransform
        )
    )

    tactile_xyz: XYZTransform = dataclasses.field(
        default_factory=(
            IdentityXYZTransform
        )
    )

    rgb: RGBTransform = dataclasses.field(
        default_factory=(
            RGBZeroOneTransform
        )
    )

    force: ForceTransform = dataclasses.field(
        default_factory=(
            IdentityForceTransform
        )
    )
