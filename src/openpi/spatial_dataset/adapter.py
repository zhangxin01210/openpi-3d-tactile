"""
OpenPI 3D + tactile：Spatial Derived Dataset Join Adapter
（openpi.spatial_dataset.adapter）

作用
----
将任意已有的训练 dataset sample：

    {
        "episode_index": ...,
        "frame_index": ...,
        "observation.state": ...,
        "action": ...,
        ...
    }

与已经生成的：

    spatial/v1

按：

    (episode_index, frame_index)

进行精确 join。

输出：

    {
        ... 原始 sample ...,

        "spatial": {
            "visual": {
                "xyz_m": ...,
                "rgb": ...,
                "rgb_valid": ...,
                "point_mask": ...,
            },

            "tactile": {
                "xyz_m": ...,
                "force": ...,
                "force_norm": ...,
                "finger_id": ...,
                "taxel_id": ...,
                "point_mask": ...,
            },
        },
    }

这一层为什么重要
----------------
它把：

    “数据集里有没有 spatial modality”

和：

    “模型如何编码 spatial modality”

彻底分开。

因此后续：

    PointNet
    PointNet++
    Joint Point Cloud
    Point Transformer
    Visual/Tactile Dual Branch
    Cross Attention
    Spatial Attention
    Visual-only / Tactile-only ablation

都不需要修改 dataset join 逻辑。

本模块明确不负责
----------------
- normalization
- contact threshold
- feature concatenation
- modality embedding
- finger embedding
- taxel embedding
- early / late fusion
- point encoder
- spatial token 数量

这些都是研究假设，应放在：

    transform / encoder / fusion config

而不是 dataset adapter。

为什么按 episode/frame join，而不是直接按全局 index
-------------------------------------------------
虽然当前：

    base dataset total frames = 2289
    spatial/v1 total frames = 2289

但未来：

    - dataset 可能重排
    - shuffle 可能发生
    - episode 可能筛选
    - action horizon loader 可能改变访问顺序

因此不能依赖：

    base_dataset[i] <-> spatial_dataset[i]

必须使用数据本身的身份：

    episode_index
    frame_index

进行 join。

数组 copy 策略
-------------
SpatialDerivedDataset 返回的是只读 mmap view。

PyTorch DataLoader / tensor conversion 对只读 NumPy array
可能产生 warning 或潜在写入风险。

因此默认：

    copy_arrays=True

每次只复制当前 sample 的 spatial arrays，
不会把整个 derived dataset 加载到 RAM。

如果只是调试 / QA，可以：

    copy_arrays=False

直接保留 mmap view。

输入 contract
-------------
base dataset 的每个 item 至少需要：

    "episode_index"
    "frame_index"

它们可以是：

    Python int
    NumPy scalar
    具有 .item() 的 scalar tensor

输出 spatial contract
---------------------
visual：

    xyz_m:
        [Nv,3] float32

    rgb:
        [Nv,3] uint8

    rgb_valid:
        [Nv] bool

    point_mask:
        [Nv] bool
        当前全部 True。

tactile：

    xyz_m:
        [Nt,3] float32

    force:
        [Nt,3] float32

    force_norm:
        [Nt] float32

    finger_id:
        [Nt] int32
        注意：derived static 中原始是 int8，
        adapter 提升为 int32，方便后续 embedding / JAX / Torch。

    taxel_id:
        [Nt] int32
        derived static 中原始是 int16，
        adapter 提升为 int32。

    point_mask:
        [Nt] bool
        当前全部 True。

基本使用
--------
    base_dataset = ...

    spatial_dataset = SpatialDerivedDataset(
        "data/press_0828_17",
        version="v1",
    )

    dataset = SpatialAugmentedDataset(
        base_dataset,
        spatial_dataset,
    )

    item = dataset[0]

    print(
        item["spatial"]["visual"]["xyz_m"].shape
    )

注意
----
- 如果 base sample 已经存在 "spatial" key，默认直接报错，
  避免静默覆盖。
- adapter 不依赖 JAX / Flax / Torch。
- adapter 不改变原 base sample object，而是返回新的 dict。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Protocol
from typing import SupportsIndex
from typing import TypeVar

import numpy as np

from openpi.spatial_dataset.derived import SpatialDerivedDataset


# =============================================================================
# 1. Minimal dataset protocol
# =============================================================================

SampleT = TypeVar(
    "SampleT",
    bound=Mapping[str, Any],
)


class RandomAccessDataset(
    Protocol[
        SampleT
    ]
):
    """
    与 OpenPI training.data_loader.Dataset 相同的最小随机访问语义。

    这里故意不 import training.data_loader，
    避免 spatial_dataset 反向依赖整个训练栈。
    """

    def __getitem__(
        self,
        index: SupportsIndex,
    ) -> SampleT:
        ...

    def __len__(
        self,
    ) -> int:
        ...


# =============================================================================
# 2. Scalar conversion
# =============================================================================

def _scalar_to_int(
    value: Any,
    *,
    name: str,
) -> int:
    """
    将常见 scalar 类型安全转换成 Python int。

    支持：
        int
        np.integer
        0-D ndarray
        torch / jax 风格 .item() scalar

    不接受多元素数组。
    """
    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        if value.size != 1:
            raise ValueError(
                f"{name} must be scalar, "
                f"got ndarray shape {value.shape}"
            )

        return int(
            value.reshape(
                ()
            ).item()
        )

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(
        item_method
    ):
        scalar = item_method()

        if isinstance(
            scalar,
            (
                int,
                np.integer,
            ),
        ):
            return int(
                scalar
            )

        try:
            return int(
                scalar
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} could not be converted "
                f"from scalar {scalar!r}"
            ) from exc

    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be an integer scalar, "
            f"got {type(value)!r}"
        ) from exc


# =============================================================================
# 3. Array materialization
# =============================================================================

def _array(
    value: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
    copy: bool,
) -> np.ndarray:
    """
    将 mmap view 转成 adapter 输出 array。

    copy=True：
        生成独立、可写、C-contiguous ndarray。

    copy=False：
        尽量保留原 mmap / view。
    """
    if copy:
        return np.array(
            value,
            dtype=dtype,
            copy=True,
            order="C",
        )

    return np.asarray(
        value,
        dtype=dtype,
    )


# =============================================================================
# 4. Spatial join adapter
# =============================================================================

class SpatialAugmentedDataset:
    """
    将 derived spatial modality join 到任意 base dataset。

    Join key：

        episode_index + frame_index
    """

    def __init__(
        self,
        base_dataset: RandomAccessDataset,
        spatial_dataset: SpatialDerivedDataset,
        *,
        copy_arrays: bool = True,
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
        output_key: str = "spatial",
    ) -> None:
        self.base_dataset = (
            base_dataset
        )

        self.spatial_dataset = (
            spatial_dataset
        )

        self.copy_arrays = bool(
            copy_arrays
        )

        self.episode_key = str(
            episode_key
        )

        self.frame_key = str(
            frame_key
        )

        self.output_key = str(
            output_key
        )

        if not self.episode_key:
            raise ValueError(
                "episode_key cannot be empty"
            )

        if not self.frame_key:
            raise ValueError(
                "frame_key cannot be empty"
            )

        if not self.output_key:
            raise ValueError(
                "output_key cannot be empty"
            )

    def __len__(
        self,
    ) -> int:
        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index: SupportsIndex,
    ) -> dict[str, Any]:
        """
        读取 base sample，并按 episode/frame join spatial sample。
        """
        base_item = self.base_dataset[
            index
        ]

        if not isinstance(
            base_item,
            Mapping,
        ):
            raise TypeError(
                "Base dataset item must be a mapping, "
                f"got {type(base_item)!r}"
            )

        if (
            self.episode_key
            not in base_item
        ):
            raise KeyError(
                f"Base sample missing "
                f"{self.episode_key!r}"
            )

        if (
            self.frame_key
            not in base_item
        ):
            raise KeyError(
                f"Base sample missing "
                f"{self.frame_key!r}"
            )

        if (
            self.output_key
            in base_item
        ):
            raise KeyError(
                f"Base sample already contains "
                f"{self.output_key!r}; refusing "
                "to silently overwrite it."
            )

        episode_index = _scalar_to_int(
            base_item[
                self.episode_key
            ],
            name=self.episode_key,
        )

        frame_index = _scalar_to_int(
            base_item[
                self.frame_key
            ],
            name=self.frame_key,
        )

        spatial_sample = (
            self.spatial_dataset.get(
                episode_index=episode_index,
                frame_index=frame_index,
            )
        )

        spatial = self._to_dict(
            spatial_sample
        )

        # 不修改 base dataset 返回的原对象。
        output = dict(
            base_item
        )

        output[
            self.output_key
        ] = spatial

        return output

    # =========================================================================
    # 5. Spatial sample -> nested dict
    # =========================================================================

    def _to_dict(
        self,
        sample,
    ) -> dict[str, Any]:
        """
        DerivedSpatialSample -> unbatched NumPy spatial tree。

        这里保留物理语义；
        不执行 normalization / feature engineering。
        """
        visual_xyz = _array(
            sample.visual_xyz_m,
            dtype=np.float32,
            copy=self.copy_arrays,
        )

        visual_rgb = _array(
            sample.visual_rgb,
            dtype=np.uint8,
            copy=self.copy_arrays,
        )

        visual_rgb_valid = _array(
            sample.visual_rgb_valid,
            dtype=np.bool_,
            copy=self.copy_arrays,
        )

        tactile_xyz = _array(
            sample.tactile_xyz_m,
            dtype=np.float32,
            copy=self.copy_arrays,
        )

        tactile_force = _array(
            sample.tactile_force_base,
            dtype=np.float32,
            copy=self.copy_arrays,
        )

        tactile_force_norm = _array(
            sample.tactile_force_norm,
            dtype=np.float32,
            copy=self.copy_arrays,
        )

        # ID 类字段在 model input 侧统一 int32。
        #
        # 这样后续无论：
        #     JAX embedding
        #     Torch embedding
        #
        # 都不需要保留磁盘压缩用的 int8/int16 dtype。
        finger_id = _array(
            sample.finger_id,
            dtype=np.int32,
            copy=self.copy_arrays,
        )

        taxel_id = _array(
            sample.taxel_id,
            dtype=np.int32,
            copy=self.copy_arrays,
        )

        visual_count = int(
            visual_xyz.shape[
                0
            ]
        )

        tactile_count = int(
            tactile_xyz.shape[
                0
            ]
        )

        visual_mask = np.ones(
            visual_count,
            dtype=np.bool_,
        )

        tactile_mask = np.ones(
            tactile_count,
            dtype=np.bool_,
        )

        return {
            "visual": {
                "xyz_m": (
                    visual_xyz
                ),
                "rgb": (
                    visual_rgb
                ),
                "rgb_valid": (
                    visual_rgb_valid
                ),
                "point_mask": (
                    visual_mask
                ),
            },
            "tactile": {
                "xyz_m": (
                    tactile_xyz
                ),
                "force": (
                    tactile_force
                ),
                "force_norm": (
                    tactile_force_norm
                ),
                "finger_id": (
                    finger_id
                ),
                "taxel_id": (
                    taxel_id
                ),
                "point_mask": (
                    tactile_mask
                ),
            },
        }
