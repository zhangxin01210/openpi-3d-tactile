"""
OpenPI 3D + tactile 扩展：统一空间观测数据协议（SpatialObservation）

作用
----
本模块只负责定义“空间预处理最终输出的数据长什么样”，也就是整个项目中
3D 视觉与触觉数据之间的统一 contract。

它回答：
    1. 有哪些字段；
    2. 每个字段的 shape / dtype；
    3. 几何数据使用哪个坐标系；
    4. 单位是什么；
    5. 哪些物理与语义一致性必须满足。

设计原则
--------
1. schema 描述“数据是什么”，而不是“数据怎么算出来”。
2. 所有 3D 坐标统一使用 base_link 坐标系，单位为米。
3. visual point 数量属于实验配置，不在 schema 中硬编码为 4096。
4. 当前 XHand tactile 布局为 5 根手指 × 120 taxels = 600，因此默认严格检查。
5. visual_rgb_valid 必须保留：几何有效但 RGB 投影无效时，不静默删除 3D 点。

当前 baseline
-------------
visual: 4096 points（由 config 决定）
tactile: 600 taxels = 5 fingers × 120 taxels

基本使用
--------
    obs = SpatialObservation(...)
    obs.validate(expected_visual_points=4096)
    batch_dict = obs.to_dict()

注意
----
- tactile_force_base 的方向已经变换到 base_link。
- 力值仍是 dataset-native unit，不是 Newton。
- 对象创建后会自动执行 validate()，让错误尽量早暴露。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


# =============================================================================
# 1. 核心数据协议
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpatialObservation:
    """单帧 3D + tactile 空间观测。

    数据流中的位置：
        raw data
            -> preprocessing
            -> SpatialObservation
            -> dataset / dataloader / model / deployment

    字段说明：
        visual_xyz_m:       [Nv, 3] float32，base_link，单位 m
        visual_rgb:         [Nv, 3] uint8
        visual_rgb_valid:   [Nv] bool

        tactile_xyz_m:      [600, 3] float32，base_link，单位 m
        tactile_force_base: [600, 3] float32，base_link，dataset-native unit
        tactile_force_norm: [600] float32
        finger_id:          [600] int8，0..4
        taxel_id:           [600] int16，1..120
    """

    # -------------------------------------------------------------------------
    # 1.1 Schema 级元信息
    # 这些值描述整个数据协议，而不是某一帧，因此使用 ClassVar。
    # -------------------------------------------------------------------------
    SCHEMA_VERSION: ClassVar[int] = 1
    COORDINATE_FRAME: ClassVar[str] = "base_link"
    XYZ_UNIT: ClassVar[str] = "m"
    FORCE_UNIT: ClassVar[str] = "dataset_native"

    # -------------------------------------------------------------------------
    # 1.2 帧级 metadata
    # -------------------------------------------------------------------------
    frame_index: int
    timestamp_s: float

    # -------------------------------------------------------------------------
    # 1.3 Visual 3D observation
    # -------------------------------------------------------------------------
    visual_xyz_m: np.ndarray
    visual_rgb: np.ndarray
    visual_rgb_valid: np.ndarray

    # -------------------------------------------------------------------------
    # 1.4 Tactile spatial observation
    # -------------------------------------------------------------------------
    tactile_xyz_m: np.ndarray
    tactile_force_base: np.ndarray
    tactile_force_norm: np.ndarray
    finger_id: np.ndarray
    taxel_id: np.ndarray

    # =========================================================================
    # 2. 对象创建后的自动检查
    # =========================================================================

    def __post_init__(self) -> None:
        """对象创建完成后自动检查 contract，尽早暴露数据错误。"""
        self.validate()

    # =========================================================================
    # 3. 常用只读属性
    # =========================================================================

    @property
    def visual_count(self) -> int:
        """返回当前帧视觉点数量 Nv。"""
        return int(self.visual_xyz_m.shape[0])

    @property
    def tactile_count(self) -> int:
        """返回当前帧 tactile 点数量 Nt。"""
        return int(self.tactile_xyz_m.shape[0])

    # =========================================================================
    # 4. 核心 contract 校验
    # =========================================================================

    def validate(
        self,
        *,
        expected_visual_points: int | None = None,
        expected_tactile_points: int = 600,
    ) -> None:
        """检查当前 SpatialObservation 是否满足数据与物理 contract。

        expected_visual_points:
            可选。当前 baseline 可传 4096。
            不默认写死，因为视觉采样点数属于实验超参数。

        expected_tactile_points:
            默认 600。当前 XHand 为 5 × 120 taxels，这是硬件约束。
        """

        # ---------------------------------------------------------------------
        # 4.1 metadata 合法性
        # ---------------------------------------------------------------------
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {self.frame_index}")
        if not np.isfinite(self.timestamp_s):
            raise ValueError(f"timestamp_s must be finite, got {self.timestamp_s}")

        # ---------------------------------------------------------------------
        # 4.2 Visual：array / shape / dtype
        # ---------------------------------------------------------------------
        self._require_array(
            "visual_xyz_m", self.visual_xyz_m,
            ndim=2, trailing_shape=(3,), dtype=np.float32,
        )
        self._require_array(
            "visual_rgb", self.visual_rgb,
            ndim=2, trailing_shape=(3,), dtype=np.uint8,
        )
        self._require_array(
            "visual_rgb_valid", self.visual_rgb_valid,
            ndim=1, trailing_shape=(), dtype=np.bool_,
        )

        nv = self.visual_count
        if self.visual_rgb.shape[0] != nv or self.visual_rgb_valid.shape[0] != nv:
            raise ValueError(
                "Visual fields must have the same leading dimension: "
                f"xyz={self.visual_xyz_m.shape}, rgb={self.visual_rgb.shape}, "
                f"rgb_valid={self.visual_rgb_valid.shape}"
            )
        if expected_visual_points is not None and nv != expected_visual_points:
            raise ValueError(f"Expected {expected_visual_points} visual points, got {nv}")

        # ---------------------------------------------------------------------
        # 4.3 Tactile：array / shape / dtype
        # ---------------------------------------------------------------------
        self._require_array(
            "tactile_xyz_m", self.tactile_xyz_m,
            ndim=2, trailing_shape=(3,), dtype=np.float32,
        )
        self._require_array(
            "tactile_force_base", self.tactile_force_base,
            ndim=2, trailing_shape=(3,), dtype=np.float32,
        )
        self._require_array(
            "tactile_force_norm", self.tactile_force_norm,
            ndim=1, trailing_shape=(), dtype=np.float32,
        )
        self._require_array(
            "finger_id", self.finger_id,
            ndim=1, trailing_shape=(), dtype=np.int8,
        )
        self._require_array(
            "taxel_id", self.taxel_id,
            ndim=1, trailing_shape=(), dtype=np.int16,
        )

        # ---------------------------------------------------------------------
        # 4.4 Tactile 所有字段长度必须一致
        # ---------------------------------------------------------------------
        nt = self.tactile_count
        tactile_lengths = {
            "xyz": self.tactile_xyz_m.shape[0],
            "force": self.tactile_force_base.shape[0],
            "force_norm": self.tactile_force_norm.shape[0],
            "finger_id": self.finger_id.shape[0],
            "taxel_id": self.taxel_id.shape[0],
        }
        if len(set(tactile_lengths.values())) != 1:
            raise ValueError(
                f"Tactile fields must have the same leading dimension: {tactile_lengths}"
            )
        if nt != expected_tactile_points:
            raise ValueError(f"Expected {expected_tactile_points} tactile points, got {nt}")

        # ---------------------------------------------------------------------
        # 4.5 关键连续数值不允许出现 NaN / Inf
        # ---------------------------------------------------------------------
        self._require_finite("visual_xyz_m", self.visual_xyz_m)
        self._require_finite("tactile_xyz_m", self.tactile_xyz_m)
        self._require_finite("tactile_force_base", self.tactile_force_base)
        self._require_finite("tactile_force_norm", self.tactile_force_norm)

        # ---------------------------------------------------------------------
        # 4.6 finger_id 范围检查
        # ---------------------------------------------------------------------
        invalid_finger = (self.finger_id < 0) | (self.finger_id > 4)
        if np.any(invalid_finger):
            bad = np.unique(self.finger_id[invalid_finger])
            raise ValueError(
                f"finger_id must be in [0, 4], got invalid values {bad.tolist()}"
            )

        # ---------------------------------------------------------------------
        # 4.7 taxel_id 范围检查
        # ---------------------------------------------------------------------
        invalid_taxel = (self.taxel_id < 1) | (self.taxel_id > 120)
        if np.any(invalid_taxel):
            bad = np.unique(self.taxel_id[invalid_taxel])
            raise ValueError(
                f"taxel_id must be in [1, 120], got invalid values {bad.tolist()}"
            )

        # ---------------------------------------------------------------------
        # 4.8 当前 XHand 物理布局：
        #     每根手指必须恰好拥有 taxel 1..120，各出现一次。
        # ---------------------------------------------------------------------
        for finger in range(5):
            ids = np.sort(self.taxel_id[self.finger_id == finger])
            expected_ids = np.arange(1, 121, dtype=np.int16)
            if not np.array_equal(ids, expected_ids):
                raise ValueError(
                    f"finger {finger} must contain each taxel_id 1..120 exactly once; "
                    f"got {len(ids)} entries"
                )

        # ---------------------------------------------------------------------
        # 4.9 force_norm 必须与 force vector 自洽。
        #     这是冗余字段，因此需要防止二者在处理中发生不同步。
        # ---------------------------------------------------------------------
        computed_norm = np.linalg.norm(
            self.tactile_force_base, axis=-1
        ).astype(np.float32)
        if not np.allclose(
            computed_norm,
            self.tactile_force_norm,
            rtol=1e-5,
            atol=1e-4,
        ):
            max_err = float(np.max(np.abs(computed_norm - self.tactile_force_norm)))
            raise ValueError(
                "tactile_force_norm is inconsistent with tactile_force_base; "
                f"max absolute error={max_err:.6g}"
            )

    # =========================================================================
    # 5. 导出为 OpenPI / DataLoader 更容易使用的 dict
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """转换为 nested NumPy dict。

        这里仍然保持 visual / tactile 分开：
            preprocessing 描述物理 observation；
            model 再决定双 branch、concat 或 token fusion。
        """
        return {
            "spatial": {
                "visual": {
                    "xyz_m": self.visual_xyz_m,
                    "rgb": self.visual_rgb,
                    "rgb_valid": self.visual_rgb_valid,
                },
                "tactile": {
                    "xyz_m": self.tactile_xyz_m,
                    "force_base": self.tactile_force_base,
                    "force_norm": self.tactile_force_norm,
                    "finger_id": self.finger_id,
                    "taxel_id": self.taxel_id,
                },
            },
            "spatial_meta": {
                "schema_version": np.int32(self.SCHEMA_VERSION),
                "frame_index": np.int64(self.frame_index),
                "timestamp_s": np.float64(self.timestamp_s),
            },
        }

    # =========================================================================
    # 6. 内部辅助函数
    # 前导 "_" 表示内部实现细节，不属于主要 public API。
    # =========================================================================

    @staticmethod
    def _require_array(
        name: str,
        value: np.ndarray,
        *,
        ndim: int,
        trailing_shape: tuple[int, ...],
        dtype: np.dtype[Any] | type[np.generic],
    ) -> None:
        """统一检查 array 类型、维度、尾部 shape 与 dtype。"""
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be numpy.ndarray, got {type(value).__name__}")
        if value.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}, got shape {value.shape}")
        if trailing_shape and value.shape[-len(trailing_shape):] != trailing_shape:
            raise ValueError(
                f"{name} must end with shape {trailing_shape}, got {value.shape}"
            )
        if value.dtype != np.dtype(dtype):
            raise TypeError(
                f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}"
            )

    @staticmethod
    def _require_finite(name: str, value: np.ndarray) -> None:
        """检查连续数值数组中是否存在 NaN 或 Inf。"""
        finite = np.isfinite(value)
        if not np.all(finite):
            count = int(np.size(value) - np.count_nonzero(finite))
            raise ValueError(f"{name} contains {count} non-finite values")
