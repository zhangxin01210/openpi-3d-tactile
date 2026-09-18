"""
OpenPI 3D + tactile 扩展：Tactile 空间观测构建（tactile.py）

作用
----
本模块负责把原始 XHand tactile state 与当前手部 FK 结果转换成统一的
600 点 tactile spatial observation。

完整物理链：

    observation.state
        ↓
    拆出 5 个 finger tactile block
        ↓
    每个 finger 取 120 × 3 raw force
        ↓
    vendor axis mapping
        ↓
    force in finger link2 frame
        ↓
    当前 FK 给出 T_base_link2
        ↓
    taxel local position:
        p_base = R_base_link2 @ p_link2 + t_base_link2

    tactile force:
        f_base = R_base_link2 @ f_link2

        注意：force vector 只旋转，不加 translation
        ↓
    TactileGeometryResult
        xyz_m       [600,3]
        force_base  [600,3]
        force_norm  [600]
        finger_id   [600]
        taxel_id    [600]

本模块不负责
------------
- 解析 URDF
- 计算 robot / XHand FK
- 从 teacher URDF / JSON 读取 taxel local geometry
- 把 tactile 与 visual 拼成 SpatialObservation
- force 单位换算成 Newton
- 训练时 normalization

这些职责故意分开：
    tactile.py 只实现“已经知道 local geometry 和 link pose 后，触觉如何变成
    base_link 下的空间观测”。

为什么这样拆
------------
如果 tactile.py 自己去解析 URDF、自己做 FK、自己读 dataset，会把：
    sensor layout
    kinematics
    dataset format
    force mapping
全部绑死在一个文件里。

现在的接口只依赖：
    1. raw state
    2. taxel local geometry
    3. T_base_link2

因此以后即使换：
    - FK backend
    - URDF parser
    - 实时机器人状态来源
    - 离线 dataset

本模块核心逻辑都不需要改。

当前 XHand baseline
------------------
finger 顺序：
    0 thumb
    1 index
    2 middle
    3 ring
    4 pinky

每根手指：
    tactile block = 384 values
    raw force 从 block offset 24 开始
    120 taxels × 3 axes = 360 values

当前 state_dim = 1972 时：
    tactile total = 5 × 384 = 1920
    inferred prefix = 1972 - 1920 = 52

force axis mapping：
    thumb / T30:
        [Fx,Fy,Fz] -> [Fx,Fy,Fz]

    index/middle/ring/pinky / T16:
        [Fx,Fy,Fz] -> [Fz,Fx,Fy]

力值单位：
    保持 dataset-native，不做 Newton 换算。

基本使用
--------
>>> from openpi.spatial.config import make_baseline_config
>>> from openpi.spatial.tactile import build_tactile_observation
>>>
>>> cfg = make_baseline_config()
>>>
>>> result = build_tactile_observation(
...     state=state,
...     taxel_xyz_link_m=taxel_xyz_link_m,   # [5,120,3]
...     T_base_link_by_name=T_base_link_by_name,
...     config=cfg.tactile,
... )
>>>
>>> result.xyz_m.shape
(600, 3)
>>> result.force_base.shape
(600, 3)

注意
----
1. `taxel_xyz_link_m[finger, taxel]` 必须已经表达在该 finger 的 link2 frame。
2. `T_base_link_by_name` 的 key 必须对应 TactileConfig.finger_link_names。
3. raw force 的正负号保持当前已经验证的 positive convention。
4. 本模块不静默猜测缺失 link transform；缺一个就立即报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from openpi.spatial.config import TactileConfig


# =============================================================================
# 1. 对外结果对象
# =============================================================================

@dataclass(frozen=True, slots=True)
class TactileGeometryResult:
    """
    单帧 tactile spatial preprocessing 结果。

    xyz_m:
        [600,3] float32
        600 个 taxel 在 base_link 中的位置，单位米。

    force_base:
        [600,3] float32
        600 个 force vector，方向已旋转到 base_link。
        数值单位仍是 dataset-native。

    force_norm:
        [600] float32
        force_base 的欧氏范数。

    finger_id:
        [600] int8
        0..4。

    taxel_id:
        [600] int16
        每根手指内部编号 1..120。

    raw_state_prefix:
        原始 state 中 tactile blocks 之前的 prefix 长度。
        当前 1972 维 state 应推导出 52。
        该字段用于 QA / provenance，不需要进入模型。
    """

    xyz_m: np.ndarray
    force_base: np.ndarray
    force_norm: np.ndarray
    finger_id: np.ndarray
    taxel_id: np.ndarray
    raw_state_prefix: int

    def __post_init__(self) -> None:
        _require_array(
            "xyz_m",
            self.xyz_m,
            shape=(600, 3),
            dtype=np.float32,
        )
        _require_array(
            "force_base",
            self.force_base,
            shape=(600, 3),
            dtype=np.float32,
        )
        _require_array(
            "force_norm",
            self.force_norm,
            shape=(600,),
            dtype=np.float32,
        )
        _require_array(
            "finger_id",
            self.finger_id,
            shape=(600,),
            dtype=np.int8,
        )
        _require_array(
            "taxel_id",
            self.taxel_id,
            shape=(600,),
            dtype=np.int16,
        )

        if self.raw_state_prefix < 0:
            raise ValueError(
                f"raw_state_prefix must be >= 0, got {self.raw_state_prefix}"
            )

        for name, value in (
            ("xyz_m", self.xyz_m),
            ("force_base", self.force_base),
            ("force_norm", self.force_norm),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} contains NaN / Inf"
                )

        computed_norm = np.linalg.norm(
            self.force_base,
            axis=-1,
        ).astype(np.float32)

        if not np.allclose(
            computed_norm,
            self.force_norm,
            rtol=1e-5,
            atol=1e-4,
        ):
            max_error = float(
                np.max(
                    np.abs(
                        computed_norm
                        - self.force_norm
                    )
                )
            )
            raise ValueError(
                "force_norm inconsistent with force_base; "
                f"max abs error={max_error:.6g}"
            )


# =============================================================================
# 2. 原始 observation.state -> [5,120,3] raw force
# =============================================================================

def extract_raw_tactile_force(
    state: np.ndarray,
    *,
    config: TactileConfig,
) -> tuple[np.ndarray, int]:
    """
    从 observation.state 中提取 5 × 120 × 3 tactile raw force。

    关键点：
        不把 prefix=52 硬编码在算法里。

    而是根据：
        state_dim - num_fingers * tactile_block_size

    自动推导 tactile blocks 的起点。

    当前数据：
        1972 - 5*384 = 52

    返回
    ----
    raw_force:
        [num_fingers, taxels_per_finger, 3] float32

    prefix:
        tactile block 前面的 state 维度。
    """
    state_array = np.asarray(
        state
    )

    if state_array.ndim != 1:
        raise ValueError(
            f"state must be 1-D, got shape {state_array.shape}"
        )

    force_values_per_finger = (
        config.taxels_per_finger
        * 3
    )

    force_end_inside_block = (
        config.raw_force_offset
        + force_values_per_finger
    )

    if force_end_inside_block > config.tactile_block_size:
        raise ValueError(
            "TactileConfig is internally inconsistent: "
            f"raw_force_offset({config.raw_force_offset}) + "
            f"taxels_per_finger*3({force_values_per_finger}) "
            f"> tactile_block_size({config.tactile_block_size})"
        )

    tactile_total = (
        config.num_fingers
        * config.tactile_block_size
    )

    prefix = (
        state_array.size
        - tactile_total
    )

    if prefix < 0:
        raise ValueError(
            "state is too short for configured tactile layout: "
            f"state_dim={state_array.size}, "
            f"required_tactile_values={tactile_total}"
        )

    raw_force = np.empty(
        (
            config.num_fingers,
            config.taxels_per_finger,
            3,
        ),
        dtype=np.float32,
    )

    for finger_index in range(
        config.num_fingers
    ):
        block_start = (
            prefix
            + finger_index
            * config.tactile_block_size
        )

        force_start = (
            block_start
            + config.raw_force_offset
        )

        force_end = (
            force_start
            + force_values_per_finger
        )

        force_flat = state_array[
            force_start:force_end
        ]

        if force_flat.size != force_values_per_finger:
            raise RuntimeError(
                f"finger {finger_index}: expected "
                f"{force_values_per_finger} force values, "
                f"got {force_flat.size}"
            )

        raw_force[
            finger_index
        ] = force_flat.reshape(
            config.taxels_per_finger,
            3,
        ).astype(
            np.float32,
            copy=False,
        )

    return (
        raw_force,
        prefix,
    )


# =============================================================================
# 3. Vendor force-axis mapping
# =============================================================================

def map_force_axes_to_link(
    raw_force: np.ndarray,
    *,
    config: TactileConfig,
) -> np.ndarray:
    """
    把 raw sensor force axes 映射到各 finger link2 frame。

    输入：
        raw_force [5,120,3]

    输出：
        force_link [5,120,3]

    当前映射：
        thumb:
            (0,1,2)
            [Fx,Fy,Fz] -> [Fx,Fy,Fz]

        other fingers:
            (2,0,1)
            [Fx,Fy,Fz] -> [Fz,Fx,Fy]

    这里只重排轴：
        不旋转到 base_link；
        不改正负号；
        不换算力单位。
    """
    raw = np.asarray(
        raw_force,
        dtype=np.float32,
    )

    expected_shape = (
        config.num_fingers,
        config.taxels_per_finger,
        3,
    )

    if raw.shape != expected_shape:
        raise ValueError(
            f"raw_force expected shape {expected_shape}, got {raw.shape}"
        )

    force_link = np.empty_like(
        raw,
        dtype=np.float32,
    )

    for finger_index, order in enumerate(
        config.force_axis_order_by_finger
    ):
        force_link[
            finger_index
        ] = raw[
            finger_index
        ][
            :,
            list(order),
        ]

    return force_link


# =============================================================================
# 4. Taxel local geometry -> base_link
# =============================================================================

def transform_taxel_positions_to_base(
    taxel_xyz_link_m: np.ndarray,
    *,
    T_base_link_by_name: Mapping[str, np.ndarray],
    config: TactileConfig,
) -> np.ndarray:
    """
    将每根手指 link2 frame 中的 taxel local position 变到 base_link。

    输入
    ----
    taxel_xyz_link_m:
        [5,120,3]
        第 i 根手指的 120 个 taxel 坐标，
        已表达在 config.finger_link_names[i] frame 中。

    T_base_link_by_name:
        {
            link_name: T_base_link [4,4]
        }

    返回
    ----
    xyz_base:
        [5,120,3] float32

    数学：
        p_base = R_base_link @ p_link + t_base_link
    """
    local_xyz = np.asarray(
        taxel_xyz_link_m,
        dtype=np.float32,
    )

    expected_shape = (
        config.num_fingers,
        config.taxels_per_finger,
        3,
    )

    if local_xyz.shape != expected_shape:
        raise ValueError(
            "taxel_xyz_link_m expected shape "
            f"{expected_shape}, got {local_xyz.shape}"
        )

    if not np.all(
        np.isfinite(local_xyz)
    ):
        raise ValueError(
            "taxel_xyz_link_m contains NaN / Inf"
        )

    xyz_base = np.empty_like(
        local_xyz,
        dtype=np.float32,
    )

    for finger_index, link_name in enumerate(
        config.finger_link_names
    ):
        T_base_link = _get_transform(
            T_base_link_by_name,
            link_name=link_name,
        )

        rotation = T_base_link[
            :3,
            :3,
        ].astype(
            np.float32,
        )

        translation = T_base_link[
            :3,
            3,
        ].astype(
            np.float32,
        )

        xyz_base[
            finger_index
        ] = (
            local_xyz[
                finger_index
            ]
            @ rotation.T
            + translation[None, :]
        )

    return xyz_base


# =============================================================================
# 5. Link-frame force -> base_link
# =============================================================================

def rotate_forces_to_base(
    force_link: np.ndarray,
    *,
    T_base_link_by_name: Mapping[str, np.ndarray],
    config: TactileConfig,
) -> np.ndarray:
    """
    将每根手指 link2 frame 中的 force vector 旋转到 base_link。

    数学：
        f_base = R_base_link @ f_link

    与 point position 不同：
        force vector 没有空间位置，因此绝对不能加 translation。
    """
    force = np.asarray(
        force_link,
        dtype=np.float32,
    )

    expected_shape = (
        config.num_fingers,
        config.taxels_per_finger,
        3,
    )

    if force.shape != expected_shape:
        raise ValueError(
            f"force_link expected shape {expected_shape}, got {force.shape}"
        )

    force_base = np.empty_like(
        force,
        dtype=np.float32,
    )

    for finger_index, link_name in enumerate(
        config.finger_link_names
    ):
        T_base_link = _get_transform(
            T_base_link_by_name,
            link_name=link_name,
        )

        rotation = T_base_link[
            :3,
            :3,
        ].astype(
            np.float32,
        )

        force_base[
            finger_index
        ] = (
            force[
                finger_index
            ]
            @ rotation.T
        )

    return force_base


# =============================================================================
# 6. finger_id / taxel_id
# =============================================================================

def make_taxel_ids(
    *,
    config: TactileConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    构造稳定的 600 点身份索引。

    展平顺序：
        thumb 1..120
        index 1..120
        middle 1..120
        ring 1..120
        pinky 1..120
    """
    finger_id = np.repeat(
        np.arange(
            config.num_fingers,
            dtype=np.int8,
        ),
        config.taxels_per_finger,
    )

    taxel_id = np.tile(
        np.arange(
            1,
            config.taxels_per_finger + 1,
            dtype=np.int16,
        ),
        config.num_fingers,
    )

    return (
        finger_id,
        taxel_id,
    )


# =============================================================================
# 7. 完整 tactile preprocessing 入口
# =============================================================================

def build_tactile_observation(
    *,
    state: np.ndarray,
    taxel_xyz_link_m: np.ndarray,
    T_base_link_by_name: Mapping[str, np.ndarray],
    config: TactileConfig,
) -> TactileGeometryResult:
    """
    完成单帧 tactile spatial preprocessing。

    这是 tactile.py 面向 preprocess.py 的主要 public API。

    流程
    ----
    1. observation.state -> raw force
    2. vendor force-axis mapping
    3. taxel local xyz -> base_link
    4. force link2 -> base_link
    5. force norm
    6. finger_id / taxel_id
    7. flatten 为固定 600 点
    """
    # -------------------------------------------------------------------------
    # 7.1 从 raw state 提取 tactile force
    # -------------------------------------------------------------------------
    raw_force, prefix = (
        extract_raw_tactile_force(
            state,
            config=config,
        )
    )

    # -------------------------------------------------------------------------
    # 7.2 vendor axis mapping
    # -------------------------------------------------------------------------
    force_link = (
        map_force_axes_to_link(
            raw_force,
            config=config,
        )
    )

    # -------------------------------------------------------------------------
    # 7.3 taxel local position -> base_link
    # -------------------------------------------------------------------------
    xyz_base = (
        transform_taxel_positions_to_base(
            taxel_xyz_link_m,
            T_base_link_by_name=T_base_link_by_name,
            config=config,
        )
    )

    # -------------------------------------------------------------------------
    # 7.4 force vector -> base_link
    # -------------------------------------------------------------------------
    force_base = (
        rotate_forces_to_base(
            force_link,
            T_base_link_by_name=T_base_link_by_name,
            config=config,
        )
    )

    # -------------------------------------------------------------------------
    # 7.5 展平为和 SpatialObservation 一致的 600 点顺序
    # -------------------------------------------------------------------------
    xyz_flat = xyz_base.reshape(
        -1,
        3,
    ).astype(
        np.float32,
        copy=False,
    )

    force_flat = force_base.reshape(
        -1,
        3,
    ).astype(
        np.float32,
        copy=False,
    )

    force_norm = np.linalg.norm(
        force_flat,
        axis=-1,
    ).astype(
        np.float32,
    )

    finger_id, taxel_id = (
        make_taxel_ids(
            config=config,
        )
    )

    return TactileGeometryResult(
        xyz_m=xyz_flat,
        force_base=force_flat,
        force_norm=force_norm,
        finger_id=finger_id,
        taxel_id=taxel_id,
        raw_state_prefix=prefix,
    )


# =============================================================================
# 8. 内部 transform / array helper
# =============================================================================

def _get_transform(
    transforms: Mapping[str, np.ndarray],
    *,
    link_name: str,
) -> np.ndarray:
    """
    从 mapping 中读取并验证 T_base_link。

    不允许缺失 link 时静默使用 identity。
    """
    if link_name not in transforms:
        raise KeyError(
            f"Missing T_base_link for {link_name!r}. "
            f"Available links: {sorted(transforms.keys())}"
        )

    T = np.asarray(
        transforms[
            link_name
        ],
        dtype=np.float64,
    )

    if T.shape != (4, 4):
        raise ValueError(
            f"{link_name}: transform must be [4,4], got {T.shape}"
        )

    if not np.all(
        np.isfinite(T)
    ):
        raise ValueError(
            f"{link_name}: transform contains NaN / Inf"
        )

    if not np.allclose(
        T[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),
        atol=1e-6,
    ):
        raise ValueError(
            f"{link_name}: invalid homogeneous last row {T[3]}"
        )

    rotation = T[
        :3,
        :3,
    ]

    orthogonal_error = float(
        np.linalg.norm(
            rotation.T @ rotation
            - np.eye(3),
            ord="fro",
        )
    )

    determinant = float(
        np.linalg.det(
            rotation
        )
    )

    if orthogonal_error > 1e-4:
        raise ValueError(
            f"{link_name}: rotation is not orthogonal; "
            f"error={orthogonal_error:.6g}"
        )

    if abs(
        determinant - 1.0
    ) > 1e-4:
        raise ValueError(
            f"{link_name}: rotation det must be ~1, "
            f"got {determinant:.8f}"
        )

    return T


def _require_array(
    name: str,
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> None:
    """检查固定 shape / dtype ndarray。"""
    if not isinstance(
        value,
        np.ndarray,
    ):
        raise TypeError(
            f"{name} must be numpy.ndarray, "
            f"got {type(value).__name__}"
        )

    if value.shape != shape:
        raise ValueError(
            f"{name} expected shape {shape}, got {value.shape}"
        )

    if value.dtype != np.dtype(
        dtype
    ):
        raise TypeError(
            f"{name} expected dtype {np.dtype(dtype)}, "
            f"got {value.dtype}"
        )
