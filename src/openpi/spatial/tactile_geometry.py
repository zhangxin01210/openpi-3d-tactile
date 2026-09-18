"""
OpenPI 3D + tactile 扩展：XHand taxel 官方几何加载（tactile_geometry.py）

作用
----
本模块只负责回答一个问题：

    “XHand 每个 taxel 在对应 finger link2 坐标系中的固定位置是什么？”

它读取厂家提供的 transformed JSON，把官方毫米坐标统一转换为米，
并组织成 tactile.py 需要的：

    taxel_xyz_link_m [5, 120, 3]

其中 finger 顺序固定为：

    0 thumb
    1 index
    2 middle
    3 ring
    4 pinky

为什么单独做这个模块
--------------------
taxel local geometry 是“传感器安装后的固定几何”，它和每帧 robot FK
不是一回事。

本模块负责：
    official transformed JSON
        -> link2-frame taxel geometry

kinematics.py 负责：
    robot state + URDF
        -> T_base_link2(q_t)

tactile.py 最后负责：
    p_base = R_base_link2 @ p_link2 + t_base_link2

这样“固定传感器几何”和“动态机器人运动学”完全解耦。

官方坐标约定
------------
厂家 transformed JSON 已经把 sensor-body frame 转换到 URDF link2 frame，
因此 transformed 文件中的坐标可以直接作为 p_link2 使用，不能再次执行
sensor -> link2 变换。

当前右手：
    thumb:
        T30 transformed geometry

    index / middle / ring / pinky:
        共用 T16 transformed geometry

注意：
    四根 T16 手指共用的是“同一局部几何模板”；
    它们在机器人中的不同空间位置由各自 T_base_link2 决定。

单位
----
官方 JSON 坐标单位：mm

本项目统一输出：
    meter

即：
    xyz_m = xyz_mm / 1000

可选审计功能
------------
如果同时拥有 raw sensor-body JSON 和 transformed JSON，本模块可以根据
官方文档给出的固定变换重新计算 transformed 坐标，并做逐 taxel parity：

T16:
    x' = z + 5.35
    y' = x
    z' = y + 15.86

T30:
    x' = x
    y' = y + 16.42
    z' = z + 9.41

这项审计的目的不是运行时重复变换，而是确认：
    1. 你拿到的 transformed 文件确实是官方 link2 geometry；
    2. 没有左右手文件混用；
    3. measurement point ordering 没有错。

基本使用
--------
>>> from pathlib import Path
>>> from openpi.spatial.config import make_baseline_config
>>> from openpi.spatial.tactile_geometry import load_xhand_taxel_geometry
>>>
>>> cfg = make_baseline_config()
>>>
>>> geometry = load_xhand_taxel_geometry(
...     t16_transformed_path=Path("configs/ur7e_xhand/points_t16_transformed.json"),
...     t30_right_transformed_path=Path(
...         "configs/ur7e_xhand/points_t30_right_hand_transformed.json"
...     ),
...     tactile_config=cfg.tactile,
... )
>>>
>>> geometry.xyz_link_m.shape
(5, 120, 3)

注意
----
1. 本模块不读取 URDF，也不执行 FK。
2. transformed JSON 已经是 link2 frame；不要重复 sensor->link2。
3. 只接受 point id 恰好为 1..120 的几何文件。
4. 当前没有构造 surface normal；这里只有 taxel position。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.spatial.config import TactileConfig


# =============================================================================
# 1. 对外结果对象
# =============================================================================

@dataclass(frozen=True, slots=True)
class TaxelGeometryBundle:
    """
    XHand 五指固定 taxel local geometry。

    xyz_link_m:
        [5,120,3] float32
        每个 taxel 在对应 finger link2 frame 中的位置，单位米。

    finger_names:
        与 xyz_link_m 第 0 维一致。

    t16_source:
        四根非拇指使用的官方 T16 transformed JSON。

    t30_source:
        右手拇指使用的官方 T30 transformed JSON。

    point_ids:
        [120] int16
        固定为 1..120。
    """

    xyz_link_m: np.ndarray
    finger_names: tuple[str, ...]
    t16_source: Path
    t30_source: Path
    point_ids: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(
            self.xyz_link_m,
            np.ndarray,
        ):
            raise TypeError(
                "xyz_link_m must be numpy.ndarray"
            )

        if self.xyz_link_m.shape != (
            5,
            120,
            3,
        ):
            raise ValueError(
                "xyz_link_m must have shape (5,120,3), "
                f"got {self.xyz_link_m.shape}"
            )

        if self.xyz_link_m.dtype != np.float32:
            raise TypeError(
                "xyz_link_m must be float32, "
                f"got {self.xyz_link_m.dtype}"
            )

        if not np.all(
            np.isfinite(
                self.xyz_link_m
            )
        ):
            raise ValueError(
                "xyz_link_m contains NaN / Inf"
            )

        if len(
            self.finger_names
        ) != 5:
            raise ValueError(
                "finger_names must contain 5 fingers"
            )

        expected_ids = np.arange(
            1,
            121,
            dtype=np.int16,
        )

        if not np.array_equal(
            self.point_ids,
            expected_ids,
        ):
            raise ValueError(
                "point_ids must be exactly 1..120"
            )


@dataclass(frozen=True, slots=True)
class GeometryAuditResult:
    """
    raw sensor-body JSON 与 transformed JSON 的 parity 结果。

    error_mm:
        每个 taxel 的 3D 欧氏距离误差。

    median_mm / p95_mm / max_mm:
        便于快速判断 transformed 文件是否和官方固定变换一致。
    """

    sensor_model: str
    point_count: int
    median_mm: float
    p95_mm: float
    max_mm: float


# =============================================================================
# 2. Public API：加载右手五指 transformed geometry
# =============================================================================

def load_xhand_taxel_geometry(
    *,
    t16_transformed_path: Path,
    t30_right_transformed_path: Path,
    tactile_config: TactileConfig,
) -> TaxelGeometryBundle:
    """
    加载右手 XHand 五指 taxel local geometry。

    当前硬件对应关系：
        thumb  -> T30
        index  -> T16
        middle -> T16
        ring   -> T16
        pinky  -> T16

    transformed JSON 本身已经位于各 finger 的 link2 frame，
    因此这里只做：
        read
        sort by point id
        validate
        mm -> m
        expand to five fingers

    不再执行任何 sensor->link2 transform。
    """
    _validate_expected_finger_layout(
        tactile_config
    )

    t16_path = Path(
        t16_transformed_path
    ).expanduser().resolve()

    t30_path = Path(
        t30_right_transformed_path
    ).expanduser().resolve()

    t16_mm, t16_ids = (
        read_measurement_points_mm(
            t16_path
        )
    )

    t30_mm, t30_ids = (
        read_measurement_points_mm(
            t30_path
        )
    )

    if not np.array_equal(
        t16_ids,
        t30_ids,
    ):
        raise ValueError(
            "T16 and T30 point ids differ; "
            "cannot construct a stable 1..120 taxel ordering"
        )

    # -------------------------------------------------------------------------
    # 五指局部几何：
    #   thumb 使用 T30；
    #   其余四指共用相同 T16 local template。
    # -------------------------------------------------------------------------
    xyz_mm = np.empty(
        (
            tactile_config.num_fingers,
            tactile_config.taxels_per_finger,
            3,
        ),
        dtype=np.float32,
    )

    xyz_mm[0] = t30_mm

    for finger_index in range(
        1,
        tactile_config.num_fingers,
    ):
        xyz_mm[
            finger_index
        ] = t16_mm

    xyz_m = (
        xyz_mm
        * np.float32(
            1e-3
        )
    ).astype(
        np.float32,
        copy=False,
    )

    return TaxelGeometryBundle(
        xyz_link_m=xyz_m,
        finger_names=tuple(
            tactile_config.finger_names
        ),
        t16_source=t16_path,
        t30_source=t30_path,
        point_ids=t16_ids,
    )


# =============================================================================
# 3. 官方 JSON reader
# =============================================================================

def read_measurement_points_mm(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从厂家 JSON 读取 120 个 measurement points。

    支持字段：
        measurement_points
        points

    每个元素必须至少包含：
        point
        x
        y
        z

    返回
    ----
    xyz_mm:
        [120,3] float32

    point_ids:
        [120] int16，必须严格为 1..120
    """
    json_path = Path(
        path
    ).expanduser().resolve()

    if not json_path.is_file():
        raise FileNotFoundError(
            f"Tactile geometry JSON not found: {json_path}"
        )

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"{json_path} top-level JSON must be object"
        )

    points = (
        data.get(
            "measurement_points"
        )
        or data.get(
            "points"
        )
    )

    if points is None:
        raise KeyError(
            f"{json_path} has no measurement_points / points. "
            f"Available keys: {sorted(data.keys())}"
        )

    if not isinstance(
        points,
        list,
    ):
        raise TypeError(
            f"{json_path}: measurement_points must be list"
        )

    parsed: list[
        tuple[int, float, float, float]
    ] = []

    for index, item in enumerate(
        points
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                f"{json_path}: point[{index}] must be object"
            )

        required = (
            "point",
            "x",
            "y",
            "z",
        )

        missing = [
            key
            for key in required
            if key not in item
        ]

        if missing:
            raise KeyError(
                f"{json_path}: point[{index}] missing {missing}"
            )

        parsed.append(
            (
                int(
                    item["point"]
                ),
                float(
                    item["x"]
                ),
                float(
                    item["y"]
                ),
                float(
                    item["z"]
                ),
            )
        )

    parsed.sort(
        key=lambda row: row[0]
    )

    point_ids = np.asarray(
        [
            row[0]
            for row in parsed
        ],
        dtype=np.int16,
    )

    xyz_mm = np.asarray(
        [
            [
                row[1],
                row[2],
                row[3],
            ]
            for row in parsed
        ],
        dtype=np.float32,
    )

    expected_ids = np.arange(
        1,
        121,
        dtype=np.int16,
    )

    if not np.array_equal(
        point_ids,
        expected_ids,
    ):
        raise ValueError(
            f"{json_path}: expected point ids exactly 1..120, "
            f"got {point_ids.tolist()}"
        )

    if xyz_mm.shape != (
        120,
        3,
    ):
        raise ValueError(
            f"{json_path}: expected xyz shape (120,3), "
            f"got {xyz_mm.shape}"
        )

    if not np.all(
        np.isfinite(
            xyz_mm
        )
    ):
        raise ValueError(
            f"{json_path}: coordinates contain NaN / Inf"
        )

    return (
        xyz_mm,
        point_ids,
    )


# =============================================================================
# 4. 官方 sensor-body -> link2 固定变换
#
# 注意：
# 这些函数用于“审计 raw JSON 与 transformed JSON 是否一致”，
# 不用于正常运行时 transformed geometry 的再次变换。
# =============================================================================

def sensor_points_to_link2_mm(
    xyz_sensor_mm: np.ndarray,
    *,
    sensor_model: str,
) -> np.ndarray:
    """
    根据官方文档，把 sensor-body points 转到 URDF link2 frame。

    支持：
        T16
        T30

    T16:
        x' = z + 5.35
        y' = x
        z' = y + 15.86

    T30:
        x' = x
        y' = y + 16.42
        z' = z + 9.41
    """
    xyz = np.asarray(
        xyz_sensor_mm,
        dtype=np.float32,
    )

    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
    ):
        raise ValueError(
            "xyz_sensor_mm must have shape [N,3], "
            f"got {xyz.shape}"
        )

    model = str(
        sensor_model
    ).strip().upper()

    transformed = np.empty_like(
        xyz,
        dtype=np.float32,
    )

    if model == "T16":
        transformed[:, 0] = (
            xyz[:, 2]
            + np.float32(
                5.35
            )
        )
        transformed[:, 1] = (
            xyz[:, 0]
        )
        transformed[:, 2] = (
            xyz[:, 1]
            + np.float32(
                15.86
            )
        )

    elif model == "T30":
        transformed[:, 0] = (
            xyz[:, 0]
        )
        transformed[:, 1] = (
            xyz[:, 1]
            + np.float32(
                16.42
            )
        )
        transformed[:, 2] = (
            xyz[:, 2]
            + np.float32(
                9.41
            )
        )

    else:
        raise ValueError(
            f"Unsupported sensor_model {sensor_model!r}; "
            "expected 'T16' or 'T30'"
        )

    return transformed


# =============================================================================
# 5. 可选 parity audit：raw JSON vs transformed JSON
# =============================================================================

def audit_transformed_geometry(
    *,
    raw_sensor_path: Path,
    transformed_path: Path,
    sensor_model: str,
) -> GeometryAuditResult:
    """
    验证 transformed JSON 是否等于：
        official sensor-body JSON
        + 官方固定 sensor->link2 变换。

    这是一次性 / migration QA，不是正常 preprocessing 必经步骤。
    """
    raw_mm, raw_ids = (
        read_measurement_points_mm(
            raw_sensor_path
        )
    )

    transformed_mm, transformed_ids = (
        read_measurement_points_mm(
            transformed_path
        )
    )

    if not np.array_equal(
        raw_ids,
        transformed_ids,
    ):
        raise ValueError(
            "raw and transformed point ids are not identical"
        )

    expected_mm = (
        sensor_points_to_link2_mm(
            raw_mm,
            sensor_model=sensor_model,
        )
    )

    error_mm = np.linalg.norm(
        expected_mm.astype(
            np.float64
        )
        - transformed_mm.astype(
            np.float64
        ),
        axis=1,
    )

    return GeometryAuditResult(
        sensor_model=str(
            sensor_model
        ).upper(),
        point_count=int(
            len(error_mm)
        ),
        median_mm=float(
            np.median(
                error_mm
            )
        ),
        p95_mm=float(
            np.percentile(
                error_mm,
                95,
            )
        ),
        max_mm=float(
            np.max(
                error_mm
            )
        ),
    )


# =============================================================================
# 6. 小型几何 QA
# =============================================================================

def geometry_summary(
    bundle: TaxelGeometryBundle,
) -> dict[str, Any]:
    """
    返回便于日志打印的 local geometry 摘要。

    不做 visualization，只给：
        每根手指 xyz min / max
        点数
        source path
    """
    summary: dict[
        str,
        Any,
    ] = {
        "t16_source": str(
            bundle.t16_source
        ),
        "t30_source": str(
            bundle.t30_source
        ),
        "fingers": {},
    }

    for finger_index, finger_name in enumerate(
        bundle.finger_names
    ):
        xyz = bundle.xyz_link_m[
            finger_index
        ]

        summary["fingers"][
            finger_name
        ] = {
            "count": int(
                len(xyz)
            ),
            "min_m": xyz.min(
                axis=0
            ).astype(
                float
            ).tolist(),
            "max_m": xyz.max(
                axis=0
            ).astype(
                float
            ).tolist(),
        }

    return summary


# =============================================================================
# 7. 内部 layout contract
# =============================================================================

def _validate_expected_finger_layout(
    config: TactileConfig,
) -> None:
    """
    当前 loader 专门对应右手 XHand1 五指布局。

    如果未来真的换成不同手型 / 不同 finger 数量，
    应新增对应 geometry loader，而不是静默套用本模板。
    """
    expected_fingers = (
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    )

    if tuple(
        config.finger_names
    ) != expected_fingers:
        raise ValueError(
            "Current XHand geometry loader expects finger order "
            f"{expected_fingers}, got {tuple(config.finger_names)}"
        )

    if (
        config.num_fingers != 5
        or config.taxels_per_finger != 120
    ):
        raise ValueError(
            "Current XHand geometry loader expects "
            "5 fingers × 120 taxels"
        )
