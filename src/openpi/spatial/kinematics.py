"""
OpenPI 3D + tactile 扩展：URDF 前向运动学与 state->joint 映射（kinematics.py）

作用
----
本模块负责把机器人当前 observation.state 转换成 URDF 各 link 在 base_link
坐标系下的位姿，尤其为 tactile.py 提供五根手指 link2 的：

    T_base_link2

完整链路：

    observation.state
        ↓
    state_mapping.csv
        ↓
    joint position q
        ↓
    verified URDF
        ↓
    Forward Kinematics (FK)
        ↓
    T_base_link
        ↓
    tactile.py

本模块只回答：
    “机器人当前姿态下，每个 link 在 base_link 中在哪里、朝向哪里？”

本模块不负责
------------
- taxel local geometry
- tactile raw force 解析
- sensor axis mapping
- RGB-D / point cloud
- dataset 文件读取
- 修改 / 拟合 URDF

模块边界
--------
tactile_geometry.py：
    固定几何：
        taxel 在 finger link2 中的局部位置

kinematics.py：
    动态运动学：
        finger link2 在 base_link 中的当前位姿

tactile.py：
    两者组合：
        p_base = R_base_link2 @ p_link2 + t_base_link2
        f_base = R_base_link2 @ f_link2

严格性原则
----------
1. 不允许 movable joint 缺失时静默使用 q=0。
2. fixed joint 不需要 q。
3. mimic joint 由 master joint 自动计算：
       q_mimic = multiplier * q_master + offset
4. state_mapping 只负责 state index -> URDF joint name；
   不在本文件偷偷做 sign flip / zero offset / degree-radian conversion。
5. 当前 mapping 已经过项目审计，因此这里按“state 中已是 URDF 所需 joint value”
   使用；若未来数据格式改变，应在 mapping / adapter 层显式处理。

URDF transform 约定
-------------------
对 joint：

    parent_link
        ↓  joint origin
    joint frame
        ↓  joint motion
    child_link

实现：

    T_parent_child
        = T_origin_xyz_rpy @ T_joint_motion(q)

其中 URDF rpy 使用 roll-pitch-yaw：

    R_origin = Rz(yaw) @ Ry(pitch) @ Rx(roll)

最终递推：

    T_root_child
        = T_root_parent @ T_parent_child

基本使用
--------
>>> from pathlib import Path
>>> import numpy as np
>>> from openpi.spatial.kinematics import RobotKinematics
>>> from openpi.spatial.kinematics import load_state_mapping_csv
>>> from openpi.spatial.kinematics import joint_positions_from_state
>>>
>>> fk = RobotKinematics(
...     Path("configs/ur7e_xhand/ur7e_xhand_verified.urdf")
... )
>>>
>>> mapping = load_state_mapping_csv(
...     Path("configs/ur7e_xhand/state_mapping.csv")
... )
>>>
>>> q = joint_positions_from_state(
...     state,
...     mapping=mapping,
...     kinematics=fk,
... )
>>>
>>> transforms = fk.compute(
...     q,
...     root="base_link",
... )
>>>
>>> transforms["right_hand_index_rota_link2"].shape
(4, 4)

为 tactile 直接获取五根 link2：
>>> tactile_tf = tactile_link_transforms(
...     state,
...     mapping=mapping,
...     kinematics=fk,
...     finger_link_names=cfg.tactile.finger_link_names,
... )

注意
----
- 该实现只处理 URDF tree，不处理闭链。
- 当前 UR7e + XHand URDF 是 tree，适用。
- 所有角度默认 rad，长度默认 m，与 URDF 规范一致。
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Mapping
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np


# =============================================================================
# 1. URDF joint 数据结构
# =============================================================================

@dataclass(frozen=True, slots=True)
class MimicSpec:
    """
    URDF mimic joint 规则：

        q_this = multiplier * q_master + offset
    """

    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True, slots=True)
class JointSpec:
    """
    FK 需要的最小 URDF joint 信息。
    """

    name: str
    joint_type: str

    parent_link: str
    child_link: str

    origin_xyz_m: np.ndarray
    origin_rpy_rad: np.ndarray

    axis: np.ndarray
    mimic: MimicSpec | None

    def __post_init__(self) -> None:
        for name, value in (
            ("origin_xyz_m", self.origin_xyz_m),
            ("origin_rpy_rad", self.origin_rpy_rad),
            ("axis", self.axis),
        ):
            if not isinstance(
                value,
                np.ndarray,
            ):
                raise TypeError(
                    f"{self.name}.{name} must be numpy.ndarray"
                )

            if value.shape != (3,):
                raise ValueError(
                    f"{self.name}.{name} must have shape (3,), "
                    f"got {value.shape}"
                )

            if not np.all(
                np.isfinite(value)
            ):
                raise ValueError(
                    f"{self.name}.{name} contains NaN / Inf"
                )


# =============================================================================
# 2. RobotKinematics：解析 URDF + 严格 FK
# =============================================================================

class RobotKinematics:
    """
    轻量 URDF FK engine。

    设计目标不是替代完整 robotics library，而是提供当前项目真正需要的：
        - fixed / revolute / continuous / prismatic
        - mimic joint
        - 严格 missing-q 检查
        - root -> all descendant links 的 homogeneous transforms

    这样 preprocessing core 不依赖 ROS / Pinocchio / scipy。
    """

    def __init__(
        self,
        urdf_path: Path,
    ) -> None:
        self.urdf_path = Path(
            urdf_path
        ).expanduser().resolve()

        if not self.urdf_path.is_file():
            raise FileNotFoundError(
                f"URDF not found: {self.urdf_path}"
            )

        (
            self.links,
            self.joints,
            self.children_by_parent,
        ) = _parse_urdf(
            self.urdf_path
        )

        self.joint_by_name = {
            joint.name: joint
            for joint in self.joints
        }

        self.movable = frozenset(
            joint.name
            for joint in self.joints
            if joint.joint_type != "fixed"
        )

        self.mimic_joints = frozenset(
            joint.name
            for joint in self.joints
            if joint.mimic is not None
        )

        self.measured_movable = frozenset(
            self.movable
            - self.mimic_joints
        )

    def compute(
        self,
        q: Mapping[str, float],
        *,
        root: str = "base_link",
    ) -> dict[str, np.ndarray]:
        """
        计算 root -> 所有 descendant link 的 4×4 transform。

        参数
        ----
        q:
            joint_name -> joint value
            revolute / continuous: rad
            prismatic: meter

        root:
            FK 根 link，当前项目使用 base_link。

        返回
        ----
        dict:
            {
                link_name: T_root_link [4,4]
            }

        严格行为
        --------
        对 root 子树中的所有“非 mimic movable joints”：
            q 必须显式存在。

        不允许：
            missing joint -> q=0
        """
        if root not in self.links:
            raise KeyError(
                f"FK root {root!r} not in URDF links"
            )

        q_numeric = {
            str(name): float(value)
            for name, value in q.items()
        }

        for name, value in q_numeric.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Joint {name!r} has non-finite q={value}"
                )

        required = self.required_measured_joints(
            root=root
        )

        missing = sorted(
            required
            - set(q_numeric)
        )

        if missing:
            raise KeyError(
                "Strict FK missing measured movable joints: "
                f"{missing}"
            )

        transforms: dict[
            str,
            np.ndarray,
        ] = {
            root: np.eye(
                4,
                dtype=np.float64,
            )
        }

        # DFS / stack 遍历 URDF tree。
        stack = [
            root
        ]

        while stack:
            parent_link = stack.pop()
            T_root_parent = transforms[
                parent_link
            ]

            for joint in self.children_by_parent.get(
                parent_link,
                ()
            ):
                joint_value = self._joint_value(
                    joint,
                    q_numeric,
                )

                T_parent_child = (
                    _origin_transform(
                        joint.origin_xyz_m,
                        joint.origin_rpy_rad,
                    )
                    @ _joint_motion_transform(
                        joint,
                        joint_value,
                    )
                )

                T_root_child = (
                    T_root_parent
                    @ T_parent_child
                )

                transforms[
                    joint.child_link
                ] = T_root_child

                stack.append(
                    joint.child_link
                )

        return transforms

    def required_measured_joints(
        self,
        *,
        root: str = "base_link",
    ) -> set[str]:
        """
        返回 root 子树中需要从 state 显式提供的 movable joints。

        mimic joint 不计入，因为它由 master joint 计算。
        """
        if root not in self.links:
            raise KeyError(
                f"root {root!r} not in URDF"
            )

        required: set[
            str
        ] = set()

        stack = [
            root
        ]

        while stack:
            parent = stack.pop()

            for joint in self.children_by_parent.get(
                parent,
                ()
            ):
                if (
                    joint.joint_type != "fixed"
                    and joint.mimic is None
                ):
                    required.add(
                        joint.name
                    )

                stack.append(
                    joint.child_link
                )

        return required

    def _joint_value(
        self,
        joint: JointSpec,
        q: Mapping[str, float],
    ) -> float:
        """
        得到一个 joint 的实际 q。

        fixed:
            0

        measured movable:
            q[joint.name]

        mimic:
            multiplier * q[master] + offset
        """
        if joint.joint_type == "fixed":
            return 0.0

        if joint.mimic is None:
            return float(
                q[joint.name]
            )

        master = joint.mimic.joint

        if master not in q:
            raise KeyError(
                f"Mimic joint {joint.name!r} requires master "
                f"{master!r}, but master q is missing"
            )

        return (
            joint.mimic.multiplier
            * float(
                q[master]
            )
            + joint.mimic.offset
        )


# =============================================================================
# 3. state_mapping.csv loader
# =============================================================================

def load_state_mapping_csv(
    path: Path,
) -> dict[str, int]:
    """
    加载 state index -> URDF joint name mapping。

    返回：
        joint_name -> observation.state index

    当前真实 state_mapping.csv 的关键字段是：

        index
        dataset_name
        urdf_joint_name

    同时它还包含：
        sign
        zero_offset
        formula
        coupling_or_mimic
        evidence
        verified
        ...

    这些额外列主要用于 provenance / 审计，不在这里重新解释或二次变换。
    当前 verified FK reference pipeline 直接使用：

        q[joint_name] = observation.state[index]

    因此本 loader 只做：
        1. 找到 urdf_joint_name
        2. 找到 index
        3. 若存在 dataset_name，只保留 *.pos 行

    也兼容少量历史 header alias。

    如果文件不是支持的 header 命名，会明确报错并打印实际 header，
    不会根据列位置静默猜。
    """
    csv_path = Path(
        path
    ).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"state mapping CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                f"{csv_path}: CSV has no header"
            )

        fieldnames = [
            str(name).strip()
            for name in reader.fieldnames
        ]

        joint_key = _first_present(
            fieldnames,
            (
                "urdf_joint_name",
                "joint_name",
                "joint",
                "urdf_joint",
                "name",
            ),
        )

        index_key = _first_present(
            fieldnames,
            (
                "state_index",
                "index",
                "observation_state_index",
                "state_idx",
                "idx",
            ),
        )

        # 当前真实 state_mapping.csv 是一份 provenance-rich 表，
        # 同一个硬件变量可能同时记录 position / velocity / command 等信息。
        # FK 只消费 joint position；如果存在 dataset_name，则只接受 *.pos。
        dataset_name_key = _first_present(
            fieldnames,
            (
                "dataset_name",
                "field_key",
            ),
        )

        if (
            joint_key is None
            or index_key is None
        ):
            raise ValueError(
                f"{csv_path}: unsupported mapping CSV header. "
                f"Found columns={fieldnames}; "
                "need one joint-name column and one state-index column."
            )

        mapping: dict[
            str,
            int,
        ] = {}

        used_indices: dict[
            int,
            str,
        ] = {}

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            joint_name = str(
                row[
                    joint_key
                ]
            ).strip()

            index_text = str(
                row[
                    index_key
                ]
            ).strip()

            if not joint_name:
                continue

            # 如果表里带 dataset_name / field_key，则只保留 joint position。
            # 这与旧 verified pipeline 的 state_mapping 使用方式一致：
            #     row["dataset_name"].endswith(".pos")
            # 避免未来同一个 URDF joint 的 velocity / current 行误进入 FK。
            if dataset_name_key is not None:
                dataset_name = str(
                    row[
                        dataset_name_key
                    ]
                ).strip()

                if dataset_name and not dataset_name.endswith(
                    ".pos"
                ):
                    continue

            if not index_text:
                raise ValueError(
                    f"{csv_path}:{row_number}: "
                    f"missing state index for {joint_name!r}"
                )

            try:
                state_index = int(
                    float(
                        index_text
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{row_number}: invalid state index "
                    f"{index_text!r} for {joint_name!r}"
                ) from exc

            if state_index < 0:
                raise ValueError(
                    f"{csv_path}:{row_number}: negative state index "
                    f"{state_index}"
                )

            if joint_name in mapping:
                raise ValueError(
                    f"{csv_path}:{row_number}: duplicate joint "
                    f"{joint_name!r}"
                )

            if state_index in used_indices:
                raise ValueError(
                    f"{csv_path}:{row_number}: state index {state_index} "
                    f"used by both {used_indices[state_index]!r} "
                    f"and {joint_name!r}"
                )

            mapping[
                joint_name
            ] = state_index

            used_indices[
                state_index
            ] = joint_name

    if not mapping:
        raise ValueError(
            f"{csv_path}: mapping is empty"
        )

    return mapping


# =============================================================================
# 4. observation.state -> q
# =============================================================================

def joint_positions_from_state(
    state: np.ndarray,
    *,
    mapping: Mapping[str, int],
    kinematics: RobotKinematics,
    root: str = "base_link",
) -> dict[str, float]:
    """
    根据 mapping 从 observation.state 提取 strict FK 所需 q。

    不做：
        sign flip
        zero offset
        unit conversion
        clipping

    当前项目已经对 mapping provenance 做过审计；
    此处只负责机械地执行已冻结 mapping。
    """
    state_array = np.asarray(
        state
    )

    if state_array.ndim != 1:
        raise ValueError(
            f"state must be 1-D, got {state_array.shape}"
        )

    required = kinematics.required_measured_joints(
        root=root
    )

    missing_mapping = sorted(
        required
        - set(mapping)
    )

    if missing_mapping:
        raise KeyError(
            "state mapping lacks measured joints required by FK: "
            f"{missing_mapping}"
        )

    q: dict[
        str,
        float,
    ] = {}

    for joint_name in sorted(
        required
    ):
        index = int(
            mapping[
                joint_name
            ]
        )

        if index >= state_array.size:
            raise IndexError(
                f"{joint_name}: state index {index} >= "
                f"state_dim {state_array.size}"
            )

        value = float(
            state_array[
                index
            ]
        )

        if not np.isfinite(value):
            raise ValueError(
                f"{joint_name}: state[{index}] is non-finite ({value})"
            )

        q[
            joint_name
        ] = value

    return q


# =============================================================================
# 5. 面向 tactile.py 的直接接口
# =============================================================================

def tactile_link_transforms(
    state: np.ndarray,
    *,
    mapping: Mapping[str, int],
    kinematics: RobotKinematics,
    finger_link_names: Sequence[str],
    root: str = "base_link",
) -> dict[str, np.ndarray]:
    """
    从 observation.state 直接得到五根 tactile finger link2 的 T_base_link。

    返回：
        {
            "right_hand_thumb_rota_link2":  [4,4],
            "right_hand_index_rota_link2":  [4,4],
            ...
        }

    该结果可以直接传给：
        build_tactile_observation(...)
    """
    q = joint_positions_from_state(
        state,
        mapping=mapping,
        kinematics=kinematics,
        root=root,
    )

    transforms = kinematics.compute(
        q,
        root=root,
    )

    output: dict[
        str,
        np.ndarray,
    ] = {}

    for link_name in finger_link_names:
        if link_name not in transforms:
            raise KeyError(
                f"FK result missing tactile link {link_name!r}"
            )

        output[
            link_name
        ] = np.asarray(
            transforms[
                link_name
            ],
            dtype=np.float64,
        )

    return output


# =============================================================================
# 6. URDF parser
# =============================================================================

def _parse_urdf(
    path: Path,
) -> tuple[
    frozenset[str],
    tuple[JointSpec, ...],
    dict[str, tuple[JointSpec, ...]],
]:
    """
    解析 FK 所需的最小 URDF 信息。
    """
    root = ET.parse(
        path
    ).getroot()

    links = frozenset(
        str(
            link.get(
                "name"
            )
        )
        for link in root.findall(
            "link"
        )
        if link.get(
            "name"
        )
    )

    joints: list[
        JointSpec
    ] = []

    children_temp: dict[
        str,
        list[JointSpec],
    ] = {}

    for element in root.findall(
        "joint"
    ):
        name = element.get(
            "name"
        )

        joint_type = element.get(
            "type"
        )

        if (
            not name
            or not joint_type
        ):
            raise ValueError(
                "URDF joint missing name/type"
            )

        parent_element = element.find(
            "parent"
        )
        child_element = element.find(
            "child"
        )

        if (
            parent_element is None
            or child_element is None
        ):
            raise ValueError(
                f"URDF joint {name!r} missing parent/child"
            )

        parent_link = parent_element.get(
            "link"
        )
        child_link = child_element.get(
            "link"
        )

        if (
            not parent_link
            or not child_link
        ):
            raise ValueError(
                f"URDF joint {name!r} has empty parent/child"
            )

        origin_element = element.find(
            "origin"
        )

        if origin_element is None:
            xyz = np.zeros(
                3,
                dtype=np.float64,
            )
            rpy = np.zeros(
                3,
                dtype=np.float64,
            )
        else:
            xyz = _parse_vector3(
                origin_element.get(
                    "xyz",
                    "0 0 0",
                ),
                field_name=(
                    f"{name}.origin.xyz"
                ),
            )

            rpy = _parse_vector3(
                origin_element.get(
                    "rpy",
                    "0 0 0",
                ),
                field_name=(
                    f"{name}.origin.rpy"
                ),
            )

        axis_element = element.find(
            "axis"
        )

        if joint_type in (
            "revolute",
            "continuous",
            "prismatic",
        ):
            if axis_element is None:
                # URDF 默认 joint axis = +X。
                axis = np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            else:
                axis = _parse_vector3(
                    axis_element.get(
                        "xyz",
                        "1 0 0",
                    ),
                    field_name=(
                        f"{name}.axis"
                    ),
                )

            norm = float(
                np.linalg.norm(
                    axis
                )
            )

            if norm <= 1e-12:
                raise ValueError(
                    f"{name}: zero joint axis"
                )

            axis = (
                axis
                / norm
            )

        elif joint_type == "fixed":
            axis = np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )

        else:
            raise ValueError(
                f"Unsupported URDF joint type {joint_type!r} "
                f"for joint {name!r}"
            )

        mimic_element = element.find(
            "mimic"
        )

        mimic = None

        if mimic_element is not None:
            master = mimic_element.get(
                "joint"
            )

            if not master:
                raise ValueError(
                    f"{name}: mimic joint missing master joint"
                )

            mimic = MimicSpec(
                joint=master,
                multiplier=float(
                    mimic_element.get(
                        "multiplier",
                        "1",
                    )
                ),
                offset=float(
                    mimic_element.get(
                        "offset",
                        "0",
                    )
                ),
            )

        joint = JointSpec(
            name=name,
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz_m=xyz,
            origin_rpy_rad=rpy,
            axis=axis,
            mimic=mimic,
        )

        joints.append(
            joint
        )

        children_temp.setdefault(
            parent_link,
            [],
        ).append(
            joint
        )

    # 基本 tree sanity：
    child_links = [
        joint.child_link
        for joint in joints
    ]

    if len(
        set(
            child_links
        )
    ) != len(
        child_links
    ):
        raise ValueError(
            "URDF is not a tree: a child link has multiple parent joints"
        )

    children_by_parent = {
        parent: tuple(
            children
        )
        for parent, children in children_temp.items()
    }

    return (
        links,
        tuple(
            joints
        ),
        children_by_parent,
    )


# =============================================================================
# 7. Joint origin / motion transform
# =============================================================================

def _origin_transform(
    xyz_m: np.ndarray,
    rpy_rad: np.ndarray,
) -> np.ndarray:
    """
    URDF <origin xyz=... rpy=...> -> 4×4 transform。

    URDF fixed-axis roll-pitch-yaw：
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    roll, pitch, yaw = [
        float(
            value
        )
        for value in rpy_rad
    ]

    rotation = (
        _rotation_z(
            yaw
        )
        @ _rotation_y(
            pitch
        )
        @ _rotation_x(
            roll
        )
    )

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[
        :3,
        :3,
    ] = rotation

    transform[
        :3,
        3,
    ] = xyz_m

    return transform


def _joint_motion_transform(
    joint: JointSpec,
    q: float,
) -> np.ndarray:
    """
    joint motion transform。

    revolute / continuous:
        围绕 joint.axis 旋转 q rad

    prismatic:
        沿 joint.axis 平移 q m

    fixed:
        identity
    """
    transform = np.eye(
        4,
        dtype=np.float64,
    )

    if joint.joint_type == "fixed":
        return transform

    if joint.joint_type in (
        "revolute",
        "continuous",
    ):
        transform[
            :3,
            :3,
        ] = _axis_angle_rotation(
            joint.axis,
            q,
        )

        return transform

    if joint.joint_type == "prismatic":
        transform[
            :3,
            3,
        ] = (
            joint.axis
            * q
        )

        return transform

    raise RuntimeError(
        f"Unhandled joint type {joint.joint_type!r}"
    )


# =============================================================================
# 8. Rotation math
# =============================================================================

def _axis_angle_rotation(
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rodrigues formula：
        unit axis + angle -> 3×3 rotation matrix
    """
    x, y, z = [
        float(
            value
        )
        for value in axis
    ]

    c = math.cos(
        angle
    )
    s = math.sin(
        angle
    )
    one_minus_c = (
        1.0 - c
    )

    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float64,
    )


def _rotation_x(
    angle: float,
) -> np.ndarray:
    c = math.cos(
        angle
    )
    s = math.sin(
        angle
    )

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rotation_y(
    angle: float,
) -> np.ndarray:
    c = math.cos(
        angle
    )
    s = math.sin(
        angle
    )

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rotation_z(
    angle: float,
) -> np.ndarray:
    c = math.cos(
        angle
    )
    s = math.sin(
        angle
    )

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# =============================================================================
# 9. 小型 helper
# =============================================================================

def _parse_vector3(
    text: str,
    *,
    field_name: str,
) -> np.ndarray:
    """
    解析 URDF "x y z" 字符串。
    """
    values = np.fromstring(
        str(
            text
        ),
        sep=" ",
        dtype=np.float64,
    )

    if values.shape != (3,):
        raise ValueError(
            f"{field_name} must contain 3 numbers, got {text!r}"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            f"{field_name} contains NaN / Inf"
        )

    return values


def _first_present(
    fieldnames: Sequence[str],
    candidates: Sequence[str],
) -> str | None:
    """
    在 CSV header 中找第一个匹配 alias。
    """
    lookup = {
        str(
            field
        ).strip().lower(): str(
            field
        ).strip()
        for field in fieldnames
    }

    for candidate in candidates:
        key = str(
            candidate
        ).lower()

        if key in lookup:
            return lookup[
                key
            ]

    return None
