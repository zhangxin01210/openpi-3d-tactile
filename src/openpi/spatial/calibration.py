"""
OpenPI 3D + tactile 扩展：相机标定加载与标准化（calibration.py）

作用
----
本模块负责把“磁盘上的标定文件格式”转换成 geometry.py 能直接使用的
统一数值对象 CameraCalibration。

当前正式来源分成两层：

    cameras_0903.json
        RGB-D 相机自身内部标定：
            - color intrinsics
            - depth intrinsics
            - depth_scale
            - T_color_depth
            - 可能还包含历史 T_base_color

    extrinsic_0909.yaml
        相机相对机器人 base_link 的更新外参：
            - T_base_color

最终得到：

    dict[str, CameraCalibration]

例如：

    {
        "front": CameraCalibration(...),
        "left": CameraCalibration(...),
    }

然后 geometry.py 完全不需要知道 JSON / YAML 的原始 schema。

本模块不负责
------------
- depth -> point cloud
- RGB projection
- ROI / voxel / sampling
- tactile / FK
- dataset 读取
- diagnostic FRONT [0,-12,+5] mm 或 K1 修正
  （这些仍由 geometry.py + DiagnosticConfig 显式控制）

为什么需要单独这一层
--------------------
真实研究工程里，算法内部对象与磁盘文件格式不应该绑死。

例如以后：
    - 从双相机变成三相机；
    - 换另一批标定文件；
    - 标定文件字段名轻微变化；
    - 部署端从 ROS / SDK 获取标定；

只需要把数据标准化成 CameraCalibration，
geometry.py 的数学实现不应该跟着重写。

加载流程
--------
    repo root
        ↓
    CalibrationConfig 中 repo-relative path
        ↓
    JSON / YAML loader
        ↓
    按 camera role 找到原始 camera entry
        ↓
    parse intrinsics / depth scale / transforms
        ↓
    用 extrinsic_0909 覆盖 T_base_color
        ↓
    CameraCalibration
        ↓
    geometry.py

基本使用
--------
>>> from pathlib import Path
>>> from openpi.spatial.config import make_baseline_config
>>> from openpi.spatial.calibration import load_camera_calibrations
>>>
>>> cfg = make_baseline_config()
>>> bundle = load_camera_calibrations(
...     repo_root=Path("."),
...     calibration_config=cfg.calibration,
...     camera_roles=cfg.visual.camera_roles,
... )
>>>
>>> bundle.cameras.keys()
dict_keys(['front', 'left'])
>>> bundle.cameras["front"].depth_scale_m_per_unit
0.001

模块化说明
----------
camera_roles 是唯一决定“加载哪些相机”的入口。

因此以后：

    ("front",)
    ("front", "left")
    ("front", "left", "right")

都走同一套 loader，不在代码中硬编码双相机。

注意
----
1. loader 不会静默猜一个缺失的关键标定。
   如果 depth intrinsics / color intrinsics / depth_scale /
   T_color_depth / T_base_color 缺失，会立即报错并打印可用 key。

2. extrinsic file 中若存在某 role 的 T_base_color，
   默认优先覆盖 camera_config 里的旧值。

3. quaternion 统一按 xyzw 解释。

4. transform 约定与 geometry.py 完全一致：

       T_A_B:
           p_A = R_A_B @ p_B + t_A_B
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np
import yaml

from openpi.spatial.config import CalibrationConfig
from openpi.spatial.config import PinholeIntrinsics
from openpi.spatial.geometry import CameraCalibration


# =============================================================================
# 1. 对外返回对象：CalibrationBundle
# =============================================================================

@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    """
    一次 calibration 加载结果。

    cameras
        标准化后的 role -> CameraCalibration。

    camera_config_path
        实际读取的 RGB-D 内部标定文件。

    camera_extrinsic_path
        实际读取的 robot-camera 外参文件。

    requested_roles
        本次调用要求加载的 camera roles。

    extrinsic_overridden_roles
        哪些相机的 T_base_color 确实被 external extrinsic 覆盖。

    这个对象的价值主要是 provenance：
        以后训练某个 checkpoint 时，可以明确知道用了哪套标定来源。
    """

    cameras: dict[str, CameraCalibration]
    camera_config_path: Path
    camera_extrinsic_path: Path
    requested_roles: tuple[str, ...]
    extrinsic_overridden_roles: tuple[str, ...]


# =============================================================================
# 2. Public API：加载完整 camera calibration
# =============================================================================

def load_camera_calibrations(
    *,
    repo_root: Path,
    calibration_config: CalibrationConfig,
    camera_roles: Sequence[str],
) -> CalibrationBundle:
    """
    加载并标准化指定 camera roles 的完整 RGB-D calibration。

    参数
    ----
    repo_root:
        当前 Git 仓库根目录。

    calibration_config:
        config.py 中的 CalibrationConfig。

    camera_roles:
        需要加载的相机角色，例如：
            ("front",)
            ("front", "left")
            ("front", "left", "right")

    返回
    ----
    CalibrationBundle

    关键行为
    --------
    1. 从 cameras_0903.json 读取 RGB-D 内部标定；
    2. 从 extrinsic_0909.yaml 读取最新 T_base_color；
    3. external extrinsic 优先覆盖 camera config 中的历史外参；
    4. 每个 role 最终都标准化为 CameraCalibration。
    """
    root = Path(repo_root).expanduser().resolve()

    roles = tuple(str(role).strip() for role in camera_roles)

    if not roles:
        raise ValueError("camera_roles must contain at least one role")

    if any(not role for role in roles):
        raise ValueError(
            f"camera_roles cannot contain empty names: {roles}"
        )

    if len(set(roles)) != len(roles):
        raise ValueError(
            f"camera_roles must be unique, got {roles}"
        )

    camera_config_path = _resolve_repo_relative(
        root,
        calibration_config.camera_config_path,
    )

    camera_extrinsic_path = _resolve_repo_relative(
        root,
        calibration_config.camera_extrinsic_path,
    )

    camera_doc = _load_json(
        camera_config_path
    )

    extrinsic_doc = _load_yaml(
        camera_extrinsic_path
    )

    camera_entries = _extract_role_entries(
        camera_doc,
        source_name="camera config",
        preferred_container_keys=(
            "cameras",
            "camera_configs",
            "camera",
        ),
    )

    extrinsic_entries = _extract_role_entries(
        extrinsic_doc,
        source_name="extrinsic config",
        preferred_container_keys=(
            "cameras",
            "extrinsics",
            "camera_extrinsics",
            "transforms",
            # 兼容外参文件直接按：
            # T_base_color:
            #   front: {...}
            #   left: {...}
            # 组织的形式。
            "T_base_color",
        ),
        allow_empty=True,
    )

    calibrations: dict[
        str,
        CameraCalibration,
    ] = {}

    overridden_roles: list[str] = []

    for role in roles:
        if role not in camera_entries:
            raise KeyError(
                f"Camera role {role!r} not found in "
                f"{camera_config_path}. "
                f"Available roles: {sorted(camera_entries)}"
            )

        camera_entry = camera_entries[role]

        external_entry = extrinsic_entries.get(
            role
        )

        calibration, overridden = (
            _build_camera_calibration(
                role=role,
                camera_entry=camera_entry,
                external_entry=external_entry,
            )
        )

        calibrations[role] = calibration

        if overridden:
            overridden_roles.append(role)

    return CalibrationBundle(
        cameras=calibrations,
        camera_config_path=camera_config_path,
        camera_extrinsic_path=camera_extrinsic_path,
        requested_roles=roles,
        extrinsic_overridden_roles=tuple(
            overridden_roles
        ),
    )


# =============================================================================
# 3. 单相机标准化
# =============================================================================

def _build_camera_calibration(
    *,
    role: str,
    camera_entry: Mapping[str, Any],
    external_entry: Mapping[str, Any] | None,
) -> tuple[CameraCalibration, bool]:
    """
    把单个 role 的原始配置标准化为 CameraCalibration。

    返回：
        calibration
        external_extrinsic_was_used
    """

    # -------------------------------------------------------------------------
    # 3.1 depth intrinsics
    # -------------------------------------------------------------------------
    depth_intrinsics_raw = _find_value(
        camera_entry,
        direct_keys=(
            "depth_intrinsics",
            "depth_K",
        ),
        nested_paths=(
            ("depth", "intrinsics"),
            ("depth_camera", "intrinsics"),
        ),
        field_name=f"{role}.depth_intrinsics",
    )

    depth_intrinsics = _parse_intrinsics(
        depth_intrinsics_raw,
        field_name=f"{role}.depth_intrinsics",
    )

    # -------------------------------------------------------------------------
    # 3.2 color intrinsics
    # -------------------------------------------------------------------------
    color_intrinsics_raw = _find_value(
        camera_entry,
        direct_keys=(
            "color_intrinsics",
            "rgb_intrinsics",
            "color_K",
        ),
        nested_paths=(
            ("color", "intrinsics"),
            ("rgb", "intrinsics"),
            ("color_camera", "intrinsics"),
        ),
        field_name=f"{role}.color_intrinsics",
    )

    color_intrinsics = _parse_intrinsics(
        color_intrinsics_raw,
        field_name=f"{role}.color_intrinsics",
    )

    # -------------------------------------------------------------------------
    # 3.3 depth scale
    # -------------------------------------------------------------------------
    depth_scale_raw = _find_value(
        camera_entry,
        direct_keys=(
            "depth_scale",
            "depth_scale_m_per_unit",
            "meters_per_depth_unit",
        ),
        nested_paths=(
            ("depth", "scale"),
            ("depth", "depth_scale"),
            ("depth", "meters_per_unit"),
        ),
        field_name=f"{role}.depth_scale",
    )

    depth_scale = _parse_positive_float(
        depth_scale_raw,
        field_name=f"{role}.depth_scale",
    )

    # -------------------------------------------------------------------------
    # 3.4 T_color_depth
    # -------------------------------------------------------------------------
    T_color_depth_raw = _find_value(
        camera_entry,
        direct_keys=(
            "T_color_depth",
            "depth_to_color",
            "T_depth_to_color",
        ),
        nested_paths=(
            ("transforms", "T_color_depth"),
            ("extrinsics", "T_color_depth"),
        ),
        field_name=f"{role}.T_color_depth",
    )

    T_color_depth = _parse_transform(
        T_color_depth_raw,
        field_name=f"{role}.T_color_depth",
    )

    # -------------------------------------------------------------------------
    # 3.5 T_base_color
    #
    # external extrinsic 优先；只有 external 没提供时才回退到
    # camera config 自身的历史值。
    # -------------------------------------------------------------------------
    external_T = _try_extract_base_color_transform(
        external_entry,
        role=role,
    )

    external_used = (
        external_T is not None
    )

    if external_T is not None:
        T_base_color = external_T
    else:
        base_color_raw = _find_value(
            camera_entry,
            direct_keys=(
                "T_base_color",
                "color_to_base",
            ),
            nested_paths=(
                ("transforms", "T_base_color"),
                ("extrinsics", "T_base_color"),
            ),
            field_name=f"{role}.T_base_color",
        )

        T_base_color = _parse_transform(
            base_color_raw,
            field_name=f"{role}.T_base_color",
        )

    return (
        CameraCalibration(
            role=role,
            depth_intrinsics=depth_intrinsics,
            color_intrinsics=color_intrinsics,
            depth_scale_m_per_unit=depth_scale,
            T_color_depth=T_color_depth,
            T_base_color=T_base_color,
        ),
        external_used,
    )


# =============================================================================
# 4. Intrinsics parser
# =============================================================================

def _parse_intrinsics(
    raw: Any,
    *,
    field_name: str,
) -> PinholeIntrinsics:
    """
    把常见 pinhole intrinsics 表达标准化为 PinholeIntrinsics。

    当前支持：
        {"fx":..., "fy":..., "cx":..., "cy":...}

    也兼容：
        {"focal_length_x":..., ...}
    """
    if not isinstance(
        raw,
        Mapping,
    ):
        raise TypeError(
            f"{field_name} must be a mapping, "
            f"got {type(raw).__name__}"
        )

    aliases = {
        "fx": (
            "fx",
            "focal_length_x",
        ),
        "fy": (
            "fy",
            "focal_length_y",
        ),
        "cx": (
            "cx",
            "principal_point_x",
            "ppx",
        ),
        "cy": (
            "cy",
            "principal_point_y",
            "ppy",
        ),
    }

    values: dict[str, float] = {}

    for canonical, keys in aliases.items():
        value = None

        for key in keys:
            if key in raw:
                value = raw[key]
                break

        if value is None:
            raise KeyError(
                f"{field_name} missing {canonical}. "
                f"Available keys: {sorted(raw.keys())}"
            )

        values[canonical] = float(
            value
        )

    return PinholeIntrinsics(
        fx=values["fx"],
        fy=values["fy"],
        cx=values["cx"],
        cy=values["cy"],
    )


# =============================================================================
# 5. Transform parser
# =============================================================================

def _parse_transform(
    raw: Any,
    *,
    field_name: str,
) -> np.ndarray:
    """
    把常见 rigid transform 表达标准化成 float64 [4,4] matrix。

    支持：

    A. 直接 4×4 list / ndarray

    B. dict:
        translation_m: [tx,ty,tz]
        quaternion_xyzw: [qx,qy,qz,qw]

    C. dict:
        translation_m: [...]
        rotation_matrix_rows:
            [[...],[...],[...]]

    D. dict:
        matrix:
            [[4×4]]

    也兼容少量历史字段：
        translation
        translation_xyz
        position_xyz
    """
    # -------------------------------------------------------------------------
    # 5.1 直接 matrix
    # -------------------------------------------------------------------------
    if isinstance(
        raw,
        (list, tuple, np.ndarray),
    ):
        matrix = np.asarray(
            raw,
            dtype=np.float64,
        )

        _validate_transform_matrix(
            matrix,
            field_name=field_name,
        )

        return matrix

    if not isinstance(
        raw,
        Mapping,
    ):
        raise TypeError(
            f"{field_name} must be transform mapping or 4x4 matrix, "
            f"got {type(raw).__name__}"
        )

    # -------------------------------------------------------------------------
    # 5.2 dict 内直接 matrix
    # -------------------------------------------------------------------------
    for matrix_key in (
        "matrix",
        "matrix_4x4",
        "transform_matrix",
    ):
        if matrix_key in raw:
            matrix = np.asarray(
                raw[matrix_key],
                dtype=np.float64,
            )

            _validate_transform_matrix(
                matrix,
                field_name=field_name,
            )

            return matrix

    # -------------------------------------------------------------------------
    # 5.3 translation
    # -------------------------------------------------------------------------
    translation = None

    for key in (
        "translation_m",
        "translation",
        "translation_xyz",
        "position_xyz",
        "xyz",
    ):
        if key in raw:
            translation = np.asarray(
                raw[key],
                dtype=np.float64,
            )
            break

    if translation is None:
        raise KeyError(
            f"{field_name} missing translation. "
            f"Available keys: {sorted(raw.keys())}"
        )

    if translation.shape != (3,):
        raise ValueError(
            f"{field_name} translation must have shape [3], "
            f"got {translation.shape}"
        )

    # -------------------------------------------------------------------------
    # 5.4 rotation
    # -------------------------------------------------------------------------
    rotation = None

    if "quaternion_xyzw" in raw:
        rotation = _quaternion_xyzw_to_matrix(
            raw["quaternion_xyzw"],
            field_name=(
                f"{field_name}.quaternion_xyzw"
            ),
        )

    elif "rotation_matrix_rows" in raw:
        rotation = np.asarray(
            raw["rotation_matrix_rows"],
            dtype=np.float64,
        )

    elif "rotation_matrix" in raw:
        rotation = np.asarray(
            raw["rotation_matrix"],
            dtype=np.float64,
        )

    else:
        raise KeyError(
            f"{field_name} missing rotation. "
            "Expected quaternion_xyzw or rotation_matrix_rows. "
            f"Available keys: {sorted(raw.keys())}"
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            f"{field_name} rotation must have shape [3,3], "
            f"got {rotation.shape}"
        )

    matrix = np.eye(
        4,
        dtype=np.float64,
    )

    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation

    _validate_transform_matrix(
        matrix,
        field_name=field_name,
    )

    return matrix


def _quaternion_xyzw_to_matrix(
    quaternion: Any,
    *,
    field_name: str,
) -> np.ndarray:
    """
    quaternion xyzw -> 3×3 rotation matrix。

    不依赖 scipy，减少 preprocessing core 的额外 dependency。
    """
    q = np.asarray(
        quaternion,
        dtype=np.float64,
    )

    if q.shape != (4,):
        raise ValueError(
            f"{field_name} must have shape [4], got {q.shape}"
        )

    if not np.all(
        np.isfinite(q)
    ):
        raise ValueError(
            f"{field_name} contains NaN / Inf"
        )

    norm = float(
        np.linalg.norm(q)
    )

    if norm <= 1e-12:
        raise ValueError(
            f"{field_name} has zero norm"
        )

    x, y, z, w = (
        q / norm
    )

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [
                1.0 - 2.0 * (yy + zz),
                2.0 * (xy - wz),
                2.0 * (xz + wy),
            ],
            [
                2.0 * (xy + wz),
                1.0 - 2.0 * (xx + zz),
                2.0 * (yz - wx),
            ],
            [
                2.0 * (xz - wy),
                2.0 * (yz + wx),
                1.0 - 2.0 * (xx + yy),
            ],
        ],
        dtype=np.float64,
    )


# =============================================================================
# 6. External extrinsic：从 role entry 中找到 T_base_color
# =============================================================================

def _try_extract_base_color_transform(
    entry: Mapping[str, Any] | None,
    *,
    role: str,
) -> np.ndarray | None:
    """
    尝试从 external extrinsic 的 role entry 中提取 T_base_color。

    支持两种常见结构：

    A.
        front:
            T_base_color:
                translation_m: ...
                quaternion_xyzw: ...

    B.
        front:
            from_frame: ...
            to_frame: ...
            translation_m: ...
            quaternion_xyzw: ...

    第二种情况下，整个 role entry 自身就是 transform spec。
    """
    if entry is None:
        return None

    if not isinstance(
        entry,
        Mapping,
    ):
        raise TypeError(
            f"Extrinsic entry for {role!r} must be mapping, "
            f"got {type(entry).__name__}"
        )

    # -------------------------------------------------------------------------
    # 6.1 T_base_color 被包在字段里
    # -------------------------------------------------------------------------
    for key in (
        "T_base_color",
        "base_from_color",
        "color_to_base",
    ):
        if key in entry:
            return _parse_transform(
                entry[key],
                field_name=(
                    f"external.{role}.{key}"
                ),
            )

    # -------------------------------------------------------------------------
    # 6.2 role entry 自身就是 transform spec
    # -------------------------------------------------------------------------
    transform_keys = {
        "translation_m",
        "translation",
        "translation_xyz",
        "position_xyz",
        "xyz",
        "quaternion_xyzw",
        "rotation_matrix_rows",
        "rotation_matrix",
        "matrix",
        "matrix_4x4",
        "transform_matrix",
    }

    if (
        transform_keys
        & set(entry.keys())
    ):
        return _parse_transform(
            entry,
            field_name=(
                f"external.{role}"
            ),
        )

    return None


# =============================================================================
# 7. 顶层 role mapping 提取
# =============================================================================

def _extract_role_entries(
    doc: Any,
    *,
    source_name: str,
    preferred_container_keys: Sequence[str],
    allow_empty: bool = False,
) -> dict[str, Mapping[str, Any]]:
    """
    从 JSON / YAML 顶层结构中提取 role -> mapping。

    支持：

    A.
        {
            "cameras": {
                "front": {...},
                "left": {...}
            }
        }

    B.
        {
            "front": {...},
            "left": {...}
        }

    C. list:
        [
            {"role": "front", ...},
            {"role": "left", ...}
        ]

    这样以后标定文件 schema 有轻微变化时，
    只需要适配这一层，而不是污染 geometry.py。
    """
    if doc is None:
        if allow_empty:
            return {}
        raise ValueError(
            f"{source_name} is empty"
        )

    # -------------------------------------------------------------------------
    # 7.1 优先找显式 container
    # -------------------------------------------------------------------------
    if isinstance(
        doc,
        Mapping,
    ):
        for key in preferred_container_keys:
            if key not in doc:
                continue

            container = doc[key]

            extracted = _normalize_role_container(
                container,
                source_name=(
                    f"{source_name}.{key}"
                ),
            )

            if extracted:
                return extracted

        # ---------------------------------------------------------------------
        # 7.2 顶层本身就是 role mapping
        # ---------------------------------------------------------------------
        direct: dict[
            str,
            Mapping[str, Any],
        ] = {}

        for key, value in doc.items():
            if isinstance(
                value,
                Mapping,
            ):
                direct[str(key)] = value

        if direct:
            return direct

    # -------------------------------------------------------------------------
    # 7.3 顶层是 list
    # -------------------------------------------------------------------------
    if isinstance(
        doc,
        list,
    ):
        extracted = _normalize_role_container(
            doc,
            source_name=source_name,
        )

        if extracted:
            return extracted

    if allow_empty:
        return {}

    raise ValueError(
        f"Could not extract camera roles from {source_name}. "
        f"Top-level type={type(doc).__name__}"
    )


def _normalize_role_container(
    container: Any,
    *,
    source_name: str,
) -> dict[str, Mapping[str, Any]]:
    """
    把 mapping / list container 统一成 role -> entry。
    """
    if isinstance(
        container,
        Mapping,
    ):
        result: dict[
            str,
            Mapping[str, Any],
        ] = {}

        for role, entry in container.items():
            if isinstance(
                entry,
                Mapping,
            ):
                result[str(role)] = entry

        return result

    if isinstance(
        container,
        list,
    ):
        result = {}

        for index, entry in enumerate(
            container
        ):
            if not isinstance(
                entry,
                Mapping,
            ):
                continue

            role = None

            for key in (
                "role",
                "camera_role",
                "name",
                "camera_name",
            ):
                if key in entry:
                    role = str(
                        entry[key]
                    )
                    break

            if role is None:
                raise KeyError(
                    f"{source_name}[{index}] has no role/name field. "
                    f"Available keys: {sorted(entry.keys())}"
                )

            result[role] = entry

        return result

    return {}


# =============================================================================
# 8. 通用 nested-field 查找
# =============================================================================

def _find_value(
    mapping: Mapping[str, Any],
    *,
    direct_keys: Sequence[str],
    nested_paths: Sequence[
        tuple[str, ...]
    ],
    field_name: str,
) -> Any:
    """
    从有限、明确的 alias 中查找字段。

    这里不是无限“猜 schema”：
    只兼容我们当前项目和少量常见历史命名。
    如果都找不到就立即报错。
    """
    for key in direct_keys:
        if key in mapping:
            return mapping[key]

    for path in nested_paths:
        current: Any = mapping
        found = True

        for key in path:
            if (
                not isinstance(
                    current,
                    Mapping,
                )
                or key not in current
            ):
                found = False
                break

            current = current[key]

        if found:
            return current

    raise KeyError(
        f"Cannot find {field_name}. "
        f"Available top-level keys: {sorted(mapping.keys())}"
    )


# =============================================================================
# 9. JSON / YAML IO
# =============================================================================

def _load_json(
    path: Path,
) -> Any:
    """读取 JSON，并在文件缺失时提供明确错误。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"Camera config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def _load_yaml(
    path: Path,
) -> Any:
    """读取 YAML，并在文件缺失时提供明确错误。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"Camera extrinsic config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


# =============================================================================
# 10. 路径与数值 validation
# =============================================================================

def _resolve_repo_relative(
    repo_root: Path,
    relative_path: Path,
) -> Path:
    """
    把 repo-relative path 转成 absolute path。

    CalibrationConfig 已禁止 absolute path；
    这里再次保持这个 invariant。
    """
    path = Path(
        relative_path
    )

    if path.is_absolute():
        raise ValueError(
            "Expected repo-relative path, got absolute path: "
            f"{path}"
        )

    return (
        repo_root
        / path
    ).resolve()


def _parse_positive_float(
    raw: Any,
    *,
    field_name: str,
) -> float:
    """解析必须 >0 的 scalar。"""
    value = float(
        raw
    )

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be finite and >0, got {value}"
        )

    return value


def _validate_transform_matrix(
    matrix: np.ndarray,
    *,
    field_name: str,
) -> None:
    """
    检查 homogeneous rigid transform 的基本结构。

    这里只检查：
        shape
        finite
        last row
        rotation 正交性 / determinant

    这样可以尽早抓住：
        - 错误 reshape
        - quaternion parse 错
        - 非刚体 matrix
    """
    if matrix.shape != (4, 4):
        raise ValueError(
            f"{field_name} must have shape [4,4], got {matrix.shape}"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            f"{field_name} contains NaN / Inf"
        )

    if not np.allclose(
        matrix[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        atol=1e-6,
    ):
        raise ValueError(
            f"{field_name} invalid homogeneous last row: "
            f"{matrix[3].tolist()}"
        )

    rotation = matrix[:3, :3]

    orthogonal_error = float(
        np.linalg.norm(
            rotation.T @ rotation
            - np.eye(3),
            ord="fro",
        )
    )

    determinant = float(
        np.linalg.det(rotation)
    )

    if orthogonal_error > 1e-4:
        raise ValueError(
            f"{field_name} rotation is not orthogonal; "
            f"Frobenius error={orthogonal_error:.6g}"
        )

    if abs(
        determinant - 1.0
    ) > 1e-4:
        raise ValueError(
            f"{field_name} rotation det must be ~1, "
            f"got {determinant:.8f}"
        )
