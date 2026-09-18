"""
OpenPI 3D + tactile 扩展：Visual 3D 几何与多相机点云预处理

作用
----
本模块实现 visual branch 中“真正的几何计算”，负责把一个或多个 RGB-D
相机的观测转换为统一的 base_link 视觉点云。

当前完整流程：

    N 个 RGB-D camera
        ↓
    depth unprojection
        ↓
    depth optical frame
        ↓  T_color_depth
    color optical frame
        ↓  T_base_color
    base_link
        ↓
    workspace ROI
        ↓
    multi-camera merge
        ↓
    voxel representative
        ↓
    sampler（当前 baseline: Morton stride）
        ↓
    从每个点的 source camera 回投 RGB
        ↓
    VisualGeometryResult
        xyz_m         [N, 3]
        rgb           [N, 3]
        rgb_valid     [N]

本模块不负责
------------
- 从 YAML / JSON 读取 calibration 文件
- 读取 LeRobot dataset / mp4
- tactile / FK
- 构造最终 SpatialObservation
- PointNet++ / π0

这些职责将在其他模块完成。

坐标约定
--------
统一采用：

    T_A_B:
        将 B frame 中的点变换到 A frame

        p_A = R_A_B @ p_B + t_A_B

NumPy 中点按照 [N,3] 行存储，因此实现为：

    points_A = points_B @ R_A_B.T + t_A_B

当前正式视觉 baseline
--------------------
camera_roles:
    ("front", "left")

depth -> base:
    depth_intrinsics
    -> T_color_depth
    -> T_base_color

workspace:
    frozen base_link ROI

sampling:
    5 mm voxel
    -> Morton stride
    -> 4096 points

RGB:
    每个 3D 点只回投到产生它的 source camera。
    RGB 无效时保留 3D 几何，同时 rgb_valid=False。

模块化设计
----------
1. 相机数量不硬编码：
   所有入口都遍历 VisualPreprocessConfig.camera_roles。
   单相机 / 双相机 / 三相机不需要改本文件算法。

2. calibration 与 geometry 解耦：
   本模块只接收 CameraCalibration 数值对象，不关心它来自哪个 YAML。

3. sampler 有统一入口：
   sample_indices(..., sampler="morton_stride")
   后续增加 FPS / random 时，只需要增加 sampler implementation，
   上层 preprocess 接口不变。

4. diagnostic override 显式：
   FRONT 外参 / K1 只有在 DiagnosticConfig 开启时才应用。

基本使用（数值 calibration 已加载的情况下）
-----------------------------------------
>>> from openpi.spatial.config import make_baseline_config
>>> from openpi.spatial.geometry import build_visual_geometry
>>>
>>> cfg = make_baseline_config()
>>> result = build_visual_geometry(
...     depth_by_role={
...         "front": front_depth,
...         "left": left_depth,
...     },
...     rgb_by_role={
...         "front": front_rgb,
...         "left": left_rgb,
...     },
...     calibrations=calibrations,
...     config=cfg.visual,
...     roi=cfg.roi,
... )
>>> result.xyz_m.shape
(4096, 3)

注意
----
- 本模块的 3D geometry baseline 默认不包含此前诊断得到的 FRONT
  [0, -12, +5] mm 修正。
- FRONT K1 只影响 RGB color projection，不改变 depth-derived 3D xyz。
- 当前实现保留每个 voxel 中真实存在的一个原始点，不生成 voxel center。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Mapping

import numpy as np

from openpi.spatial.config import DiagnosticConfig
from openpi.spatial.config import PinholeIntrinsics
from openpi.spatial.config import VisualPreprocessConfig
from openpi.spatial.config import WorkspaceROI


# =============================================================================
# 1. Runtime calibration 数据结构
# =============================================================================

@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """
    单个 RGB-D 相机在 geometry 层需要的最小 calibration。

    role
        相机逻辑名称，例如 "front" / "left" / "right"。

    depth_intrinsics
        depth optical camera 的 pinhole intrinsics。

    color_intrinsics
        color optical camera 的 pinhole intrinsics。

    depth_scale_m_per_unit
        原始 depth 数值乘以该值后得到米。

    T_color_depth
        shape = [4,4]
        depth optical -> color optical。

    T_base_color
        shape = [4,4]
        color optical -> base_link。
    """

    role: str
    depth_intrinsics: PinholeIntrinsics
    color_intrinsics: PinholeIntrinsics
    depth_scale_m_per_unit: float
    T_color_depth: np.ndarray
    T_base_color: np.ndarray

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("CameraCalibration.role cannot be empty")

        if self.depth_scale_m_per_unit <= 0:
            raise ValueError(
                "depth_scale_m_per_unit must be > 0, got "
                f"{self.depth_scale_m_per_unit}"
            )

        _require_transform(
            "T_color_depth",
            self.T_color_depth,
        )
        _require_transform(
            "T_base_color",
            self.T_base_color,
        )


@dataclass(frozen=True, slots=True)
class CameraGeometryCache:
    """
    针对固定 depth image shape 预计算的 camera geometry cache。

    目的：
        在线 / 批量转换时，不要每一帧重复构造 pixel rays。

    rays_base_per_meter:
        shape = [H*W, 3]

        对每个 depth pixel，如果 depth=1m，
        该 pixel 沿 depth ray 对应到 base_link 后的“方向部分”。

    base_origin_m:
        depth optical origin 在 base_link 中的位置。

    因此每个 pixel 的 base point 可快速写成：

        p_base = ray_base * z_m + base_origin_m
    """

    role: str
    image_height: int
    image_width: int
    rays_base_per_meter: np.ndarray
    base_origin_m: np.ndarray
    depth_scale_m_per_unit: float


@dataclass(frozen=True, slots=True)
class VisualGeometryResult:
    """
    Visual preprocessing 的最终固定点数输出。

    xyz_m
        shape = [N,3], float32, base_link / meter。

    rgb
        shape = [N,3], uint8。

    rgb_valid
        shape = [N], bool。

    source_camera_index
        shape = [N], int16。
        每个点来自 config.camera_roles 中哪一个 camera。
        这是 geometry/debug metadata，不要求进入最终模型。

    voxel_candidate_count
        voxel 后、fixed-N sampling 前有多少候选点。

    merged_roi_count
        所有相机 ROI 点融合后、voxel 前有多少点。
    """

    xyz_m: np.ndarray
    rgb: np.ndarray
    rgb_valid: np.ndarray
    source_camera_index: np.ndarray

    voxel_candidate_count: int
    merged_roi_count: int

    def __post_init__(self) -> None:
        _require_array(
            "xyz_m",
            self.xyz_m,
            ndim=2,
            trailing_shape=(3,),
            dtype=np.float32,
        )
        _require_array(
            "rgb",
            self.rgb,
            ndim=2,
            trailing_shape=(3,),
            dtype=np.uint8,
        )
        _require_array(
            "rgb_valid",
            self.rgb_valid,
            ndim=1,
            trailing_shape=(),
            dtype=np.bool_,
        )
        _require_array(
            "source_camera_index",
            self.source_camera_index,
            ndim=1,
            trailing_shape=(),
            dtype=np.int16,
        )

        n = self.xyz_m.shape[0]
        if (
            self.rgb.shape[0] != n
            or self.rgb_valid.shape[0] != n
            or self.source_camera_index.shape[0] != n
        ):
            raise ValueError(
                "VisualGeometryResult fields must have the same leading "
                f"dimension, got xyz={self.xyz_m.shape}, "
                f"rgb={self.rgb.shape}, "
                f"rgb_valid={self.rgb_valid.shape}, "
                f"source_camera_index={self.source_camera_index.shape}"
            )

        if not np.all(np.isfinite(self.xyz_m)):
            raise ValueError("xyz_m contains NaN / Inf")


# =============================================================================
# 2. 基础 3D transformation / projection
# =============================================================================

def transform_points(
    points: np.ndarray,
    T_target_source: np.ndarray,
) -> np.ndarray:
    """
    将 [N,3] points 从 source frame 变换到 target frame。

    坐标约定：
        p_target = R_target_source @ p_source + t_target_source
    """
    points = np.asarray(points)
    T = np.asarray(T_target_source)

    _require_transform(
        "T_target_source",
        T,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape [N,3], got {points.shape}"
        )

    return (
        points @ T[:3, :3].T
        + T[:3, 3]
    )


def invert_transform(
    T_target_source: np.ndarray,
) -> np.ndarray:
    """返回 rigid transform 的逆：T_source_target。"""
    T = np.asarray(
        T_target_source,
        dtype=np.float64,
    )
    _require_transform(
        "T_target_source",
        T,
    )

    R = T[:3, :3]
    t = T[:3, 3]

    inverse = np.eye(
        4,
        dtype=np.float64,
    )
    inverse[:3, :3] = R.T
    inverse[:3, 3] = -(R.T @ t)
    return inverse


def project_pinhole(
    points_camera_m: np.ndarray,
    intrinsics: PinholeIntrinsics,
) -> np.ndarray:
    """
    将 camera optical frame 中 [N,3] 点投影到 image pixel。

    返回：
        uv shape = [N,2]
        uv[:,0] = u / column
        uv[:,1] = v / row

    z<=0 的点会得到 NaN，由调用方判为无效。
    """
    points = np.asarray(
        points_camera_m,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points_camera_m must have shape [N,3], "
            f"got {points.shape}"
        )

    z = points[:, 2]

    uv = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64,
    )

    valid_z = (
        np.isfinite(z)
        & (z > 0)
    )

    x = points[valid_z, 0]
    y = points[valid_z, 1]
    zv = z[valid_z]

    uv[valid_z, 0] = (
        intrinsics.fx * x / zv
        + intrinsics.cx
    )
    uv[valid_z, 1] = (
        intrinsics.fy * y / zv
        + intrinsics.cy
    )

    return uv


# =============================================================================
# 3. Diagnostic calibration override
# =============================================================================

def apply_diagnostic_overrides(
    calibrations: Mapping[str, CameraCalibration],
    diagnostics: DiagnosticConfig,
) -> dict[str, CameraCalibration]:
    """
    根据 DiagnosticConfig 返回新的 calibration mapping。

    不原地修改输入 calibrations。

    当前只有 FRONT diagnostic：
        1. base-frame translation correction
        2. fitted color K1

    其中：
        extrinsic correction 会改变 FRONT 3D xyz；
        intrinsic K1 只改变 FRONT RGB projection。
    """
    output = dict(calibrations)

    if "front" not in output:
        return output

    front = output["front"]

    # -------------------------------------------------------------------------
    # 3.1 可选 FRONT 外参平移修正
    # -------------------------------------------------------------------------
    if diagnostics.enable_front_extrinsic_correction:
        T_base_color = np.array(
            front.T_base_color,
            dtype=np.float64,
            copy=True,
        )

        delta = np.asarray(
            diagnostics.front_extrinsic_translation_base_m,
            dtype=np.float64,
        )

        if delta.shape != (3,):
            raise ValueError(
                "front_extrinsic_translation_base_m "
                f"must have shape [3], got {delta.shape}"
            )

        # 这是 base-frame 中的纯平移左乘，因此 rotation 不变：
        # t_new = t_old + delta_base
        T_base_color[:3, 3] += delta

        front = replace(
            front,
            T_base_color=T_base_color,
        )

    # -------------------------------------------------------------------------
    # 3.2 可选 FRONT color K1
    # -------------------------------------------------------------------------
    if diagnostics.enable_front_intrinsic_k1:
        front = replace(
            front,
            color_intrinsics=diagnostics.front_color_k1,
        )

    output["front"] = front
    return output


# =============================================================================
# 4. 预计算 camera ray cache
# =============================================================================

def build_camera_cache(
    calibration: CameraCalibration,
    *,
    image_height: int,
    image_width: int,
) -> CameraGeometryCache:
    """
    为一个相机和固定 depth image shape 构造 geometry cache。

    这里把两步 rigid transform 合并：

        depth
          -> T_color_depth
        color
          -> T_base_color
        base

    得到：
        T_base_depth = T_base_color @ T_color_depth

    之后每帧只需要：
        p_base = ray_base * z + t_base_depth
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            "image_height / image_width must be > 0"
        )

    k = calibration.depth_intrinsics

    u, v = np.meshgrid(
        np.arange(
            image_width,
            dtype=np.float32,
        ),
        np.arange(
            image_height,
            dtype=np.float32,
        ),
    )

    ray_depth = np.column_stack(
        (
            ((u - k.cx) / k.fx).reshape(-1),
            ((v - k.cy) / k.fy).reshape(-1),
            np.ones(
                image_height * image_width,
                dtype=np.float32,
            ),
        )
    ).astype(
        np.float32,
        copy=False,
    )

    T_base_depth = (
        np.asarray(
            calibration.T_base_color,
            dtype=np.float64,
        )
        @ np.asarray(
            calibration.T_color_depth,
            dtype=np.float64,
        )
    )

    R_base_depth = T_base_depth[:3, :3].astype(
        np.float32,
    )
    t_base_depth = T_base_depth[:3, 3].astype(
        np.float32,
    )

    rays_base = (
        ray_depth @ R_base_depth.T
    ).astype(
        np.float32,
        copy=False,
    )

    return CameraGeometryCache(
        role=calibration.role,
        image_height=image_height,
        image_width=image_width,
        rays_base_per_meter=rays_base,
        base_origin_m=t_base_depth,
        depth_scale_m_per_unit=calibration.depth_scale_m_per_unit,
    )


# =============================================================================
# 5. 单相机 depth -> base_link + ROI
# =============================================================================

def depth_to_base_roi(
    depth_raw: np.ndarray,
    cache: CameraGeometryCache,
    *,
    roi: WorkspaceROI,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把一个 depth frame 转换为 base_link ROI 内的 3D 点。

    返回
    ----
    xyz_m:
        [M,3], float32

    source_pixel_flat:
        [M], int32
        对应原始 depth image flatten 后的 pixel index。
        主要用于 debug / provenance；RGB 关联目前使用 3D 回投。
    """
    depth = np.asarray(depth_raw)

    expected_shape = (
        cache.image_height,
        cache.image_width,
    )

    if depth.shape != expected_shape:
        raise ValueError(
            f"{cache.role} depth shape expected "
            f"{expected_shape}, got {depth.shape}"
        )

    z_m = (
        depth.reshape(-1).astype(
            np.float32,
        )
        * np.float32(
            cache.depth_scale_m_per_unit
        )
    )

    depth_valid = (
        np.isfinite(z_m)
        & (z_m >= min_depth_m)
        & (z_m <= max_depth_m)
    )

    source_pixel = np.flatnonzero(
        depth_valid
    ).astype(
        np.int32,
    )

    z_valid = z_m[depth_valid]

    xyz_base = (
        cache.rays_base_per_meter[depth_valid]
        * z_valid[:, None]
        + cache.base_origin_m[None, :]
    ).astype(
        np.float32,
        copy=False,
    )

    roi_valid = (
        (xyz_base[:, 0] >= roi.x_min_m)
        & (xyz_base[:, 0] <= roi.x_max_m)
        & (xyz_base[:, 1] >= roi.y_min_m)
        & (xyz_base[:, 1] <= roi.y_max_m)
        & (xyz_base[:, 2] >= roi.z_min_m)
        & (xyz_base[:, 2] <= roi.z_max_m)
    )

    return (
        xyz_base[roi_valid],
        source_pixel[roi_valid],
    )


# =============================================================================
# 6. N-camera merge
# =============================================================================

def merge_camera_depths(
    depth_by_role: Mapping[str, np.ndarray],
    *,
    camera_roles: tuple[str, ...],
    caches: Mapping[str, CameraGeometryCache],
    roi: WorkspaceROI,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将任意数量 camera 的 ROI 点云融合。

    这里不写任何 front / left 特化逻辑。

    camera_roles 控制：
        1. 哪些 camera 被使用
        2. camera index 的稳定顺序

    返回
    ----
    xyz_m:
        [M,3]

    source_camera_index:
        [M]
        第 i 个点来自 camera_roles 中哪一个 camera。

    source_pixel_flat:
        [M]
        对应 source depth image 的 flatten pixel id。
    """
    xyz_parts: list[np.ndarray] = []
    camera_parts: list[np.ndarray] = []
    pixel_parts: list[np.ndarray] = []

    for camera_index, role in enumerate(
        camera_roles
    ):
        if role not in depth_by_role:
            raise KeyError(
                f"Missing depth frame for camera role {role!r}"
            )

        if role not in caches:
            raise KeyError(
                f"Missing geometry cache for camera role {role!r}"
            )

        xyz, pixel = depth_to_base_roi(
            depth_by_role[role],
            caches[role],
            roi=roi,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )

        xyz_parts.append(xyz)
        camera_parts.append(
            np.full(
                len(xyz),
                camera_index,
                dtype=np.int16,
            )
        )
        pixel_parts.append(pixel)

    if not xyz_parts:
        raise ValueError(
            "camera_roles produced no camera inputs"
        )

    xyz_merged = np.concatenate(
        xyz_parts,
        axis=0,
    )
    source_camera_index = np.concatenate(
        camera_parts,
        axis=0,
    )
    source_pixel_flat = np.concatenate(
        pixel_parts,
        axis=0,
    )

    return (
        xyz_merged,
        source_camera_index,
        source_pixel_flat,
    )


# =============================================================================
# 7. Voxel representative
# =============================================================================

def voxel_representatives(
    xyz_m: np.ndarray,
    source_camera_index: np.ndarray,
    source_pixel_flat: np.ndarray,
    *,
    roi: WorkspaceROI,
    voxel_size_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    每个 occupied voxel 保留一个“真实观测点”。

    注意：
        这里不是生成 voxel center，而是从该 voxel 中选择第一个真实点，
        因此输出坐标仍来自实际 RGB-D observation。

    返回：
        xyz_voxel
        camera_voxel
        pixel_voxel
        ix
        iy
        iz
    """
    xyz = np.asarray(
        xyz_m,
        dtype=np.float32,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz_m must have shape [N,3], got {xyz.shape}"
        )

    if voxel_size_m <= 0:
        raise ValueError(
            "voxel_size_m must be > 0"
        )

    n = len(xyz)
    if (
        len(source_camera_index) != n
        or len(source_pixel_flat) != n
    ):
        raise ValueError(
            "xyz / source_camera_index / source_pixel_flat "
            "must have the same length"
        )

    ix = np.floor(
        (xyz[:, 0] - roi.x_min_m)
        / voxel_size_m
    ).astype(
        np.int32,
    )
    iy = np.floor(
        (xyz[:, 1] - roi.y_min_m)
        / voxel_size_m
    ).astype(
        np.int32,
    )
    iz = np.floor(
        (xyz[:, 2] - roi.z_min_m)
        / voxel_size_m
    ).astype(
        np.int32,
    )

    nx = (
        int(
            np.ceil(
                (roi.x_max_m - roi.x_min_m)
                / voxel_size_m
            )
        )
        + 1
    )
    ny = (
        int(
            np.ceil(
                (roi.y_max_m - roi.y_min_m)
                / voxel_size_m
            )
        )
        + 1
    )

    linear_voxel = (
        ix.astype(np.int64)
        + np.int64(nx)
        * (
            iy.astype(np.int64)
            + np.int64(ny)
            * iz.astype(np.int64)
        )
    )

    # np.unique 返回每个 voxel 第一次出现的位置。
    _, first_index = np.unique(
        linear_voxel,
        return_index=True,
    )

    # unique 会按 voxel id 排序；再按原 observation 顺序排序，
    # 使“每个 voxel 保留第一个真实点”的语义稳定。
    first_index.sort()

    return (
        xyz[first_index],
        np.asarray(
            source_camera_index,
            dtype=np.int16,
        )[first_index],
        np.asarray(
            source_pixel_flat,
            dtype=np.int32,
        )[first_index],
        ix[first_index],
        iy[first_index],
        iz[first_index],
    )


# =============================================================================
# 8. Morton stride sampler
# =============================================================================

def morton_codes(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
) -> np.ndarray:
    """
    将整数 voxel 坐标编码为 3D Morton / Z-order code。

    Morton order 会把空间上相近的 voxel 尽量排列得相近。
    后续沿 Morton 排序序列均匀 stride 取点，可得到较均匀的空间覆盖。
    """
    x = np.asarray(
        ix,
        dtype=np.uint64,
    )
    y = np.asarray(
        iy,
        dtype=np.uint64,
    )
    z = np.asarray(
        iz,
        dtype=np.uint64,
    )

    def _part1by2(
        value: np.ndarray,
    ) -> np.ndarray:
        value = value.copy()
        value &= np.uint64(
            0x1FFFFF
        )
        value = (
            value
            | (value << np.uint64(32))
        ) & np.uint64(
            0x1F00000000FFFF
        )
        value = (
            value
            | (value << np.uint64(16))
        ) & np.uint64(
            0x1F0000FF0000FF
        )
        value = (
            value
            | (value << np.uint64(8))
        ) & np.uint64(
            0x100F00F00F00F00F
        )
        value = (
            value
            | (value << np.uint64(4))
        ) & np.uint64(
            0x10C30C30C30C30C3
        )
        value = (
            value
            | (value << np.uint64(2))
        ) & np.uint64(
            0x1249249249249249
        )
        return value

    return (
        _part1by2(x)
        | (
            _part1by2(y)
            << np.uint64(1)
        )
        | (
            _part1by2(z)
            << np.uint64(2)
        )
    )


def morton_stride_indices(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    *,
    num_points: int,
) -> np.ndarray:
    """
    对 voxel candidate 做固定点数 Morton-stride sampling。

    如果 candidate 少于 num_points，直接报错。
    当前不静默 repeat / padding，因为那会改变数据语义。
    """
    candidate_count = len(ix)

    if num_points <= 0:
        raise ValueError(
            "num_points must be > 0"
        )

    if candidate_count < num_points:
        raise ValueError(
            "Not enough voxel candidates for fixed-N sampling: "
            f"candidate_count={candidate_count}, "
            f"num_points={num_points}"
        )

    if candidate_count == num_points:
        return np.arange(
            candidate_count,
            dtype=np.int64,
        )

    order = np.argsort(
        morton_codes(
            ix,
            iy,
            iz,
        ),
        kind="mergesort",
    )

    positions = np.rint(
        np.linspace(
            0,
            candidate_count - 1,
            num_points,
        )
    ).astype(
        np.int64,
    )

    return order[positions]


def sample_indices(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    *,
    num_points: int,
    sampler: str,
) -> np.ndarray:
    """
    sampler 的统一入口。

    当前仅实现：
        morton_stride

    未来增加：
        fps
        random
        learned sampler
    时，上层 build_visual_geometry() 不需要改变接口。
    """
    if sampler == "morton_stride":
        return morton_stride_indices(
            ix,
            iy,
            iz,
            num_points=num_points,
        )

    raise ValueError(
        f"Unsupported sampler {sampler!r}. "
        "Currently supported: ('morton_stride',)"
    )


# =============================================================================
# 9. RGB association
# =============================================================================

def colorize_from_source_cameras(
    xyz_base_m: np.ndarray,
    source_camera_index: np.ndarray,
    *,
    camera_roles: tuple[str, ...],
    rgb_by_role: Mapping[str, np.ndarray],
    calibrations: Mapping[str, CameraCalibration],
    invalid_rgb: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 base_link 中的 3D 点回投到其 source camera 的 color image。

    关键原则：
        RGB projection 不负责决定 3D 点是否存在。

    因此投影失败时：
        xyz 仍保留；
        rgb 填 invalid_rgb；
        rgb_valid=False。

    返回
    ----
    rgb:
        [N,3], uint8

    rgb_valid:
        [N], bool
    """
    xyz = np.asarray(
        xyz_base_m,
        dtype=np.float64,
    )

    source_camera_index = np.asarray(
        source_camera_index,
        dtype=np.int16,
    )

    n = len(xyz)

    if source_camera_index.shape != (n,):
        raise ValueError(
            "source_camera_index must have shape [N], got "
            f"{source_camera_index.shape}"
        )

    invalid_rgb_array = np.asarray(
        invalid_rgb,
        dtype=np.uint8,
    )

    if invalid_rgb_array.shape != (3,):
        raise ValueError(
            "invalid_rgb must contain exactly 3 values"
        )

    rgb = np.tile(
        invalid_rgb_array[None, :],
        (n, 1),
    )

    rgb_valid = np.zeros(
        n,
        dtype=np.bool_,
    )

    for camera_index, role in enumerate(
        camera_roles
    ):
        if role not in rgb_by_role:
            raise KeyError(
                f"Missing RGB frame for camera role {role!r}"
            )

        if role not in calibrations:
            raise KeyError(
                f"Missing calibration for camera role {role!r}"
            )

        mask = (
            source_camera_index
            == camera_index
        )

        if not np.any(mask):
            continue

        calibration = calibrations[role]
        rgb_image = np.asarray(
            rgb_by_role[role]
        )

        if (
            rgb_image.ndim != 3
            or rgb_image.shape[2] != 3
        ):
            raise ValueError(
                f"{role} RGB must have shape [H,W,3], "
                f"got {rgb_image.shape}"
            )

        T_color_base = invert_transform(
            calibration.T_base_color
        )

        points_color = transform_points(
            xyz[mask],
            T_color_base,
        )

        uv = project_pinhole(
            points_color,
            calibration.color_intrinsics,
        )

        # 与已验证旧 pipeline 一致：
        # 使用最近 pixel 的整数索引。
        pixel = np.rint(
            np.nan_to_num(
                uv,
                nan=-1e9,
                posinf=-1e9,
                neginf=-1e9,
            )
        ).astype(
            np.int64,
        )

        height, width = (
            rgb_image.shape[:2]
        )

        valid = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(points_color).all(axis=1)
            & (points_color[:, 2] > 0)
            & (pixel[:, 0] >= 0)
            & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0)
            & (pixel[:, 1] < height)
        )

        global_index = np.flatnonzero(
            mask
        )

        if np.any(valid):
            selected = global_index[valid]

            rgb[selected] = rgb_image[
                pixel[valid, 1],
                pixel[valid, 0],
            ]

            rgb_valid[selected] = True

    return (
        rgb.astype(
            np.uint8,
            copy=False,
        ),
        rgb_valid,
    )


# =============================================================================
# 10. 完整 Visual preprocessing 入口
# =============================================================================

def build_visual_geometry(
    *,
    depth_by_role: Mapping[str, np.ndarray],
    rgb_by_role: Mapping[str, np.ndarray],
    calibrations: Mapping[str, CameraCalibration],
    config: VisualPreprocessConfig,
    roi: WorkspaceROI,
    diagnostics: DiagnosticConfig | None = None,
    caches: Mapping[str, CameraGeometryCache] | None = None,
) -> VisualGeometryResult:
    """
    完成一帧 visual spatial preprocessing。

    这是 geometry.py 面向上层 preprocess.py 的主要 public API。

    流程
    ----
    1. 根据 config.camera_roles 选择 N 个相机
    2. 可选 diagnostic calibration override
    3. depth -> base_link + ROI
    4. N-camera merge
    5. voxel representative
    6. fixed-N sampling
    7. source-camera RGB association
    8. 返回 VisualGeometryResult

    caches
    ------
    若不传：
        本函数根据当前 depth shape 临时构建 cache。

    在线部署 / 整集转换：
        建议由更上层 SpatialPreprocessor 预先构建并复用 cache，
        避免每帧重复创建 pixel rays。
    """
    active_calibrations = dict(
        calibrations
    )

    if diagnostics is not None:
        active_calibrations = apply_diagnostic_overrides(
            active_calibrations,
            diagnostics,
        )

    # -------------------------------------------------------------------------
    # 10.1 检查 camera inputs，并准备 cache
    # -------------------------------------------------------------------------
    active_caches: dict[
        str,
        CameraGeometryCache,
    ] = {}

    for role in config.camera_roles:
        if role not in active_calibrations:
            raise KeyError(
                f"Missing calibration for camera role {role!r}"
            )

        if role not in depth_by_role:
            raise KeyError(
                f"Missing depth frame for camera role {role!r}"
            )

        depth = np.asarray(
            depth_by_role[role]
        )

        if depth.ndim != 2:
            raise ValueError(
                f"{role} depth must be [H,W], got {depth.shape}"
            )

        if caches is not None and role in caches:
            cache = caches[role]

            if (
                cache.image_height,
                cache.image_width,
            ) != depth.shape:
                raise ValueError(
                    f"{role} cache shape "
                    f"{(cache.image_height, cache.image_width)} "
                    f"does not match depth shape {depth.shape}"
                )

            # 如果 diagnostics 改变了 extrinsic，则旧 cache 已经失效。
            if (
                diagnostics is not None
                and role == "front"
                and diagnostics.enable_front_extrinsic_correction
            ):
                cache = build_camera_cache(
                    active_calibrations[role],
                    image_height=depth.shape[0],
                    image_width=depth.shape[1],
                )
        else:
            cache = build_camera_cache(
                active_calibrations[role],
                image_height=depth.shape[0],
                image_width=depth.shape[1],
            )

        active_caches[role] = cache

    # -------------------------------------------------------------------------
    # 10.2 N-camera depth -> base_link ROI
    # -------------------------------------------------------------------------
    (
        xyz_merged,
        source_camera_merged,
        source_pixel_merged,
    ) = merge_camera_depths(
        depth_by_role,
        camera_roles=config.camera_roles,
        caches=active_caches,
        roi=roi,
        min_depth_m=config.min_depth_m,
        max_depth_m=config.max_depth_m,
    )

    merged_roi_count = len(
        xyz_merged
    )

    # -------------------------------------------------------------------------
    # 10.3 Voxel representative
    # -------------------------------------------------------------------------
    (
        xyz_voxel,
        source_camera_voxel,
        source_pixel_voxel,
        ix,
        iy,
        iz,
    ) = voxel_representatives(
        xyz_merged,
        source_camera_merged,
        source_pixel_merged,
        roi=roi,
        voxel_size_m=config.voxel_size_m,
    )

    voxel_candidate_count = len(
        xyz_voxel
    )

    # -------------------------------------------------------------------------
    # 10.4 Fixed-N sampling
    # -------------------------------------------------------------------------
    selected = sample_indices(
        ix,
        iy,
        iz,
        num_points=config.num_points,
        sampler=config.sampler,
    )

    xyz_sampled = xyz_voxel[
        selected
    ].astype(
        np.float32,
        copy=False,
    )

    source_camera_sampled = (
        source_camera_voxel[
            selected
        ].astype(
            np.int16,
            copy=False,
        )
    )

    # source_pixel 当前主要作为 provenance/debug 信息。
    # 这里保留局部变量，后续如需要可扩展进 VisualGeometryResult。
    _source_pixel_sampled = (
        source_pixel_voxel[
            selected
        ]
    )

    # -------------------------------------------------------------------------
    # 10.5 RGB association
    # -------------------------------------------------------------------------
    rgb, rgb_valid = colorize_from_source_cameras(
        xyz_sampled,
        source_camera_sampled,
        camera_roles=config.camera_roles,
        rgb_by_role=rgb_by_role,
        calibrations=active_calibrations,
    )

    return VisualGeometryResult(
        xyz_m=xyz_sampled,
        rgb=rgb,
        rgb_valid=rgb_valid,
        source_camera_index=source_camera_sampled,
        voxel_candidate_count=voxel_candidate_count,
        merged_roi_count=merged_roi_count,
    )


# =============================================================================
# 11. 内部 validation helper
# =============================================================================

def _require_transform(
    name: str,
    value: np.ndarray,
) -> None:
    """检查 4×4 homogeneous transform 的基本结构。"""
    array = np.asarray(
        value
    )

    if array.shape != (4, 4):
        raise ValueError(
            f"{name} must have shape [4,4], got {array.shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains NaN / Inf"
        )

    if not np.allclose(
        array[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),
        atol=1e-6,
    ):
        raise ValueError(
            f"{name} has invalid homogeneous last row: {array[3]}"
        )


def _require_array(
    name: str,
    value: np.ndarray,
    *,
    ndim: int,
    trailing_shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> None:
    """统一检查 ndarray 的 ndim / trailing shape / dtype。"""
    if not isinstance(
        value,
        np.ndarray,
    ):
        raise TypeError(
            f"{name} must be numpy.ndarray, "
            f"got {type(value).__name__}"
        )

    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have ndim={ndim}, "
            f"got shape {value.shape}"
        )

    if (
        trailing_shape
        and value.shape[
            -len(trailing_shape):
        ] != trailing_shape
    ):
        raise ValueError(
            f"{name} must end with shape "
            f"{trailing_shape}, got {value.shape}"
        )

    if value.dtype != np.dtype(
        dtype
    ):
        raise TypeError(
            f"{name} must have dtype "
            f"{np.dtype(dtype)}, got {value.dtype}"
        )
