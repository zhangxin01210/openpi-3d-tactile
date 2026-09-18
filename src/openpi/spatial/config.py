"""
OpenPI 3D + tactile 扩展：空间预处理配置（SpatialPreprocessConfig）

作用
----
本模块只负责描述“空间预处理应该按什么参数运行”，不执行任何真正的
RGB-D 重建、点云融合、采样、FK、触觉力变换或文件读取。

它与 schema.py 的分工是：

    schema.py
        定义最终数据“长什么样”。

    config.py
        定义这次实验“按什么参数做”。

    geometry.py / tactile.py
        定义具体“怎么算”。

    preprocess.py
        把 config + geometry + tactile 组织成完整预处理流水线。

设计目标
--------
1. 模块化：
   - 单相机 / 双相机 / 三相机，只改 camera_roles；
   - 2048 / 4096 / 8192 点，只改 num_points；
   - 5 mm / 3 mm voxel，只改 voxel_size_m；
   - 更换 sampling 方法，只改 sampler 名称；
   - diagnostic calibration 通过显式开关控制。

2. 配置与算法解耦：
   config 只保存参数，不读取 YAML、不解析 URDF、不执行矩阵运算。

3. 不写机器相关绝对路径：
   所有资产路径均为 repo-relative path，保证本地、部署机、训练服务器
   可以共享同一份 Git commit。

4. baseline 与 diagnostic 分离：
   当前正式 baseline 默认不启用此前诊断得到的 FRONT 外参 / 内参修正。

5. 相机“内部标定”与“机器人外参”分离：
   - cameras_0903.json 提供每台相机的 color/depth intrinsics、
     depth_scale、T_color_depth 等 RGB-D 内部标定；
   - extrinsic_0909.yaml 只负责覆盖/提供 T_base_color 等机器人外参。
   两者是不同来源，不能混成一个文件概念。

当前 baseline
-------------
Visual:
    camera_roles = ("front", "left")
    voxel_size_m = 0.005
    num_points = 4096
    sampler = "morton_stride"
    depth range = [0.10, 2.50] m

Tactile:
    5 fingers × 120 taxels = 600
    thumb / T30: [Fx,Fy,Fz] -> [Fx,Fy,Fz]
    index/middle/ring/pinky / T16: [Fx,Fy,Fz] -> [Fz,Fx,Fy]

Diagnostic-only:
    FRONT base-frame translation correction = [0, -0.012, +0.005] m
    FRONT fitted color K1:
        fx = 634.8879216573664
        fy = 617.5755080760925
        cx = 318.8070728820338
        cy = 235.91048429182663

基本使用
--------
>>> from openpi.spatial.config import make_baseline_config
>>>
>>> cfg = make_baseline_config()
>>> print(cfg.visual.camera_roles)
('front', 'left')
>>> print(cfg.visual.num_points)
4096

切换为单相机：
>>> cfg = cfg.with_camera_roles("front")

切换为三相机：
>>> cfg = cfg.with_camera_roles("front", "left", "right")

将视觉采样改为 8192 点：
>>> cfg = cfg.with_visual_sampling(num_points=8192)

只为诊断实验启用 FRONT 外参修正：
>>> cfg = cfg.with_front_diagnostics(extrinsic=True)

注意
----
这里的“可插拔”首先通过稳定接口 + config-driven 实现，而不是一开始
搭建复杂 plugin framework。等项目真正出现第二种 sampler / 第二种
geometry backend 后，再在实现层增加 registry，会更干净。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path


# =============================================================================
# 1. 工作空间 ROI
# =============================================================================

@dataclass(frozen=True, slots=True)
class WorkspaceROI:
    """
    base_link 坐标系中的轴对齐工作空间 ROI，单位为米。

    该 ROI 决定哪些 3D 点允许进入后续 voxel / sampling 阶段。
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        if self.x_min_m >= self.x_max_m:
            raise ValueError("ROI requires x_min_m < x_max_m")
        if self.y_min_m >= self.y_max_m:
            raise ValueError("ROI requires y_min_m < y_max_m")
        if self.z_min_m >= self.z_max_m:
            raise ValueError("ROI requires z_min_m < z_max_m")

    @property
    def bounds_m(self) -> tuple[tuple[float, float], ...]:
        """返回 ((xmin,xmax), (ymin,ymax), (zmin,zmax))。"""
        return (
            (self.x_min_m, self.x_max_m),
            (self.y_min_m, self.y_max_m),
            (self.z_min_m, self.z_max_m),
        )


# =============================================================================
# 2. Visual preprocessing 配置
# =============================================================================

@dataclass(frozen=True, slots=True)
class VisualPreprocessConfig:
    """
    RGB-D -> visual point cloud 的实验配置。

    camera_roles
        参与融合的相机角色。
        geometry 实现必须遍历该 tuple，因此不能在算法里硬编码 front / left。

    voxel_size_m
        policy preprocessing 的 voxel size。
        当前正式 baseline = 5 mm。

    num_points
        voxel 后最终保留的视觉点数。
        当前 baseline = 4096。

    sampler
        sampling backend 的逻辑名称。
        当前 = "morton_stride"。
        后续若增加 FPS / random / learned sampler，只应增加对应实现，
        不应修改上层 preprocess 接口。

    min_depth_m / max_depth_m
        接受的 depth 范围。
    """

    camera_roles: tuple[str, ...] = ("front", "left")

    voxel_size_m: float = 0.005
    num_points: int = 4096
    sampler: str = "morton_stride"

    min_depth_m: float = 0.10
    max_depth_m: float = 2.50

    def __post_init__(self) -> None:
        if not self.camera_roles:
            raise ValueError("camera_roles must contain at least one camera")

        normalized = tuple(role.strip() for role in self.camera_roles)
        if any(not role for role in normalized):
            raise ValueError("camera_roles cannot contain empty names")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"camera_roles must be unique, got {self.camera_roles}"
            )

        if self.voxel_size_m <= 0:
            raise ValueError(
                f"voxel_size_m must be > 0, got {self.voxel_size_m}"
            )

        if self.num_points <= 0:
            raise ValueError(
                f"num_points must be > 0, got {self.num_points}"
            )

        if not self.sampler.strip():
            raise ValueError("sampler cannot be empty")

        if self.min_depth_m <= 0:
            raise ValueError(
                f"min_depth_m must be > 0, got {self.min_depth_m}"
            )

        if self.max_depth_m <= self.min_depth_m:
            raise ValueError(
                "max_depth_m must be greater than min_depth_m"
            )


# =============================================================================
# 3. Tactile 硬件与力轴映射配置
# =============================================================================

@dataclass(frozen=True, slots=True)
class TactileConfig:
    """
    当前 XHand tactile 空间布局与 force-axis mapping。

    这里描述“映射规则是什么”；
    真正的向量重排 / FK / rotation 由 tactile.py 实现。

    force_axis_order_by_finger
        使用 axis index 表示原始 force vector 如何重排。

        (0,1,2) 表示：
            [Fx,Fy,Fz] -> [Fx,Fy,Fz]

        (2,0,1) 表示：
            [Fx,Fy,Fz] -> [Fz,Fx,Fy]
    """

    finger_names: tuple[str, ...] = (
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    )

    finger_link_names: tuple[str, ...] = (
        "right_hand_thumb_rota_link2",
        "right_hand_index_rota_link2",
        "right_hand_mid_link2",
        "right_hand_ring_link2",
        "right_hand_pinky_link2",
    )

    taxels_per_finger: int = 120

    force_axis_order_by_finger: tuple[
        tuple[int, int, int],
        ...,
    ] = (
        (0, 1, 2),  # thumb / T30: identity
        (2, 0, 1),  # index / T16
        (2, 0, 1),  # middle / T16
        (2, 0, 1),  # ring / T16
        (2, 0, 1),  # pinky / T16
    )

    # 原始 observation.state 中 tactile block 的布局。
    tactile_block_size: int = 384
    raw_force_offset: int = 24

    def __post_init__(self) -> None:
        num_fingers = len(self.finger_names)

        if num_fingers <= 0:
            raise ValueError("finger_names cannot be empty")

        if len(set(self.finger_names)) != num_fingers:
            raise ValueError("finger_names must be unique")

        if len(self.finger_link_names) != num_fingers:
            raise ValueError(
                "finger_link_names must match finger_names length"
            )

        if len(self.force_axis_order_by_finger) != num_fingers:
            raise ValueError(
                "force_axis_order_by_finger must match finger_names length"
            )

        if self.taxels_per_finger <= 0:
            raise ValueError("taxels_per_finger must be > 0")

        if self.tactile_block_size <= 0:
            raise ValueError("tactile_block_size must be > 0")

        if self.raw_force_offset < 0:
            raise ValueError("raw_force_offset must be >= 0")

        for finger, order in zip(
            self.finger_names,
            self.force_axis_order_by_finger,
            strict=True,
        ):
            if tuple(sorted(order)) != (0, 1, 2):
                raise ValueError(
                    f"{finger} force axis order must be a permutation "
                    f"of (0,1,2), got {order}"
                )

    @property
    def num_fingers(self) -> int:
        """当前 tactile 手指数。"""
        return len(self.finger_names)

    @property
    def num_taxels(self) -> int:
        """当前 tactile 总 taxel 数。"""
        return self.num_fingers * self.taxels_per_finger


# =============================================================================
# 4. 正式 calibration / geometry asset 配置
# =============================================================================

@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """
    项目正式使用的 calibration / geometry asset 路径。

    所有路径均是 repo-relative path。

    相机 calibration 被刻意拆成两层：

    camera_config_path
        RGB-D 相机自身的内部标定与 stream 参数。
        当前来自旧项目的 cameras_0903.json，包含：
            - color_intrinsics
            - depth_intrinsics
            - depth_scale
            - T_color_depth
            - serial / role 等元信息

    camera_extrinsic_path
        相机相对于机器人 base_link 的外部标定。
        当前使用 extrinsic_0909.yaml，主要提供/覆盖：
            - T_base_color

    这个拆分非常重要：
        “RGB-D 内部几何”与“camera -> robot 外参”不是一回事，
        后续重新做手眼标定时，不应该连 depth/color 内部标定一起改。

    本类只保存路径，不负责：
        - 检查文件是否存在
        - 读取 JSON / YAML
        - 解析 URDF
        - 构造 transformation matrix

    这些 IO / 解析工作属于 calibration.py / tactile.py。
    """

    camera_config_path: Path = Path(
        "configs/ur7e_xhand/cameras_0903.json"
    )

    camera_extrinsic_path: Path = Path(
        "configs/ur7e_xhand/extrinsic_0909.yaml"
    )

    robot_urdf_path: Path = Path(
        "configs/ur7e_xhand/ur7e_xhand_verified.urdf"
    )

    tactile_geometry_urdf_path: Path = Path(
        "configs/ur7e_xhand/ur5_xhand_teacher_tactile.urdf"
    )

    state_mapping_path: Path = Path(
        "configs/ur7e_xhand/state_mapping.csv"
    )

    def __post_init__(self) -> None:
        for name, path in (
            ("camera_config_path", self.camera_config_path),
            ("camera_extrinsic_path", self.camera_extrinsic_path),
            ("robot_urdf_path", self.robot_urdf_path),
            (
                "tactile_geometry_urdf_path",
                self.tactile_geometry_urdf_path,
            ),
            ("state_mapping_path", self.state_mapping_path),
        ):
            if path.is_absolute():
                raise ValueError(
                    f"{name} must be repo-relative, got absolute path: {path}"
                )


# =============================================================================
# 5. Diagnostic-only calibration 配置
# =============================================================================

@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """最小 pinhole intrinsics 表达，用于 diagnostic override。"""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be > 0")


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    """
    只用于诊断 / ablation 的 calibration 开关。

    重要：
        默认必须全部关闭。
        这些参数当前没有进入正式 3D baseline。

    front_extrinsic_translation_base_m
        之前 2D / projection diagnostic 找到的 FRONT base-frame 平移：
            [0, -12, +5] mm

    front_color_k1
        之前 fair-K 实验得到的 FRONT color intrinsics candidate。
        它只影响 color projection，不应该改变 depth-derived 3D geometry。
    """

    enable_front_extrinsic_correction: bool = False
    enable_front_intrinsic_k1: bool = False

    front_extrinsic_translation_base_m: tuple[
        float,
        float,
        float,
    ] = (
        0.0,
        -0.012,
        0.005,
    )

    front_color_k1: PinholeIntrinsics = PinholeIntrinsics(
        fx=634.8879216573664,
        fy=617.5755080760925,
        cx=318.8070728820338,
        cy=235.91048429182663,
    )


# =============================================================================
# 6. 总配置：完整 Spatial preprocessing 实验定义
# =============================================================================

@dataclass(frozen=True, slots=True)
class SpatialPreprocessConfig:
    """
    完整空间预处理配置。

    这是 preprocess.py 未来主要接收的配置对象：

        cfg = make_baseline_config()
        preprocessor = SpatialPreprocessor(cfg)

    为方便常见 ablation，本类提供少量 immutable helper。
    每个 helper 都返回新的 config，不原地修改旧 config。
    """

    roi: WorkspaceROI
    visual: VisualPreprocessConfig
    tactile: TactileConfig
    calibration: CalibrationConfig
    diagnostics: DiagnosticConfig

    # -------------------------------------------------------------------------
    # 6.1 相机数量 / 组合切换
    # -------------------------------------------------------------------------
    def with_camera_roles(
        self,
        *camera_roles: str,
    ) -> SpatialPreprocessConfig:
        """
        返回仅修改 camera_roles 的新配置。

        示例
        ----
        单相机：
            cfg.with_camera_roles("front")

        三相机：
            cfg.with_camera_roles("front", "left", "right")
        """
        new_visual = replace(
            self.visual,
            camera_roles=tuple(camera_roles),
        )
        return replace(
            self,
            visual=new_visual,
        )

    # -------------------------------------------------------------------------
    # 6.2 Visual sampling ablation
    # -------------------------------------------------------------------------
    def with_visual_sampling(
        self,
        *,
        num_points: int | None = None,
        voxel_size_m: float | None = None,
        sampler: str | None = None,
    ) -> SpatialPreprocessConfig:
        """
        返回仅修改 visual sampling 参数的新配置。

        未提供的参数保持原值。

        示例
        ----
        8192 点：
            cfg.with_visual_sampling(num_points=8192)

        3 mm voxel + 8192 点：
            cfg.with_visual_sampling(
                voxel_size_m=0.003,
                num_points=8192,
            )
        """
        new_visual = replace(
            self.visual,
            num_points=(
                self.visual.num_points
                if num_points is None
                else num_points
            ),
            voxel_size_m=(
                self.visual.voxel_size_m
                if voxel_size_m is None
                else voxel_size_m
            ),
            sampler=(
                self.visual.sampler
                if sampler is None
                else sampler
            ),
        )
        return replace(
            self,
            visual=new_visual,
        )

    # -------------------------------------------------------------------------
    # 6.3 Diagnostic 开关
    # -------------------------------------------------------------------------
    def with_front_diagnostics(
        self,
        *,
        extrinsic: bool | None = None,
        intrinsic: bool | None = None,
    ) -> SpatialPreprocessConfig:
        """
        返回仅修改 FRONT diagnostic 开关的新配置。

        示例
        ----
        只开外参 diagnostic：
            cfg.with_front_diagnostics(extrinsic=True)

        两个都开：
            cfg.with_front_diagnostics(
                extrinsic=True,
                intrinsic=True,
            )
        """
        new_diagnostics = replace(
            self.diagnostics,
            enable_front_extrinsic_correction=(
                self.diagnostics.enable_front_extrinsic_correction
                if extrinsic is None
                else extrinsic
            ),
            enable_front_intrinsic_k1=(
                self.diagnostics.enable_front_intrinsic_k1
                if intrinsic is None
                else intrinsic
            ),
        )

        return replace(
            self,
            diagnostics=new_diagnostics,
        )


# =============================================================================
# 7. 当前正式 baseline factory
# =============================================================================

def make_baseline_config() -> SpatialPreprocessConfig:
    """
    构造当前已经冻结的正式 spatial preprocessing baseline。

    这里集中保存“当前 baseline 到底是什么”，而不是让这些 magic numbers
    散落到 geometry.py / tactile.py / scripts 中。

    当前 baseline
    ---------------
    camera:
        front + left

    ROI:
        来自已人工验收的 workspace_bounds。

    visual:
        5 mm voxel
        Morton stride
        4096 points

    tactile:
        5 × 120 = 600 taxels
        T30 / T16 vendor axis mapping

    diagnostics:
        全部关闭
    """
    return SpatialPreprocessConfig(
        roi=WorkspaceROI(
            x_min_m=-0.194484675325666,
            x_max_m=0.904719882815225,
            y_min_m=-0.141058700425284,
            y_max_m=0.765338105814797,
            z_min_m=0.0295315698214935,
            z_max_m=0.641479513645171,
        ),
        visual=VisualPreprocessConfig(
            camera_roles=("front", "left"),
            voxel_size_m=0.005,
            num_points=4096,
            sampler="morton_stride",
            min_depth_m=0.10,
            max_depth_m=2.50,
        ),
        tactile=TactileConfig(),
        calibration=CalibrationConfig(),
        diagnostics=DiagnosticConfig(
            enable_front_extrinsic_correction=False,
            enable_front_intrinsic_k1=False,
        ),
    )
