"""
OpenPI 3D + tactile 扩展：统一空间预处理入口（preprocess.py）

作用
----
本模块把前面已经分别验证通过的 visual / tactile 子链真正合并成一个
“离线训练和在线部署共用”的 canonical spatial preprocessor。

完整输入：

    RGB-D
    tactile raw state
    robot joint state

完整输出：

    SpatialObservation
        visual_xyz_m         [Nv,3]
        visual_rgb           [Nv,3]
        visual_rgb_valid     [Nv]
        tactile_xyz_m        [600,3]
        tactile_force_base   [600,3]
        tactile_force_norm   [600]
        finger_id            [600]
        taxel_id             [600]

其中 baseline：
    Nv = 4096

核心原则
--------
1. 离线 dataset conversion 和在线 deployment 必须调用同一个 preprocess()。
2. 本模块不读取具体 dataset 文件，不依赖 LeRobot / ROS bag / 某个采集格式。
3. calibration / FK / taxel geometry 都在初始化阶段加载并冻结。
4. 单帧 preprocess 只消费“已经解码好的 raw observation”。
5. model / training 只看到 SpatialObservation，不知道 ArUco、URDF、相机标定细节。

架构
----
静态资产（初始化一次）：

    camera calibration
    verified robot URDF
    state_mapping.csv
    official transformed taxel geometry
        ↓
    SpatialPreprocessor.from_repo_root(...)
        ↓
    SpatialPreprocessor

每帧动态输入：

    frame_index
    timestamp_s
    observation.state
    depth_by_role
    rgb_by_role
        ↓
    preprocess(...)
        ↓
    visual:
        calibration + RGB-D
        -> geometry.py
        -> 4096 visual points

    tactile:
        state + mapping + FK
        -> T_base_link2
        + official taxel geometry
        -> tactile.py
        -> 600 tactile points

        ↓
    SpatialObservation

为什么这里不读取 Dataset
-----------------------
如果 preprocess.py 自己 import 旧 Dataset：

    offline:
        能用

    online:
        无法复用

就又会出现两套 preprocessing。

因此：
    dataset adapter 负责“把磁盘数据解码成 raw arrays”
    deployment adapter 负责“从传感器拿 raw arrays”

但二者最后都调用同一个：

    SpatialPreprocessor.preprocess(...)

这就是 train / deploy preprocessing 不分叉的关键边界。

基本使用
--------
>>> from pathlib import Path
>>> from openpi.spatial.config import make_baseline_config
>>> from openpi.spatial.preprocess import SpatialPreprocessor
>>>
>>> cfg = make_baseline_config()
>>>
>>> preprocessor = SpatialPreprocessor.from_repo_root(
...     repo_root=Path("."),
...     config=cfg,
... )
>>>
>>> observation = preprocessor.preprocess(
...     frame_index=213,
...     timestamp_s=0.0,
...     state=state,
...     depth_by_role={
...         "front": front_depth,
...         "left": left_depth,
...     },
...     rgb_by_role={
...         "front": front_rgb,
...         "left": left_rgb,
...     },
... )
>>>
>>> observation.visual_xyz_m.shape
(4096, 3)
>>> observation.tactile_xyz_m.shape
(600, 3)

默认 tactile geometry 资产
--------------------------
为了不把机器绝对路径写入 core code，默认使用 repo-relative：

    configs/ur7e_xhand/tactile/points_t16_transformed.json
    configs/ur7e_xhand/tactile/points_t30_right_hand_transformed.json

如果以后更换 tactile geometry，可以在 from_repo_root() 显式传入新路径。

注意
----
- 本模块假设前面已经通过 calibration / geometry / FK / tactile parity。
- 不在这里重新做任何 diagnostic 修正。
- DiagnosticConfig 的开关仍由 config 显式控制，默认全部关闭。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from openpi.spatial.calibration import CalibrationBundle
from openpi.spatial.calibration import load_camera_calibrations
from openpi.spatial.config import SpatialPreprocessConfig
from openpi.spatial.geometry import build_visual_geometry
from openpi.spatial.kinematics import RobotKinematics
from openpi.spatial.kinematics import load_state_mapping_csv
from openpi.spatial.kinematics import tactile_link_transforms
from openpi.spatial.schema import SpatialObservation
from openpi.spatial.tactile import build_tactile_observation
from openpi.spatial.tactile_geometry import TaxelGeometryBundle
from openpi.spatial.tactile_geometry import load_xhand_taxel_geometry


# =============================================================================
# 1. Repo-relative 默认 tactile geometry
# =============================================================================

DEFAULT_T16_TRANSFORMED_PATH = Path(
    "configs/ur7e_xhand/tactile/points_t16_transformed.json"
)

DEFAULT_T30_RIGHT_TRANSFORMED_PATH = Path(
    "configs/ur7e_xhand/tactile/points_t30_right_hand_transformed.json"
)


# =============================================================================
# 2. SpatialPreprocessor
# =============================================================================

@dataclass(slots=True)
class SpatialPreprocessor:
    """
    离线 / 在线共用的 canonical spatial preprocessor。

    config:
        所有可切换 preprocessing 参数。

    calibration_bundle:
        已加载并标准化的 camera calibration。

    taxel_geometry:
        官方 transformed tactile local geometry。

    kinematics:
        verified URDF 对应的 FK engine。

    state_mapping:
        URDF joint name -> observation.state index。
    """

    config: SpatialPreprocessConfig
    calibration_bundle: CalibrationBundle
    taxel_geometry: TaxelGeometryBundle
    kinematics: RobotKinematics
    state_mapping: dict[str, int]

    def __post_init__(self) -> None:
        self._validate_static_contract()

    # =========================================================================
    # 2.1 Factory：从 repo-relative 资产构造一次
    # =========================================================================

    @classmethod
    def from_repo_root(
        cls,
        *,
        repo_root: Path,
        config: SpatialPreprocessConfig,
        t16_transformed_path: Path = DEFAULT_T16_TRANSFORMED_PATH,
        t30_right_transformed_path: Path = DEFAULT_T30_RIGHT_TRANSFORMED_PATH,
    ) -> "SpatialPreprocessor":
        """
        从当前 repo 的静态资产创建 preprocessor。

        这一步通常只做一次：
            offline conversion：每个 worker / process 一次
            online deployment：程序启动一次

        不应该每 frame 重复：
            parse URDF
            read calibration
            read taxel JSON
            read mapping CSV
        """
        root = Path(
            repo_root
        ).expanduser().resolve()

        calibration_bundle = (
            load_camera_calibrations(
                repo_root=root,
                calibration_config=config.calibration,
                camera_roles=config.visual.camera_roles,
            )
        )

        t16_path = _resolve_repo_relative(
            root,
            t16_transformed_path,
        )

        t30_path = _resolve_repo_relative(
            root,
            t30_right_transformed_path,
        )

        taxel_geometry = (
            load_xhand_taxel_geometry(
                t16_transformed_path=t16_path,
                t30_right_transformed_path=t30_path,
                tactile_config=config.tactile,
            )
        )

        robot_urdf_path = (
            _resolve_repo_relative(
                root,
                config.calibration.robot_urdf_path,
            )
        )

        state_mapping_path = (
            _resolve_repo_relative(
                root,
                config.calibration.state_mapping_path,
            )
        )

        kinematics = RobotKinematics(
            robot_urdf_path
        )

        state_mapping = (
            load_state_mapping_csv(
                state_mapping_path
            )
        )

        return cls(
            config=config,
            calibration_bundle=calibration_bundle,
            taxel_geometry=taxel_geometry,
            kinematics=kinematics,
            state_mapping=state_mapping,
        )

    # =========================================================================
    # 2.2 Canonical per-frame API
    # =========================================================================

    def preprocess(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        state: np.ndarray,
        depth_by_role: Mapping[str, np.ndarray],
        rgb_by_role: Mapping[str, np.ndarray],
    ) -> SpatialObservation:
        """
        把单帧 raw observation 转成 SpatialObservation。

        参数
        ----
        frame_index:
            原始数据中的 frame id。

        timestamp_s:
            当前帧 timestamp，单位秒。
            如果某个 dataset 暂时没有可靠 timestamp，
            adapter 应显式给出它自己的定义，而不是让本函数猜。

        state:
            observation.state，一维数组。

        depth_by_role:
            {
                "front": depth_raw,
                "left": depth_raw,
                ...
            }

        rgb_by_role:
            {
                "front": rgb_uint8,
                "left": rgb_uint8,
                ...
            }

        返回
        ----
        SpatialObservation

        该对象就是未来：
            offline derived dataset
            online policy input adapter
        共用的物理空间表示。
        """
        self._validate_dynamic_input(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            state=state,
            depth_by_role=depth_by_role,
            rgb_by_role=rgb_by_role,
        )

        # ---------------------------------------------------------------------
        # 2.2.1 Visual branch
        #
        # RGB-D -> base_link visual point cloud
        # -> ROI -> voxel -> fixed-N sampling -> RGB association
        # ---------------------------------------------------------------------
        visual = build_visual_geometry(
            depth_by_role=dict(
                depth_by_role
            ),
            rgb_by_role=dict(
                rgb_by_role
            ),
            calibrations=(
                self.calibration_bundle.cameras
            ),
            config=self.config.visual,
            roi=self.config.roi,
            diagnostics=(
                self.config.diagnostics
            ),
        )

        # ---------------------------------------------------------------------
        # 2.2.2 Kinematics branch
        #
        # state -> q -> verified FK -> five T_base_link2
        # ---------------------------------------------------------------------
        tactile_transforms = (
            tactile_link_transforms(
                np.asarray(
                    state
                ),
                mapping=self.state_mapping,
                kinematics=self.kinematics,
                finger_link_names=(
                    self.config.tactile.finger_link_names
                ),
                root="base_link",
            )
        )

        # ---------------------------------------------------------------------
        # 2.2.3 Tactile branch
        #
        # raw force + local taxel geometry + current FK
        # -> 600 base_link tactile points / force vectors
        # ---------------------------------------------------------------------
        tactile = (
            build_tactile_observation(
                state=np.asarray(
                    state
                ),
                taxel_xyz_link_m=(
                    self.taxel_geometry.xyz_link_m
                ),
                T_base_link_by_name=(
                    tactile_transforms
                ),
                config=self.config.tactile,
            )
        )

        # ---------------------------------------------------------------------
        # 2.2.4 Canonical schema
        #
        # 注意：
        # VisualGeometryResult / TactileGeometryResult 都是 preprocessing
        # 内部对象；从这里开始，上层只看到 SpatialObservation。
        # ---------------------------------------------------------------------
        observation = SpatialObservation(
            frame_index=int(
                frame_index
            ),
            timestamp_s=float(
                timestamp_s
            ),
            visual_xyz_m=np.asarray(
                visual.xyz_m,
                dtype=np.float32,
            ),
            visual_rgb=np.asarray(
                visual.rgb,
                dtype=np.uint8,
            ),
            visual_rgb_valid=np.asarray(
                visual.rgb_valid,
                dtype=np.bool_,
            ),
            tactile_xyz_m=np.asarray(
                tactile.xyz_m,
                dtype=np.float32,
            ),
            tactile_force_base=np.asarray(
                tactile.force_base,
                dtype=np.float32,
            ),
            tactile_force_norm=np.asarray(
                tactile.force_norm,
                dtype=np.float32,
            ),
            finger_id=np.asarray(
                tactile.finger_id,
                dtype=np.int8,
            ),
            taxel_id=np.asarray(
                tactile.taxel_id,
                dtype=np.int16,
            ),
        )

        # ---------------------------------------------------------------------
        # 2.2.5 在系统边界再次检查最终 contract
        # ---------------------------------------------------------------------
        observation.validate(
            expected_visual_points=(
                self.config.visual.num_points
            ),
            expected_tactile_points=(
                self.config.tactile.num_taxels
            ),
        )

        return observation

    # =========================================================================
    # 2.3 Static contract
    # =========================================================================

    def _validate_static_contract(
        self,
    ) -> None:
        """
        检查初始化阶段的模块接缝。

        这些错误如果存在，应该在程序启动时暴露，
        而不是跑到第几千帧才发现。
        """
        expected_roles = tuple(
            self.config.visual.camera_roles
        )

        loaded_roles = tuple(
            self.calibration_bundle.cameras.keys()
        )

        if set(
            loaded_roles
        ) != set(
            expected_roles
        ):
            raise ValueError(
                "Loaded camera calibration roles do not match config: "
                f"config={expected_roles}, loaded={loaded_roles}"
            )

        if tuple(
            self.taxel_geometry.finger_names
        ) != tuple(
            self.config.tactile.finger_names
        ):
            raise ValueError(
                "Taxel geometry finger order does not match TactileConfig: "
                f"geometry={self.taxel_geometry.finger_names}, "
                f"config={self.config.tactile.finger_names}"
            )

        required_joints = (
            self.kinematics.required_measured_joints(
                root="base_link"
            )
        )

        missing_mapping = sorted(
            required_joints
            - set(
                self.state_mapping
            )
        )

        if missing_mapping:
            raise KeyError(
                "state_mapping does not cover strict FK joints: "
                f"{missing_mapping}"
            )

        for link_name in (
            self.config.tactile.finger_link_names
        ):
            if link_name not in self.kinematics.links:
                raise KeyError(
                    "TactileConfig references link absent from URDF: "
                    f"{link_name!r}"
                )

    # =========================================================================
    # 2.4 Per-frame raw-input contract
    # =========================================================================

    def _validate_dynamic_input(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        state: np.ndarray,
        depth_by_role: Mapping[str, np.ndarray],
        rgb_by_role: Mapping[str, np.ndarray],
    ) -> None:
        """
        检查 raw frame 是否满足 canonical preprocessor 输入 contract。

        这里只检查结构，不重复执行 calibration / geometry 内部 validation。
        """
        if int(
            frame_index
        ) < 0:
            raise ValueError(
                f"frame_index must be >=0, got {frame_index}"
            )

        timestamp = float(
            timestamp_s
        )

        if not np.isfinite(
            timestamp
        ):
            raise ValueError(
                f"timestamp_s must be finite, got {timestamp_s}"
            )

        state_array = np.asarray(
            state
        )

        if state_array.ndim != 1:
            raise ValueError(
                f"state must be 1-D, got {state_array.shape}"
            )

        if not np.all(
            np.isfinite(
                state_array
            )
        ):
            raise ValueError(
                "state contains NaN / Inf"
            )

        required_roles = tuple(
            self.config.visual.camera_roles
        )

        missing_depth = [
            role
            for role in required_roles
            if role not in depth_by_role
        ]

        missing_rgb = [
            role
            for role in required_roles
            if role not in rgb_by_role
        ]

        if missing_depth:
            raise KeyError(
                f"Missing depth for camera roles: {missing_depth}"
            )

        if missing_rgb:
            raise KeyError(
                f"Missing RGB for camera roles: {missing_rgb}"
            )

        for role in required_roles:
            depth = np.asarray(
                depth_by_role[
                    role
                ]
            )

            rgb = np.asarray(
                rgb_by_role[
                    role
                ]
            )

            if depth.ndim != 2:
                raise ValueError(
                    f"{role}: depth must be [H,W], got {depth.shape}"
                )

            if (
                rgb.ndim != 3
                or rgb.shape[
                    2
                ] != 3
            ):
                raise ValueError(
                    f"{role}: RGB must be [H,W,3], got {rgb.shape}"
                )

            if not np.issubdtype(
                depth.dtype,
                np.number,
            ):
                raise TypeError(
                    f"{role}: depth dtype must be numeric, got {depth.dtype}"
                )

            if rgb.dtype != np.uint8:
                raise TypeError(
                    f"{role}: RGB dtype must be uint8, got {rgb.dtype}"
                )


# =============================================================================
# 3. Repo-relative path helper
# =============================================================================

def _resolve_repo_relative(
    repo_root: Path,
    relative_path: Path,
) -> Path:
    """
    解析 repo-relative asset path。

    Core code 中不允许硬编码用户机器的 /home/... 路径。
    """
    path = Path(
        relative_path
    )

    if path.is_absolute():
        raise ValueError(
            "Expected repo-relative asset path, got absolute path: "
            f"{path}"
        )

    return (
        repo_root
        / path
    ).resolve()
