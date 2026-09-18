"""
OpenPI UR7e + XHand：机器人输入 / 输出 transform
（openpi.policies.xhand_policy）

作用
----
将 UR7e + XHand 数据集 / 部署端 observation 转换成 OpenPI
``Observation`` 所需要的 canonical image/state/action 结构。

当前数据 contract
-----------------
原始 LeRobot 数据：

    observation.state:
        [1972]

    action:
        [18]

    observation.images.cam_front:
        RGB

    observation.images.cam_left:
        RGB

    observation.images.cam_right:
        RGB

1972 维 state 中包含：

    0:6
        UR7e 6 个实际关节位置

    6:12
        UR7e 关节速度

    12:28
        EE 4x4 pose

    28:52
        12 个 XHand 关节 position / torque 交错

    52:
        tactile / temperature / raw force ...

模型 proprio state
------------------
spatial/v1 已经把 tactile 独立表示成：

    tactile_xyz_m
    tactile_force_base
    tactile_force_norm
    finger_id
    taxel_id

因此低维 proprio state 不再重复携带 raw tactile。

当前模型 state 固定为：

    UR7e 6 joint positions
        +
    XHand 12 joint positions

共：

    18 dims

具体索引：

    arm:
        state[0:6]

    hand:
        state[28:52:2]

该索引已经与当前数据集 meta/info.json 和
configs/ur7e_xhand/state_mapping.csv 交叉核对。

图像映射
--------
OpenPI π0 当前保留三个固定 image slots：

    base_0_rgb
    left_wrist_0_rgb
    right_wrist_0_rgb

对于当前三路相机：

    cam_front -> base_0_rgb
    cam_left  -> left_wrist_0_rgb
    cam_right -> right_wrist_0_rgb

这些名称是 OpenPI 的模型 slot 名称；
并不表示 cam_left / cam_right 必须物理安装在 wrist。

动作
----
数据集 action 为：

    6 个 UR7e joint target positions
        +
    12 个 XHand joint target positions

共 18 维。

当前数据集 action 为 18-D 绝对关节目标位姿。

baseline 直接使用 absolute action，不做 DeltaActions。
如果未来要研究 delta action representation，应作为显式消融配置，
而不是改变数据 contract。

输出 transform 会从 π0 padded 32-D action 中取回前 18 维。

Spatial sidecar
---------------
本文件不读取、不修改 spatial 数据。

SpatialAugmentedDataset 将 spatial/v1 join 到 sample 后，
data_loader 会在普通 OpenPI transforms 期间暂存 ``spatial``，
完成本 transform 后再恢复。

这样 xhand_policy.py 不依赖任何具体 spatial encoder。

基本使用
--------
训练：

    RepackTransform
        ↓
    XHandInputs
        ↓
    Normalize
        ↓
    ModelTransformFactory
        ↓
    Observation.from_dict

推理：

    robot observation
        ↓
    XHandInputs
        ↓
    model

注意
----
- 原始 1972-D state 长度不足 52 时直接报错。
- RGB 支持 uint8 HWC 和 float CHW/HWC。
- 不在这里读取 depth；depth 已经用于离线生成 spatial/v1。
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# =============================================================================
# 1. Frozen robot-state contract
# =============================================================================

ARM_POSITION_SLICE = slice(
    0,
    6,
)

HAND_POSITION_START = 28
HAND_POSITION_STOP = 52
HAND_POSITION_STEP = 2

ROBOT_STATE_DIM = 18
ROBOT_ACTION_DIM = 18


def extract_joint_state(
    state: np.ndarray,
) -> np.ndarray:
    """
    从 1972-D sensor state 中提取 18-D robot proprio state。

    支持：
        [1972]
        [..., 1972]
    """
    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape[
        -1
    ] < HAND_POSITION_STOP:
        raise ValueError(
            "UR7e+XHand observation.state too short: "
            f"got last dimension {state.shape[-1]}, "
            f"need at least {HAND_POSITION_STOP}."
        )

    arm = state[
        ...,
        ARM_POSITION_SLICE,
    ]

    hand = state[
        ...,
        HAND_POSITION_START:
        HAND_POSITION_STOP:
        HAND_POSITION_STEP,
    ]

    output = np.concatenate(
        [
            arm,
            hand,
        ],
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )

    if output.shape[
        -1
    ] != ROBOT_STATE_DIM:
        raise RuntimeError(
            "Unexpected XHand joint-state dimension: "
            f"{output.shape[-1]}"
        )

    return output


# =============================================================================
# 2. Image helper
# =============================================================================

def _parse_image(
    image,
) -> np.ndarray:
    """
    统一成 uint8 HWC。

    LeRobot 读取视频时常见：
        float32 CHW in [0,1]

    部署端常见：
        uint8 HWC
    """
    image = np.asarray(
        image
    )

    if (
        image.ndim
        != 3
    ):
        raise ValueError(
            "XHand image must be rank-3, "
            f"got shape {image.shape}"
        )

    if np.issubdtype(
        image.dtype,
        np.floating,
    ):
        # 兼容：
        #   [0,1] float
        #   [0,255] float
        maximum = float(
            np.max(
                image
            )
        ) if image.size else 0.0

        if maximum <= 1.0 + 1e-6:
            image = (
                image
                * 255.0
            )

        image = np.clip(
            image,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

    elif image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8
        )

    # CHW -> HWC
    if (
        image.shape[
            0
        ]
        == 3
        and image.shape[
            -1
        ]
        != 3
    ):
        image = einops.rearrange(
            image,
            "c h w -> h w c",
        )

    if image.shape[
        -1
    ] != 3:
        raise ValueError(
            "XHand RGB image must have 3 channels, "
            f"got shape {image.shape}"
        )

    return image


# =============================================================================
# 3. Input transform
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class XHandInputs(
    transforms.DataTransformFn
):
    """
    UR7e + XHand -> canonical OpenPI model input。
    """

    model_type: _model.ModelType

    def __call__(
        self,
        data: dict,
    ) -> dict:
        images = data[
            "images"
        ]

        front = _parse_image(
            images[
                "cam_front"
            ]
        )

        left = _parse_image(
            images[
                "cam_left"
            ]
        )

        right = _parse_image(
            images[
                "cam_right"
            ]
        )

        state = extract_joint_state(
            data[
                "state"
            ]
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": (
                    front
                ),
                "left_wrist_0_rgb": (
                    left
                ),
                "right_wrist_0_rgb": (
                    right
                ),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(
                data[
                    "actions"
                ],
                dtype=np.float32,
            )

            if actions.shape[
                -1
            ] != ROBOT_ACTION_DIM:
                raise ValueError(
                    "XHand action dimension mismatch: "
                    f"got {actions.shape[-1]}, "
                    f"expected {ROBOT_ACTION_DIM}."
                )

            inputs[
                "actions"
            ] = actions

        if "prompt" in data:
            inputs[
                "prompt"
            ] = data[
                "prompt"
            ]

        return inputs


# =============================================================================
# 4. Output transform
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class XHandOutputs(
    transforms.DataTransformFn
):
    """
    从 π0 padded action 中取回 UR7e + XHand 的 18 维 action。
    """

    def __call__(
        self,
        data: dict,
    ) -> dict:
        actions = np.asarray(
            data[
                "actions"
            ]
        )

        return {
            "actions": actions[
                ...,
                :ROBOT_ACTION_DIM,
            ]
        }


# =============================================================================
# 5. Minimal inference / transform example
# =============================================================================

def make_xhand_example() -> dict:
    """
    创建不含 spatial sidecar 的最小 raw observation 示例。

    spatial 在真实流程中由独立模块提供。
    """
    return {
        "images": {
            "cam_front": np.zeros(
                (
                    480,
                    640,
                    3,
                ),
                dtype=np.uint8,
            ),
            "cam_left": np.zeros(
                (
                    480,
                    640,
                    3,
                ),
                dtype=np.uint8,
            ),
            "cam_right": np.zeros(
                (
                    480,
                    640,
                    3,
                ),
                dtype=np.uint8,
            ),
        },
        "state": np.zeros(
            1972,
            dtype=np.float32,
        ),
        "prompt": (
            "press the button 4 times "
            "and put it into the box"
        ),
    }
