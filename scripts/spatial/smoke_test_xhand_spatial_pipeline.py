"""
OpenPI 3D + tactile：UR7e + XHand spatial pipeline smoke test
（smoke_test_xhand_spatial_pipeline.py）

作用
----
在正式计算 normalization statistics / 开始训练之前，
验证一条真实数据链：

    LeRobot press_0828_17
        ↓
    action horizon loader
        ↓
    spatial/v1 join
        ↓
    XHand robot transform
        ↓
    OpenPI model transforms
        ↓
    Observation
        ↓
    JointPointNet spatial encoder
        ↓
    PREFIX / SUFFIX / BOTH router

本脚本不初始化完整 PaliGemma / π0，
因此比正式 model forward 轻很多。

它主要抓以下错误：

    - episode/frame 与 spatial/v1 错位
    - 1972-D state 提取索引错误
    - absolute action 被偷偷转成 delta
    - 18-D state/action padding 错误
    - spatial sidecar 在 Repack / Normalize / model transforms 中丢失
    - tactile/visual shape 错误
    - JointPointNet 输入 contract 错误
    - conditioning PREFIX / SUFFIX / BOTH 路由错误

默认检查
--------
config:
    pi0_xhand_spatial_joint_pointnet_suffix

sample:
    global index 213
    对应 episode 0 / frame 213

已知 frame 213 tactile identity：
    strongest finger_id = 1
    strongest taxel_id  = 55
    norm ≈ 51.2445

absolute action contract：
    transformed actions[..., :18]
    必须与 raw dataset action 完全一致。

基本使用
--------
    PYTHONPATH=src python scripts/spatial/smoke_test_xhand_spatial_pipeline.py

检查 PREFIX：

    ... --config-name pi0_xhand_spatial_joint_pointnet_prefix

检查 BOTH：

    ... --config-name pi0_xhand_spatial_joint_pointnet_both

注意
----
- 需要服务器上的完整 OpenPI/JAX 环境。
- 不要求已经计算 norm stats；本脚本使用 skip_norm_stats=True。
- 不加载 base checkpoint。
- 不初始化完整 π0，因此不能替代最后的 1-step train smoke test。
"""

from __future__ import annotations

import argparse
import dataclasses

from flax import nnx
import jax
import numpy as np

from openpi.models.spatial_encoders.router import (
    SpatialConditioningRouter,
)
from openpi.policies import xhand_policy
from openpi.training import config as _config
from openpi.training import data_loader


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the XHand + spatial/v1 "
            "OpenPI training input pipeline."
        )
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default=(
            "pi0_xhand_spatial_"
            "joint_pointnet_suffix"
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=213,
    )

    return parser.parse_args()


# =============================================================================
# 2. Helpers
# =============================================================================

def _as_numpy(
    value,
) -> np.ndarray:
    return np.asarray(
        value
    )


def _assert_allclose(
    actual,
    expected,
    *,
    name: str,
    atol: float = 1e-6,
) -> None:
    actual = _as_numpy(
        actual
    )

    expected = _as_numpy(
        expected
    )

    if not np.allclose(
        actual,
        expected,
        atol=atol,
        rtol=0.0,
    ):
        difference = float(
            np.max(
                np.abs(
                    actual
                    - expected
                )
            )
        )

        raise AssertionError(
            f"{name} mismatch; "
            f"max_abs_diff={difference}"
        )


# =============================================================================
# 3. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    train_config = _config.get_config(
        args.config_name
    )

    data_config = train_config.data.create(
        train_config.assets_dirs,
        train_config.model,
    )

    print(
        "===== CONFIG ====="
    )
    print(
        "name:",
        train_config.name,
    )
    print(
        "repo:",
        data_config.repo_id,
    )
    print(
        "action_horizon:",
        train_config.model.action_horizon,
    )
    print(
        "model action_dim:",
        train_config.model.action_dim,
    )
    print(
        "spatial:",
        data_config.spatial,
    )

    conditioning = getattr(
        train_config.model,
        "conditioning",
        None,
    )

    if conditioning is None:
        raise AssertionError(
            "Selected config is not a spatial model config."
        )

    print(
        "conditioning:",
        conditioning.target.value,
    )

    # -------------------------------------------------------------------------
    # 3.1 Raw LeRobot + spatial join
    # -------------------------------------------------------------------------
    raw_dataset = (
        data_loader.create_torch_dataset(
            data_config,
            train_config.model.action_horizon,
            train_config.model,
        )
    )

    raw = raw_dataset[
        args.sample_index
    ]

    print(
        "\n===== RAW + SPATIAL JOIN ====="
    )

    episode_index = int(
        _as_numpy(
            raw[
                "episode_index"
            ]
        ).reshape(
            ()
        )
    )

    frame_index = int(
        _as_numpy(
            raw[
                "frame_index"
            ]
        ).reshape(
            ()
        )
    )

    print(
        "episode/frame:",
        episode_index,
        frame_index,
    )

    if (
        args.sample_index == 213
        and (
            episode_index != 0
            or frame_index != 213
        )
    ):
        raise AssertionError(
            "Global index 213 is expected to map "
            "to episode 0 / frame 213."
        )

    spatial = raw[
        "spatial"
    ]

    visual = spatial[
        "visual"
    ]

    tactile = spatial[
        "tactile"
    ]

    print(
        "visual xyz:",
        visual[
            "xyz_m"
        ].shape,
        visual[
            "xyz_m"
        ].dtype,
    )

    print(
        "tactile xyz:",
        tactile[
            "xyz_m"
        ].shape,
        tactile[
            "xyz_m"
        ].dtype,
    )

    if visual[
        "xyz_m"
    ].shape != (
        4096,
        3,
    ):
        raise AssertionError(
            "Unexpected visual spatial shape."
        )

    if tactile[
        "xyz_m"
    ].shape != (
        600,
        3,
    ):
        raise AssertionError(
            "Unexpected tactile spatial shape."
        )

    strongest = int(
        np.argmax(
            tactile[
                "force_norm"
            ]
        )
    )

    print(
        "strongest tactile:",
        "finger=",
        int(
            tactile[
                "finger_id"
            ][
                strongest
            ]
        ),
        "taxel=",
        int(
            tactile[
                "taxel_id"
            ][
                strongest
            ]
        ),
        "norm=",
        float(
            tactile[
                "force_norm"
            ][
                strongest
            ]
        ),
    )

    if args.sample_index == 213:
        if int(
            tactile[
                "finger_id"
            ][
                strongest
            ]
        ) != 1:
            raise AssertionError(
                "Frame 213 strongest finger mismatch."
            )

        if int(
            tactile[
                "taxel_id"
            ][
                strongest
            ]
        ) != 55:
            raise AssertionError(
                "Frame 213 strongest taxel mismatch."
            )

    # -------------------------------------------------------------------------
    # 3.2 Full OpenPI transform chain, but without normalization
    # -------------------------------------------------------------------------
    transformed_dataset = (
        data_loader.transform_dataset(
            raw_dataset,
            data_config,
            skip_norm_stats=True,
        )
    )

    transformed = transformed_dataset[
        args.sample_index
    ]

    print(
        "\n===== TRANSFORMED SAMPLE ====="
    )

    print(
        "state:",
        transformed[
            "state"
        ].shape,
        transformed[
            "state"
        ].dtype,
    )

    print(
        "actions:",
        transformed[
            "actions"
        ].shape,
        transformed[
            "actions"
        ].dtype,
    )

    print(
        "tokenized prompt:",
        transformed[
            "tokenized_prompt"
        ].shape,
    )

    if transformed[
        "state"
    ].shape != (
        train_config.model.action_dim,
    ):
        raise AssertionError(
            "Padded state shape mismatch."
        )

    if transformed[
        "actions"
    ].shape != (
        train_config.model.action_horizon,
        train_config.model.action_dim,
    ):
        raise AssertionError(
            "Padded action shape mismatch."
        )

    # -------------------------------------------------------------------------
    # 3.3 State extraction must remain the known 18-D proprio
    # -------------------------------------------------------------------------
    expected_state = (
        xhand_policy.extract_joint_state(
            raw[
                "observation.state"
            ]
        )
    )

    _assert_allclose(
        transformed[
            "state"
        ][
            :18
        ],
        expected_state,
        name=(
            "18-D proprio state"
        ),
    )

    _assert_allclose(
        transformed[
            "state"
        ][
            18:
        ],
        np.zeros(
            train_config.model.action_dim
            - 18,
            dtype=np.float32,
        ),
        name=(
            "state zero padding"
        ),
    )

    print(
        "state contract:",
        "18-D proprio + zero pad PASS",
    )

    # -------------------------------------------------------------------------
    # 3.4 Absolute action must remain absolute
    # -------------------------------------------------------------------------
    raw_actions = _as_numpy(
        raw[
            "action"
        ]
    ).astype(
        np.float32
    )

    transformed_actions = _as_numpy(
        transformed[
            "actions"
        ]
    )

    _assert_allclose(
        transformed_actions[
            ...,
            :18
        ],
        raw_actions,
        name=(
            "absolute action preservation"
        ),
    )

    _assert_allclose(
        transformed_actions[
            ...,
            18:
        ],
        np.zeros(
            (
                transformed_actions.shape[
                    0
                ],
                train_config.model.action_dim
                - 18,
            ),
            dtype=np.float32,
        ),
        name=(
            "action zero padding"
        ),
    )

    print(
        "action contract:",
        "absolute 18-D + zero pad PASS",
    )

    # -------------------------------------------------------------------------
    # 3.5 spatial must bypass ordinary OpenPI normalization/transforms unchanged
    # -------------------------------------------------------------------------
    transformed_spatial = (
        transformed[
            "spatial"
        ]
    )

    _assert_allclose(
        transformed_spatial[
            "visual"
        ][
            "xyz_m"
        ],
        visual[
            "xyz_m"
        ],
        name=(
            "visual spatial preservation"
        ),
    )

    _assert_allclose(
        transformed_spatial[
            "tactile"
        ][
            "force_norm"
        ],
        tactile[
            "force_norm"
        ],
        name=(
            "tactile spatial preservation"
        ),
    )

    print(
        "spatial bypass:",
        "PASS",
    )

    # -------------------------------------------------------------------------
    # 3.6 Real batched Observation
    # -------------------------------------------------------------------------
    loader = (
        data_loader.create_torch_data_loader(
            data_config,
            model_config=(
                train_config.model
            ),
            action_horizon=(
                train_config.model.action_horizon
            ),
            batch_size=1,
            skip_norm_stats=True,
            shuffle=False,
            num_batches=1,
            num_workers=0,
            framework="jax",
        )
    )

    observation, batch_actions = next(
        iter(
            loader
        )
    )

    print(
        "\n===== BATCHED OBSERVATION ====="
    )

    print(
        "state:",
        observation.state.shape,
    )

    print(
        "actions:",
        batch_actions.shape,
    )

    if observation.spatial is None:
        raise AssertionError(
            "Observation.spatial was lost."
        )

    if observation.spatial.visual is not None:
        print(
            "visual:",
            observation.spatial.visual.xyz_m.shape,
        )

    if observation.spatial.tactile is not None:
        print(
            "tactile:",
            observation.spatial.tactile.xyz_m.shape,
        )

    # -------------------------------------------------------------------------
    # 3.7 Encoder + router without initializing full π0
    # -------------------------------------------------------------------------
    model_config = train_config.model

    encoder = model_config.create_spatial_encoder(
        rngs=nnx.Rngs(
            0
        )
    )

    # 使用小的测试 downstream width。
    # 这里只验证 router contract，不初始化 PaliGemma。
    router = SpatialConditioningRouter(
        encoder=encoder,
        encoder_token_dim=(
            model_config.spatial_encoder_token_dim
        ),
        conditioning=(
            model_config.conditioning
        ),
        prefix_dim=(
            64
            if model_config.conditioning.use_prefix
            else None
        ),
        suffix_dim=(
            48
            if model_config.conditioning.use_suffix
            else None
        ),
        rngs=nnx.Rngs(
            1
        ),
    )

    spatial_input = observation.spatial

    if not model_config.use_visual:
        spatial_input = dataclasses.replace(
            spatial_input,
            visual=None,
        )

    if not model_config.use_tactile:
        spatial_input = dataclasses.replace(
            spatial_input,
            tactile=None,
        )

    conditioned = router(
        spatial_input
    )

    print(
        "\n===== ENCODER / ROUTER ====="
    )

    if conditioned.has_prefix:
        print(
            "prefix tokens:",
            conditioned.prefix_tokens.shape,
        )

        if conditioned.prefix_tokens.shape != (
            1,
            1,
            64,
        ):
            raise AssertionError(
                "Unexpected routed prefix shape."
            )

    if conditioned.has_suffix:
        print(
            "suffix tokens:",
            conditioned.suffix_tokens.shape,
        )

        if conditioned.suffix_tokens.shape != (
            1,
            1,
            48,
        ):
            raise AssertionError(
                "Unexpected routed suffix shape."
            )

    print(
        "\nXHAND_SPATIAL_PIPELINE_PASS"
    )


if __name__ == "__main__":
    main()
