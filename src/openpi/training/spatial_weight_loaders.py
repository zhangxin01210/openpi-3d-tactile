"""
OpenPI 3D + tactile：Spatial-aware checkpoint weight loader
（openpi.training.spatial_weight_loaders）

作用
----
从普通预训练 π0 checkpoint 初始化带 spatial branch 的模型。

问题背景
--------
普通 π0 checkpoint 中没有：

    spatial_router/
        encoder/
        prefix_adapter/
        suffix_adapter/

如果直接使用 upstream CheckpointWeightLoader，
这些新参数不会自动从当前随机初始化模型中补回。

本 loader 的目标是：

    1. checkpoint 中已有的 base π0 参数
       -> 从 checkpoint 加载；

    2. checkpoint 中不存在的 spatial 参数
       -> 保留当前模型随机初始化值；

    3. 如果缺失的是其他非 spatial 参数
       -> 立即报错。

这样可以避免：

    “因为 checkpoint / model architecture 不匹配，
     某些普通参数也没加载，但训练仍悄悄继续”

的危险情况。

默认允许保留的 missing parameter
--------------------------------
    spatial_router/.*

以及：

    .*lora.*

后者与 upstream CheckpointWeightLoader 的兼容行为一致，
便于未来 LoRA spatial 实验。

为什么单独放一个文件
--------------------
不直接修改：

    openpi.training.weight_loaders.CheckpointWeightLoader

原因：

    - upstream 普通 π0 行为不应改变；
    - spatial 只是一个可选实验分支；
    - 后续 upstream merge 更容易。

训练配置中只需要：

    weight_loader=SpatialCheckpointWeightLoader(
        "gs://.../pi0_base/params"
    )

参数树 contract
---------------
输入：

    params
        当前模型随机初始化后的完整参数树。

加载：

    loaded_params
        base checkpoint 参数树。

最终输出：

    merged_params

结构必须与当前 ``params`` 完全一致。

因此训练器后续看到的是完整模型参数：

    base π0 weights
        +
    random initialized spatial weights

严格检查
--------
对于 checkpoint 与当前模型共有的 parameter：

    - key 必须匹配；
    - shape 必须匹配；
    - dtype 不同时转换成当前模型 dtype。

checkpoint 中多余、当前模型不存在的 key：

    忽略。

当前模型中缺失、且不满足 allowed missing regex 的 key：

    报错。

基本使用
--------
    from openpi.training.spatial_weight_loaders import (
        SpatialCheckpointWeightLoader,
    )

    loader = SpatialCheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    )

    params = loader.load(
        initialized_params
    )

可扩展
------
如果以后新增另一类“允许随机初始化”的模块：

    SpatialCheckpointWeightLoader(
        params_path=...,
        additional_missing_regexes=(
            r"another_module/.*",
        ),
    )

而不需要修改 loader 本身。

注意
----
- 本文件不负责 optimizer state。
- 本文件不负责 checkpoint resume。
- 本文件用于“预训练 base model -> 新 spatial architecture”的初始化。
- resume 自己训练出的 spatial checkpoint 时，应使用完整 checkpoint 恢复流程。
"""

from __future__ import annotations

import dataclasses
import re

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download


def _flatten_params(params: at.Params):
    return flax.traverse_util.flatten_dict(
        params,
    )


def _path_to_str(path) -> str:
    if isinstance(path, tuple):
        return "/".join(str(part) for part in path)
    return str(path)


# =============================================================================
# 1. Loader
# =============================================================================

@dataclasses.dataclass(
    frozen=True
)
class SpatialCheckpointAudit:
    loaded: tuple[str, ...]
    allowed_missing: tuple[str, ...]
    unexpected_missing: tuple[str, ...]
    unexpected_loaded: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


@dataclasses.dataclass(
    frozen=True
)
class SpatialCheckpointWeightLoader:
    """
    加载 base π0 checkpoint，并保留随机初始化的 spatial 参数。
    """

    params_path: str

    # 默认同时兼容：
    #   1. spatial branch
    #   2. upstream LoRA missing parameter
    additional_missing_regexes: tuple[
        str,
        ...,
    ] = ()

    @property
    def allowed_missing_regexes(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        return (
            r"spatial_router/.*",
            r".*lora.*",
            *self.additional_missing_regexes,
        )

    def load(
        self,
        params: at.Params,
    ) -> at.Params:
        """
        将 base checkpoint 合并到当前完整模型参数树。
        """
        loaded_params = _model.restore_params(
            download.maybe_download(
                self.params_path
            ),
            restore_type=np.ndarray,
        )

        return _merge_spatial_checkpoint(
            loaded_params=loaded_params,
            reference_params=params,
            allowed_missing_regexes=(
                self.allowed_missing_regexes
            ),
        )

    def audit(
        self,
        params: at.Params,
    ) -> SpatialCheckpointAudit:
        loaded_params = _model.restore_params(
            download.maybe_download(
                self.params_path
            ),
            restore_type=np.ndarray,
        )

        return audit_spatial_checkpoint(
            loaded_params=loaded_params,
            reference_params=params,
            allowed_missing_regexes=(
                self.allowed_missing_regexes
            ),
        )


# =============================================================================
# 2. Merge
# =============================================================================

def _merge_spatial_checkpoint(
    *,
    loaded_params: at.Params,
    reference_params: at.Params,
    allowed_missing_regexes: tuple[
        str,
        ...,
    ],
) -> at.Params:
    """
    严格合并 checkpoint 与当前随机初始化参数。

    reference_params：
        当前模型完整结构，是最终输出结构的唯一标准。
    """
    flat_reference = _flatten_params(
        reference_params
    )

    flat_loaded = _flatten_params(
        loaded_params
    )

    patterns = tuple(
        re.compile(
            pattern
        )
        for pattern
        in allowed_missing_regexes
    )

    result = {}

    # -------------------------------------------------------------------------
    # 2.1 checkpoint 中已有且当前模型仍需要的参数
    # -------------------------------------------------------------------------
    for key, loaded_value in flat_loaded.items():
        reference_value = (
            flat_reference.get(
                key
            )
        )

        # checkpoint 可能包含当前 architecture 已经不用的参数。
        # 与 upstream loader 一样，忽略这些 extra key。
        if reference_value is None:
            continue

        loaded_shape = getattr(
            loaded_value,
            "shape",
            None,
        )

        reference_shape = getattr(
            reference_value,
            "shape",
            None,
        )

        if (
            loaded_shape is not None
            and reference_shape is not None
            and tuple(
                loaded_shape
            )
            != tuple(
                reference_shape
            )
        ):
            raise ValueError(
                "Checkpoint parameter shape mismatch for "
                f"{_path_to_str(key)!r}: loaded={loaded_shape}, "
                f"current={reference_shape}"
            )

        loaded_dtype = getattr(
            loaded_value,
            "dtype",
            None,
        )

        reference_dtype = getattr(
            reference_value,
            "dtype",
            None,
        )

        if (
            loaded_dtype is not None
            and reference_dtype is not None
            and loaded_dtype
            != reference_dtype
        ):
            loaded_value = (
                loaded_value.astype(
                    reference_dtype
                )
            )

        result[
            key
        ] = loaded_value

    # -------------------------------------------------------------------------
    # 2.2 找出当前 architecture 中 checkpoint 缺失的参数
    # -------------------------------------------------------------------------
    missing_keys = sorted(
        set(
            flat_reference
        )
        - set(
            result
        )
    )

    unexpected_missing = []

    for key in missing_keys:
        allowed = any(
            pattern.fullmatch(
                _path_to_str(key)
            )
            is not None
            for pattern
            in patterns
        )

        if allowed:
            # 使用当前模型随机初始化值。
            result[
                key
            ] = flat_reference[
                key
            ]

        else:
            unexpected_missing.append(
                key
            )

    if unexpected_missing:
        preview = "\n".join(
            f"  - {_path_to_str(key)}"
            for key
            in unexpected_missing[
                :30
            ]
        )

        more = (
            ""
            if len(
                unexpected_missing
            )
            <= 30
            else (
                "\n  ... and "
                f"{len(unexpected_missing) - 30} more"
            )
        )

        raise ValueError(
            "Base checkpoint is missing parameters that are "
            "not declared as newly initialized modules.\n"
            "Unexpected missing keys:\n"
            f"{preview}"
            f"{more}\n"
            "Allowed missing regexes: "
            f"{allowed_missing_regexes}"
        )

    # -------------------------------------------------------------------------
    # 2.3 最后检查输出结构必须与当前模型完全一致
    # -------------------------------------------------------------------------
    final_keys = set(
        result
    )

    reference_keys = set(
        flat_reference
    )

    if final_keys != reference_keys:
        missing_after_merge = sorted(
            reference_keys
            - final_keys
        )

        extra_after_merge = sorted(
            final_keys
            - reference_keys
        )

        raise RuntimeError(
            "Merged parameter tree does not match "
            "the current model structure. "
            f"missing={missing_after_merge[:20]}, "
            f"extra={extra_after_merge[:20]}"
        )

    return flax.traverse_util.unflatten_dict(
        result,
    )


def audit_spatial_checkpoint(
    *,
    loaded_params: at.Params,
    reference_params: at.Params,
    allowed_missing_regexes: tuple[str, ...],
) -> SpatialCheckpointAudit:
    flat_reference = _flatten_params(reference_params)
    flat_loaded = _flatten_params(loaded_params)
    patterns = tuple(re.compile(pattern) for pattern in allowed_missing_regexes)

    loaded = []
    shape_mismatches = []

    for key, loaded_value in flat_loaded.items():
        reference_value = flat_reference.get(key)
        if reference_value is None:
            continue

        loaded_shape = getattr(loaded_value, "shape", None)
        reference_shape = getattr(reference_value, "shape", None)
        if (
            loaded_shape is not None
            and reference_shape is not None
            and tuple(loaded_shape) != tuple(reference_shape)
        ):
            shape_mismatches.append(
                f"{_path_to_str(key)}: loaded={tuple(loaded_shape)} current={tuple(reference_shape)}"
            )
            continue

        loaded.append(key)

    missing = sorted(set(flat_reference) - set(loaded))
    allowed_missing = []
    unexpected_missing = []

    for key in missing:
        path = _path_to_str(key)
        allowed = any(pattern.fullmatch(path) is not None for pattern in patterns)
        if allowed:
            allowed_missing.append(path)
        else:
            unexpected_missing.append(path)

    unexpected_loaded = sorted(_path_to_str(key) for key in set(flat_loaded) - set(flat_reference))

    return SpatialCheckpointAudit(
        loaded=tuple(sorted(_path_to_str(key) for key in loaded)),
        allowed_missing=tuple(sorted(allowed_missing)),
        unexpected_missing=tuple(unexpected_missing),
        unexpected_loaded=tuple(unexpected_loaded),
        shape_mismatches=tuple(sorted(shape_mismatches)),
    )
