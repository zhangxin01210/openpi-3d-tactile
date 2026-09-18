"""
OpenPI 3D + tactile：数据集内置 Spatial Derived Modality 导出器
（export_spatial_derived_dataset.py）

作用
----
将当前仓库中的原始机器人数据集：

    data/<dataset>/
    ├── data/
    ├── images/
    ├── meta/
    └── videos/

通过唯一的 canonical：

    openpi.spatial.SpatialPreprocessor

离线导出为同一数据集中的 derived spatial modality：

    data/<dataset>/
    └── spatial/
        └── v1/
            ├── manifest.json
            ├── static/
            │   ├── finger_id.npy
            │   └── taxel_id.npy
            └── episodes/
                ├── episode_000000/
                │   ├── manifest.json
                │   ├── shard_000000/
                │   ├── shard_000001/
                │   └── ...
                └── ...

核心边界
--------
本脚本只负责：

    RawSpatialDataset
        ↓
    SpatialPreprocessor
        ↓
    NPY shards

它不重新实现：
    - depth -> 3D
    - calibration
    - FK
    - tactile geometry
    - voxelization
    - sampling
    - RGB association

因此 offline derived-data 生成和未来 online deployment
仍然共享同一个 SpatialPreprocessor。

原始数据与 derived data 的身份
------------------------------
原来的：

    data/
    images/
    meta/
    videos/

是 source of truth。

新增的：

    spatial/v1/

是可删除、可重新生成的 derived modality。

本脚本：
    - 不修改原始 data/
    - 不修改 images/
    - 不修改 videos/
    - 不修改 meta/info.json
    - 不把 spatial 伪装成 LeRobot 原生 feature

未来模型侧 reader 显式读取：

    <dataset>/spatial/v1

即可。

V1 输出 schema
--------------
每个 frame：

    visual_xyz_m         [Nv,3] float32
    visual_rgb           [Nv,3] uint8
    visual_rgb_valid     [Nv] bool

    tactile_xyz_m        [600,3] float32
    tactile_force_base   [600,3] float32
    tactile_force_norm   [600] float32

静态 topology：

    finger_id            [600] int8
    taxel_id             [600] int16

默认：

    Nv = 4096
    coordinate frame = base_link
    xyz unit = m
    force unit = dataset_native

流式设计
--------
对于一个 episode：

    parquet rows
        ───────────────►

    front video
        ───────────────►

    left video
        ───────────────►

三条流按 frame_index 对齐后立即 preprocess。

因此：
    - parquet 只顺序扫描一次
    - 每个 camera video 只顺序 decode 一次
    - 不需要把整段 episode RGB 常驻内存
    - shard_size 只控制 derived output buffer 大小

这比“每个 shard 重新调用视频抽帧器”更适合全量导出。

版本原则
--------
如果未来改变：

    - visual point count
    - voxel size
    - sampler
    - camera roles
    - calibration
    - tactile geometry
    - force semantics

不要覆盖已经冻结的 v1。

应使用：

    spatial/v2/
    spatial/v3/

基本使用
--------
1. 四帧 smoke test：

    PYTHONPATH=src python scripts/spatial/export_spatial_derived_dataset.py \
        --dataset data/press_0828_17 \
        --episodes 0 \
        --frames 0,50,100,213 \
        --version v1_test \
        --cameras front,left \
        --num-points 4096 \
        --shard-size 4

默认输出：

    data/press_0828_17/spatial/v1_test/

2. 正式全量：

    PYTHONPATH=src python scripts/spatial/export_spatial_derived_dataset.py \
        --dataset data/press_0828_17 \
        --episodes all \
        --frames all \
        --version v1 \
        --cameras front,left \
        --num-points 4096 \
        --shard-size 128

3. 如果 dataset 参数省略：

    --dataset data/press_0828_17

是当前默认值。

注意
----
- 多 episode 导出时，--frames 必须为 all。
- timestamp 直接来自 source parquet，不通过 FPS 推断。
- RGB frame_index 当前按 episode 内视频 decode ordinal 对齐。
- exporter 会检查 parquet frame 与每一路 RGB frame 严格一致。
- version root 已存在时默认拒绝覆盖。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterator

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import DEFAULT_T16_TRANSFORMED_PATH
from openpi.spatial.preprocess import DEFAULT_T30_RIGHT_TRANSFORMED_PATH
from openpi.spatial.preprocess import SpatialPreprocessor
from openpi.spatial_dataset.source import RawSpatialDataset
from openpi.spatial_dataset.source import VideoFrame


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export canonical spatial derived data "
            "inside the source dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/press_0828_17"
        ),
        help=(
            "数据集根目录。相对路径以当前 OpenPI repo 为基准。"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help=(
            '使用 "all" 或逗号分隔 episode ids，'
            "例如 0,1,2。"
        ),
    )

    parser.add_argument(
        "--frames",
        type=str,
        default="all",
        help=(
            '使用 "all" 或逗号分隔 frame ids。'
            "显式 frame list 只允许单 episode。"
        ),
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "仅对 --frames all 生效；1 表示全部 frame。"
        ),
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="derived modality 版本，例如 v1 / v1_test / v2。",
    )

    parser.add_argument(
        "--cameras",
        type=str,
        default="front,left",
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "特殊情况下覆盖输出路径。"
            "标准用法无需填写，默认 <dataset>/spatial/<version>/。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "显式删除并重建已存在的 version root。"
            "正式版本通常应改 version，而不是覆盖。"
        ),
    )

    return parser.parse_args()


# =============================================================================
# 2. 路径与参数
# =============================================================================

def resolve_dataset_root(
    *,
    repo_root: Path,
    dataset_arg: Path,
) -> Path:
    """
    dataset 相对路径统一相对于当前 OpenPI repo。
    """
    if dataset_arg.is_absolute():
        root = dataset_arg
    else:
        root = (
            repo_root
            / dataset_arg
        )

    root = (
        root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise FileNotFoundError(
            root
        )

    required = (
        root / "data",
        root / "meta",
        root / "videos",
    )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise ValueError(
            "Dataset root does not match expected layout. "
            f"Missing: {missing}"
        )

    return root


def validate_version_name(
    value: str,
) -> str:
    version = value.strip()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        version,
    ):
        raise ValueError(
            "--version must match "
            "[A-Za-z0-9][A-Za-z0-9._-]*"
        )

    if version in (
        ".",
        "..",
    ):
        raise ValueError(
            "Invalid --version"
        )

    return version


def resolve_output_root(
    *,
    repo_root: Path,
    dataset_root: Path,
    version: str,
    output_arg: Path | None,
) -> Path:
    if output_arg is None:
        return (
            dataset_root
            / "spatial"
            / version
        ).resolve()

    if output_arg.is_absolute():
        return (
            output_arg
            .expanduser()
            .resolve()
        )

    return (
        repo_root
        / output_arg
    ).resolve()


def prepare_output_root(
    *,
    output_root: Path,
    overwrite: bool,
) -> None:
    """
    默认不覆盖已有 derived version。
    """
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                "Spatial derived version already exists:\n"
                f"  {output_root}\n"
                "Use a new --version, remove a test version, "
                "or explicitly pass --overwrite."
            )

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )


def parse_camera_roles(
    text: str,
) -> tuple[str, ...]:
    roles = tuple(
        token.strip()
        for token in text.split(
            ","
        )
        if token.strip()
    )

    if not roles:
        raise ValueError(
            "--cameras cannot be empty"
        )

    if len(
        roles
    ) != len(
        set(
            roles
        )
    ):
        raise ValueError(
            "--cameras contains duplicates"
        )

    return roles


def select_episode_ids(
    *,
    text: str,
    available: tuple[int, ...],
) -> list[int]:
    if text.strip().lower() == "all":
        return list(
            available
        )

    selected = [
        int(
            token.strip()
        )
        for token in text.split(
            ","
        )
        if token.strip()
    ]

    if not selected:
        raise ValueError(
            "--episodes resolved to empty list"
        )

    if len(
        selected
    ) != len(
        set(
            selected
        )
    ):
        raise ValueError(
            "--episodes contains duplicates"
        )

    missing = sorted(
        set(
            selected
        )
        - set(
            available
        )
    )

    if missing:
        raise KeyError(
            "Requested episodes do not exist: "
            f"{missing}"
        )

    return sorted(
        selected
    )


def select_frame_ids(
    *,
    text: str,
    available: list[int],
    stride: int,
) -> list[int]:
    if stride <= 0:
        raise ValueError(
            "--frame-stride must be > 0"
        )

    if text.strip().lower() == "all":
        return list(
            available[
                ::stride
            ]
        )

    selected = [
        int(
            token.strip()
        )
        for token in text.split(
            ","
        )
        if token.strip()
    ]

    if not selected:
        raise ValueError(
            "--frames resolved to empty list"
        )

    if len(
        selected
    ) != len(
        set(
            selected
        )
    ):
        raise ValueError(
            "--frames contains duplicates"
        )

    missing = sorted(
        set(
            selected
        )
        - set(
            available
        )
    )

    if missing:
        raise KeyError(
            "Requested frames do not exist in episode: "
            f"{missing}"
        )

    return sorted(
        selected
    )


# =============================================================================
# 3. Source state helpers
# =============================================================================

def build_state_index(
    dataset: RawSpatialDataset,
) -> dict[int, dict[str, Any]]:
    """
    建立 lightweight frame -> state/timestamp 索引。

    不包含 depth / RGB。
    """
    document = dataset.states()

    frame_ids = np.asarray(
        document[
            "frame_index"
        ]
    )

    timestamps = np.asarray(
        document[
            "timestamp"
        ],
        dtype=np.float64,
    )

    states = np.asarray(
        document[
            "observation.state"
        ]
    )

    if not (
        len(
            frame_ids
        )
        == len(
            timestamps
        )
        == len(
            states
        )
    ):
        raise ValueError(
            "Source state columns have inconsistent lengths"
        )

    output = {}

    for row_index, frame_raw in enumerate(
        frame_ids
    ):
        frame_id = int(
            frame_raw
        )

        if frame_id in output:
            raise ValueError(
                f"Duplicate frame_index: {frame_id}"
            )

        timestamp = float(
            timestamps[
                row_index
            ]
        )

        state = np.asarray(
            states[
                row_index
            ]
        )

        if not np.isfinite(
            timestamp
        ):
            raise ValueError(
                f"Frame {frame_id}: non-finite timestamp"
            )

        if state.ndim != 1:
            raise ValueError(
                f"Frame {frame_id}: state must be 1-D, "
                f"got {state.shape}"
            )

        output[
            frame_id
        ] = {
            "timestamp_s": timestamp,
            "state": state,
        }

    return output


# =============================================================================
# 4. 同步 parquet / RGB streams
# =============================================================================

def synchronized_raw_frames(
    *,
    dataset: RawSpatialDataset,
    frame_ids: list[int],
    camera_roles: tuple[str, ...],
    state_index: dict[int, dict[str, Any]],
) -> Iterator[
    tuple[
        int,
        dict[str, Any],
    ]
]:
    """
    一个 episode 内同步三类输入：

        parquet depth row
        RGB stream(s)
        state/timestamp index

    每一路 RGB video 只创建一个 iterator，
    因此整个 selected frame set 只顺序 decode 一次。

    输出：

        frame_id,
        {
            timestamp_s,
            state,
            depth_by_role,
            rgb_by_role,
        }
    """
    selected_set = set(
        frame_ids
    )

    unknown_rgb = sorted(
        set(
            camera_roles
        )
        - set(
            dataset.camera_roles
        )
    )

    unknown_depth = sorted(
        set(
            camera_roles
        )
        - set(
            dataset.depth_roles
        )
    )

    if unknown_rgb:
        raise KeyError(
            "Requested cameras missing RGB streams: "
            f"{unknown_rgb}"
        )

    if unknown_depth:
        raise KeyError(
            "Requested cameras missing depth streams: "
            f"{unknown_depth}"
        )

    row_iterator = dataset.iter_rows(
        selected=selected_set,
        depth_roles=camera_roles,
    )

    video_iterators: dict[
        str,
        Iterator[VideoFrame],
    ] = {
        role: iter(
            dataset.iter_video_frames(
                role,
                selected=selected_set,
            )
        )
        for role in camera_roles
    }

    produced = []

    for row in row_iterator:
        frame_id = int(
            row[
                "frame_index"
            ]
        )

        if frame_id not in state_index:
            raise KeyError(
                f"Frame {frame_id}: missing state/timestamp index"
            )

        rgb_by_role = {}

        for role in camera_roles:
            try:
                video_frame = next(
                    video_iterators[
                        role
                    ]
                )
            except StopIteration as exc:
                raise RuntimeError(
                    f"RGB stream {role!r} ended before "
                    f"parquet frame {frame_id}"
                ) from exc

            if (
                video_frame.frame_index
                != frame_id
            ):
                raise RuntimeError(
                    "Parquet / RGB frame misalignment: "
                    f"parquet={frame_id}, "
                    f"camera={role}, "
                    f"rgb={video_frame.frame_index}"
                )

            rgb_by_role[
                role
            ] = video_frame.rgb

        depth_by_role = {
            role: np.asarray(
                row[
                    f"observation.depths.cam_{role}"
                ]
            )
            for role in camera_roles
        }

        produced.append(
            frame_id
        )

        yield (
            frame_id,
            {
                "timestamp_s": (
                    state_index[
                        frame_id
                    ][
                        "timestamp_s"
                    ]
                ),
                "state": (
                    state_index[
                        frame_id
                    ][
                        "state"
                    ]
                ),
                "depth_by_role": (
                    depth_by_role
                ),
                "rgb_by_role": (
                    rgb_by_role
                ),
            },
        )

    if produced != frame_ids:
        raise RuntimeError(
            "Source stream did not yield the selected frames "
            "in the expected order.\n"
            f"expected={frame_ids}\n"
            f"actual={produced}"
        )

    # 所有 selected frames 都处理完后，
    # 每一路 video iterator 也必须正好耗尽。
    for role, iterator in video_iterators.items():
        try:
            extra = next(
                iterator
            )
        except StopIteration:
            continue

        raise RuntimeError(
            f"RGB stream {role!r} yielded unexpected extra "
            f"selected frame {extra.frame_index}"
        )


# =============================================================================
# 5. Observation buffer -> NPY arrays
# =============================================================================

def append_observation(
    *,
    observations: list[Any],
    observation: Any,
    static_ids: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """
    将一个 SpatialObservation 放入当前 shard buffer，
    同时冻结 / 检查 tactile topology。
    """
    current_ids = {
        "finger_id": np.asarray(
            observation.finger_id,
            dtype=np.int8,
        ),
        "taxel_id": np.asarray(
            observation.taxel_id,
            dtype=np.int16,
        ),
    }

    if static_ids is None:
        static_ids = {
            name: array.copy()
            for name, array
            in current_ids.items()
        }

    else:
        for name, array in current_ids.items():
            if not np.array_equal(
                array,
                static_ids[
                    name
                ],
            ):
                raise RuntimeError(
                    f"Frame {observation.frame_index}: "
                    f"{name} violates static topology contract"
                )

    observations.append(
        observation
    )

    return static_ids


def observations_to_arrays(
    observations: list[Any],
) -> dict[str, np.ndarray]:
    """
    将一个 shard buffer 转成固定 shape arrays。
    """
    if not observations:
        raise ValueError(
            "Cannot serialize empty observation buffer"
        )

    return {
        "frame_index": np.asarray(
            [
                item.frame_index
                for item in observations
            ],
            dtype=np.int64,
        ),
        "timestamp_s": np.asarray(
            [
                item.timestamp_s
                for item in observations
            ],
            dtype=np.float64,
        ),
        "visual_xyz_m": np.stack(
            [
                item.visual_xyz_m
                for item in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        "visual_rgb": np.stack(
            [
                item.visual_rgb
                for item in observations
            ],
            axis=0,
        ).astype(
            np.uint8,
            copy=False,
        ),
        "visual_rgb_valid": np.stack(
            [
                item.visual_rgb_valid
                for item in observations
            ],
            axis=0,
        ).astype(
            np.bool_,
            copy=False,
        ),
        "tactile_xyz_m": np.stack(
            [
                item.tactile_xyz_m
                for item in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        "tactile_force_base": np.stack(
            [
                item.tactile_force_base
                for item in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        "tactile_force_norm": np.stack(
            [
                item.tactile_force_norm
                for item in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
    }


# =============================================================================
# 6. Atomic shard writer
# =============================================================================

def write_shard_atomic(
    *,
    episode_root: Path,
    shard_index: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    临时目录写完后再 rename。

    因此正式 shard 不会出现半写状态。
    """
    shard_name = (
        f"shard_{shard_index:06d}"
    )

    final_dir = (
        episode_root
        / shard_name
    )

    temp_dir = (
        episode_root
        / f".{shard_name}.tmp"
    )

    if final_dir.exists():
        raise FileExistsError(
            final_dir
        )

    if temp_dir.exists():
        shutil.rmtree(
            temp_dir
        )

    temp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    files = {}

    try:
        for field_name, array in arrays.items():
            path = (
                temp_dir
                / f"{field_name}.npy"
            )

            np.save(
                path,
                array,
                allow_pickle=False,
            )

            files[
                field_name
            ] = {
                "file": (
                    f"{shard_name}/{field_name}.npy"
                ),
                "shape": [
                    int(
                        value
                    )
                    for value
                    in array.shape
                ],
                "dtype": str(
                    array.dtype
                ),
                "bytes": int(
                    path.stat().st_size
                ),
            }

        os.replace(
            temp_dir,
            final_dir,
        )

    except Exception:
        if temp_dir.exists():
            shutil.rmtree(
                temp_dir
            )

        raise

    return {
        "name": shard_name,
        "count": int(
            arrays[
                "frame_index"
            ].shape[
                0
            ]
        ),
        "first_frame": int(
            arrays[
                "frame_index"
            ][
                0
            ]
        ),
        "last_frame": int(
            arrays[
                "frame_index"
            ][
                -1
            ]
        ),
        "files": files,
    }


# =============================================================================
# 7. Static topology
# =============================================================================

def write_static_ids(
    *,
    output_root: Path,
    static_ids: dict[str, np.ndarray],
) -> dict[str, Any]:
    static_root = (
        output_root
        / "static"
    )

    static_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {}

    for name, array in static_ids.items():
        path = (
            static_root
            / f"{name}.npy"
        )

        np.save(
            path,
            array,
            allow_pickle=False,
        )

        result[
            name
        ] = {
            "file": (
                f"static/{name}.npy"
            ),
            "shape": [
                int(
                    value
                )
                for value
                in array.shape
            ],
            "dtype": str(
                array.dtype
            ),
            "sha256": sha256_file(
                path
            ),
        }

    return result


# =============================================================================
# 8. Provenance
# =============================================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            block = file.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def jsonable(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): jsonable(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            jsonable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def build_asset_provenance(
    *,
    repo_root: Path,
    config: Any,
) -> dict[str, Any]:
    """
    记录真正影响 numerical spatial output 的静态资产。
    """
    relative_assets = {
        "camera_internal_config": (
            config
            .calibration
            .camera_config_path
        ),
        "camera_external_config": (
            config
            .calibration
            .camera_extrinsic_path
        ),
        "robot_urdf": (
            config
            .calibration
            .robot_urdf_path
        ),
        "state_mapping": (
            config
            .calibration
            .state_mapping_path
        ),
        "tactile_t16_transformed": (
            DEFAULT_T16_TRANSFORMED_PATH
        ),
        "tactile_t30_right_transformed": (
            DEFAULT_T30_RIGHT_TRANSFORMED_PATH
        ),
    }

    result = {}

    for name, relative_path in relative_assets.items():
        path = (
            repo_root
            / Path(
                relative_path
            )
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        result[
            name
        ] = {
            "path": str(
                relative_path
            ),
            "sha256": sha256_file(
                path
            ),
            "bytes": int(
                path.stat().st_size
            ),
        }

    return result


def build_source_meta_provenance(
    dataset_root: Path,
) -> dict[str, Any]:
    """
    只 hash source metadata，不 hash 大视频。
    """
    names = (
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
    )

    result = {}

    for name in names:
        path = (
            dataset_root
            / "meta"
            / name
        )

        if not path.is_file():
            continue

        result[
            name
        ] = {
            "sha256": sha256_file(
                path
            ),
            "bytes": int(
                path.stat().st_size
            ),
        }

    return result


def git_provenance(
    repo_root: Path,
) -> dict[str, Any]:
    """
    记录生成 derived data 时使用的代码版本。
    """
    try:
        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        return {
            "commit": commit,
            "dirty": bool(
                status.strip()
            ),
        }

    except Exception:
        return {
            "commit": None,
            "dirty": None,
        }


# =============================================================================
# 9. Episode export
# =============================================================================

def export_episode(
    *,
    dataset_root: Path,
    episode_id: int,
    output_root: Path,
    preprocessor: SpatialPreprocessor,
    camera_roles: tuple[str, ...],
    frames_arg: str,
    frame_stride: int,
    shard_size: int,
    static_ids: dict[str, np.ndarray] | None,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """
    单 episode streaming export。

    一个 episode 内：
        parquet 一次
        每 camera video 一次
        observations 按 shard_size 缓冲后落盘
    """
    dataset = RawSpatialDataset(
        dataset_root,
        episode=episode_id,
    )

    state_index = build_state_index(
        dataset
    )

    available_frames = sorted(
        state_index
    )

    selected_frames = select_frame_ids(
        text=frames_arg,
        available=available_frames,
        stride=frame_stride,
    )

    if not selected_frames:
        raise RuntimeError(
            f"Episode {episode_id}: no frames selected"
        )

    episode_name = (
        f"episode_{episode_id:06d}"
    )

    episode_root = (
        output_root
        / "episodes"
        / episode_name
    )

    if episode_root.exists():
        raise FileExistsError(
            episode_root
        )

    episode_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    start_wall = time.perf_counter()

    shard_entries = []
    observation_buffer = []
    shard_index = 0
    processed_count = 0

    try:
        raw_stream = synchronized_raw_frames(
            dataset=dataset,
            frame_ids=selected_frames,
            camera_roles=camera_roles,
            state_index=state_index,
        )

        for frame_id, raw in raw_stream:
            observation = preprocessor.preprocess(
                frame_index=frame_id,
                timestamp_s=float(
                    raw[
                        "timestamp_s"
                    ]
                ),
                state=raw[
                    "state"
                ],
                depth_by_role=raw[
                    "depth_by_role"
                ],
                rgb_by_role=raw[
                    "rgb_by_role"
                ],
            )

            static_ids = append_observation(
                observations=observation_buffer,
                observation=observation,
                static_ids=static_ids,
            )

            processed_count += 1

            if (
                len(
                    observation_buffer
                )
                >= shard_size
            ):
                arrays = observations_to_arrays(
                    observation_buffer
                )

                entry = write_shard_atomic(
                    episode_root=episode_root,
                    shard_index=shard_index,
                    arrays=arrays,
                )

                shard_entries.append(
                    entry
                )

                print(
                    f"    shard {shard_index:06d}: "
                    f"{entry['count']} frames | "
                    f"progress "
                    f"{processed_count}/"
                    f"{len(selected_frames)}"
                )

                observation_buffer.clear()
                shard_index += 1

        # 最后不足一个 shard 的尾部。
        if observation_buffer:
            arrays = observations_to_arrays(
                observation_buffer
            )

            entry = write_shard_atomic(
                episode_root=episode_root,
                shard_index=shard_index,
                arrays=arrays,
            )

            shard_entries.append(
                entry
            )

            print(
                f"    shard {shard_index:06d}: "
                f"{entry['count']} frames | "
                f"progress "
                f"{processed_count}/"
                f"{len(selected_frames)}"
            )

            observation_buffer.clear()

        if (
            processed_count
            != len(
                selected_frames
            )
        ):
            raise RuntimeError(
                f"Episode {episode_id}: processed "
                f"{processed_count} frames, expected "
                f"{len(selected_frames)}"
            )

        elapsed = (
            time.perf_counter()
            - start_wall
        )

        episode_manifest = {
            "episode_index": int(
                episode_id
            ),
            "source_frame_count": int(
                len(
                    available_frames
                )
            ),
            "selected_frame_count": int(
                len(
                    selected_frames
                )
            ),
            "first_selected_frame": int(
                selected_frames[
                    0
                ]
            ),
            "last_selected_frame": int(
                selected_frames[
                    -1
                ]
            ),
            "first_timestamp_s": float(
                state_index[
                    selected_frames[
                        0
                    ]
                ][
                    "timestamp_s"
                ]
            ),
            "last_timestamp_s": float(
                state_index[
                    selected_frames[
                        -1
                    ]
                ][
                    "timestamp_s"
                ]
            ),
            "shards": shard_entries,
            "wall_seconds": float(
                elapsed
            ),
        }

        with (
            episode_root
            / "manifest.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                episode_manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return (
            {
                "episode_index": int(
                    episode_id
                ),
                "path": (
                    f"episodes/{episode_name}"
                ),
                "selected_frame_count": int(
                    len(
                        selected_frames
                    )
                ),
                "source_frame_count": int(
                    len(
                        available_frames
                    )
                ),
                "wall_seconds": float(
                    elapsed
                ),
            },
            static_ids,
        )

    except Exception:
        # 当前 episode 如果失败，不留下半成品目录。
        if episode_root.exists():
            shutil.rmtree(
                episode_root
            )

        raise


# =============================================================================
# 10. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    total_start = time.perf_counter()

    repo_root = Path(
        "."
    ).resolve()

    dataset_root = resolve_dataset_root(
        repo_root=repo_root,
        dataset_arg=args.dataset,
    )

    version = validate_version_name(
        args.version
    )

    output_root = resolve_output_root(
        repo_root=repo_root,
        dataset_root=dataset_root,
        version=version,
        output_arg=args.output,
    )

    if args.shard_size <= 0:
        raise ValueError(
            "--shard-size must be > 0"
        )

    if args.num_points <= 0:
        raise ValueError(
            "--num-points must be > 0"
        )

    if args.frame_stride <= 0:
        raise ValueError(
            "--frame-stride must be > 0"
        )

    camera_roles = parse_camera_roles(
        args.cameras
    )

    available_episodes = (
        RawSpatialDataset.discover_episode_ids(
            dataset_root
        )
    )

    selected_episodes = select_episode_ids(
        text=args.episodes,
        available=available_episodes,
    )

    if (
        len(
            selected_episodes
        )
        > 1
        and args.frames.strip().lower()
        != "all"
    ):
        raise ValueError(
            "Explicit --frames is only allowed for "
            "a single episode. "
            "Use --frames all for multi-episode export."
        )

    # -------------------------------------------------------------------------
    # 10.1 在正式创建 output 之前，先检查第一 episode camera contract。
    # -------------------------------------------------------------------------
    probe = RawSpatialDataset(
        dataset_root,
        episode=selected_episodes[
            0
        ],
    )

    missing_rgb = sorted(
        set(
            camera_roles
        )
        - set(
            probe.camera_roles
        )
    )

    missing_depth = sorted(
        set(
            camera_roles
        )
        - set(
            probe.depth_roles
        )
    )

    if missing_rgb:
        raise KeyError(
            f"Missing RGB cameras: {missing_rgb}"
        )

    if missing_depth:
        raise KeyError(
            f"Missing depth cameras: {missing_depth}"
        )

    # -------------------------------------------------------------------------
    # 10.2 创建 spatial version root
    # -------------------------------------------------------------------------
    prepare_output_root(
        output_root=output_root,
        overwrite=args.overwrite,
    )

    print(
        "dataset:",
        dataset_root,
    )
    print(
        "spatial output:",
        output_root,
    )
    print(
        "available episodes:",
        len(
            available_episodes
        ),
    )
    print(
        "selected episodes:",
        selected_episodes,
    )
    print(
        "camera roles:",
        camera_roles,
    )
    print(
        "visual N:",
        args.num_points,
    )

    # -------------------------------------------------------------------------
    # 10.3 唯一 canonical SpatialPreprocessor
    # -------------------------------------------------------------------------
    config = (
        make_baseline_config()
        .with_camera_roles(
            *camera_roles
        )
        .with_visual_sampling(
            num_points=args.num_points,
        )
    )

    preprocessor = (
        SpatialPreprocessor.from_repo_root(
            repo_root=repo_root,
            config=config,
        )
    )

    # -------------------------------------------------------------------------
    # 10.4 Provenance
    # -------------------------------------------------------------------------
    asset_provenance = build_asset_provenance(
        repo_root=repo_root,
        config=config,
    )

    source_meta = build_source_meta_provenance(
        dataset_root
    )

    git_info = git_provenance(
        repo_root
    )

    # -------------------------------------------------------------------------
    # 10.5 Episode export
    # -------------------------------------------------------------------------
    static_ids = None
    episode_entries = []

    try:
        for ordinal, episode_id in enumerate(
            selected_episodes,
            start=1,
        ):
            print(
                f"\n[{ordinal}/{len(selected_episodes)}] "
                f"episode {episode_id}"
            )

            entry, static_ids = export_episode(
                dataset_root=dataset_root,
                episode_id=episode_id,
                output_root=output_root,
                preprocessor=preprocessor,
                camera_roles=camera_roles,
                frames_arg=args.frames,
                frame_stride=args.frame_stride,
                shard_size=args.shard_size,
                static_ids=static_ids,
            )

            episode_entries.append(
                entry
            )

        if static_ids is None:
            raise RuntimeError(
                "No spatial observations were exported"
            )

        # ---------------------------------------------------------------------
        # 10.6 Static tactile topology
        # ---------------------------------------------------------------------
        static_manifest = write_static_ids(
            output_root=output_root,
            static_ids=static_ids,
        )

        # ---------------------------------------------------------------------
        # 10.7 Dataset-level manifest
        # ---------------------------------------------------------------------
        total_wall = (
            time.perf_counter()
            - total_start
        )

        total_selected_frames = int(
            sum(
                entry[
                    "selected_frame_count"
                ]
                for entry in episode_entries
            )
        )

        manifest = {
            "derived_schema_version": 1,
            "spatial_version": version,
            "status": "complete",
            "source_dataset": {
                "name": dataset_root.name,
                "available_episode_count": int(
                    len(
                        available_episodes
                    )
                ),
                "selected_episode_count": int(
                    len(
                        selected_episodes
                    )
                ),
                "selected_episodes": [
                    int(
                        value
                    )
                    for value in selected_episodes
                ],
                "source_meta_sha256": source_meta,
            },
            "spatial_contract": {
                "coordinate_frame": "base_link",
                "xyz_unit": "m",
                "force_unit": "dataset_native",
                "visual_points_per_frame": int(
                    args.num_points
                ),
                "tactile_points_per_frame": 600,
                "camera_roles": list(
                    camera_roles
                ),
            },
            "storage": {
                "format": "npy_shards_v1",
                "shard_size": int(
                    args.shard_size
                ),
                "static": static_manifest,
                "episodes": episode_entries,
            },
            "preprocess_config": jsonable(
                asdict(
                    config
                )
            ),
            "asset_provenance": asset_provenance,
            "generator": {
                "openpi_repo_git": git_info,
                "source_reader": (
                    "openpi.spatial_dataset.source."
                    "RawSpatialDataset"
                ),
            },
            "runtime": {
                "total_selected_frames": (
                    total_selected_frames
                ),
                "total_wall_seconds": float(
                    total_wall
                ),
                "average_wall_ms_per_frame_including_io_decode_write": (
                    float(
                        total_wall
                        / total_selected_frames
                        * 1000.0
                    )
                    if total_selected_frames
                    else None
                ),
            },
        }

        manifest_path = (
            output_root
            / "manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "\n===== COMPLETE ====="
        )
        print(
            "dataset:",
            dataset_root,
        )
        print(
            "spatial version:",
            version,
        )
        print(
            "output:",
            output_root,
        )
        print(
            "episodes:",
            len(
                episode_entries
            ),
        )
        print(
            "frames:",
            total_selected_frames,
        )
        print(
            "wall seconds:",
            f"{total_wall:.2f}",
        )
        print(
            "manifest:",
            manifest_path,
        )

    except Exception:
        # 本次 version root 是 exporter 独占创建的。
        # 任意失败都整体删除，避免 incomplete version 被误当正式数据。
        if output_root.exists():
            shutil.rmtree(
                output_root
            )

        raise


if __name__ == "__main__":
    main()
