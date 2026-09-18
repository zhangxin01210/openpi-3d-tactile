"""
OpenPI 3D + tactile：离线派生空间数据集导出器
（export_spatial_derived_dataset.py）

作用
----
把旧采集数据中的：

    RGB-D + observation.state + timestamp

通过已经验收完成的：

    SpatialPreprocessor

离线转换成固定 schema 的 derived spatial dataset。

这是下一阶段的第一份正式数据工程文件。

核心原则
--------
1. 不复制 preprocessing 逻辑。
   本脚本只负责：
       读旧数据
       -> 调 SpatialPreprocessor.preprocess()
       -> 写磁盘

2. offline / online preprocessing 仍然共用：
       src/openpi/spatial/preprocess.py

3. 输出使用简单、透明、可 mmap 的 NumPy shard 格式，
   暂时不绑定某个训练框架或 LeRobot writer。

4. 每个 shard 目录独立、固定 shape；
   后续 OpenPI dataset adapter 只需要读取这些 .npy，
   不再重新跑 RGB-D / FK / tactile geometry。

输出 schema
-----------
根目录：

    manifest.json
    static/
        finger_id.npy          [600] int8
        taxel_id.npy           [600] int16

    shard_000000/
        frame_index.npy        [B] int64
        timestamp_s.npy        [B] float64

        visual_xyz_m.npy       [B,Nv,3] float32
        visual_rgb.npy         [B,Nv,3] uint8
        visual_rgb_valid.npy   [B,Nv] bool

        tactile_xyz_m.npy      [B,600,3] float32
        tactile_force_base.npy [B,600,3] float32
        tactile_force_norm.npy [B,600] float32

    shard_000001/
        ...

为什么 finger_id / taxel_id 只存一次
-------------------------------
它们在所有 frame 中都是固定拓扑：

    finger_id:
        0..4，每根 120 个

    taxel_id:
        每根 1..120

所以没有必要每一帧重复写 600 个 id。
导出时仍会逐帧检查它们是否与 static contract 一致。

manifest.json 记录
------------------
- derived schema version
- source dataset / episode
- camera roles
- visual num_points
- voxel size / sampler
- units / coordinate frame
- shard 列表
- calibration / URDF / mapping / tactile geometry 的 SHA256
- 完整 SpatialPreprocessConfig
- 总 frame 数

这样以后看到一个 derived dataset，可以知道它到底由哪套静态资产生成。

为什么用 shard + .npy
---------------------
优点：
    - 无额外依赖
    - 文件格式透明
    - np.load(..., mmap_mode="r") 可直接 mmap
    - 单个 shard 损坏不会污染整套数据
    - 后续可以非常容易适配 PyTorch / JAX / OpenPI

当前先不做：
    - zarr
    - HDF5 并行 writer
    - parquet 嵌套 tensor
    - 自定义数据库

这些等真正出现第二种存储需求再做。

基本使用
--------
先只导出 4 帧做 smoke test：

    PYTHONPATH=src python scripts/export_spatial_derived_dataset.py \
        --legacy-repo ../3D_tactile \
        --dataset data/press_0828_17 \
        --episode 0 \
        --frames 0,50,100,213 \
        --cameras front,left \
        --num-points 4096 \
        --shard-size 4 \
        --output output/derived_spatial/press_0828_17_ep0_test

确认后全量：

    PYTHONPATH=src python scripts/export_spatial_derived_dataset.py \
        --legacy-repo ../3D_tactile \
        --dataset data/press_0828_17 \
        --episode 0 \
        --frames all \
        --cameras front,left \
        --num-points 4096 \
        --shard-size 128 \
        --output output/derived_spatial/press_0828_17_ep0

如果输出目录已存在：
    默认拒绝覆盖。

显式覆盖：
    --overwrite

注意
----
- timestamp 直接读取 ds.states()["timestamp"]，不猜 FPS。
- video decode / disk IO 是 exporter 成本，不属于 SpatialPreprocessor latency。
- 当前 baseline 仍为 4096 visual；8192 保留为后续 encoder ablation 候选。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import DEFAULT_T16_TRANSFORMED_PATH
from openpi.spatial.preprocess import DEFAULT_T30_RIGHT_TRANSFORMED_PATH
from openpi.spatial.preprocess import SpatialPreprocessor


# =============================================================================
# 1. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical SpatialObservation into mmap-friendly NPY shards."
    )

    parser.add_argument(
        "--legacy-repo",
        type=Path,
        default=Path("../3D_tactile"),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="相对于 legacy repo 的 dataset 路径。",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frames",
        type=str,
        default="all",
        help='使用 "all" 或逗号分隔 frame ids，例如 0,50,100,213。',
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help='仅对 --frames all 生效；1 表示全部 frame。',
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
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# 2. Legacy Dataset adapter
# =============================================================================

def import_legacy_dataset(
    legacy_repo: Path,
):
    """
    动态导入旧 Dataset / video_frames。

    旧 repo dependency 只停留在 scripts/ exporter，
    不进入 src/openpi/spatial core。
    """
    legacy_src = (
        legacy_repo
        / "pointcloud_delivery"
        / "src"
    ).resolve()

    if not legacy_src.is_dir():
        raise FileNotFoundError(
            f"Legacy src not found: {legacy_src}"
        )

    sys.path.insert(
        0,
        str(
            legacy_src
        ),
    )

    try:
        from dataset import Dataset
        from dataset import video_frames
    finally:
        sys.path.pop(
            0
        )

    return (
        Dataset,
        video_frames,
    )


def normalize_video_frames(
    decoded,
    *,
    frames: list[int],
) -> dict[
    int,
    np.ndarray,
]:
    """
    兼容旧 video_frames 的两种返回形式：

        images
        (images, video_info)

    images 又可能是：
        frame-indexed mapping
        sequence

    统一成：
        frame_id -> RGB ndarray
    """
    images = (
        decoded[
            0
        ]
        if isinstance(
            decoded,
            tuple,
        )
        else decoded
    )

    if isinstance(
        images,
        dict,
    ):
        output = {
            int(
                frame
            ): np.asarray(
                image,
                dtype=np.uint8,
            )
            for frame, image in images.items()
        }
    else:
        if len(
            images
        ) != len(
            frames
        ):
            raise RuntimeError(
                "Decoded RGB sequence length does not match requested frames: "
                f"{len(images)} vs {len(frames)}"
            )

        output = {
            int(
                frame
            ): np.asarray(
                images[
                    index
                ],
                dtype=np.uint8,
            )
            for index, frame in enumerate(
                frames
            )
        }

    missing = sorted(
        set(
            frames
        )
        - set(
            output
        )
    )

    if missing:
        raise KeyError(
            f"Decoded video missing frames: {missing}"
        )

    return output


# =============================================================================
# 3. State / timestamp index
# =============================================================================

def build_state_index(
    dataset,
) -> dict[
    int,
    dict[str, Any],
]:
    """
    一次性读取 ds.states()。

    当前 legacy dataset 已提供：
        frame_index
        timestamp
        observation.state

    因此 timestamp 使用真实记录值，不通过 frame/FPS 猜测。
    """
    document = dataset.states()

    required = (
        "frame_index",
        "timestamp",
        "observation.state",
    )

    missing_fields = [
        field
        for field in required
        if field not in document
    ]

    if missing_fields:
        raise KeyError(
            "Legacy states table missing fields: "
            f"{missing_fields}"
        )

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
            "states() columns have inconsistent lengths"
        )

    index = {}

    for row_index, frame_id_raw in enumerate(
        frame_ids
    ):
        frame_id = int(
            frame_id_raw
        )

        if frame_id in index:
            raise ValueError(
                f"Duplicate frame_index in states(): {frame_id}"
            )

        timestamp = float(
            timestamps[
                row_index
            ]
        )

        if not np.isfinite(
            timestamp
        ):
            raise ValueError(
                f"Frame {frame_id}: non-finite timestamp {timestamp}"
            )

        state = np.asarray(
            states[
                row_index
            ]
        )

        if state.ndim != 1:
            raise ValueError(
                f"Frame {frame_id}: state must be 1-D, got {state.shape}"
            )

        index[
            frame_id
        ] = {
            "timestamp_s": timestamp,
            "state": state,
        }

    return index


def select_frames(
    *,
    text: str,
    available_frames: list[int],
    stride: int,
) -> list[int]:
    """
    frame 选择：
        all
        或显式逗号列表
    """
    if stride <= 0:
        raise ValueError(
            "--frame-stride must be > 0"
        )

    if text.strip().lower() == "all":
        return available_frames[
            ::stride
        ]

    frames = [
        int(
            token.strip()
        )
        for token in text.split(
            ","
        )
        if token.strip()
    ]

    if not frames:
        raise ValueError(
            "--frames resolved to empty list"
        )

    if len(
        set(
            frames
        )
    ) != len(
        frames
    ):
        raise ValueError(
            "--frames contains duplicate frame ids"
        )

    available = set(
        available_frames
    )

    missing = sorted(
        set(
            frames
        )
        - available
    )

    if missing:
        raise KeyError(
            f"Requested frames not in states(): {missing}"
        )

    return sorted(
        frames
    )


# =============================================================================
# 4. Shard raw input loader
# =============================================================================

def load_raw_shard(
    *,
    dataset,
    video_frames,
    frame_ids: list[int],
    camera_roles: tuple[str, ...],
    state_index: dict[int, dict[str, Any]],
) -> dict[
    int,
    dict[str, Any],
]:
    """
    一次读取一个 shard 所需的：
        state
        timestamp
        depth
        RGB

    这样不会把整个 episode 的视频都常驻内存。
    """
    rows = {
        int(
            row[
                "frame_index"
            ]
        ): row
        for row in dataset.rows(
            frame_ids
        )
    }

    missing_rows = sorted(
        set(
            frame_ids
        )
        - set(
            rows
        )
    )

    if missing_rows:
        raise KeyError(
            f"Dataset rows missing frames: {missing_rows}"
        )

    rgb_by_role = {}

    for role in camera_roles:
        decoded = video_frames(
            dataset.video(
                role
            ),
            frame_ids,
        )

        rgb_by_role[
            role
        ] = normalize_video_frames(
            decoded,
            frames=frame_ids,
        )

    output = {}

    for frame_id in frame_ids:
        row = rows[
            frame_id
        ]

        depth = {}

        for role in camera_roles:
            key = (
                f"observation.depths.cam_{role}"
            )

            if key not in row:
                raise KeyError(
                    f"Frame {frame_id}: missing depth key {key!r}"
                )

            depth[
                role
            ] = np.asarray(
                row[
                    key
                ]
            )

        output[
            frame_id
        ] = {
            "timestamp_s": state_index[
                frame_id
            ][
                "timestamp_s"
            ],
            "state": state_index[
                frame_id
            ][
                "state"
            ],
            "depth_by_role": depth,
            "rgb_by_role": {
                role: rgb_by_role[
                    role
                ][
                    frame_id
                ]
                for role in camera_roles
            },
        }

    return output


# =============================================================================
# 5. Derived shard builder
# =============================================================================

def build_derived_shard(
    *,
    preprocessor: SpatialPreprocessor,
    frame_ids: list[int],
    raw_frames: dict[int, dict[str, Any]],
    static_ids: dict[str, np.ndarray] | None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    对一个 shard 的每帧调用 canonical SpatialPreprocessor。

    static_ids:
        第一 shard 为 None；
        第一次 observation 会建立 finger_id / taxel_id contract。

    返回：
        shard arrays
        static ids
    """
    observations = []

    for frame_id in frame_ids:
        raw = raw_frames[
            frame_id
        ]

        observation = (
            preprocessor.preprocess(
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
        )

        observations.append(
            observation
        )

        if static_ids is None:
            static_ids = {
                "finger_id": np.asarray(
                    observation.finger_id,
                    dtype=np.int8,
                ).copy(),
                "taxel_id": np.asarray(
                    observation.taxel_id,
                    dtype=np.int16,
                ).copy(),
            }
        else:
            if not np.array_equal(
                observation.finger_id,
                static_ids[
                    "finger_id"
                ],
            ):
                raise RuntimeError(
                    f"Frame {frame_id}: finger_id violates static topology contract"
                )

            if not np.array_equal(
                observation.taxel_id,
                static_ids[
                    "taxel_id"
                ],
            ):
                raise RuntimeError(
                    f"Frame {frame_id}: taxel_id violates static topology contract"
                )

    if static_ids is None:
        raise RuntimeError(
            "Cannot build empty derived shard"
        )

    arrays = {
        "frame_index": np.asarray(
            [
                observation.frame_index
                for observation in observations
            ],
            dtype=np.int64,
        ),
        "timestamp_s": np.asarray(
            [
                observation.timestamp_s
                for observation in observations
            ],
            dtype=np.float64,
        ),
        "visual_xyz_m": np.stack(
            [
                observation.visual_xyz_m
                for observation in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        "visual_rgb": np.stack(
            [
                observation.visual_rgb
                for observation in observations
            ],
            axis=0,
        ).astype(
            np.uint8,
            copy=False,
        ),
        "visual_rgb_valid": np.stack(
            [
                observation.visual_rgb_valid
                for observation in observations
            ],
            axis=0,
        ).astype(
            np.bool_,
            copy=False,
        ),
        "tactile_xyz_m": np.stack(
            [
                observation.tactile_xyz_m
                for observation in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        "tactile_force_base": np.stack(
            [
                observation.tactile_force_base
                for observation in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        "tactile_force_norm": np.stack(
            [
                observation.tactile_force_norm
                for observation in observations
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
    }

    return (
        arrays,
        static_ids,
    )


# =============================================================================
# 6. Atomic shard writer
# =============================================================================

def write_shard_atomic(
    *,
    output_root: Path,
    shard_index: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    先写临时目录，再 rename 成正式 shard。

    如果进程中途退出，不会留下“看起来完整但只写了一半”的正式 shard。
    """
    shard_name = (
        f"shard_{shard_index:06d}"
    )

    final_dir = (
        output_root
        / shard_name
    )

    temporary_dir = (
        output_root
        / (
            f".{shard_name}.tmp"
        )
    )

    if final_dir.exists():
        raise FileExistsError(
            final_dir
        )

    if temporary_dir.exists():
        shutil.rmtree(
            temporary_dir
        )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    file_manifest = {}

    try:
        for field_name, array in arrays.items():
            path = (
                temporary_dir
                / f"{field_name}.npy"
            )

            np.save(
                path,
                array,
                allow_pickle=False,
            )

            file_manifest[
                field_name
            ] = {
                "file": (
                    f"{shard_name}/{field_name}.npy"
                ),
                "shape": [
                    int(
                        value
                    )
                    for value in array.shape
                ],
                "dtype": str(
                    array.dtype
                ),
                "bytes": int(
                    path.stat().st_size
                ),
            }

        os.replace(
            temporary_dir,
            final_dir,
        )

    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(
                temporary_dir
            )
        raise

    return {
        "name": shard_name,
        "count": int(
            len(
                arrays[
                    "frame_index"
                ]
            )
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
        "files": file_manifest,
    }


# =============================================================================
# 7. Static topology writer
# =============================================================================

def write_static_ids(
    *,
    output_root: Path,
    static_ids: dict[str, np.ndarray],
) -> dict[str, Any]:
    static_dir = (
        output_root
        / "static"
    )

    static_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {}

    for field_name, array in static_ids.items():
        path = (
            static_dir
            / f"{field_name}.npy"
        )

        np.save(
            path,
            array,
            allow_pickle=False,
        )

        result[
            field_name
        ] = {
            "file": (
                f"static/{field_name}.npy"
            ),
            "shape": [
                int(
                    value
                )
                for value in array.shape
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
# 8. Provenance helpers
# =============================================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with Path(
        path
    ).open(
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
    """
    dataclass config -> JSON-safe tree。
    """
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
            for key, item in value.items()
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
    config,
) -> dict[str, Any]:
    """
    对真正影响 canonical preprocessing 的静态资产做 SHA256。
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


# =============================================================================
# 9. Output-root safety
# =============================================================================

def prepare_output_root(
    *,
    output_root: Path,
    overwrite: bool,
) -> None:
    """
    默认绝不覆盖已有 derived dataset。
    """
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                "Output already exists. "
                "Use --overwrite only if you explicitly want to replace it: "
                f"{output_root}"
            )

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )


# =============================================================================
# 10. Main
# =============================================================================

def main() -> None:
    args = parse_args()

    start_wall = time.time()

    repo_root = Path(
        "."
    ).resolve()

    legacy_repo = (
        args.legacy_repo
        if args.legacy_repo.is_absolute()
        else (
            repo_root
            / args.legacy_repo
        )
    ).resolve()

    dataset_path = (
        args.dataset
        if args.dataset.is_absolute()
        else (
            legacy_repo
            / args.dataset
        )
    ).resolve()

    output_root = (
        args.output
        if args.output.is_absolute()
        else (
            repo_root
            / args.output
        )
    ).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            dataset_path
        )

    if args.shard_size <= 0:
        raise ValueError(
            "--shard-size must be > 0"
        )

    if args.num_points <= 0:
        raise ValueError(
            "--num-points must be > 0"
        )

    camera_roles = tuple(
        token.strip()
        for token in args.cameras.split(
            ","
        )
        if token.strip()
    )

    if not camera_roles:
        raise ValueError(
            "--cameras cannot be empty"
        )

    prepare_output_root(
        output_root=output_root,
        overwrite=args.overwrite,
    )

    Dataset, video_frames = (
        import_legacy_dataset(
            legacy_repo
        )
    )

    dataset = Dataset(
        dataset_path,
        args.episode,
    )

    print(
        "[1/5] Reading states / timestamps..."
    )

    state_index = build_state_index(
        dataset
    )

    available_frames = sorted(
        state_index
    )

    selected_frames = select_frames(
        text=args.frames,
        available_frames=available_frames,
        stride=args.frame_stride,
    )

    if not selected_frames:
        raise RuntimeError(
            "No frames selected"
        )

    print(
        "source frames:",
        len(
            available_frames
        ),
    )

    print(
        "selected frames:",
        len(
            selected_frames
        ),
        "first=",
        selected_frames[
            0
        ],
        "last=",
        selected_frames[
            -1
        ],
    )

    # -------------------------------------------------------------------------
    # 10.1 Canonical preprocessor
    # -------------------------------------------------------------------------
    print(
        "[2/5] Initializing canonical SpatialPreprocessor..."
    )

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
    # 10.2 Provenance
    # -------------------------------------------------------------------------
    asset_provenance = (
        build_asset_provenance(
            repo_root=repo_root,
            config=config,
        )
    )

    # -------------------------------------------------------------------------
    # 10.3 Sharded export
    # -------------------------------------------------------------------------
    print(
        "[3/5] Exporting spatial shards..."
    )

    shard_entries = []
    static_ids = None

    total = len(
        selected_frames
    )

    for shard_index, start in enumerate(
        range(
            0,
            total,
            args.shard_size,
        )
    ):
        shard_frames = selected_frames[
            start:
            start
            + args.shard_size
        ]

        shard_start_time = time.perf_counter()

        raw_frames = load_raw_shard(
            dataset=dataset,
            video_frames=video_frames,
            frame_ids=shard_frames,
            camera_roles=camera_roles,
            state_index=state_index,
        )

        arrays, static_ids = (
            build_derived_shard(
                preprocessor=preprocessor,
                frame_ids=shard_frames,
                raw_frames=raw_frames,
                static_ids=static_ids,
            )
        )

        entry = write_shard_atomic(
            output_root=output_root,
            shard_index=shard_index,
            arrays=arrays,
        )

        elapsed = (
            time.perf_counter()
            - shard_start_time
        )

        entry[
            "wall_seconds"
        ] = float(
            elapsed
        )

        shard_entries.append(
            entry
        )

        processed = min(
            start
            + len(
                shard_frames
            ),
            total,
        )

        print(
            f"  shard {shard_index:06d}: "
            f"{len(shard_frames)} frames | "
            f"{elapsed:.2f}s | "
            f"progress {processed}/{total}"
        )

    if static_ids is None:
        raise RuntimeError(
            "No observations were exported"
        )

    # -------------------------------------------------------------------------
    # 10.4 Static topology
    # -------------------------------------------------------------------------
    print(
        "[4/5] Writing static tactile topology..."
    )

    static_manifest = (
        write_static_ids(
            output_root=output_root,
            static_ids=static_ids,
        )
    )

    # -------------------------------------------------------------------------
    # 10.5 Final manifest
    # -------------------------------------------------------------------------
    total_wall = (
        time.time()
        - start_wall
    )

    manifest = {
        "derived_schema_version": 1,
        "status": "complete",
        "source": {
            "legacy_repo": str(
                legacy_repo
            ),
            "dataset": str(
                dataset_path
            ),
            "episode": int(
                args.episode
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
            "shards": shard_entries,
        },
        "preprocess_config": jsonable(
            asdict(
                config
            )
        ),
        "asset_provenance": (
            asset_provenance
        ),
        "runtime": {
            "total_wall_seconds": float(
                total_wall
            ),
            "average_wall_ms_per_frame_including_io_decode_write": float(
                total_wall
                / len(
                    selected_frames
                )
                * 1000.0
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
        "[5/5] COMPLETE"
    )

    print(
        "output:",
        output_root,
    )

    print(
        "manifest:",
        manifest_path,
    )

    print(
        "frames:",
        len(
            selected_frames
        ),
    )

    print(
        "shards:",
        len(
            shard_entries
        ),
    )

    print(
        "wall time [s]:",
        f"{total_wall:.2f}",
    )

    print(
        "wall ms/frame incl IO+decode+write:",
        f"{manifest['runtime']['average_wall_ms_per_frame_including_io_decode_write']:.2f}",
    )


if __name__ == "__main__":
    main()
