"""
OpenPI 3D + tactile：原始机器人数据集读取层
（openpi.spatial_dataset.source）

作用
----
读取当前项目中的 LeRobot-v2-like 数据集：

    dataset_root/
    ├── data/
    ├── images/
    ├── meta/
    └── videos/

并向上层提供稳定、最小的 raw-data 接口：

    - episode discovery
    - observation.state
    - timestamp / frame_index
    - RGB-D depth
    - RGB video frame stream

本模块只负责“读取原始数据”。

它明确不负责：
    - 相机标定
    - depth -> point cloud
    - FK
    - tactile geometry
    - voxelization
    - sampling
    - SpatialObservation
    - derived spatial 写盘

这些仍然由：
    openpi.spatial.SpatialPreprocessor
负责。

为什么单独做这一层
------------------
旧工程的 scripts 曾通过修改 Python 搜索路径，
再动态导入另一个 repo 中的数据集 reader。

现在这个跨仓库运行时依赖已经不再需要。

现在数据已经迁移到当前仓库，因此这个依赖没有继续存在的理由。

新的数据流是：

    data/<dataset>/
        ↓
    RawSpatialDataset
        ↓
    SpatialPreprocessor
        ↓
    data/<dataset>/spatial/v1

这样当前仓库可以完全独立运行。

与旧 reader 相比的重要变化
--------------------------
1. camera role 不再硬编码 front / left。
   depth / RGB role 从调用参数决定。

2. RGB 提供流式接口：
       iter_video_frames()

   一个 episode / 一个 camera 可以只顺序 decode 一次。

   旧的 video_frames(path, selected) 如果被每个 shard 调用一次，
   会重复从头到尾 decode 同一个视频；全量 derived export 时代价很高。

3. parquet rows 也提供流式接口：
       iter_rows()

4. 不依赖 LeRobot Python package。
   只依赖：
       pyarrow
       av
       numpy

输入
----
root:
    数据集根目录，例如：
        data/press_0828_17

episode:
    episode index，例如：
        0

主要输出
--------
RawSpatialDataset.states():
    {
        "observation.state": np.ndarray,
        "timestamp": np.ndarray,
        "frame_index": np.ndarray,
        "episode_index": np.ndarray,
        "index": np.ndarray,
    }

RawSpatialDataset.iter_rows(...):
    逐 frame 返回：
        observation.state
        timestamp
        frame_index
        episode_index
        index
        requested depth fields

RawSpatialDataset.iter_video_frames(role, ...):
    逐 frame 返回：

        VideoFrame(
            frame_index=...,
            rgb=uint8[H,W,3],
        )

基本使用
--------
    from openpi.spatial_dataset.source import RawSpatialDataset

    ds = RawSpatialDataset(
        "data/press_0828_17",
        episode=0,
    )

    print(ds.camera_roles)
    print(ds.depth_roles)

    states = ds.states()

    for row in ds.iter_rows(
        selected=[0, 50, 100],
        depth_roles=("front", "left"),
    ):
        ...

    for frame in ds.iter_video_frames(
        "front",
        selected=[0, 50, 100],
    ):
        print(
            frame.frame_index,
            frame.rgb.shape,
        )

注意
----
- frame_index 被视为 episode 内 video ordinal。
- RGB 解码固定输出 rgb24。
- depth 当前数据实际为 Arrow list<list<uint16>>。
- 本模块不会把 depth 转换成米；depth_scale 属于 calibration/preprocess。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import av
import numpy as np
import pyarrow.parquet as pq


# =============================================================================
# 1. Lightweight public data structures
# =============================================================================

@dataclass(frozen=True, slots=True)
class VideoFrame:
    """
    一个 RGB video frame。

    frame_index:
        episode 内视频 ordinal。

    rgb:
        [H,W,3] uint8，固定 RGB 顺序。
    """

    frame_index: int
    rgb: np.ndarray


# =============================================================================
# 2. Raw dataset reader
# =============================================================================

class RawSpatialDataset:
    """
    LeRobot-v2-like 原始 episode reader。

    每个实例只绑定一个 episode。
    """

    _BASE_COLUMNS = (
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        episode: int = 0,
    ) -> None:
        self.root = (
            Path(root)
            .expanduser()
            .resolve()
        )

        if not self.root.is_dir():
            raise FileNotFoundError(
                self.root
            )

        self.episode = int(
            episode
        )

        self.info_path = (
            self.root
            / "meta"
            / "info.json"
        )

        if not self.info_path.is_file():
            raise FileNotFoundError(
                self.info_path
            )

        self.info = json.loads(
            self.info_path.read_text(
                encoding="utf-8",
            )
        )

        self._validate_info()

        chunk_size = int(
            self.info.get(
                "chunks_size",
                1000,
            )
        )

        if chunk_size <= 0:
            raise ValueError(
                "meta/info.json: chunks_size must be > 0"
            )

        self.format_args = {
            "episode_index": (
                self.episode
            ),
            "episode_chunk": (
                self.episode
                // chunk_size
            ),
        }

        self.parquet_path = (
            self.root
            / self.info[
                "data_path"
            ].format(
                **self.format_args
            )
        )

        if not self.parquet_path.is_file():
            raise FileNotFoundError(
                self.parquet_path
            )

        self._parquet_file = (
            pq.ParquetFile(
                self.parquet_path
            )
        )

        self.state_names = tuple(
            self.info[
                "features"
            ][
                "observation.state"
            ][
                "names"
            ]
        )

        if len(
            self.state_names
        ) != len(
            set(
                self.state_names
            )
        ):
            raise ValueError(
                "Duplicate observation.state names"
            )

        self.state_name_to_index = {
            name: index
            for index, name
            in enumerate(
                self.state_names
            )
        }

        self.camera_roles = tuple(
            sorted(
                self._discover_roles(
                    prefix=(
                        "observation.images.cam_"
                    )
                )
            )
        )

        self.depth_roles = tuple(
            sorted(
                self._discover_roles(
                    prefix=(
                        "observation.depths.cam_"
                    )
                )
            )
        )

    # =========================================================================
    # 3. Metadata validation / discovery
    # =========================================================================

    def _validate_info(
        self,
    ) -> None:
        required = (
            "features",
            "data_path",
            "video_path",
        )

        missing = [
            key
            for key in required
            if key not in self.info
        ]

        if missing:
            raise KeyError(
                "meta/info.json missing required fields: "
                f"{missing}"
            )

        features = self.info[
            "features"
        ]

        if (
            "observation.state"
            not in features
        ):
            raise KeyError(
                "meta/info.json missing feature "
                "'observation.state'"
            )

        if (
            "names"
            not in features[
                "observation.state"
            ]
        ):
            raise KeyError(
                "observation.state feature "
                "does not contain 'names'"
            )

    def _discover_roles(
        self,
        *,
        prefix: str,
    ) -> list[str]:
        """
        从 info.features key 中发现 camera role。
        """
        output = []

        for feature_name in self.info[
            "features"
        ]:
            if feature_name.startswith(
                prefix
            ):
                role = feature_name[
                    len(
                        prefix
                    ):
                ]

                if role:
                    output.append(
                        role
                    )

        return output

    # =========================================================================
    # 4. Dataset-level episode discovery
    # =========================================================================

    @staticmethod
    def discover_episode_ids(
        root: str | Path,
    ) -> tuple[int, ...]:
        """
        从 meta/episodes.jsonl 读取 episode ids。

        不通过 chunk 文件名猜测。
        """
        root_path = (
            Path(root)
            .expanduser()
            .resolve()
        )

        episodes_path = (
            root_path
            / "meta"
            / "episodes.jsonl"
        )

        if not episodes_path.is_file():
            raise FileNotFoundError(
                episodes_path
            )

        episode_ids = []

        with episodes_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                text = line.strip()

                if not text:
                    continue

                item = json.loads(
                    text
                )

                if (
                    "episode_index"
                    in item
                ):
                    episode_id = int(
                        item[
                            "episode_index"
                        ]
                    )

                elif "episode" in item:
                    episode_id = int(
                        item[
                            "episode"
                        ]
                    )

                else:
                    raise KeyError(
                        f"{episodes_path}:"
                        f"{line_number}: "
                        "missing episode_index / episode"
                    )

                episode_ids.append(
                    episode_id
                )

        if not episode_ids:
            raise ValueError(
                f"No episodes found in {episodes_path}"
            )

        if len(
            episode_ids
        ) != len(
            set(
                episode_ids
            )
        ):
            raise ValueError(
                "Duplicate episode ids in "
                "meta/episodes.jsonl"
            )

        return tuple(
            sorted(
                episode_ids
            )
        )

    # =========================================================================
    # 5. Paths
    # =========================================================================

    def video_path(
        self,
        role: str,
    ) -> Path:
        """
        返回某 camera role 的 episode video path。
        """
        if role not in self.camera_roles:
            raise KeyError(
                f"Unknown RGB camera role {role!r}. "
                f"Available: {self.camera_roles}"
            )

        path = (
            self.root
            / self.info[
                "video_path"
            ].format(
                **self.format_args,
                video_key=(
                    "observation.images."
                    f"cam_{role}"
                ),
            )
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        return path

    # =========================================================================
    # 6. State table
    # =========================================================================

    def states(
        self,
    ) -> dict[str, np.ndarray]:
        """
        一次读取 episode 的 lightweight state table。

        这里不读取 depth。
        """
        table = pq.read_table(
            self.parquet_path,
            columns=list(
                self._BASE_COLUMNS
            ),
        )

        return {
            key: np.asarray(
                table[
                    key
                ].to_pylist()
            )
            for key in self._BASE_COLUMNS
        }

    # =========================================================================
    # 7. Parquet row stream
    # =========================================================================

    def iter_rows(
        self,
        *,
        selected: Iterable[int] | None = None,
        depth_roles: Sequence[str] = (),
    ) -> Iterator[dict[str, object]]:
        """
        顺序流式读取 parquet rows。

        selected:
            None:
                返回全部 frame。

            iterable:
                只返回指定 frame_index。

        depth_roles:
            需要加载哪些 depth，例如：
                ("front", "left")

        重要：
            这里不会读取未请求的 depth camera。
        """
        selected_set = (
            None
            if selected is None
            else {
                int(
                    value
                )
                for value in selected
            }
        )

        depth_roles = tuple(
            str(
                role
            )
            for role in depth_roles
        )

        unknown_roles = sorted(
            set(
                depth_roles
            )
            - set(
                self.depth_roles
            )
        )

        if unknown_roles:
            raise KeyError(
                "Unknown depth roles: "
                f"{unknown_roles}. "
                f"Available: {self.depth_roles}"
            )

        depth_columns = tuple(
            f"observation.depths.cam_{role}"
            for role in depth_roles
        )

        columns = (
            *self._BASE_COLUMNS,
            *depth_columns,
        )

        found = set()

        # batch_size=1 是有意保留的：
        # 当前 depth 是 Arrow nested list。
        # 单行 batch 可以限制 decompressed Arrow working set，
        # 避免一次构造大量 Python nested-list / depth 临时对象。
        for batch in self._parquet_file.iter_batches(
            batch_size=1,
            columns=list(
                columns
            ),
        ):
            frame_column_index = (
                batch.schema.get_field_index(
                    "frame_index"
                )
            )

            frame_id = int(
                batch.column(
                    frame_column_index
                )[
                    0
                ].as_py()
            )

            if (
                selected_set is not None
                and frame_id
                not in selected_set
            ):
                continue

            row = {}

            for key in columns:
                field_index = (
                    batch.schema.get_field_index(
                        key
                    )
                )

                scalar = batch.column(
                    field_index
                )[
                    0
                ]

                if key.startswith(
                    "observation.depths."
                ):
                    row[
                        key
                    ] = self._decode_depth_scalar(
                        key=key,
                        frame_id=frame_id,
                        scalar=scalar,
                    )

                else:
                    row[
                        key
                    ] = scalar.as_py()

            found.add(
                frame_id
            )

            yield row

        if selected_set is not None:
            missing = sorted(
                selected_set
                - found
            )

            if missing:
                raise KeyError(
                    "Parquet missing requested frames: "
                    f"{missing}"
                )

    def _decode_depth_scalar(
        self,
        *,
        key: str,
        frame_id: int,
        scalar,
    ) -> np.ndarray:
        """
        当前数据集 depth schema：

            Arrow list<list<uint16>>

        输出：

            [H,W] uint16 ndarray

        不做 depth-scale 转换。
        """
        if not scalar.is_valid:
            raise ValueError(
                f"Frame {frame_id}: missing depth {key}"
            )

        feature = self.info[
            "features"
        ][
            key
        ]

        shape = tuple(
            int(
                value
            )
            for value in feature[
                "shape"
            ]
        )

        array = scalar.values

        # 兼容 list<list<uint16>>：
        # flatten 一层后再转连续 NumPy。
        flat = (
            array
            .flatten()
            .to_numpy(
                zero_copy_only=False
            )
        )

        expected_size = int(
            np.prod(
                shape
            )
        )

        if (
            flat.dtype
            != np.uint16
        ):
            raise ValueError(
                f"Frame {frame_id}: "
                f"depth {key} dtype is "
                f"{flat.dtype}, expected uint16"
            )

        if (
            flat.size
            != expected_size
        ):
            raise ValueError(
                f"Frame {frame_id}: "
                f"depth {key} size "
                f"{flat.size}, expected "
                f"{expected_size} for shape {shape}"
            )

        return (
            flat
            .reshape(
                shape
            )
            .copy()
        )

    # =========================================================================
    # 8. RGB video stream
    # =========================================================================

    def iter_video_frames(
        self,
        role: str,
        *,
        selected: Iterable[int] | None = None,
    ) -> Iterator[VideoFrame]:
        """
        单次、顺序 decode 某 camera video。

        这是 full derived export 推荐使用的接口：

            每个 episode / camera 只 decode 一遍。

        selected=None:
            yield 全部 frames。

        selected={0,50,100}:
            video 仍然只顺序 decode 一次，
            但只把指定 ordinal 转成 RGB ndarray 并 yield。

        注意：
            frame_index 当前按 decoded frame ordinal 对齐。
        """
        selected_set = (
            None
            if selected is None
            else {
                int(
                    value
                )
                for value in selected
            }
        )

        if (
            selected_set is not None
            and any(
                value < 0
                for value in selected_set
            )
        ):
            raise ValueError(
                "Video frame indices must be >= 0"
            )

        found = set()

        path = self.video_path(
            role
        )

        with av.open(
            str(
                path
            )
        ) as container:
            stream = (
                container
                .streams
                .video[
                    0
                ]
            )

            for frame_index, frame in enumerate(
                container.decode(
                    stream
                )
            ):
                if (
                    selected_set is not None
                    and frame_index
                    not in selected_set
                ):
                    continue

                rgb = frame.to_ndarray(
                    format="rgb24"
                )

                if (
                    rgb.ndim != 3
                    or rgb.shape[
                        -1
                    ]
                    != 3
                    or rgb.dtype
                    != np.uint8
                ):
                    raise ValueError(
                        f"Unexpected decoded RGB shape/dtype: "
                        f"{rgb.shape} {rgb.dtype}"
                    )

                found.add(
                    frame_index
                )

                yield VideoFrame(
                    frame_index=(
                        frame_index
                    ),
                    rgb=rgb,
                )

        if selected_set is not None:
            missing = sorted(
                selected_set
                - found
            )

            if missing:
                raise KeyError(
                    f"Video {role!r} missing "
                    f"requested frames: {missing}"
                )

    # =========================================================================
    # 9. Small convenience helper for QA / smoke tests
    # =========================================================================

    def load_video_frames(
        self,
        role: str,
        *,
        selected: Iterable[int],
    ) -> dict[int, np.ndarray]:
        """
        少量抽帧 QA 的 convenience API。

        正式全量 exporter 不应该反复调用它；
        exporter 应直接使用 iter_video_frames()。
        """
        return {
            frame.frame_index: (
                frame.rgb
            )
            for frame in self.iter_video_frames(
                role,
                selected=selected,
            )
        }
