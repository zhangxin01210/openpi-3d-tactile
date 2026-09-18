"""
OpenPI 3D + tactile：Derived Spatial Dataset Reader
（openpi.spatial_dataset.derived）

作用
----
读取已经离线生成的：

    <dataset>/spatial/<version>/

并以 memory-mapped 方式提供稳定的 spatial sample。

典型数据流：

    raw RGB-D + state
        ↓
    SpatialPreprocessor
        ↓
    spatial/v1  （离线生成，已经完成）
        ↓
    SpatialDerivedDataset
        ↓
    DerivedSpatialSample
        ↓
    后续 model transform / point encoder

本模块明确不负责
----------------
- RGB-D 解码
- depth -> point cloud
- calibration
- FK
- tactile geometry
- voxelization
- sampling
- point encoder
- tensor 拼接 / normalization
- π0 输入格式

这些边界保持分离：

    source.py
        = raw dataset reader

    openpi.spatial.*
        = canonical spatial preprocessing

    derived.py
        = 已生成 spatial/v1 的读取层

    model transform（后续）
        = spatial representation -> model tensor / tokens

为什么使用 mmap
---------------
derived spatial 数据包含大量固定 shape array。

例如：

    visual_xyz_m
        [128,4096,3] float32

    tactile_xyz_m
        [128,600,3] float32

如果训练时把整个数据集一次性加载到 RAM，没有必要。

本 reader 使用：

    np.load(..., mmap_mode="r")

只有真正访问某个 frame 时，操作系统才按页读取相应数据。

索引方式
--------
支持三种常用方式：

1. episode + frame：

    sample = dataset.get(
        episode_index=0,
        frame_index=213,
    )

2. tuple shortcut：

    sample = dataset[
        (0, 213)
    ]

3. 全局 dataset index：

    sample = dataset[0]
    sample = dataset[-1]

全局顺序严格按照 manifest 中：

    episode 顺序
        -> shard 顺序
            -> shard 内 frame 顺序

这方便后续适配 PyTorch / JAX / OpenPI Dataset。

输出
----
DerivedSpatialSample：

    episode_index: int
    frame_index: int
    timestamp_s: float

    visual_xyz_m:       [Nv,3] float32
    visual_rgb:         [Nv,3] uint8
    visual_rgb_valid:   [Nv] bool

    tactile_xyz_m:      [600,3] float32
    tactile_force_base: [600,3] float32
    tactile_force_norm: [600] float32

    finger_id:          [600] int8
    taxel_id:           [600] int16

注意：
    finger_id / taxel_id 来自 spatial/<version>/static，
    所有 frame 共享，不重复存储。

缓存策略
--------
reader 对 shard array 使用小型 LRU cache。

默认最多保留：

    8 shards

的 mmap 对象。

这不会把 8 个 shard 全部读进 RAM；
只是保留 mmap 映射和文件句柄，避免训练访问相邻样本时反复 np.load。

基本使用
--------
    from openpi.spatial_dataset.derived import SpatialDerivedDataset

    ds = SpatialDerivedDataset(
        "data/press_0828_17",
        version="v1",
    )

    print(len(ds))
    print(ds.episode_ids)

    sample = ds.get(
        episode_index=0,
        frame_index=213,
    )

    print(sample.visual_xyz_m.shape)
    print(sample.tactile_xyz_m.shape)

    sample2 = ds[(0, 213)]
    sample3 = ds[0]

注意
----
- 本 reader 只接受 manifest status == "complete"。
- schema/version mismatch 会尽早报错。
- 返回 array 是只读 mmap view；不要原地修改。
- 如果模型需要 normalization / concatenate / augmentation，
  请在后续 model transform 中实现，不要污染 derived reader。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# 1. Public sample
# =============================================================================

@dataclass(frozen=True, slots=True)
class DerivedSpatialSample:
    """
    一个 frame 的物理 spatial representation。

    这里保持“数据语义”，不提前决定模型 tensor layout。
    """

    episode_index: int
    frame_index: int
    timestamp_s: float

    visual_xyz_m: np.ndarray
    visual_rgb: np.ndarray
    visual_rgb_valid: np.ndarray

    tactile_xyz_m: np.ndarray
    tactile_force_base: np.ndarray
    tactile_force_norm: np.ndarray

    finger_id: np.ndarray
    taxel_id: np.ndarray


# =============================================================================
# 2. Internal shard metadata
# =============================================================================

@dataclass(frozen=True, slots=True)
class _ShardRecord:
    """
    一个 derived shard 的轻量索引信息。
    """

    episode_index: int
    name: str
    root: Path

    count: int
    first_frame: int
    last_frame: int

    global_start: int
    global_stop: int


# =============================================================================
# 3. Reader
# =============================================================================

class SpatialDerivedDataset:
    """
    Memory-mapped spatial/<version> reader。
    """

    _REQUIRED_FIELDS = (
        "frame_index",
        "timestamp_s",
        "visual_xyz_m",
        "visual_rgb",
        "visual_rgb_valid",
        "tactile_xyz_m",
        "tactile_force_base",
        "tactile_force_norm",
    )

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        version: str = "v1",
        max_cached_shards: int = 8,
    ) -> None:
        self.dataset_root = (
            Path(
                dataset_root
            )
            .expanduser()
            .resolve()
        )

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                self.dataset_root
            )

        self.version = str(
            version
        )

        self.root = (
            self.dataset_root
            / "spatial"
            / self.version
        )

        if not self.root.is_dir():
            raise FileNotFoundError(
                self.root
            )

        if max_cached_shards <= 0:
            raise ValueError(
                "max_cached_shards must be > 0"
            )

        self.max_cached_shards = int(
            max_cached_shards
        )

        self.manifest_path = (
            self.root
            / "manifest.json"
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                self.manifest_path
            )

        self.manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8",
            )
        )

        self._validate_manifest()

        self.coordinate_frame = str(
            self.manifest[
                "spatial_contract"
            ][
                "coordinate_frame"
            ]
        )

        self.xyz_unit = str(
            self.manifest[
                "spatial_contract"
            ][
                "xyz_unit"
            ]
        )

        self.force_unit = str(
            self.manifest[
                "spatial_contract"
            ][
                "force_unit"
            ]
        )

        self.visual_points_per_frame = int(
            self.manifest[
                "spatial_contract"
            ][
                "visual_points_per_frame"
            ]
        )

        self.tactile_points_per_frame = int(
            self.manifest[
                "spatial_contract"
            ][
                "tactile_points_per_frame"
            ]
        )

        self.camera_roles = tuple(
            self.manifest[
                "spatial_contract"
            ][
                "camera_roles"
            ]
        )

        self.finger_id = self._load_static_array(
            "finger_id",
            expected_shape=(
                self.tactile_points_per_frame,
            ),
            expected_dtype=np.int8,
        )

        self.taxel_id = self._load_static_array(
            "taxel_id",
            expected_shape=(
                self.tactile_points_per_frame,
            ),
            expected_dtype=np.int16,
        )

        (
            self._shards,
            self._episode_shards,
            self._global_stops,
        ) = self._build_index()

        self.episode_ids = tuple(
            sorted(
                self._episode_shards
            )
        )

        self._length = (
            self._shards[
                -1
            ].global_stop
            if self._shards
            else 0
        )

        manifest_count = int(
            self.manifest[
                "runtime"
            ][
                "total_selected_frames"
            ]
        )

        if self._length != manifest_count:
            raise ValueError(
                "Derived dataset frame count mismatch: "
                f"index={self._length}, "
                f"manifest={manifest_count}"
            )

        # key:
        #   (episode_index, shard_name)
        #
        # value:
        #   {field_name: np.memmap}
        #
        # OrderedDict 用于简单 LRU。
        self._array_cache: OrderedDict[
            tuple[int, str],
            dict[str, np.ndarray],
        ] = OrderedDict()

    # =========================================================================
    # 4. Manifest / static validation
    # =========================================================================

    def _validate_manifest(
        self,
    ) -> None:
        if (
            self.manifest.get(
                "status"
            )
            != "complete"
        ):
            raise ValueError(
                "Spatial derived dataset is not complete"
            )

        if int(
            self.manifest.get(
                "derived_schema_version",
                -1,
            )
        ) != 1:
            raise ValueError(
                "Unsupported derived_schema_version: "
                f"{self.manifest.get('derived_schema_version')}"
            )

        if (
            self.manifest.get(
                "spatial_version"
            )
            != self.version
        ):
            raise ValueError(
                "Spatial version mismatch: "
                f"requested={self.version!r}, "
                f"manifest={self.manifest.get('spatial_version')!r}"
            )

        required_top_level = (
            "source_dataset",
            "spatial_contract",
            "storage",
            "runtime",
        )

        missing = [
            key
            for key in required_top_level
            if key not in self.manifest
        ]

        if missing:
            raise KeyError(
                "Derived manifest missing fields: "
                f"{missing}"
            )

        storage_format = (
            self.manifest[
                "storage"
            ].get(
                "format"
            )
        )

        if storage_format != "npy_shards_v1":
            raise ValueError(
                "Unsupported spatial storage format: "
                f"{storage_format!r}"
            )

    def _load_static_array(
        self,
        name: str,
        *,
        expected_shape: tuple[int, ...],
        expected_dtype: np.dtype | type,
    ) -> np.ndarray:
        static_entry = (
            self.manifest[
                "storage"
            ][
                "static"
            ].get(
                name
            )
        )

        if static_entry is None:
            raise KeyError(
                f"Missing static field {name!r}"
            )

        path = (
            self.root
            / static_entry[
                "file"
            ]
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if tuple(
            array.shape
        ) != expected_shape:
            raise ValueError(
                f"Static {name}: shape "
                f"{array.shape}, expected "
                f"{expected_shape}"
            )

        expected_dtype = np.dtype(
            expected_dtype
        )

        if array.dtype != expected_dtype:
            raise ValueError(
                f"Static {name}: dtype "
                f"{array.dtype}, expected "
                f"{expected_dtype}"
            )

        return array

    # =========================================================================
    # 5. Build lightweight index
    # =========================================================================

    def _build_index(
        self,
    ) -> tuple[
        list[_ShardRecord],
        dict[int, list[_ShardRecord]],
        list[int],
    ]:
        """
        只读取 manifest，建立：

            global index -> shard
            episode -> shards

        不在初始化时 mmap 所有 feature arrays。
        """
        all_shards: list[
            _ShardRecord
        ] = []

        episode_shards: dict[
            int,
            list[_ShardRecord],
        ] = {}

        global_cursor = 0

        episode_entries = (
            self.manifest[
                "storage"
            ][
                "episodes"
            ]
        )

        for episode_entry in episode_entries:
            episode_index = int(
                episode_entry[
                    "episode_index"
                ]
            )

            episode_root = (
                self.root
                / episode_entry[
                    "path"
                ]
            )

            episode_manifest_path = (
                episode_root
                / "manifest.json"
            )

            if not episode_manifest_path.is_file():
                raise FileNotFoundError(
                    episode_manifest_path
                )

            episode_manifest = json.loads(
                episode_manifest_path.read_text(
                    encoding="utf-8",
                )
            )

            manifest_episode_index = int(
                episode_manifest[
                    "episode_index"
                ]
            )

            if (
                manifest_episode_index
                != episode_index
            ):
                raise ValueError(
                    "Episode manifest index mismatch: "
                    f"{manifest_episode_index} "
                    f"vs {episode_index}"
                )

            records = []

            for shard_entry in episode_manifest[
                "shards"
            ]:
                count = int(
                    shard_entry[
                        "count"
                    ]
                )

                if count <= 0:
                    raise ValueError(
                        "Shard count must be > 0"
                    )

                shard_root = (
                    episode_root
                    / shard_entry[
                        "name"
                    ]
                )

                if not shard_root.is_dir():
                    raise FileNotFoundError(
                        shard_root
                    )

                record = _ShardRecord(
                    episode_index=(
                        episode_index
                    ),
                    name=str(
                        shard_entry[
                            "name"
                        ]
                    ),
                    root=shard_root,
                    count=count,
                    first_frame=int(
                        shard_entry[
                            "first_frame"
                        ]
                    ),
                    last_frame=int(
                        shard_entry[
                            "last_frame"
                        ]
                    ),
                    global_start=(
                        global_cursor
                    ),
                    global_stop=(
                        global_cursor
                        + count
                    ),
                )

                global_cursor += count

                records.append(
                    record
                )

                all_shards.append(
                    record
                )

            expected_count = int(
                episode_manifest[
                    "selected_frame_count"
                ]
            )

            actual_count = sum(
                item.count
                for item in records
            )

            if actual_count != expected_count:
                raise ValueError(
                    f"Episode {episode_index}: "
                    f"shard count total "
                    f"{actual_count}, expected "
                    f"{expected_count}"
                )

            episode_shards[
                episode_index
            ] = records

        global_stops = [
            record.global_stop
            for record in all_shards
        ]

        return (
            all_shards,
            episode_shards,
            global_stops,
        )

    # =========================================================================
    # 6. Lazy mmap cache
    # =========================================================================

    def _load_shard_arrays(
        self,
        record: _ShardRecord,
    ) -> dict[str, np.ndarray]:
        key = (
            record.episode_index,
            record.name,
        )

        cached = self._array_cache.get(
            key
        )

        if cached is not None:
            self._array_cache.move_to_end(
                key
            )

            return cached

        arrays = {}

        for field_name in self._REQUIRED_FIELDS:
            path = (
                record.root
                / f"{field_name}.npy"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

            arrays[
                field_name
            ] = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

        self._validate_shard_arrays(
            record=record,
            arrays=arrays,
        )

        self._array_cache[
            key
        ] = arrays

        self._array_cache.move_to_end(
            key
        )

        while (
            len(
                self._array_cache
            )
            > self.max_cached_shards
        ):
            self._array_cache.popitem(
                last=False
            )

        return arrays

    def _validate_shard_arrays(
        self,
        *,
        record: _ShardRecord,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """
        shard 第一次 mmap 时做结构验证。

        不逐元素扫描，不影响 lazy loading。
        """
        count = record.count

        expected = {
            "frame_index": (
                (count,),
                np.dtype(
                    np.int64
                ),
            ),
            "timestamp_s": (
                (count,),
                np.dtype(
                    np.float64
                ),
            ),
            "visual_xyz_m": (
                (
                    count,
                    self.visual_points_per_frame,
                    3,
                ),
                np.dtype(
                    np.float32
                ),
            ),
            "visual_rgb": (
                (
                    count,
                    self.visual_points_per_frame,
                    3,
                ),
                np.dtype(
                    np.uint8
                ),
            ),
            "visual_rgb_valid": (
                (
                    count,
                    self.visual_points_per_frame,
                ),
                np.dtype(
                    np.bool_
                ),
            ),
            "tactile_xyz_m": (
                (
                    count,
                    self.tactile_points_per_frame,
                    3,
                ),
                np.dtype(
                    np.float32
                ),
            ),
            "tactile_force_base": (
                (
                    count,
                    self.tactile_points_per_frame,
                    3,
                ),
                np.dtype(
                    np.float32
                ),
            ),
            "tactile_force_norm": (
                (
                    count,
                    self.tactile_points_per_frame,
                ),
                np.dtype(
                    np.float32
                ),
            ),
        }

        for field_name, (
            expected_shape,
            expected_dtype,
        ) in expected.items():
            array = arrays[
                field_name
            ]

            if tuple(
                array.shape
            ) != expected_shape:
                raise ValueError(
                    f"{record.root}: "
                    f"{field_name} shape "
                    f"{array.shape}, expected "
                    f"{expected_shape}"
                )

            if array.dtype != expected_dtype:
                raise ValueError(
                    f"{record.root}: "
                    f"{field_name} dtype "
                    f"{array.dtype}, expected "
                    f"{expected_dtype}"
                )

        frame_index = arrays[
            "frame_index"
        ]

        if int(
            frame_index[
                0
            ]
        ) != record.first_frame:
            raise ValueError(
                f"{record.root}: first_frame mismatch"
            )

        if int(
            frame_index[
                -1
            ]
        ) != record.last_frame:
            raise ValueError(
                f"{record.root}: last_frame mismatch"
            )

        if (
            frame_index.shape[
                0
            ]
            > 1
            and np.any(
                frame_index[
                    1:
                ]
                <= frame_index[
                    :-1
                ]
            )
        ):
            raise ValueError(
                f"{record.root}: frame_index "
                "must be strictly increasing"
            )

    # =========================================================================
    # 7. Locate sample
    # =========================================================================

    def _locate_global(
        self,
        index: int,
    ) -> tuple[
        _ShardRecord,
        int,
    ]:
        if self._length == 0:
            raise IndexError(
                "Derived dataset is empty"
            )

        if index < 0:
            index += self._length

        if (
            index < 0
            or index >= self._length
        ):
            raise IndexError(
                f"Global index out of range: {index}"
            )

        shard_position = bisect.bisect_right(
            self._global_stops,
            index,
        )

        record = self._shards[
            shard_position
        ]

        local_index = (
            index
            - record.global_start
        )

        return (
            record,
            local_index,
        )

    def _locate_frame(
        self,
        *,
        episode_index: int,
        frame_index: int,
    ) -> tuple[
        _ShardRecord,
        int,
    ]:
        records = self._episode_shards.get(
            int(
                episode_index
            )
        )

        if records is None:
            raise KeyError(
                f"Unknown episode {episode_index}. "
                f"Available: {self.episode_ids}"
            )

        target = int(
            frame_index
        )

        for record in records:
            if not (
                record.first_frame
                <= target
                <= record.last_frame
            ):
                continue

            arrays = self._load_shard_arrays(
                record
            )

            frame_ids = arrays[
                "frame_index"
            ]

            position = int(
                np.searchsorted(
                    frame_ids,
                    target,
                )
            )

            if (
                position
                < frame_ids.shape[
                    0
                ]
                and int(
                    frame_ids[
                        position
                    ]
                )
                == target
            ):
                return (
                    record,
                    position,
                )

        raise KeyError(
            f"Frame {frame_index} not found "
            f"in episode {episode_index}"
        )

    # =========================================================================
    # 8. Materialize one sample view
    # =========================================================================

    def _sample_from_location(
        self,
        *,
        record: _ShardRecord,
        local_index: int,
    ) -> DerivedSpatialSample:
        arrays = self._load_shard_arrays(
            record
        )

        frame_index = int(
            arrays[
                "frame_index"
            ][
                local_index
            ]
        )

        timestamp_s = float(
            arrays[
                "timestamp_s"
            ][
                local_index
            ]
        )

        return DerivedSpatialSample(
            episode_index=(
                record.episode_index
            ),
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            visual_xyz_m=(
                arrays[
                    "visual_xyz_m"
                ][
                    local_index
                ]
            ),
            visual_rgb=(
                arrays[
                    "visual_rgb"
                ][
                    local_index
                ]
            ),
            visual_rgb_valid=(
                arrays[
                    "visual_rgb_valid"
                ][
                    local_index
                ]
            ),
            tactile_xyz_m=(
                arrays[
                    "tactile_xyz_m"
                ][
                    local_index
                ]
            ),
            tactile_force_base=(
                arrays[
                    "tactile_force_base"
                ][
                    local_index
                ]
            ),
            tactile_force_norm=(
                arrays[
                    "tactile_force_norm"
                ][
                    local_index
                ]
            ),
            finger_id=(
                self.finger_id
            ),
            taxel_id=(
                self.taxel_id
            ),
        )

    # =========================================================================
    # 9. Public API
    # =========================================================================

    def __len__(
        self,
    ) -> int:
        return self._length

    def get(
        self,
        *,
        episode_index: int,
        frame_index: int,
    ) -> DerivedSpatialSample:
        """
        按 episode/frame 精确读取。
        """
        record, local_index = (
            self._locate_frame(
                episode_index=episode_index,
                frame_index=frame_index,
            )
        )

        return self._sample_from_location(
            record=record,
            local_index=local_index,
        )

    def get_global(
        self,
        index: int,
    ) -> DerivedSpatialSample:
        """
        按全局 dataset index 读取。
        """
        record, local_index = (
            self._locate_global(
                int(
                    index
                )
            )
        )

        return self._sample_from_location(
            record=record,
            local_index=local_index,
        )

    def __getitem__(
        self,
        key: int | tuple[int, int],
    ) -> DerivedSpatialSample:
        """
        支持：

            ds[0]
            ds[-1]
            ds[(episode_index, frame_index)]
        """
        if isinstance(
            key,
            tuple,
        ):
            if len(
                key
            ) != 2:
                raise IndexError(
                    "Tuple index must be "
                    "(episode_index, frame_index)"
                )

            return self.get(
                episode_index=int(
                    key[
                        0
                    ]
                ),
                frame_index=int(
                    key[
                        1
                    ]
                ),
            )

        return self.get_global(
            int(
                key
            )
        )

    def episode_length(
        self,
        episode_index: int,
    ) -> int:
        """
        返回 derived version 中某 episode 的 frame 数。
        """
        records = self._episode_shards.get(
            int(
                episode_index
            )
        )

        if records is None:
            raise KeyError(
                f"Unknown episode {episode_index}"
            )

        return sum(
            record.count
            for record in records
        )

    def clear_cache(
        self,
    ) -> None:
        """
        清除 mmap shard cache。

        一般训练过程中无需手动调用；
        长时间交互式分析时可用于主动释放引用。
        """
        self._array_cache.clear()

    @property
    def cached_shard_count(
        self,
    ) -> int:
        return len(
            self._array_cache
        )
