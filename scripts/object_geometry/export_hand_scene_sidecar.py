#!/usr/bin/env python3
"""Build the independent causal ``object_geometry/v1_hand_scene`` sidecar.

This tool does not estimate meshes, object poses, contact, or grasp state.  It
copies the metric base_link object clouds from a completed calibrated geometry
sidecar and joins a producer-owned, per-frame causal state record.  That keeps
the output of an external mesh-scale/6D-registration pipeline explicit while
preventing a training exporter from silently fabricating relationship labels.

State input layout (all paths are relative to ``--state-root``)::

    e023/f000120.npz

Required arrays are documented in ``docs/object_geometry/hand_scene_v0.md``.
In particular ``causal_max_source_frame`` must be no greater than the frame it
labels; it makes accidental future-frame smoothing visible at this boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.object_geometry_dataset.derived import FUNNEL_MASK_SIZE
from openpi.object_geometry_dataset.derived import HAND_SCENE_FIELDS
from openpi.object_geometry_dataset.derived import POINT_COUNT
from openpi.object_geometry_dataset.derived import TACTILE_POINT_COUNT
from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

STATE_REQUIRED = (
    "causal_max_source_frame",
    "manipulated_relation_valid",
    "target_relation_valid",
    "manipulated_geometry_quality",
    "target_geometry_quality",
    "grasp_reference_force_norm",
    "grasp_short_delta_force_norm",
    "grasp_contact_valid",
    "grasp_quality",
)


def _state_path(root: Path, episode_index: int, frame_index: int) -> Path:
    return root / f"e{episode_index:03d}" / f"f{frame_index:06d}.npz"


def _scalar(record: dict[str, np.ndarray], name: str, dtype: Any) -> np.ndarray:
    value = np.asarray(record[name], dtype=dtype)
    if value.shape != ():
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return value


def _quality(record: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = _scalar(record, name, np.float32)
    if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _vector(record: dict[str, np.ndarray], name: str, dtype: Any) -> np.ndarray:
    value = np.asarray(record[name], dtype=dtype)
    if value.shape != (TACTILE_POINT_COUNT,):
        raise ValueError(f"{name} must have shape ({TACTILE_POINT_COUNT},), got {value.shape}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def _resize_nearest(mask: np.ndarray) -> np.ndarray:
    """Resize one current-frame mask without a CV dependency or interpolation."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"funnel_mask must be rank 2, got {mask.shape}")
    if mask.shape == (FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE):
        return mask
    y = np.floor(np.arange(FUNNEL_MASK_SIZE) * mask.shape[0] / FUNNEL_MASK_SIZE).astype(np.int64)
    x = np.floor(np.arange(FUNNEL_MASK_SIZE) * mask.shape[1] / FUNNEL_MASK_SIZE).astype(np.int64)
    return mask[np.ix_(y, x)]


def _load_state(path: Path, *, frame_index: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as raw:
        record = {name: raw[name].copy() for name in raw.files}
    missing = [name for name in STATE_REQUIRED if name not in record]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")
    max_source = int(_scalar(record, "causal_max_source_frame", np.int64))
    if max_source > frame_index:
        raise ValueError(f"{path}: causal_max_source_frame={max_source} is after labelled frame {frame_index}")
    output = {
        "manipulated_relation_valid": _scalar(record, "manipulated_relation_valid", bool),
        "target_relation_valid": _scalar(record, "target_relation_valid", bool),
        "manipulated_geometry_quality": _quality(record, "manipulated_geometry_quality"),
        "target_geometry_quality": _quality(record, "target_geometry_quality"),
        "grasp_reference_force_norm": _vector(record, "grasp_reference_force_norm", np.float32),
        "grasp_short_delta_force_norm": _vector(record, "grasp_short_delta_force_norm", np.float32),
        "grasp_contact_valid": _vector(record, "grasp_contact_valid", bool),
        "grasp_quality": _quality(record, "grasp_quality"),
    }
    if "funnel_mask" not in record:
        output.update(
            funnel_mask=np.zeros((FUNNEL_MASK_SIZE, FUNNEL_MASK_SIZE), dtype=bool),
            funnel_mask_valid=np.zeros((), dtype=bool),
            funnel_mask_quality=np.asarray(0.0, dtype=np.float32),
            funnel_mask_hand_offset_xy=np.zeros((2,), dtype=np.float32),
        )
    else:
        required_mask_fields = {"funnel_mask_valid", "funnel_mask_quality", "funnel_mask_hand_offset_xy"}
        if not required_mask_fields <= record.keys():
            raise ValueError(f"{path}: funnel_mask requires {sorted(required_mask_fields)}")
        hand_offset = np.asarray(record["funnel_mask_hand_offset_xy"], dtype=np.float32)
        if hand_offset.shape != (2,) or not np.isfinite(hand_offset).all():
            raise ValueError(f"{path}: funnel_mask_hand_offset_xy must be finite with shape (2,)")
        output.update(
            funnel_mask=_resize_nearest(record["funnel_mask"]),
            funnel_mask_valid=_scalar(record, "funnel_mask_valid", bool),
            funnel_mask_quality=_quality(record, "funnel_mask_quality"),
            funnel_mask_hand_offset_xy=hand_offset,
        )
    assert tuple(output) == HAND_SCENE_FIELDS
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_episodes(value: str | None, source: ObjectGeometryDerivedDataset) -> set[int]:
    available = {episode for episode, _ in source._records}  # noqa: SLF001 - exporter owns this dataset boundary.
    if value is None:
        return available
    output: set[int] = set()
    for token in value.split(","):
        start_stop = token.split("-", 1)
        if len(start_stop) == 2:
            start, stop = map(int, start_stop)
            output.update(range(start, stop + 1))
        else:
            output.add(int(token))
    unknown = output - available
    if unknown:
        raise ValueError(f"Requested episodes absent from source sidecar: {sorted(unknown)}")
    return output


def build_sidecar(
    *,
    source_dataset_root: Path,
    source_version: str,
    state_root: Path,
    output_dataset_root: Path,
    version: str,
    episodes: str | None,
    generation_commit: str,
) -> dict[str, Any]:
    """Build and return the manifest; exposed for CPU-only contract tests."""
    source = ObjectGeometryDerivedDataset(source_dataset_root, version=source_version)
    if source.has_hand_scene:
        raise ValueError("Source is already a Hand-Scene sidecar; use its original external state records instead")
    if not isinstance(source.manifest.get("calibration"), dict):
        raise ValueError("Source object geometry sidecar must carry formal calibration provenance")
    root = output_dataset_root / "object_geometry" / version
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing sidecar: {root}")
    selected_episodes = _parse_episodes(episodes, source)
    staged: dict[int, list[dict[str, np.ndarray]]] = {}
    for episode, frame in source._records:  # noqa: SLF001 - deterministic manifest order.
        if episode not in selected_episodes:
            continue
        sample = source.get(episode_index=episode, frame_index=frame)
        state = _load_state(_state_path(state_root, episode, frame), frame_index=frame)
        staged.setdefault(episode, []).append(
            {
                "episode_index": np.asarray(episode, dtype=np.int64),
                "frame_index": np.asarray(frame, dtype=np.int64),
                **{
                    field: getattr(sample, field)
                    for name in ("manipulated", "target")
                    for field in (
                        f"{name}_xyz_m",
                        f"{name}_mask",
                        f"{name}_source",
                        f"{name}_age",
                        f"{name}_confidence",
                    )
                },
                **state,
            }
        )
    if not staged:
        raise ValueError("No records selected for Hand-Scene sidecar")
    root.mkdir(parents=True, exist_ok=False)
    records = []
    fields = (
        "episode_index",
        "frame_index",
        *(
            field
            for name in ("manipulated", "target")
            for field in (f"{name}_xyz_m", f"{name}_mask", f"{name}_source", f"{name}_age", f"{name}_confidence")
        ),
        *HAND_SCENE_FIELDS,
    )
    for episode, episode_records in staged.items():
        shard_name = f"shard_e{episode:03d}"
        shard = root / shard_name
        shard.mkdir()
        for field in fields:
            np.save(shard / f"{field}.npy", np.asarray([record[field] for record in episode_records]))
        records.extend(
            {"episode_index": episode, "frame_index": int(record["frame_index"]), "shard": shard_name, "row": row}
            for row, record in enumerate(episode_records)
        )
    source_manifest = source.root / "manifest.json"
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "generation_commit": generation_commit,
        "source_object_geometry": {
            "dataset_root": str(source_dataset_root),
            "version": source_version,
            "manifest_sha256": _sha256(source_manifest),
        },
        "calibration": source.manifest["calibration"],
        "object_geometry_contract": {
            "version": version,
            "point_count": POINT_COUNT,
            "units": "meters",
            "coordinate_frame": "base_link",
            "source_semantics": source.manifest.get("object_geometry_contract", {}).get("source_semantics"),
        },
        "hand_scene_contract": {
            "version": "v1_hand_scene",
            "tactile_point_count": TACTILE_POINT_COUNT,
            "funnel_mask_size": FUNNEL_MASK_SIZE,
            "relation_valid_semantics": (
                "current-frame causal relation state; manipulated=false after a lift invalidates Hand-Funnel"
            ),
            "geometry_quality_semantics": "[0,1] external registration/geometry quality; zero invalidates relation token",
            "grasp_semantics": (
                "current force comes from spatial tactile input; reference and short delta are causal taxel force magnitudes"
            ),
            "mask_semantics": "current-frame downsampled SAM/object mask; invalid or missing uses zero mask and false valid flag",
            "future_frame_policy": "causal_max_source_frame <= labelled frame is required for every record",
        },
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--source-version", default="v1_front_candidate")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-dataset-root", type=Path, required=True)
    parser.add_argument("--version", default="v1_hand_scene")
    parser.add_argument("--episodes", help="Comma/range selection, e.g. 23 or 0-69; default all source episodes")
    parser.add_argument("--generation-commit", default="unknown")
    args = parser.parse_args()
    if args.version != "v1_hand_scene":
        parser.error("Hand-Scene V0 only writes --version v1_hand_scene")
    manifest = build_sidecar(
        source_dataset_root=args.source_dataset_root.expanduser().resolve(),
        source_version=args.source_version,
        state_root=args.state_root.expanduser().resolve(),
        output_dataset_root=args.output_dataset_root.expanduser().resolve(),
        version=args.version,
        episodes=args.episodes,
        generation_commit=args.generation_commit,
    )
    print(json.dumps({"status": manifest["status"], "records": len(manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()
