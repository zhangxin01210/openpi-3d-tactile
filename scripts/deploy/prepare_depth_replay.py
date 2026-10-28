#!/usr/bin/env python3
"""Audit legacy replay depth against ORIGINAL LeRobot sensor depth; optionally derive aligned depth.

Fail closed: identical HxW shapes are never evidence of alignment. This tool
only accepts front_depth_raw when bitwise identical to the parquet front
depth for the same frame. It then uses the existing calibrated RGBTrack
registration implementation, and leaves all original NPZ files untouched.

Requires the original dataset. If it is unavailable or depth differs, do not
rename arrays or fabricate color-aligned depth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path, required=True, help="Unmodified legacy replay directory")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Original LeRobot dataset, not a generated replay")
    parser.add_argument("--episode", type=int, default=23)
    parser.add_argument("--front-calibration-profile", type=Path,
                        default=Path("configs/ur7e_xhand/front_calibration_candidate.json"))
    parser.add_argument("--output-dir", type=Path, help="Without this: read-only audit of three representative frames")
    parser.add_argument("--max-frames", type=int, default=10, help="Only used for output mode; start at frame 0")
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    rgbtrack_path = str(root / "scripts/object_geometry/pose_tracking")
    if rgbtrack_path not in sys.path:
        sys.path.insert(0, rgbtrack_path)
    from openpi.spatial_dataset.source import RawSpatialDataset
    from rgbtrack_adapter import _load_front_camera_for_depth_registration, _rgb_registered_depth

    original_files = sorted(source_dir.glob("frame_*.npz"))
    if not original_files:
        raise FileNotFoundError(f"No frame_*.npz in {source_dir}")
    all_indices = [int(path.stem.removeprefix("frame_")) for path in original_files]
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Duplicate frame index in replay filenames")
    if args.output_dir is None:
        selected = [0, all_indices[len(all_indices) // 2], all_indices[-1]]
        missing = sorted(set(selected) - set(all_indices))
        if missing:
            raise ValueError(f"Missing required audit frames: {missing}")
        selected_paths = [source_dir / f"frame_{i:06d}.npz" for i in selected]
    else:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        if all_indices[0] != 0:
            raise ValueError("Replay conversion must begin with seed frame 0")
        selected_paths = original_files[:args.max_frames]
        selected = [int(p.stem.removeprefix("frame_")) for p in selected_paths]
        if selected != list(range(len(selected))):
            raise ValueError(f"Output replay must be a contiguous 0-based prefix: {selected}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        if output_dir == source_dir or output_dir == dataset_root:
            raise ValueError("Output cannot overwrite source replay or dataset")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")

    profile = args.front_calibration_profile
    if not profile.is_absolute():
        profile = root / profile
    camera = _load_front_camera_for_depth_registration(root, profile)
    if not np.isfinite(camera.depth_scale_m_per_unit) or camera.depth_scale_m_per_unit <= 0:
        raise ValueError("Invalid front depth_scale_m_per_unit")
    dataset = RawSpatialDataset(dataset_root, episode=args.episode)
    rows = dataset.iter_rows(selected=selected, depth_roles=("front",))
    audit_rows = []

    for path, row in zip(selected_paths, rows, strict=True):
        frame = int(path.stem.removeprefix("frame_"))
        if int(row["frame_index"]) != frame:
            raise ValueError(f"Parquet/replay frame mismatch at {path}: {row['frame_index']} != {frame}")
        with np.load(path, allow_pickle=False) as packet:
            if "front_depth_raw" not in packet.files:
                raise KeyError(f"{path}: legacy front_depth_raw absent")
            source = {name: np.asarray(packet[name]) for name in packet.files}
        if int(np.asarray(source["frame_index"]).reshape(())) != frame:
            raise ValueError(f"NPZ index mismatch: {path}")
        old_depth = source["front_depth_raw"]
        original_depth = np.asarray(row["observation.depths.cam_front"])
        if (old_depth.shape != original_depth.shape or old_depth.dtype != original_depth.dtype
                or not np.array_equal(old_depth, original_depth)):
            raise ValueError(
                f"{path}: front_depth_raw is NOT identical to original parquet depth. "
                "Cannot assume it is native. Stop and inspect original replay producer."
            )
        original_state = np.asarray(row["observation.state"], dtype=np.float32)
        state = np.asarray(source["robot_state"], dtype=np.float32)
        if original_state.shape != state.shape or not np.array_equal(original_state, state):
            raise ValueError(f"{path}: robot_state differs from original dataset frame {frame}")

        rgb = np.asarray(source["front_rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"{path}: invalid RGB")
        registered_m = _rgb_registered_depth(original_depth, camera, rgb.shape[:2])
        if registered_m.shape != rgb.shape[:2] or not np.isfinite(registered_m).all():
            raise ValueError(f"{path}: registered depth invalid")
        if np.any(registered_m < 0):
            raise ValueError(f"{path}: registered depth contains negative values")
        # The existing online perception multiplies this field by depth_scale.
        # Undo that scale here; retain float32 to avoid extra integer quantization.
        registered_sensor_units = (
            registered_m / np.float32(camera.depth_scale_m_per_unit)
        ).astype(np.float32)
        row_audit = {
            "frame_index": frame,
            "native_shape": list(original_depth.shape),
            "rgb_shape": list(rgb.shape),
            "native_equals_parquet": True,
            "state_equals_parquet": True,
            "native_nonzero": int(np.count_nonzero(original_depth)),
            "aligned_nonzero": int(np.count_nonzero(registered_sensor_units)),
            "aligned_depth_unit": "original_sensor_units_float32",
        }
        audit_rows.append(row_audit)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            source.pop("front_depth_raw")
            source["front_depth_native"] = original_depth
            source["front_depth_color_aligned"] = registered_sensor_units
            destination = output_dir / path.name
            with destination.with_name(destination.name + ".tmp").open("xb") as f:
                np.savez_compressed(f, **source)
            destination.with_name(destination.name + ".tmp").replace(destination)
        print(json.dumps(row_audit, ensure_ascii=False), flush=True)

    report = {
        "status": "VERIFIED_NATIVE_AND_REGISTERED",
        "input_replay": str(source_dir),
        "dataset_root": str(dataset_root),
        "episode": args.episode,
        "frames_checked": len(audit_rows),
        "registration_source": "rgbtrack_adapter._rgb_registered_depth",
        "calibration_profile": str(profile.resolve()),
        "calibration_sha256": _sha256(profile.resolve()),
        "depth_scale_m_per_unit": float(camera.depth_scale_m_per_unit),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "rows": audit_rows,
    }
    if output_dir is not None:
        (output_dir / "depth_conversion_provenance.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
