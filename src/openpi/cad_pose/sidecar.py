"""Write an independent predicted-geometry sidecar from raw RGB-D frames only."""

import json
from pathlib import Path
import re
import tempfile

import numpy as np

from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.spatial_dataset.source import RawSpatialDataset


def export_sidecar(
    *,
    task,
    runtime,
    repo,
    profile,
    checkpoint_sha256,
    camera_hashes,
    version="pretrained_pose_v1",
    raw_factory=RawSpatialDataset,
    shard_size=128,
    acceptance=None,
):
    if version in {"v1", "v1_hand_scene"} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version) or shard_size < 1:
        raise ValueError("choose a safe, independent predicted sidecar version")
    output = Path(task.root) / "object_geometry" / version
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version}.building-", dir=output.parent))
    records, arrays, shard_id = [], [], 0

    def flush():
        nonlocal shard_id
        if not arrays:
            return
        shard = f"shard_{shard_id:06d}"
        folder = staging / shard
        folder.mkdir()
        for key in arrays[0]:
            np.save(folder / f"{key}.npy", np.stack([x[key] for x in arrays]), allow_pickle=False)
        for i, item in enumerate(arrays):
            records.append(
                {
                    "episode_index": int(item["episode_index"]),
                    "frame_index": int(item["frame_index"]),
                    "shard": shard,
                    "row": i,
                }
            )
        arrays.clear()
        shard_id += 1

    for episode in task.episodes:
        raw = raw_factory(task.root, episode=episode)
        count = 0
        for row, video in zip(raw.iter_rows(depth_roles=("front",)), raw.iter_video_frames("front"), strict=True):
            frame = int(row["frame_index"])
            if frame != count or video.frame_index != frame:
                raise ValueError(f"RGB-D frame mismatch: episode {episode}, frame {frame}")
            _, geometry = runtime.infer(video.rgb, row["observation.depths.cam_front"])
            arrays.append({"episode_index": np.int64(episode), "frame_index": np.int64(frame), **geometry})
            count += 1
            if len(arrays) >= shard_size:
                flush()
        if count == 0:
            raise ValueError(f"episode {episode}: no raw RGB-D frames")
    flush()
    manifest = {
        "status": "complete",
        "dataset": task.dataset,
        "producer": "openpi.cad_pose.sidecar",
        "object_geometry_contract": {"version": version, "point_count": 512},
        "coordinate_frame": "base_link",
        "length_unit": "m",
        "role_to_object": task.roles,
        "pose_checkpoint_sha256": checkpoint_sha256,
        "ordered_cad_sha256": __import__("hashlib").sha256(np.ascontiguousarray(runtime.cad).tobytes()).hexdigest(),
        "camera_hashes": camera_hashes,
        "calibration": front_calibration_provenance(Path(repo), profile),
        "acceptance": acceptance,
        "preprocess_version": "front_rgbd_spatial_v1",
        "model_version": "xhand_cad_pose_transformer_v1",
        "validity_semantics": "finite model output only; no calibrated pose confidence",
        "records": records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    staging.rename(output)
    return output
