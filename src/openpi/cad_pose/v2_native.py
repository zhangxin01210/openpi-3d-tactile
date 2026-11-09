"""Publish and verify the actual single-CAD V2 candidate as an independent native sidecar."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


def _load_repo_script(name):
    """Load the tracked V2 implementation even when the launcher is in scripts/."""
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[3] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing V2 repository script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_candidate = _load_repo_script("v2_ho_candidate_reader_smoke")
ROLES = _candidate.ROLES
TASKS = _candidate.TASKS
V2CandidateReader = _candidate.V2CandidateReader

SCHEMA = "single_rgbd_cad_pose_v2_v1"
NATIVE_SCHEMA = "openpi_rgbd_single_cad_v2_geometry_v1"
PREPROCESS = {
    "rgb_resize": "opencv_inter_area",
    "depth_resize": "opencv_inter_nearest",
    "rgb_scale": 255.0,
    "depth_clip_m": 2.0,
    "depth_input": "raw_uint16_unaligned",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path, value):
    """Publish one JSON file atomically without replacing an existing file."""
    path = Path(path)
    temp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    with temp.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def checkpoint_metadata(path):
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or not {"model_state_dict", "width", "height", "depth_scale", "script_sha256", "tasks"} <= value.keys()
    ):
        raise ValueError("V2 checkpoint schema or required fields missing")
    if (
        int(value["width"]) < 64
        or int(value["height"]) < 64
        or float(value["depth_scale"]) <= 0
        or set(value["tasks"]) != set(TASKS)
    ):
        raise ValueError("V2 checkpoint input or task contract invalid")
    return value


def validate_sources(candidate_root, cache_root, checkpoint, task):
    candidate_root, cache_root, checkpoint = map(Path, (candidate_root, cache_root, checkpoint))
    reader = V2CandidateReader(candidate_root, task)
    prov, summary = reader.provenance, reader.summary
    ckpt = checkpoint_metadata(checkpoint)
    current_script_sha = sha256(Path(__file__).resolve().parents[3] / "cad_pose_rgbd_single_v2.py")
    if ckpt["script_sha256"] != current_script_sha:
        raise ValueError("V2 model script SHA differs from checkpoint; synchronize the exact training script")
    config_path, splits_path = cache_root / "cache_config.json", cache_root / "episode_splits.json"
    config, splits = read_json(config_path), read_json(splits_path)
    cad_path = cache_root / task / "cad.npy"
    digest = sha256(checkpoint)
    if (
        prov.get("schema") != "v2_pose_pred_full_candidate_v1"
        or summary.get("schema") != "v2_pose_pred_full_candidate_v1"
        or set(prov.get("tasks", [])) != set(TASKS)
        or prov.get("checkpoint_schema") != SCHEMA
        or prov.get("checkpoint_sha256") != digest
        or summary.get("source_checkpoint_sha256") != digest
        or prov.get("checkpoint_script_sha256") != ckpt["script_sha256"]
        or prov.get("rgbd_cache_config_sha256") != sha256(config_path)
        or summary.get("source_cache_config_sha256") != prov.get("rgbd_cache_config_sha256")
        or prov.get("rgbd_episode_splits_sha256") != sha256(splits_path)
        or prov.get("cad_sha256_by_task", {}).get(task) != sha256(cad_path)
        or config.get("schema") != SCHEMA
        or not config.get("script_sha256")
        or (config.get("width"), config.get("height")) != (ckpt["width"], ckpt["height"])
        or (prov.get("image_width"), prov.get("image_height")) != (ckpt["width"], ckpt["height"])
        or float(prov.get("depth_scale", -1)) != float(ckpt["depth_scale"])
        or prov.get("role_names") != list(ROLES)
        or prov.get("inference_inputs_only") != ["rgb", "raw_depth", "CAD"]
    ):
        raise ValueError(f"{task}: V2 candidate, checkpoint, CAD or RGB-D preprocessing provenance mismatch")
    groups = splits[task]
    episodes = [int(e) for split in ("train", "val", "test") for e in groups[split]]
    if len(episodes) != len(set(episodes)) or sorted(episodes) != list(reader.episode_ids):
        raise ValueError(f"{task}: candidate episodes differ from cache split union")
    task_summary = summary.get("by_task", {}).get(task, {})
    counts = [int(read_json(candidate_root / task / f"e{ep:05d}_candidate.json")["frames"]) for ep in episodes]
    if task_summary.get("episodes") != len(episodes) or task_summary.get("frames") != sum(counts):
        raise ValueError(f"{task}: candidate summary and episode counts differ")
    return reader, ckpt, config, splits


def raw_frame_ids(dataset_root, episode):
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    info = read_json(root / "meta/info.json")
    chunk_size = int(info.get("chunks_size", 1000))
    if chunk_size <= 0:
        raise ValueError("invalid LeRobot chunks_size")
    path = root / info["data_path"].format(episode_index=episode, episode_chunk=episode // chunk_size)
    table = pq.read_table(path, columns=["episode_index", "frame_index"])
    ep = np.asarray(table.column("episode_index"))
    frame = np.asarray(table.column("frame_index"))
    if not len(frame) or not np.all(ep == episode) or not np.array_equal(frame, np.arange(len(frame))):
        raise ValueError(f"raw episode/frame indices invalid in {dataset_root} e{episode}")
    return frame


def raw_file_fingerprints(raw):
    result = {}
    for name, source_path in (("parquet", raw.parquet_path), ("front_video", raw.video_path("front"))):
        path = Path(source_path)
        stat = path.stat()
        result[name] = {
            "path": str(path.relative_to(raw.root)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        }
    return result


def verify_raw_cache(dataset_root, cache_root, task, episode, config):
    """Prove cache image identity using the V2 exporter resize, never read labels."""
    import cv2

    from openpi.spatial_dataset.source import RawSpatialDataset

    raw = RawSpatialDataset(dataset_root, episode=episode)
    folder = Path(cache_root) / task / f"e{episode:05d}"
    rgb = np.load(folder / "rgb.npy", mmap_mode="r", allow_pickle=False)
    depth = np.load(folder / "depth.npy", mmap_mode="r", allow_pickle=False)
    h, w = int(config["height"]), int(config["width"])
    if (
        rgb.shape != (len(rgb), h, w, 3)
        or rgb.dtype != np.uint8
        or depth.shape != (len(rgb), h, w)
        or depth.dtype != np.uint16
    ):
        raise ValueError(f"{task} e{episode}: invalid cached RGB-D shape/dtype")
    count = 0
    for row, video in zip(raw.iter_rows(depth_roles=("front",)), raw.iter_video_frames("front"), strict=True):
        frame = int(row["frame_index"])
        if frame != count or video.frame_index != frame or frame >= len(rgb):
            raise ValueError(f"{task} e{episode}: raw/cache frame identity mismatch")
        resized_rgb = cv2.resize(video.rgb, (w, h), interpolation=cv2.INTER_AREA)
        resized_depth = cv2.resize(row["observation.depths.cam_front"], (w, h), interpolation=cv2.INTER_NEAREST)
        if not np.array_equal(resized_rgb, rgb[frame]) or not np.array_equal(resized_depth, depth[frame]):
            raise ValueError(f"{task} e{episode} f{frame}: current RGB-D differs from V2 cache")
        count += 1
    if count != len(rgb):
        raise ValueError(f"{task} e{episode}: raw/cache frame counts differ")
    return raw_file_fingerprints(raw)


def validate_raw_files(dataset_root, raw_sources):
    for episode, fields in raw_sources.items():
        if set(fields) != {"parquet", "front_video"}:
            raise ValueError(f"episode {episode}: incomplete raw RGB-D provenance")
        for name, item in fields.items():
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("raw source path escapes dataset root")
            path = Path(dataset_root) / relative
            stat = path.stat()
            if stat.st_size != item["size"] or sha256(path) != item["sha256"]:
                raise ValueError(f"episode {episode}: raw {name} changed since V2 publication")


def replay_samples(reader, ckpt, cache_root, task, device):
    """Verify actual RGB-D + each CAD query against stored candidate; never open GT arrays."""
    import torch

    model_v2 = _load_repo_script("cad_pose_rgbd_single_v2")

    model = model_v2.SingleCadRGBDPose().to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    episode = reader.episode_ids[0]
    data = reader.load_episode(episode)
    directory = Path(cache_root) / task / f"e{episode:05d}"
    rgb = np.load(directory / "rgb.npy", mmap_mode="r", allow_pickle=False)
    depth = np.load(directory / "depth.npy", mmap_mode="r", allow_pickle=False)
    n = len(data["frame_index"])
    if (
        rgb.shape != (n, ckpt["height"], ckpt["width"], 3)
        or rgb.dtype != np.uint8
        or depth.shape != rgb.shape[:3]
        or depth.dtype != np.uint16
    ):
        raise ValueError("V2 cached RGB-D shape/dtype differs from checkpoint")
    maximum = 0.0
    with torch.inference_mode():
        for frame in sorted({0, n // 2, n - 1}):
            r = (
                torch.from_numpy(np.repeat(np.asarray(rgb[frame : frame + 1]), 2, axis=0).copy())
                .to(device)
                .permute(0, 3, 1, 2)
                .float()
                / 255.0
            )
            d = (
                torch.from_numpy(np.repeat(np.asarray(depth[frame : frame + 1]), 2, axis=0).copy())
                .to(device)
                .unsqueeze(1)
                .float()
            )
            d = (d * float(ckpt["depth_scale"])).clamp(0, 2.0) / 2.0
            cad = torch.from_numpy(data["cad_xyz_local_m"]).to(device)
            with torch.autocast(
                device_type=torch.device(device).type, enabled=torch.device(device).type == "cuda", dtype=torch.bfloat16
            ):
                translation, rotation = model(r, d, cad)
            pred = np.zeros((2, 4, 4), np.float32)
            pred[:, 3, 3] = 1
            pred[:, :3, :3] = rotation.float().cpu().numpy()
            pred[:, :3, 3] = translation.float().cpu().numpy()
            error = float(np.max(np.abs(pred - data["pred_T_base_object"][frame])))
            if not np.isfinite(error) or error > 0.005:
                raise ValueError(
                    f"{task} e{episode} f{frame}: V2 inference replay differs by {error}; check device/numerics"
                )
            maximum = max(maximum, error)
    return {
        "episode": episode,
        "frames": sorted({0, n // 2, n - 1}),
        "max_transform_abs_error": maximum,
        "tolerance": 0.005,
        "device": str(device),
        "gt_used": False,
    }


def dataset_meta_hashes(root):
    return {name: sha256(Path(root) / "meta" / name) for name in ("info.json", "episodes.jsonl")}


def publish_task(*, candidate_root, cache_root, checkpoint, dataset_root, task, device="cuda:0"):
    """Atomic per-episode resumable staging; never overwrite an existing sidecar."""
    from openpi.spatial_dataset.source import RawSpatialDataset

    root = Path(dataset_root).resolve()
    if root.name != task or task not in TASKS:
        raise ValueError("dataset root/task mismatch")
    reader, ckpt, config, splits = validate_sources(candidate_root, cache_root, checkpoint, task)
    episodes = tuple(RawSpatialDataset.discover_episode_ids(root))
    if episodes != reader.episode_ids:
        raise ValueError(f"{task}: raw and candidate episode IDs differ")
    output = root / "object_geometry/pretrained_pose_v2"
    if output.exists() or output.is_symlink():
        from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset

        if output.is_symlink():
            raise FileExistsError(f"refusing existing sidecar symlink {output}")
        native = ObjectGeometryDerivedDataset(root, version="pretrained_pose_v2")
        validate_native_manifest(native.manifest, root, checkpoint, cache_root, reader=native)
        if native.manifest.get("candidate_provenance_sha256") != sha256(
            Path(candidate_root) / "provenance.json"
        ) or native.manifest.get("candidate_summary_sha256") != sha256(Path(candidate_root) / "summary.json"):
            raise ValueError(f"existing sidecar uses different candidate source: {output}")
        print(f"[{task}] already published; verified without overwrite", flush=True)
        return output
    stage = root / "object_geometry/.pretrained_pose_v2.staging"
    if stage.is_symlink():
        raise ValueError(f"refusing staging symlink: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    replay = replay_samples(reader, ckpt, cache_root, task, device)
    current_script_sha = sha256(Path(__file__).resolve().parents[3] / "cad_pose_rgbd_single_v2.py")
    replay["current_model_script_sha256"] = current_script_sha
    replay["checkpoint_script_matches_current"] = current_script_sha == ckpt["script_sha256"]
    if not replay["checkpoint_script_matches_current"]:
        raise ValueError("V2 model script changed after candidate provenance validation")
    source = {
        "schema": NATIVE_SCHEMA,
        "dataset": task,
        "pose_checkpoint_sha256": sha256(checkpoint),
        "checkpoint_schema": SCHEMA,
        "checkpoint_script_sha256": ckpt["script_sha256"],
        "candidate_provenance_sha256": sha256(Path(candidate_root) / "provenance.json"),
        "candidate_summary_sha256": sha256(Path(candidate_root) / "summary.json"),
        "rgbd_cache_config_sha256": sha256(Path(cache_root) / "cache_config.json"),
        "episode_splits_sha256": sha256(Path(cache_root) / "episode_splits.json"),
        "cad_file_sha256": sha256(Path(cache_root) / task / "cad.npy"),
        "dataset_meta_sha256": dataset_meta_hashes(root),
        "preprocess": {
            **PREPROCESS,
            "width": int(ckpt["width"]),
            "height": int(ckpt["height"]),
            "depth_scale": float(ckpt["depth_scale"]),
        },
        "cache_script_sha256": config["script_sha256"],
        "raw_input_binding": "all frames current raw RGB-D resized and compared with V2 cache; source file fingerprints recorded",
        "calibration_source": "V2 predicts base_link from unaligned raw RGB-D; model camera calibration provenance not recorded",
        "validity_semantics": "finite_prediction_not_visibility",
        "gt_used": False,
    }
    source_path = stage / "source.json"
    if source_path.exists() and read_json(source_path) != source:
        raise ValueError("staging source differs; inspect existing staging directory")
    if not source_path.exists():
        write_json_once(source_path, source)
    records = []
    shard_hashes = {}
    raw_sources = {}
    for episode in episodes:
        data = reader.load_episode(episode)
        ids = data["frame_index"]
        if not np.array_equal(ids, raw_frame_ids(root, episode)):
            raise ValueError(f"{task} e{episode}: candidate/raw frame indices differ")
        if not np.array_equal(
            data["cad_xyz_local_m"], np.load(Path(cache_root) / task / "cad.npy", allow_pickle=False)
        ):
            raise ValueError("candidate CAD differs from approved cache")
        n = len(ids)
        predicted_t = data["pred_T_base_object"]
        rotation = predicted_t[:, :, :3, :3]
        reconstructed = np.einsum("nrij,rkj->nrki", rotation, data["cad_xyz_local_m"]) + predicted_t[:, :, None, :3, 3]
        if (
            not np.allclose(predicted_t[:, :, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=3e-3, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=3e-3, rtol=0)
            or not np.allclose(data["candidate_xyz_m"], reconstructed, atol=1e-5, rtol=0)
        ):
            raise ValueError(f"{task} e{episode}: predicted SE(3)/CAD geometry mismatch")
        shard = f"shard_e{episode:05d}"
        folder = stage / shard
        meta_path = folder / "source.json"
        candidate_meta = Path(candidate_root) / task / f"e{episode:05d}_candidate.json"
        meta = read_json(candidate_meta)
        if (
            meta.get("schema") != "v2_pose_pred_full_candidate_v1"
            or meta.get("frames") != n
            or meta.get("rgbd_cache_config_sha256") != reader.provenance["rgbd_cache_config_sha256"]
            or meta.get("cad_sha256") != reader.provenance["cad_sha256_by_task"][task]
        ):
            raise ValueError(f"{task} e{episode}: candidate episode provenance mismatch")
        cache_dir = Path(cache_root) / task / f"e{episode:05d}"
        cache_hashes = {name: sha256(cache_dir / f"{name}.npy") for name in ("rgb", "depth")}
        expected_ep = {"candidate_file_sha256": meta["file_sha256"], "frames": n, "cache_sha256": cache_hashes}
        if folder.exists() or meta_path.exists():
            if not folder.is_dir() or not meta_path.is_file():
                raise ValueError(f"partial staged episode: {folder}")
            source_ep = read_json(meta_path)
            if any(source_ep.get(key) != value for key, value in expected_ep.items()) or not source_ep.get(
                "raw_source_files"
            ):
                raise ValueError(f"mismatched staged episode: {folder}")
            validate_raw_files(root, {str(episode): source_ep["raw_source_files"]})
            if not np.array_equal(np.load(folder / "frame_index.npy", mmap_mode="r"), ids) or not np.all(
                np.load(folder / "episode_index.npy", mmap_mode="r") == episode
            ):
                raise ValueError(f"staged episode/frame index differs: {folder}")
            for role_i, role in enumerate(ROLES):
                if not np.array_equal(
                    np.load(folder / f"{role}_xyz_m.npy", mmap_mode="r"), data["candidate_xyz_m"][:, role_i]
                ):
                    raise ValueError(f"staged prediction differs: {folder}")
                for suffix, expected in (("mask", True), ("source", 1), ("age", 0), ("confidence", 0.0)):
                    array = np.load(folder / f"{role}_{suffix}.npy", mmap_mode="r", allow_pickle=False)
                    if array.shape != (n, 512) or not np.all(array == expected):
                        raise ValueError(f"staged {role}_{suffix} differs: {folder}")
        else:
            source_ep = {**expected_ep, "raw_source_files": verify_raw_cache(root, cache_root, task, episode, config)}
            temp = stage / f".{shard}.tmp_{os.getpid()}"
            temp.mkdir()
            np.save(temp / "episode_index.npy", np.full(n, episode, np.int64), allow_pickle=False)
            np.save(temp / "frame_index.npy", ids.astype(np.int64), allow_pickle=False)
            for i, role in enumerate(ROLES):
                np.save(temp / f"{role}_xyz_m.npy", data["candidate_xyz_m"][:, i], allow_pickle=False)
                for suffix, dtype, val in (
                    ("mask", bool, True),
                    ("source", np.int8, 1),
                    ("age", np.int16, 0),
                    ("confidence", np.float32, 0),
                ):
                    np.save(temp / f"{role}_{suffix}.npy", np.full((n, 512), val, dtype=dtype), allow_pickle=False)
            (temp / "source.json").write_text(json.dumps(source_ep, indent=2) + "\n")
            temp.rename(folder)
        raw_sources[str(episode)] = source_ep["raw_source_files"]
        fields = ["episode_index", "frame_index"] + [
            f"{role}_{suffix}" for role in ROLES for suffix in ("xyz_m", "mask", "source", "age", "confidence")
        ]
        shard_hashes[shard] = {field: sha256(folder / f"{field}.npy") for field in fields}
        records.extend(
            {"episode_index": episode, "frame_index": int(f), "shard": shard, "row": i} for i, f in enumerate(ids)
        )
        print(f"[{task}] e{episode:05d}: {n} frames verified", flush=True)
    manifest = {
        **source,
        "status": "complete",
        "producer": "openpi.cad_pose.v2_native",
        "object_geometry_contract": {"version": "pretrained_pose_v2", "point_count": 512},
        "coordinate_frame": "base_link",
        "length_unit": "m",
        "role_to_object": dict(zip(ROLES, TASKS[task], strict=True)),
        "replay": replay,
        "shard_sha256": shard_hashes,
        "raw_source_files": raw_sources,
        "records": records,
    }
    path = stage / "manifest.json"
    if path.exists():
        existing = read_json(path)
        # Replay may vary slightly between devices; the prior replay must still be valid.
        old_replay = existing.pop("replay", None)
        new_compare = {k: v for k, v in manifest.items() if k != "replay"}
        if existing != new_compare or not isinstance(old_replay, dict) or old_replay.get("gt_used") is not False:
            raise ValueError("existing staged manifest differs; inspect manually")
    else:
        write_json_once(path, manifest)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    stage.rename(output)
    return output


def validate_native_manifest(manifest, dataset_root, checkpoint, cache_root, *, reader=None):
    """Verify the V2 contract without depending on any V1 checkpoint fields."""
    dataset_root, checkpoint, cache_root = map(Path, (dataset_root, checkpoint, cache_root))
    task = dataset_root.name
    if task not in TASKS:
        raise ValueError(f"unknown V2 task {task}")
    ckpt = checkpoint_metadata(checkpoint)
    cfg_path, splits_path = cache_root / "cache_config.json", cache_root / "episode_splits.json"
    cfg = read_json(cfg_path)
    expected_preprocess = {
        **PREPROCESS,
        "width": int(ckpt["width"]),
        "height": int(ckpt["height"]),
        "depth_scale": float(ckpt["depth_scale"]),
    }
    replay = manifest.get("replay") or {}
    if (
        manifest.get("schema") != NATIVE_SCHEMA
        or manifest.get("producer") != "openpi.cad_pose.v2_native"
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != task
        or manifest.get("object_geometry_contract") != {"version": "pretrained_pose_v2", "point_count": 512}
        or manifest.get("coordinate_frame") != "base_link"
        or manifest.get("length_unit") != "m"
        or manifest.get("role_to_object") != dict(zip(ROLES, TASKS[task], strict=True))
        or manifest.get("pose_checkpoint_sha256") != sha256(checkpoint)
        or manifest.get("checkpoint_schema") != SCHEMA
        or manifest.get("checkpoint_script_sha256") != ckpt["script_sha256"]
        or manifest.get("rgbd_cache_config_sha256") != sha256(cfg_path)
        or manifest.get("episode_splits_sha256") != sha256(splits_path)
        or manifest.get("cad_file_sha256") != sha256(cache_root / task / "cad.npy")
        or manifest.get("dataset_meta_sha256") != dataset_meta_hashes(dataset_root)
        or cfg.get("schema") != SCHEMA
        or cfg.get("script_sha256") != manifest.get("cache_script_sha256")
        or (cfg.get("width"), cfg.get("height")) != (ckpt["width"], ckpt["height"])
        or manifest.get("preprocess") != expected_preprocess
        or manifest.get("gt_used") is not False
        or manifest.get("validity_semantics") != "finite_prediction_not_visibility"
        or not manifest.get("candidate_provenance_sha256")
        or not manifest.get("candidate_summary_sha256")
        or manifest.get("raw_input_binding")
        != "all frames current raw RGB-D resized and compared with V2 cache; source file fingerprints recorded"
        or not manifest.get("calibration_source")
        or replay.get("gt_used") is not False
        or not replay.get("frames")
        or not isinstance(replay.get("max_transform_abs_error"), int | float)
        or not np.isfinite(replay["max_transform_abs_error"])
        or replay["max_transform_abs_error"] > 0.005
    ):
        raise ValueError(f"{task}: invalid V2 checkpoint/CAD/preprocess/native manifest provenance")
    if reader is not None:
        from openpi.spatial_dataset.source import RawSpatialDataset

        episodes = RawSpatialDataset.discover_episode_ids(dataset_root)
        if tuple(episodes) != reader.episode_ids:
            raise ValueError(f"{task}: native V2 and raw episode IDs differ")
        splits = read_json(splits_path)[task]
        split_ids = [int(e) for part in ("train", "val", "test") for e in splits[part]]
        if len(split_ids) != len(set(split_ids)) or sorted(split_ids) != list(episodes):
            raise ValueError(f"{task}: V2 split episode mapping differs")
        raw_episode_lengths = {}
        with (dataset_root / "meta/episodes.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                entry = json.loads(line)
                episode_id = int(entry["episode_index"])
                if episode_id in raw_episode_lengths or int(entry.get("length", 0)) <= 0:
                    raise ValueError(f"{task}: invalid raw episode metadata")
                raw_episode_lengths[episode_id] = int(entry["length"])
        if set(raw_episode_lengths) != set(episodes):
            raise ValueError(f"{task}: raw episode metadata incomplete")
        for record in manifest["records"]:
            episode_id, frame = int(record["episode_index"]), int(record["frame_index"])
            if record.get("shard") != f"shard_e{episode_id:05d}" or int(record.get("row", -1)) != frame:
                raise ValueError(f"{task}: native V2 manifest shard/row index differs")
        raw_sources = manifest.get("raw_source_files")
        if not isinstance(raw_sources, dict) or set(raw_sources) != {str(e) for e in episodes}:
            raise ValueError(f"{task}: missing raw RGB-D fingerprints")
        validate_raw_files(dataset_root, raw_sources)
        hashes = manifest.get("shard_sha256")
        if not isinstance(hashes, dict) or len(hashes) != len(episodes):
            raise ValueError(f"{task}: missing native shard digests")
        fields = {"episode_index", "frame_index"} | {
            f"{role}_{suffix}" for role in ROLES for suffix in ("xyz_m", "mask", "source", "age", "confidence")
        }
        for episode in episodes:
            shard = f"shard_e{episode:05d}"
            if set(hashes.get(shard, {})) != fields:
                raise ValueError(f"{task}: missing native shard field digests for {shard}")
            for field in fields:
                if sha256(reader.root / shard / f"{field}.npy") != hashes[shard][field]:
                    raise ValueError(f"{task}: modified native V2 geometry: {shard}/{field}")
            frames = reader.frame_ids(episode)
            if not frames or frames != tuple(range(len(frames))) or len(frames) != raw_episode_lengths[episode]:
                raise ValueError(f"{task}: invalid sidecar frame index for episode {episode}")
    return {k: v for k, v in manifest.items() if k != "records"}
