#!/usr/bin/env python3
"""Read-only, per-configuration preflight; never initializes model weights."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

from openpi.training import config as config_lib
from openpi.training import data_loader


def names():
    return [f"{prefix}_xhand_{dataset}_{mode}_v2" for dataset, variants in (
        ("1028_press_button", (("pi0", "rgb"), ("pi05", "rgb"), ("pi0", "st"), ("pi0", "full"))),
        ("1030_yellow_socket", (("pi0", "rgb"), ("pi05", "rgb"), ("pi0", "st"), ("pi0", "full"))),
        ("1018_peg_all", (("pi0", "st"), ("pi0", "full"))),
    ) for prefix, mode in variants]


def inspect(name, *, cache=None):
    if cache is None:
        cache = {}
    result = {"config": name, "checks": {}, "errors": [], "warnings": []}
    def check(label, fn):
        try:
            value = fn()
            result["checks"][label] = "OK"
            return value
        except Exception as exc:
            result["checks"][label] = "FAIL"
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    def validate_config():
        cfg = config_lib.get_config(name)
        dataset = next(d for d in ("1028_press_button", "1030_yellow_socket", "1018_peg_all") if d in name)
        mode = name.removesuffix("_v2").rsplit("_", 1)[-1]
        if cfg.data.repo_id != f"data/{dataset}" or bool(cfg.model.pi05) != name.startswith("pi05_"):
            raise ValueError("model type or dataset binding mismatch")
        spatial = mode in {"st", "full"}
        if bool(cfg.data.spatial) != spatial or bool(cfg.data.object_geometry) != (mode == "full"):
            raise ValueError("spatial/object sidecar switch mismatch")
        if spatial:
            encoder = cfg.model.encoder
            if (cfg.data.spatial.version != "v1" or not encoder.visual_hierarchy_v2 or encoder.tactile_regions is None
                or encoder.default_token_count != (68 if mode == "full" else 58)
                or encoder.enable_object_relations != (mode == "full") or cfg.model.conditioning.target != "suffix"):
                raise ValueError("V2 spatial configuration, token count or suffix routing mismatch")
        if cfg.data.object_geometry and cfg.data.object_geometry.version != "v1":
            raise ValueError("object geometry version must be v1")
        return cfg
    cfg = check("CONFIG", validate_config)
    if cfg is None:
        return result
    repo_root = Path(__file__).resolve().parents[2]
    root = data_loader.resolve_local_lerobot_root(cfg.data.repo_id)
    if root is None:
        raise ValueError(f"V2 configuration must use a local dataset: {cfg.data.repo_id}")
    check("DATASET", lambda: root.is_dir() or (_ for _ in ()).throw(FileNotFoundError(root)))
    info_path = root / "meta/info.json"
    check("LEROBOT_META", lambda: info_path.is_file() or (_ for _ in ()).throw(FileNotFoundError(info_path)))
    weight = Path(cfg.weight_loader.params_path)
    check("WEIGHT", lambda: weight.exists() or (_ for _ in ()).throw(FileNotFoundError(weight)))
    assets_dir = Path(cfg.data.assets.assets_dir)
    if not assets_dir.is_absolute():
        assets_dir = repo_root / assets_dir
    stats_path = assets_dir / cfg.data.assets.asset_id / "norm_stats.json"
    norm_mode = "st" if root.name.startswith("1018") else "rgb"
    norm_config = f"pi0_xhand_{root.name}_{norm_mode}_v2"
    check("NORM", lambda: stats_path.is_file() or (_ for _ in ()).throw(
        FileNotFoundError(f"{stats_path}; run scripts/compute_norm_stats.py --config-name {norm_config}")))
    if stats_path.is_file():
        def check_stats():
            from openpi.shared import normalize
            stats = normalize.load(stats_path.parent)
            for key in ("state", "actions"):
                if key not in stats or np.asarray(stats[key].mean).shape != (18,):
                    raise ValueError(f"normalization {key} must have 18 dimensions")
                if cfg.model.pi05 and (stats[key].q01 is None or stats[key].q99 is None):
                    raise ValueError(f"normalization {key} requires q01 and q99 for pi05")
        check("NORM_CONTENTS", check_stats)
    if cfg.data.spatial is not None:
        spatial_path = root / "spatial" / cfg.data.spatial.version / "manifest.json"
        check("SPATIAL", lambda: spatial_path.is_file() or (_ for _ in ()).throw(FileNotFoundError(spatial_path)))
    if cfg.data.object_geometry is not None:
        geometry_path = root / "object_geometry" / cfg.data.object_geometry.version / "manifest.json"
        def inspect_manifest():
            manifest = json.loads(geometry_path.read_text())
            roles = manifest.get("role_to_object") or manifest.get("source", {}).get("roles")
            expected = {"1028": {"manipulated": "button", "target": "box"},
                        "1030": {"manipulated": "red", "target": "yellow"},
                        "1018": {"manipulated": "funnel", "target": "target_flask"}}[root.name[:4]]
            if roles != expected:
                raise ValueError(f"{geometry_path}: role_to_object expected {expected}, found {roles}; inspect source manifest")
            frame = (manifest.get("coordinate_frame") or manifest.get("object_geometry_contract", {}).get("coordinate_frame")
                     or (manifest.get("calibration") or {}).get("coordinate_frame"))
            if frame != "base_link":
                raise ValueError(f"{geometry_path}: missing base_link coordinate frame")
            if manifest.get("object_geometry_contract", {}).get("point_count") != 512:
                raise ValueError(f"{geometry_path}: object point_count must be 512")
            return manifest
        manifest = check("OBJECT", inspect_manifest)
        if manifest is not None:
            def calibration():
                return data_loader.validate_object_geometry_calibration(cfg.data, cfg.model, manifest)
            legacy = check("CALIBRATION", calibration)
            if legacy:
                result["checks"]["CALIBRATION"] = "LEGACY_UNRECORDED"
                result["warnings"].append(
                    "LEGACY_UNRECORDED: object geometry has no historical calibration provenance; "
                    "its calibration cannot be considered verified")
    def join_and_mask():
        factory = dataclasses.replace(cfg.data, assets=dataclasses.replace(cfg.data.assets, assets_dir=str(assets_dir)))
        data = factory.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
        if not len(dataset):
            raise ValueError("dataset is empty")
        expected_episodes, expected_frames = {"1018": (70, 32343), "1028": (100, 26485),
                                              "1030": (111, 34963)}[root.name[:4]]
        if len(dataset) != expected_frames:
            raise ValueError(f"LeRobot frame count expected {expected_frames}, found {len(dataset)}")
        from openpi.spatial_dataset.source import RawSpatialDataset
        raw_episodes = tuple(RawSpatialDataset.discover_episode_ids(root))
        if len(raw_episodes) != expected_episodes:
            raise ValueError(f"episode count expected {expected_episodes}, found {len(raw_episodes)}")
        if data.spatial:
            import pyarrow.parquet as pq

            from openpi.spatial_dataset.derived import SpatialDerivedDataset

            spatial_index = SpatialDerivedDataset(root, version=data.spatial.version)
            if spatial_index.coordinate_frame != "base_link" or spatial_index.xyz_unit != "m":
                raise ValueError("spatial geometry must use base_link coordinates in metres")
            if spatial_index.force_unit != "dataset_native":
                raise ValueError(f"unexpected tactile force unit: {spatial_index.force_unit}")
            for finger in range(5):
                ids = spatial_index.taxel_id[spatial_index.finger_id == finger]
                if not np.array_equal(np.sort(ids), np.arange(1, 121)):
                    raise ValueError(f"finger {finger}: expected each canonical taxel id 1..120 exactly once")
            if tuple(spatial_index.episode_ids) != raw_episodes:
                raise ValueError("LeRobot and spatial episode sets differ")
            object_index = None
            hand_reader = None
            if data.object_geometry:
                from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
                from openpi.training.xhand_multimodal_dataset import XHandGeometryHandJoin
                object_index = ObjectGeometryDerivedDataset(root, version=data.object_geometry.version)
                if object_index.episode_ids != raw_episodes:
                    raise ValueError("LeRobot and object episode sets differ")
                hand_reader = XHandGeometryHandJoin(dataset, root, object_version=data.object_geometry.version,
                                                   spatial_version=data.spatial.version)
            count = 0
            index_cache = cache.setdefault("indices", {})
            spatial_key = (str(root), data.spatial.version, data.spatial.camera_roles,
                           data.spatial.front_calibration_profile)
            spatial_done = cache.setdefault("spatial_scans", set())
            object_key = (spatial_key, data.object_geometry.version) if data.object_geometry else None
            object_done = cache.setdefault("object_scans", set())
            for episode in raw_episodes:
                frame_key = (str(root), episode)
                if frame_key not in index_cache:
                    raw = RawSpatialDataset(root, episode=episode)
                    index_cache[frame_key] = pq.read_table(raw.parquet_path, columns=["frame_index"])["frame_index"].to_numpy()
                frames = index_cache[frame_key]
                if not np.array_equal(frames, np.arange(len(frames))):
                    raise ValueError(f"episode {episode}: raw frame indices not contiguous")
                if spatial_index.episode_length(episode) != len(frames):
                    raise ValueError(f"episode {episode}: spatial frame count mismatch")
                if spatial_key not in spatial_done:
                    next_frame = 0
                    for shard in spatial_index._episode_shards[episode]:  # noqa: SLF001
                        if shard.first_frame != next_frame:
                            raise ValueError(f"episode {episode}: spatial shard gap or overlap at frame {next_frame}")
                        arrays = spatial_index._load_shard_arrays(shard)  # noqa: SLF001
                        if not np.array_equal(arrays["frame_index"], frames[shard.first_frame:shard.last_frame + 1]):
                            raise ValueError(f"episode {episode}: spatial shard {shard.name} frame indices mismatch")
                        next_frame = shard.last_frame + 1
                    if next_frame != len(frames):
                        raise ValueError(f"episode {episode}: spatial shard coverage incomplete")
                if object_index and object_key not in object_done and object_index.frame_ids(episode) != tuple(range(len(frames))):
                    raise ValueError(f"episode {episode}: object frame indices mismatch")
                if hand_reader and object_key not in object_done:
                    record = hand_reader.hand_records.get(episode)
                    if not record or int(record["frames"]) != len(frames):
                        raise ValueError(f"episode {episode}: hand frame count mismatch")
                    hand_reader._hand_xyz(episode, len(frames) - 1)  # noqa: SLF001
                count += len(frames)
            if count != len(dataset):
                raise ValueError(f"raw frame count {count} differs from LeRobot {len(dataset)}")
            spatial_done.add(spatial_key)
            if object_key is not None:
                object_done.add(object_key)
        for index in sorted({0, len(dataset) - 1}):
            item = dataset[index]
            if data.spatial:
                spatial = item["spatial"]
                for label, count in (("visual", 4096), ("tactile", 600)):
                    xyz = np.asarray(spatial[label]["xyz_m"])
                    mask = np.asarray(spatial[label]["point_mask"], dtype=bool)
                    if xyz.shape != (count, 3) or mask.shape != (count,) or not np.isfinite(xyz[mask]).all():
                        raise ValueError(f"{label}: invalid xyz/mask")
                    if label == "tactile":
                        for field, shape in (("force", (600, 3)), ("force_norm", (600,))):
                            value = np.asarray(spatial[label][field])
                            if value.shape != shape or not np.isfinite(value[mask]).all():
                                raise ValueError(f"tactile {field}: invalid shape or nonfinite valid data")
                if data.object_geometry:
                    hand = spatial["hand_keypoints"]
                    if np.asarray(hand["xyz_m"]).shape != (18, 3) or np.asarray(hand["point_mask"]).shape != (18,):
                        raise ValueError("hand keypoints xyz/mask shape mismatch")
                    obj = spatial["object_geometry"]
                    for role in ("manipulated", "target"):
                        xyz, mask = np.asarray(obj[f"{role}_xyz_m"]), np.asarray(obj[f"{role}_mask"], dtype=bool)
                        if xyz.shape != (512, 3) or mask.shape != (512,) or not np.isfinite(xyz[mask]).all():
                            raise ValueError(f"{role}: invalid xyz/mask")
        return len(dataset)
    blocking = ("DATASET", "LEROBOT_META", "NORM", "NORM_CONTENTS", "SPATIAL", "OBJECT", "CALIBRATION")
    if root.is_dir() and not any(result["checks"].get(label) == "FAIL" for label in blocking):
        spatial = cfg.data.spatial
        objects = cfg.data.object_geometry
        join_key = (str(root), cfg.model.action_horizon,
                    None if spatial is None else (spatial.version, spatial.camera_roles, spatial.front_calibration_profile),
                    None if objects is None else objects.version)
        joins = cache.setdefault("joins", {})
        if join_key in joins:
            joined = joins[join_key]
            result["checks"]["JOIN_AND_MASK"] = "OK (cached)"
        else:
            joined = check("JOIN_AND_MASK", join_and_mask)
            if joined is not None:
                joins[join_key] = joined
        if joined is not None:
            result["checks"]["JOIN"] = "OK"
            result["checks"]["MASK"] = "OK"
            if cfg.data.spatial and not cfg.data.object_geometry:
                # create_torch_dataset checked the spatial manifest against the configured profile.
                result["checks"]["CALIBRATION"] = "OK"
            if cfg.data.object_geometry:
                result["checks"]["KEYPOINTS"] = "OK"
                result["checks"]["ROLES"] = result["checks"].get("OBJECT", "FAIL")
    else:
        result["checks"]["JOIN_AND_MASK"] = "SKIPPED"
    result["ready"] = not result["errors"] and all(v in {"OK", "OK (cached)", "LEGACY_UNRECORDED"}
                                                   for v in result["checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-configs", action="store_true")
    group.add_argument("--config", choices=names())
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = []
    cache = {}
    for name in (names() if args.all_configs else [args.config]):
        print(f"[START] {name}", flush=True)
        try:
            result = inspect(name, cache=cache)
        except Exception as exc:
            result = {"config": name, "checks": {}, "errors": [f"{type(exc).__name__}: {exc}"], "ready": False}
        results.append(result)
        print(f"[{ 'READY FOR SMOKE' if result.get('ready') else 'FAIL' }] {result['config']}", flush=True)
        for label, value in result["checks"].items():
            print(f"  [{label} {value}]", flush=True)
        for error in result["errors"]:
            print(f"    {error}", flush=True)
        for warning in result.get("warnings", []):
            print(f"  [WARNING] {warning}", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    else:
        print(json.dumps(results, indent=2))
    return 0 if all(item.get("ready") for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
