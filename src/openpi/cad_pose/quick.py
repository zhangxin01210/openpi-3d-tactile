"""Bounded, sampled pose-data inspection; never asserts full source verification."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from scripts.cad_pose.preflight import ROLES
from scripts.preprocess.export_object_geometry_rgbtrack import SURFACE_SAMPLING_SEED
from scripts.preprocess.export_object_geometry_rgbtrack import cloud_from_pose
from scripts.preprocess.export_object_geometry_rgbtrack import load_surface_points


def first_episode_index(path):
    """Read the manifest header and first episode only, without materializing its complete index.

    Existing exporters write pretty-printed top-level records last. Reject other
    layouts instead of silently scanning an arbitrary complete data set.
    """
    with Path(path).open(encoding="utf-8") as stream:
        prefix = []
        size = 0
        for line in stream:
            size += len(line)
            if size > 8 * 1024 * 1024:
                raise ValueError("geometry manifest header exceeds 8 MiB before records")
            if line.rstrip("\r\n") == '  "records": [':
                break
            prefix.append(line)
        else:
            raise ValueError("unsupported manifest layout: expected top-level records array")
        header = json.loads("".join(prefix) + '  "records": []\n}')
        decoder = json.JSONDecoder()
        buffer = ""
        records = []
        episode = None
        for line in stream:
            buffer += line
            stripped = buffer.lstrip(" \t\r\n,")
            if stripped.startswith("]"):
                break
            try:
                item, end = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                if len(buffer) > 1024 * 1024:
                    raise ValueError("invalid or oversized geometry index entry") from None
                continue
            buffer = stripped[end:]
            if not isinstance(item, dict):
                raise ValueError("invalid geometry index entry")
            ep = int(item["episode_index"])
            if episode is None:
                episode = ep
            if ep != episode:
                break
            records.append(item)
        if len(records) < 3 or [int(x["frame_index"]) for x in records] != list(range(len(records))):
            raise ValueError("first geometry episode has fewer than three contiguous frames")
        return header, episode, records


def selected_geometry(root, records):
    """Memory map only index and geometry arrays for the three selected records."""
    result = []
    for rec in records:
        folder = Path(root) / str(rec["shard"])
        row = int(rec["row"])

        def load(name, *, _folder=folder, _row=row):
            return np.load(_folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)[_row]

        if int(load("episode_index")) != int(rec["episode_index"]) or int(load("frame_index")) != int(
            rec["frame_index"]
        ):
            raise ValueError("geometry shard and manifest index disagree")
        result.append(
            {
                role: (np.asarray(load(f"{role}_xyz_m")), np.asarray(load(f"{role}_mask"), dtype=bool))
                for role in ("manipulated", "target")
            }
        )
    return result


def selected_pose_rows(path, *, object_name, frames):
    """Decode only selected JSONL records; sequentially skip earlier bytes."""
    targets = set(frames)
    selected = {}
    with Path(path).open(encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            if ordinal not in targets:
                continue
            row = json.loads(line)
            if (
                row.get("frame_index") != ordinal
                or row.get("object_name") != object_name
                or type(row.get("valid")) is not bool
            ):
                raise ValueError(f"selected pose/object index invalid: {path} frame {ordinal}")
            selected[ordinal] = row
            if len(selected) == len(targets):
                break
    if set(selected) != targets:
        raise ValueError(f"selected pose frames missing in {path}")
    return selected


def quick_audit(repo, data_root, dataset, profile, *, do_smoke=True, allow_legacy_intrinsics=False):
    """Three-frame check; SHA source equality and unselected frames remain unverified."""
    root = Path(data_root) / dataset
    prep = Path(repo) / "output/preprocess" / dataset
    result = {"dataset": dataset, "status": "blocked", "errors": [], "checks": {}, "mode": "quick"}
    checks = result["checks"]
    timings = {}
    started = time.monotonic()

    def stage(name):
        nonlocal started
        now = time.monotonic()
        timings[name] = round(now - started, 3)
        print(f"  {dataset}: {name} {timings[name]:.2f}s", flush=True)
        started = now

    try:
        manifest, episode, records = first_episode_index(root / "object_geometry/v1/manifest.json")
        if (
            manifest.get("status") != "complete"
            or manifest.get("object_geometry_contract", {}).get("point_count") != 512
        ):
            raise ValueError("geometry v1 is incomplete or not 512 points")
        chosen = [records[0], records[len(records) // 2], records[-1]]
        frames = [int(x["frame_index"]) for x in chosen]
        samples = selected_geometry(root / "object_geometry/v1", chosen)
        checks.update(episode=episode, frames=frames, header_layout="first episode only")
        stage("metadata_and_geometry_index")
        expected_cal = front_calibration_provenance(Path(repo), profile)
        roles = ROLES[dataset]
        legacy = manifest.get("source", {}).get("mode") == "geometry_only_drop_hand_scene_extras_keep_raw_q2"
        cad = {}
        hash_mismatches = []
        if legacy:
            if dataset != "1018_peg_all" or manifest.get("source", {}).get("roles") != roles:
                raise ValueError("legacy object role provenance mismatch")
            from openpi.cad_pose.legacy import legacy_calibration_check

            checks["legacy_calibration"] = legacy_calibration_check(
                manifest.get("calibration"), expected_cal, allow_legacy_intrinsics=allow_legacy_intrinsics
            )
            assets = manifest.get("asset_provenance")
            if not isinstance(assets, dict) or not set(roles.values()).issubset(assets):
                source_info = manifest.get("source", {})
                old_digest = source_info.get("original_manifest_sha256")
                old_version = source_info.get("original_version")
                if old_version != "v1_hand_scene_q2" or not isinstance(old_digest, str) or len(old_digest) != 64:
                    raise ValueError("Peg cannot train: missing source manifest linkage")
                old_path = root / "object_geometry" / old_version / "manifest.json"
                old_header, _, _ = first_episode_index(old_path)
                if old_header.get("producer") != "mesh_6d_hand_scene_v1" or old_header.get("status") != "complete":
                    raise ValueError("Peg cannot train: unrecognized historical geometry producer")
                assets = old_header.get("generation", {}).get("assets")
                checks["legacy_source"] = {
                    "path": str(old_path),
                    "recorded_sha256": old_digest,
                    "hash_status": "not recomputed in quick; verified at training startup",
                    "asset_metadata": "historical generation.assets",
                }
            if not isinstance(assets, dict):
                raise ValueError("Peg cannot train: missing legacy CAD ordering/scale provenance")
            config = json.loads((Path(repo) / "configs/object_geometry/datasets/1018_peg_all/config.json").read_text())
            from openpi.object_geometry.production import load_mesh_surface_points

            for name in roles.values():
                cfg, provenance = config["assets"][name], assets[name]
                if (
                    provenance.get("source_sha256") != cfg["sha256"]
                    or provenance.get("sampler") != "deterministic_surface_points.v1"
                    or int(provenance.get("seed", -1)) != cfg["sampling_seed"]
                    or float(provenance.get("scale_m_per_unit", -1)) != cfg["scale_m_per_unit"]
                ):
                    raise ValueError(f"Peg cannot train: {name} CAD ordering/scale provenance mismatch")
                cad[name] = load_mesh_surface_points(
                    Path(repo) / cfg["mesh"],
                    scale_m_per_unit=cfg["scale_m_per_unit"],
                    seed=cfg["sampling_seed"],
                    include_source_sha256=False,
                )[0]
            checks["adapter"] = "legacy_ordered_fit"
        else:
            release = json.loads((prep / "rgbtrack/release_v1.json").read_text())
            approval = json.loads((prep / "geometry/approved/manifest.json").read_text())
            if (
                manifest.get("dataset") != dataset
                or manifest.get("role_to_object") != roles
                or manifest.get("coordinate_frame") != "base_link"
                or manifest.get("length_unit") != "m"
                or release.get("dataset") != dataset
                or set(release.get("objects", [])) != set(roles.values())
                or approval.get("status") != "human_accepted"
                or manifest.get("surface_sampling_seed") != SURFACE_SAMPLING_SEED
            ):
                raise ValueError("published role, CAD, sampling or coordinate contract mismatch")
            pose_dir = prep / "rgbtrack/episodes" / f"e{episode:05d}"
            expected_run = prep / "rgbtrack/runs" / release["run_tag"] / "episodes" / f"e{episode:05d}"
            if not pose_dir.is_symlink() or pose_dir.resolve() != expected_run.resolve():
                raise ValueError("pose episode link differs from release")
            run = json.loads((pose_dir / "manifest.json").read_text())
            err = calibration_provenance_error(run.get("calibration"), expected_cal)
            if err:
                raise ValueError(f"pose calibration: {err}")
            for i, name in enumerate(release["objects"]):
                asset = approval["assets"][name]
                if float(asset.get("scale_m_per_unit", -1)) != 1:
                    raise ValueError(f"{name}: CAD metric scale mismatch")
                if (
                    not asset.get("sha256")
                    or not manifest.get("approved_meshes_sha256", {}).get(name)
                    or asset.get("sha256") != manifest.get("approved_meshes_sha256", {}).get(name)
                ):
                    hash_mismatches.append(
                        {
                            "object": name,
                            "approved": asset.get("sha256"),
                            "geometry": manifest.get("approved_meshes_sha256", {}).get(name),
                        }
                    )
                cad[name] = load_surface_points(prep / "geometry/approved" / f"{name}.glb", SURFACE_SAMPLING_SEED + i)
            checks["adapter"] = "button_ordered_fit" if dataset == "1028_press_button" else "published_sampled"
            if dataset == "1028_press_button":
                checks["pose_source"] = "cleaned Geometry v1; old PoseRecord not used (frame versions differ)"
                checks["recorded_pose_sha256"] = manifest.get("pose_files_sha256_by_episode", {}).get(str(episode), {})
        fit_geometry = legacy or dataset == "1028_press_button"
        stage("cad_and_calibration")
        rows = (
            {}
            if fit_geometry
            else {
                role: selected_pose_rows(pose_dir / f"{name}_poses.jsonl", object_name=name, frames=frames)
                for role, name in roles.items()
            }
        )
        from openpi.cad_pose.geometry import fit_corresponding_rigid_pose

        valid_counts = dict.fromkeys(roles, 0)
        residuals = []
        checks["sampled_residuals"] = residuals
        checks["recorded_cad_hash_mismatches"] = hash_mismatches
        poses_for_smoke = {}
        for frame, sample in zip(frames, samples, strict=True):
            poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
            valid = np.zeros(2, bool)
            for i, (role, name) in enumerate(roles.items()):
                points, mask = sample[role]
                if fit_geometry:
                    if mask.shape != (512,):
                        raise ValueError(f"{role}: invalid geometry mask shape at frame {frame}")
                    if mask.any():
                        if not mask.all():
                            raise ValueError(f"{role}: partial fixed-point correspondence at frame {frame}")
                        poses[i], rmse = fit_corresponding_rigid_pose(cad[name], points, mask, max_rmse_m=1e-4)
                        residuals.append({"frame": frame, "role": role, "rigid_fit_rmse_m": rmse})
                        valid[i] = True
                else:
                    row = rows[role][frame]
                    if mask.shape != (512,) or bool(mask.any()) != row["valid"] or (row["valid"] and not mask.all()):
                        raise ValueError(f"frame {frame} {role}: pose/geometry validity differs")
                    if row["valid"]:
                        delta = points - cloud_from_pose(row, cad[name])
                        error = float(np.max(np.abs(delta)))
                        if not np.isfinite(error) or error > 1e-4:
                            raise ValueError(f"frame {frame} {role}: sampled ordered geometry differs by {error}m")
                        residuals.append({"frame": frame, "role": role, "max_abs_m": error})
                        poses[i] = row["T_base_link_object"]
                        valid[i] = True
                valid_counts[role] += int(valid[i])
            poses_for_smoke[frame] = (poses, valid)
        if not all(valid_counts.values()) or not poses_for_smoke:
            raise ValueError("not enough valid sampled poses for both roles")
        checks["sampled_valid_frames"] = valid_counts
        stage("three_frame_correspondence")
        if hash_mismatches:
            raise ValueError(
                "recorded CAD source hashes differ; sampled residuals are reported but source remains unconfirmed"
            )
        if do_smoke:
            import torch

            from openpi.cad_pose.model import XHandCADPoseTransformerV1
            from openpi.cad_pose.model import pose_loss
            from openpi.cad_pose.preprocess import FrontRGBDPreprocessor
            from openpi.spatial_dataset.source import RawSpatialDataset

            torch.set_num_threads(2)
            raw = RawSpatialDataset(root, episode=episode)
            preprocessor = FrontRGBDPreprocessor(Path(repo), profile)
            model = XHandCADPoseTransformerV1().eval()
            cad_t = torch.from_numpy(np.stack([cad[roles[r]] for r in ("manipulated", "target")])[None])
            smoke_frames = []
            for row, video, frame in zip(
                raw.iter_rows(selected=frames, depth_roles=("front",)),
                raw.iter_video_frames("front", selected=frames),
                frames,
                strict=True,
            ):
                if int(row["frame_index"]) != frame or video.frame_index != frame:
                    raise ValueError("RGB/depth/pose frame mismatch")
                poses, valid = poses_for_smoke[frame]
                scene, mask = preprocessor(video.rgb, row["observation.depths.cam_front"])
                with torch.no_grad():
                    pred = model(torch.from_numpy(scene[None]), torch.from_numpy(mask[None]), cad_t)
                    if not all(torch.isfinite(pred[key]).all() for key in ("rotation", "translation")):
                        raise ValueError("non-finite model smoke prediction")
                    loss = (
                        pose_loss(pred, torch.from_numpy(poses[None]), cad_t, torch.from_numpy(valid[None]))
                        if valid.any()
                        else None
                    )
                if loss is not None and not torch.isfinite(loss):
                    raise ValueError("non-finite model smoke loss")
                smoke_frames.append(
                    {"frame": frame, "loss": None if loss is None else float(loss), "scene_shape": list(scene.shape)}
                )
            if len(smoke_frames) != len(frames):
                raise ValueError("missing selected RGB-D frames")
            checks["smoke"] = smoke_frames
        stage("rgbd_and_model_smoke")
        checks["pose_hash_and_alignment"] = "sampled_pass"
        checks["profile_sha256"] = expected_cal["profile"]["sha256"]
        checks["source_hashes"] = "not recomputed; source identity and unsampled frames remain unverified"
        result["errors"].append(
            "Only three frames sampled. Existing SHA mismatch cannot be ruled out or resolved without hashing; geometric accuracy needs RGB-D review."
        )
        result["status"] = "requires_visual_review"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AssertionError,
        ImportError,
        RuntimeError,
        StopIteration,
    ) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        timings["current_stage_until_exit"] = round(time.monotonic() - started, 3)
        result["timings_seconds"] = timings
    return result


def _worker(conn, args, quick, allow_legacy_intrinsics):
    try:
        if quick:
            result = quick_audit(*args, allow_legacy_intrinsics=allow_legacy_intrinsics)
        else:
            from scripts.cad_pose.preflight import audit_task

            result = audit_task(*args)
        conn.send(result)
    except BaseException as exc:
        conn.send({"dataset": args[2], "status": "blocked", "errors": [f"{type(exc).__name__}: {exc}"], "checks": {}})
    finally:
        conn.close()


def run_timed_task(repo, data_root, dataset, profile, *, timeout=120, quick=True, allow_legacy_intrinsics=False):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_worker, args=(child, (repo, data_root, dataset, profile), quick, allow_legacy_intrinsics)
    )
    started = time.monotonic()
    process.start()
    child.close()
    try:
        if parent.poll(timeout):
            report = parent.recv()
            report["elapsed_seconds_including_worker_startup"] = round(time.monotonic() - started, 3)
            return report
        return {
            "dataset": dataset,
            "status": "timeout",
            "checks": {},
            "errors": [f"{'quick' if quick else 'full'} task exceeded {timeout}s; worker terminated"],
        }
    except EOFError:
        return {
            "dataset": dataset,
            "status": "blocked",
            "checks": {},
            "errors": ["quick worker exited without a result"],
        }
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def run_quick_task(repo, data_root, dataset, profile, *, timeout=120, allow_legacy_intrinsics=False):
    return run_timed_task(
        repo, data_root, dataset, profile, timeout=timeout, quick=True, allow_legacy_intrinsics=allow_legacy_intrinsics
    )
