#!/usr/bin/env python3
"""Read-only Button V2 geometry/policy counterfactual diagnostic; never connects to a robot.

The batch=2 replay shares the committed exporter implementation and is NOT a
claim about an external deployment runtime. Output is restricted to output/.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

TASK = "1028_press_button"
CONFIG = "pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/press_the_button"
ROLES = ("manipulated", "target")  # button, box; confirmed V2 candidate role order
OUTPUT = ROOT / "output/button_pose_v2_offline_diagnosis"


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {
        "count": int(x.size),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        **{f"over_{n}mm_fraction": float(np.mean(x > n)) for n in (5, 10, 20)},
    }


def signed_stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0}
    return {
        "count": int(x.size),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "positive_fraction": float(np.mean(x > 0)),
        "negative_fraction": float(np.mean(x < 0)),
    }


def event_proxy(force):
    """Tactile rise is an unverified contact proxy, not a button-contact label."""
    f = np.asarray(force, dtype=float)
    if len(f) < 10 or not np.isfinite(f).all():
        return None
    base = f[: max(5, len(f) // 10)]
    median = float(np.median(base))
    threshold = median + max(3 * 1.4826 * float(np.median(np.abs(base - median))), 0.05)
    above = np.flatnonzero(f > threshold)
    if len(above) < 3:
        return None
    peak = int(np.argmax(f))
    # A bounded neighborhood, not a claim of measured contact duration.
    return {"onset": int(above[0]), "peak": peak, "end": int(above[-1]), "threshold": threshold, "baseline": median}


def phase(frame, event):
    if event is None:
        return "unclassified"
    return "before" if frame < event["onset"] else ("after" if frame > event["end"] else "during_proxy")


def select_episodes(splits, available, count, seed):
    if count < 3:
        raise ValueError("Need at least 3 episodes to cover the available Pose split groups")
    rng = np.random.default_rng(seed)
    available = set(available)
    result = []
    declared = [int(ep) for name in ("train", "val", "test") for ep in splits.get(name, [])]
    if len(declared) != len(set(declared)):
        raise ValueError("Pose episode splits overlap or contain duplicate IDs")
    groups = {}
    for split in ("train", "val", "test"):
        groups[split] = list(map(int, splits.get(split, [])))
        groups[split] = list(rng.permutation([x for x in groups[split] if x in available]))
        if groups[split]:
            result.append((int(groups[split].pop()), split))
    while len(result) < count and any(groups.values()):
        for split in ("train", "val", "test"):
            if groups[split] and len(result) < count:
                result.append((int(groups[split].pop()), split))
    if len(result) < count:
        raise ValueError(f"Only {len(result)} episodes available, requested {count}")
    return result


def choose_frames(n, event, count=3):
    if event is None:
        return sorted(set(map(int, np.linspace(0, n - 1, count))))
    return sorted(
        {max(0, min(n - 1, x)) for x in (event["onset"] - 3, event["onset"], event["onset"] + 3, event["end"] + 3)}
    )


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def raw_inputs(raw, ids, width, height):
    import cv2

    ids = set(map(int, ids))
    rgb = {f.frame_index: f.rgb for f in raw.iter_video_frames("front", selected=ids)}
    result = {}
    for row in raw.iter_rows(selected=ids, depth_roles=("front",)):
        i = int(row["frame_index"])
        d = np.asarray(row["observation.depths.cam_front"])
        im = np.asarray(rgb[i])
        if im.dtype != np.uint8 or d.dtype != np.uint16:
            raise ValueError(f"e{raw.episode} f{i}: expected RGB uint8 and raw unregistered depth uint16")
        result[i] = (
            cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA),
            cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST),
        )
    if set(result) != ids:
        raise ValueError(f"e{raw.episode}: incomplete RGB-D frame selection")
    return result


def predict_two(model, ckpt, cad, rgb, depth, device):
    import torch

    r = torch.from_numpy(np.repeat(rgb[None], 2, axis=0).copy()).to(device).permute(0, 3, 1, 2).float() / 255
    d = torch.from_numpy(np.repeat(depth[None], 2, axis=0).copy()).to(device).unsqueeze(1).float()
    d = (d * float(ckpt["depth_scale"])).clamp(0, 2) / 2
    c = torch.from_numpy(np.asarray(cad, np.float32).copy()).to(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16),
    ):
        t, rot = model(r, d, c)
    t, rot = t.float().cpu().numpy(), rot.float().cpu().numpy()
    xyz = np.einsum("rij,rpj->rpi", rot, cad) + t[:, None, :]
    if not np.isfinite(xyz).all():
        raise ValueError("Pose V2 replay generated nonfinite geometry")
    return xyz, t, rot


def rotation_difference(a, b):
    cos = np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def checkpoint_scan(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Policy checkpoint root not found: {root}")
    candidates = [
        p
        for p in root.glob("*/*")
        if p.is_dir() and p.name.isdigit() and (p / "params").is_dir() and (p / "assets").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete <experiment>/<step>/params+assets under {root}")
    # Prefer the most recently modified run, then its highest complete step.
    by_run = {}
    for p in candidates:
        by_run.setdefault(p.parent, []).append(p)
    run = max(by_run, key=lambda p: max(x.stat().st_mtime_ns for x in by_run[p]))
    return max(by_run[run], key=lambda p: int(p.name)), [str(x) for x in sorted(candidates)]


def make_training_input(raw, chain, horizon):
    """Preserve exactly the joined spatial packet; never pass dataset actions."""
    spatial = raw["spatial"]
    inp = dict(raw)
    inp.pop("spatial")
    inp["action"] = np.zeros((horizon, 18), np.float32)
    transformed = chain(inp)
    transformed.pop("actions", None)
    transformed["spatial"] = spatial
    return transformed


def make_policy(cfg, data, checkpoint):
    """Use exact training transforms once and checkpoint norm stats; no online runtime shortcut."""
    import jax.numpy as jnp

    from openpi import transforms as tr
    from openpi.models import model as model_lib
    from openpi.policies.policy import Policy
    from openpi.training.checkpoints import load_norm_stats

    norm = load_norm_stats(checkpoint / "assets", data.asset_id)
    if norm is None:
        raise ValueError("Checkpoint has no norm stats")
    model = cfg.model.load(
        model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16), remove_extra_params=False
    )
    # Repack contract expects action but it is deliberately supplied as zeros;
    # remove it before model input. Actual future labels are never passed.
    chain = tr.compose(
        [
            *data.repack_transforms.inputs,
            *data.data_transforms.inputs,
            tr.Normalize(norm, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )

    def convert(raw):
        return make_training_input(raw, chain, cfg.model.action_horizon)

    return Policy(
        model,
        transforms=[convert],
        output_transforms=[
            *data.model_transforms.outputs,
            tr.Unnormalize(norm, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ],
    )


def run(args):
    import torch

    import cad_pose_rgbd_single_v2 as v2
    from openpi.cad_pose.geometry import fit_corresponding_rigid_pose
    from openpi.cad_pose.v2_native import checkpoint_metadata
    from openpi.cad_pose.v2_native import sha256
    from openpi.cad_pose.v2_native import validate_native_manifest
    from openpi.object_geometry_dataset.derived import ObjectGeometryDerivedDataset
    from openpi.spatial.kinematics import RobotKinematics
    from openpi.spatial.kinematics import joint_positions_from_state
    from openpi.spatial.kinematics import load_state_mapping_csv
    from openpi.spatial_dataset.derived import SpatialDerivedDataset
    from openpi.spatial_dataset.source import RawSpatialDataset
    from scripts.deploy.run_policy_dry_run import _ground_truth_actions
    from scripts.deploy.tactile_motion_probe import _fk_intent

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "data" / TASK
    cache = ROOT / "output/cad_pose_rgbd_v2/cache"
    checkpoint = ROOT / "output/cad_pose_rgbd_v2/train_20261109_162628/best.pt"
    sidecar = ObjectGeometryDerivedDataset(dataset, version="pretrained_pose_v2")
    spatial = SpatialDerivedDataset(dataset, version="v1")
    if len(sidecar.episode_ids) != 88:
        raise ValueError(f"Expected cleaned Button 88 episodes; got {len(sidecar.episode_ids)}")
    splits = json.loads((cache / "episode_splits.json").read_text())[TASK]
    selected = select_episodes(splits, sidecar.episode_ids, args.episodes, args.seed)
    cad = np.load(cache / TASK / "cad.npy", allow_pickle=False).astype(np.float32)
    if cad.shape != (2, 512, 3) or not np.isfinite(cad).all():
        raise ValueError("Expected finite ordered CAD (2,512,3)")
    ckpt = checkpoint_metadata(checkpoint)
    manifest = sidecar.manifest
    if (
        manifest.get("pose_checkpoint_sha256") != sha256(checkpoint)
        or manifest.get("cad_file_sha256") != sha256(cache / TASK / "cad.npy")
        or manifest.get("coordinate_frame") != "base_link"
        or manifest.get("length_unit") != "m"
        or manifest.get("role_to_object") != {"manipulated": "button", "target": "box"}
        or manifest.get("gt_used") is not False
        or manifest.get("checkpoint_script_sha256") != ckpt["script_sha256"]
        or ckpt["script_sha256"] != sha256(ROOT / "cad_pose_rgbd_single_v2.py")
    ):
        raise ValueError("Sidecar Pose checkpoint/CAD/roles/frame or prediction provenance mismatch")
    # Validate the native contract and small metadata without auditing every episode.
    validate_native_manifest(manifest, dataset, checkpoint, cache)
    for ep, _ in selected:
        shard = f"shard_e{ep:05d}"
        for role in ROLES:
            for suffix in ("xyz_m", "mask"):
                field = f"{role}_{suffix}"
                expected = manifest.get("shard_sha256", {}).get(shard, {}).get(field)
                if not expected or sha256(sidecar.root / shard / f"{field}.npy") != expected:
                    raise ValueError(f"e{ep}: sidecar {field} hash mismatch")
    model = v2.SingleCadRGBDPose().to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    fk = RobotKinematics(ROOT / "configs/ur7e_xhand/ur7e_xhand_verified.urdf")
    mapping = load_state_mapping_csv(ROOT / "configs/ur7e_xhand/state_mapping.csv")
    stability, online, demo, selected_items, replay, labels_by_episode = [], [], [], [], {}, {}
    results = {
        "status": "running",
        "seed": args.seed,
        "episodes": selected,
        "pose_split_note": "Pose cache episode split; not a verified pi0 train/validation split",
        "contact": "unverified tactile force proxy; first 10% baseline + max(3*MAD,0.05), not a physical contact label",
        "online_runtime": "not available in committed policy factory; batch=2 equivalent exporter replay, not actual deployment runtime",
        "pose_checkpoint_sha256": sha256(checkpoint),
        "cad_sha256": sha256(cache / TASK / "cad.npy"),
        "device": str(device),
        "bf16": device.type == "cuda",
        "batch_size": 2,
        "events": {},
        "preprocessing": {
            "rgb": "decoded rgb24 uint8, INTER_AREA resize, /255",
            "depth": "unregistered uint16, INTER_NEAREST resize, scale, clip 2m, /2",
            "depth_scale": ckpt["depth_scale"],
            "width": ckpt["width"],
            "height": ckpt["height"],
        },
        "coordinate_frame": "base_link metres; camera-left is not established",
        "errors": [],
    }
    for j, (ep, split) in enumerate(selected, 1):
        print(f"[{j}/{len(selected)}] episode {ep}, {split}: geometry/tactile/FK", flush=True)
        ids = list(sidecar.frame_ids(ep))
        if ids != list(range(len(ids))):
            raise ValueError(f"e{ep}: frame ids not contiguous from 0")
        raw = RawSpatialDataset(dataset, episode=ep)
        table = raw.states()
        if list(map(int, table["frame_index"])) != ids:
            raise ValueError(f"e{ep}: raw and geometry frame index mismatch")
        action = _ground_truth_actions(dataset, ep)
        labels_by_episode[ep] = action
        states = np.asarray(table["observation.state"])
        force = []
        centers = np.empty((len(ids), 2, 3))
        hands = np.empty((len(ids), 3))
        good = np.zeros((len(ids), 2), dtype=bool)
        measured_ee = np.empty((len(ids), 3), dtype=float)
        for i in ids:
            s = spatial.get(episode_index=ep, frame_index=i)
            force.append(float(np.linalg.norm(np.asarray(s.tactile_force_base), axis=-1).sum()))
            # Geometric taxel centroid is a hand landmark, not an EE pose.
            hands[i] = np.nanmean(np.asarray(s.tactile_xyz_m), axis=0)
            g = sidecar.get(episode_index=ep, frame_index=i)
            for r, role in enumerate(ROLES):
                mask = getattr(g, role + "_mask")
                good[i, r] = bool(np.all(mask))
                centers[i, r] = getattr(g, role + "_xyz_m")[mask].mean(axis=0) if mask.any() else np.nan
        event = event_proxy(force)
        results["events"][str(ep)] = event
        keys = choose_frames(len(ids), event)
        selected_items.extend((ep, f, split, event, int(table["index"][f])) for f in keys)
        samples = sorted(set(range(0, len(ids), args.online_stride)) | set(keys))
        rgbd = raw_inputs(raw, samples, int(ckpt["width"]), int(ckpt["height"]))
        rgb_cache = np.load(cache / TASK / f"e{ep:05d}/rgb.npy", mmap_mode="r", allow_pickle=False)
        dep_cache = np.load(cache / TASK / f"e{ep:05d}/depth.npy", mmap_mode="r", allow_pickle=False)
        if len(rgb_cache) != len(ids) or len(dep_cache) != len(ids):
            raise ValueError(f"e{ep}: cache/geometry lengths differ")
        for i in ids:
            for r, role in enumerate(ROLES):
                step = (
                    float(np.linalg.norm(centers[i, r] - centers[i - 1, r]) * 1000)
                    if i and good[i, r] and good[i - 1, r]
                    else None
                )
                rel = float(np.linalg.norm(centers[i, r] - hands[i]) * 1000) if good[i, r] else None
                stability.append(
                    {
                        "episode": ep,
                        "split": split,
                        "frame": i,
                        "role": role,
                        "phase": phase(i, event),
                        "x_m": centers[i, r, 0],
                        "y_m": centers[i, r, 1],
                        "z_m": centers[i, r, 2],
                        "step_mm": step,
                        "hand_centroid_distance_mm": rel,
                        "tactile_force_sum": force[i],
                        "valid": bool(good[i, r]),
                    }
                )
            # FK both measured state and demonstrated absolute joint target.
            q = joint_positions_from_state(states[i], mapping=mapping, kinematics=fk)
            measured_ee[i] = fk.compute(q, root="base_link")["right_hand_link"][:3, 3]
            actual_step = (measured_ee[i] - measured_ee[i - 1]) * 1000 if i else np.zeros(3)
            intent = _fk_intent(fk, mapping, states[i], np.asarray([action[i]]), centers[i, 1])
            demo.append(
                {
                    "episode": ep,
                    "split": split,
                    "frame": i,
                    "phase": phase(i, event),
                    "dx_mm": intent["x_mm"][0],
                    "dy_mm": intent["y_mm"][0],
                    "dz_mm": intent["z_mm"][0],
                    "measured_x_m": measured_ee[i, 0],
                    "measured_y_m": measured_ee[i, 1],
                    "measured_z_m": measured_ee[i, 2],
                    "measured_step_x_mm": actual_step[0] if i else None,
                    "measured_step_y_mm": actual_step[1] if i else None,
                    "measured_step_z_mm": actual_step[2] if i else None,
                }
            )
        for i in samples:
            rgb, depth = rgbd[i]
            if not np.array_equal(rgb, rgb_cache[i]) or not np.array_equal(depth, dep_cache[i]):
                raise ValueError(f"e{ep} f{i}: raw RGB-D differs from export cache, cannot compare same input")
            xyz, t, rot = predict_two(model, ckpt, cad, rgb, depth, device)
            replay[(ep, i)] = xyz
            g = sidecar.get(episode_index=ep, frame_index=i)
            for r, role in enumerate(ROLES):
                old = getattr(g, role + "_xyz_m")
                mask = getattr(g, role + "_mask")
                if not np.asarray(mask).all():
                    raise ValueError(
                        f"e{ep} f{i} {role}: invalid V2 mask; refusing to conflate masks or infer visibility"
                    )
                transform, residual = fit_corresponding_rigid_pose(cad[r], old, mask, max_rmse_m=0.001)
                distances = np.linalg.norm(xyz[r] - old, axis=1) * 1000
                online.append(
                    {
                        "episode": ep,
                        "split": split,
                        "frame": i,
                        "phase": phase(i, event),
                        "role": role,
                        "translation_diff_mm": float(np.linalg.norm(t[r] - transform[:3, 3]) * 1000),
                        "rotation_diff_deg": rotation_difference(transform[:3, :3], rot[r]),
                        "point_mean_mm": float(distances.mean()),
                        "point_p95_mm": float(np.percentile(distances, 95)),
                        "centroid_diff_mm": float(np.linalg.norm(xyz[r].mean(axis=0) - old.mean(axis=0)) * 1000),
                        "ordered_fit_rmse_mm": residual * 1000,
                    }
                )
        if args.plots:
            import matplotlib as mpl

            mpl.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            for r, role in enumerate(ROLES):
                for k, axis in enumerate("xyz"):
                    ax[r].plot(centers[:, r, k] * 1000, label=axis)
                ax[r].set_ylabel(role + " center mm")
                ax[r].legend()
            ax[2].plot(force, label="tactile force sum (proxy)")
            if event:
                ax[2].axvline(event["onset"], color="red", label="proxy onset")
            ax[2].legend()
            for k, axis in enumerate("xyz"):
                ax[3].plot(measured_ee[:, k] * 1000, label=axis)
            ax[3].set_ylabel("measured FK mm")
            ax[3].legend()
            ax[3].set_xlabel("frame")
            fig.tight_layout()
            fig.savefig(out / f"episode_{ep:05d}_trajectory.png", dpi=120)
            plt.close(fig)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fields = [
        "episode",
        "split",
        "frame",
        "role",
        "phase",
        "x_m",
        "y_m",
        "z_m",
        "step_mm",
        "hand_centroid_distance_mm",
        "tactile_force_sum",
        "valid",
    ]
    write_csv(out / "pose_stability.csv", stability, fields)
    write_csv(
        out / "pose_online_vs_offline.csv",
        online,
        [
            "episode",
            "split",
            "frame",
            "phase",
            "role",
            "translation_diff_mm",
            "rotation_diff_deg",
            "point_mean_mm",
            "point_p95_mm",
            "centroid_diff_mm",
            "ordered_fit_rmse_mm",
        ],
    )
    write_csv(
        out / "demonstration_fk.csv",
        demo,
        [
            "episode",
            "split",
            "frame",
            "phase",
            "dx_mm",
            "dy_mm",
            "dz_mm",
            "measured_x_m",
            "measured_y_m",
            "measured_z_m",
            "measured_step_x_mm",
            "measured_step_y_mm",
            "measured_step_z_mm",
        ],
    )
    results["pose_stability_mm"] = {
        role: {
            p: stats(
                [
                    r["step_mm"]
                    for r in stability
                    if r["role"] == role and (p == "all" or r["phase"] == p) and r["step_mm"] is not None
                ]
            )
            for p in ("all", "before", "during_proxy", "after", "unclassified")
        }
        for role in ROLES
    }
    results["pose_replay"] = {
        role: {
            p: {
                k: stats([r[k] for r in online if r["role"] == role and r["phase"] == p])
                for k in ("centroid_diff_mm", "translation_diff_mm", "point_mean_mm")
            }
            | {
                "rotation_diff_deg": signed_stats(
                    [r["rotation_diff_deg"] for r in online if r["role"] == role and r["phase"] == p]
                )
            }
            for p in ("before", "during_proxy", "after", "unclassified")
        }
        for role in ROLES
    }
    results["geometry_jump_candidates"] = sorted(
        [r for r in stability if r["step_mm"] is not None and r["step_mm"] > 20],
        key=lambda r: r["step_mm"],
        reverse=True,
    )[:50]
    results["demonstration_base_link_fk_mm"] = {
        p: {k: signed_stats([r[k] for r in demo if r["phase"] == p]) for k in ("dx_mm", "dy_mm", "dz_mm")}
        for p in ("before", "during_proxy", "after", "unclassified")
    }
    results["demonstration_measured_fk_steps_mm"] = {
        p: {
            k: signed_stats([r[k] for r in demo if r["phase"] == p and r[k] is not None])
            for k in ("measured_step_x_mm", "measured_step_y_mm", "measured_step_z_mm")
        }
        for p in ("before", "during_proxy", "after", "unclassified")
    }
    results["policy"] = "not run"
    (out / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n")
    if args.skip_policy:
        write_csv(out / "policy_action_comparison.csv", [], ["episode", "frame", "condition", "action_path"])
        results["status"] = "partial: policy skipped by request"
    else:
        print("Loading policy and training DataLoader (may validate the full V2 provenance)...", flush=True)
        import jax

        if device.type == "cuda" and not jax.devices("gpu"):
            raise RuntimeError("JAX GPU unavailable; refusing silent CPU fallback")
        from openpi.training import config
        from openpi.training import data_loader

        cfg = config.get_config(CONFIG)
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        ds = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
        cp, candidates = checkpoint_scan(ROOT / "checkpoints" / CONFIG)
        provenance_path = cp / "assets" / data.asset_id / "pose_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Policy checkpoint lacks pose provenance: {provenance_path}")
        trained_provenance = json.loads(provenance_path.read_text())
        for key in (
            "pose_checkpoint_sha256",
            "cad_file_sha256",
            "dataset_meta_sha256",
            "preprocess",
            "candidate_provenance_sha256",
            "candidate_summary_sha256",
            "shard_sha256",
        ):
            if key not in trained_provenance or trained_provenance[key] != manifest.get(key):
                raise ValueError(f"Policy training provenance differs from current V2 sidecar: {key}")
        policy = make_policy(cfg, data, cp)
        results["policy_checkpoint"] = str(cp)
        results["checkpoint_candidates"] = candidates
        rows = []
        noise = (
            np.random.default_rng(args.seed)
            .normal(size=(cfg.model.action_horizon, cfg.model.action_dim))
            .astype(np.float32)
        )
        for ep, i, split, event, index in selected_items:
            item = dict(ds[index])
            if int(item["episode_index"]) != ep or int(item["frame_index"]) != i:
                raise ValueError(f"DataLoader index mismatch e{ep} f{i}")
            state = np.asarray(item["observation.state"])
            original = item["spatial"]["object_geometry"]
            # Freeze a past frame only; proxy onset is retrospectively selected and is not an online trigger.
            frozen_i = max(0, (event["onset"] - 1) if event and i >= event["onset"] else i)
            if event and i > event["onset"] + args.freeze_frames:
                frozen_i = i
            frozen = sidecar.get(episode_index=ep, frame_index=frozen_i).as_dict()
            variants = {"sidecar": original, "batch2_replay": dict(original), "frozen_pre_proxy": dict(original)}
            for k, role in enumerate(ROLES):
                variants["batch2_replay"][role + "_xyz_m"] = replay[(ep, i)][k]
                variants["frozen_pre_proxy"][role + "_xyz_m"] = frozen[role + "_xyz_m"]
                variants["frozen_pre_proxy"][role + "_mask"] = frozen[role + "_mask"]
            # Future labels are read only after inference has finished.
            outputs = {}
            for name, geom in variants.items():
                raw_item = dict(item)
                raw_item.pop("action", None)
                raw_item["spatial"] = dict(item["spatial"], object_geometry=geom)
                policy._rng = jax.random.key(args.seed)  # noqa: SLF001 -- paired diffusion RNG
                actions = np.asarray(policy.infer(raw_item, noise=noise)["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 18 or not np.isfinite(actions).all():
                    raise ValueError(f"Unexpected policy actions {actions.shape}")
                outputs[name] = actions
            labels = labels_by_episode[ep]
            for name, actions in outputs.items():
                path = f"action_e{ep:05d}_f{i:06d}_{name}.npy"
                np.save(out / path, actions, allow_pickle=False)
                intent = _fk_intent(fk, mapping, state, actions, original["target_xyz_m"])
                valid = [n for n in range(len(actions)) if i + n in labels]
                gt = np.stack([labels[i + n] for n in valid])
                rows.append(
                    {
                        "episode": ep,
                        "split": split,
                        "frame": i,
                        "phase": phase(i, event),
                        "condition": name,
                        "frozen_frame": frozen_i if name == "frozen_pre_proxy" else "",
                        "action_path": path,
                        "chunk_vs_sidecar_mae_rad": float(np.mean(np.abs(actions - outputs["sidecar"]))),
                        "chunk_vs_demo_mae_rad": float(np.mean(np.abs(actions[valid] - gt))),
                        "first_dx_mm": intent["x_mm"][0],
                        "first_dy_mm": intent["y_mm"][0],
                        "first_dz_mm": intent["z_mm"][0],
                        "end_dx_mm": intent["x_mm"][-1],
                        "end_dy_mm": intent["y_mm"][-1],
                        "end_dz_mm": intent["z_mm"][-1],
                    }
                )
            print(f"Policy e{ep} f{i}: three interventions complete", flush=True)
        write_csv(
            out / "policy_action_comparison.csv",
            rows,
            [
                "episode",
                "split",
                "frame",
                "phase",
                "condition",
                "frozen_frame",
                "action_path",
                "chunk_vs_sidecar_mae_rad",
                "chunk_vs_demo_mae_rad",
                "first_dx_mm",
                "first_dy_mm",
                "first_dz_mm",
                "end_dx_mm",
                "end_dy_mm",
                "end_dz_mm",
            ],
        )
        results["policy"] = {
            name: {
                "n": sum(r["condition"] == name for r in rows),
                "chunk_vs_sidecar_mae_rad": signed_stats(
                    [r["chunk_vs_sidecar_mae_rad"] for r in rows if r["condition"] == name]
                ),
                "end_base_xyz_mm_by_phase": {
                    p: {
                        k: signed_stats([r[k] for r in rows if r["condition"] == name and r["phase"] == p])
                        for k in ("end_dx_mm", "end_dy_mm", "end_dz_mm")
                    }
                    for p in ("before", "during_proxy", "after", "unclassified")
                },
            }
            for name in variants
        }
        results["status"] = "completed; human review and deployment runtime/logs still required"
    (out / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n")
    summary_lines = [
        "# Button Pose V2 offline diagnosis",
        "",
        f"Status: {results['status']}",
        "",
        "## Measurements",
        "",
        "- Temporal step statistics (mm, by role and tactile-proxy phase): `pose_stability_mm` in summary.json; "
        "individual >20 mm candidates are listed in `geometry_jump_candidates`.",
        "- Offline vs batch=2 replay: `pose_replay` in summary.json and pose_online_vs_offline.csv. "
        "The replay is sampled across each full episode; it is not the missing deployment runtime.",
        "- Policy: paired action changes and base_link FK displacements appear in `policy` and "
        "policy_action_comparison.csv. Full denormalized action chunks are saved as .npy files.",
        "- Demonstrations: signed base_link FK changes are in `demonstration_base_link_fk_mm` and "
        "demonstration_fk.csv. Recorded-state FK steps and action-target FK are separately labeled.",
        "",
        "## Interpretation limits",
        "",
        "- Tactile onset/phase is an unverified proxy, not a button-contact label. "
        "No-contact episodes are marked unclassified.",
        "- The frozen geometry is a past or current frame. The freeze window is retrospectively selected "
        "using the proxy and is not an online trigger.",
        "- Camera-left is not established from base_link axes. Pose split membership is not proof of "
        "the pi0 training/test split.",
        "- A cause, real-world movement and actual deployment-runtime consistency cannot be established "
        "from these measurements alone; review the CSV/plots and collect synchronized real-robot logs.",
    ]
    (out / "summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"Results: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--online-stride", type=int, default=10)
    p.add_argument("--freeze-frames", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-policy", action="store_true", help="run A/B/D only")
    p.add_argument("--no-plots", dest="plots", action="store_false")
    a = p.parse_args()
    if not 3 <= a.episodes <= 12 or a.online_stride < 1 or a.freeze_frames < 0:
        p.error("episodes must be 3..12, stride >= 1, freeze-frames >= 0")
    # Never mix partial or older experiment outputs with this run.
    a.output_dir = OUTPUT / (datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S_") + str(os.getpid()))
    try:
        run(a)
    except Exception as exc:
        # Keep partial measurements and failure details instead of claiming a PASS.
        import traceback

        a.output_dir.mkdir(parents=True, exist_ok=True)
        path = a.output_dir / "summary.json"
        prior = json.loads(path.read_text()) if path.is_file() else {}
        prior["status"] = "incomplete: " + str(exc)
        prior.setdefault("errors", []).append(traceback.format_exc())
        path.write_text(json.dumps(prior, indent=2) + "\n")
        (a.output_dir / "summary.md").write_text(
            "# Incomplete Button offline diagnosis\n\n"
            + str(exc)
            + "\n\nSee summary.json for the traceback. No unmeasured conclusion is implied.\n"
        )
        raise


if __name__ == "__main__":
    main()
