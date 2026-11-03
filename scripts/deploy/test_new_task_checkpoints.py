"""Offline deployment audit of the eight 1028/1030 exports; never connects to a robot.

Run from the checkout with PYTHONPATH=src and the OpenPI Python environment.
Each checkpoint runs in a fresh subprocess to release JAX device allocations.
Task configs are reconstructed from existing configs, not claimed training configs.
Strict parameter auditing prevents silently discarding unexpected trained weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("1028_press_button", "1030_yellow_socket")
BASES = {
    "pi0_rgb": "pi0_xhand_rgb_baseline",
    "pi05_rgb": "pi05_xhand_rgb_baseline",
    "pi0_st": "pi0_xhand_spatial_structured_suffix_front",
    "pi0_full": "pi0_xhand_spatial_structured_suffix_front_relational",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def deploy_client():
    spec = importlib.util.spec_from_file_location(
        "offline_deploy_client", ROOT / "deploy/ur7e_xhand_deploy_3d_tactile_multi.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_config(task, method, checkpoint):
    from openpi.training import config

    family, variant = method.split("_")
    name = f"{family}_xhand_{task}_{variant}_v2"
    try:
        return config.get_config(name)
    except ValueError:
        pass
    base = config.get_config(BASES[method])
    asset_files = list(checkpoint.glob("*/norm_stats.json"))
    if len(asset_files) != 1:
        raise ValueError(f"Expected exactly one normalization asset: {asset_files}")
    model = base.model
    # The export metadata explicitly contains VLM LoRA and no expert LoRA.
    if method in ("pi0_st", "pi0_full"):
        model = dataclasses.replace(model, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m")
    data = dataclasses.replace(
        base.data,
        repo_id=str(ROOT / "data" / task),
        assets=config.AssetsConfig(assets_dir=str(checkpoint), asset_id=asset_files[0].parent.name),
    )
    if data.spatial is not None:
        data = dataclasses.replace(
            data, spatial=dataclasses.replace(data.spatial, dataset_root=str(ROOT / "data" / task))
        )
    return dataclasses.replace(base, model=model, data=data)


def parameter_audit(cfg, checkpoint):
    import flax.nnx as nnx
    import flax.traverse_util
    import jax

    model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    expected = flax.traverse_util.flatten_dict(nnx.state(model).to_pure_dict())
    expected = {"/".join(map(str, k)): list(v.shape) for k, v in expected.items()}
    metadata = json.loads((checkpoint / "_METADATA").read_text())["tree_metadata"]
    got = {}
    for record in metadata.values():
        keys = [entry["key"] for entry in record["key_metadata"]]
        assert keys[0] == "params"
        keys = keys[1:]
        if keys[-1] == "value":
            keys = keys[:-1]
        got["/".join(keys)] = record["value_metadata"]["write_shape"]
    return {
        "expected_leaves": len(expected),
        "checkpoint_leaves": len(got),
        "missing": {k: expected[k] for k in expected.keys() - got.keys()},
        "extra": {k: got[k] for k in got.keys() - expected.keys()},
        "shape_mismatch": {
            k: {"expected": expected[k], "checkpoint": got[k]}
            for k in expected.keys() & got.keys()
            if expected[k] != got[k]
        },
    }


def prepare_packets(output, episodes):
    import av
    import pyarrow.parquet as pq

    for task in TASKS:
        dataset = ROOT / "data" / task
        task_text = json.loads((dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
        records = []
        for episode in episodes:
            parquet = dataset / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
            table = pq.read_table(parquet)
            n = table.num_rows
            # Three consecutive frames at the start, middle, and final full chunk.
            indices = sorted({start + i for start in (0, n // 2, max(0, n - 52)) for i in range(3) if start + i < n})
            decoded = {}
            for camera in ("cam_front", "cam_left", "cam_right"):
                video = (
                    dataset
                    / f"videos/chunk-{episode // 1000:03d}/observation.images.{camera}/episode_{episode:06d}.mp4"
                )
                with av.open(str(video)) as container:
                    for index, frame in enumerate(container.decode(video=0)):
                        if index in indices:
                            decoded[camera, index] = frame.to_ndarray(format="rgb24")
                        if index >= max(indices):
                            break
            actions = np.stack(table["action"].to_pylist()).astype(np.float32)
            for index in indices:
                row = table.slice(index, 1).to_pylist()[0]
                arrays = {
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "frame_index": np.int64(index),
                    "episode_index": np.int64(episode),
                    "timestamp_s": np.float64(row["timestamp"]),
                    "prompt": np.asarray(task_text),
                    "target_actions": actions[index : index + 50],
                }
                for camera in ("cam_front", "cam_left", "cam_right"):
                    arrays[f"observation.images.{camera}"] = decoded[camera, index]
                    arrays[f"observation.depths.{camera}"] = np.asarray(
                        row[f"observation.depths.{camera}"], dtype=np.uint16
                    )
                path = output / "packets" / task / f"ep{episode:03d}_f{index:06d}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                records.append(
                    {
                        "episode": episode,
                        "frame": index,
                        "path": str(path),
                        "front_depth_nonzero_fraction": float(np.mean(arrays["observation.depths.cam_front"] > 0)),
                        "raw_tactile_nonzero_fraction": float(np.mean(arrays["observation.state"][52:] != 0)),
                    }
                )
            print(f"Prepared {task} episode {episode}: {len(indices)} samples", flush=True)
        write_json(
            output / "packets" / task / "manifest.json",
            {
                "task": task,
                "prompt": task_text,
                "records": records,
                "precomputed_spatial_used": False,
                "temporal_sampling": "sparse 3-frame segments; not a full causal episode replay",
            },
        )


def worker(task, method, output):
    from openpi.policies import policy_config
    from openpi.policies.spatial_online import XHandSpatialOnlinePreprocess

    family, variant = method.split("_")
    candidates = list((ROOT / "checkpoints" / f"{family}_xhand_{task}_{variant}_v2").glob("*/19999"))
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous/missing checkpoints: {candidates}")
    checkpoint = candidates[0]
    dest = output / f"{task}_{method}"
    result = {
        "task": task,
        "method": method,
        "checkpoint": str(checkpoint),
        "robot_command_sent": False,
        "base_config": BASES[method],
        "status": "STARTED",
        "samples": [],
    }
    try:
        cfg = make_config(task, method, checkpoint)
        result["resolved_config"] = cfg.name
        result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        result["model_config"] = repr(cfg.model)
        stats = next(checkpoint.glob("*/norm_stats.json"))
        result["norm_stats"] = {
            "path": str(stats),
            "sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "quantile_norm": cfg.model.model_type.value != "pi0",
        }
        result["parameter_audit"] = parameter_audit(cfg, checkpoint)
        client = deploy_client()
        args = SimpleNamespace(
            spatial_cameras=None, prompt=None, task="", fps=15, realsense_height=480, realsense_width=640
        )
        client.configure_input_mode_from_metadata(args, cfg.policy_metadata)
        result["metadata"] = cfg.policy_metadata
        result["spatial_preprocessing"] = []
        packets = sorted((output / "packets" / task).glob("*.npz"))
        observations = []
        # Audit RGB-D + FK + raw tactile independently of model compatibility.
        preprocess = (
            XHandSpatialOnlinePreprocess(
                camera_roles=("front",), front_calibration_profile=cfg.data.spatial.front_calibration_profile
            )
            if variant != "rgb"
            else None
        )
        for packet in packets:
            with np.load(packet, allow_pickle=False) as raw:
                values = dict(raw)
            args.task = str(values.pop("prompt"))
            target = values.pop("target_actions")
            source = {"observation.state": values["observation.state"]}
            source.update(
                {k: v for k, v in values.items() if k.startswith(("observation.images.", "observation.depths."))}
            )
            for camera in ("cam_front", "cam_left", "cam_right"):
                source[camera] = values[f"observation.images.{camera}"]
            obs = client.build_pi0_observation(
                source,
                values["observation.state"],
                ["cam_front", "cam_left", "cam_right"],
                args,
                int(values["frame_index"]),
                int(values["frame_index"]),
            )
            obs["episode_index"] = values["episode_index"]
            obs["timestamp_s"] = values["timestamp_s"]
            assert "spatial" not in obs
            assert "object_geometry" not in obs
            if preprocess is not None:
                start = time.perf_counter()
                spatial = preprocess(obs)["spatial"]
                for branch in ("visual", "tactile"):
                    assert np.isfinite(spatial[branch]["xyz_m"]).all()
                assert np.isfinite(spatial["tactile"]["force"]).all()
                result["spatial_preprocessing"].append(
                    {
                        "packet": packet.name,
                        "ms": (time.perf_counter() - start) * 1000,
                        "visual_shape": list(spatial["visual"]["xyz_m"].shape),
                        "tactile_shape": list(spatial["tactile"]["xyz_m"].shape),
                        "rgb_valid_count": int(spatial["visual"]["rgb_valid"].sum()),
                        "force_norm_max": float(spatial["tactile"]["force_norm"].max()),
                    }
                )
            observations.append((packet.name, obs, target))
        write_json(dest / "result.json", result)
        print(f"{task} {method}: loading trained checkpoint", flush=True)
        start = time.perf_counter()
        policy = policy_config.create_trained_policy(cfg, checkpoint)
        result["load_s"] = time.perf_counter() - start
        audit = result["parameter_audit"]
        if audit["missing"] or audit["extra"] or audit["shape_mismatch"]:
            raise ValueError("Strict parameter audit failed; refusing partial model inference")
        prediction_arrays = {}
        for name, obs, target in observations:
            start = time.perf_counter()
            prediction = policy.infer(obs)
            wall_ms = (time.perf_counter() - start) * 1000
            action = np.asarray(prediction["actions"])
            assert action.shape == (cfg.model.action_horizon, 18), action.shape
            assert np.isfinite(action).all()
            residual = action[: len(target)] - target
            record = {
                "packet": name,
                "wall_ms": wall_ms,
                "action_shape": list(action.shape),
                "chunk_mae_rad": float(np.abs(residual).mean()),
                "arm_mae_rad": float(np.abs(residual[:, :6]).mean()),
                "hand_mae_rad": float(np.abs(residual[:, 6:]).mean()),
                "first_action_mae_rad": float(np.abs(residual[0]).mean()),
            }
            result["samples"].append(record)
            prediction_arrays[name.removesuffix(".npz")] = action
            print(f"{task} {method} {name}: {wall_ms:.1f}ms MAE={record['chunk_mae_rad']:.4f}", flush=True)
        np.savez_compressed(dest / "actions.npz", **prediction_arrays)
        result["status"] = "PASS"
        result["summary"] = {
            "count": len(result["samples"]),
            "first_infer_ms": result["samples"][0]["wall_ms"],
            "warm_median_ms": float(np.median([r["wall_ms"] for r in result["samples"][1:]])),
            **{
                k: float(np.mean([r[k] for r in result["samples"]]))
                for k in ("chunk_mae_rad", "arm_mae_rad", "hand_mae_rad", "first_action_mae_rad")
            },
        }
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    write_json(dest / "result.json", result)


def main():
    global ROOT  # noqa: PLW0603 - choose the isolated checkout for all worker helpers.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/deploy/new_tasks_20261102")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 50, 99])
    parser.add_argument("--worker", nargs=2, metavar=("TASK", "METHOD"))
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    ROOT = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(*args.worker, output)
        return
    if not args.skip_prepare:
        prepare_packets(output, args.episodes)
    results = []
    for task in TASKS:
        for method in BASES:
            dest = output / f"{task}_{method}"
            dest.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            with (dest / "run.log").open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--repo-root",
                        str(ROOT),
                        "--output-dir",
                        str(output),
                        "--worker",
                        task,
                        method,
                    ],
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            path = dest / "result.json"
            result = (
                json.loads(path.read_text())
                if path.exists()
                else {"task": task, "method": method, "status": "PROCESS_FAILED"}
            )
            result["returncode"] = proc.returncode
            results.append(result)
            write_json(output / "summary.json", results)
            print(f"{task} {method}: {result['status']}", flush=True)
    raise SystemExit(0 if all(r["status"] == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
