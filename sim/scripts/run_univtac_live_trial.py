#!/usr/bin/env python3
# ruff: noqa: E402, SLF001
"""Exercise a live UniVTAC observation/action loop with optional late slot offset.

The recorded provider is a reference controller, not a learned policy. The
websocket provider uses the OpenPI msgpack protocol and accepts a real policy
server when a compatible checkpoint is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT.parent / "openpi-sim-runtime" / "third_party" / "UniVTAC-full"
CLIENT_ROOT = PROJECT_ROOT / "packages" / "openpi-client" / "src"
sys.path[:0] = [str(SOURCE_ROOT), str(CLIENT_ROOT)]

from envs.utils.env_parser import build_task_env_cfg
from envs.utils.env_parser import load_task_config
from isaaclab.app import AppLauncher
from unproject_univtac_depth import unproject_opengl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "sim/configs/univtac_insert_hole_depth.yml")
parser.add_argument("--hdf5", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--provider", choices=("recorded", "websocket"), default="recorded")
parser.add_argument("--websocket-uri", default="ws://127.0.0.1:8000")
parser.add_argument("--offset-mm", type=float, nargs=2, metavar=("X", "Y"), default=(0.0, 0.0))
parser.add_argument("--perturb-index", type=int, default=170)
parser.add_argument("--max-actions", type=int, default=0, help="0 means all recorded actions")
parser.add_argument("--include-pointcloud", action="store_true", help="Send 1024 world points per camera")
parser.add_argument("--probe-contact", action="store_true", help="Export UIPC peg-slot normal contact primitives")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1

config, config_path = load_task_config(args.config)
launcher = AppLauncher(args)
app = launcher.app

from openpi_client import msgpack_numpy
import torch
from uipc.core import ContactSystemFeature
from uipc.geometry import Geometry
import websockets.sync.client


def array(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def snapshot(task, index: int) -> tuple[dict, dict]:
    task._update_render()
    observed = task._get_observations()
    images = {name: array(observed["observation"][name]["rgb"]) for name in ("head", "wrist")}
    tactile = dict(observed["tactile"])
    state = array(observed["embodiment"]["joint"][:9]).astype(np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid live joint state {state.shape}")
    if set(tactile) != {"left_tactile", "right_tactile"}:
        raise ValueError(f"Unexpected tactile streams: {list(tactile)}")
    request = {
        # These are the exact online keys consumed by UniVTACInputs.
        "observation/head_rgb": images["head"],
        "observation/wrist_rgb": images["wrist"],
        "observation/left_tactile_rgb": array(tactile["left_tactile"]["rgb"]),
        "observation/right_tactile_rgb": array(tactile["right_tactile"]["rgb"]),
        "observation/state": state,
        "prompt": "insert the peg into the hole",
        # Used by the recorded reference server; ignored by trained policies.
        "source_index": index,
    }
    if args.include_pointcloud:
        cloud_xyz = []
        cloud_rgb = []
        for name in ("head", "wrist"):
            camera = observed["observation"][name]
            xyz, pixel_indices = unproject_opengl(
                array(camera["depth"]), array(camera["intrinsic"]), array(camera["pose_w_opengl"]),
                stride=4, max_depth_m=3.0,
            )
            if len(xyz) < 1024:
                raise ValueError(f"Only {len(xyz)} live points from {name}")
            selected = np.linspace(0, len(xyz) - 1, 1024, dtype=np.int64)
            cloud_xyz.append(xyz[selected])
            cloud_rgb.append(images[name].reshape(-1, 3)[pixel_indices[selected]])
        xyz = np.concatenate(cloud_xyz)
        rgb = np.concatenate(cloud_rgb)
        request["spatial"] = {"visual": {
            "xyz_m": xyz,
            "rgb": rgb,
            "rgb_valid": np.ones(len(xyz), dtype=bool),
            "point_mask": np.ones(len(xyz), dtype=bool),
        }}
    for key in ("observation/head_rgb", "observation/wrist_rgb", "observation/left_tactile_rgb", "observation/right_tactile_rgb"):
        image = request[key]
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError("Invalid live camera/tactile image")
    metrics = {
        "joint": state.tolist(),
        "ee": array(observed["embodiment"]["ee"][:7]).tolist(),
        "actor_pose": {name: array(pose).tolist() for name, pose in observed["actor"].items()},
        "camera_rgb_mean": {name: float(images[name].mean()) for name in images},
        "camera_valid_depth_fraction": {
            name: float(np.mean(np.isfinite(array(observed["observation"][name]["depth"]))
                             & (array(observed["observation"][name]["depth"]) > 0)))
            for name in images
        },
        "tactile_press_max_mm": {
            name: float(array(sensor["press_depth"]).max()) for name, sensor in tactile.items()
        },
        "world_pointcloud_count": len(request["spatial"]["visual"]["xyz_m"]) if args.include_pointcloud else 0,
    }
    return request, metrics


def websocket_action(connection, request: dict) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    connection.send(msgpack_numpy.packb(request))
    raw = connection.recv()
    if isinstance(raw, str):
        raise RuntimeError(f"Policy server error: {raw}")
    response = msgpack_numpy.unpackb(raw)
    actions = np.asarray(response["actions"], dtype=np.float32)
    action = actions[0] if actions.ndim == 2 else actions
    return action.copy(), (time.perf_counter() - start) * 1000


def peg_slot_contact(task, feature) -> dict:
    """Count active UIPC normal contact primitives spanning peg and slot vertices."""
    offsets = np.asarray(task.uipc_sim._surf_vertex_offsets, dtype=np.int64)
    prism_id = task.prism.body.obj_id
    slot_id = task.slot.body.obj_id
    prism_lo, prism_hi = offsets[prism_id - 1:prism_id + 1]
    slot_lo, slot_hi = offsets[slot_id - 1:slot_id + 1]
    result = {
        "primitive_count": 0,
        "normal_energy_sum": 0.0,
        "by_type": {},
        "peg_normal_barrier_force_xyz_N": [0.0, 0.0, 0.0],
        "slot_normal_barrier_force_xyz_N": [0.0, 0.0, 0.0],
        "gradient_topology_mismatch_count": 0,
        "mixed_actor_primitive_count": 0,
        "gradient_translation_residual_max_scaled": 0.0,
    }
    peg_force = np.zeros(3, dtype=np.float64)
    slot_force = np.zeros(3, dtype=np.float64)
    dt2 = float(task.physics_dt) ** 2
    # PH topology includes an implicit half-plane ID, not two surface vertex
    # ranges. Only simplex-simplex contacts can be mapped to two actors here.
    surface_pair_types = {"PP+N", "PE+N", "PT+N", "EE+N"}
    for primitive_type in feature.contact_primitive_types():
        if primitive_type not in surface_pair_types:
            continue
        geometry = Geometry()
        feature.contact_energy(primitive_type, geometry)
        topology_attr = geometry.instances().find("topo")
        energy_attr = geometry.instances().find("energy")
        if topology_attr is None or energy_attr is None:
            continue
        topology = np.asarray(topology_attr.view(), dtype=np.int64)
        energies = np.asarray(energy_attr.view(), dtype=np.float64).reshape(-1)
        if topology.size == 0:
            continue
        topology = topology.reshape(len(energies), -1)
        has_prism = np.any((topology >= prism_lo) & (topology < prism_hi), axis=1)
        has_slot = np.any((topology >= slot_lo) & (topology < slot_hi), axis=1)
        peg_vertices = (topology >= prism_lo) & (topology < prism_hi)
        slot_vertices = (topology >= slot_lo) & (topology < slot_hi)
        valid_pair_vertices = np.all(peg_vertices | slot_vertices, axis=1)
        result["mixed_actor_primitive_count"] += int(np.count_nonzero(
            has_prism & has_slot & ~valid_pair_vertices & np.isfinite(energies) & (energies > 0)
        ))
        active = has_prism & has_slot & valid_pair_vertices & np.isfinite(energies) & (energies > 0)
        count = int(np.count_nonzero(active))
        result["by_type"][primitive_type] = count
        result["primitive_count"] += count
        result["normal_energy_sum"] += float(energies[active].sum())
        if count:
            gradient_geometry = Geometry()
            feature.contact_gradient(primitive_type, gradient_geometry)
            index_attr = gradient_geometry.instances().find("i")
            gradient_attr = gradient_geometry.instances().find("grad")
            if index_attr is None or gradient_attr is None:
                raise RuntimeError(f"Missing UIPC contact gradient for {primitive_type}")
            gradient_indices = np.asarray(index_attr.view(), dtype=np.int64).reshape(topology.shape)
            gradient_values = np.asarray(gradient_attr.view(), dtype=np.float64).reshape(*topology.shape, 3)
            mismatches = int(np.count_nonzero(np.any(gradient_indices != topology, axis=1)))
            result["gradient_topology_mismatch_count"] += mismatches
            if mismatches:
                raise RuntimeError(f"UIPC contact topology/gradient index mismatch in {primitive_type}")
            translation_residual = np.linalg.norm(gradient_values[active].sum(axis=1), axis=1)
            typical_gradient = np.linalg.norm(gradient_values[active], axis=2).sum(axis=1)
            relative_translation_residual = translation_residual / np.maximum(typical_gradient, 1e-12)
            result["gradient_translation_residual_max_scaled"] = max(
                result["gradient_translation_residual_max_scaled"],
                float(np.max(relative_translation_residual)),
            )
            peg_force -= np.sum(gradient_values[active] * peg_vertices[active, :, None], axis=(0, 1)) / dt2
            slot_force -= np.sum(gradient_values[active] * slot_vertices[active, :, None], axis=(0, 1)) / dt2
    result["peg_normal_barrier_force_xyz_N"] = peg_force.tolist()
    result["slot_normal_barrier_force_xyz_N"] = slot_force.tolist()
    result["peg_normal_barrier_force_magnitude_N"] = float(np.linalg.norm(peg_force))
    result["action_reaction_residual_N"] = float(np.linalg.norm(peg_force + slot_force))
    return result


def main() -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "pass": False, "stage": "launch", "seed": args.seed,
        "provider": args.provider, "source_hdf5": str(args.hdf5.resolve()),
        "config": str(config_path), "offset_mm": list(args.offset_mm),
        "perturb_index": args.perturb_index, "trace": [],
        "request_schema": "univtac_policy_raw_v1",
        "world_pointcloud_in_request": args.include_pointcloud,
        "contact_truth_available": False,
        "contact_measurement_note": "No peg-slot contact pair/force is exported unless --probe-contact succeeds; actor pose and gripper GelSight indentation are diagnostic proxies only.",
    }
    task = None
    connection = None
    try:
        with h5py.File(args.hdf5, "r") as h5:
            recorded = np.asarray(h5["embodiment/joint"][:, :9], dtype=np.float32)
            recorded_steps = np.asarray(h5["step"])
        count = min(len(recorded), args.max_actions or len(recorded))
        if count < 1 or args.perturb_index < 0 or args.perturb_index >= count:
            raise ValueError(f"Invalid action count/perturb index: {count}/{args.perturb_index}")
        module, cfg, _, _ = build_task_env_cfg("insert_hole", config, args.config.stem, "eval", save_dir=output.parent / "scratch")
        cfg.reset_time_limit = 600.0
        cfg.save_frequency = 0
        cfg.video_frequency = 0
        cfg.step_lim = max(cfg.step_lim, count)
        task = module.Task(cfg, mode="eval")
        task.reset(seed=args.seed)
        report["stage"] = "reset"
        report["action_count_requested"] = count
        report["configured_decimation"] = cfg.decimation
        contact_feature = None
        if args.probe_contact:
            contact_feature = task.uipc_sim.world.features().find(ContactSystemFeature)
            if contact_feature is None:
                raise RuntimeError("UIPC ContactSystemFeature not available")
            report["contact_primitive_types"] = list(contact_feature.contact_primitive_types())
        if args.provider == "websocket":
            connection = websockets.sync.client.connect(args.websocket_uri, compression=None, max_size=None, open_timeout=10)
            report["server_metadata"] = msgpack_numpy.unpackb(connection.recv())
        for index in range(count):
            perturb_applied = index == args.perturb_index and any(args.offset_mm)
            if perturb_applied:
                pose = task.slot.get_pose()
                pose[:2] = pose[:2] + np.asarray(args.offset_mm, dtype=np.float64) * 0.001
                task.slot.set_pose(pose)
            request, before = snapshot(task, index)
            if args.provider == "websocket":
                action, latency_ms = websocket_action(connection, request)
            else:
                action, latency_ms = recorded[index], 0.0
            if action.shape != (9,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid 9D policy action at {index}: {action.shape}")
            exec_success, latched_success = task.take_action(
                torch.as_tensor(action[:8], dtype=torch.float32, device=task.device),
                action_type="qpos", stop_on_success=False, is_save=False,
            )
            after = array(task._get_observations()["embodiment"]["joint"][:9])
            row = {
                "index": index, "source_step": int(recorded_steps[index]),
                "observation": before,
                "target_joint": action.tolist(),
                "recorded_joint": recorded[index].tolist(),
                "real_joint_after": after.tolist(),
                "joint_error_abs_max": float(np.max(np.abs(after - action))),
                "inference_latency_ms": latency_ms,
                "perturb_applied": perturb_applied,
                "exec_success": bool(exec_success),
                "success_latched": bool(latched_success),
            }
            if contact_feature is not None and index >= max(0, args.perturb_index - 5):
                row["peg_slot_contact"] = peg_slot_contact(task, contact_feature)
            report["trace"].append(row)
            if index % 25 == 0 or perturb_applied:
                print(f"trial step {index}/{count}: max_joint_error={row['joint_error_abs_max']:.6f}", flush=True)
        report["stage"] = "completed"
        report["terminal_success"] = bool(task.check_success())
        report["success_latched"] = bool(task.eval_success)
        report["all_actions_executed"] = bool(task.take_action_cnt == count)
        report["joint_mae"] = float(np.mean([
            np.mean(np.abs(np.asarray(row["real_joint_after"]) - np.asarray(row["target_joint"])))
            for row in report["trace"]
        ]))
        if contact_feature is not None:
            report["contact_truth_available"] = True
            report["contact_measurement_note"] = (
                "UIPC active PP/PE/PT/EE normal-contact primitive topology contains only prism and slot surface "
                "vertices and spans both actors; PH implicit-plane and mixed-actor contacts are excluded; "
                "normal_energy is contact potential energy scaled by dt^2; selected negative gradient/dt^2 "
                "is a provisional normal barrier-force estimate, not independently calibrated contact force."
            )
            report["peg_slot_contact_primitive_total"] = sum(
                row["peg_slot_contact"]["primitive_count"] for row in report["trace"]
                if "peg_slot_contact" in row
            )
        report["pass"] = report["all_actions_executed"] and all(row["exec_success"] for row in report["trace"])
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        if connection is not None:
            connection.close()
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "trace"}, indent=2), flush=True)
        if task is not None:
            task.close()
        app.close()


if __name__ == "__main__":
    main()
