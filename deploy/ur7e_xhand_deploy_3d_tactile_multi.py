#!/usr/bin/env python3 
"""
Deploy the current OpenPI / pi0 policy on the UR7e + XHand robot.

This client runs on the robot machine. It sends raw online observations to the
policy server. The exact payload is selected from the server policy metadata:
the native RGB baseline sends RGB and state only, while the spatial policies
also send the configured depth images. The server builds the RGB-D point cloud
and tactile spatial input with the same SpatialPreprocessor used during
training.

Useful safety/debug flags:

# Only run inference and print action chunks; do not send actions to the robot.
--dry-run

# Validate dataset/schema wiring and exit before connecting to robot.
--check-config

# Record raw model requests, action chunks, and per-frame control data.
--record-dir ./diagnostic_runs

Render a completed run offline with deploy/render_pi0_diagnostics.py.


sudo -E "$(which python)" \
  deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip 192.168.1.100 \
  --server-port 8990 \
  --dataset-dir /path/to/dataset \
  --record-dir ./diagnostic_runs \
  --duration 60 \
  --dry-run \
  --yes
  
uv run --group dev python deploy/render_pi0_diagnostics.py \
  diagnostic_runs/生成的运行目录 \
  --video
"""

import argparse
import functools
import importlib
import io
import json
import logging
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

# SELF-CONTAINED DEPLOYMENT CONTRACT: this file is copied by itself to the
# robot-side LeRobot checkout. Keep the safety guard embedded here; no project-
# local sibling module is importable on that host. Tests in the source checkout
# re-export these same classes from deploy/contact_safety.py.
TACTILE_START = 52
FINGERS = 5
FINGER_STRIDE = 384
TAXELS = 120
FORCE_OFFSET = 24
EXPECTED_STATE = TACTILE_START + FINGERS * FINGER_STRIDE


class SafetyTrip(RuntimeError):
    """Latched unsafe or unavailable sensor reading; stop the entire run."""


def _positive_scalar(value, name):
    v=float(value)
    if not np.isfinite(v) or v<=0:
        raise ValueError(f"{name} must be a positive finite value")
    return v


def load_safety_limits(path: str | Path) -> dict:
    """Only explicitly reviewed hardware limits can enable robot actuation."""
    path=Path(path).expanduser().resolve()
    cfg=json.loads(path.read_text(encoding="utf-8"))
    required={
        "schema_version","reviewed_for_glass_task",
        "tactile_peak_limits_raw","tactile_emergency_step_change_limits_raw",
        "tactile_replan_step_change_limits_raw", "replan_enabled",
        "replan_min_interval_s", "arm_tcp_force_norm_limit_n",
        "arm_tcp_torque_norm_limit_nm", "max_observation_gap_s"
    }
    if not isinstance(cfg,dict) or set(cfg)!=required or cfg["schema_version"]!=1:
        raise ValueError(f"Safety config must contain exactly {sorted(required)}")
    if cfg["reviewed_for_glass_task"] is not True:
        raise ValueError("Safety configuration is not explicitly reviewed for glass manipulation")
    # Live UR7eXHand.get_observation() provides two separate vectors.
    # Their semantics are fixed at the device interface: [Fx,Fy,Fz] in N,
    # [Tx,Ty,Tz] in Nm. Never accept a configurable hand torque substitute.
    for key in ("tactile_peak_limits_raw","tactile_emergency_step_change_limits_raw",
                "tactile_replan_step_change_limits_raw"):
        v=np.asarray(cfg[key],dtype=float)
        if v.shape!=(FINGERS,) or not np.isfinite(v).all() or np.any(v<=0):
            raise ValueError(f"{key} must specify five positive raw-sensor thresholds")
    emergency=np.asarray(cfg["tactile_emergency_step_change_limits_raw"],dtype=float)
    replan=np.asarray(cfg["tactile_replan_step_change_limits_raw"],dtype=float)
    if np.any(replan>=emergency):
        raise ValueError("All replan thresholds must be strictly below emergency step-change thresholds")
    if type(cfg["replan_enabled"]) is not bool:
        raise ValueError("replan_enabled must be a boolean")
    interval=float(cfg["replan_min_interval_s"])
    if not np.isfinite(interval) or interval<0:
        raise ValueError("replan_min_interval_s must be a finite nonnegative number")
    for key in ("arm_tcp_force_norm_limit_n","arm_tcp_torque_norm_limit_nm","max_observation_gap_s"):
        _positive_scalar(cfg[key],key)
    return cfg


def raw_tactile_finger_peaks(state) -> np.ndarray:
    x=np.asarray(state,dtype=np.float64)
    if x.shape!=(EXPECTED_STATE,) or not np.isfinite(x).all():
        raise SafetyTrip(f"Missing/nonfinite canonical 1972-value XHand state: {x.shape}")
    blocks=x[TACTILE_START:].reshape(FINGERS,FINGER_STRIDE)
    forces=blocks[:,FORCE_OFFSET:].reshape(FINGERS,TAXELS,3)
    return np.linalg.norm(forces,axis=-1).max(axis=1)


class ContactSafetyGuard:
    def __init__(self, config: Mapping, *, clock=time.monotonic):
        self.cfg=dict(config)
        self.clock=clock
        self.last_time=None
        self.last_peaks=None
        self.tripped=None
        self.last_replan_time=None
        self.replan_events=0

    def check(self, *, observation: Mapping, state):
        if self.tripped is not None:
            raise SafetyTrip(f"Safety already latched: {self.tripped}")
        try:
            now=float(self.clock())
            if not np.isfinite(now):
                raise SafetyTrip("Safety monotonic clock invalid")
            if self.last_time is not None and now-self.last_time>float(self.cfg["max_observation_gap_s"]):
                raise SafetyTrip(f"Safety polling gap {now-self.last_time:.3f}s exceeds configured maximum")
            peaks=raw_tactile_finger_peaks(state)
            limits=np.asarray(self.cfg["tactile_peak_limits_raw"],dtype=float)
            overflow=np.flatnonzero(peaks>=limits)
            if len(overflow):
                i=int(overflow[0])
                raise SafetyTrip(f"tactile_peak finger={i} raw={peaks[i]:.5g} >= {limits[i]:.5g}")
            if self.last_peaks is not None:
                step=np.abs(peaks-self.last_peaks)
                emergency=np.asarray(self.cfg["tactile_emergency_step_change_limits_raw"],dtype=float)
                jumps=np.flatnonzero(step>=emergency)
                if len(jumps):
                    i=int(jumps[0])
                    raise SafetyTrip(f"tactile_emergency_step_change finger={i} raw_delta={step[i]:.5g} >= {emergency[i]:.5g}")
            else:
                step=np.zeros(FINGERS,dtype=float)
            for key in ("arm_tcp_force","arm_tcp_torque"):
                if key not in observation:
                    raise SafetyTrip(f"Required measured UR TCP sensor missing: {key}; "
                                     "XHand hand_joint_i.torque is motor current, NOT UR torque")
            tcp_force=np.asarray(observation["arm_tcp_force"],dtype=np.float64)
            tcp_torque=np.asarray(observation["arm_tcp_torque"],dtype=np.float64)
            if tcp_force.shape!=(3,) or tcp_torque.shape!=(3,):
                raise SafetyTrip(f"Invalid UR TCP force/torque shapes: {tcp_force.shape}, {tcp_torque.shape}")
            if not np.isfinite(tcp_force).all() or not np.isfinite(tcp_torque).all():
                raise SafetyTrip("Invalid non-finite UR TCP force/torque")
            force=float(np.linalg.norm(tcp_force))
            torque=float(np.linalg.norm(tcp_torque))
            if force>=float(self.cfg["arm_tcp_force_norm_limit_n"]):
                raise SafetyTrip(f"arm_tcp_force_norm {force:.5g} N exceeds threshold")
            if torque>=float(self.cfg["arm_tcp_torque_norm_limit_nm"]):
                raise SafetyTrip(f"arm_tcp_torque_norm {torque:.5g} Nm exceeds threshold")
            replan_fingers=[]
            if bool(self.cfg.get("replan_enabled",False)) and self.last_peaks is not None:
                replan_limit=np.asarray(self.cfg["tactile_replan_step_change_limits_raw"],dtype=float)
                events=np.flatnonzero(step>=replan_limit)
                if len(events) and (
                    self.last_replan_time is None
                    or now-self.last_replan_time>=float(self.cfg["replan_min_interval_s"])
                ):
                    replan_fingers=[int(i) for i in events]
                    self.last_replan_time=now
                    self.replan_events+=1
            self.last_time=now
            self.last_peaks=peaks.copy()
            return {"tactile_finger_peaks_raw":peaks.tolist(),
                    "tactile_finger_step_change_raw":step.tolist(),
                    "arm_tcp_force_norm_n":force,"arm_tcp_torque_norm_nm":torque,
                    "replan_recommended":bool(replan_fingers),
                    "replan_triggered_fingers":replan_fingers,
                    "replan_events_so_far":self.replan_events}
        except SafetyTrip as exc:
            self.tripped=str(exc)
            raise
        except (ValueError,TypeError,KeyError) as exc:
            self.tripped=f"Safety sensor failure: {exc}"
            raise SafetyTrip(self.tripped) from exc
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_DATASET_ROOT = REPO_ROOT / "data"
DEFAULT_DATASET_NAME = "grasp_pipette"
DEFAULT_TASK = "Pick up the glass funnel, align its stem with the opening of the small beaker, and insert it until it is fully seated."
DEFAULT_SERVER_PORT = 8990
TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_BLOCK_START = 52
TACTILE_RAW_FORCE_OFFSET = 24
TACTILE_TAXELS_PER_FINGER = 120
TACTILE_FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
TACTILE_SENSOR_IDS = tuple(range(0x11, 0x16))
TACTILE_FORCE_AXES = ("fx", "fy", "fz")
INDEX_SPLAY_ACTION_KEY = "hand_joint_3.pos"
INDEX_SPLAY_DEADZONE_MIN_RAD = -0.2
INDEX_SPLAY_DEADZONE_MAX_RAD = 0.0
DEBUG_INFER_PRINT_LIMIT = 3
DIAGNOSTIC_CONTROL_BLOCK_FRAMES = 30

FALLBACK_STATE_NAMES = [
    *[f"arm_joint_{i}.pos" for i in range(6)],
    *[f"arm_joint_{i}.vel" for i in range(6)],
    *[f"arm_ee_pose.{i:02d}" for i in range(16)],
    *[f"hand_joint_{i}.pos" for i in range(12)],
    *[f"hand_joint_{i}.torque" for i in range(12)],
]

FALLBACK_ACTION_NAMES = [
    *[f"arm_joint_{i}.pos" for i in range(6)],
    *[f"hand_joint_{i}.pos" for i in range(12)],
]


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


def summarize_array(value: np.ndarray) -> str:
    array = np.asarray(value)
    summary = f"shape={array.shape}, dtype={array.dtype}"
    if array.size == 0:
        return summary + ", empty"
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        if finite.size:
            summary += (
                f", min={float(finite.min()):.6g}, max={float(finite.max()):.6g}, "
                f"mean={float(finite.mean()):.6g}"
            )
        else:
            summary += ", no finite values"
    return summary


def debug_print_client_observation(observation: dict, *, query_index: int, mode: str) -> None:
    print(f"=== CLIENT OBS BEFORE SEND #{query_index} mode={mode} ===", flush=True)
    for key in sorted(observation):
        value = observation[key]
        if isinstance(value, np.ndarray):
            print(f"[client] {key}: {summarize_array(value)}", flush=True)
        else:
            print(f"[client] {key}: {type(value).__name__}={value}", flush=True)

    state = observation.get("observation.state", observation.get("observation/state"))
    if isinstance(state, np.ndarray):
        current_state = state[-1] if state.ndim >= 2 else state
        print(f"[client] current state summary: {summarize_array(current_state)}", flush=True)


def extract_tactile_diagnostics(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-finger raw tactile blocks without changing their vendor representation."""
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    tactile_size = TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    if state_array.size < tactile_size:
        raise ValueError(f"State has {state_array.size} values, fewer than the {tactile_size} tactile values")

    tactile_blocks = state_array[-tactile_size:].reshape(TACTILE_SENSOR_COUNT, TACTILE_BLOCK_SIZE).copy()
    raw_force = tactile_blocks[:, TACTILE_RAW_FORCE_OFFSET:].reshape(
        TACTILE_SENSOR_COUNT, TACTILE_TAXELS_PER_FINGER, 3
    ).copy()
    raw_force_norm = np.linalg.norm(raw_force, axis=-1).astype(np.float32)
    return tactile_blocks, raw_force, raw_force_norm


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class DiagnosticRecorder:
    """Low-overhead asynchronous recorder for one policy run."""

    def __init__(
        self,
        *,
        root_dir: Path,
        run_index: int,
        args: argparse.Namespace,
        state_names: list[str],
        action_names: list[str],
        camera_names: list[str],
    ) -> None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root_dir.expanduser().resolve() / f"{timestamp}_run_{run_index:03d}"
        self.inference_dir = self.run_dir / "inference"
        self.control_dir = self.run_dir / "control"
        self.inference_dir.mkdir(parents=True, exist_ok=False)
        self.control_dir.mkdir()

        self._queue: queue.Queue = queue.Queue(maxsize=args.record_queue_size)
        self._stop_token = object()
        self._worker_errors: list[str] = []
        self._dropped_records = 0
        self._dropped_paths: list[str] = []
        self._control_records: list[dict[str, object]] = []
        self._control_block_index = 0
        self._closed = False
        self._record_tactile = bool(getattr(args, "spatial_use_tactile", True))
        recorded_args = dict(vars(args))
        if recorded_args.get("api_key"):
            recorded_args["api_key"] = "<redacted>"
        self._manifest = {
            "format_version": 1,
            "status": "recording",
            "started_at": datetime.now().astimezone().isoformat(),
            "run_index": run_index,
            "script": str(Path(__file__).resolve()),
            "state_names": state_names,
            "action_names": action_names,
            "camera_names": camera_names,
            "spatial_camera_roles": list(args.spatial_cameras),
            "tactile": {
                "enabled": self._record_tactile,
                "finger_names": list(TACTILE_FINGER_NAMES),
                "block_size": TACTILE_BLOCK_SIZE,
                "raw_force_offset": TACTILE_RAW_FORCE_OFFSET,
                "taxels_per_finger": TACTILE_TAXELS_PER_FINGER,
            },
            "args": _json_safe(recorded_args),
        }
        self._write_json_atomic(self.run_dir / "manifest.json", self._manifest)
        self._worker = threading.Thread(target=self._writer_loop, name="pi0-diagnostic-writer", daemon=True)
        self._worker.start()
        print(f"Diagnostic recording: {self.run_dir}", flush=True)

    @staticmethod
    def _write_json_atomic(path: Path, value: dict) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(_json_safe(value), file, ensure_ascii=True, indent=2)
        os.replace(temp_path, path)

    @staticmethod
    def _write_npz_atomic(path: Path, arrays: dict[str, object]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "wb") as file:
            np.savez(file, **arrays)
        os.replace(temp_path, path)

    def _writer_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._stop_token:
                    return
                path, arrays = task
                self._write_npz_atomic(path, arrays)
            except Exception as exc:
                self._worker_errors.append(f"{type(exc).__name__}: {exc}")
                logging.exception("Failed to write diagnostic record")
            finally:
                self._queue.task_done()

    def _submit(self, relative_path: Path, arrays: dict[str, object], *, block: bool = False) -> None:
        if self._closed:
            return
        task = (self.run_dir / relative_path, arrays)
        if block:
            self._queue.put(task)
            return
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            self._dropped_records += 1
            self._dropped_paths.append(str(relative_path))
            logging.warning("Diagnostic queue is full; dropped %s", relative_path)

    def record_query_input(self, *, query_id: int, observation: dict, elapsed_s: float) -> None:
        state = np.asarray(observation["observation.state"], dtype=np.float32).copy()
        arrays: dict[str, object] = {
            "query_id": np.asarray(query_id, dtype=np.int64),
            "frame_index": np.asarray(observation["frame_index"], dtype=np.int64),
            "timestamp_s": np.asarray(observation["timestamp_s"], dtype=np.float64),
            "client_elapsed_s": np.asarray(elapsed_s, dtype=np.float64),
            "wall_time_ns": np.asarray(time.time_ns(), dtype=np.int64),
            "state": state,
            "prompt": np.asarray(str(observation.get("prompt", ""))),
        }
        if self._record_tactile:
            tactile_blocks, tactile_raw_force, tactile_raw_force_norm = extract_tactile_diagnostics(state)
            arrays.update(
                {
                    "tactile_blocks": tactile_blocks,
                    "tactile_raw_force": tactile_raw_force,
                    "tactile_raw_force_norm": tactile_raw_force_norm,
                }
            )
        for key, value in observation.items():
            if key.startswith("observation.images."):
                arrays[f"rgb_{key.removeprefix('observation.images.')}"] = np.asarray(value).copy()
            elif key.startswith("observation.depths."):
                arrays[f"depth_{key.removeprefix('observation.depths.')}"] = np.asarray(value).copy()
        self._submit(Path("inference") / f"query_{query_id:06d}_input.npz", arrays)

    def record_query_output(
        self,
        *,
        query_id: int,
        frame_idx: int,
        action_chunk: np.ndarray | None,
        inference_result: dict | None,
        client_roundtrip_ms: float,
        error: str | None = None,
    ) -> None:
        policy_timing = (inference_result or {}).get("policy_timing", {})
        server_timing = (inference_result or {}).get("server_timing", {})
        arrays = {
            "query_id": np.asarray(query_id, dtype=np.int64),
            "frame_index": np.asarray(frame_idx, dtype=np.int64),
            "success": np.asarray(error is None, dtype=np.bool_),
            "error": np.asarray(error or ""),
            "action_chunk": (
                np.asarray(action_chunk, dtype=np.float32).copy()
                if action_chunk is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
            "server_action_chunk": (
                np.asarray((inference_result or {}).get("actions"), dtype=np.float32).copy()
                if (inference_result or {}).get("actions") is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
            "client_roundtrip_ms": np.asarray(client_roundtrip_ms, dtype=np.float64),
            "policy_infer_ms": np.asarray(policy_timing.get("infer_ms", np.nan), dtype=np.float64),
            "server_infer_ms": np.asarray(server_timing.get("infer_ms", np.nan), dtype=np.float64),
        }
        self._submit(Path("inference") / f"query_{query_id:06d}_output.npz", arrays)

    def record_control_frame(
        self,
        *,
        frame_idx: int,
        elapsed_s: float,
        state: np.ndarray,
        raw_action: np.ndarray,
        command_action: np.ndarray,
        query_id: int,
        chunk_step: int,
        loop_ms: float,
        action_sent: bool,
        fallback: bool,
    ) -> None:
        self._control_records.append(
            {
                "frame_index": frame_idx,
                "elapsed_s": elapsed_s,
                "wall_time_ns": time.time_ns(),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "raw_action": np.asarray(raw_action, dtype=np.float32).copy(),
                "command_action": np.asarray(command_action, dtype=np.float32).copy(),
                "query_id": query_id,
                "chunk_step": chunk_step,
                "loop_ms": loop_ms,
                "action_sent": action_sent,
                "fallback": fallback,
            }
        )
        if len(self._control_records) >= DIAGNOSTIC_CONTROL_BLOCK_FRAMES:
            self._flush_control_block()

    def _flush_control_block(self, *, block: bool = False) -> None:
        if not self._control_records:
            return
        keys = tuple(self._control_records[0])
        arrays = {key: np.asarray([record[key] for record in self._control_records]) for key in keys}
        relative_path = Path("control") / f"block_{self._control_block_index:06d}.npz"
        self._control_records.clear()
        self._control_block_index += 1
        self._submit(relative_path, arrays, block=block)

    def close(self, *, frame_count: int, end_reason: str) -> None:
        if self._closed:
            return
        # The run is over, so preserving the final partial block takes priority over latency.
        self._queue.join()
        self._flush_control_block(block=True)
        self._closed = True
        self._queue.put(self._stop_token)
        self._queue.join()
        self._worker.join(timeout=5.0)
        self._manifest.update(
            {
                "status": "complete" if not self._worker_errors else "write_errors",
                "finished_at": datetime.now().astimezone().isoformat(),
                "end_reason": end_reason,
                "frame_count": frame_count,
                "control_block_count": self._control_block_index,
                "dropped_records": self._dropped_records,
                "dropped_paths": self._dropped_paths,
                "writer_errors": self._worker_errors,
            }
        )
        self._write_json_atomic(self.run_dir / "manifest.json", self._manifest)
        print(
            f"Diagnostic recording saved to {self.run_dir} "
            f"(dropped={self._dropped_records}, write_errors={len(self._worker_errors)})",
            flush=True,
        )
        renderer = SCRIPT_DIR / "render_pi0_diagnostics.py"
        if renderer.exists():
            print(f"Render with: {sys.executable} {renderer} {self.run_dir} --video", flush=True)


def import_msgpack():
    try:
        return importlib.import_module("msgpack")
    except ImportError as exc:
        raise ImportError(
            "Missing Python package 'msgpack'. Install it in the deployment environment, e.g. "
            "`pip install msgpack websockets`."
        ) from exc


def import_websockets_sync_client():
    try:
        return importlib.import_module("websockets.sync.client")
    except ImportError as exc:
        raise ImportError(
            "Missing Python package 'websockets'. Install it in the deployment environment, e.g. "
            "`pip install msgpack websockets`."
        ) from exc


def init_logging() -> None:
    try:
        lerobot_utils = importlib.import_module("lerobot.utils.utils")
        lerobot_utils.init_logging()
    except (ImportError, AttributeError):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


class WebsocketClientPolicy:
    """PI0 websocket inference client."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        proxy: str | bool | None = None,
        open_timeout: float = 10.0,
    ) -> None:
        self._msgpack = import_msgpack()
        self._websockets_sync_client = import_websockets_sync_client()
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = functools.partial(self._msgpack.Packer, default=pack_array)()
        self._unpackb = functools.partial(self._msgpack.unpackb, object_hook=unpack_array)
        self._api_key = api_key
        self._proxy = proxy
        self._open_timeout = open_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for PI0 server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = self._websockets_sync_client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=self._proxy,
                    open_timeout=self._open_timeout,
                )
                metadata = self._unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logging.info("Still waiting for PI0 server: %s", exc)
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpackb(response)

    def reset(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy UR7e + XHand PI0 policy client")

    # PI0 weights and policy inference live on the remote server.
    parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="PI0 websocket server IP or ws:// URI")
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Let websockets use HTTP(S)/SOCKS proxy environment variables. Default is direct connection.",
    )
    parser.add_argument("--server-open-timeout", type=float, default=10.0)

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset directory containing meta/info.json. Overrides --dataset-root/--dataset-name.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt sent to PI0 server.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=float, default=60.0, help="Run duration in seconds")
    parser.add_argument(
        "--run-mode",
        choices=["single", "multi"],
        default="multi",
        help="single keeps the original one-shot behavior; multi keeps hardware connected across repeated runs.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=0,
        help="Number of runs in multi mode. Use 0 to keep running until Ctrl+C.",
    )
    parser.add_argument(
        "--between-run-seconds",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--arm-ip", type=str, default="192.168.1.102")
    parser.add_argument("--arm-control-freq", type=float, default=15.0)
    parser.add_argument("--arm-max-relative-target", type=float, default=0.25)

    parser.add_argument("--hand-protocol", choices=["RS485", "EtherCAT"], default="EtherCAT")
    parser.add_argument("--hand-serial-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--hand-ethercat-interface", default="")
    parser.add_argument("--hand-control-freq", type=float, default=None)

    parser.add_argument("--realsense-front-serial", type=str, default="347622074420")
    parser.add_argument("--realsense-left-serial", type=str, default="347622075196")
    parser.add_argument("--realsense-right-serial", type=str, default="409122273624")
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=15)
    parser.add_argument(
        "--spatial-cameras",
        choices=("auto", "front,left", "front", "none"),
        default="auto",
        help=(
            "Depth camera roles. 'auto' follows server policy metadata: front for the new front-only "
            "models and none for the RGB baseline."
        ),
    )
    parser.add_argument("--camera-read-timeout-ms", type=float, default=80.0)
    parser.add_argument("--query-frequency", type=int, default=48, help="Request a new action chunk every N frames.")
    parser.add_argument("--max-action-chunk-size", type=int, default=30, help="Max actions to use from each server chunk.")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--no-home", action="store_true", help="Skip the physical home reset before each run, including dry runs")
    parser.add_argument(
        "--no-tactile-reset",
        action="store_true",
        help="Do not zero XHand tactile sensors before each PI0 run.",
    )
    parser.add_argument(
        "--tactile-reset-verify-threshold",
        type=float,
        default=2.0,
        help="Warn when unloaded tactile force remains above this value immediately after reset.",
    )
    parser.add_argument(
        "--no-index-splay-deadzone",
        action="store_true",
        help="Do not clamp the index-splay action dead zone before sending policy actions.",
    )
    parser.add_argument(
        "--index-splay-deadzone-min",
        type=float,
        default=INDEX_SPLAY_DEADZONE_MIN_RAD,
        help="Lower bound for the index-splay action dead zone.",
    )
    parser.add_argument(
        "--index-splay-deadzone-max",
        type=float,
        default=INDEX_SPLAY_DEADZONE_MAX_RAD,
        help="Upper bound for the index-splay action dead zone.",
    )
    parser.add_argument("--contact-safety-config", type=Path,
                        help="Mandatory reviewed XHand tactile + measured arm TCP wrench limits for any actuating run")
    parser.add_argument("--dry-run", action="store_true", help="Run inference without sending policy actions; home reset still runs unless --no-home is set")
    parser.add_argument("--live-packet-url", type=str,
                        help="SENSOR-ONLY HTTP receiver /frame on inference host. "
                             "No Pi0 websocket, no homing or policy actions.")
    parser.add_argument("--live-packet-frames", type=int, default=48,
                        help="Max raw LeRobot frame packets to send in sensor-only mode")
    parser.add_argument("--capture-observation-audit", type=Path,
                        help="Read-only robot/camera observation probe: no PI0 server, home reset or action sending")
    parser.add_argument("--capture-audit-frames", type=int, default=5,
                        help="How many live observations to audit without any policy control")
    parser.add_argument("--check-config", action="store_true", help="Validate dataset wiring and exit")
    parser.add_argument("--yes", action="store_true", help="Skip interactive start confirmation")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Enable lightweight asynchronous diagnostic recording under this directory.",
    )
    parser.add_argument(
        "--record-queue-size",
        type=int,
        default=16,
        help="Maximum pending diagnostic writes; full queues drop records instead of delaying control.",
    )

    args = parser.parse_args()
    args.spatial_cameras = (
        None
        if args.spatial_cameras == "auto"
        else ()
        if args.spatial_cameras == "none"
        else tuple(args.spatial_cameras.split(","))
    )
    return args


def _normalize_spatial_camera_roles(value) -> tuple[str, ...] | None:
    """Normalize policy metadata roles while accepting old server representations."""
    if value is None:
        return None
    if isinstance(value, str):
        roles = () if value in {"", "none"} else tuple(value.split(","))
    else:
        roles = tuple(value)
    normalized = tuple(str(role).removeprefix("cam_") for role in roles)
    unsupported = set(normalized) - {"front", "left"}
    if unsupported:
        raise ValueError(
            f"Server policy requests unsupported spatial camera roles {sorted(unsupported)}; "
            "this deployment client supports front and left."
        )
    return normalized


def configure_input_mode_from_metadata(args: argparse.Namespace, metadata: dict | None) -> None:
    """Resolve camera/depth/tactile requirements from the policy server metadata.

    The deployment client remains usable with older servers that did not publish
    spatial metadata. In that case it keeps the historical front+left+tactile
    behavior. New models publish explicit ``spatial_use_*`` fields, so the RGB
    baseline does not accidentally receive depth or require tactile recording.
    """
    metadata = metadata or {}
    has_spatial_metadata = any(
        key in metadata
        for key in ("spatial_camera_roles", "spatial_use_visual", "spatial_use_tactile", "rgb_only_baseline")
    )
    metadata_roles = _normalize_spatial_camera_roles(metadata.get("spatial_camera_roles"))
    if metadata_roles is None:
        if "spatial_use_visual" in metadata:
            metadata_roles = ("front", "left") if bool(metadata["spatial_use_visual"]) else ()
        elif args.spatial_cameras is not None:
            # A pre-metadata server can still be driven with the explicit CLI
            # choice that the old deployment script supported.
            metadata_roles = tuple(args.spatial_cameras)
        else:
            # Legacy server: preserve the old client default.
            metadata_roles = ("front", "left")

    if args.spatial_cameras is None:
        args.spatial_cameras = metadata_roles
    elif tuple(args.spatial_cameras) != metadata_roles:
        raise ValueError(
            f"Client --spatial-cameras={args.spatial_cameras} differs from server "
            f"spatial_camera_roles={metadata_roles}. Use --spatial-cameras auto or match the server policy."
        )

    legacy_server = not has_spatial_metadata
    args.spatial_use_visual = bool(
        metadata.get("spatial_use_visual", bool(args.spatial_cameras))
    )
    args.spatial_use_tactile = bool(
        metadata.get("spatial_use_tactile", True if legacy_server else bool(args.spatial_cameras))
    )
    args.rgb_only_baseline = bool(metadata.get("rgb_only_baseline", False))
    args.policy_input_mode = "rgb" if not args.spatial_use_visual and not args.spatial_use_tactile else "spatial"
    # Some older/native servers invoke XHandInputs directly and expect the
    # canonical {"images": ..., "state": ...} schema instead of running the
    # deployment repack transform first. Keep the normal raw schema, but add
    # these aliases when metadata is absent or the selected policy is RGB-only.
    args.send_native_input_aliases = legacy_server or args.policy_input_mode == "rgb"


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir.expanduser().resolve()
    return (args.dataset_root / args.dataset_name).expanduser().resolve()


def load_feature_names(dataset_dir: Path) -> tuple[list[str], list[str], list[str]]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        logging.warning("Dataset info not found at %s; using built-in UR7e + XHand feature order", info_path)
        return FALLBACK_STATE_NAMES, FALLBACK_ACTION_NAMES, ["cam_front", "cam_left", "cam_right"]

    info = read_json(info_path)
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")
    camera_names = sorted(
        key.removeprefix("observation.images.")
        for key, ft in features.items()
        if key.startswith("observation.images.") and ft.get("dtype") in {"image", "video"}
    )

    if not state_names:
        logging.warning("observation.state names not found in %s; using built-in state order", info_path)
        state_names = FALLBACK_STATE_NAMES
    if not action_names:
        logging.warning("action names not found in %s; using built-in action order", info_path)
        action_names = FALLBACK_ACTION_NAMES
    if not camera_names:
        camera_names = ["cam_front", "cam_left", "cam_right"]

    return list(state_names), list(action_names), camera_names


def build_robot(args: argparse.Namespace):
    try:
        camera_configs = importlib.import_module("lerobot.cameras.configs")
        realsense = importlib.import_module("lerobot.cameras.realsense")
        ur7e_config = importlib.import_module("lerobot.robots.ur7e.ur7e_config")
        ur7e_xhand = importlib.import_module("lerobot.robots.ur7e_xhand.ur7e_xhand")
        ur7e_xhand_config = importlib.import_module("lerobot.robots.ur7e_xhand.ur7e_xhand_config")
        xhand_config = importlib.import_module("lerobot.robots.xhand.xhand_config")
    except ImportError as exc:
        raise ImportError(
            "Missing LeRobot UR7e/XHand deployment modules. Run this script in the robot deployment environment."
        ) from exc

    common_camera = dict(
        fps=args.realsense_fps,
        width=args.realsense_width,
        height=args.realsense_height,
        color_mode=camera_configs.ColorMode.RGB,
    )
    cameras = {
        "cam_front": realsense.RealSenseCameraConfig(
            serial_number_or_name=args.realsense_front_serial,
            use_depth="front" in args.spatial_cameras,
            **common_camera,
        ),
        "cam_left": realsense.RealSenseCameraConfig(
            serial_number_or_name=args.realsense_left_serial,
            use_depth="left" in args.spatial_cameras,
            **common_camera,
        ),
        "cam_right": realsense.RealSenseCameraConfig(
            serial_number_or_name=args.realsense_right_serial,
            use_depth=False,
            **common_camera,
        ),
    }

    robot_config = ur7e_xhand_config.UR7eXHandConfig(
        arm_config=ur7e_config.UR7eConfig(
            robot_ip=args.arm_ip,
            control_mode="servoj",
            speed=0.9,
            acceleration=0.5,
            servo_time=1.0 / args.arm_control_freq,
            servo_lookahead_time=1.0 / args.arm_control_freq,
            max_relative_target=args.arm_max_relative_target,
            # Required by the independent contact interlock; UR7eXHand
            # exposes arm_tcp_force (3 N) and arm_tcp_torque (3 Nm).
            # These are EXTRA observation keys; do not append to 1972-state.
            record_tcp_force_torque=True,
            cameras={},
        ),
        hand_config=xhand_config.XHandConfig(
            protocol=args.hand_protocol,
            serial_port=args.hand_serial_port,
            ethercat_interface=args.hand_ethercat_interface,
            control_frequency=args.hand_control_freq or args.fps,
            cameras={},
        ),
        cameras=cameras,
        synchronize_actions=True,
        action_timeout=max(0.2, 2.0 / args.arm_control_freq),
        camera_read_timeout_ms=args.camera_read_timeout_ms,
        check_arm_hand_collision=True,
        emergency_stop_both=True,
    )
    return ur7e_xhand.UR7eXHand(robot_config)


def build_env_state(obs: dict, state_names: list[str]) -> np.ndarray:
    missing = [name for name in state_names if name not in obs]
    if missing:
        raise KeyError(f"Missing observation state keys: {missing}")
    return np.array([obs[name] for name in state_names], dtype=np.float32)


def state_schema_has_raw_xhand_state(state_names: list[str]) -> bool:
    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    return len(state_names) == required_dim


def validate_state_schema(state_names: list[str], args: argparse.Namespace) -> None:
    """Validate the minimum state needed by the selected server-side policy."""
    native_xhand_dim = TACTILE_BLOCK_START
    if len(state_names) < native_xhand_dim:
        raise ValueError(
            "XHand deployment requires at least the 52-value arm/hand state prefix, "
            f"got {len(state_names)}. Check --dataset-dir/--dataset-root/--dataset-name."
        )
    if args.spatial_use_tactile and not state_schema_has_raw_xhand_state(state_names):
        required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
        raise ValueError(
            "The selected spatial policy uses tactile input and requires the full raw UR7e + XHand "
            f"observation.state with exactly {required_dim} values, got {len(state_names)}. "
            "Check --dataset-dir/--dataset-root/--dataset-name."
        )


def get_current_action(obs: dict, action_names: list[str]) -> np.ndarray:
    missing = [name for name in action_names if name not in obs]
    if missing:
        raise KeyError(f"Missing current action keys in observation: {missing}")
    return np.array([obs[name] for name in action_names], dtype=np.float32)


def get_image(obs: dict, camera_name: str, height: int, width: int) -> np.ndarray:
    image = obs.get(camera_name)
    if isinstance(image, dict):
        for key in ("image", "color", "rgb", "color_image"):
            if key in image:
                return np.asarray(image[key], dtype=np.uint8)
    if image is None:
        raise KeyError(f"Missing RGB image for {camera_name}; expected {width}x{height} camera observation")
    return np.asarray(image, dtype=np.uint8)


def get_depth(obs: dict, camera_name: str) -> np.ndarray:
    image_like = obs.get(camera_name)
    if isinstance(image_like, dict):
        for key in ("depth", "depth_image", "z16"):
            if key in image_like:
                return np.asarray(image_like[key])

    candidates = (
        f"depths.{camera_name}",
        f"observation.depths.{camera_name}",
        f"observation/{camera_name}_depth",
        f"{camera_name}_depth",
        f"depth_{camera_name}",
        f"{camera_name}.depth",
    )
    for key in candidates:
        value = obs.get(key)
        if value is not None:
            return np.asarray(value)
    raise KeyError(
        f"Missing depth image for {camera_name}; tried keys: {', '.join(candidates)}. "
        "Spatial deployment requires raw RGB-D obs so the server can build point clouds."
    )


def build_pi0_observation(
    obs: dict,
    env_state: np.ndarray,
    camera_names: list[str],
    args: argparse.Namespace,
    current_action_step: int,
    frame_idx: int,
) -> dict:
    observation = {
        "observation.state": env_state.astype(np.float32),
        "prompt": args.prompt or args.task,
        "current_action_step": current_action_step,
        "frame_index": np.asarray(frame_idx, dtype=np.int64),
        "timestamp_s": np.asarray(frame_idx / max(args.fps, 1), dtype=np.float64),
    }
    native_images = {}
    for camera_name in camera_names:
        image = get_image(obs, camera_name, args.realsense_height, args.realsense_width)
        observation[f"observation.images.{camera_name}"] = image
        native_images[camera_name] = image
    if getattr(args, "send_native_input_aliases", False):
        observation["images"] = native_images
        observation["state"] = observation["observation.state"]
    for role in args.spatial_cameras:
        camera_name = f"cam_{role}"
        observation[f"observation.depths.{camera_name}"] = get_depth(obs, camera_name)
    return observation


def normalize_action_chunk(actions, expected_dim: int, max_action_chunk_size: int) -> np.ndarray:
    action_chunk = np.asarray(actions, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]
    if action_chunk.ndim != 2:
        raise ValueError(f"Server returned actions with invalid shape {action_chunk.shape}; expected [T, {expected_dim}]")
    if action_chunk.shape[1] != expected_dim:
        raise ValueError(f"Server returned action dim {action_chunk.shape[1]}, expected {expected_dim}")
    if action_chunk.shape[0] == 0 or not np.all(np.isfinite(action_chunk)):
        raise ValueError("Server returned an empty or non-finite action chunk")
    return action_chunk[:max_action_chunk_size]


def action_array_to_dict(action: np.ndarray, action_names: list[str]) -> dict[str, float]:
    if len(action) != len(action_names):
        raise ValueError(f"Action length {len(action)} does not match action names {len(action_names)}")
    return {name: float(value) for name, value in zip(action_names, action, strict=True)}


def format_dry_run_action(action_dict: dict[str, float], obs: dict) -> str:
    groups = []
    for prefix in ("arm_", "hand_"):
        entries = [(name, value) for name, value in action_dict.items() if name.startswith(prefix)]
        values = ", ".join(f"{name.removeprefix(prefix)}={value:.4f}" for name, value in entries)
        deltas = [abs(value - float(obs[name])) for name, value in entries if name in obs]
        delta_text = f" max_abs_delta={max(deltas):.4f} rad" if deltas else ""
        groups.append(f"{prefix.removesuffix('_')}=[{values}]{delta_text}")
    return " | ".join(groups)


def read_tactile_calc_force_from_hand(hand_observation: dict[str, float]) -> np.ndarray:
    force = np.zeros((len(TACTILE_FINGER_NAMES), len(TACTILE_FORCE_AXES)), dtype=np.float32)
    for sensor_idx in range(len(TACTILE_FINGER_NAMES)):
        for axis_idx, axis in enumerate(TACTILE_FORCE_AXES):
            force[sensor_idx, axis_idx] = hand_observation[
                f"tactile_sensor_{sensor_idx}.calc_force.{axis}"
            ]
    return force


def reset_tactile_sensors_before_run(robot, run_index: int, verify_threshold: float) -> None:
    if getattr(robot.hand, "_device", None) is None:
        logging.warning("Cannot reset XHand tactile sensors because the SDK device is unavailable.")
        return

    print("\nZeroing XHand tactile sensors while unloaded...")
    print("Keep all five fingertips free of contact until the PI0 run starts.")
    reset_errors = []
    for sensor_id, finger_name in zip(TACTILE_SENSOR_IDS, TACTILE_FINGER_NAMES, strict=True):
        result = robot.hand._device.reset_sensor(robot.hand._hand_id, sensor_id)
        error_code = int(result.error_code)
        error_message = str(result.error_message)
        if error_code != 0:
            reset_errors.append((finger_name, error_code, error_message))

    time.sleep(0.5)
    verification_samples = []
    for _ in range(5):
        verification_samples.append(read_tactile_calc_force_from_hand(robot.hand.get_observation()))
        time.sleep(0.05)

    verification_force = np.stack(verification_samples)
    verification_norm = np.linalg.norm(verification_force, axis=2)
    max_norms = verification_norm.max(axis=0)
    peak_text = ", ".join(
        f"{finger}={max_norm:.2f}"
        for finger, max_norm in zip(TACTILE_FINGER_NAMES, max_norms, strict=True)
    )
    print(f"Run {run_index} post-reset unloaded max norms: {peak_text}")

    if reset_errors:
        logging.warning("Tactile reset SDK errors for run %s: %s", run_index, reset_errors)

    failed_sensors = [
        (finger, float(max_norm))
        for finger, max_norm in zip(TACTILE_FINGER_NAMES, max_norms, strict=True)
        if max_norm > verify_threshold
    ]
    if failed_sensors:
        logging.warning(
            "TACTILE RESET CHECK WARNING for run %s: unloaded force remains above %.2f: %s. "
            "PI0 will continue, but the tactile input may be out of the training distribution.",
            run_index,
            verify_threshold,
            ", ".join(f"{finger}={value:.2f}" for finger, value in failed_sensors),
        )


def apply_index_splay_deadzone(
    action: np.ndarray,
    action_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    if args.no_index_splay_deadzone:
        return action

    try:
        idx = action_names.index(INDEX_SPLAY_ACTION_KEY)
    except ValueError:
        return action

    value = float(action[idx])
    if args.index_splay_deadzone_min <= value <= args.index_splay_deadzone_max:
        action = action.copy()
        action[idx] = 0.0
    return action


def make_hold_action_chunk(obs: dict, action_names: list[str], n_action_steps: int) -> np.ndarray:
    current_action = get_current_action(obs, action_names)
    return np.repeat(current_action[None, :], repeats=n_action_steps, axis=0)


def request_action_chunk(
    *,
    client: WebsocketClientPolicy,
    observation: dict,
    action_names: list[str],
    max_action_chunk_size: int,
    label: str,
) -> tuple[np.ndarray, dict, float]:
    infer_start = time.perf_counter()
    inference_result = client.infer(observation)
    action_chunk = normalize_action_chunk(
        inference_result["actions"],
        expected_dim=len(action_names),
        max_action_chunk_size=max_action_chunk_size,
    )
    infer_ms = (time.perf_counter() - infer_start) * 1000
    policy_ms = inference_result.get("policy_timing", {}).get("infer_ms")
    server_ms = inference_result.get("server_timing", {}).get("infer_ms")
    print(
        f"Got {label} action chunk {action_chunk.shape}: "
        f"client_roundtrip={infer_ms:.1f}ms, policy={policy_ms}, server={server_ms}",
        flush=True,
    )
    return action_chunk, inference_result, infer_ms


def seed_action_target_fallback(robot) -> None:
    fallback = {key: 0.0 for key in robot.action_target_position_features}

    try:
        arm_obs = robot.arm.get_observation()
        arm_target_position = [arm_obs[f"ee_pose.{i:02d}"] for i in (3, 7, 11)]
        arm_target_orientation = [arm_obs[f"ee_pose.{i:02d}"] for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)]
        fallback.update(
            {f"arm_target_position_{i}": float(value) for i, value in enumerate(arm_target_position)}
        )
        fallback.update(
            {f"arm_target_orientation_{i}": float(value) for i, value in enumerate(arm_target_orientation)}
        )
    except Exception as exc:
        logging.warning("Could not seed arm target fallback from current pose: %s", exc)

    robot._last_action_target_values = fallback


def require_ethercat_permissions(args: argparse.Namespace) -> None:
    if args.hand_protocol != "EtherCAT" or args.check_config:
        return
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        python_bin = Path(sys.executable).resolve()
        raise SystemExit(
            "XHand EtherCAT requires root privileges for raw socket access.\n"
            "Run the script with the lefranx interpreter directly, for example:\n"
            f"  sudo -E {python_bin} {Path(__file__).resolve()} --hand-protocol EtherCAT ...\n"
            "Tip: keep the same deployment arguments you just used."
        )


def read_stdin_byte(fd: int, timeout: float, stop_event: threading.Event) -> bytes | None:
    while not stop_event.is_set():
        ready, _, _ = select.select([fd], [], [], timeout)
        if ready:
            return os.read(fd, 1)
        return None
    return None


def read_terminal_key(fd: int, stop_event: threading.Event) -> str | None:
    ch = read_stdin_byte(fd, 0.05, stop_event)
    if not ch:
        return None

    if ch == b"\x1b":
        second = read_stdin_byte(fd, 0.08, stop_event)
        if second in (b"[", b"O"):
            code = read_stdin_byte(fd, 0.08, stop_event)
            return {
                b"A": "up",
                b"B": "down",
                b"C": "right",
                b"D": "left",
            }.get(code, "escape")
        return "escape"

    if ch in (b"\r", b"\n"):
        return "enter"

    try:
        return ch.decode("utf-8")
    except UnicodeDecodeError:
        return None


@contextmanager
def terminal_stop_listener():
    """Remember a RIGHT-arrow stop request even while inference or robot IO is busy."""
    stop_requested = threading.Event()
    listener_stop = threading.Event()

    if not sys.stdin.isatty():
        yield stop_requested
        return

    fd = sys.stdin.fileno()
    try:
        old_attrs = termios.tcgetattr(fd)
    except termios.error:
        yield stop_requested
        return

    def listen() -> None:
        while not listener_stop.is_set():
            if read_terminal_key(fd, listener_stop) == "right":
                stop_requested.set()

    thread = threading.Thread(target=listen, name="pi0-right-arrow-listener", daemon=True)
    try:
        tty.setcbreak(fd)
        attrs = termios.tcgetattr(fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        thread.start()
        yield stop_requested
    finally:
        listener_stop.set()
        thread.join(timeout=0.2)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def reset_robot_to_home_for_run(robot, args: argparse.Namespace, run_index: int) -> None:
    if not args.no_home:
        print("Resetting robot to home...")
        if not robot.reset_to_home():
            raise RuntimeError(f"Home reset failed before run {run_index}; refusing to start with an unknown initial pose")
        time.sleep(2.0)
    else:
        print("Skipping home reset (--no-home); initial observation uses the robot's current pose.")

    seed_action_target_fallback(robot)
    if not args.spatial_use_tactile:
        print("Skipping tactile reset: the selected policy does not use tactile input.")
    elif not args.no_tactile_reset:
        reset_tactile_sensors_before_run(
            robot,
            run_index=run_index,
            verify_threshold=args.tactile_reset_verify_threshold,
        )


def prepare_robot_for_policy_run(robot, args: argparse.Namespace, run_index: int, total_runs: int | None) -> None:
    run_label = f"{run_index}" if total_runs is None else f"{run_index}/{total_runs}"
    print(f"\n=== Preparing PI0 run {run_label} ===")

    if args.run_mode == "multi":
        preparation = "start" if args.no_home else "reset home and start"
        input(
            f"Press ENTER to {preparation} PI0 run {run_label}. "
            "Press RIGHT arrow during the run to finish this run: "
        )
        reset_robot_to_home_for_run(robot, args, run_index)
        return

    reset_robot_to_home_for_run(robot, args, run_index)
    if not args.yes:
        input(f"Press ENTER to start PI0 run {run_label}. Use Ctrl+C to stop: ")


def run_policy_once(
    *,
    robot,
    client: WebsocketClientPolicy,
    state_names: list[str],
    action_names: list[str],
    camera_names: list[str],
    args: argparse.Namespace,
    run_index: int,
) -> int:
    dt = 1.0 / args.fps
    frame_idx = 0
    previous_action = None
    action_chunk = None
    chunk_idx = 0
    n_action_steps = 1
    query_count = 0
    current_action_step = 0
    next_query_id = 0
    active_query_id = -1
    active_fallback = False
    end_reason = "duration"
    safety = ContactSafetyGuard(args.contact_safety_limits) if args.contact_safety_limits is not None else None
    client.reset()
    recorder = (
        DiagnosticRecorder(
            root_dir=args.record_dir,
            run_index=run_index,
            args=args,
            state_names=state_names,
            action_names=action_names,
            camera_names=camera_names,
        )
        if args.record_dir is not None
        else None
    )

    if args.run_mode == "multi":
        print(f"Starting spatial PI0 control loop for run {run_index}. Press RIGHT arrow to finish this run.")
    else:
        print(f"Starting spatial PI0 control loop for run {run_index}...")
    start_run = time.perf_counter()
    finished_by_key = False
    try:
        with terminal_stop_listener() as stop_requested:
            while time.perf_counter() - start_run < args.duration:
                if args.run_mode == "multi" and stop_requested.is_set():
                    print(f"\nRight arrow pressed. Finishing PI0 run {run_index}.")
                    finished_by_key = True
                    end_reason = "right_arrow"
                    break

                loop_start = time.perf_counter()
                obs = robot.get_observation()
                env_state = build_env_state(obs, state_names)
                contact_event = None
                if safety is not None:
                    contact_event = safety.check(observation=obs, state=env_state)
                if contact_event is not None and contact_event["replan_recommended"]:
                    # Moderate tactile event: invalidate unexecuted actions.
                    # This is an explicitly enabled POLICY SCHEDULER, not a safety stop.
                    action_chunk = None
                    print(f"[CONTACT REPLAN] frame={frame_idx} finger(s)="
                          f"{contact_event['replan_triggered_fingers']}; "
                          "discarded remaining chunk, querying current observation",flush=True)

                should_query = (
                    action_chunk is None
                    or chunk_idx >= n_action_steps
                    or query_count % args.query_frequency == 0
                )
                if should_query:
                    query_id = next_query_id
                    next_query_id += 1
                    active_query_id = query_id
                    observation = build_pi0_observation(
                        obs=obs,
                        env_state=env_state,
                        camera_names=camera_names,
                        args=args,
                        current_action_step=current_action_step,
                        frame_idx=frame_idx,
                    )
                    if recorder is not None:
                        recorder.record_query_input(
                            query_id=query_id,
                            observation=observation,
                            elapsed_s=time.perf_counter() - start_run,
                        )
                    if current_action_step < DEBUG_INFER_PRINT_LIMIT:
                        debug_print_client_observation(
                            observation,
                            query_index=current_action_step,
                            mode=args.policy_input_mode,
                        )
                    infer_attempt_start = time.perf_counter()
                    try:
                        action_chunk, inference_result, infer_ms = request_action_chunk(
                            client=client,
                            observation=observation,
                            action_names=action_names,
                            max_action_chunk_size=args.max_action_chunk_size,
                            label=args.policy_input_mode,
                        )
                        if recorder is not None:
                            recorder.record_query_output(
                                query_id=query_id,
                                frame_idx=frame_idx,
                                action_chunk=action_chunk,
                                inference_result=inference_result,
                                client_roundtrip_ms=infer_ms,
                            )
                        n_action_steps = action_chunk.shape[0]
                        current_action_step += 1
                        active_fallback = False
                    except Exception as exc:
                        infer_ms = (time.perf_counter() - infer_attempt_start) * 1000.0
                        print(f"Inference failed; holding current pose: {exc}", flush=True)
                        if recorder is not None:
                            recorder.record_query_output(
                                query_id=query_id,
                                frame_idx=frame_idx,
                                action_chunk=None,
                                inference_result=None,
                                client_roundtrip_ms=infer_ms,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        action_chunk = make_hold_action_chunk(obs, action_names, n_action_steps)
                        active_fallback = True
                    chunk_idx = 0

                # Inference may block for hundreds of milliseconds. Refresh sensors
                # BEFORE the next command; never execute a stale chunk blindly.
                if should_query and safety is not None:
                    obs = robot.get_observation()
                    env_state = build_env_state(obs, state_names)
                    after_infer_contact = safety.check(observation=obs, state=env_state)
                    if after_infer_contact["replan_recommended"]:
                        # Never execute a stale action after contact changed
                        # during blocking inference. A safe new query happens
                        # only on a later loop iteration, with fresh sensors.
                        print(f"[CONTACT REPLAN] frame={frame_idx} changed during inference; "
                              "discarding action chunk; no command sent",flush=True)
                        action_chunk = None
                        frame_idx += 1
                        continue
                query_count += 1
                executed_chunk_step = chunk_idx
                raw_action = action_chunk[chunk_idx].copy()
                chunk_idx += 1

                if active_fallback:
                    action = raw_action
                elif previous_action is not None:
                    action = args.smoothing_alpha * raw_action + (1.0 - args.smoothing_alpha) * previous_action
                else:
                    action = raw_action
                if not active_fallback:
                    action = action * args.action_scale
                    action = apply_index_splay_deadzone(action, action_names, args)
                previous_action = action.copy()

                action_dict = action_array_to_dict(action, action_names)
                if args.dry_run:
                    print(
                        f"[dry-run would_send] run={run_index} frame={frame_idx} "
                        f"chunk={chunk_idx}/{n_action_steps} "
                        f"{'fallback_hold ' if active_fallback else ''}"
                        f"{format_dry_run_action(action_dict, obs)}",
                        flush=True,
                    )
                else:
                    robot.send_action(action_dict)

                elapsed = time.perf_counter() - loop_start
                if recorder is not None:
                    recorder.record_control_frame(
                        frame_idx=frame_idx,
                        elapsed_s=time.perf_counter() - start_run,
                        state=env_state,
                        raw_action=raw_action,
                        command_action=action,
                        query_id=active_query_id,
                        chunk_step=executed_chunk_step,
                        loop_ms=elapsed * 1000.0,
                        action_sent=not args.dry_run,
                        fallback=active_fallback,
                    )
                if frame_idx % max(args.fps, 1) == 0:
                    print(
                        f"run={run_index} frame={frame_idx} chunk={chunk_idx}/{n_action_steps} "
                        f"loop={elapsed * 1000:.1f}ms"
                    )
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                frame_idx += 1
    except SafetyTrip as exc:
        end_reason = "contact_safety_trip"
        print(f"[CONTACT SAFETY] Latched failure: {exc}. No further commands in this run.",flush=True)
        # Application-level stop: must be validated against physical UR/XHand
        # behavior, independent hardware E-stop / protective stop still needed.
        if not args.dry_run:
            robot.stop()
        raise
    except KeyboardInterrupt:
        end_reason = "keyboard_interrupt"
        raise
    except Exception:
        end_reason = "error"
        raise
    finally:
        if recorder is not None:
            recorder.close(frame_count=frame_idx, end_reason=end_reason)

    suffix = " by RIGHT arrow" if finished_by_key else ""
    print(f"Spatial PI0 run {run_index} finished{suffix} after {frame_idx} frames.")
    return frame_idx



def _audit_shape(value):
    """Minimal, non-sensitive schema report; do not reinterpret raw camera depth."""
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(map(str,value))}
    try:
        arr=np.asarray(value)
    except (TypeError,ValueError):
        return {"kind": type(value).__name__}
    if arr.dtype.kind in "iufb":
        result={"kind":"array","shape":list(arr.shape),"dtype":str(arr.dtype)}
        if arr.size and arr.dtype.kind in "iuf":
            finite=arr[np.isfinite(arr)]
            result["nonfinite_count"]=int(arr.size-finite.size)
            result["nonzero_count"]=int(np.count_nonzero(finite))
        return result
    return {"kind":type(value).__name__}


def capture_observation_audit(robot, args: argparse.Namespace, state_names: list[str]) -> None:
    """Read actual live fields only. Output is NOT a validated perception packet.

    Depth comes from the existing LeRobot accessor but may be RAW, aligned,
    rescaled, or filtered. Never assign front_depth_native or aligned keys.
    """
    out=args.capture_observation_audit.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite live observation audit: {out}")
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    n=int(args.capture_audit_frames)
    if n<1 or n>100:
        raise ValueError("--capture-audit-frames must be 1..100")
    print("[AUDIT-ONLY] Connecting to read sensors. No policy server, homing, tactile reset or actions.",flush=True)
    for i in range(n):
        start_ns=time.monotonic_ns()
        obs=robot.get_observation()
        end_ns=time.monotonic_ns()
        state=build_env_state(obs,state_names)
        raw_keys=sorted(map(str,obs))
        arm_obs=robot.arm.get_observation()
        arm_keys=sorted(map(str,arm_obs))
        if state.shape!=(1972,) or not np.isfinite(state).all():
            raise ValueError(f"Live state does not meet training contract [1972]: {state.shape}")
        arrays={"robot_state":state,"capture_start_monotonic_ns":np.asarray(start_ns,dtype=np.int64),
                "capture_end_monotonic_ns":np.asarray(end_ns,dtype=np.int64)}
        diagnostics={}
        # Only store the SDK's actual values, including separate UR TCP force
        # and torque vectors; NEVER invent zeros when missing.
        for key,unit in (("arm_tcp_force","N"),("arm_tcp_torque","Nm")):
            raw=obs.get(key)
            if raw is None:
                diagnostics[key]={"status":"missing","expected_shape":[3],"unit":unit}
                continue
            sample=np.asarray(raw,dtype=np.float64)
            if sample.shape!=(3,) or not np.isfinite(sample).all():
                raise ValueError(f"Invalid live {key} shape or values: {sample.shape}")
            arrays[key]=sample
            diagnostics[key]={"status":"captured_unverified", "shape":[3],"unit":unit,
                              "norm":float(np.linalg.norm(sample)),
                              "sdk_provenance":"record_tcp_force_torque=True, not calibrated or freshness-verified"}

        for role in ("front","left","right"):
            camera_name="cam_"+role
            try:
                arrays[role+"_rgb"]=get_image(obs,camera_name,args.realsense_height,args.realsense_width)
                diagnostics[role+"_rgb"]={"status":"captured","shape":list(arrays[role+"_rgb"].shape)}
            except (KeyError,TypeError,ValueError) as exc:
                diagnostics[role+"_rgb"]={"status":"missing","error":str(exc)}
        try:
            # The existing LeRobot getter does NOT prove camera depth is native.
            candidate=np.asarray(get_depth(obs,"cam_front"))
            if candidate.ndim!=2 or candidate.dtype.kind not in "iuf" or not np.isfinite(candidate).all():
                raise ValueError(f"Unexpected front depth candidate shape/dtype {candidate.shape}/{candidate.dtype}")
            arrays["front_depth_coordinate_system_UNVERIFIED"]=candidate
            diagnostics["front_depth"]={"status":"captured_unverified",
                                        "shape":list(candidate.shape),"dtype":str(candidate.dtype),
                                        "depth_coordinate_system":"UNKNOWN",
                                        "depth_scale_m_per_unit":"UNKNOWN"}
        except (KeyError,TypeError,ValueError) as exc:
            diagnostics["front_depth"]={"status":"missing_or_invalid","error":str(exc)}
        try:
            # Raw 120-taxel/5-finger force vectors, using the TRAINING 1972 layout.
            peaks=raw_tactile_finger_peaks(state)
            diagnostics["tactile"]={
                "per_finger_max_raw":peaks.astype(float).tolist(),
                "all_raw_taxels_zero":bool(np.all(peaks==0.)),
                "raw_tactile_freshness_verified":False,
                "warning":"All-zero taxels do not prove working XHand sensors; SDK may hold prior/placeholder data",
            }
        except (ValueError,TypeError) as exc:
            diagnostics["tactile"]={"error":str(exc)}
        torque_key_hints=tuple(k for k in raw_keys+arm_keys if any(
            word in k.lower() for word in ("wrench","actual_tcp_force","tcp_force","joint_torque","joint_effort","ft_sensor")))
        record={"audit_index":i,"start_monotonic_ns":start_ns,"end_monotonic_ns":end_ns,
                "observation_keys":raw_keys,"arm_observation_keys":arm_keys,
                "observation_schema":{key:_audit_shape(value) for key,value in obs.items()},
                "arm_observation_schema":{key:_audit_shape(value) for key,value in arm_obs.items()},
                "torque_or_wrench_key_candidates_NOT_VERIFIED":list(torque_key_hints),
                "diagnostics":diagnostics,
                "never_sent_robot_action":True}
        records.append(record)
        file=out/f"audit_{i:04d}.npz"
        with file.open("xb") as stream:
            np.savez_compressed(stream,**arrays)
        print(f"[AUDIT-ONLY] {i+1}/{n}: rgb={[k for k in arrays if k.endswith('_rgb')]} "
              f"depth={diagnostics['front_depth']['status']} "
              f"UR_force={diagnostics['arm_tcp_force']['status']} "
              f"UR_torque={diagnostics['arm_tcp_torque']['status']}",flush=True)
        if i+1<n:
            time.sleep(max(0.,1.0/max(1,args.fps)-(time.monotonic_ns()-start_ns)*1e-9))
    report={"schema_version":1,"mode":"READ_ONLY_UNVERIFIED_SENSOR_DISCOVERY",
            "frames":n,"no_home_reset":True,"no_actions_sent":True,"pi0_server_used":False,
            "depth_provenance":"UNVERIFIED; do not use audit NPZ as front_depth_native in policy",
            "tcp_wrench_provenance":"UR7eConfig.record_tcp_force_torque=True; split arm_tcp_force/arm_tcp_torque; "
                                     "units/reference frame/timestamp/force bias require physical verification",
            "records":records}
    (out/"observation_schema.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("[AUDIT-ONLY] schema:",out/"observation_schema.json",flush=True)




def capture_live_packet_stream(robot,args:argparse.Namespace,state_names:list[str])->None:
    """Send EXACT LeRobot RGB/raw depth/state fields to inference computer.

    This branch never touches PI0 websocket, sends an arm action, homes or
    resets tactile. Receiver registers front depth with shared training
    rgbtrack_adapter rather than reimplementing calibration on robot host.
    """
    target=args.live_packet_url
    if not isinstance(target,str) or not target.startswith("http://") or not target.endswith("/frame"):
        raise ValueError("--live-packet-url must be http://<trusted-inference-host>:<port>/frame")
    token=os.environ.get("OPENPI_LIVE_PACKET_TOKEN","")
    if len(token)<16:
        raise ValueError("Set the same OPENPI_LIVE_PACKET_TOKEN on robot and inference host")
    n=int(args.live_packet_frames)
    if not 1<=n<=1000:
        raise ValueError("--live-packet-frames must be in 1..1000")
    dt=1.0/max(1,args.fps)
    print(f"[LIVE SENSOR-ONLY] frames={n}, destination={target}; NO actuation",flush=True)
    for frame in range(n):
        started=time.monotonic()
        observation=robot.get_observation()
        state=build_env_state(observation,state_names)
        if state.shape!=(1972,) or not np.isfinite(state).all():
            raise ValueError(f"Live robot state invalid: {state.shape}")
        rgb={}
        for role in ("front","left","right"):
            image=get_image(observation,f"cam_{role}",args.realsense_height,args.realsense_width)
            if image.dtype!=np.uint8 or image.shape!=(args.realsense_height,args.realsense_width,3):
                raise ValueError(f"Invalid {role} RGB {image.shape} {image.dtype}")
            rgb[role+"_rgb"]=image
        # In LeRobot training this SAME observation.depths.cam_front was
        # persisted to parquet; do not align, normalize or resample here.
        raw_depth=np.asarray(get_depth(observation,"cam_front"))
        if raw_depth.dtype!=np.uint16 or raw_depth.shape!=(args.realsense_height,args.realsense_width):
            raise ValueError(f"Live LeRobot front depth unexpected: {raw_depth.dtype} {raw_depth.shape}")
        packet=dict(rgb,front_depth_native=raw_depth,robot_state=state,
                    timestamp_s=np.asarray(started,dtype=np.float64),
                    frame_index=np.asarray(frame,dtype=np.int64))
        buffer=io.BytesIO()
        np.savez_compressed(buffer,**packet)
        request=urllib.request.Request(
            target,data=buffer.getvalue(),method="POST",
            headers={"Content-Type":"application/octet-stream",
                     "X-Live-Packet-Token":token})
        try:
            with urllib.request.urlopen(request,timeout=30.) as response:
                ack=json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message=exc.read().decode("utf-8",errors="replace")
            raise RuntimeError(f"Live receiver rejected frame {frame}: HTTP {exc.code} {message}") from exc
        if ack.get("accepted_frame")!=frame or ack.get("no_robot_action") is not True:
            raise RuntimeError(f"Invalid live receiver acknowledgement at frame={frame}: {ack}")
        print(f"[LIVE SENSOR-ONLY] forwarded frame={frame}/{n-1}, "
              f"elapsed_ms={(time.monotonic()-started)*1000:.1f}",flush=True)
        remaining=dt-(time.monotonic()-started)
        if remaining>0:
            time.sleep(remaining)



def iter_run_indices(args: argparse.Namespace):
    if args.run_mode == "single":
        yield 1, 1
        return
    if args.num_runs == 0:
        run_index = 1
        while True:
            yield run_index, None
            run_index += 1
    else:
        for run_index in range(1, args.num_runs + 1):
            yield run_index, args.num_runs


def main() -> int:
    args = parse_args()
    init_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1]")
    if args.query_frequency <= 0:
        raise ValueError("--query-frequency must be positive")
    if args.max_action_chunk_size <= 0:
        raise ValueError("--max-action-chunk-size must be positive")
    if args.record_queue_size <= 0:
        raise ValueError("--record-queue-size must be positive")
    if not 1 <= args.capture_audit_frames <= 100:
        raise ValueError("--capture-audit-frames must be 1..100")
    if args.capture_observation_audit is not None and args.check_config:
        raise ValueError("--capture-observation-audit and --check-config cannot be combined")
    if args.capture_observation_audit is not None and args.live_packet_url is not None:
        raise ValueError("Choose either offline observation audit or live packet forwarding")
    if args.live_packet_url is not None and args.check_config:
        raise ValueError("--live-packet-url and --check-config cannot be combined")
    if args.num_runs < 0:
        raise ValueError("--num-runs must be non-negative")
    if args.tactile_reset_verify_threshold < 0:
        raise ValueError("--tactile-reset-verify-threshold must be non-negative")
    if args.index_splay_deadzone_min > args.index_splay_deadzone_max:
        raise ValueError("--index-splay-deadzone-min must be <= --index-splay-deadzone-max")
    if (not args.dry_run and args.contact_safety_config is None and not args.check_config
            and args.capture_observation_audit is None and args.live_packet_url is None):
        raise ValueError("Actuation forbidden without --contact-safety-config; this is not a hardware E-stop")
    args.contact_safety_limits = (
        load_safety_limits(args.contact_safety_config)
        if args.contact_safety_config is not None else None
    )
    require_ethercat_permissions(args)

    dataset_dir = resolve_dataset_dir(args)
    state_names, action_names, camera_names = load_feature_names(dataset_dir)
    supported_cameras = {"cam_front", "cam_left", "cam_right"}
    unsupported_cameras = set(camera_names) - supported_cameras
    if unsupported_cameras:
        raise ValueError(
            f"Dataset expects unsupported cameras {sorted(unsupported_cameras)}; "
            f"available cameras are {sorted(supported_cameras)}"
        )
    missing_cameras = supported_cameras - set(camera_names)
    if missing_cameras:
        raise ValueError(
            f"This XHand policy requires all three RGB cameras; missing {sorted(missing_cameras)} "
            f"from {dataset_dir / 'meta' / 'info.json'}"
        )
    if len(action_names) != len(FALLBACK_ACTION_NAMES):
        raise ValueError(f"This XHand deployment requires 18 action names, got {len(action_names)}")

    print("=== UR7e + XHand PI0 Deployment Client ===")
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Dataset: {dataset_dir}")
    print(f"State dim: {len(state_names)}")
    print(f"Action dim: {len(action_names)}")
    print(f"Cameras: {', '.join(camera_names)}")
    requested_depth = (
        "auto (server metadata)"
        if args.spatial_cameras is None
        else ", ".join(args.spatial_cameras) if args.spatial_cameras else "none"
    )
    print(f"Requested spatial depth cameras: {requested_depth}")
    print(f"Prompt: {args.prompt or args.task}")
    print(f"FPS: {args.fps}, duration: {args.duration}s")
    print(f"Run mode: {args.run_mode}, num_runs: {'until Ctrl+C' if args.num_runs == 0 else args.num_runs}")
    print(f"Dry run: {args.dry_run}")
    print(f"Diagnostic recording: {args.record_dir if args.record_dir is not None else 'disabled'}")

    if args.capture_observation_audit is not None:
        # Standalone sensor-only mode: no policy server, no reset-to-home,
        # no contact controller, and never execute a model action.
        args.spatial_cameras=("front",)
        args.spatial_use_visual=True
        args.spatial_use_tactile=True
        validate_state_schema(state_names,args)
        robot=build_robot(args)
        try:
            robot.connect(calibrate=False)
            if not robot.is_connected:
                raise RuntimeError("Sensor audit robot connection failed")
            capture_observation_audit(robot,args,state_names)
        finally:
            if robot.is_connected:
                robot.disconnect()
        return 0

    if args.live_packet_url is not None:
        # Exactly like audit: standalone robot-side sensor-only streaming.
        args.spatial_cameras=("front",)
        args.spatial_use_visual=True
        args.spatial_use_tactile=True
        validate_state_schema(state_names,args)
        if args.realsense_width!=640 or args.realsense_height!=480:
            raise ValueError("Live packet receiver expects training RGB-D 640x480")
        if not os.environ.get("OPENPI_LIVE_PACKET_TOKEN"):
            raise ValueError("OPENPI_LIVE_PACKET_TOKEN is required for live packet forwarding")
        robot=build_robot(args)
        try:
            robot.connect(calibrate=False)
            if not robot.is_connected:
                raise RuntimeError("Sensor-only robot connection failed")
            capture_live_packet_stream(robot,args,state_names)
        finally:
            if robot.is_connected:
                robot.disconnect()
        return 0

    if args.check_config:
        configure_input_mode_from_metadata(args, metadata=None)
        validate_state_schema(state_names, args)
        print(f"Resolved policy input mode: {args.policy_input_mode}")
        print(f"Resolved spatial depth cameras: {', '.join(args.spatial_cameras) if args.spatial_cameras else 'none'}")
        print("Config check passed. Exiting before server/robot connection.")
        return 0

    client = WebsocketClientPolicy(
        args.server_ip,
        args.server_port,
        api_key=args.api_key,
        proxy=True if args.use_env_proxy else None,
        open_timeout=args.server_open_timeout,
    )
    metadata = client.get_server_metadata()
    args.server_metadata = metadata
    if metadata:
        print(f"Server metadata: {metadata}")
    else:
        print(
            "WARNING: server returned no policy metadata; using legacy spatial defaults and "
            "including native XHand input aliases for compatibility. Update/restart the policy "
            "server if this is not the intended model.",
            flush=True,
        )
    configure_input_mode_from_metadata(args, metadata)
    validate_state_schema(state_names, args)
    print(f"Policy input mode: {args.policy_input_mode}")
    print(f"Spatial depth cameras: {', '.join(args.spatial_cameras) if args.spatial_cameras else 'none'}")
    print(f"Spatial visual input: {args.spatial_use_visual}")
    print(f"Spatial tactile input: {args.spatial_use_tactile}")
    print(f"Native XHand input aliases: {args.send_native_input_aliases}")

    robot = build_robot(args)

    try:
        print("Connecting robot and cameras...")
        robot.connect(calibrate=False)
        if not robot.is_connected:
            raise RuntimeError("Robot failed to connect")

        for run_index, total_runs in iter_run_indices(args):
            prepare_robot_for_policy_run(robot, args, run_index, total_runs)
            run_policy_once(
                robot=robot,
                client=client,
                state_names=state_names,
                action_names=action_names,
                camera_names=camera_names,
                args=args,
                run_index=run_index,
            )

    except KeyboardInterrupt:
        print("\nStopping PI0 deployment...")
    except Exception as exc:
        print(f"Error in PI0 control loop: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if robot.is_connected:
            print("Disconnecting robot...")
            try:
                robot.stop()
            finally:
                robot.disconnect()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
