#!/usr/bin/env python3
"""Causal per-frame SAM2 -> RGBTrack -> Hand-Scene packet producer.

Replay input is a directory of one-frame NPZ files. Each file is consumed in
frame order and contains only the current raw observation; no SAM2 video
predictor or future-frame window is used. The same frame schema is suitable
for a live sensor adapter writing one packet at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OBJECTS = ("funnel", "target_flask")
COLORS_BGR = {"funnel": (0, 140, 255), "target_flask": (225, 90, 150)}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _load_training_geometry_spec(
    path: Path,
    *,
    repo_root: Path,
    meshes: Mapping[str, Path],
    scale_manifest: Mapping[str, float],
) -> dict[str, Any]:
    """Resolve the mesh identity, scales, and deterministic seeds used by training."""

    config_path = path.expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), dict):
        raise ValueError(f"Unsupported training geometry config: {config_path}")
    assets: dict[str, dict[str, Any]] = {}
    for name in OBJECTS:
        raw = payload["assets"].get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"Training geometry config has no {name} asset")
        configured_mesh = Path(str(raw["mesh"]))
        if not configured_mesh.is_absolute():
            configured_mesh = (repo_root / configured_mesh).resolve()
        expected_mesh = meshes[name].expanduser().resolve()
        if configured_mesh != expected_mesh:
            raise ValueError(
                f"{name} mesh differs from training geometry config: {expected_mesh} != {configured_mesh}"
            )
        actual_hash = hashlib.sha256(expected_mesh.read_bytes()).hexdigest()
        expected_hash = raw.get("sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"{name} mesh SHA-256 differs from training geometry config")
        scale = float(raw["scale_m_per_unit"])
        if name not in scale_manifest or not np.isclose(scale, float(scale_manifest[name]), atol=0.0, rtol=0.0):
            raise ValueError(f"{name} scale differs between training config and scale manifest")
        if "sampling_seed" not in raw:
            raise ValueError(f"Training geometry config has no deterministic sampling_seed for {name}")
        assets[name] = {
            "mesh": str(expected_mesh),
            "sha256": actual_hash,
            "scale_m_per_unit": scale,
            "sampling_seed": int(raw["sampling_seed"]),
        }
    motion = payload.get("motion_assumptions", {})
    return {
        "path": str(config_path),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "assets": assets,
        "target_flask_motion": str(motion.get("target_flask", "dynamic")),
        "static_prior_start_frame": int(motion.get("static_prior_start_frame", 1)),
        "rgbtrack": dict(payload.get("rgbtrack", {}).get("validated_parameters", {})),
    }


def _install_training_rgbtrack_runtime(
    *,
    repo_root: Path,
    runtime_dir: Path,
    replay_dir: Path,
    episode: int,
    funnel_mesh: Path,
    flask_mesh: Path,
    scale_manifest: Path,
    calibration_profile: Path,
    weights_root: Path,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Install the exact upright/recovery hooks used by offline production.

    The existing entry point exposes an install-only mode so both training
    replay and deployment use the same candidate selection and recovery code.
    """

    required = {
        "upright_max_tilt_deg",
        "upright_max_rank",
        "follow_max_tilt_deg",
        "recovery_center_max_px",
        "recovery_depth_l1_max_m",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"Training geometry config misses RGBTrack constraints: {missing}")
    if runtime_dir.exists():
        raise FileExistsError(f"RGBTrack runtime audit directory already exists: {runtime_dir}")
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "scripts/object_geometry/pose_tracking"))
    import rgbtrack_native_upright

    argv = [
        str(repo_root / "scripts/object_geometry/pose_tracking/rgbtrack_native_upright.py"),
        "--follow-max-tilt-deg", str(parameters["follow_max_tilt_deg"]),
        "--recovery-center-max-px", str(parameters["recovery_center_max_px"]),
        "--recovery-depth-l1-max-m", str(parameters["recovery_depth_l1_max_m"]),
        "--repo-root", str(repo_root),
        "--dataset", str(replay_dir),
        "--episode", str(int(episode)),
        "--mask-root", str(replay_dir),
        "--funnel-glb", str(funnel_mesh),
        "--flask-glb", str(flask_mesh),
        "--scale-manifest", str(scale_manifest),
        "--weights-root", str(weights_root),
        "--front-calibration-profile", str(calibration_profile),
        "--output", str(runtime_dir),
        "--objects", "funnel", "target_flask",
        "--mode", "depth",
        "--upright-max-tilt-deg", str(parameters["upright_max_tilt_deg"]),
        "--upright-max-rank", str(parameters["upright_max_rank"]),
        "--install-runtime-only",
    ]
    previous_argv = sys.argv
    try:
        sys.argv = argv
        return_code = rgbtrack_native_upright.main()
    finally:
        sys.argv = previous_argv
    if int(return_code) != 0:
        raise RuntimeError(f"RGBTrack training-policy hook installer returned {return_code}")
    return {
        "status": "installed",
        "source": "scripts/object_geometry/pose_tracking/rgbtrack_native_upright.py + rgbtrack_upright_full.py",
        "parameters": {key: parameters[key] for key in sorted(required)},
        "audit_directory": str(runtime_dir),
        "same_hooks_as_training_pose_pipeline": True,
    }


def _read_frames(path: Path, *, follow: bool = False):
    seen: set[Path] = set()
    previous = -1
    while True:
        files = [file for file in sorted(path.glob("frame_*.npz")) if file not in seen]
        for file in files:
            with np.load(file, allow_pickle=False) as source:
                values = {key: np.asarray(source[key]) for key in source.files}
            frame_index = int(np.asarray(values.get("frame_index", 0)).reshape(()))
            if "frame_index" not in values:
                frame_index = int(file.stem.removeprefix("frame_"))
            if frame_index <= previous:
                raise ValueError(f"Replay frame indices must increase strictly: {file}")
            previous = frame_index
            seen.add(file)
            values["frame_index"] = np.asarray(frame_index, dtype=np.int64)
            yield file, values
        if not follow:
            if not seen:
                raise FileNotFoundError(f"No frame_*.npz replay packets in {path}")
            return
        time.sleep(0.02)


def _validate_packet(file: Path, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    required = (
        "front_rgb", "left_rgb", "right_rgb", "front_depth_raw", "robot_state", "timestamp_s", "frame_index"
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{file}: missing required replay arrays {missing}")
    result: dict[str, Any] = {key: np.asarray(data[key]) for key in required}
    for key in ("front_rgb", "left_rgb", "right_rgb"):
        image = result[key]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{file}: {key} must be uint8 RGB [H,W,3], got {image.shape} {image.dtype}")
    front = result["front_rgb"]
    depth = result["front_depth_raw"]
    if depth.ndim != 2 or depth.shape != front.shape[:2] or not np.issubdtype(depth.dtype, np.number):
        raise ValueError(
            f"{file}: front_depth_raw must be color-aligned [H,W]={front.shape[:2]}, got {depth.shape}"
        )
    state = result["robot_state"].astype(np.float32, copy=False)
    if state.shape != (1972,) or not np.isfinite(state).all():
        raise ValueError(f"{file}: robot_state must be finite [1972], got {state.shape}")
    timestamp = float(result["timestamp_s"].reshape(()))
    if not np.isfinite(timestamp):
        raise ValueError(f"{file}: timestamp_s must be finite")
    result["robot_state"] = state
    result["timestamp_s"] = np.asarray(timestamp, dtype=np.float64)
    result["frame_index"] = np.asarray(int(result["frame_index"]), dtype=np.int64)
    for name in OBJECTS:
        mask_key = f"{name}_seed_mask"
        if mask_key in data:
            mask = np.asarray(data[mask_key], dtype=bool)
            if mask.shape != front.shape[:2]:
                raise ValueError(f"{file}: {mask_key} shape {mask.shape} != front RGB {front.shape[:2]}")
            result[mask_key] = mask
    if "tactile_force_norm" in data:
        force = np.asarray(data["tactile_force_norm"], dtype=np.float32)
        if force.shape != (600,):
            raise ValueError(f"{file}: tactile_force_norm must be [600], got {force.shape}")
        result["tactile_force_norm"] = force
    return result


def _mask_prompt(mask: np.ndarray) -> np.ndarray:
    low = cv2.resize(mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    # SAM2 accepts logits here. Use fixed margins around the binary seed mask.
    return np.where(low[None] > 0, 8.0, -8.0).astype(np.float32)


class CausalSAM2Stream:
    """SAM2 image-predictor adapter with current-frame prompts and prior logits."""

    def __init__(self, *, checkpoint: Path, config: str, config_root: Path, device: str, seed_prompts: dict[str, Any], corrections: dict[str, Any], min_area: int, max_fraction: float, min_score: float = 0.05):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        previous_cwd = Path.cwd()
        try:
            os.chdir(config_root)
            model = build_sam2(
                config,
                str(checkpoint),
                device=device,
                apply_postprocessing=True,
                mode="eval",
            )
        finally:
            os.chdir(previous_cwd)
        self.predictor = SAM2ImagePredictor(model)
        self.torch = torch
        self.device = device
        self.prompts = seed_prompts
        self.corrections = corrections
        self.seed_frame = int(seed_prompts.get("seed_frame", 0))
        self.min_area = int(min_area)
        self.max_fraction = float(max_fraction)
        self.min_score = float(min_score)
        if self.min_area <= 0 or not 0.0 < self.max_fraction <= 1.0:
            raise ValueError("SAM2 mask area limits must be positive and max_fraction in (0, 1]")
        if not np.isfinite(self.min_score) or not 0.0 <= self.min_score <= 1.0:
            raise ValueError("SAM2 min_score must be finite and in [0, 1]")
        self.low_res: dict[str, np.ndarray | None] = {name: None for name in OBJECTS}
        self.last_masks: dict[str, np.ndarray | None] = {name: None for name in OBJECTS}
        self.last_area = {name: 0 for name in OBJECTS}
        self.last_valid = {name: False for name in OBJECTS}

    def _spec(self, frame_index: int, name: str) -> tuple[dict[str, Any] | None, bool]:
        frame_corrections = self.corrections.get(str(frame_index), {})
        if name in frame_corrections:
            return dict(frame_corrections[name]), True
        if frame_index == self.seed_frame:
            objects = self.prompts.get("objects", {})
            if name in objects:
                return dict(objects[name]), False
        return None, False

    def _predict_one(self, frame_index: int, name: str, frame_data: Mapping[str, np.ndarray], shape: tuple[int, int]):
        prompt, corrected = self._spec(frame_index, name)
        seed_mask = frame_data.get(f"{name}_seed_mask") if frame_index == self.seed_frame else None
        correction_mask = None
        if corrected and prompt is not None and "mask" in prompt:
            mask_path = Path(prompt["mask"]).expanduser()
            decoded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise FileNotFoundError(mask_path)
            if tuple(decoded.shape) != shape:
                raise ValueError(f"{name} correction mask shape {decoded.shape} != {shape}")
            correction_mask = decoded > 0

        # A reviewed mask is the observation for its current frame. Do not
        # ask SAM2 to reinterpret it and silently replace the human annotation.
        # Its low-resolution logits seed causal propagation on the next frame.
        manual_mask = seed_mask if seed_mask is not None else correction_mask
        if manual_mask is not None:
            mask = np.asarray(manual_mask, dtype=bool).copy()
            area = int(mask.sum())
            max_area = int(shape[0] * shape[1] * self.max_fraction)
            valid = self.min_area <= area <= max_area
            self.last_valid[name] = valid
            if valid:
                self.last_area[name] = area
                self.last_masks[name] = mask.copy()
                self.low_res[name] = _mask_prompt(mask)
            if seed_mask is not None:
                status = "manual_seed" if valid else "invalid_manual_seed_area"
            else:
                status = "corrected" if valid else "invalid_manual_correction_area"
            return mask, 1.0, status, valid, bool(corrected and valid)

        if prompt is None and self.low_res[name] is None and seed_mask is None:
            return np.zeros(shape, dtype=bool), 0.0, "uninitialized", False, False
        kwargs: dict[str, Any] = {"multimask_output": True}
        if self.low_res[name] is not None:
            kwargs["mask_input"] = self.low_res[name]
            kwargs["multimask_output"] = False
            prior = self.last_masks[name]
            if prior is not None and prior.any():
                ys, xs = np.nonzero(prior)
                pad = max(5, int(round(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.12)))
                x0, x1 = max(0, int(xs.min()) - pad), min(shape[1] - 1, int(xs.max()) + pad)
                y0, y1 = max(0, int(ys.min()) - pad), min(shape[0] - 1, int(ys.max()) + pad)
                kwargs["box"] = np.asarray([x0, y0, x1, y1], dtype=np.float32)
                interior = cv2.distanceTransform(prior.astype(np.uint8), cv2.DIST_L2, 5)
                py, px = np.unravel_index(int(np.argmax(interior)), interior.shape)
                other = next((other_mask for other_name, other_mask in self.last_masks.items()
                              if other_name != name and other_mask is not None), None)
                candidates = [
                    (x0, y0), (x1, y0), (x1, y1), (x0, y1),
                    (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2),
                    ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1),
                ]
                negatives = []
                for nx, ny in candidates:
                    if prior[ny, nx] or (other is not None and other[ny, nx]):
                        continue
                    negatives.append((nx, ny))
                    if len(negatives) == 2:
                        break
                points = [(int(px), int(py)), *negatives]
                kwargs["point_coords"] = np.asarray(points, dtype=np.float32)
                kwargs["point_labels"] = np.asarray([1, *([0] * len(negatives))], dtype=np.int32)
        if seed_mask is not None:
            kwargs["mask_input"] = _mask_prompt(seed_mask)
            kwargs["multimask_output"] = False
        if prompt is not None:
            if "box_xyxy" in prompt:
                box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
                if box.shape != (4,) or not np.isfinite(box).all():
                    raise ValueError(f"{name} box_xyxy must be finite [4]")
                kwargs["box"] = box
            if "positive" in prompt or "negative" in prompt:
                pos = np.asarray(prompt.get("positive", []), dtype=np.float32).reshape(-1, 2)
                neg = np.asarray(prompt.get("negative", []), dtype=np.float32).reshape(-1, 2)
                points = np.concatenate((pos, neg), axis=0)
                labels = np.concatenate((np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)))
                if not len(pos):
                    raise ValueError(f"{name} requires at least one positive point")
                kwargs["point_coords"] = points
                kwargs["point_labels"] = labels
            if "mask" in prompt:
                raise AssertionError("Manual correction masks are consumed by the exact-mask path above")
        if not any(key in kwargs for key in ("box", "point_coords", "mask_input")):
            raise ValueError(f"{name} has no seed prompt or causal mask")
        masks, scores, logits = self.predictor.predict(**kwargs)
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        logits = np.asarray(logits)
        selected = int(np.argmax(scores))
        mask = np.asarray(masks[selected], dtype=bool)
        low_res = logits[selected]
        while low_res.ndim > 3 and low_res.shape[0] == 1:
            low_res = low_res[0]
        if low_res.ndim == 2:
            low_res = low_res[None]
        if low_res.shape != (1, 256, 256):
            raise ValueError(f"SAM2 returned unsupported low-res prompt logits shape: {low_res.shape}")
        area = int(mask.sum())
        max_area = int(shape[0] * shape[1] * self.max_fraction)
        ratio_bad = (
            not corrected
            and self.last_area[name] > 0
            and area > 0
            and max(area / self.last_area[name], self.last_area[name] / area) > 4.0
        )
        score_bad = float(scores[selected]) < self.min_score
        valid = self.min_area <= area <= max_area and not ratio_bad and not score_bad
        recovered = valid and not self.last_valid[name] and self.last_area[name] > 0
        invalid_status = "invalid_low_score" if score_bad else "invalid_area_or_jump"
        status = "corrected" if corrected and valid else ("recovered" if recovered else ("observed" if valid else invalid_status))
        if valid:
            self.low_res[name] = low_res.astype(np.float32, copy=True)
            self.last_masks[name] = mask.copy()
            self.last_area[name] = area
        self.last_valid[name] = valid
        return mask, float(scores[selected]), status, valid, corrected or recovered

    def predict(self, frame_index: int, rgb: np.ndarray, data: Mapping[str, np.ndarray]):
        enabled = str(self.device).startswith("cuda")
        previous = {
            name: (self.low_res[name], self.last_masks[name], self.last_area[name])
            for name in OBJECTS
        }
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16, enabled=enabled
        ):
            self.predictor.set_image(rgb)
            results = {
                name: self._predict_one(frame_index, name, data, rgb.shape[:2])
                for name in OBJECTS
            }
        funnel, target = results["funnel"][0], results["target_flask"][0]
        overlap = int(np.logical_and(funnel, target).sum())
        min_area = max(1, min(int(funnel.sum()), int(target.sum())))
        conflict = overlap / min_area > 0.15
        if conflict:
            results = {
                name: (mask, score, "overlapping_objects", False, recovery)
                for name, (mask, score, _status, _valid, recovery) in results.items()
            }
            for name in OBJECTS:
                self.low_res[name], self.last_masks[name], self.last_area[name] = previous[name]
                self.last_valid[name] = False
        return results, overlap


def _draw_overlay(rgb: np.ndarray, masks: Mapping[str, np.ndarray], rows: Mapping[str, dict[str, Any]], frame_index: int) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name in OBJECTS:
        mask = np.asarray(masks[name], dtype=bool)
        color = COLORS_BGR[name]
        if mask.any():
            tint = image.copy()
            tint[mask] = color
            image = cv2.addWeighted(tint, 0.25, image, 0.75, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
        row = rows[name]
        short_name = "target" if name == "target_flask" else "funnel"
        label = (
            f"{short_name}: A={row['area_px']} {row['status']} "
            f"V={int(row['valid'])} R={int(row['recovery'])}"
        )
        y = 24 if name == "funnel" else 50
        cv2.putText(image, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"frame={frame_index} causal", (8, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def _invalid_pose(frame_index: int) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "T_base_link_object": np.eye(4, dtype=np.float64),
        "valid": False,
        "confidence": 0.0,
        "status": "invalid_no_observation",
        "pose_source": "unknown",
        "observed_pose_valid": False,
    }


def _save_packet(path: Path, values: Mapping[str, Any], geometry: Mapping[str, Any]) -> None:
    arrays = dict(values)
    for key, value in geometry.items():
        if key == "hand_scene":
            for subkey, subvalue in value.items():
                if subkey == "causal_tactile":
                    for tactile_key, tactile_value in subvalue.items():
                        arrays[f"object_geometry__hand_scene__causal_tactile__{tactile_key}"] = np.asarray(tactile_value)
                else:
                    arrays[f"object_geometry__hand_scene__{subkey}"] = np.asarray(subvalue)
        else:
            arrays[f"object_geometry__{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "checkpoints/sam2/sam2.1_hiera_small.pt")
    parser.add_argument("--sam2-source", type=Path, default=Path.home() / ".cache/openpi_deploy/sam2")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2-backend", choices=("video_memory", "legacy_image"), default="video_memory",
                        help="Use the reviewed Level-1 SAM2 video-memory algorithm; legacy_image is for A/B comparison only.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--scale-manifest", type=Path, default=ROOT / "data/pose_eval/hand_scene_metric_mesh_scales.json")
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=ROOT / "configs/object_geometry/datasets/1018_peg_all/config.json",
        help="Canonical training geometry config; controls mesh hashes, scales, and point-sampling seeds.",
    )
    parser.add_argument("--funnel-mesh", type=Path, default=ROOT / "assets/object_geometry/synthware_19_22/funnel_F181975.glb")
    parser.add_argument("--target-flask-mesh", type=Path, default=ROOT / "assets/object_geometry/synthware_19_22/flask_F301950.glb")
    parser.add_argument("--weights-root", type=Path, default=ROOT / "checkpoints/rgbtrack")
    parser.add_argument("--front-calibration-profile", type=Path, default=ROOT / "configs/ur7e_xhand/front_calibration_candidate.json")
    parser.add_argument("--episode", type=int, default=23, help="Episode ID for provenance and existing RGBTrack hook initialization.")
    parser.add_argument("--pose-mode", choices=("depth", "no_depth", "unknown_scale"), default="depth")
    parser.add_argument("--masks-only", action="store_true", help="Run SAM2 causal masks and contract geometry without RGBTrack pose estimation.")
    parser.add_argument("--static-target-anchor-json", type=Path)
    parser.add_argument(
        "--static-target-anchor-source",
        choices=("human_reviewed", "reference_geometry"),
        help="Provenance for the static flask anchor; reference_geometry is replay-only.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--follow", action="store_true", help="Watch the packet directory for new frames from a live sensor adapter.")
    parser.add_argument("--min-mask-area-px", type=int, default=40)
    parser.add_argument("--max-mask-fraction", type=float, default=0.35)
    parser.add_argument("--min-sam2-score", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when set")
    root = args.repo_root.expanduser().resolve()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    replay_dir = args.replay_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    for name in OBJECTS:
        (output_dir / "masks" / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)

    sam2_source = args.sam2_source.expanduser().resolve()
    if str(sam2_source) not in sys.path:
        sys.path.insert(0, str(sam2_source))
    if not args.sam2_checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"SAM2 checkpoint is missing: {args.sam2_checkpoint}; see scripts/deploy/download_weights.py")
    prompts = _read_json(args.prompts)
    corrections = _read_json(args.corrections)
    if args.sam2_backend == "video_memory":
        from scripts.deploy.sam2_video_stream import CausalVideoSAM2Stream
        stream_type = CausalVideoSAM2Stream
    else:
        stream_type = CausalSAM2Stream
    sam2_stream = stream_type(
        checkpoint=args.sam2_checkpoint.expanduser().resolve(),
        config=args.sam2_config,
        config_root=sam2_source,
        device=args.device,
        seed_prompts=prompts,
        corrections=corrections,
        min_area=args.min_mask_area_px,
        max_fraction=args.max_mask_fraction,
        min_score=args.min_sam2_score,
    )

    sys.path.insert(0, str(root / "scripts/object_geometry/pose_tracking"))
    from rgbtrack_adapter import RGBTrackObject, convert_glb_to_rgbtrack_mesh, load_front_calibration
    from openpi.policies.hand_scene_online import OnlineObjectGeometryBuilder

    calibration = load_front_calibration(root, args.front_calibration_profile)
    scales = OnlineObjectGeometryBuilder.load_scale_manifest(args.scale_manifest)
    geometry_spec = _load_training_geometry_spec(
        args.geometry_config,
        repo_root=root,
        meshes={"funnel": args.funnel_mesh, "target_flask": args.target_flask_mesh},
        scale_manifest=scales,
    )
    funnel_mesh_manifest = convert_glb_to_rgbtrack_mesh(
        source_glb=args.funnel_mesh,
        output_root=output_dir / "rgbtrack_meshes",
        object_name="funnel",
        scale_m_per_unit=scales["funnel"],
    )
    flask_mesh_manifest = convert_glb_to_rgbtrack_mesh(
        source_glb=args.target_flask_mesh,
        output_root=output_dir / "rgbtrack_meshes",
        object_name="target_flask",
        scale_m_per_unit=scales["target_flask"],
    )
    static_anchor = None
    static_anchor_metadata = None
    if args.static_target_anchor_json is not None:
        anchor_payload = json.loads(args.static_target_anchor_json.read_text(encoding="utf-8"))
        static_anchor_metadata = anchor_payload if isinstance(anchor_payload, dict) else None
        anchor_value = anchor_payload.get("T_base_link_object") if isinstance(anchor_payload, dict) else anchor_payload
        static_anchor = np.asarray(anchor_value, dtype=np.float64)
        if args.static_target_anchor_source is None:
            raise ValueError("A static target prior requires --static-target-anchor-source")
        if args.static_target_anchor_source == "reference_geometry":
            try:
                replay_dir.relative_to(root / "data" / "deploy" / "replay")
            except ValueError as exc:
                raise ValueError("reference_geometry anchors are allowed only for saved replay data") from exc
            if args.follow:
                raise ValueError("reference_geometry anchors cannot be used with a live --follow stream")
    elif args.static_target_anchor_source is not None:
        raise ValueError("--static-target-anchor-source requires --static-target-anchor-json")
    if geometry_spec["target_flask_motion"] == "static" and static_anchor is None:
        raise ValueError(
            "Training geometry config declares target_flask static; supply its explicit frame-0 anchor "
            "with --static-target-anchor-json and --static-target-anchor-source"
        )
    builder = OnlineObjectGeometryBuilder.from_meshes(
        funnel_mesh=args.funnel_mesh,
        flask_mesh=args.target_flask_mesh,
        scales=scales,
        sampling_seeds={name: entry["sampling_seed"] for name, entry in geometry_spec["assets"].items()},
        static_target_anchor=static_anchor,
        static_target_anchor_provenance=args.static_target_anchor_source,
        static_target_anchor_start_frame=geometry_spec["static_prior_start_frame"],
    )
    rgbtrack_runtime = None
    if not args.masks_only:
        rgbtrack_runtime = _install_training_rgbtrack_runtime(
            repo_root=root,
            runtime_dir=output_dir / "rgbtrack_training_policy",
            replay_dir=replay_dir,
            episode=args.episode,
            funnel_mesh=args.funnel_mesh,
            flask_mesh=args.target_flask_mesh,
            scale_manifest=args.scale_manifest,
            calibration_profile=args.front_calibration_profile,
            weights_root=args.weights_root,
            parameters=geometry_spec["rgbtrack"],
        )
    tracker_by_name: dict[str, Any] = {}
    if not args.masks_only:
        tracker_by_name = {
            name: RGBTrackObject(
                repo_root=root,
                object_name=name,
                mesh_path=(funnel_mesh_manifest if name == "funnel" else flask_mesh_manifest)["output_mesh"],
                mode=args.pose_mode,
                debug_dir=output_dir / "rgbtrack_debug",
                weights_root=args.weights_root,
                init_frame=int(prompts.get("init_frames", {}).get(name, sam2_stream.seed_frame)),
                min_mask_area_px=args.min_mask_area_px,
            )
            for name in OBJECTS
        }
    # Force the optional JAX import to CPU; vision CUDA stays with PyTorch.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from openpi.spatial.config import make_baseline_config
    from openpi.spatial.preprocess import SpatialPreprocessor
    from openpi.spatial.calibration_profile import apply_front_calibration_profile

    spatial_config = make_baseline_config().with_camera_roles("front")
    profile_path = args.front_calibration_profile.expanduser().resolve()
    try:
        profile_path = profile_path.relative_to(root)
    except ValueError:
        pass
    spatial_config = apply_front_calibration_profile(spatial_config, profile_path)
    spatial_preprocessor = SpatialPreprocessor.from_repo_root(repo_root=root, config=spatial_config)

    records = output_dir / "manifest.jsonl"
    frame_count = 0
    last_timestamp = -np.inf
    with records.open("x", encoding="utf-8") as manifest:
        try:
            for number, (source_path, raw) in enumerate(_read_frames(replay_dir, follow=args.follow)):
                if args.max_frames is not None and number >= max(0, args.max_frames):
                    break
                started = time.perf_counter()
                packet = _validate_packet(source_path, raw)
                frame_index = int(packet["frame_index"])
                timestamp_s = float(packet["timestamp_s"])
                if timestamp_s < last_timestamp:
                    raise ValueError(f"Non-monotonic timestamp at frame {frame_index}")
                last_timestamp = timestamp_s
                rgb = packet["front_rgb"]
                sam2_started = time.perf_counter()
                masks, overlap = sam2_stream.predict(frame_index, rgb, packet)
                sam2_latency_ms = (time.perf_counter() - sam2_started) * 1000.0
                object_masks = {name: masks[name][0] for name in OBJECTS}
                pose_rows: dict[str, dict[str, Any]] = {}
                rgbtrack_latency_ms = 0.0
                for name in OBJECTS:
                    mask, score, status, valid, recovery = masks[name]
                    corrected = status == "corrected"
                    object_pose_started = time.perf_counter()
                    if corrected and name in tracker_by_name:
                        tracker_by_name[name] = RGBTrackObject(
                            repo_root=root,
                            object_name=name,
                            mesh_path=(funnel_mesh_manifest if name == "funnel" else flask_mesh_manifest)["output_mesh"],
                            mode=args.pose_mode,
                            debug_dir=output_dir / "rgbtrack_debug",
                            weights_root=args.weights_root,
                            init_frame=frame_index,
                            min_mask_area_px=args.min_mask_area_px,
                        )
                    if name not in tracker_by_name:
                        pose_rows[name] = _invalid_pose(frame_index)
                    else:
                        depth_m = packet["front_depth_raw"].astype(np.float32) * calibration.depth_scale_m_per_unit
                        result = tracker_by_name[name].estimate(
                            frame_index=frame_index,
                            rgb=rgb,
                            depth_m=depth_m,
                            mask=mask if valid else np.zeros_like(mask),
                            K=calibration.K_color,
                            T_base_color=calibration.T_base_color,
                        )
                        pose_rows[name] = {
                            "frame_index": frame_index,
                            "T_base_link_object": result.T_base_object,
                            "valid": bool(result.valid and valid),
                            "confidence": float(result.confidence if valid else 0.0),
                            "status": result.status if valid else status,
                            "pose_source": "rgbtrack_observed" if result.valid and valid else "rgbtrack_invalid",
                            "observed_pose_valid": bool(result.valid and valid),
                        }
                    pose_rows[name]["sam2_score"] = float(score)
                    pose_rows[name]["mask_area_px"] = int(mask.sum())
                    pose_rows[name]["mask_valid"] = bool(valid)
                    pose_rows[name]["mask_status"] = status
                    pose_rows[name]["recovery"] = bool(recovery)
                    object_pose_ms = (time.perf_counter() - object_pose_started) * 1000.0
                    pose_rows[name]["rgbtrack_latency_ms"] = object_pose_ms if name in tracker_by_name else 0.0
                    rgbtrack_latency_ms += pose_rows[name]["rgbtrack_latency_ms"]

                geometry_started = time.perf_counter()
                if "tactile_force_norm" in packet:
                    force_norm = packet["tactile_force_norm"]
                else:
                    force_norm = spatial_preprocessor.preprocess_tactile(packet["robot_state"]).force_norm
                funnel_offset = packet.get("funnel_mask_hand_offset_xy")
                geometry = builder.build_spatial_object_geometry(
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    funnel_pose=pose_rows["funnel"],
                    target_flask_pose=pose_rows["target_flask"],
                    tactile_force_norm=force_norm,
                    funnel_mask=object_masks["funnel"] if masks["funnel"][3] else None,
                    target_flask_mask=object_masks["target_flask"] if masks["target_flask"][3] else None,
                    funnel_mask_hand_offset_xy=funnel_offset,
                )
                geometry_latency_ms = (time.perf_counter() - geometry_started) * 1000.0
                geometry_dict = geometry
                geometry_dict.pop("episode_index", None)
                geometry_dict.pop("frame_index", None)
                mask_rows = {
                    name: {
                        "area_px": int(object_masks[name].sum()),
                        "valid": bool(masks[name][3]),
                        "status": masks[name][2],
                        "recovery": bool(masks[name][4]),
                        "score": float(masks[name][1]),
                    }
                    for name in OBJECTS
                }
                overlay = _draw_overlay(rgb, object_masks, mask_rows, frame_index)
                cv2.imwrite(str(output_dir / "overlays" / f"{frame_index:06d}.jpg"), overlay)
                for name in OBJECTS:
                    cv2.imwrite(
                        str(output_dir / "masks" / name / f"{frame_index:06d}.png"),
                        object_masks[name].astype(np.uint8) * 255,
                    )
                saved = {
                    "frame_index": packet["frame_index"],
                    "timestamp_s": packet["timestamp_s"],
                    "front_rgb": packet["front_rgb"],
                    "left_rgb": packet["left_rgb"],
                    "right_rgb": packet["right_rgb"],
                    "front_depth_raw": packet["front_depth_raw"],
                    "robot_state": packet["robot_state"],
                    "funnel_mask": object_masks["funnel"],
                    "target_flask_mask": object_masks["target_flask"],
                    "T_base_link_funnel": pose_rows["funnel"]["T_base_link_object"],
                    "T_base_link_target_flask": pose_rows["target_flask"]["T_base_link_object"],
                    "funnel_pose_valid": np.asarray(pose_rows["funnel"]["valid"]),
                    "target_flask_pose_valid": np.asarray(pose_rows["target_flask"]["valid"]),
                }
                if "episode_index" in packet:
                    saved["episode_index"] = packet["episode_index"]
                _save_packet(output_dir / "frames" / f"frame_{frame_index:06d}.npz", saved, geometry_dict)
                total_latency_ms = (time.perf_counter() - started) * 1000.0
                row = {
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "objects": pose_rows,
                    "masks": mask_rows,
                    "mask_overlap_px": overlap,
                    "sam2_latency_ms": sam2_latency_ms,
                    "rgbtrack_latency_ms": rgbtrack_latency_ms,
                    "geometry_latency_ms": geometry_latency_ms,
                    "spatial_latency_ms": None,
                    "policy_latency_ms": None,
                    "elapsed_ms": total_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "geometry_shape": {
                        "manipulated_xyz_m": list(geometry_dict["manipulated_xyz_m"].shape),
                        "target_xyz_m": list(geometry_dict["target_xyz_m"].shape),
                    },
                    "geometry_finite": bool(
                        np.isfinite(geometry_dict["manipulated_xyz_m"]).all()
                        and np.isfinite(geometry_dict["target_xyz_m"]).all()
                    ),
                    "source_packet": str(source_path),
                    "future_frames_used": False,
                }
                manifest.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
                manifest.flush()
                frame_count += 1
                print(f"[FRAME] {frame_index} mask={mask_rows} overlap={overlap} elapsed_ms={row['elapsed_ms']:.1f}", flush=True)
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
        except KeyboardInterrupt:
            print("[STOP] stream interrupted; finalized all completed frames", flush=True)
    if frame_count == 0:
        raise ValueError("No replay frame selected")
    constraint_report = None
    if rgbtrack_runtime is not None:
        import rgbtrack_native_upright
        constraint_report = rgbtrack_native_upright.write_runtime_report()
        if constraint_report is not None:
            rgbtrack_runtime["event_count"] = len(constraint_report["funnel_events"])
            rgbtrack_runtime["status_counts"] = constraint_report["summary"]
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "live_packet_watch" if args.follow else "causal_frame_replay",
                "future_frames_used": False,
                "sam2_checkpoint": str(args.sam2_checkpoint.expanduser().resolve()),
                "sam2_checkpoint_sha256": __import__("hashlib").sha256(args.sam2_checkpoint.read_bytes()).hexdigest(),
                "sam2_min_score": args.min_sam2_score if args.sam2_backend == "legacy_image" else None,
                "sam2_backend": args.sam2_backend,
                "sam2_score_available": args.sam2_backend == "legacy_image",
                "sam2_source": str(sam2_source),
                "rgbtrack_enabled": not args.masks_only,
                "rgbtrack_submodule_revision": "23e43f832b82f4813d6ab72773779bf8d2ee4d83",
                "scale_manifest": str(args.scale_manifest.expanduser().resolve()),
                "training_geometry": geometry_spec,
                "static_target_anchor_source": args.static_target_anchor_source,
                "static_target_anchor": static_anchor_metadata,
                "rgbtrack_training_policy": rgbtrack_runtime,
                "front_calibration": calibration.provenance,
                "num_frames": frame_count,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
