"""Causal SAM2 *video-memory* adapter for the pinned SAM2.1 predictor.

Unlike repeatedly prompting ``SAM2ImagePredictor`` with a previous mask, this
adapter calls the SAME video predictor methods used by the reviewed HOI Level-1
pipeline: ``add_new_points_or_box`` / ``add_new_mask`` and
``propagate_in_video``. Only the current frame is appended to the predictor
state, and propagation is limited to that one frame.

Important: extending ``inference_state['images']`` and ``num_frames`` is a
version-pinned adaptation of SAM2's internal state, not a public SAM2 streaming
API. Run the real-data parity QA before using this in robot-control loops.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

OBJECT_IDS = {"target_flask": 1, "funnel": 2}  # Same ids as Level-1.


class CausalVideoSAM2Stream:
    """One evolving SAM2VideoPredictor state; no future frames or hand subtraction."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        config: str,
        config_root: Path,
        device: str,
        seed_prompts: dict[str, Any],
        corrections: dict[str, Any],
        min_area: int,
        max_fraction: float,
        min_score: float = 0.05,
        max_frames_per_episode: int = 2000,
    ) -> None:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if min_area < 1 or not 0 < max_fraction <= 1:
            raise ValueError("Invalid mask area thresholds")
        if max_frames_per_episode < 1:
            raise ValueError("max_frames_per_episode must be positive")
        before = Path.cwd()
        try:
            os.chdir(config_root)
            self.predictor = build_sam2_video_predictor(
                config, str(checkpoint), device=device,
                apply_postprocessing=True,
            )
        finally:
            os.chdir(before)
        self.torch = torch
        self.device = device
        self.prompts = seed_prompts
        self.corrections = corrections
        self.seed_frame = int(seed_prompts.get("seed_frame", seed_prompts.get("frame_index", 0)))
        self.init_frames = {
            name: int(seed_prompts.get("init_frames", {}).get(name, self.seed_frame))
            for name in OBJECT_IDS
        }
        self.min_area = int(min_area)
        self.max_fraction = float(max_fraction)
        self.max_frames_per_episode = max_frames_per_episode
        # Video Predictor emits mask logits, NOT per-frame IoU scores.
        self.score_available = False
        self.state = None
        self.shape: tuple[int, int] | None = None
        self.previous_frame: int | None = None
        self.seeded: set[str] = set()
        self.last_area = {name: 0 for name in OBJECT_IDS}
        self.last_valid = {name: False for name in OBJECT_IDS}

    def _normalized_frame(self, rgb: np.ndarray):
        from PIL import Image

        # Match sam2.utils.misc._load_img_as_tensor (PIL resize / ImageNet).
        size = int(self.predictor.image_size)
        image = np.asarray(Image.fromarray(rgb).resize((size, size)), dtype=np.float32) / 255.0
        frame = self.torch.from_numpy(image).permute(2, 0, 1)
        mean = frame.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = frame.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (frame - mean) / std

    def _append_current_image(self, rgb: np.ndarray) -> int:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected uint8 HWC RGB frame")
        if self.shape is not None and rgb.shape[:2] != self.shape:
            raise ValueError("Camera resolution changed mid-episode")
        from PIL import Image

        if self.state is None:
            # Official init_state creates the SAM2 memory structures using
            # ONLY the first available frame. Replace warm-up JPEG pixels with
            # original RGB, invalidate its cached visual features.
            with tempfile.TemporaryDirectory(prefix="sam2_video_seed_") as temp:
                Image.fromarray(rgb).save(Path(temp) / "00000.jpg", quality=100, subsampling=0)
                state = self.predictor.init_state(
                    str(temp), offload_video_to_cpu=True, offload_state_to_cpu=True,
                )
            if not isinstance(state.get("images"), (list, self.torch.Tensor)):
                raise RuntimeError("SAM2 state interface changed; streaming adaptation not validated")
            state["images"] = [self._normalized_frame(rgb)]
            state["num_frames"] = 1
            state["cached_features"].clear()
            self.shape = rgb.shape[:2]
            self.state = state
            return 0
        if len(self.state["images"]) != self.state["num_frames"]:
            raise RuntimeError("SAM2 video-state frame count mismatch")
        if self.state["num_frames"] >= self.max_frames_per_episode:
            raise RuntimeError("SAM2 video state limit reached; reset between episodes")
        self.state["images"].append(self._normalized_frame(rgb))
        self.state["num_frames"] += 1
        return self.state["num_frames"] - 1

    def _prompt(self, frame_index: int, name: str, packet: Mapping[str, np.ndarray]):
        corrected = self.corrections.get(str(frame_index), {}).get(name)
        if corrected is not None:
            return dict(corrected), True, None
        if frame_index == self.init_frames[name] and name not in self.seeded:
            mask = packet.get(f"{name}_seed_mask")
            return self.prompts.get("objects", {}).get(name), False, mask
        return None, False, None

    def _register_prompt(self, ordinal: int, frame_index: int, name: str, packet: Mapping[str, np.ndarray]):
        import cv2

        prompt, corrected, seed_mask = self._prompt(frame_index, name, packet)
        if prompt is None and seed_mask is None:
            return None
        prompt = prompt or {}
        if seed_mask is None and "mask" in prompt:
            image = cv2.imread(str(Path(prompt["mask"]).expanduser()), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(prompt["mask"])
            seed_mask = image > 0
        if seed_mask is not None:
            seed_mask = np.asarray(seed_mask, dtype=bool)
            if seed_mask.shape != self.shape:
                raise ValueError(f"{name} seed mask shape {seed_mask.shape}, expected {self.shape}")
            self.predictor.add_new_mask(self.state, frame_idx=ordinal,
                                        obj_id=OBJECT_IDS[name], mask=seed_mask)
            source = "manual_correction_mask" if corrected else "manual_seed_mask"
        else:
            # Match Level-1 reviewed positive/negative point semantics.
            # Box-only prompt is explicitly rejected in parity mode.
            if "positive" not in prompt or "negative" not in prompt:
                raise ValueError(f"{name}: reviewed positive AND negative point lists required")
            positives = np.asarray(prompt["positive"], dtype=np.float32).reshape(-1, 2)
            negatives = np.asarray(prompt["negative"], dtype=np.float32).reshape(-1, 2)
            if len(positives) == 0:
                raise ValueError(f"{name}: empty positive point prompts")
            points = np.concatenate((positives, negatives), axis=0)
            if (not np.isfinite(points).all() or np.any(points < 0) or
                np.any(points[:, 0] >= self.shape[1]) or np.any(points[:, 1] >= self.shape[0])):
                raise ValueError(f"{name}: nonfinite or out-of-bounds points")
            labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=self.state, frame_idx=ordinal,
                obj_id=OBJECT_IDS[name], points=points, labels=labels,
            )
            source = "manual_correction_points" if corrected else "reviewed_points"
        self.seeded.add(name)
        return (source, seed_mask)

    def predict(self, frame_index: int, rgb: np.ndarray, data: Mapping[str, np.ndarray]):
        if self.previous_frame is not None and frame_index <= self.previous_frame:
            raise ValueError("Video Predictor requires strictly increasing frame indices")
        if self.state is None and frame_index > min(self.init_frames.values()):
            raise ValueError("First frame must include the first object initialization frame")
        enabled = str(self.device).startswith("cuda")
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16, enabled=enabled
        ):
            ordinal = self._append_current_image(rgb)
            manual = {
                name: self._register_prompt(ordinal, frame_index, name, data)
                for name in OBJECT_IDS
            }
            if not self.seeded:
                raise ValueError("No SAM2 objects initialized: provide reviewed points or seed masks")
            propagated = list(self.predictor.propagate_in_video(
                self.state, start_frame_idx=ordinal, max_frame_num_to_track=0,
                reverse=False,
            ))
        if len(propagated) != 1 or int(propagated[0][0]) != ordinal:
            raise RuntimeError("SAM2 did not propagate exactly the current frame")
        _, ids, logits = propagated[0]
        ids = [int(obj_id) for obj_id in ids]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicated SAM2 object IDs")
        masks = {}
        for name, obj_id in OBJECT_IDS.items():
            if obj_id in ids:
                tensor = logits[ids.index(obj_id)]
                mask = np.asarray((tensor > 0).detach().cpu().numpy(), dtype=bool).squeeze()
                if mask.shape != self.shape:
                    raise RuntimeError(f"SAM2 produced wrong shape for {name}: {mask.shape}")
            else:
                mask = np.zeros(self.shape, dtype=bool)
            # Exact manual annotation is the output for its own seed frame.
            if manual[name] is not None and manual[name][1] is not None:
                mask = np.asarray(manual[name][1], dtype=bool).copy()
            masks[name] = mask
        overlap = int(np.logical_and(masks["funnel"], masks["target_flask"]).sum())
        smallest = max(1, min(int(m.sum()) for m in masks.values()))
        conflict = overlap > 0.15 * smallest
        result = {}
        for name, mask in masks.items():
            area = int(mask.sum())
            ratio_bad = (
                self.last_area[name] > 0 and area > 0 and
                max(area / self.last_area[name], self.last_area[name] / area) > 4.0
            )
            valid = (name in self.seeded and self.min_area <= area <=
                     int(mask.size * self.max_fraction) and not ratio_bad and not conflict)
            status = (
                ("corrected" if "correction" in manual[name][0] else "manual_seed") if manual[name] is not None else
                "video_tracked" if valid else
                "uninitialized" if name not in self.seeded else
                "invalid_overlap" if conflict else "invalid_area_or_jump"
            )
            recovered = (valid and not self.last_valid[name] and
                         self.last_area[name] > 0 and manual[name] is None)
            if valid:
                self.last_area[name] = area
            self.last_valid[name] = valid
            # Video Predictor has no calibrated per-frame IoU score.
            result[name] = (mask, -1.0, status, valid,
                            bool(recovered or (manual[name] and status == "corrected")))
        self.previous_frame = frame_index
        return result, overlap
