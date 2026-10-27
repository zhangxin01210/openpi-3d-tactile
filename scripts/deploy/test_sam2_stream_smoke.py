#!/usr/bin/env python3
"""GPU smoke for the causal SAM2 image-predictor stream adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _synthetic_scene(shift: int = 0) -> np.ndarray:
    image = np.full((512, 512, 3), 32, dtype=np.uint8)
    polygon = np.asarray([[105 + shift, 90], [265 + shift, 90], [230 + shift, 320], [140 + shift, 320]], dtype=np.int32)
    cv2.fillPoly(image, [polygon], (210, 170, 70))
    cv2.rectangle(image, (340 + shift, 175), (420 + shift, 405), (80, 190, 230), thickness=-1)
    cv2.rectangle(image, (352 + shift, 155), (408 + shift, 180), (100, 210, 240), thickness=-1)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam2/sam2.1_hiera_small.pt")
    parser.add_argument("--source", type=Path, default=Path.home() / ".cache/openpi_deploy/sam2")
    parser.add_argument("--output", type=Path, default=ROOT / "output/deploy/sam2_stream_smoke")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_online_perception import CausalSAM2Stream, _draw_overlay

    prompts = {
        "seed_frame": 0,
        "objects": {
            "funnel": {"box_xyxy": [95, 75, 280, 335]},
            "target_flask": {"box_xyxy": [330, 140, 435, 420]},
        },
    }
    stream = CausalSAM2Stream(
        checkpoint=args.checkpoint.resolve(),
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        config_root=args.source.resolve(),
        device="cuda",
        seed_prompts=prompts,
        corrections={},
        min_area=20,
        max_fraction=0.8,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame_index, image in enumerate((_synthetic_scene(0), _synthetic_scene(8))):
        masks, overlap = stream.predict(frame_index, image, {})
        mask_arrays = {name: masks[name][0] for name in ("funnel", "target_flask")}
        visual_rows = {
            name: {
                "area_px": int(mask_arrays[name].sum()),
                "valid": bool(masks[name][3]),
                "status": masks[name][2],
                "recovery": bool(masks[name][4]),
                "score": float(masks[name][1]),
            }
            for name in mask_arrays
        }
        if any(mask.shape != image.shape[:2] for mask in mask_arrays.values()):
            raise AssertionError("SAM2 returned a mask with the wrong frame dimensions")
        if any(not np.isfinite(row[1]) for row in masks.values()):
            raise AssertionError("SAM2 returned a non-finite score")
        cv2.imwrite(str(args.output / f"overlay_{frame_index:06d}.jpg"), _draw_overlay(image, mask_arrays, visual_rows, frame_index))
        for name, mask in mask_arrays.items():
            cv2.imwrite(str(args.output / f"{name}_{frame_index:06d}.png"), mask.astype(np.uint8) * 255)
        rows.append({"frame_index": frame_index, "objects": visual_rows, "overlap_px": overlap})
    report = {
        "status": "PASS",
        "test": "synthetic_causal_image_prompt_smoke",
        "device": "cuda",
        "frames_processed_in_order": [0, 1],
        "future_frames_used": False,
        "results": rows,
        "accuracy_claim": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
