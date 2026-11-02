"""Explicit six-position Level-1 prompt selection; never auto-classify scene geometry.

Five complete reviewed templates plus episode-0 flask points are reference
2D clicks, NOT reference images or 6D object poses. For position 1 (e0),
reuse episode-23 FUNNEL clicks; keep episode-0's own FLASK clicks.
All clicks are applied to the CURRENT robot camera first frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_EPISODES = (0, 23, 38, 49, 58, 65)
# Operator-facing values, unlike internal LeRobot episode indexes.
SCENE_POSITION_TO_EPISODE = {1: 0, 2: 23, 3: 38, 4: 49, 5: 58, 6: 65}
EPISODE_ZERO_FUNNEL_DONOR = 23
OBJECTS = ("funnel", "target_flask")
TEMPLATE_ROOT = Path("tools/hoi_level1_snapshot/metadata/six_positions")


def resolve_prompts(
    *,
    repo_root: Path,
    prompt_file: Path | None,
    template_episode: int | None,
    template_confirmed: bool,
    first_frame_rgb_shape: tuple[int, int, int],
) -> dict[str, Any]:
    if (prompt_file is None) == (template_episode is None):
        raise ValueError("Choose exactly one: --prompts or --position-template-episode")
    if template_episode is not None:
        if template_episode not in REFERENCE_EPISODES:
            raise ValueError(f"Unknown position reference {template_episode}; allowed {REFERENCE_EPISODES}")
        if not template_confirmed:
            raise ValueError("Template reuse requires --template-reviewed-for-current-scene; inspect current frame first")
        if template_episode == 0:
            source = repo_root / TEMPLATE_ROOT / "episode_e00_flask_prompts.json"
            donor_path = repo_root / TEMPLATE_ROOT / f"episode_e{EPISODE_ZERO_FUNNEL_DONOR:02d}_seed_prompts.json"
            donor = json.loads(donor_path.read_text(encoding="utf-8"))
            if donor.get("manual_reviewed_prompts") is not True or donor.get("episode_index") != EPISODE_ZERO_FUNNEL_DONOR:
                raise ValueError("Funnel fallback source is not a reviewed prompt template")
        else:
            source = repo_root / TEMPLATE_ROOT / f"episode_e{template_episode:02d}_seed_prompts.json"
    else:
        if template_confirmed:
            raise ValueError("--template-reviewed-for-current-scene only applies to a selected template")
        source = prompt_file.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if template_episode == 0:
        if payload.get("episode_index") != 0 or "funnel" in payload.get("objects", {}):
            raise ValueError("Episode-0 reference must contain its own flask-only clicks")
        if (payload.get("image_width"), payload.get("image_height")) != (
            donor.get("image_width"), donor.get("image_height")
        ):
            raise ValueError("Borrowed funnel point dimensions differ from Episode-0 flask clicks")
        payload = {**payload, "objects": {
            "target_flask":payload["objects"]["target_flask"],
            "funnel":donor["objects"]["funnel"],
        }}
    height, width, channels = first_frame_rgb_shape
    if channels != 3 or (payload.get("image_width"), payload.get("image_height")) != (width, height):
        raise ValueError(f"Prompt image size differs from current scene {width}x{height}: {source}")
    if int(payload.get("frame_index", -1)) != 0:
        raise ValueError("Initial prompts must be defined on frame zero")
    if template_episode is not None:
        if (template_episode != 0 and payload.get("manual_reviewed_prompts") is not True) or int(payload.get("episode_index", -1)) != template_episode:
            raise ValueError(f"Repository template lacks reviewed provenance: {source}")
    for name in OBJECTS:
        prompt = payload.get("objects", {}).get(name)
        if not isinstance(prompt, dict):
            raise ValueError(f"Missing reviewed {name} points in {source}")
        for kind in ("positive", "negative"):
            points = prompt.get(kind)
            if not isinstance(points, list) or (kind == "positive" and not points):
                raise ValueError(f"{name}.{kind} needs a point list (positive nonempty)")
            for point in points:
                if (not isinstance(point, list) or len(point) != 2 or
                    not all(isinstance(v, (int, float)) for v in point) or
                    not (0 <= point[0] < width and 0 <= point[1] < height)):
                    raise ValueError(f"{name}.{kind} contains out-of-bounds/invalid click: {point}")
    return {
        "frame_index": 0,
        "seed_frame": 0,
        "init_frames": {name: 0 for name in OBJECTS},
        "objects": {name: payload["objects"][name] for name in OBJECTS},
        "initialization_provenance": {
            "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mode": "explicit_position_template" if template_episode is not None else "reviewed_prompts_file",
            "template_episode": template_episode,
            "scene_position": (
                next((k for k,v in SCENE_POSITION_TO_EPISODE.items() if v == template_episode), None)
                if template_episode is not None else None
            ),
            "borrowed_funnel_keypoints_from_episode": (
                EPISODE_ZERO_FUNNEL_DONOR if template_episode == 0 else None
            ),
            "reference_images_or_poses_reused": False,
            "template_reviewed_for_current_scene": bool(template_confirmed),
            "seed_kind": "positive_negative_points",
        },
    }


# Reference matching shares the validated preprocessing algorithms.
@dataclass
class MaskResult:
    success: bool
    masks: dict[str, np.ndarray]
    provenance: dict[str, Any]
    error: str | None = None


def initialize_masks(reference_root: Path, frame_bgr: np.ndarray, predictor: Any, *,
                     max_reference_score: float | None = None, min_reference_margin: float | None = None,
                     manual_reference: int | None = None, min_sam_score: float = 0.0) -> MaskResult:
    """Initialize two masks using the actual offline reference algorithms.

    Multi-reference automatic selection requires explicitly calibrated score and margin
    thresholds. Manual reference is an operator fallback, not a confidence claim.
    """
    from scripts.preprocess import sam2_task
    from scripts.preprocess import qa_peg_six_reference_selection as peg
    provenance: dict[str, Any] = {"candidates": []}
    try:
        if not np.isfinite(min_sam_score) or not 0 <= min_sam_score <= 1:
            raise ValueError("min_sam_score must be finite and between 0 and 1")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR image")
        root = Path(reference_root)
        manifest = json.loads((root / "manifest.json").read_text())
        names = manifest.get("objects", [])
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError("Reference manifest requires exactly two distinct objects")
        strategy = manifest.get("initialization_strategy")
        if strategy not in {"multi_reference_points", "single_reference_mask_match"}:
            raise ValueError("Reference initialization strategy mismatch")
        provenance["task"] = manifest.get("task")
        for rel, digest in manifest["sha256"].items():
            path = (root / rel).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Reference asset integrity failure: {rel}")
        refs = {int(r["reference_episode"]): r for r in manifest["references"]}
        if strategy == "multi_reference_points":
            if len(refs) < 2:
                raise ValueError("Automatic selection requires at least two references")
            features = peg.extract_features(frame_bgr)
            for ep, ref in refs.items():
                image = sam2_task.img(root / ref["path"] / "reference_frame.jpg")
                score = peg.pair_score(features, peg.extract_features(image), "table_roi")
                if not np.isfinite(score):
                    raise ValueError("Nonfinite reference score")
                provenance["candidates"].append({"episode": ep, "score": score})
            provenance["candidates"].sort(key=lambda x: x["score"])
            best, second = provenance["candidates"][:2]
            provenance["margin"] = second["score"] - best["score"]
            if manual_reference is None:
                if (max_reference_score is None or min_reference_margin is None or
                    not np.isfinite(max_reference_score) or not np.isfinite(min_reference_margin) or
                    max_reference_score < 0 or min_reference_margin < 0):
                    raise ValueError("Calibrated reference score and margin thresholds required")
                if best["score"] > max_reference_score or provenance["margin"] < min_reference_margin:
                    raise ValueError("Low-confidence Peg reference match")
                selected = best["episode"]
                provenance["source"] = "automatic_reference_points"
            else:
                if manual_reference not in refs:
                    raise ValueError("Unknown manual Peg reference")
                selected = manual_reference
                provenance["source"] = "manual_reference_points"
        else:
            if manual_reference is not None or len(refs) != 1:
                raise ValueError("Single-reference task requires exactly one reference")
            selected = next(iter(refs))
            provenance["source"] = "single_reference_template_match"
        provenance["reference_episode"] = selected
        refdir = root / refs[selected]["path"]
        reference = sam2_task.img(refdir / "reference_frame.jpg")
        if reference.shape != frame_bgr.shape:
            raise ValueError("Current and reference image dimensions differ")
        masks = {}
        provenance["objects"] = {}
        if strategy == "multi_reference_points":
            prompts = json.loads((refdir / "prompts.json").read_text())
            for name in names:
                points = prompts["objects"][name]
                pos = np.asarray(points["positive"], dtype=np.float32).reshape(-1, 2)
                neg = np.asarray(points["negative"], dtype=np.float32).reshape(-1, 2)
                if not len(pos) or any(not (0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]) for x, y in np.concatenate((pos, neg))):
                    raise ValueError(f"Invalid {name} prompts")
                mask, score = sam2_task.sam_predict(predictor, frame_bgr, pos, neg)
                masks[name] = mask
                provenance["objects"][name] = {"sam_score": float(score)}
        else:
            for name in names:
                refmask = sam2_task.read_mask(refdir / "templates" / f"{name}.png", reference.shape[:2])
                mask, score, match = sam2_task.predict_object(predictor, frame_bgr, reference, refmask)
                masks[name] = mask
                provenance["objects"][name] = {"sam_score": float(score), "match": match}
                if match.get("second_vs_best_ratio") is not None and match["second_vs_best_ratio"] < 1.05:
                    raise ValueError(f"Ambiguous template match for {name}")
        for name, mask in masks.items():
            score = provenance["objects"][name]["sam_score"]
            if mask.shape != frame_bgr.shape[:2] or not np.any(mask) or not np.isfinite(score) or score < min_sam_score:
                raise ValueError(f"Invalid SAM2 result for {name}")
        return MaskResult(True, masks, provenance)
    except Exception as exc:
        return MaskResult(False, {}, provenance, f"{type(exc).__name__}: {exc}")


def save_initialization_report(result: MaskResult, output_path: Path) -> None:
    """Persist success or failure and candidate scores without serializing masks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump({"success": result.success, "error": result.error,
                   "provenance": result.provenance}, stream, indent=2, allow_nan=False)
        stream.write("\n")
