"""Explicit six-position Level-1 prompt selection; never auto-classify scene geometry.

Five complete reviewed templates plus episode-0 flask points are reference
2D clicks, NOT reference images or 6D object poses. For position 1 (e0),
reuse episode-23 FUNNEL clicks; keep episode-0's own FLASK clicks.
All clicks are applied to the CURRENT robot camera first frame.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
