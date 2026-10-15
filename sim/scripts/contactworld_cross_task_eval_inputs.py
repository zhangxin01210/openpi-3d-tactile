"""Online model inputs shared by the ContactWorld cross-task evaluator.

The three tasks use the same bilateral RGB-D/TacFF observation contract.  Only
the language prompt and action dimensionality differ.  The policy checkpoints
are task-specific, so this module keeps the four evaluated configurations
explicit rather than accidentally accepting one of the USB-only names.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from contactworld_cross_task_common import TASKS

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from contactworld_v3_eval_inputs import snapshot  # noqa: E402


SUFFIXES = ("01_rgb", "02_fused_pc", "07_fused_pc_base_pooled", "10_fused_pc_base_map_matched")
CONFIGS = tuple(
    f"pi0_cw3_{task}_{suffix}" for task in TASKS for suffix in SUFFIXES
)
_CLOUD_SUFFIXES = {SUFFIXES[1], SUFFIXES[2], SUFFIXES[3]}
_TACTILE_SUFFIXES = {SUFFIXES[2], SUFFIXES[3]}


def config_task(config: str) -> str:
    if config not in CONFIGS:
        raise ValueError(f"Unknown cross-task config: {config}")
    for task in TASKS:
        if config.startswith(f"pi0_cw3_{task}_"):
            return task
    raise ValueError(f"Cannot parse task from config: {config}")


def config_suffix(config: str) -> str:
    task = config_task(config)
    prefix = f"pi0_cw3_{task}_"
    return config[len(prefix):]


def policy_input(snap: dict[str, np.ndarray], config: str, prompt: str) -> dict:
    """Convert an all-sensor simulator snapshot to the trained policy schema."""
    suffix = config_suffix(config)
    request = {
        "observation/front_rgb": snap["front"],
        "observation/wrist_rgb": snap["wrist"],
        "observation/state": snap["state"],
        "prompt": prompt,
    }
    visual = None
    tactile = None
    if suffix in _CLOUD_SUFFIXES:
        visual = {
            "xyz_m": snap["cloud_xyz"],
            "rgb": np.zeros((4096, 3), np.uint8),
            "rgb_valid": np.zeros(4096, bool),
            "point_mask": np.ones(4096, bool),
        }
    if suffix in _TACTILE_SUFFIXES:
        force = snap["force_base"].reshape(280, 3)
        tactile = {
            "xyz_m": snap["taxel_xyz_base"].reshape(280, 3),
            "force": force,
            "force_norm": np.linalg.norm(force, axis=-1),
            "finger_id": np.repeat(np.array([0, 1], np.int32), 140),
            "taxel_id": np.tile(np.arange(140, dtype=np.int32), 2),
            "point_mask": np.ones(280, bool),
        }
    if visual is not None or tactile is not None:
        request["spatial"] = {"visual": visual, "tactile": tactile}
    return request

