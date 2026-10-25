"""Contract tests for the 1018_peg_all production batch boundary."""

from __future__ import annotations

import json
from pathlib import Path

from openpi.object_geometry.cli import _load_config
from openpi.object_geometry.cli import _position_group


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/object_geometry/datasets/1018_peg_all/config.json"


def test_regression_episode_groups_and_e23_mask_override() -> None:
    config, _, _ = _load_config(CONFIG)
    assert _position_group(config, 23)["name"] == "position_1"
    assert _position_group(config, 38)["name"] == "position_2"
    assert _position_group(config, 65)["name"] == "position_5"
    assert config["episode_overrides"]["mask_root"]["23"].endswith("level1_sequence_e23_maskfix_v1/masks")
    assert config["paths"]["spatial_version"] == "v1_front_candidate"
    assert config["output_version"] == "v1_hand_scene_q2"
    assert config["qa_thresholds"]["allow_missing_tactile"] is False


def test_config_is_valid_json_and_declares_batch_statuses() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf8"))
    assert raw["review"]["statuses"] == ["human_accepted", "auto_qa_passed", "pending_review", "rejected"]
    assert raw["review"]["reference_human_accepted"] == [23, 38, 65]
