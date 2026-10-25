"""Contract tests for the 1018_peg_all production batch boundary."""

from __future__ import annotations

import json
from pathlib import Path

from openpi.object_geometry.cli import _load_config
from openpi.object_geometry.cli import _load_approval
from openpi.object_geometry.cli import _position_group
from openpi.object_geometry.cli import _write_auto_qa_review
from openpi.object_geometry.production import sha256_file


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


def test_auto_qa_preserves_existing_human_approval(tmp_path: Path) -> None:
    pose_run = tmp_path / "pose_run"
    pose_run.mkdir()
    (pose_run / "manifest.json").write_text(json.dumps({"episode": 23}), encoding="utf8")
    review_root = tmp_path / "scratch" / "pose_reviews" / "e023"
    review_root.mkdir(parents=True)
    human_review = {
        "schema_version": 1,
        "episode": 23,
        "status": "human_accepted",
        "approved_by": "reviewer",
        "review_note": "validated RGB+3D",
        "pose_run": str(pose_run),
        "pose_manifest_sha256": sha256_file(pose_run / "manifest.json"),
    }
    approved_path = review_root / "approved_pose.json"
    approved_path.write_text(json.dumps(human_review, indent=2) + "\n", encoding="utf8")

    quality = {"episode": 23, "status": "auto_qa_passed", "auto_qa_passed": True, "checks": {"all": True}, "metrics": {"frame_count": 10}}
    preserved = _write_auto_qa_review(review_root, quality)
    assert preserved is not None
    assert json.loads(approved_path.read_text(encoding="utf8"))["status"] == "human_accepted"
    assert (review_root / "auto_qa.json").is_file()

    loaded = _load_approval(tmp_path / "scratch", 23)
    assert loaded["status"] == "human_accepted"
    assert loaded["approved_by"] == "reviewer"
    assert loaded["auto_qa_passed"] is True
    assert loaded["metrics"]["frame_count"] == 10
