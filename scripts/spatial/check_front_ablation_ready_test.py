"""Manifest checks for the three-card front-only ablation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from check_front_ablation_ready import check_asset_provenance, check_manifest


def fixture():
    profile = {
        "front_extrinsic_translation_base_m": [0, -0.012, 0.005],
        "front_color_intrinsics": {"fx": 634.9, "fy": 617.6, "cx": 318.8, "cy": 235.9},
    }
    manifest = {
        "status": "complete", "spatial_version": "v1_front_candidate",
        "spatial_contract": {"camera_roles": ["front"], "visual_points_per_frame": 4096,
                             "tactile_points_per_frame": 600},
        "preprocess_config": {"visual": {"camera_roles": ["front"]}, "diagnostics": {
            "enable_front_extrinsic_correction": True,
            "enable_front_intrinsic_k1": True,
            "front_extrinsic_translation_base_m": profile["front_extrinsic_translation_base_m"],
            "front_color_k1": profile["front_color_intrinsics"],
        }},
        "storage": {"episodes": [{"selected_frame_count": 2, "source_frame_count": 2}]},
    }
    info = {"total_episodes": 1, "total_frames": 2}
    return manifest, info, profile


def test_complete_corrected_front_sidecar_is_accepted() -> None:
    manifest, info, profile = fixture()
    check_manifest(manifest, info, profile, version="v1_front_candidate")


@pytest.mark.parametrize("change", ("roles", "calibration", "frames"))
def test_stale_or_incomplete_sidecar_is_rejected(change: str) -> None:
    manifest, info, profile = fixture()
    manifest = copy.deepcopy(manifest)
    if change == "roles":
        manifest["spatial_contract"]["camera_roles"] = ["front", "left"]
    elif change == "calibration":
        manifest["preprocess_config"]["diagnostics"]["front_color_k1"]["fx"] += 1
    else:
        manifest["storage"]["episodes"][0]["selected_frame_count"] = 1
    with pytest.raises(ValueError):
        check_manifest(manifest, info, profile, version="v1_front_candidate")


def test_changed_calibration_asset_is_rejected(tmp_path: Path) -> None:
    names = (
        "camera_internal_config", "camera_external_config", "robot_urdf",
        "state_mapping", "tactile_t16_transformed", "tactile_t30_right_transformed",
    )
    asset = tmp_path / "asset.json"
    asset.write_text("before")
    manifest = {"asset_provenance": {
        name: {"path": "asset.json", "sha256": hashlib.sha256(b"before").hexdigest()}
        for name in names
    }}
    check_asset_provenance(manifest, repo_root=tmp_path)
    asset.write_text("after")
    with pytest.raises(ValueError):
        check_asset_provenance(manifest, repo_root=tmp_path)
