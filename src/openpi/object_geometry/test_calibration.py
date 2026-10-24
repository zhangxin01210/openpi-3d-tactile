"""Regression tests for the formal front-calibration geometry contract."""

from __future__ import annotations

import copy
from pathlib import Path

from openpi.object_geometry.calibration import calibration_provenance_error
from openpi.object_geometry.calibration import front_calibration_provenance
from openpi.object_geometry.calibration import make_front_geometry_config

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE = "configs/ur7e_xhand/front_calibration_candidate.json"


def test_front_geometry_config_matches_profiled_spatial_contract() -> None:
    config, profile = make_front_geometry_config(REPO_ROOT, PROFILE)

    assert profile == REPO_ROOT / PROFILE
    assert config.visual.camera_roles == ("front",)
    assert config.diagnostics.enable_front_extrinsic_correction
    assert config.diagnostics.front_extrinsic_translation_base_m == (0.0, -0.012, 0.005)
    assert config.diagnostics.enable_front_intrinsic_k1
    assert (
        config.diagnostics.front_color_k1.fx,
        config.diagnostics.front_color_k1.fy,
        config.diagnostics.front_color_k1.cx,
        config.diagnostics.front_color_k1.cy,
    ) == (634.8879216573664, 617.5755080760925, 318.8070728820338, 235.91048429182663)

    provenance = front_calibration_provenance(REPO_ROOT, profile, config=config)
    assert provenance["coordinate_frame"] == "base_link"
    assert provenance["projection_contract"]["depth_to_base"] == "active front T_base_color"
    assert provenance["projection_contract"]["rgb_mask_projection"] == "active front color intrinsics"
    assert calibration_provenance_error(provenance, provenance) is None


def test_profile_digest_and_applied_intrinsics_are_validated() -> None:
    config, profile = make_front_geometry_config(REPO_ROOT, PROFILE)
    expected = front_calibration_provenance(REPO_ROOT, profile, config=config)

    wrong_profile = copy.deepcopy(expected)
    wrong_profile["profile"]["sha256"] = "not-the-requested-profile"
    assert "profile sha256" in calibration_provenance_error(wrong_profile, expected)

    wrong_intrinsics = copy.deepcopy(expected)
    wrong_intrinsics["applied_front_color_intrinsics"]["cx"] += 1.0
    assert "applied_front_color_intrinsics" in calibration_provenance_error(wrong_intrinsics, expected)
