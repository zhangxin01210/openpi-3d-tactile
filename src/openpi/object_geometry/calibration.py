"""Calibration contract shared by formal Level-1 geometry generation and export.

The object clouds are selected in the front RGB image and represented in
``base_link``.  Both the depth-to-base transform and the RGB projection must
therefore use exactly the same front profile as the spatial sidecar.  Keeping
the small provenance record here makes that contract inspectable without
loading a dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openpi.spatial.calibration_profile import apply_front_calibration_profile
from openpi.spatial.config import SpatialPreprocessConfig
from openpi.spatial.config import make_baseline_config


def resolve_front_calibration_profile(repo_root: Path, profile: Path | str) -> Path:
    """Resolve a profile relative to the checkout and require an ordinary file."""
    candidate = Path(profile).expanduser()
    if not candidate.is_absolute():
        candidate = Path(repo_root) / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Front calibration profile does not exist: {candidate}")
    return candidate


def make_front_geometry_config(
    repo_root: Path, profile: Path | str | None,
) -> tuple[SpatialPreprocessConfig, Path | None]:
    """Build the exact single-front-camera config used by the spatial sidecar."""
    config = make_baseline_config().with_camera_roles("front")
    if profile is None:
        return config, None
    resolved = resolve_front_calibration_profile(repo_root, profile)
    return apply_front_calibration_profile(config, resolved), resolved


def front_calibration_provenance(
    repo_root: Path,
    profile: Path | str | None,
    *,
    config: SpatialPreprocessConfig | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe record of the calibration actually used.

    ``profile_path`` is informational; consumers compare the profile digest and
    applied values so a checkout can move without weakening validation.
    """
    if config is None:
        config, resolved = make_front_geometry_config(repo_root, profile)
    elif profile is None:
        resolved = None
    else:
        resolved = resolve_front_calibration_profile(repo_root, profile)

    diagnostics = config.diagnostics
    applied_intrinsics: dict[str, float] | None = None
    if diagnostics.enable_front_intrinsic_k1:
        intrinsic = diagnostics.front_color_k1
        applied_intrinsics = {
            "fx": float(intrinsic.fx),
            "fy": float(intrinsic.fy),
            "cx": float(intrinsic.cx),
            "cy": float(intrinsic.cy),
        }
    applied_translation: list[float] | None = None
    if diagnostics.enable_front_extrinsic_correction:
        applied_translation = [float(value) for value in diagnostics.front_extrinsic_translation_base_m]

    record: dict[str, Any] = {
        "schema_version": 1,
        "camera_role": "front",
        "coordinate_frame": "base_link",
        "calibration_mode": "profile" if resolved is not None else "baseline_unprofiled",
        "projection_contract": {
            "depth_to_base": "active front T_base_color",
            "rgb_mask_projection": "active front color intrinsics",
        },
        "applied_front_extrinsic_translation_base_m": applied_translation,
        "applied_front_color_intrinsics": applied_intrinsics,
    }
    if resolved is not None:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        try:
            profile_path = str(resolved.relative_to(Path(repo_root).resolve()))
        except ValueError:
            profile_path = str(resolved)
        record["profile"] = {
            "path": profile_path,
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "schema_version": raw.get("schema_version"),
        }
    return record


def calibration_provenance_error(actual: Any, expected: dict[str, Any]) -> str | None:
    """Explain why a source report is not compatible with the requested profile."""
    if not isinstance(actual, dict):
        return "missing calibration provenance"
    for field in (
        "schema_version",
        "camera_role",
        "coordinate_frame",
        "calibration_mode",
        "projection_contract",
        "applied_front_extrinsic_translation_base_m",
        "applied_front_color_intrinsics",
    ):
        if actual.get(field) != expected.get(field):
            return f"calibration {field} differs from requested front profile"
    actual_profile = actual.get("profile")
    expected_profile = expected.get("profile")
    if not isinstance(actual_profile, dict) or not isinstance(expected_profile, dict):
        return "missing profiled calibration provenance"
    for field in ("sha256", "schema_version"):
        if actual_profile.get(field) != expected_profile.get(field):
            return f"calibration profile {field} differs from requested front profile"
    return None
