"""Explicit front-camera diagnostic calibration profiles shared by offline and online paths."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.spatial.config import PinholeIntrinsics, SpatialPreprocessConfig


def apply_front_calibration_profile(config: SpatialPreprocessConfig, path: Path) -> SpatialPreprocessConfig:
    profile = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError(f"Unsupported front calibration profile schema: {path}")
    unknown = set(profile) - {
        "schema_version", "front_extrinsic_translation_base_m", "front_color_intrinsics", "provenance"
    }
    if unknown:
        raise ValueError(f"Unknown front calibration profile fields: {sorted(unknown)}")
    diagnostics = config.diagnostics
    if "front_extrinsic_translation_base_m" in profile:
        translation = tuple(float(x) for x in profile["front_extrinsic_translation_base_m"])
        if len(translation) != 3 or not np.all(np.isfinite(translation)):
            raise ValueError("front_extrinsic_translation_base_m must be three finite numbers")
        diagnostics = dataclasses.replace(
            diagnostics,
            enable_front_extrinsic_correction=True,
            front_extrinsic_translation_base_m=translation,
        )
    if "front_color_intrinsics" in profile:
        values = profile["front_color_intrinsics"]
        if set(values) != {"fx", "fy", "cx", "cy"}:
            raise ValueError("front_color_intrinsics requires exactly fx, fy, cx, cy")
        intrinsic = PinholeIntrinsics(**{key: float(value) for key, value in values.items()})
        if not np.all(np.isfinite((intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy))):
            raise ValueError("front_color_intrinsics must be finite")
        diagnostics = dataclasses.replace(
            diagnostics,
            enable_front_intrinsic_k1=True,
            front_color_k1=intrinsic,
        )
    return dataclasses.replace(config, diagnostics=diagnostics)
