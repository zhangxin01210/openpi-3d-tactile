"""Shared training/inference front RGB-D preprocessing; no tracker or GT inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FrontRGBDPreprocessor:
    """Use the project's calibrated raw-depth->base cloud/RGB projection once per frame."""

    def __init__(self, repo_root: Path, profile: str | Path):
        from openpi.object_geometry.calibration import make_front_geometry_config
        from openpi.spatial.preprocess import SpatialPreprocessor

        root = Path(repo_root).resolve()
        config, _ = make_front_geometry_config(root, profile)
        self._preprocessor = SpatialPreprocessor.from_repo_root(repo_root=root, config=config)

    def __call__(self, rgb: np.ndarray, raw_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb, raw_depth = np.asarray(rgb), np.asarray(raw_depth)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("front RGB must be uint8 [H,W,3]")
        if raw_depth.ndim != 2 or raw_depth.dtype != np.uint16:
            raise ValueError("front depth must be raw, unaligned uint16 [H,W]")
        visual = self._preprocessor.preprocess_visual(depth_by_role={"front": raw_depth}, rgb_by_role={"front": rgb})
        if visual.merged_roi_count == 0 or visual.voxel_candidate_count == 0:
            raise ValueError("front RGB-D produced no valid workspace points")
        # RGB association failures are explicitly masked, never presented as black observations.
        mask = np.asarray(visual.rgb_valid, dtype=bool)
        if not mask.any():
            raise ValueError("no scene points have valid RGB projection")
        scene = np.concatenate((visual.xyz_m, visual.rgb.astype(np.float32) / 255.0), axis=-1)
        return scene.astype(np.float32), mask

    def color_camera(self):
        """Return the same profile-adjusted camera used by RGB projection."""
        from openpi.spatial.geometry import apply_diagnostic_overrides
        processor = self._preprocessor
        return apply_diagnostic_overrides(processor.calibration_bundle.cameras, processor.config.diagnostics)["front"]
