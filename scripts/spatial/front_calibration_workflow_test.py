"""Synthetic regression for the two-stage front-camera residual fit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import SpatialPreprocessor


MODULE_PATH = Path(__file__).with_name("front_calibration_workflow.py")
SPEC = importlib.util.spec_from_file_location("front_calibration_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def test_two_stage_fit_recovers_translation_and_color_intrinsics(tmp_path: Path) -> None:
    pre = SpatialPreprocessor.from_repo_root(
        repo_root=workflow.REPO_ROOT,
        config=make_baseline_config().with_camera_roles("front"),
    )
    camera = pre.calibration_bundle.cameras["front"]
    T = camera.T_base_color
    k = np.array([
        camera.color_intrinsics.fx,
        camera.color_intrinsics.fy,
        camera.color_intrinsics.cx,
        camera.color_intrinsics.cy,
    ])
    delta = np.array([0.003, -0.010, 0.004])
    fitted_k = k + np.array([7.0, -5.0, 2.0, -1.5])
    rng = np.random.default_rng(42)
    camera_points = rng.uniform([-0.3, -0.2, 0.5], [0.3, 0.2, 1.0], size=(24, 3))
    base_points = camera_points @ T[:3, :3].T + T[:3, 3]
    rows = []
    for group, points, intrinsics in (("base", base_points[:12], k), ("tip", base_points[12:], fitted_k)):
        uv = workflow._project(points, T, intrinsics, delta)
        rows.extend(
            {"group": group, "frame_index": index, "base_xyz_m": point.tolist(), "observed_uv": pixel.tolist()}
            for index, (point, pixel) in enumerate(zip(points, uv, strict=True))
        )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({
        "schema_version": 1, "dataset": "synthetic", "episode": 0, "annotations": rows,
    }))
    base_profile = tmp_path / "base.json"
    workflow.fit(SimpleNamespace(
        annotations=annotations, stage="base", base_profile=None, output=base_profile,
    ))
    measured_base = json.loads(base_profile.read_text())
    np.testing.assert_allclose(measured_base["front_extrinsic_translation_base_m"], delta, atol=1e-5)
    combined = tmp_path / "combined.json"
    workflow.fit(SimpleNamespace(
        annotations=annotations, stage="tip", base_profile=base_profile, output=combined,
    ))
    measured = json.loads(combined.read_text())
    np.testing.assert_allclose(
        [measured["front_color_intrinsics"][key] for key in ("fx", "fy", "cx", "cy")],
        fitted_k,
        atol=1e-4,
    )
