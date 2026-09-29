"""Synthetic depth/CAD rasterization and residual checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).with_name("audit_front_depth_cad.py")
SPEC = importlib.util.spec_from_file_location("audit_front_depth_cad", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_depth_raster_and_signed_residual() -> None:
    assert audit.array_fingerprint(np.array([1, 2], np.uint16)) != audit.array_fingerprint(
        np.array([1, 3], np.uint16)
    )
    triangle = SimpleNamespace(
        vertices=np.array([[-0.2, -0.2, 1], [0.2, -0.2, 1], [0, 0.2, 1]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    links = {link for _, group in audit.CHAIN_GROUPS for link in group}
    meshes = dict.fromkeys(links, triangle)
    transforms = {link: np.eye(4) for link in links}
    camera = SimpleNamespace(
        depth_intrinsics=SimpleNamespace(fx=100, fy=100, cx=32, cy=32),
        T_base_color=np.eye(4), T_color_depth=np.eye(4),
    )
    pred, groups = audit.render_cad_depth(meshes, transforms, camera, (64, 64))
    assert abs(pred[32, 32] - 1) < 1e-5
    assert groups[32, 32] == 0
    assert not np.isfinite(pred[0, 0])

    obs = np.full((64, 64), 1.01, np.float32)
    report, residual, valid, matched = audit.evaluate(obs, pred, groups, 2, 0.05)
    assert valid[32, 32] and matched[32, 32]
    assert abs(residual[32, 32] - 0.01) < 1e-5
    assert abs(report["near_surface"]["signed_median_mm"] - 10) < 1e-3

    obs[32, 32] = 0
    report, _, valid, _ = audit.evaluate(obs, pred, groups, 2, 0.05)
    assert not valid[32, 32]
    assert report["observed_interior_pixels"] < report["interior_pixels"]

    sheet = audit.make_depth_root_to_tip(obs, groups, audit.CHAIN_GROUPS)
    assert sheet.shape[1] == 4 * 380
    assert np.any(sheet != 0)
