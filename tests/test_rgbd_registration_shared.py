"""Tests for shared raw-depth-to-RGB rasterization and SDK matrix convention.

Run: PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_rgbd_registration_shared.py
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from openpi.spatial.rgbd_registration import align_raw_depth_to_color


@dataclass(frozen=True)
class K:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Cam:
    depth_intrinsics: K
    color_intrinsics: K
    T_color_depth: np.ndarray
    depth_scale_m_per_unit: float = 0.001


def camera(h=40,w=50,ratio=1.,tx=0.):
    kd=K(28.,28.,w/2,h/2)
    kc=K(28.*ratio,28.*ratio,w/2,h/2)
    T=np.eye(4)
    T[0,3]=tx
    return Cam(kd,kc,T)


def test_identity_projection_is_pixel_exact():
    raw=np.full((40,50),900,np.uint16)
    got=align_raw_depth_to_color(raw,camera(),(40,50))
    assert got.shape==(40,50)
    assert np.count_nonzero(got)==got.size
    assert np.allclose(got,.9,atol=1e-5)


def test_magnification_fills_footprints_not_single_pixel_dots():
    raw=np.full((40,50),900,np.uint16)
    sparse_zeros=raw.size
    got=align_raw_depth_to_color(raw,camera(ratio=1.6),(40,50))
    assert np.count_nonzero(got)>0.80*sparse_zeros
    assert np.allclose(got[got>0],.9,atol=1e-5)


def test_nearest_z_wins_occlusion():
    raw=np.full((40,50),1200,np.uint16)
    raw[18:24,22:29]=650
    got=align_raw_depth_to_color(raw,camera(ratio=1.6),(40,50))
    assert np.min(got[got>0])<.7
    assert np.max(got)<1.201
    assert np.any(got==np.float32(.65000004)) or np.any(np.isclose(got,.65,atol=1e-5))
    # Central foreground stays in front of the 1.2m background.
    assert got[20,25]<.8


def test_invalid_and_out_of_range_raw_depth_not_used():
    raw=np.full((40,50),900,np.uint16)
    raw[0,0]=0
    raw[0,1]=65535
    got=align_raw_depth_to_color(raw,camera(),(40,50))
    assert got[0,0]==0
    assert got[0,1]==0
    assert got[20,20]>0


def test_real_sdk_rotation_is_valid():
    R=np.array([
       [0.999976634979248, -0.00312922359444201, 0.006081835366785526],
       [0.003140633925795555,0.9999933242797852,-0.0018675047904253006],
       [-0.006075950805097818,0.0018865619786083698,0.9999797344207764],
    ])
    legacy=R.T.copy()
    assert np.allclose(legacy.T,R,atol=1e-12)
    assert np.allclose(R@R.T,np.eye(3),atol=2e-5)
    # Explicit row-vs-column-vector distinction at a nonzero 3D point.
    v=np.array([-.08,.15,.87])
    assert np.linalg.norm(R@v - legacy@v)>0.005
