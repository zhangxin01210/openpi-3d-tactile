"""Raw depth optical pixels -> registered RGB optical metric depth.

Single authoritative alignment for RGBTrack. Training point clouds should use
raw depth with their original depth-camera intrinsics and T_base_depth: do NOT
round-trip through this color-aligned raster.

Each raw depth pixel covers a finite source pixel *area*. Four projected corner
rays define a conservative destination rectangle (normally ~1-3px across for
D435I 640x480). Destination pixels covered by multiple depth footprints keep
the nearest depth (Z-buffer), so a background pixel does not overwrite a nearer
foreground surface. No global interpolation or inpainting is performed.

Requires raw, UNALIGNED depth. Data known to be aligned already must bypass
this function. Lens distortion is not modeled here: the calibration files used
by this project report all-zero distortion coefficients for these profiles.
"""

from __future__ import annotations

import numpy as np


def align_raw_depth_to_color(
    raw_depth: np.ndarray,
    calibration,
    color_hw: tuple[int, int],
    *,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.5,
    max_footprint_px: int = 6,
) -> np.ndarray:
    """Return [Hc,Wc] float32 depth (meters) in COLOR optical coordinates.

    `calibration` is the existing CameraCalibration: depth_intrinsics,
    color_intrinsics, T_color_depth (DEPTH -> COLOR), depth_scale_m_per_unit.

    Unobserved destination pixels are zero. Does not fabricate measurements.
    Raises on implausibly large footprints instead of silently painting areas.
    """
    raw = np.asarray(raw_depth)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"raw_depth must be a numeric HxW array; got {raw.shape}/{raw.dtype}")
    hc, wc = map(int, color_hw)
    if hc < 1 or wc < 1:
        raise ValueError(f"color_hw must be positive, got {color_hw}")
    if not (0 < min_depth_m < max_depth_m):
        raise ValueError("Require 0 < min_depth_m < max_depth_m")
    if max_footprint_px < 1 or max_footprint_px > 16:
        raise ValueError("max_footprint_px must be in 1..16")
    kd, kc = calibration.depth_intrinsics, calibration.color_intrinsics
    for role, k in (("depth", kd), ("color", kc)):
        if not all(np.isfinite(x) and x > 0 for x in (float(k.fx), float(k.fy))):
            raise ValueError(f"{role} intrinsics invalid")
    T = np.asarray(calibration.T_color_depth, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1], atol=1e-7):
        raise ValueError("T_color_depth must be a finite 4x4 transform")
    scale = float(calibration.depth_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid depth scale")

    z = raw.astype(np.float32, copy=False) * np.float32(scale)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    out = np.zeros((hc, wc), dtype=np.float32)
    if not np.any(valid):
        return out
    v, u = np.nonzero(valid)
    z = z[v, u].astype(np.float64)
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    R, t = T[:3, :3], T[:3, 3]

    # Four source pixel corners at the *same measured depth* define its footprint.
    # For RGB-D devices here, the rectangles normally contain 1-4 color pixels.
    xs = []
    ys = []
    zc = None
    for du, dv in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
        xd = (u + du - float(kd.cx)) * z / float(kd.fx)
        yd = (v + dv - float(kd.cy)) * z / float(kd.fy)
        xcol = R[0,0]*xd + R[0,1]*yd + R[0,2]*z + t[0]
        ycol = R[1,0]*xd + R[1,1]*yd + R[1,2]*z + t[1]
        zcol = R[2,0]*xd + R[2,1]*yd + R[2,2]*z + t[2]
        # The center/corner color-z differs by under a millimeter for real D435I;
        # average corner-z is a stable representative of this source pixel depth.
        if zc is None:
            zc = np.zeros_like(zcol)
        zc += .25*zcol
        with np.errstate(divide="ignore", invalid="ignore"):
            xs.append(float(kc.fx)*xcol/zcol + float(kc.cx))
            ys.append(float(kc.fy)*ycol/zcol + float(kc.cy))

    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)
    finite = (np.isfinite(xs).all(axis=0) & np.isfinite(ys).all(axis=0) &
              np.isfinite(zc) & (zc > 0))
    # Filter non-finite before integer conversion to avoid warnings/overflows.
    if not np.any(finite):
        return out
    xs, ys, zc = xs[:,finite], ys[:,finite], zc[finite].astype(np.float32)
    xmin = np.ceil(np.min(xs, axis=0)).astype(np.int64)
    xmax = np.floor(np.max(xs, axis=0)).astype(np.int64)
    ymin = np.ceil(np.min(ys, axis=0)).astype(np.int64)
    ymax = np.floor(np.max(ys, axis=0)).astype(np.int64)

    widths = xmax - xmin + 1
    heights = ymax - ymin + 1
    in_range = ((xmax >= 0) & (xmin < wc) & (ymax >= 0) & (ymin < hc) &
                (widths > 0) & (heights > 0))
    too_wide = in_range & ((widths > max_footprint_px) | (heights > max_footprint_px))
    if np.any(too_wide):
        raise ValueError(
            f"Depth-to-color footprint exceeds {max_footprint_px}px in "
            f"{int(too_wide.sum())} pixels; check intrinsics/profile/units"
        )
    if not np.any(in_range):
        return out
    xmin, xmax = xmin[in_range], xmax[in_range]
    ymin, ymax = ymin[in_range], ymax[in_range]
    zc = zc[in_range]
    flat = np.full(hc*wc, np.inf, dtype=np.float32)
    # Vectorized Z-buffer per footprint offset; only near-foreground wins.
    # Usually 2x2; loop bound is the actual largest footprint in the image.
    max_dx = int(np.max(xmax - xmin))
    max_dy = int(np.max(ymax - ymin))
    for dy in range(max_dy+1):
        yy = ymin + dy
        row = (yy <= ymax) & (yy >= 0) & (yy < hc)
        if not np.any(row):
            continue
        for dx in range(max_dx+1):
            xx = xmin + dx
            good = row & (xx <= xmax) & (xx >= 0) & (xx < wc)
            if np.any(good):
                np.minimum.at(flat, yy[good]*wc + xx[good], zc[good])
    flat[~np.isfinite(flat)] = 0.
    return flat.reshape(hc, wc)
