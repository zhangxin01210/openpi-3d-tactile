"""Offline raw RGB-D registration and CAD overlays for front-camera audits."""

from __future__ import annotations

import cv2
import numpy as np

from root_to_tip_mesh import COLORS, FINGER_COLORS, label_panel, make_sheet


PALETTE_BGR = tuple(tuple(reversed(color)) for color in COLORS + FINGER_COLORS)


def align_depth_to_color(depth_m, camera, color_shape, *, splat: str = "nearest"):
    """Project raw depth samples into color pixels with a frontmost z-buffer.

    Values are color-optical Z in meters; invalid pixels remain +inf. The four-pixel
    variant is for visualization coverage only, not a new depth measurement.
    """
    if splat not in ("nearest", "four"):
        raise ValueError(f"Unknown splat mode: {splat}")
    h_color, w_color = color_shape[:2]
    h_depth, w_depth = depth_m.shape
    k_depth, k_color = camera.depth_intrinsics, camera.color_intrinsics
    index = np.flatnonzero(np.isfinite(depth_m) & (depth_m > 0))
    v, u = np.divmod(index, w_depth)
    z = depth_m.ravel()[index].astype(np.float64)
    points_depth = np.column_stack((
        (u - k_depth.cx) * z / k_depth.fx,
        (v - k_depth.cy) * z / k_depth.fy,
        z,
    ))
    T = camera.T_color_depth
    points_color = points_depth @ T[:3, :3].T + T[:3, 3]
    keep = np.isfinite(points_color).all(axis=1) & (points_color[:, 2] > 0)
    points_color = points_color[keep]
    zz = points_color[:, 2].astype(np.float32)
    uu = k_color.fx * points_color[:, 0] / zz + k_color.cx
    vv = k_color.fy * points_color[:, 1] / zz + k_color.cy
    good = np.isfinite(uu) & np.isfinite(vv)
    uu, vv, zz = uu[good], vv[good], zz[good]
    zbuffer = np.full(h_color * w_color, np.inf, np.float32)
    if splat == "nearest":
        candidates = ((np.rint(uu).astype(np.int64), np.rint(vv).astype(np.int64)),)
    else:
        left, top = np.floor(uu).astype(np.int64), np.floor(vv).astype(np.int64)
        candidates = tuple((left + du, top + dv) for dv in (0, 1) for du in (0, 1))
    for x, y in candidates:
        inside = (x >= 0) & (x < w_color) & (y >= 0) & (y < h_color)
        np.minimum.at(zbuffer, y[inside] * w_color + x[inside], zz[inside])
    return zbuffer.reshape(h_color, w_color)


def depth_range(depth_m):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return 0.2, 1.2
    lo, hi = np.percentile(depth_m[valid], (2, 98))
    return float(lo), float(max(hi, lo + 1e-6))


def colorize_depth(depth_m, limits):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    out = np.zeros((*depth_m.shape, 3), np.uint8)
    lo, hi = limits
    scaled = np.zeros(depth_m.shape, np.uint8)
    scaled[valid] = np.clip(255 * (1 - (depth_m[valid] - lo) / (hi - lo)), 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    out[valid] = colors[valid]
    return out


def depth_discontinuities(depth_m, threshold_m=0.025):
    """Depth jumps between measured neighbors; holes are excluded."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    edge = np.zeros(depth_m.shape, bool)
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    dx = np.zeros_like(depth_m)
    dy = np.zeros_like(depth_m)
    np.subtract(depth_m[:, 1:], depth_m[:, :-1], out=dx[:, 1:], where=horizontal)
    np.subtract(depth_m[1:, :], depth_m[:-1, :], out=dy[1:, :], where=vertical)
    edge[:, 1:] |= horizontal & (np.abs(dx[:, 1:]) > threshold_m)
    edge[1:, :] |= vertical & (np.abs(dy[1:, :]) > threshold_m)
    return edge


def draw_cad_contours(background, group_ids, group_defs):
    out = background.copy()
    for index, _ in enumerate(group_defs):
        mask = (group_ids == index).astype(np.uint8)
        if np.any(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, PALETTE_BGR[index], 2, cv2.LINE_AA)
    return out


def make_color_cad_comparison(rgb, baseline_groups, corrected_groups):
    """Overlay both visible CAD silhouettes on exactly the same RGB pixels."""
    if rgb.shape[:2] != baseline_groups.shape or baseline_groups.shape != corrected_groups.shape:
        raise ValueError("RGB and both CAD masks must share the color pixel grid")
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for groups, color in ((baseline_groups, (0, 0, 255)), (corrected_groups, (0, 255, 0))):
        mask = (groups >= 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return label_panel(out, "same RGB pixels / baseline red / corrected green")


def make_link_sheet(background, group_ids, group_defs, *, title="raw image / native pixel grid"):
    panels = [label_panel(background, title)]
    for index, (name, _) in enumerate(group_defs):
        single = np.where(group_ids == index, index, -1)
        panels.append(label_panel(draw_cad_contours(background, single, group_defs),
                                  f"{name} ONLY / visible CAD"))
    cumulative = background.copy()
    for index, (name, _) in enumerate(group_defs):
        single = np.where(group_ids == index, index, -1)
        cumulative = draw_cad_contours(cumulative, single, group_defs)
        panels.append(label_panel(cumulative, f"cumulative through {name}"))
    return make_sheet(panels, width=380, cols=4)


def make_joint_sheet(rgb, raw_depth_m, aligned_nearest, aligned_four,
                     depth_groups, color_groups, group_defs):
    """Make a six-panel audit; RGB-grid panels never resize raw depth into RGB."""
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    limits = depth_range(raw_depth_m)
    rgb_cad = draw_cad_contours(rgb_bgr, color_groups, group_defs)
    raw_cad = draw_cad_contours(colorize_depth(raw_depth_m, limits), depth_groups, group_defs)

    valid_four = np.isfinite(aligned_four)
    pseudo = colorize_depth(aligned_four, limits)
    registered = rgb_bgr.copy()
    registered[valid_four] = np.clip(
        0.50 * registered[valid_four] + 0.50 * pseudo[valid_four], 0, 255
    ).astype(np.uint8)
    registered = draw_cad_contours(registered, color_groups, group_defs)

    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(gray, 80, 160) != 0
    depth_edges = depth_discontinuities(aligned_four)
    edges = (rgb_bgr.astype(np.float32) * 0.40).astype(np.uint8)
    edges[rgb_edges] = (40, 230, 40)
    edges[depth_edges] = (40, 40, 255)
    edges[rgb_edges & depth_edges] = (0, 255, 255)
    edges = draw_cad_contours(edges, color_groups, group_defs)

    coverage = np.zeros_like(rgb_bgr)
    nearest_mask = np.isfinite(aligned_nearest)
    coverage[valid_four & ~nearest_mask] = (0, 180, 255)
    coverage[nearest_mask] = (70, 220, 70)
    panels = (
        label_panel(rgb_cad, "RGB + CAD / color optical"),
        label_panel(registered, "RGB + registered raw depth + CAD / color optical"),
        label_panel(edges, "RGB edges green / depth jumps red / CAD multicolor"),
        label_panel(raw_cad, "raw depth + CAD / depth optical"),
        label_panel(colorize_depth(aligned_nearest, limits), "registered nearest-only / color optical"),
        label_panel(coverage, "green measured pixel / amber four-splat only"),
    )
    return np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
