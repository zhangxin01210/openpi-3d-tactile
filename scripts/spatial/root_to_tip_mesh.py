"""Expanded-canvas URDF visual-mesh overlays for front camera audits."""

from __future__ import annotations

import math
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


CHAIN_GROUPS = (
    ("C0_base", ("base_link_inertia",)),
    ("C1_shoulder", ("shoulder_link",)),
    ("C2_upper_arm", ("upper_arm_link",)),
    ("C3_forearm", ("forearm_link",)),
    ("C4_wrist1", ("wrist_1_link",)),
    ("C5_wrist2", ("wrist_2_link",)),
    ("C6_wrist3", ("wrist_3_link",)),
    ("C7_adapter", ("Flange_base_link",)),
    ("C8_palm", ("right_hand_link", "right_hand_back_link", "right_hand_ee_link")),
)
COLORS = (
    (255, 70, 70), (255, 170, 40), (240, 230, 50),
    (70, 220, 80), (50, 220, 220), (50, 130, 255),
    (110, 70, 255), (220, 70, 220), (255, 255, 255),
)
FINGER_PREFIXES = (
    ("thumb", "right_hand_thumb_"),
    ("index", "right_hand_index_"),
    ("middle", "right_hand_mid"),
    ("ring", "right_hand_ring"),
    ("pinky", "right_hand_pinky"),
)
FINGER_COLORS = (
    (230, 60, 60), (55, 105, 230), (50, 180, 85),
    (175, 80, 210), (235, 155, 45),
)


def load_marker_camera_report(report_path: Path, key: str) -> tuple[np.ndarray, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if key not in report:
        raise KeyError(f"{report_path} has no {key!r}")
    T = np.asarray(report[key], dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{key} must be a finite homogeneous 4x4 transform")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(T[:3, :3]), 1, atol=1e-3
    ):
        raise ValueError(f"{key} rotation must be in SO(3)")
    return T, report


def resolve_mesh_urdf(repo_root: Path, supplied: Path | None) -> Path:
    candidates = [supplied] if supplied is not None else [
        repo_root / "configs/ur7e_xhand/root_to_tip_assets/pointcloud_delivery/configs/ur7e_xhand_verified.urdf",
        repo_root / "assets/root_to_tip/pointcloud_delivery/configs/ur7e_xhand_verified.urdf",
        repo_root / "3D_tactile/pointcloud_delivery/configs/ur7e_xhand_verified.urdf",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            selected = candidate.expanduser().resolve()
            canonical = repo_root / "configs/ur7e_xhand/ur7e_xhand_verified.urdf"
            if canonical.is_file() and selected.read_bytes() != canonical.read_bytes():
                raise ValueError(
                    f"Mesh URDF {selected} differs from the FK URDF {canonical}; "
                    "use the matching verified URDF and meshes."
                )
            return selected
    raise FileNotFoundError(
        "Root-to-tip requires the verified URDF and its visual meshes. "
        f"Tried: {[str(p) for p in candidates]}. "
        "Ensure configs/ur7e_xhand/root_to_tip_assets is present, or pass --mesh-urdf."
    )


def _visual_mesh_path(filename: str, urdf: Path) -> Path:
    prefix = "package://ur_description/"
    if filename.startswith(prefix):
        return (
            urdf.parent.parent / "diagnostics/ur_description_source" / filename.removeprefix(prefix)
        ).resolve()
    if filename.startswith("package://") or Path(filename).is_absolute():
        raise ValueError(f"Visual mesh must have a portable relative path: {filename}")
    return (urdf.parent / filename).resolve()


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if origin is not None:
        T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        T[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return T


def load_visual_meshes(urdf: Path, *, include_fingers: bool = False) -> dict[str, trimesh.Trimesh]:
    """Load only links drawn in the audit; preserve DAE scene transforms."""
    root = ET.parse(urdf).getroot()
    required = {link for _, links in CHAIN_GROUPS for link in links}
    if include_fingers:
        required.update(
            link.get("name") for link in root.findall("link")
            if any(link.get("name", "").startswith(prefix) for _, prefix in FINGER_PREFIXES)
            and link.find("visual/geometry/mesh") is not None
        )
    meshes: dict[str, trimesh.Trimesh] = {}
    missing: list[str] = []
    for link in root.findall("link"):
        name = link.get("name")
        if name not in required:
            continue
        parts = []
        for visual in link.findall("visual"):
            elem = visual.find("geometry/mesh")
            if elem is None:
                continue
            filename = elem.get("filename")
            if filename is None:
                raise ValueError(f"{name} visual mesh has no filename")
            path = _visual_mesh_path(filename, urdf)
            if not path.is_file():
                missing.append(str(path))
                continue
            loaded = trimesh.load(path, force="scene", process=False)
            if isinstance(loaded, trimesh.Scene):
                try:
                    mesh = loaded.to_geometry()
                    if isinstance(mesh, trimesh.Scene):
                        mesh = loaded.dump(concatenate=True)
                except Exception:
                    mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Not a triangular visual mesh: {path}")
            mesh = mesh.copy()
            scale = np.fromstring(elem.get("scale", "1 1 1"), sep=" ")
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            mesh.vertices *= scale
            mesh.apply_transform(_origin_matrix(visual.find("origin")))
            parts.append(mesh)
        if parts:
            meshes[name] = trimesh.util.concatenate(parts)
    if missing or required - meshes.keys():
        raise FileNotFoundError(
            "Missing visual meshes for root-to-tip:\n"
            + "\n".join(missing + [f"No visual mesh for link {x}" for x in sorted(required - meshes.keys())])
        )
    return meshes


def _projected_mesh(mesh: trimesh.Trimesh, T_base_link: np.ndarray, camera) -> tuple[np.ndarray, np.ndarray]:
    points_base = mesh.vertices @ T_base_link[:3, :3].T + T_base_link[:3, 3]
    T = camera.T_base_color
    points_color = (points_base - T[:3, 3]) @ T[:3, :3]
    z = points_color[:, 2]
    uv = np.full((len(z), 2), np.nan)
    good = np.isfinite(points_color).all(axis=1) & (z > 1e-7)
    K = camera.color_intrinsics
    uv[good, 0] = K.fx * points_color[good, 0] / z[good] + K.cx
    uv[good, 1] = K.fy * points_color[good, 1] / z[good] + K.cy
    return uv, z


def compute_expanded_bounds(rgb_shape, meshes, transforms, camera, pad: int, max_extra: int):
    h, w = rgb_shape[:2]
    xs, ys = [0.0, float(w - 1)], [0.0, float(h - 1)]
    for _, links in CHAIN_GROUPS:
        for link in links:
            if link not in transforms:
                raise KeyError(f"FK missing {link}")
            uv, z = _projected_mesh(meshes[link], transforms[link], camera)
            valid = np.isfinite(uv).all(axis=1) & (z > 1e-7)
            u, v = uv[valid, 0], uv[valid, 1]
            reasonable = (np.abs(u) < w + max_extra * 3) & (np.abs(v) < h + max_extra * 3)
            xs.extend(u[reasonable].tolist())
            ys.extend(v[reasonable].tolist())
    xmin = max(math.floor(min(xs) - pad), -max_extra)
    ymin = max(math.floor(min(ys) - pad), -max_extra)
    xmax = min(math.ceil(max(xs) + pad), w - 1 + max_extra)
    ymax = min(math.ceil(max(ys) + pad), h - 1 + max_extra)
    return min(xmin, -pad), min(ymin, -pad), max(xmax, w - 1 + pad), max(ymax, h - 1 + pad)


def make_base_canvas(rgb: np.ndarray, bounds) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    h, w = rgb.shape[:2]
    canvas = np.zeros((ymax - ymin + 1, xmax - xmin + 1, 3), np.uint8)
    canvas[-ymin:-ymin+h, -xmin:-xmin+w] = rgb
    cv2.rectangle(canvas, (-xmin, -ymin), (-xmin+w-1, -ymin+h-1), (170, 170, 170), 2, cv2.LINE_AA)
    return canvas


def rasterize_mesh(canvas, mesh, T_base_link, camera, bounds, color, alpha):
    uv, z = _projected_mesh(mesh, T_base_link, camera)
    faces = np.asarray(mesh.faces, int)
    faces = faces[np.all(z[faces] > 1e-7, axis=1)]
    tri = uv[faces]
    tri = tri[np.isfinite(tri).all(axis=(1, 2))] + np.asarray([-bounds[0], -bounds[1]])
    h, w = canvas.shape[:2]
    polys = []
    for t in tri:
        p = np.rint(t).astype(np.int32)
        if np.max(p[:, 0]) < -20 or np.min(p[:, 0]) > w + 20:
            continue
        if np.max(p[:, 1]) < -20 or np.min(p[:, 1]) > h + 20:
            continue
        polys.append(p)
    mask = np.zeros((h, w), np.uint8)
    if polys:
        cv2.fillPoly(mask, polys, 255)
    layer = np.zeros_like(canvas)
    layer[mask > 0] = color
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    out = np.clip(canvas * (1.0 - a) + layer * a, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
    return out


def render_group(canvas, links, meshes, transforms, camera, bounds, color, alpha):
    out = canvas.copy()
    for link in links:
        out = rasterize_mesh(out, meshes[link], transforms[link], camera, bounds, color, alpha)
    return out


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    out = np.zeros((image.shape[0] + 30, image.shape[1], 3), np.uint8)
    out[30:] = image
    cv2.putText(out, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_sheet(panels, width: int = 460, cols: int = 3) -> np.ndarray:
    resized = []
    for panel in panels:
        scale = width / panel.shape[1]
        resized.append(cv2.resize(panel, (width, max(1, round(panel.shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST))
    max_h = max(panel.shape[0] for panel in resized)
    fixed = [np.pad(panel, ((0, max_h - panel.shape[0]), (0, 0), (0, 0))) for panel in resized]
    while len(fixed) % cols:
        fixed.append(np.zeros_like(fixed[0]))
    return np.concatenate([np.concatenate(fixed[i:i+cols], axis=1)
                           for i in range(0, len(fixed), cols)], axis=0)
