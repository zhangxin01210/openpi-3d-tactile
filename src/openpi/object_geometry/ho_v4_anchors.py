"""Fixed HO V4 anchor assets and provenance verification (CPU only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ALGORITHM = "pca_box_corners_distinct_fps_v1"
COUNT = 16
ROLES = ("manipulated", "target")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _exporter():
    path = _REPO_ROOT / "scripts/preprocess/export_object_geometry_rgbtrack.py"
    spec = importlib.util.spec_from_file_location("_openpi_rgbtrack_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"HO V4 required asset or geometry source missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _points_hash(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype="<f4").tobytes()).hexdigest()


def choose_indices(points):
    """PCA box corners, greedy unique nearest points, then deterministic FPS."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.shape != (512, 3) or not np.isfinite(xyz).all():
        raise ValueError("expected 512 finite 3D surface points")
    mean = xyz.mean(axis=0)
    values, vectors = np.linalg.eigh((xyz - mean).T @ (xyz - mean))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    # Eigenvectors in a degenerate eigenspace are not uniquely defined.
    # Project fixed object-frame axes into each such subspace and use
    # Gram-Schmidt, so symmetric shapes do not depend on LAPACK's basis.
    tolerance = max(abs(values[0]) * 1e-10, 1e-18)
    start = 0
    while start < 3:
        end = start + 1
        while end < 3 and abs(values[end] - values[start]) <= tolerance:
            end += 1
        if end - start > 1:
            projection = vectors[:, start:end] @ vectors[:, start:end].T
            basis = []
            for axis in np.eye(3):
                candidate = projection @ axis
                for previous in basis:
                    candidate -= previous * np.dot(previous, candidate)
                length = np.linalg.norm(candidate)
                if length > 1e-8:
                    basis.append(candidate / length)
                if len(basis) == end - start:
                    break
            if len(basis) != end - start:
                raise ValueError("cannot resolve degenerate PCA eigenspace")
            vectors[:, start:end] = np.stack(basis, axis=1)
        else:
            axis = vectors[:, start]
            if axis[np.argmax(np.abs(axis))] < 0:
                vectors[:, start] *= -1
        start = end
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] *= -1
    projected = (xyz - mean) @ vectors
    low, high = projected.min(axis=0), projected.max(axis=0)
    selected = []
    for corner in range(8):
        corner_p = np.where([(corner >> axis) & 1 for axis in range(3)], high, low)
        d2 = np.sum((projected - corner_p) ** 2, axis=1)
        d2[selected] = np.inf
        selected.append(int(np.argmin(d2)))
    corner_indices = selected.copy()
    nearest = np.min(np.sum((xyz[:, None, :] - xyz[selected][None, :, :]) ** 2, axis=-1), axis=1)
    while len(selected) < COUNT:
        nearest[selected] = -1
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((xyz - xyz[index]) ** 2, axis=1))
    return selected, corner_indices, selected[8:]


def load_sources(repo_root, dataset, *, sample_surface=True):
    """Verify release, approved meshes and the existing sidecar are the same export."""
    repo = Path(repo_root).resolve()
    prep = repo / "output/preprocess" / dataset
    release_path = prep / "rgbtrack/release_v1.json"
    approved_path = prep / "geometry/approved/manifest.json"
    sidecar_path = repo / "data" / dataset / "object_geometry/v1/manifest.json"
    release, approved, sidecar = map(_read, (release_path, approved_path, sidecar_path))
    objects = release.get("objects")
    policy = release.get("policy", {})
    roles = {"manipulated": policy.get("dynamic_object"), "target": policy.get("static_object")}
    if (release.get("dataset") != dataset or not isinstance(objects, list) or len(objects) != 2
            or len(set(objects)) != 2 or set(objects) != set(roles.values())):
        raise ValueError("RGBTrack release dataset, object ordering, or roles are invalid")
    if approved.get("status") != "human_accepted" or set(approved.get("objects", [])) != set(objects):
        raise ValueError("geometry is not approved or object names mismatch")
    exporter = _exporter()
    if (sidecar.get("status") != "complete" or sidecar.get("dataset") != dataset
            or sidecar.get("producer") != "scripts/preprocess/export_object_geometry_rgbtrack.py"
            or sidecar.get("role_to_object") != roles
            or sidecar.get("source_release_sha256") != exporter.sha256(release_path)
            or sidecar.get("approved_geometry_sha256") != exporter.sha256(approved_path)
            or sidecar.get("surface_sampling_seed") != exporter.SURFACE_SAMPLING_SEED
            or sidecar.get("object_geometry_contract", {}).get("point_count") != 512):
        raise ValueError("object_geometry/v1 source, roles or 512-point sampling provenance mismatch")
    result = {}
    for role in ROLES:
        name = roles[role]
        mesh = prep / "geometry/approved" / f"{name}.glb"
        mesh_hash = exporter.sha256(mesh)
        entry = approved.get("assets", {}).get(name, {})
        if (entry.get("sha256") != mesh_hash or float(entry.get("scale_m_per_unit", -1)) != 1.0
                or sidecar.get("approved_meshes_sha256", {}).get(name) != mesh_hash):
            raise ValueError(f"{role}: approved mesh hash or units differ from geometry sidecar")
        seed = exporter.SURFACE_SAMPLING_SEED + objects.index(name)
        points = exporter.load_surface_points(mesh, seed) if sample_surface else None
        result[role] = {"dataset": dataset, "object_name": name, "object_role": role,
                        "mesh_sha256": mesh_hash, "surface_sampling_seed": seed,
                        "surface_points_sha256": _points_hash(points) if points is not None else None,
                        "source_release_sha256": sidecar["source_release_sha256"],
                        "approved_geometry_sha256": sidecar["approved_geometry_sha256"],
                        "sampler_code_sha256": exporter.sha256(Path(exporter.__file__)), "points": points}
    return result


def make_asset(source):
    points = source["points"]
    indices, corners, fps = choose_indices(points)
    if choose_indices(points) != (indices, corners, fps):
        raise ValueError("anchor selection is not deterministic")
    return {**{k: v for k, v in source.items() if k != "points"}, "algorithm": ALGORITHM,
            "generator_version": "ho_v4_anchors_v1", "generator_code_sha256": _exporter().sha256(Path(__file__)),
            "indices": indices,
            "xyz_object_m": points[indices].tolist(), "surface_xyz_object_m": points.tolist(),
            "corner_indices": corners, "fps_indices": fps}


def load_assets(repo_root, dataset, asset_root="assets/ho_v4_anchors"):
    sources = load_sources(repo_root, dataset, sample_surface=False)
    root = Path(asset_root)
    if not root.is_absolute():
        root = Path(repo_root) / root
    output = {}
    for role in ROLES:
        path = root / dataset / f"{role}.json"
        asset = _read(path)
        source = sources[role]
        for key in ("dataset", "object_name", "object_role", "mesh_sha256", "surface_sampling_seed",
                    "source_release_sha256", "approved_geometry_sha256", "sampler_code_sha256"):
            if asset.get(key) != source[key]:
                raise ValueError(f"{path}: {key} does not match current approved geometry/export")
        indices = asset.get("indices")
        if (asset.get("algorithm") != ALGORITHM or asset.get("generator_version") != "ho_v4_anchors_v1"
                or asset.get("generator_code_sha256") != _exporter().sha256(Path(__file__))
                or not isinstance(indices, list) or len(indices) != COUNT
                or any(type(i) is not int or not 0 <= i < 512 for i in indices) or len(set(indices)) != COUNT
                or asset.get("corner_indices") != indices[:8] or asset.get("fps_indices") != indices[8:]):
            raise ValueError(f"{path}: invalid fixed anchor index contract")
        points = np.asarray(asset.get("surface_xyz_object_m"), dtype=np.float32)
        if (points.shape != (512, 3) or not np.isfinite(points).all()
                or _points_hash(points) != asset.get("surface_points_sha256")):
            raise ValueError(f"{path}: sampled surface bytes do not match the recorded SHA256")
        if not np.array_equal(np.asarray(asset.get("xyz_object_m"), dtype=np.float32), points[indices]):
            raise ValueError(f"{path}: anchor coordinates do not match the sampled surface indices")
        output[role] = {"indices": np.asarray(indices, dtype=np.int32),
                        "xyz_object_m": points[indices], "surface_points_sha256": asset["surface_points_sha256"],
                        "surface_distances_m": np.linalg.norm(points[:, None, :]
                            - points[indices][None, :, :], axis=-1)}
    return output


def validate_frame_geometry(geometry, assets):
    """Check ordered point identities up to a rigid pose; fully invalid objects stay masked."""
    for role in ROLES:
        asset = assets[role]
        xyz = np.asarray(geometry[f"{role}_xyz_m"], dtype=np.float32)
        mask = np.asarray(geometry[f"{role}_mask"], dtype=bool)
        if xyz.shape != (512, 3) or mask.shape != (512,):
            raise ValueError(f"HO V4 {role}: expected 512 ordered object points and mask")
        if not mask.any():
            continue
        if not np.isfinite(xyz[mask]).all():
            raise ValueError(f"HO V4 {role}: valid object points must be finite")
        indices = asset["indices"]
        anchor_mask = mask[indices]
        if not anchor_mask.any():
            raise ValueError(f"HO V4 {role}: cannot verify surface identity without any valid fixed anchor")
        current = np.linalg.norm(xyz[mask, None, :] - xyz[indices[anchor_mask]][None, :, :], axis=-1)
        expected = asset["surface_distances_m"][mask][:, anchor_mask]
        if not np.allclose(current, expected, atol=5e-5, rtol=1e-4):
            raise ValueError(f"HO V4 {role}: ordered surface points do not match registered anchor identities")


def validate_online_geometry(geometry, assets):
    """Require source fingerprints and ordered geometry from the online producer."""
    provenance = geometry.get("surface_points_sha256")
    if not isinstance(provenance, dict):
        raise ValueError("HO V4 online object_geometry requires surface_points_sha256 by role")
    for role in ROLES:
        if provenance.get(role) != assets[role]["surface_points_sha256"]:
            raise ValueError(f"HO V4 online {role} surface sampling provenance mismatch")
    validate_frame_geometry(geometry, assets)
