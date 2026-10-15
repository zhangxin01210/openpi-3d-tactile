"""Deterministic contact-centred tactile and local geometry features.

The routines in this module are deliberately NumPy-only.  The offline dataset
builder, the LeRobot adapter, and the old Isaac Gym evaluation runtime can
therefore use exactly the same geometry convention.

This is a representation of force on the *finger pads*.  It does not claim to
observe the unmeasured USB--socket contact point.
"""

from __future__ import annotations

import collections

import numpy as np


FINGERS = 2
TAXELS = 140
GRID_SHAPE = (10, 14)
ANCHORS_PER_FINGER = 2
PATCH_POINTS_PER_ANCHOR = 16
RELATION_CANDIDATES = 48

# ContactWorld TacFF values are simulator units.  This small floor only
# distinguishes a completely zero pad from numerical noise; relative strength
# determines the connected contact regions.
ACTIVE_FLOOR = np.float32(5e-5)
RELATION_RADIUS_M = np.float32(0.16)
BACKSIDE_TOLERANCE_M = np.float32(0.004)
SELF_SURFACE_MARGIN_M = np.float32(0.003)


def taxel_uv() -> np.ndarray:
    """Fixed, normalized (u, v) coordinates for the 10 x 14 pad grid."""
    row, col = np.meshgrid(
        np.linspace(-1.0, 1.0, GRID_SHAPE[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, GRID_SHAPE[1], dtype=np.float32),
        indexing="ij",
    )
    return np.stack((col, row), axis=-1).reshape(TAXELS, 2)


_UV = taxel_uv()


def _components(active: np.ndarray) -> list[np.ndarray]:
    """Four-connected components in a 10 x 14 boolean taxel map."""
    active = np.asarray(active, dtype=bool).reshape(GRID_SHAPE)
    seen = np.zeros(GRID_SHAPE, dtype=bool)
    output = []
    for row in range(GRID_SHAPE[0]):
        for col in range(GRID_SHAPE[1]):
            if not active[row, col] or seen[row, col]:
                continue
            queue = collections.deque([(row, col)])
            seen[row, col] = True
            component = []
            while queue:
                rr, cc = queue.popleft()
                component.append(rr * GRID_SHAPE[1] + cc)
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (0 <= nr < GRID_SHAPE[0] and 0 <= nc < GRID_SHAPE[1]
                            and active[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            output.append(np.asarray(component, dtype=np.int32))
    return output


def _summary(force: np.ndarray) -> np.ndarray:
    """Thirteen signed/absolute force summaries, one per finger."""
    norm = np.linalg.norm(force, axis=-1)
    return np.concatenate((
        force.mean(axis=0),
        np.abs(force).mean(axis=0),
        np.sqrt(np.mean(force ** 2, axis=0) + 1e-12),
        np.abs(force).max(axis=0),
        np.asarray([(norm > ACTIVE_FLOOR).mean()], dtype=np.float32),
    )).astype(np.float32)


def _empty_finger(xyz: np.ndarray) -> dict[str, np.ndarray]:
    pad = xyz.mean(axis=0).astype(np.float32)
    return {
        "patch_xyz_local": np.zeros((ANCHORS_PER_FINGER, PATCH_POINTS_PER_ANCHOR, 2), np.float32),
        "patch_force": np.zeros((ANCHORS_PER_FINGER, PATCH_POINTS_PER_ANCHOR, 3), np.float32),
        "patch_mask": np.zeros((ANCHORS_PER_FINGER, PATCH_POINTS_PER_ANCHOR), bool),
        "anchor_xyz_base": np.repeat(pad[None], ANCHORS_PER_FINGER, axis=0),
        "anchor_uv": np.zeros((ANCHORS_PER_FINGER, 2), np.float32),
        "anchor_stats": np.zeros((ANCHORS_PER_FINGER, 10), np.float32),
        "anchor_mask": np.zeros(ANCHORS_PER_FINGER, bool),
        "finger_summary": np.zeros(13, np.float32),
        "finger_center_base": pad,
        "finger_valid": np.bool_(False),
    }


def build_centered_finger(xyz: np.ndarray, force: np.ndarray) -> dict[str, np.ndarray]:
    """Build two contact-centred 16-taxel patches for one pad.

    Components are selected from a relative activity map.  Each selected patch
    keeps the 16 grid positions nearest that component's centre, including
    nearby zero-force taxels so that shape is retained rather than merely the
    strongest individual taxels.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32)
    if xyz.shape != (TAXELS, 3) or force.shape != (TAXELS, 3):
        raise ValueError((xyz.shape, force.shape))
    norm = np.linalg.norm(force, axis=-1)
    total = float(norm.sum())
    if total < float(ACTIVE_FLOOR):
        return _empty_finger(xyz)
    threshold = max(float(ACTIVE_FLOOR), 0.25 * float(norm.max()))
    components = _components(norm >= threshold)
    if not components:
        components = [np.asarray([int(np.argmax(norm))], dtype=np.int32)]
    components.sort(key=lambda indices: float(norm[indices].sum()), reverse=True)

    result = _empty_finger(xyz)
    result["finger_summary"] = _summary(force)
    all_weights = norm / max(total, 1e-12)
    result["finger_center_base"] = (xyz * all_weights[:, None]).sum(axis=0).astype(np.float32)
    result["finger_valid"] = np.bool_(True)

    for anchor, indices in enumerate(components[:ANCHORS_PER_FINGER]):
        weights = norm[indices]
        weights = weights / max(float(weights.sum()), 1e-12)
        center_xyz = (xyz[indices] * weights[:, None]).sum(axis=0)
        center_uv = (_UV[indices] * weights[:, None]).sum(axis=0)
        nearest = np.argsort(np.sum((_UV - center_uv[None]) ** 2, axis=-1), kind="stable")[:PATCH_POINTS_PER_ANCHOR]
        rel_uv = _UV[nearest] - center_uv[None]
        radius = np.sqrt(np.sum((_UV[indices] - center_uv[None]) ** 2, axis=-1))
        stats = np.concatenate((
            force[indices].sum(axis=0),
            np.asarray([
                norm[indices].sum(),
                len(indices) / float(TAXELS),
                np.sum(weights * radius),
                norm[indices].max(),
            ], dtype=np.float32),
            center_uv.astype(np.float32),
            np.asarray([1.0], dtype=np.float32),
        )).astype(np.float32)
        result["patch_xyz_local"][anchor] = rel_uv
        result["patch_force"][anchor] = force[nearest]
        result["patch_mask"][anchor] = True
        result["anchor_xyz_base"][anchor] = center_xyz
        result["anchor_uv"][anchor] = center_uv
        result["anchor_stats"][anchor] = stats
        result["anchor_mask"][anchor] = True
    return result


def build_relation_candidates(cloud_xyz: np.ndarray, pad_xyz: np.ndarray,
                              centered: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Return local non-backside scene candidates for both tactile pads.

    The inward half-space is a conservative self filter: points behind a pad
    cannot belong to the held object or interaction region and are excluded.
    This is not a full robot segmentation, which the recorded V3 sidecar does
    not contain; the audit marks this distinction explicitly.
    """
    cloud_xyz = np.asarray(cloud_xyz, dtype=np.float32)
    pad_xyz = np.asarray(pad_xyz, dtype=np.float32)
    if cloud_xyz.ndim != 2 or cloud_xyz.shape[1] != 3 or pad_xyz.shape != (FINGERS, TAXELS, 3):
        raise ValueError((cloud_xyz.shape, pad_xyz.shape))
    query_xyz = np.stack([item["finger_center_base"] for item in centered]).astype(np.float32)
    candidates = np.zeros((FINGERS, RELATION_CANDIDATES, 3), np.float32)
    mask = np.zeros((FINGERS, RELATION_CANDIDATES), bool)
    candidate_distance = np.zeros((FINGERS, RELATION_CANDIDATES), np.float32)
    pad_centers = pad_xyz.mean(axis=1)
    for finger in range(FINGERS):
        inward = pad_centers[1 - finger] - pad_centers[finger]
        inward = inward / max(float(np.linalg.norm(inward)), 1e-8)
        delta = cloud_xyz - query_xyz[finger]
        distance = np.linalg.norm(delta, axis=-1)
        facing = delta @ inward
        keep = ((distance >= SELF_SURFACE_MARGIN_M) & (distance <= RELATION_RADIUS_M)
                & (facing >= -BACKSIDE_TOLERANCE_M))
        kept = np.flatnonzero(keep)
        if not len(kept):
            continue
        order = kept[np.argsort(distance[kept], kind="stable")[:RELATION_CANDIDATES]]
        candidates[finger, :len(order)] = cloud_xyz[order]
        mask[finger, :len(order)] = True
        candidate_distance[finger, :len(order)] = distance[order]
    return {
        "query_xyz_base": query_xyz,
        "query_features": np.stack([item["finger_summary"] for item in centered]).astype(np.float32),
        "query_valid": np.asarray([item["finger_valid"] for item in centered], dtype=bool),
        "candidate_xyz_base": candidates,
        "candidate_mask": mask,
        "candidate_distance": candidate_distance,
    }


def build_frame(cloud_xyz: np.ndarray, tactile_xyz: np.ndarray, tactile_force: np.ndarray) -> dict[str, np.ndarray]:
    """Build every approved contact representation for one pre-action frame."""
    tactile_xyz = np.asarray(tactile_xyz, dtype=np.float32)
    tactile_force = np.asarray(tactile_force, dtype=np.float32)
    if tactile_xyz.shape != (FINGERS, TAXELS, 3) or tactile_force.shape != (FINGERS, TAXELS, 3):
        raise ValueError((tactile_xyz.shape, tactile_force.shape))
    centered = [build_centered_finger(tactile_xyz[finger], tactile_force[finger]) for finger in range(FINGERS)]
    relation = build_relation_candidates(cloud_xyz, tactile_xyz, centered)
    output = {
        key: np.stack([item[key] for item in centered]).astype(np.float32 if key not in {"patch_mask", "anchor_mask", "finger_valid"} else bool)
        for key in ("patch_xyz_local", "patch_force", "patch_mask", "anchor_xyz_base", "anchor_uv",
                    "anchor_stats", "anchor_mask", "finger_summary", "finger_center_base", "finger_valid")
    }
    output.update(relation)
    return output


def build_sequence(cloud_xyz: np.ndarray, tactile_xyz: np.ndarray, tactile_force: np.ndarray) -> dict[str, np.ndarray]:
    """Vector-free deterministic sequence builder used by the sidecar exporter."""
    cloud_xyz = np.asarray(cloud_xyz, dtype=np.float32)
    tactile_xyz = np.asarray(tactile_xyz, dtype=np.float32)
    tactile_force = np.asarray(tactile_force, dtype=np.float32)
    if (cloud_xyz.ndim != 3 or cloud_xyz.shape[1:] != (4096, 3)
            or tactile_xyz.shape[:2] != (len(cloud_xyz), FINGERS)
            or tactile_force.shape != tactile_xyz.shape):
        raise ValueError((cloud_xyz.shape, tactile_xyz.shape, tactile_force.shape))
    rows = [build_frame(cloud_xyz[index], tactile_xyz[index], tactile_force[index])
            for index in range(len(cloud_xyz))]
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}
