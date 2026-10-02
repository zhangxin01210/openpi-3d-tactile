"""Convert ContactWorld right TacFF channels to physical base-frame vectors.

The source simulator projects its world force onto sensor-local axes in this
order: [normal=-Fy, shear_x=-Fx, shear_y=Fz]. Taxel world poses are available
online in TactileFieldSensor after the force field observation is computed.
The released Zarr does not include these poses, so this conversion is only
valid during same-frame replay/capture.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def tactile_field_to_base(force_grid: np.ndarray, taxel_pos_world: np.ndarray,
                          taxel_quat_world: np.ndarray, base_pos_world: np.ndarray,
                          base_quat_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(force_grid, dtype=np.float64)
    xyz_world = np.asarray(taxel_pos_world, dtype=np.float64)
    quat_world = np.asarray(taxel_quat_world, dtype=np.float64)
    base_pos = np.asarray(base_pos_world, dtype=np.float64)
    base_quat = np.asarray(base_quat_world, dtype=np.float64)
    if (field.shape != (10, 14, 3) or xyz_world.shape != (140, 3) or
            quat_world.shape != (140, 4) or base_pos.shape != (3,) or
            base_quat.shape != (4,)):
        raise ValueError("Unexpected ContactWorld TacFF/taxel/base geometry")
    if not all(np.isfinite(x).all() for x in (field, xyz_world, quat_world, base_pos, base_quat)):
        raise ValueError("Nonfinite tactile geometry")
    normal, shear_x, shear_y = field.reshape(140, 3).T
    force_local_xyz = np.stack((-shear_x, -normal, shear_y), axis=-1)
    world_rotations = Rotation.from_quat(quat_world)
    force_world = world_rotations.apply(force_local_xyz)
    base_inverse = Rotation.from_quat(base_quat).inv()
    xyz_base = base_inverse.apply(xyz_world - base_pos)
    force_base = base_inverse.apply(force_world)
    # Round trip against the source channel projection, checking signs/order.
    local_check = world_rotations.inv().apply(Rotation.from_quat(base_quat).apply(force_base))
    reconstructed = np.stack((-local_check[:, 1], -local_check[:, 0],
                              local_check[:, 2]), axis=-1)
    if np.max(np.abs(reconstructed - field.reshape(140, 3))) > 1e-6:
        raise ValueError("TacFF local/base round-trip mismatch")
    return xyz_base.astype(np.float32), force_base.astype(np.float32)
