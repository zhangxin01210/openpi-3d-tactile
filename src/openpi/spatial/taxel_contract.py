"""Canonical XHand taxel ordering and force-independent geometric centers."""
import numpy as np


def validate_taxel_order(tactile):
    finger = np.asarray(tactile["finger_id"])
    taxel = np.asarray(tactile["taxel_id"])
    if not np.array_equal(finger, np.repeat(np.arange(5), 120)) or not np.array_equal(taxel, np.tile(np.arange(1, 121), 5)):
        raise ValueError("XHand taxel order must be five fingers with taxel IDs 1..120 in each finger")


def finger_geometry_centers(tactile):
    validate_taxel_order(tactile)
    xyz = np.asarray(tactile["xyz_m"], dtype=np.float32).reshape(5, 120, 3)
    mask = np.asarray(tactile["point_mask"], dtype=bool).reshape(5, 120) & np.isfinite(xyz).all(axis=-1)
    counts = mask.sum(axis=1)
    centers = np.where(mask[..., None], xyz, 0.0).sum(axis=1) / np.maximum(counts[:, None], 1)
    return {"xyz_m": centers.astype(np.float32), "point_mask": counts > 0}


def online_finger_geometry_centers(preprocessor, state):
    """Reuse the FK and static taxel transform without reading force values."""
    from openpi.spatial.kinematics import tactile_link_transforms
    from openpi.spatial.tactile import make_taxel_ids
    from openpi.spatial.tactile import transform_taxel_positions_to_base

    cfg = preprocessor.config.tactile
    transforms = tactile_link_transforms(state, mapping=preprocessor.state_mapping,
        kinematics=preprocessor.kinematics, finger_link_names=cfg.finger_link_names, root="base_link")
    xyz = transform_taxel_positions_to_base(preprocessor.taxel_geometry.xyz_link_m,
        T_base_link_by_name=transforms, config=cfg).reshape(600, 3)
    finger, taxel = make_taxel_ids(config=cfg)
    return finger_geometry_centers({"xyz_m": xyz, "finger_id": finger, "taxel_id": taxel,
                                    "point_mask": np.ones(600, dtype=bool)})
