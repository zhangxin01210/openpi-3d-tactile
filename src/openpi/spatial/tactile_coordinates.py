"""Configuration-driven XHand finger-link2 tactile input adaptation."""
from pathlib import Path

import numpy as np

from openpi.spatial.config import make_baseline_config
from openpi.spatial.preprocess import DEFAULT_T16_TRANSFORMED_PATH
from openpi.spatial.preprocess import DEFAULT_T30_RIGHT_TRANSFORMED_PATH
from openpi.spatial.tactile import extract_raw_tactile_force
from openpi.spatial.tactile import make_taxel_ids
from openpi.spatial.tactile import map_force_axes_to_link
from openpi.spatial.tactile_geometry import load_xhand_taxel_geometry


class FingerLink2TactileAdapter:
    """Use official static link geometry and the mapped raw force in both data paths."""

    def __init__(self, repo_root: str | Path):
        root = Path(repo_root)
        self.config = make_baseline_config().tactile
        geometry = load_xhand_taxel_geometry(
            t16_transformed_path=root / DEFAULT_T16_TRANSFORMED_PATH,
            t30_right_transformed_path=root / DEFAULT_T30_RIGHT_TRANSFORMED_PATH,
            tactile_config=self.config,
        )
        self.xyz = np.asarray(geometry.xyz_link_m, dtype=np.float32).reshape(600, 3)
        self.finger_id, self.taxel_id = make_taxel_ids(config=self.config)

    def __call__(self, state, tactile):
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        raw, _ = extract_raw_tactile_force(np.asarray(state, dtype=np.float32), config=self.config)
        force = map_force_axes_to_link(raw, config=self.config).reshape(600, 3)
        if not np.array_equal(tactile["finger_id"], self.finger_id) or not np.array_equal(tactile["taxel_id"], self.taxel_id):
            raise ValueError("tactile taxel IDs do not match the official link2 geometry")
        # Rotation preserves norm. A mismatch indicates an old, misaligned sidecar.
        if not np.allclose(np.linalg.norm(force, axis=-1), tactile["force_norm"], rtol=1e-5, atol=1e-4):
            raise ValueError("raw tactile force does not match spatial sidecar force norms")
        return {**tactile, "xyz_m": self.xyz.copy(), "force": force}
