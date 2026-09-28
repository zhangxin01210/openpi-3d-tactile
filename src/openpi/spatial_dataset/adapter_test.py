"""Modality ablation must remove unused branches from the training batch."""

import numpy as np

from openpi.spatial_dataset.adapter import SpatialAugmentedDataset
from openpi.spatial_dataset.derived import DerivedSpatialSample


def test_spatial_augmented_dataset_filters_unused_modality():
    sample = DerivedSpatialSample(
        episode_index=0,
        frame_index=0,
        timestamp_s=0.0,
        visual_xyz_m=np.zeros((4, 3), np.float32),
        visual_rgb=np.zeros((4, 3), np.uint8),
        visual_rgb_valid=np.ones(4, bool),
        tactile_xyz_m=np.zeros((2, 3), np.float32),
        tactile_force_base=np.zeros((2, 3), np.float32),
        tactile_force_norm=np.zeros(2, np.float32),
        finger_id=np.zeros(2, np.int8),
        taxel_id=np.arange(2, dtype=np.int16),
    )

    class Sidecar:
        def get(self, *, episode_index, frame_index):
            assert (episode_index, frame_index) == (0, 0)
            return sample

    base = [{"episode_index": 0, "frame_index": 0}]
    both = SpatialAugmentedDataset(base, Sidecar())[0]["spatial"]
    visual = SpatialAugmentedDataset(base, Sidecar(), use_tactile=False)[0]["spatial"]
    tactile = SpatialAugmentedDataset(base, Sidecar(), use_visual=False)[0]["spatial"]
    assert set(both) == {"visual", "tactile"}
    assert set(visual) == {"visual"}
    assert set(tactile) == {"tactile"}
