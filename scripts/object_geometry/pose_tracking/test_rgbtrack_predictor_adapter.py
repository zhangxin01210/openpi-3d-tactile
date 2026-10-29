"""Upstream FoundationPose accesses attributes beyond scorer.predict()."""
from __future__ import annotations

import numpy as np

from scripts.object_geometry.pose_tracking.rgbtrack_adapter import _BatchedRGBTrackPredictor


class FakePredictor:
    def __init__(self):
        self.model = object()
        self.cfg = {"input": "test"}
        self.batches = []

    def predict(self, *, ob_in_cams):
        import torch
        self.batches.append(len(ob_in_cams))
        return torch.arange(len(ob_in_cams), dtype=torch.float32), None

    def predict_depth_batches(self, *args, **kwargs):
        return "upstream_depth_path", None


def test_preserves_upstream_attributes_and_special_methods():
    source = FakePredictor()
    adapter = _BatchedRGBTrackPredictor(source, batch_size=2)
    assert adapter.model is source.model
    assert adapter.cfg is source.cfg
    assert adapter.predict_depth_batches()[0] == "upstream_depth_path"


def test_candidate_score_batches_are_bounded():
    predictor = FakePredictor()
    adapter = _BatchedRGBTrackPredictor(predictor, batch_size=2)
    scores, vis = adapter.predict(ob_in_cams=np.zeros((5, 4, 4)))
    assert predictor.batches == [2, 2, 1]
    assert scores.shape == (5,)
    assert vis is None
