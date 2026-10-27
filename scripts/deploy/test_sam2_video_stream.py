"""Unit tests for causal video predictor contract (fake SAM2, no real image claim)."""
import numpy as np
import pytest
import torch
from scripts.deploy.sam2_video_stream import CausalVideoSAM2Stream


class FakeVideo:
    image_size = 32
    def __init__(self):
        self.masks = {}
        self.calls = []
    def init_state(self, path, *, offload_video_to_cpu, offload_state_to_cpu):
        assert offload_video_to_cpu and offload_state_to_cpu
        return dict(images=torch.zeros(1, 3, 32, 32), num_frames=1, cached_features={0: "jpeg"})
    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.masks[obj_id] = torch.as_tensor(np.asarray(mask).copy())
        self.calls.append(("mask", frame_idx, obj_id))
    def add_new_points_or_box(self, *, inference_state, frame_idx, obj_id, points, labels):
        assert list(labels) == [1, 0]
        out = torch.zeros(12, 16, dtype=torch.bool)
        out[int(points[0, 1]), int(points[0, 0])] = True
        self.masks[obj_id] = out
        self.calls.append(("points", frame_idx, obj_id))
    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse):
        assert start_frame_idx == state["num_frames"] - 1
        assert max_frame_num_to_track == 0 and reverse is False
        ids = list(self.masks)
        self.calls.append(("propagate", start_frame_idx))
        yield start_frame_idx, ids, torch.stack([self.masks[i].float()[None] * 2 - 1 for i in ids])


def stream(prompts=None):
    s = CausalVideoSAM2Stream.__new__(CausalVideoSAM2Stream)
    s.predictor = FakeVideo()
    s.torch = torch
    s.device = "cpu"
    s.prompts = prompts or {"seed_frame": 0, "objects": {}}
    s.corrections = {}
    s.seed_frame = 0
    s.init_frames = {"target_flask": 0, "funnel": 0}
    s.min_area = 1
    s.max_fraction = .9
    s.max_frames_per_episode = 10
    s.score_available = False
    s.state = None
    s.shape = None
    s.previous_frame = None
    s.seeded = set()
    s.last_area = {"target_flask": 0, "funnel": 0}
    s.last_valid = {"target_flask": False, "funnel": False}
    return s


def test_manual_masks_and_one_step_video_memory():
    s = stream()
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    f = np.zeros((12, 16), bool); f[2:5, 2:5] = True
    t = np.zeros((12, 16), bool); t[7:10, 10:13] = True
    out, overlap = s.predict(0, rgb, {"funnel_seed_mask": f, "target_flask_seed_mask": t})
    assert overlap == 0
    np.testing.assert_array_equal(out["funnel"][0], f)
    np.testing.assert_array_equal(out["target_flask"][0], t)
    assert out["funnel"][2] == "manual_seed"
    assert out["funnel"][1] == -1.0
    assert s.state["cached_features"] == {}
    out, _ = s.predict(1, rgb, {})
    assert out["funnel"][2] == "video_tracked"
    assert s.state["num_frames"] == 2
    assert [c for c in s.predictor.calls if c[0] == "propagate"] == [
        ("propagate", 0), ("propagate", 1)]


def test_reviewed_level1_point_semantics_and_ids():
    s = stream({"seed_frame": 0, "objects": {
        "funnel": {"positive": [[3, 3]], "negative": [[1, 1]]},
        "target_flask": {"positive": [[12, 8]], "negative": [[8, 8]]}}})
    out, _ = s.predict(0, np.zeros((12, 16, 3), np.uint8), {})
    assert all(v[3] for v in out.values())
    assert ("points", 0, 1) in s.predictor.calls
    assert ("points", 0, 2) in s.predictor.calls


def test_reject_box_only_and_mismatched_resolution():
    s = stream({"seed_frame": 0, "objects": {
        "funnel": {"box_xyxy": [0, 0, 5, 5]},
        "target_flask": {"positive": [[12, 8]], "negative": [[8, 8]]}}})
    with pytest.raises(ValueError, match="positive AND negative"):
        s.predict(0, np.zeros((12, 16, 3), np.uint8), {})


def test_delayed_funnel_seed():
    s = stream({"seed_frame": 0, "objects": {
        "funnel": {"positive": [[3, 3]], "negative": [[1, 1]]},
        "target_flask": {"positive": [[12, 8]], "negative": [[8, 8]]}}})
    s.init_frames["funnel"] = 2
    rgb = np.zeros((12, 16, 3), np.uint8)
    assert not s.predict(0, rgb, {})[0]["funnel"][3]
    s.predict(1, rgb, {})
    assert s.predict(2, rgb, {})[0]["funnel"][3]
    assert ("points", 2, 2) in s.predictor.calls
