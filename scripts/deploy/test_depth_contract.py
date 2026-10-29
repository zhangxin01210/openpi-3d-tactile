"""Contract regression: RGBTrack's registered depth is not Spatial's native depth."""
from pathlib import Path

import numpy as np
import pytest

from scripts.deploy.run_online_perception import _validate_packet


def _packet():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    return {
        "front_rgb": image,
        "left_rgb": image,
        "right_rgb": image,
        "front_depth_native": np.ones((6, 8), dtype=np.uint16) * 1000,
        "front_depth_color_aligned": np.ones((12, 16), dtype=np.uint16) * 1000,
        "robot_state": np.zeros(1972, dtype=np.float32),
        "timestamp_s": np.asarray(0.0),
        "frame_index": np.asarray(0),
    }


def test_depth_spaces_remain_distinct():
    result = _validate_packet(Path("f.npz"), _packet())
    assert result["front_depth_native"].shape == (6, 8)
    assert result["front_depth_color_aligned"].shape == (12, 16)


def test_legacy_ambiguous_depth_is_rejected():
    packet = _packet()
    packet["front_depth_raw"] = packet.pop("front_depth_native")
    with pytest.raises(KeyError, match="front_depth_native"):
        _validate_packet(Path("f.npz"), packet)


def test_aligned_depth_must_match_rgb_but_native_need_not():
    packet = _packet()
    packet["front_depth_color_aligned"] = np.ones((6, 8), np.uint16)
    with pytest.raises(ValueError, match="match front RGB"):
        _validate_packet(Path("f.npz"), packet)


def test_depth_must_be_finite():
    packet = _packet()
    packet["front_depth_native"] = np.ones((6, 8), dtype=np.float32)
    packet["front_depth_native"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        _validate_packet(Path("f.npz"), packet)
