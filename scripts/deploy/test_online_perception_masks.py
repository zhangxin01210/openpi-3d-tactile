from __future__ import annotations

import numpy as np

from scripts.deploy.run_online_perception import CausalSAM2Stream


def test_manual_seed_mask_is_preserved_and_becomes_causal_prompt() -> None:
    stream = CausalSAM2Stream.__new__(CausalSAM2Stream)
    stream.seed_frame = 0
    stream.prompts = {"seed_frame": 0, "objects": {}}
    stream.corrections = {}
    stream.min_area = 4
    stream.max_fraction = 0.9
    stream.low_res = {"funnel": None, "target_flask": None}
    stream.last_masks = {"funnel": None, "target_flask": None}
    stream.last_area = {"funnel": 0, "target_flask": 0}
    stream.last_valid = {"funnel": False, "target_flask": False}

    seed = np.zeros((32, 48), dtype=bool)
    seed[4:12, 9:17] = True
    mask, score, status, valid, recovered = stream._predict_one(
        0, "funnel", {"funnel_seed_mask": seed}, seed.shape
    )

    np.testing.assert_array_equal(mask, seed)
    assert (score, status, valid, recovered) == (1.0, "manual_seed", True, False)
    assert stream.low_res["funnel"].shape == (1, 256, 256)
    np.testing.assert_array_equal(stream.last_masks["funnel"], seed)
    assert np.any(stream.low_res["funnel"] > 0)
    assert np.any(stream.low_res["funnel"] < 0)


def test_manual_correction_mask_is_preserved_for_its_frame(tmp_path) -> None:
    import cv2

    stream = CausalSAM2Stream.__new__(CausalSAM2Stream)
    stream.seed_frame = 0
    stream.prompts = {"seed_frame": 0, "objects": {}}
    correction = tmp_path / "funnel.png"
    expected = np.zeros((24, 32), dtype=np.uint8)
    expected[7:16, 10:20] = 255
    assert cv2.imwrite(str(correction), expected)
    stream.corrections = {"3": {"funnel": {"mask": str(correction)}}}
    stream.min_area = 4
    stream.max_fraction = 0.9
    stream.low_res = {"funnel": None, "target_flask": None}
    stream.last_masks = {"funnel": None, "target_flask": None}
    stream.last_area = {"funnel": 0, "target_flask": 0}
    stream.last_valid = {"funnel": False, "target_flask": False}

    mask, score, status, valid, recovered = stream._predict_one(
        3, "funnel", {}, expected.shape
    )

    np.testing.assert_array_equal(mask, expected > 0)
    assert (score, status, valid, recovered) == (1.0, "corrected", True, True)
