from __future__ import annotations

import numpy as np

from openpi.spatial.tactile_ema import CausalTactileEMA
from openpi.spatial.tactile_ema import precompute_episode_tactile_ema


def test_ema_is_causal_timestamped_and_resets() -> None:
    filt = CausalTactileEMA(tau_s=1.0, taxel_count=2)
    first = filt.update(timestamp_s=0.0, force_norm=np.array([1.0, 2.0]))
    assert np.all(first.reference_valid)
    np.testing.assert_allclose(first.current_minus_ema_force_norm, 0.0)

    second = filt.update(timestamp_s=1.0, force_norm=np.array([3.0, 2.0]))
    expected_alpha = 1.0 - np.exp(-1.0)
    assert np.isclose(second.alpha, expected_alpha)
    np.testing.assert_allclose(second.ema_before_force_norm, [1.0, 2.0])
    np.testing.assert_allclose(second.current_minus_ema_force_norm, [2.0, 0.0])
    np.testing.assert_allclose(second.current_minus_previous_force_norm, [2.0, 0.0])

    filt.reset()
    reset = filt.update(timestamp_s=0.0, force_norm=np.array([3.0, 2.0]))
    np.testing.assert_allclose(reset.current_minus_ema_force_norm, 0.0)


def test_missing_frame_does_not_create_cross_gap_delta() -> None:
    filt = CausalTactileEMA(tau_s=1.0, taxel_count=1)
    filt.update(timestamp_s=0.0, force_norm=np.array([1.0]))
    missing = filt.update(timestamp_s=0.1, force_norm=None)
    assert not bool(missing.sample_valid[0])
    np.testing.assert_allclose(missing.ema_after_force_norm, [1.0])
    recovered = filt.update(timestamp_s=0.2, force_norm=np.array([4.0]))
    assert not bool(recovered.previous_valid[0])
    np.testing.assert_allclose(recovered.current_minus_previous_force_norm, [0.0])


def test_precompute_sorts_frames_and_does_not_depend_on_sampling_order() -> None:
    frames = [
        {"frame_index": 2, "timestamp_s": 0.2, "force_norm": np.array([3.0])},
        {"frame_index": 0, "timestamp_s": 0.0, "force_norm": np.array([1.0])},
        {"frame_index": 1, "timestamp_s": 0.1, "force_norm": np.array([2.0])},
    ]
    ordered = precompute_episode_tactile_ema(frames, taxel_count=1)
    shuffled = precompute_episode_tactile_ema(list(reversed(frames)), taxel_count=1)
    for frame in ordered:
        np.testing.assert_allclose(
            ordered[frame].ema_after_force_norm,
            shuffled[frame].ema_after_force_norm,
        )


def test_short_force_change_is_distinct_from_slow_reference() -> None:
    filt = CausalTactileEMA(tau_s=2.0, taxel_count=1)
    filt.update(timestamp_s=0.0, force_norm=np.array([1.0]))
    jump = filt.update(timestamp_s=0.1, force_norm=np.array([6.0]))
    sustained = filt.update(timestamp_s=2.1, force_norm=np.array([6.0]))
    assert float(abs(jump.current_minus_ema_force_norm[0])) > float(abs(sustained.current_minus_ema_force_norm[0]))
