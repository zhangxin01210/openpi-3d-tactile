"""Causal per-taxel tactile EMA shared by offline training and online inference.

The filter operates on force magnitudes, not on a calibrated grasp-force
estimate.  It has no access to future frames and explicitly breaks the
``current - previous`` link after a missing/invalid sample.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TactileEMAState:
    """One causal update result in taxel force-norm space."""

    ema_before_force_norm: np.ndarray
    ema_after_force_norm: np.ndarray
    current_minus_ema_force_norm: np.ndarray
    current_minus_previous_force_norm: np.ndarray
    reference_valid: np.ndarray
    previous_valid: np.ndarray
    sample_valid: np.ndarray
    alpha: float


class CausalTactileEMA:
    """Per-taxel exponential moving reference with explicit missing semantics."""

    def __init__(self, *, tau_s: float = 1.0, taxel_count: int = 600) -> None:
        self.tau_s = float(tau_s)
        self.taxel_count = int(taxel_count)
        if not np.isfinite(self.tau_s) or self.tau_s <= 0:
            raise ValueError("tau_s must be finite and positive")
        if self.taxel_count <= 0:
            raise ValueError("taxel_count must be positive")
        self.reset()

    def reset(self) -> None:
        """Reset episode state.  Call this at every episode boundary."""

        self._ema = np.zeros((self.taxel_count,), dtype=np.float32)
        self._previous = np.zeros((self.taxel_count,), dtype=np.float32)
        self._reference_valid = np.zeros((self.taxel_count,), dtype=bool)
        self._previous_valid = np.zeros((self.taxel_count,), dtype=bool)
        self._last_timestamp_s: float | None = None

    def update(
        self,
        *,
        timestamp_s: float,
        force_norm: np.ndarray | Iterable[float] | None,
        valid_mask: np.ndarray | Iterable[bool] | None = None,
    ) -> TactileEMAState:
        """Consume one current frame and return only causal quantities.

        Invalid taxels leave the EMA unchanged.  They also invalidate the
        previous-frame delta for the next valid sample, so a later frame never
        turns a gap into a fabricated short-time impulse.
        """

        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        if force_norm is None:
            values = np.zeros((self.taxel_count,), dtype=np.float32)
            valid = np.zeros((self.taxel_count,), dtype=bool)
        else:
            values = np.asarray(force_norm, dtype=np.float32)
            if values.shape != (self.taxel_count,):
                raise ValueError(f"force_norm must have shape ({self.taxel_count},), got {values.shape}")
            valid = np.isfinite(values)
            if valid_mask is not None:
                supplied = np.asarray(valid_mask, dtype=bool)
                if supplied.shape != values.shape:
                    raise ValueError(f"valid_mask must have shape {values.shape}, got {supplied.shape}")
                valid &= supplied
            values = np.where(valid, values, 0.0).astype(np.float32)

        before = self._ema.copy()
        previous_valid_before = self._previous_valid.copy()
        after = before.copy()
        current_minus_ema = np.zeros_like(values)
        current_minus_previous = np.zeros_like(values)

        dt = 0.0 if self._last_timestamp_s is None else max(0.0, timestamp - self._last_timestamp_s)
        alpha = float(1.0 - np.exp(-dt / self.tau_s)) if self._last_timestamp_s is not None else 0.0

        newly_initialized = valid & ~self._reference_valid
        continuing = valid & self._reference_valid
        # Initialisation is deliberately a zero-change event: the first valid
        # sample establishes a reference rather than pretending to be a load
        # impulse.
        before[newly_initialized] = values[newly_initialized]
        current_minus_ema[continuing] = values[continuing] - before[continuing]
        current_minus_previous[valid & previous_valid_before] = values[valid & previous_valid_before] - self._previous[valid & previous_valid_before]
        after[newly_initialized] = values[newly_initialized]
        after[continuing] = before[continuing] + alpha * (values[continuing] - before[continuing])

        self._ema = after
        self._reference_valid |= valid
        self._previous = np.where(valid, values, self._previous)
        self._previous_valid = valid.copy()
        # Timestamp advances even on a missing frame.  The EMA itself does not
        # update, while the next alpha reflects elapsed wall-clock time.
        self._last_timestamp_s = timestamp

        return TactileEMAState(
            ema_before_force_norm=before,
            ema_after_force_norm=after.copy(),
            current_minus_ema_force_norm=current_minus_ema,
            current_minus_previous_force_norm=current_minus_previous,
            reference_valid=self._reference_valid.copy(),
            previous_valid=previous_valid_before,
            sample_valid=valid,
            alpha=alpha,
        )


def precompute_episode_tactile_ema(
    frames: Iterable[dict[str, object]],
    *,
    tau_s: float = 1.0,
    taxel_count: int = 600,
) -> dict[int, TactileEMAState]:
    """Precompute an episode in frame order, independent of input iteration order."""

    ordered = sorted((dict(frame) for frame in frames), key=lambda frame: int(frame["frame_index"]))
    filter_state = CausalTactileEMA(tau_s=tau_s, taxel_count=taxel_count)
    output: dict[int, TactileEMAState] = {}
    for frame in ordered:
        frame_index = int(frame["frame_index"])
        if frame_index in output:
            raise ValueError(f"duplicate frame_index {frame_index}")
        output[frame_index] = filter_state.update(
            timestamp_s=float(frame["timestamp_s"]),
            force_norm=frame.get("force_norm"),
            valid_mask=frame.get("valid_mask"),
        )
    return output
