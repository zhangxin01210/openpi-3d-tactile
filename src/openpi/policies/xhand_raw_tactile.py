"""Raw XHand 18D proprioception plus 1800D current tactile force."""
import numpy as np


def extract_raw_tactile_state(state):
    raw = np.asarray(state, dtype=np.float32)
    if raw.shape[-1] != 1972:
        raise ValueError(f"raw tactile state requires 1972 values, got {raw.shape}")
    proprio = np.concatenate((raw[..., :6], raw[..., 28:52:2]), axis=-1)
    tactile = np.concatenate([raw[..., 52 + i * 384 + 24:52 + i * 384 + 24 + 360] for i in range(5)], axis=-1)
    result = np.concatenate((proprio, tactile), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("raw tactile state contains NaN or Inf")
    return result
