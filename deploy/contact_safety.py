"""Repository-side compatibility exports for the single-file robot deployment client.

The authoritative implementations live IN the standalone robot script,
`ur7e_xhand_deploy_3d_tactile_multi.py`, because on the actual UR/XHand
machine the user copies only that file into the LeRobot scripts directory.

This module exists solely to preserve test/import compatibility in the
openpi repository; do not copy it to the robot host.
"""
from deploy.ur7e_xhand_deploy_3d_tactile_multi import (
    ContactSafetyGuard,
    SafetyTrip,
    load_safety_limits,
    raw_tactile_finger_peaks,
    TACTILE_START,
    FINGERS,
    FINGER_STRIDE,
    TAXELS,
    FORCE_OFFSET,
    EXPECTED_STATE,
)

__all__ = [
    "ContactSafetyGuard", "SafetyTrip", "load_safety_limits",
    "raw_tactile_finger_peaks", "TACTILE_START", "FINGERS",
    "FINGER_STRIDE", "TAXELS", "FORCE_OFFSET", "EXPECTED_STATE",
]
