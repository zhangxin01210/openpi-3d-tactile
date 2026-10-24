"""Optional SAM2 integration boundary.

The core package never imports this module. The prototype's complete SAM2
workflow remains in ``tools/hoi_level1_snapshot/scripts/level1_sam2_multi.py``;
this adapter only provides a narrow callback boundary for future integration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from openpi.object_geometry.providers.base import ObjectMaskProvider


class Sam2MaskProvider(ObjectMaskProvider):
    """Wrap a prepared SAM2 frame reader without importing SAM2 at module load."""

    def __init__(self, frame_reader: Callable[[int], Mapping[str, Any]]) -> None:
        self._frame_reader = frame_reader

    def masks_for_frame(self, frame_index: int) -> Mapping[str, Any]:
        return self._frame_reader(int(frame_index))


def build_sam2_video_predictor(*args: Any, **kwargs: Any) -> Any:
    """Lazily construct SAM2's predictor for callers that installed SAM2."""

    try:
        from sam2.build_sam import build_sam2_video_predictor as builder
    except ImportError as exc:
        raise ImportError(
            "SAM2 is optional; install/use the prototype SAM2 environment before calling this helper"
        ) from exc
    return builder(*args, **kwargs)
