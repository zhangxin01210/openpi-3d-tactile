"""Mask-provider boundary for the dependency-free geometry core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class ObjectMaskProvider(ABC):
    """Provide named boolean masks for a frame without prescribing a tracker."""

    @abstractmethod
    def masks_for_frame(self, frame_index: int) -> Mapping[str, Any]:
        """Return object-name to image-mask mappings for ``frame_index``."""
