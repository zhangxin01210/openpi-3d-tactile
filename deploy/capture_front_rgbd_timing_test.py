"""Per-stream timestamp comparison must respect clock domains."""

from __future__ import annotations

from capture_front_rgbd_timing import comparable_delta_ms


def test_only_comparable_frame_timestamp_domains_are_subtracted() -> None:
    color = {"timestamp_ms": 103.5, "timestamp_domain": "hardware_clock"}
    depth = {"timestamp_ms": 100.0, "timestamp_domain": "hardware_clock"}
    assert comparable_delta_ms(color, depth) == 3.5
    assert comparable_delta_ms(color, {**depth, "timestamp_domain": "system_time"}) is None
