"""Focused source/provenance checks for the formal object-geometry exporter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from openpi.object_geometry.calibration import front_calibration_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "configs/ur7e_xhand/front_calibration_candidate.json"
CLOUD_RUN = "front_candidate_stride1"
MEMORY_RUN = "abc_v1_front_candidate_stride1"


def _exporter_module():
    name = "object_geometry_export_sidecar_test_module"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts/object_geometry/export_sidecar.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_formal_source(episode_root: Path, calibration: dict) -> None:
    cloud_root = episode_root / f"cloud_frames_{CLOUD_RUN}"
    memory_root = episode_root / MEMORY_RUN / "frames"
    cloud_root.mkdir(parents=True)
    memory_root.mkdir(parents=True)
    current = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    history = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)
    np.savez_compressed(
        cloud_root / "00000.npz",
        frame_index=np.int32(0),
        funnel_xyz_base_m=current,
        target_flask_xyz_base_m=current,
    )
    np.savez_compressed(
        memory_root / "00000.npz",
        B_funnel_supported_history_xyz_base_m=history,
        B_funnel_support_count=np.array([2], dtype=np.int16),
        B_flask_static_history_xyz_base_m=history,
        B_flask_support_count=np.array([3], dtype=np.int16),
    )
    (episode_root / f"sequence_report_{CLOUD_RUN}.json").write_text(
        json.dumps({"stride": 1, "calibration": calibration}), encoding="utf8"
    )
    (memory_root.parent / "report.json").write_text(
        json.dumps(
            {
                "source_cloud_run": cloud_root.name,
                "source_sequence_report": str(episode_root / f"sequence_report_{CLOUD_RUN}.json"),
                "calibration": calibration,
                "source_calibration": calibration,
            }
        ),
        encoding="utf8",
    )


def test_exporter_accepts_only_matching_profiled_stride_one_sources(tmp_path: Path) -> None:
    exporter = _exporter_module()
    calibration = front_calibration_provenance(REPO_ROOT, PROFILE)
    episode_root = tmp_path / "level1_sequence_e23"
    _write_formal_source(episode_root, calibration)

    record, error = exporter._episode_record(  # noqa: SLF001 - validate the export source boundary directly.
        episode_root,
        23,
        require_stride=1,
        cloud_run_name=CLOUD_RUN,
        memory_run_name=MEMORY_RUN,
        expected_calibration=calibration,
    )
    assert error is None
    assert record is not None
    assert record["records"][0]["manipulated_mask"][0]
    assert record["records"][0]["target_mask"][1]

    report = episode_root / MEMORY_RUN / "report.json"
    bad = json.loads(report.read_text(encoding="utf8"))
    bad["source_calibration"] = {"calibration_mode": "baseline_unprofiled"}
    report.write_text(json.dumps(bad), encoding="utf8")
    record, error = exporter._episode_record(  # noqa: SLF001 - validate the export source boundary directly.
        episode_root,
        23,
        require_stride=1,
        cloud_run_name=CLOUD_RUN,
        memory_run_name=MEMORY_RUN,
        expected_calibration=calibration,
    )
    assert record is None
    assert error is not None
    assert "memory source" in error
