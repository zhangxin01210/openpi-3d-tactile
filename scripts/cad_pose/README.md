# CAD pose prototype: current scope

This directory is **not a ready-to-train pipeline**. It does not yet include a
four-task trainer, evaluation CLI, dataset-wide visualizer, predicted sidecar
exporter, Pi0 configurations, or a serving integration. No existing model is
switched to predicted geometry. Do not use these components to claim a trained
or validated pose model.

`preflight.py` is read-only with respect to source datasets. It writes a report
under `output/cad_pose_transformer_v1/reports/` and exits nonzero even when
provenance checks pass, because the report cannot establish that RGB-D/CAD
alignment is geometrically correct. It lists missing raw data, release files and
pose streams; for supported published releases it checks hashes, calibration,
frame IDs, and sampled ordered-geometry residuals. Peg's legacy geometry has
no demonstrated correspondence or compatible pose stream in this checkout and
requires a separate adapter and keyframe review. Eraser's actual format is not
available locally and must be inspected, not guessed.

```bash
.venv/bin/python scripts/cad_pose/preflight.py --repo .
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/pytest -q src/openpi/cad_pose/test_cad_pose.py
```

The existing GLB exporter uses `trimesh`, which is not installed in all OpenPI
environments. For an isolated invocation with package access, an example is:

```bash
uv run --no-sync --with 'numpy==1.26.4' --with 'trimesh==4.12.2' \
  python scripts/cad_pose/preflight.py --repo .
```

The model and one-batch optimizer tests are synthetic. The reusable components
include a PyTorch cross-attention model, canonical front RGB-D preprocessing,
explicit-symmetry loss/metrics, ordered rigid-fit validation, read-only
published-pose streaming, and a tracker-free runtime. The runtime requires a
checkpoint with explicitly matching model version, per-dataset ordered CAD hash,
and camera/profile hashes. A finite output does not establish detection
confidence or geometric accuracy.
