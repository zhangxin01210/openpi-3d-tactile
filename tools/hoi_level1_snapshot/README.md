# HOI Level-1 Migration Snapshot

This is a focused safety snapshot of the tested Level-1 object-geometry workflow
from `/workspace/mnt/sqzhang26/hoi`, captured on 2026-10-23. It is a reference
copy; it does not move or modify the prototype workspace.

## Confirmed active source

- `scripts/level1_manual_six_positions_v4.py`: reviewed six-position prompt
  templates and guarded spread/install workflow.
- `scripts/level1_sequence_multi.py`: episode preparation and RGB-D candidate
  point-cloud A build after reviewed masks.
- `scripts/level1_sam2_multi.py`: optional SAM2 frame-0 preview, explicit
  approval, causal forward tracking, and tracking review metadata.
- `scripts/level1_contact_memory_multi.py`: causal A/B/C diagnostic build;
  B uses motion-supported hand-local history and base-frame target history.
- `scripts/level1_object_cloud_probe.py`: calibration-aware RGB-D projection
  helpers used by the sequence and memory scripts.
- `scripts/level1_template_init.py`: frame-0 annotation/template helper used
  by the sequence and six-position workflow.
- `scripts/level1_position_scan_v2.py` and `scripts/level1_position_scan.py`:
  position grouping/inspection utilities.
- `scripts/level1_ep.sh`: per-episode orchestration entry point.
- `scripts/level1_batch_preview_all.sh`: all-70 prompt spread, preparation,
  frame-0 preview, and contact-sheet workflow.
- `scripts/level1_multi_summary.py`: small multi-episode diagnostic summary.
- `scripts/README.md`: the working operator notes and limitations.

## Preserved experimental state

`metadata/six_positions/` contains the exact small prompt records available in
the prototype: e0's manually clicked flask prompt plus its preview manifest,
and the manually reviewed two-object prompts for e23, e38, e49, e58, and e65.
The e0 funnel initialization was derived by the active v4 script from the
existing reviewed e0 funnel mask; that mask is intentionally not copied here.

`metadata/tracking_review/` contains the five small explicit review records for
e23, e38, e49, e58, and e65. Their hashes refer to the full mask reports that
remain in the prototype workspace.

## Dependencies outside this snapshot

- Prototype data and generated outputs remain under `/workspace/mnt/sqzhang26/hoi`.
- The dataset is `/workspace/mnt/sqzhang26/openpi-3d-tactile/data/1018_peg_all`.
- RGB-D projection uses the formal repository's existing calibration and
  spatial geometry APIs.
- SAM2 is only used by `level1_sam2_multi.py`, under the prototype SAM2
  environment and repository at `/workspace/mnt/sqzhang26/hoi/repos/sam2` and
  `/workspace/mnt/sqzhang26/hoi/envs/sam2`.
- The scripts expect the OpenPI environment at
  `/workspace/mnt/sqzhang26/openpi-3d-tactile/.venv` for non-SAM2 stages.

## Commands used by the tested workflow

```bash
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP prepare
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP seed
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP preview
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP approve
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP track
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP review-track
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP build
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP memory
bash /workspace/mnt/sqzhang26/hoi/scripts/level1_batch_preview_all.sh
```

The prototype's tested visualization build used `--stride 3`; future formal
training export must use `--stride 1`.

## Intentionally excluded

Generated frame directories, full SAM2 mask sequences, videos, RGB-D point
clouds, HTML/contact sheets, model weights, SAM2 checkpoints, virtualenvs,
archives, cache files, and superseded/debug scripts are not part of this
snapshot. They remain available in `/workspace/mnt/sqzhang26/hoi`.

Known limitations are preserved as facts: candidate depth under a transparent
mask can include background; the e0 funnel prompt depends on its existing
reviewed mask; the manipulated-object B history may persist after release; and
the tactile modality remains native OpenPI state rather than Level-1 anchor
geometry. No release behavior is claimed to be fixed here.
