# Level-1 Object Geometry Prototype

Status: migration snapshot captured on 2026-10-23. The tested prototype remains
at `/workspace/mnt/sqzhang26/hoi`; this document records how the formal repo can
reconstruct the workflow without committing its generated products.

## Source and data

- HOI source: `/workspace/mnt/sqzhang26/hoi`
- Dataset: `/workspace/mnt/sqzhang26/openpi-3d-tactile/data/1018_peg_all`
- Episodes: 70 total, grouped as `e0-e15`, `e16-e30`, `e31-e45`, `e46-e53`,
  `e54-e61`, and `e62-e69`.
- Reviewed representatives: e0, e23, e38, e49, e58, e65.
- Prompt and review metadata: `tools/hoi_level1_snapshot/metadata/`.

The group template sources are e0, e23, e38, e49, e58, and e65 in that order.
For e0, the exact reviewed flask clicks are preserved; the active v4 workflow
derives the funnel clicks from the existing e0 funnel mask.

## Environments and commands

- OpenPI environment: `/workspace/mnt/sqzhang26/openpi-3d-tactile/.venv`
- SAM2 environment: `/workspace/mnt/sqzhang26/hoi/envs/sam2`
- SAM2 checkout: `/workspace/mnt/sqzhang26/hoi/repos/sam2`
- Per-episode orchestration: `bash /workspace/mnt/sqzhang26/hoi/scripts/level1_ep.sh EP STEP`
- All-70 frame-0 workflow: `bash /workspace/mnt/sqzhang26/hoi/scripts/level1_batch_preview_all.sh`
- Existing successful non-SAM2 stages use `PYTHONPATH=src` from the formal repo;
  preview/track/review use the SAM2 environment.

## Outputs and scope

Prototype outputs are under `/workspace/mnt/sqzhang26/hoi/outputs`, including
`level1_sequence_e<ID>` directories and the all-70 preview contact sheet. They
are intentionally not committed. Visualization builds used stride 3; formal
training export must eventually use stride 1.

Level-1 A is current RGB-D depth projected under reviewed object masks. B is
strictly causal history: manipulated-object history is propagated in hand-local
coordinates while the grasp hypothesis is active, then transformed to the
base frame; target-object history is accumulated in the stable base frame.
Minor target-mask leakage into nearby support geometry was accepted for Level-1.
Tactile remains an existing native model modality; tactile anchor geometry is
not part of this migration.

## Known issue

The current prototype can leave manipulated-object B history active after
object release. Future formal training export must clear or rapidly decay that
history when the grasp is released or invalid. This migration preserves the
tested behavior and does not claim that release handling is fixed.
