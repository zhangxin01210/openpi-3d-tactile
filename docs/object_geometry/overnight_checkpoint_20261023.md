# Overnight Checkpoint: 2026-10-23

Captured before continuing migration work on branch
`feat/object-geometry-migration`.

## Git

- Pushed baseline snapshot: `2cfba08` (`chore: preserve tested HOI level1 prototype sources`).
- Existing unrelated work remains unstaged: `scripts/train.py`,
  `README_position_scan_v2.md`, and `scripts/compute_xhand_norm_stats_fast.py`.
- The interrupted turn left uncommitted formal-core files under
  `src/openpi/object_geometry/` and wrappers under `scripts/object_geometry/`.

## Prototype state observed

- 70/70 Level-1 episode clip directories exist.
- 70/70 `seed_prompts.json` files exist; e0 is the older special-case setup and
  e23/e38/e49/e58/e65 are the reviewed position representatives.
- Frame-0 previews exist for e1-e36 and e38, e49, e58, e65; other episodes do
  not currently have preview images.
- Full tracked mask reports exist for e0, e23, e38, e49, e58, and e65 only.
- Explicit tracking-review records exist for e23, e38, e49, e58, and e65.
- e0 has an older complete cloud/memory output and report, but no current
  `tracking_reviewed.json` in its episode directory.
- No full all-70 tracking, stride-1 sidecar, or formal QA package was found.

## Safety

No existing spatial pipeline, dataset, outputs, jobs, or unrelated work were
modified while taking this checkpoint. Existing long-running FactileLDM jobs
were observed and left untouched.
