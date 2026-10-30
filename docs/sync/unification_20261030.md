# Three-machine code unification — 2026-10-30

**Purpose:** one Git history before moving the primary remote to Gitee. This is a **source integration checkpoint**, not a hardware release or new preprocessing design.

## Ancestry

- Cloud A800 tracked HEAD: `928898d`.
- Laptop and deployment main tracked HEAD: `3260d63` (one commit after cloud).
- Deployment PR1 worktree detached at `e79fd63`; this contains the latest review-branch deployment changes.
- This integration branch started at `e79fd63`, and therefore retains all tracked ancestors of the other three worktrees.
- The original PR #1 remains a draft. This separate sync branch is **not** a request to authorize robot actuation.

## Changes from 2026-10-30 uploaded worktree archives

Both original source archives were inspected without executing their scripts. The cloud tracked patch modifies three files; the changes have been integrated here:

1. `scripts/object_geometry/precompute_tactile_ema.py`: read episode frame IDs from spatial mmap shards instead of a nonexistent `_records` index.
2. `src/openpi/object_geometry/cli.py`: correct manipulated/target valid point count field names and only compute sequential pose jumps for adjacent valid frames.
3. `scripts/train.py`: preserve cloud loss progress logging.

Deployment-main tracked patch modifies ten files. Prior PR1 integration already included the main RGBTrack shared-predictor/runtime hooks, causal geometry state, geometry forwarding and checkpoint layout fixes. The **remaining local-only differences are preserved, not blindly overlaid**, including the visualization NPZ replay/512-point audit, model-timing instrumentation, stricter RGBTrack import audit and changed static-prior config. That last config change would alter config digests used by existing dataset QA; it requires versioned migration review.

Uploaded untracked sources include 18 cloud entries and 12 deployment-main entries. One untracked `src/openpi/policies/hand_scene_online.py` overlaps the newer committed implementation and one RGBTrack file is an old backup. The other 28 scripts/docs are retained in the separate recovery artifact for conflict-free addition at their original paths after review. They are predominantly research/evaluation utilities, not new required production pipeline steps.

## Safety / release conditions

- Preserve the original uncommitted worktrees until the recovery bundle has been applied and verified. No `reset --hard`, `clean -fd`, or force push.
- Do not commit datasets, checkpoint files, caches, `mycpp/build`, symlinks to weights or `.venv`.
- The uploaded submodule reports show an old `segment-anything-2-real-time` mapping error inside RGBTrack and an uninitialized XMem child; these are **not** resolved merely by changing the main Git remote.
- `.gitmodules` still points RGBTrack (as well as ALOHA and LIBERO) to GitHub. Gitee mirroring of the primary repo does not make submodule origins reachable automatically.
- Run import/syntax/tests on the appropriate machines. No A800 RGBTrack run, real-sidecar parity or robot actuation has been performed during this sync.
- The old `v1_hand_scene_q2` data and its provenance are immutable. Data preprocessing changes, including selective HTML generation and novice onboarding, occur **after Gitee migration**, on a follow-up branch.

## Gitee migration gate

1. Integrate/review the recovery artifact on this branch in a clean clone, or explicitly archive optional experiments without overwriting active code.
2. Run feasible source tests and compare critical training/deployment schemas; document hardware-dependent blockers.
3. Create an empty Gitee repository, configure its SSH remote on the checked-out unified branch, and push the branch as the new Gitee main.
4. Update all three computers to the exact same commit after backing up any old uncommitted changes.
5. Only then begin the separate new-data preprocessing refactor.
