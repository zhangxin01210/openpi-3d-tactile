# Deployment integration convergence — 2026-10-28

## Single goal and boundaries

Deploy an already trained spatial-tactile Pi0 policy on UR + XHand for
funnel insertion into target flask under six historically reviewed layouts.
The initial Erlenmeyer support is NOT the target flask. The only approved
camera geometry is front/left; the uncalibrated wrist camera must not enter
pose estimation. Keep scene spatial points + tactile points + hand-object
relations and retain the trained action/output schema. There is no live
control or claim of task success in this PR.

PR is an isolated REVIEW branch. The original non-clean deployment worktree
is authoritative for uncommitted work and must not be reset or overwritten.

## Source reuse and remaining integration risks

| Interface | Training authority | Deployment | Status |
| --- | --- | --- | --- |
| Six position prompts | `tools/hoi_level1_snapshot/scripts/level1_manual_six_positions_v4.py`; six source episodes 0/23/38/49/58/65 | `scripts/deploy/position_initialization.py` resolves reviewed two-object point prompts, explicit current-scene review and image geometry | PARTIAL: manual selection only; e0 full two-object prompt is not bundled; no automatic scene classification or UI |
| SAM2 tracking | `level1_sam2_multi.py`, reviewed positive/negative prompts and native SAM2VideoPredictor | `sam2_video_stream.py` causal version-pinned memory adaptation | PARTIAL: algorithm family same, video memory quality and exact training parity NOT empirically verified |
| RGB-D | Original LeRobot `RawSpatialDataset`, shared spatial camera calibration | Native depth originates parquet for replay, `_rgb_registered_depth` creates RGB-registered metric z-buffer; two distinct packet arrays | PARTIAL: code adapted; legacy replay depth provenance unknown; live camera SDK must deliver both spaces |
| Object tracking | Training `rgbtrack_native_upright.py` constrained orientation/recovery | Same runtime hooks plus `RGBTrackObject` wrapped scorer/refiner | PARTIAL: imports, CUDA render, synthetic box and real CAD initialization passed on deployment host; no real tracking validation |
| Metric geometry | `object_geometry.production.load_mesh_surface_points`, `_geometry_record` | `OnlineObjectGeometryBuilder` calls same functions/seeds and causal hand scene state | Unit-tests only; no matched real train-sidecar comparison |
| Static flask anchor | Frame-0 human-reviewed scene-specific anchor frozen from frame 1 | Explicit `--static-target-anchor-json` required for full pose mode | BLOCKED: no general initialize-review-accept mechanism for six layouts; never substitute old absolute frame |
| Spatial/tactile | `SpatialPreprocessor` and causal 600-taxel state | shared preprocess; checks explicit taxel input | Unit-tests only; verify real sensor values and update once per frame |
| Pi0 input/output | 68 structured tokens for one evaluated config (33 visual + 25 tactile + 10 relation) and existing policy action semantics | online geometry + `spatial_online.py` into dry-run | PARTIAL: 22 local unit tests and previous isolated Pi0 checkpoint dry-run; end-to-end actual episode under this PR not passed |
| Hardware | front/left cameras + UR/XHand driver | NPZ packet boundary | NO live SDK writer, actuation disabled |

**Crucial caution:** Episode numbers 0/23/38/49/58/65 designate reference
prompt sets, NOT a deployed classifier. Historical same-location grouping is
not proof that clicks work on a new image. Prompt selection must be manually
reviewed on the CURRENT first RGB frame. Packet seed masks are not allowed to
silently supersede the selected point prompts.

## Evidence

- Deployment host: 20 tests + 2 predictor wrapper tests passed, RGBTrack
  phases A/B/C passed, and genuine two-mesh FoundationPose initialization
  reported `shared_models=true`, `ok=true`.
- Full 468-frame Episode 23 replay directory exists. It contains only legacy
  `front_depth_raw`; first-frame exact seed masks exist. No training sidecar
  was located.
- On frame 0 `front_depth_raw` differed from original parquet values;
  that establishes *only* that it cannot be assumed raw. This motivated
  building from original parquet directly. No transformed replay was run yet.

## Change classification

- KEEP: common geometry production functions, separate depth-space
  contract, SAM2 video memory backend, runtime compatibility, shared
  RGBTrack predictors, pi0 geometry forwarding.
- REFACTOR: old legacy depth conversion provenance; fixed prompt
  initialization replaced with explicit template selection; freeze of
  static flask anchor requires approved current-scene pose.
- TEMPORARY QA ONLY: check_rgbtrack_objects.py,
  prepare_depth_replay.py, synthetic setup/phase checks. These must not be
  called from the live per-frame runtime.
- DO NOT AUTOMATE: six-layout recognition without evidence, frame-0
  static flask pose approval, robot actuation.

## Remaining release gates

1. Run position/depth tests and convert only short real replay using
   original data, inspect depth overlay. Fix mismatches rather than rename.
2. Run real SAM2 forward memory with each of the six reviewed references,
   check masks/occlusions against independent human review.
3. Establish per-layout static target anchor from current RGBTrack pose
   with human review and explicit provenance, never a memorized base pose.
4. Compare same-frame real geometry/tactile token arrays with the
   corresponding training-sidecar; no substitution of missing geometry.
5. Verify policy consumes trained geometry tokens and produces finite
   actions in dry-run, benchmark latency and memory under joint inference.
6. Only then test command transport under separate human-approved robot
   safety procedure.

PR MUST remain draft/unmerged until these gates are satisfied.
