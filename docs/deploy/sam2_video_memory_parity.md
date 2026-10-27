# SAM2 Video-Memory Parity (2026-10-27)

## Root cause and change

Reviewed training-side Level-1 masks were generated with
`tools/hoi_level1_snapshot/scripts/level1_sam2_multi.py`:

- `build_sam2_video_predictor`;
- object IDs **target_flask=1, funnel=2**;
- human-reviewed positive/negative points on frame 0;
- `add_new_points_or_box` and `propagate_in_video` with temporal video memory;
- preview and manual tracking review.

The prior deployment default instead invoked `SAM2ImagePredictor` on each
frame with synthetic previous-mask box/point prompts. This is a different
algorithm; it is a plausible mask-drift source, **not proven** to be the
cause of the reported hand-in-mask failure.

The new `scripts/deploy/sam2_video_stream.py` uses the **same video predictor
API and temporal memory mechanism** as the training pipeline, not the previous
image-predictor re-prompting scheme. Every call appends only the present RGB
frame to the predictor state, then calls `propagate_in_video` solely on the
newly appended ordinal. This is causal and preserves separate fixed object IDs.

**Important limitation:** SAM2 2b90b9f5's official `init_state` reads a
whole static folder, not an online stream. We use its *internal* mutable state
`images` / `num_frames` to append normalized frames, and hence this is a
pinned-version adapter rather than an officially supported streaming API.
It uses original RGB frame pixels, unlike the JPEG-based offline pipeline.
There is no empirical claim of bitwise parity or real-mask correctness.
Long-running states accumulate memory and must reset between episodes.
A 2000-frame guard fails closed.

The old image predictor remains selectable with
`--sam2-backend legacy_image` solely for controlled A/B comparisons.
Default is `--sam2-backend video_memory`.

The patch does **not** implement URDF hand-mask subtraction, change the
XHand/UR controller, change pose/geometrical representation, or issue actions.

## Inputs and prompts

Run the new backend using the existing packet replay format, reviewed
`seed_prompts.json`, or the exact-mask packet fields
`funnel_seed_mask` and `target_flask_seed_mask`.
Video backend requires manually reviewed *positive and negative lists* for
point-prompt initialization. Unreviewed box-only prompts are rejected.
Mask corrections are passed directly to SAM2's `add_new_mask`; point
corrections go through `add_new_points_or_box`. A corrected frame retains
status `corrected` so existing RGBTrack reset behavior remains intact.

Frame 0 exact mask PNGs are accepted as seeds and returned unmodified on
that frame, but note this is **different conditioning** from the original
training-side reviewed clicks unless the very same masks were used there.
To compare training and deployment segmentation use the exact same reviewed
point prompts instead of mask seeds, on identical RGB source frames.

The video predictor supplies no comparable per-frame IoU score. The output
records score `-1.0` as unavailable and its metadata sets
`sam2_score_available=false`. Do not use the legacy `min_sam2_score`
threshold as a video-memory quality assertion. Mask area/jump/overlap checks
are safety heuristics only, not segmentation accuracy metrics.

## Offline / replay validation (must run on deployment machine)

The GitHub-only change has *not* run real SAM2 or RGBTrack.

1. Run the existing test suite and new unit tests:
   `PYTHONPATH=.:src .venv-perception/bin/python -m pytest -q scripts/deploy/test_sam2_video_stream.py`
2. Confirm `--sam2-backend video_memory` starts on the installed pinned SAM2
   and handles independently seeded flask/funnel.
3. Re-run saved Episode 23 without actions (prefer mask-only first):
   ```bash
   PYTHONPATH=.:src .venv-perception/bin/python scripts/deploy/run_online_perception.py \
     --replay-dir data/deploy/replay/e023 \
     --output-dir output/deploy/qa/e023_video_memory \
     --prompts data/deploy/seed_prompts.json \
     --sam2-backend video_memory --masks-only
   ```
   Replace the `--prompts` path with the *actual reviewed Episode 23*
   prompt file, and ensure the seed source is the same as the training run.
   If the dataset bundle already carries exact mask seeds, document this
   difference to the point-prompted teacher.
4. Run a second replay to a NEW output directory with
   `--sam2-backend legacy_image`. Compare frame-aligned binary masks and
   overlay sheets against the reviewed original Level-1 sequence (where present).
   Inspect occlusion and hand intrusion manually; do not equate automated
   mask area checks with proof.
5. Only then enable RGBTrack and validate joint SAM2+RGBTrack+Pi0 latency;
   all action transport remains disabled until the user approves live testing.

## Parity inventory

| Part | Reviewed training path | Deployment (video_memory) | Status |
| --- | --- | --- | --- |
| SAM2 type | VideoPredictor | VideoPredictor | Same algorithm family |
| Two IDs | flask 1 / funnel 2 | flask 1 / funnel 2 | Matched |
| Manual points | Reviewed + / - | Reviewed + / - | Matched if identical |
| Temporal features | Native memory | Native memory | Native mechanism, unverified streaming append |
| Frame input | JPEG folder | Raw current RGB normalized | Not pixel-identical |
| QA review | Human preview and tracked contact sheet | Replay overlays | Human parity review still required |
| Error handling | Area/jump warnings | Area/jump invalid gates | Not equivalent |
| Robot control | Separate | Unchanged | No new actuation |
