# Hand-Scene V0 training contract

`pi0_xhand_spatial_structured_suffix_hand_scene_v0` preserves the reviewed
front spatial stream and appends geometry relations:

```text
33 RGB-D scene tokens + 25 spatial tactile tokens
+ 5 Hand-Funnel tokens + 5 Hand-Target Flask tokens = 68
```

It does not alter the original front baseline, the legacy 5 HO + 9 OO pilot,
Pi0 checkpoint-loader code, or the XHand action contract.  The dataset action
is still 18-D absolute joint target; the existing model transform pads it to
the model's 32-D action interface.

## Sidecar layout

The configuration reads a new, coexistence-safe directory:

```text
<dataset-root>/object_geometry/v1_hand_scene/
  manifest.json
  shard_e023/
    manipulated_xyz_m.npy          float32 [F, 512, 3]
    manipulated_mask.npy           bool    [F, 512]
    ... existing source/age/confidence arrays for manipulated and target ...
    manipulated_relation_valid.npy bool    [F]
    target_relation_valid.npy      bool    [F]
    manipulated_geometry_quality.npy float32 [F]
    target_geometry_quality.npy      float32 [F]
    grasp_reference_force_norm.npy   float32 [F, 600]
    grasp_short_delta_force_norm.npy float32 [F, 600]
    grasp_contact_valid.npy          bool    [F, 600]
    grasp_quality.npy                float32 [F]
    funnel_mask.npy                  bool    [F, 32, 32]
    funnel_mask_valid.npy            bool    [F]
    funnel_mask_quality.npy          float32 [F]
    funnel_mask_hand_offset_xy.npy   float32 [F, 2]
```

`manipulated_xyz_m` is the calibrated funnel cloud and `target_xyz_m` is the
calibrated target-flask cloud.  Both are fixed `[512, 3]`, in **meters** in
`base_link`; they must come from the external mesh-scale and 6D registration
pipeline, not from this exporter.  Per-object quality is finite `[0, 1]`;
zero makes the corresponding relation invalid.  The embedded object clouds
continue to carry their own per-point valid, source, age and confidence fields.

`manipulated_relation_valid=false` explicitly removes Hand-Funnel tokens once
the funnel's old hand-attached geometry becomes invalid after a lift.  It must
be derived from current and past information only.  The target relation may
remain valid while `target_geometry_quality` becomes low; V0 does not freeze
the bottle pose.

The grasp arrays use each taxel's scalar force magnitude.  Current magnitude
comes from the normal 600-taxel spatial input; reference and short-time delta
are supplied by this sidecar.  This avoids comparing force vectors expressed
under changing hand poses.  `causal_max_source_frame` is an exporter-only
required state-record field and must be `<= frame_index`.

The mask is only a current-frame downsampled funnel silhouette.
`funnel_mask_hand_offset_xy` is the current normalized image-space offset from
the projected hand reference to the funnel-mask centroid, so the optional
branch sees relative visible position rather than an unanchored silhouette.
`false` valid or a missing mask is encoded as zeros and never prevents geometry training.
The initial V0 configuration sets
`hand_scene_enable_funnel_mask_encoder=False`; changing that configuration
field enables an eight-feature lightweight mask encoder.  Neither mode fills a
missing mask from future frames.

## Build `v1_hand_scene`

First generate a formal profiled `v1_front_candidate` geometry sidecar with
the existing exporter.  Then have the external causal 6D/mesh pipeline write
one compressed state record per source frame at:

```text
<state-root>/e023/f000120.npz
```

The record must contain all arrays listed above (the mask arrays are optional
as a group) and `causal_max_source_frame`.  It is deliberately an input
contract rather than an invented detector or a static hand-follow rule.

From the repository root, the build command is:

```bash
PYTHONPATH=src python scripts/object_geometry/export_hand_scene_sidecar.py \
  --source-dataset-root data/1018_peg_all \
  --source-version v1_front_candidate \
  --state-root data/hand_scene_states \
  --output-dataset-root data/1018_peg_all \
  --episodes 23 --generation-commit "$(git rev-parse HEAD)"
```

The exporter refuses to overwrite outputs, requires source calibration
provenance, preserves the metric source clouds byte-for-byte as arrays, and
rejects a record whose causal source frame is in the future.

## Computer-4 smoke test and training

Use checkout-relative paths or set the two environment variables to mounted
locations.  A one-step smoke test checks data joins, model initialization,
68-token routing, weight loading, and backward execution on the selected GPU:

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
export OPENPI_HAND_SCENE_DATASET_ROOT=data/1018_peg_all
export OPENPI_HAND_SCENE_BASE_WEIGHTS=checkpoints/pi0_base/params
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  python scripts/train.py pi0_xhand_spatial_structured_suffix_hand_scene_v0 \
  --exp-name hand_scene_v0_smoke --num-train-steps 1 --batch-size 1 --wandb-enabled false
```

After a successful smoke run, omit the three final override flags and use a
new experiment name for the normal configuration schedule.  This document does
not claim that real e23 registration, masks, or GPU execution were validated
on a machine without those resources.
