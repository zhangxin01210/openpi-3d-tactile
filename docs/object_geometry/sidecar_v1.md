# Object Geometry Sidecar

The formal front relational configuration reads
`object_geometry/v1_front_candidate`, an independent fixed-size sidecar joined by
`(episode_index, frame_index)` after the existing `SpatialAugmentedDataset`.
It does not modify or replace `spatial/v1_front_candidate`.

`object_geometry/v1` remains reserved for the older unprofiled Level-1
prototype. It must not be joined with the profiled front spatial sidecar.

Each object has 512 deterministic slots:

```text
<object>_xyz_m       float32 [512, 3], base-frame meters
<object>_mask        bool    [512]
<object>_source      int8    [512], 0=current A, 1=history B
<object>_age         int16   [512], 0=current, positive=history marker
<object>_confidence  float32 [512], deterministic validity/support value
```

The exporter is [export_sidecar.py](/workspace/mnt/sqzhang26/openpi-3d-tactile/scripts/object_geometry/export_sidecar.py).
It accepts only the coexistence-safe `cloud_frames_front_candidate_stride1` and
`abc_v1_front_candidate_stride1` runs by default. It validates each build and
memory report against `configs/ur7e_xhand/front_calibration_candidate.json`,
including the profile SHA-256, applied RGB intrinsics, front base-frame
translation, and the `base_link` projection convention. It refuses stride-3 or
legacy/unprofiled sources, records failed episodes in `manifest.json`, and
marks the sidecar `incomplete` until every requested episode has matching
stride-1 A/B records. Fewer than 512 points are padded with `mask=false`; no
fake-valid points are duplicated.

The prototype does not expose per-point historical last-seen age or calibrated
probabilistic confidence. Therefore v1 uses a documented historical marker and
deterministic support/validity confidence rather than inventing learned values.
