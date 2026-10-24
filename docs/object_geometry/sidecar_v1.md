# Object Geometry Sidecar v1

`object_geometry/v1` is an independent fixed-size sidecar joined by
`(episode_index, frame_index)` after the existing `SpatialAugmentedDataset`.
It does not modify or replace `spatial/v1`.

Each object has 512 deterministic slots:

```text
<object>_xyz_m       float32 [512, 3], base-frame meters
<object>_mask        bool    [512]
<object>_source      int8    [512], 0=current A, 1=history B
<object>_age         int16   [512], 0=current, positive=history marker
<object>_confidence  float32 [512], deterministic validity/support value
```

The exporter is [export_sidecar.py](/workspace/mnt/sqzhang26/openpi-3d-tactile/scripts/object_geometry/export_sidecar.py).
It refuses stride-3 visualization clouds when invoked with the default
`--require-stride 1`, records failed episodes in `manifest.json`, and marks the
sidecar `incomplete` until every requested episode has matching stride-1 A/B
records. Fewer than 512 points are padded with `mask=false`; no fake-valid
points are duplicated.

The prototype does not expose per-point historical last-seen age or calibrated
probabilistic confidence. Therefore v1 uses a documented historical marker and
deterministic support/validity confidence rather than inventing learned values.
