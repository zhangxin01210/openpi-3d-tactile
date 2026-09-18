# Spatial Preprocessing V1 Milestone

Date: 2026-09-18

## 1. Milestone

The repository now has a single canonical preprocessing path:

```text
RGB-D + observation.state
        |
        v
SpatialPreprocessor
        |
        +-- visual geometry
        |     depth -> depth frame -> color frame -> base_link
        |     -> ROI -> 5 mm voxel -> Morton sampling
        |     -> RGB association
        |
        +-- tactile geometry
              official transformed taxel geometry
              + state -> verified URDF FK
              + raw tactile force parsing / axis mapping
              -> base_link
        |
        v
SpatialObservation
```

Baseline output per frame:

```text
visual_xyz_m         [4096, 3] float32
visual_rgb           [4096, 3] uint8
visual_rgb_valid     [4096] bool

tactile_xyz_m        [600, 3] float32
tactile_force_base   [600, 3] float32
tactile_force_norm   [600] float32
finger_id            [600] int8
taxel_id             [600] int16
```

Coordinate frame: `base_link`

XYZ unit: meter

Tactile force unit: `dataset_native`

## 2. Frozen baseline decisions

- Camera roles: `front`, `left`
- Visual voxel size: 5 mm
- Visual sampler: Morton stride
- Visual point count: 4096
- Tactile points: all 600 taxels; tactile is not part of visual sampling
- Depth preprocessing: raw depth, no median / bilateral / hole filling
- Diagnostic geometry offsets: OFF by default
- Train / offline and deployment / online must call the same `SpatialPreprocessor`

8192 visual points remain a strong model-side ablation candidate, but are not the V1 baseline.

## 3. Validation evidence

### 3.1 Visual geometry parity

Real `press_0828_17`, frame 213:

Front:

```text
new ROI = 90321
legacy ROI = 90321

XYZ error:
median = 0.000043449 mm
p95    = 0.000088211 mm
max    = 0.000183111 mm
```

Left:

```text
new ROI = 110651
legacy ROI = 110651

XYZ error:
median = 0.000056247 mm
p95    = 0.000105211 mm
max    = 0.000187376 mm
```

Raw RGB association agreed at >99.8% validity classification; RGB values on mutually valid points were effectively identical.

### 3.2 Official tactile geometry audit

Official raw sensor-body JSON transformed with the manufacturer equations was compared against official transformed JSON.

T16:

```text
median = 1.04e-07 mm
p95    = 3.81e-06 mm
max    = 3.82e-06 mm
```

T30:

```text
median = 2.38e-07 mm
p95    = 3.82e-06 mm
max    = 3.86e-06 mm
```

Therefore the official transformed JSON is used directly as the runtime link2-local geometry.

### 3.3 FK parity

Legacy FK and new `RobotKinematics` were compared on real frames 0, 50, 100 and 213.

```text
max translation error = 3.60e-13 mm
max rotation error    = 4.38e-14 deg
max |delta R_ij|      ~ 1e-15
```

The two FK implementations are numerically equivalent to floating-point precision.

### 3.4 Complete tactile pipeline parity

Independent reference chain versus new canonical tactile pipeline:

```text
max tactile XYZ error        = 5.65e-05 mm
max force component error    = 6.13e-06 dataset-native
max force norm error         = 6.38e-06 dataset-native
state prefix                 = 52
finger / taxel ordering      = exact
```

Frame 213 strongest tactile point:

```text
finger_id = 1  (index)
taxel_id  = 55
force norm = 51.2445 dataset-native
```

### 3.5 Integrated `SpatialObservation`

Real frame 213:

```text
visual_xyz_m      (4096, 3) float32
visual_rgb        (4096, 3) uint8
visual_rgb_valid  (4096,)   ratio = 0.689697

tactile_xyz_m     (600, 3) float32
tactile_force     (600, 3)

strongest tactile:
finger_id = 1
taxel_id  = 55
norm      = 51.2445
```

### 3.6 Single-camera operation

Real frame 50:

Front:

```text
visual = (4096, 3)
RGB-valid ratio = 0.759766
```

Left:

```text
visual = (4096, 3)
RGB-valid ratio = 0.564453
```

The canonical preprocessor therefore supports single-camera and multi-camera configurations without a separate preprocessing implementation.

## 4. Depth artifact audit

Frame 213 was audited with raw / median3 / median5 / bilateral depth.

Front:

```text
median3 edge p95 change = 8 mm
median5 edge p95 change = 17 mm
bilateral edge p95 change = 3 mm
```

Left:

```text
median3 edge p95 change = 8 mm
median5 edge p95 change = 16 mm
bilateral edge p95 change = 3 mm
```

Human inspection found no meaningful visual benefit large enough to justify the extra processing.

Decision:

```text
V1 keeps raw depth.
No median filter.
No bilateral filter.
No hole filling.
```

## 5. Visual sampling coverage

Across real frames 0, 50, 100 and 213:

2048:

```text
candidate -> sampled p95:
median/max across frames = 18.54 / 19.06 mm

within 10 mm = 51.7%
within 15 mm = 85.6%
```

4096:

```text
candidate -> sampled p95:
median/max across frames = 13.37 / 13.62 mm

within 10 mm = 78.9%
within 15 mm = 97.2%
```

8192:

```text
candidate -> sampled p95:
median/max across frames = 9.66 / 9.95 mm

within 10 mm = 95.7%
within 15 mm = 99.4%
```

4096 remains the engineering baseline. 8192 should be revisited when the point encoder is implemented because the downstream encoder cost, not preprocessing, is likely to dominate the 4096-vs-8192 tradeoff.

One apparent candidate-count mismatch on frame 50 was audited:

```text
official                = 49183
float32 + ROI origin    = 49183
float64 + ROI origin    = 49184
```

This was an audit-only floating-point voxel-boundary effect, not a production geometry error.

## 6. Canonical preprocessing latency

Two cameras, complete `SpatialPreprocessor.preprocess()` only; disk/video decode and static initialization excluded.

4096:

```text
mean / median = 71.877 / 70.909 ms
p95           = 83.625 ms
median FPS    = 14.10
```

8192:

```text
mean / median = 69.559 / 69.787 ms
p95           = 72.731 ms
median FPS    = 14.33
```

The small difference is benchmark noise. The supported conclusion is:

```text
4096 -> 8192 adds no measurable preprocessing bottleneck.
```

It does not imply that 8192 will be free in the future point encoder.

## 7. Durable files

Core:

```text
src/openpi/spatial/schema.py
src/openpi/spatial/config.py
src/openpi/spatial/calibration.py
src/openpi/spatial/geometry.py
src/openpi/spatial/tactile_geometry.py
src/openpi/spatial/kinematics.py
src/openpi/spatial/tactile.py
src/openpi/spatial/preprocess.py
```

Durable utilities:

```text
scripts/spatial/visualize_spatial_web.py
scripts/spatial/export_spatial_derived_dataset.py
scripts/spatial/diagnostics/audit_visual_sampling_coverage.py
scripts/spatial/diagnostics/benchmark_spatial_preprocess_latency.py
```

## 8. Known non-blocking items

- Small legacy/new differences remain in RGB visibility / z-buffer edge semantics. The 3D geometry itself is aligned.
- The dataset-native tactile force values are not converted to Newton in V1.
- 4096 vs 8192 is intentionally deferred to point-encoder / task ablation.
- Derived spatial data should be treated as regenerable data, not raw source-of-truth sensor data.

## 9. Next milestone

The next stage is model-input engineering:

```text
raw dataset
    +
derived spatial modality
        |
        v
OpenPI dataset adapter
        |
        v
point encoder
        |
        v
spatial tokens
        |
        v
pi0
```

The raw RGB-D / FK / tactile preprocessing chain should not be reimplemented inside the model or training dataset loader.
