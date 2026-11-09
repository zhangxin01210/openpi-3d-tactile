# RGB-D single-CAD V2 → π0 HO V2

Four independent `pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/<task>`
configs reuse the corresponding V1 π0 model and training settings. Only the
object geometry source, version-specific provenance check and metadata differ.
No V1 configuration or existing sidecar is changed.

From the cloud project root, with the frozen checkpoint and full candidate
still in their original locations:

```bash
uv run --no-sync python scripts/cad_pose/publish_v2_sidecar.py --device cuda:0
uv run --no-sync python scripts/cad_pose/smoke_v2_loader.py
uv run --no-sync python scripts/train.py pi0_xhand_pc_v2_tactile_mlp_ho_v2_pretrain_v2/peg_all --exp-name seed42
```

`OPENPI_CAD_POSE_V2_CHECKPOINT` and `OPENPI_CAD_POSE_V2_CACHE` override the
checkpoint and RGB-D cache used by the training configs. If overridden, pass
matching `--checkpoint` and `--cache-root` to the publisher. `--tasks` allows
publishing or smoke testing a subset. Existing completed sidecars are checked
and reused; they and pre-existing manifests are never overwritten. Interrupted
publication resumes from `.pretrained_pose_v2.staging` after validating the
staged episode against its candidate. Inspect unexpected partial files; do not
remove existing user data automatically.

For every candidate episode the publisher checks its hash, CAD order,
transforms and all resulting 512-point arrays, plus exact raw episode/frame
indices. The old GT transforms and visibility masks in the NPZ are never
loaded into the sidecar. To bind previously unrecorded cache provenance, it
also compares **every current raw RGB-D frame** after the V2 resize against its
cached RGB-D counterpart, and records hashes of the original video and depth
Parquet. This is read-only but may take substantial time on the full 112,939
frames. It replays two CAD queries for sampled cached RGB-D frames with the
actual checkpoint. The current model script must match the checkpoint's
recorded script SHA; synchronize the exact training script if it differs.

The native mask means `finite_prediction_not_visibility`; it is true for a
finite prediction even if an object is occluded or absent. `source=1` indicates
inference and `confidence=0` means no calibrated confidence is available. The
model does not estimate visibility. These fields must not be interpreted as
measured object presence or detection confidence.

Training validates the checkpoint schema and SHA, CAD and cache configuration,
raw data and native shard hashes, task/role mapping and complete frame indices.
It will read source files to check hashes at startup. The cache and checkpoint
must remain accessible. The model predicts `base_link` from raw RGB-D; its
original camera calibration provenance is **not recorded** in the candidate.
The spatial sidecar's own configured calibration remains checked, but this
does not prove geometric alignment between V2 predictions and Spatial V1.
Review the available projected predictions before training. V2 online policy
inference is not implemented by this training integration and fails explicitly.
