# Level-1 object-geometry workflow

These wrappers preserve the tested prototype workflow while keeping the full
SAM2/data dependency outside the formal package. The source-of-truth snapshot
is in `tools/hoi_level1_snapshot/scripts/`.

```bash
bash scripts/object_geometry/level1_ep.sh EP prepare
bash scripts/object_geometry/level1_ep.sh EP preview
bash scripts/object_geometry/level1_ep.sh EP approve
bash scripts/object_geometry/level1_ep.sh EP track
bash scripts/object_geometry/level1_ep.sh EP review-track
bash scripts/object_geometry/level1_ep.sh EP build
bash scripts/object_geometry/level1_ep.sh EP memory
bash scripts/object_geometry/level1_batch_preview_all.sh
```

Set `HOI` and `REPO` when the prototype or formal repository is in a different
location. The wrappers do not copy datasets, masks, videos, checkpoints, or
generated outputs into Git.

## Formal front-calibrated stride-1 output

Do not overwrite the existing prototype `cloud_frames` or `abc_v1`. For a
reviewed episode, generate the formal sidecar source with the separate run
names below. The build and memory steps use the same profile as
`pi0_xhand_spatial_structured_suffix_front_relational`.

```bash
REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
HOI=/workspace/mnt/sqzhang26/hoi
OUT="$HOI/outputs/level1_sequence_e23"
PROFILE=configs/ur7e_xhand/front_calibration_candidate.json

cd "$REPO"
PYTHONPATH=src .venv/bin/python tools/hoi_level1_snapshot/scripts/level1_sequence_multi.py build \
  --repo "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 --output "$OUT" \
  --stride 1 --run-name front_candidate_stride1 --front-calibration-profile "$PROFILE"

PYTHONPATH=src .venv/bin/python tools/hoi_level1_snapshot/scripts/level1_contact_memory_multi.py \
  --repo "$REPO" --dataset "$REPO/data/1018_peg_all" --episode 23 --output-root "$OUT" \
  --input-dir "$OUT/cloud_frames_front_candidate_stride1" \
  --source-html "$OUT/sequence_3d_front_candidate_stride1.html" \
  --source-sequence-report "$OUT/sequence_report_front_candidate_stride1.json" \
  --run-name abc_v1_front_candidate_stride1 --front-calibration-profile "$PROFILE"
```

After every requested episode is regenerated, export all of them together with
the following command. It writes the separate
`object_geometry/v1_front_candidate` directory and records the profile in its
manifest.

```bash
PYTHONPATH=src .venv/bin/python scripts/object_geometry/export_sidecar.py \
  --repo-root "$REPO" --front-calibration-profile "$PROFILE" \
  --hoi-output-root "$HOI/outputs" --output-root "$REPO/data/1018_peg_all" \
  --episodes 0-69 --cloud-run-name front_candidate_stride1 \
  --memory-run-name abc_v1_front_candidate_stride1 --version v1_front_candidate \
  --generation-commit "$(git rev-parse HEAD)"
```

## Hand-Scene V0 derived sidecar

`export_hand_scene_sidecar.py` never reruns pose estimation or overwrites the
formal front-calibrated geometry.  It copies the calibrated `[512,3]`
`base_link` clouds into a separate `object_geometry/v1_hand_scene` directory
and joins producer-owned causal relation/grasp state records.  The full input
schema, no-future-frame rule, and training smoke command are in
[Hand-Scene V0 contract](../../docs/object_geometry/hand_scene_v0.md).

## Sparse 3-D keypoint pilot

`pilot_sparse_3d_relations.py` is a separate lightweight prototype. It reads
the reviewed `funnel/` and `target_flask/` PNG masks, never runs SAM2, and
requires a small manual RGB initialization so it does not pretend a mask alone
identifies a funnel tip or bottle mouth. The initialization contains
`initial_frame` plus `shaft_end`/`mouth_center` and `axis_reference` pixel
coordinates for each object. The script writes causal per-frame
`keypoints.jsonl`, RGB overlays, `keypoints_3d.html`, `report.json`, and timing
statistics. Predicted points during occlusion are explicitly `valid=false`.

Initialization template (replace every `null` with reviewed RGB pixels; these
are deliberately not fabricated here):

```json
{
  "schema_version": 1,
  "objects": {
    "funnel": {
      "initial_frame": 212,
      "keypoints": {"shaft_end": [null, null], "axis_reference": [null, null]}
    },
    "target_flask": {
      "initial_frame": 0,
      "keypoints": {"mouth_center": [null, null], "axis_reference": [null, null]}
    }
  }
}
```

Example e23 invocation on computer 4 (fill in the two manual anchors in the
JSON first):

```bash
REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
HOI=/workspace/mnt/sqzhang26/hoi
python "$REPO/scripts/object_geometry/pilot_sparse_3d_relations.py" \
  --repo-root "$REPO" \
  --dataset "$REPO/data/1018_peg_all" \
  --episode 23 \
  --mask-root "$HOI/outputs/level1_sequence_e23/masks" \
  --init-json "$HOI/outputs/level1_sequence_e23/keypoint_init_e23.json" \
  --front-calibration-profile "$REPO/configs/ur7e_xhand/front_calibration_candidate.json" \
  --output "$HOI/outputs/level1_sequence_e23/sparse_3d_relations_front_candidate"
```
