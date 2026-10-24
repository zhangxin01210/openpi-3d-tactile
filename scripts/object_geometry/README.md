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
