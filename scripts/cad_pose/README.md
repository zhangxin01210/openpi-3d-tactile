# CAD Pose Transformer V1

The existing 648,457-parameter model is unchanged. Training, evaluation,
prediction visualization and sidecar export are implemented. Real data and
model accuracy have **not** been validated on this machine. Source hash checks,
ordered point agreement and synthetic tests do not establish geometric accuracy.

All commands below run from the repository root. For environments lacking
`trimesh`, prefix Python commands with:

```bash
uv run --no-sync --with 'numpy==1.26.4' --with 'trimesh==4.12.2' python
```

`preflight.py` is read-only and reports each task separately. It deliberately
exits nonzero until geometric review is performed. When a PoseRecord hash
mismatches, it compares **all** ordered geometry points and validity masks;
no provenance is rewritten. Legacy Peg needs provable sampler/mesh/calibration
provenance and rigid-fit residuals. Unordered ICP is never used.

```bash
python scripts/cad_pose/preflight.py --repo . --output output/cad_pose_audit/preflight.json
python scripts/cad_pose/visualize_gt.py --repo . --output-dir output/cad_pose_audit/gt \
  --episodes 1018_peg_all=23,65
python scripts/cad_pose/smoke.py --repo . --device cpu --output output/cad_pose_audit/smoke.json
```

GT visualization draws the existing Geometry v1 in RGB and RGB-D; it is a
candidate label, **not a prediction or automatic acceptance**. Other tasks
continue even if one dataset is unavailable. Override any episode selection
with e.g. `--episodes 1028_press_button=0,1`. Images include source and
calibration provenance in the HTML report. Missing geometry/calibration must
be investigated rather than replaced with invented pose labels.

Only after inspecting the Peg RGB-D projections and verifying the legacy
adapter, create a review JSON with `status`, `reviewer`, `reviewed_frames`,
`geometry_manifest_sha256` and `profile_sha256` from the audit; status must be
`accepted`, and at least three actual frame identifiers must be reviewed. The
review must reflect a real human decision, not a fabricated approval. If Peg
cannot prove ordered correspondence, four-task training fails closed; use
`--tasks` to independently smoke-test other tasks.

```bash
python scripts/cad_pose/train.py --repo . --device cuda:0 --batch-size 32 \
  --steps-per-epoch 1000 --epochs 100 --peg-review /path/to/peg_review.json \
  --output-dir output/cad_pose_runs/run1
# Resume with --resume output/cad_pose_runs/run1/checkpoints/epoch_....pt and the same arguments.
python scripts/cad_pose/evaluate.py --repo . --device cuda:0 \
  --checkpoint /path/to/best_epoch.pt --output-dir output/cad_pose_runs/eval1
python scripts/cad_pose/visualize.py --repo . --device cuda:0 \
  --checkpoint /path/to/best_epoch.pt --evaluation-dir output/cad_pose_runs/eval1 \
  --output-dir output/cad_pose_runs/predictions1
```

Synthetic execution requires `--synthetic` on train/evaluation/visualization;
synthetic checkpoints cannot be deployed as real checkpoints. Splits are by
episode, task sampling is balanced, `best.json` records the best checkpoint,
and epoch checkpoints are never overwritten. Loss and metrics use identity
symmetry unless verified per-object rotation symmetries are explicitly
provided via `--config`; ADD-S is diagnostic and does not certify symmetry.
Latency separates preprocessing and forward from disk/video decoding.

The optional `export_sidecar.py` consumes only raw RGB-D and a real, accepted
pose checkpoint. Its `--acceptance` JSON must contain `status: accepted`,
`reviewer` and `checkpoint_sha256`, matching actual review of this checkpoint.
It writes a new version and refuses to overwrite existing geometry. The new
Pi0 configurations require the same accepted pose checkpoint; the policy
checkpoint stores sidecar provenance and the online bridge verifies it.
The existing deployment client does not yet provide all required input; no
physical deployment has been validated. A finite prediction is not a detection
confidence or proof of accuracy.
