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

Button uses ordered CAD-to-current-Geometry rigid fits for every consumed frame;
it never truncates the old PoseRecord. Peg's differing historical color
intrinsics require the explicit `--peg-legacy-intrinsics` option; the historical
metadata is retained and the difference requires human projection review.

`preflight.py --quick` inspects the first episode's index and three selected
frames per task. It loads the three RGB-D samples into the model, and shows
phase timings and progress. The default per-task timeout is 120 seconds
(`--timeout-seconds` adjusts it). It does not recompute full source SHA256:
recorded hashes and unsampled frames are unverified, including Button's known
historical mismatch. Sample agreement is not proof of source identity or
geometric accuracy. Peg is blocked unless ordered CAD provenance and sampled
fixed-point fits pass. During training every consumed published pose is
compared against saved ordered geometry; mismatches stop the run.

The original full audit is available without `--quick` (default per-task
timeout 3600 seconds), and compares every frame for each mismatched pose SHA.
It is not automatically run at training startup. Full audit is expensive:
it hashes CAD and pose files, reads all episode indices, and scans the complete
geometry of a hash-mismatched episode. Quick stage times separate manifest/index,
CAD/calibration, three-frame correspondence and RGB-D/model initialization;
worker wall time also includes dependency startup. No source is modified.

```bash
python scripts/cad_pose/preflight.py --quick --peg-legacy-intrinsics --repo . --output output/cad_pose_audit/quick.json
python scripts/cad_pose/visualize_gt.py --repo . --output-dir output/cad_pose_audit/gt \
  --episodes 1018_peg_all=23,65
```

GT visualization draws the existing Geometry v1 in RGB and RGB-D; it is a
candidate label, **not a prediction or automatic acceptance**. Other tasks
continue even if one dataset is unavailable. Override any episode selection
with e.g. `--episodes 1028_press_button=0,1`. Images include source and
calibration provenance in the HTML report. Missing geometry/calibration must
be investigated rather than replaced with invented pose labels.

Only after inspecting the Peg RGB-D projections and verifying the legacy
adapter, create a review JSON with `status`, `reviewer`, `reviewed_frames`,
`geometry_manifest_sha256` (compute with `sha256sum` on Peg geometry manifest) and
`profile_sha256` from the quick report; status must be
`accepted`, and at least three actual frame identifiers must be reviewed. The
review must reflect a real human decision, not a fabricated approval. If Peg
cannot prove ordered correspondence, four-task training fails closed; use
`--tasks` to independently smoke-test other tasks.

```bash
# Training repeats the per-task quick check; add --full-audit for the full scan.
python scripts/cad_pose/train.py --repo . --peg-legacy-intrinsics --device cuda:0 --batch-size 32 \
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
