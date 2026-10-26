# DTCR v1

DTCR uses the existing 600-taxel `spatial/v1_front_candidate` sidecar and
the existing approved Mesh+6D object sidecar.  It does not rewrite either
sidecar or `object_geometry/v1_hand_scene_q2`.

The causal tactile state is generated in frame order with the shared
`CausalTactileEMA` (`tau_s=1.0` by default):

```bash
export PYTHONPATH="$PWD/src"
python scripts/object_geometry/precompute_tactile_ema.py \
  --dataset-root data/1018_peg_all \
  --spatial-version v1_front_candidate \
  --episodes 23,38,65 \
  --output-root scratch/object_geometry/1018_peg_all/dtcr_tactile_ema \
  --tau-s 1.0 --resume
```

Use those state files with the existing Hand-Scene exporter, writing a new
version:

```bash
python scripts/object_geometry/export_hand_scene_sidecar.py \
  --source-dataset-root data/1018_peg_all \
  --source-version v1_hand_scene_q2 \
  --state-root scratch/object_geometry/1018_peg_all/dtcr_tactile_ema \
  --output-dataset-root data/1018_peg_all \
  --version v1_hand_scene_dtcr_v1 \
  --episodes 23,38,65
```

The training configs are:

- `pi0_xhand_spatial_structured_suffix_hand_scene_oo_v1` (M2, 68 tokens);
- `pi0_xhand_spatial_dtcr_v1` (DTCR, 68 tokens);
- `pi0_xhand_spatial_dtcr_independent_v1` (independent tactile ablation).

On computer 4, after the new sidecar has passed its data/contract checks, run
a one-step GPU smoke test from the repository root.  The initial checkpoint
is the Pi0 base parameter tree (override the path with
`OPENPI_HAND_SCENE_BASE_WEIGHTS` if it is mounted elsewhere):

```bash
export OPENPI_HAND_SCENE_DATASET_ROOT=data/1018_peg_all
export OPENPI_HAND_SCENE_BASE_WEIGHTS=checkpoints/pi0_base/params
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  python scripts/train.py pi0_xhand_spatial_dtcr_v1 \
  --exp-name dtcr_v1_smoke --num-train-steps 1 --batch-size 1 --wandb-enabled false
```

Normal training uses the same command with a fresh experiment name and the
desired schedule.  The checkpoint loader starts from the base Pi0 tree; M0 or
M1 checkpoints are not required.

The EMA is a causal force-magnitude reference, not a calibrated grasp-force
estimate.  Missing samples do not update the EMA and invalidate the next
short-time delta.  Online preprocessing emits the same fields under
`spatial.causal_tactile`; the deployment object-geometry adapter should place
that record into the current frame's Hand-Scene input.
