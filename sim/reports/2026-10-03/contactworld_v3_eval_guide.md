# ContactWorld USB V3: paired 40-seed evaluation

This evaluates the 13 `pi0_cw3_usb_*` checkpoints on fresh random USB resets.
Each model sees the same seed schedule `10000..10039`, 180 simulator actions,
one policy query per four actions and the native simulator control period. The
policy always receives the two RGB cameras and the native 18D joint position
and velocity state. Depending on its training config it also receives the same
fused 4096-point robot-base cloud and/or ordered left and right 140-taxel
force fields used in V3 training. The force source, coordinate basis and token
encoder are selected by each saved training config. The third π0 image slot
remains masked. No tactile preview MP4 is a model input.

The runner uses the isolated **patched bilateral** ContactWorld checkout and
starts a new simulator process for every seed. It checks the transferred
Orbax checkpoint size against the other configurations, and refuses a partial
download. It resumes completed seeds on the same command. The 13th checkpoint
can be skipped temporarily with `--skip-incomplete`; rerunning the same
command when its transfer finishes evaluates only the missing model/seeds.

## Commands

From `/home/sai/zx/openpi-3d-tactile`:

```bash
# Check transfers and simulator source; no GPU rollout.
.venv/bin/python sim/scripts/run_contactworld_pi0_v3_matrix.py --check-only

# Optional short end-to-end check of one spatial model (separate output).
.venv/bin/python sim/scripts/run_contactworld_pi0_v3_matrix.py \
  --configs pi0_cw3_usb_04_fused_pc_local_map \
  --output /home/sai/zx/openpi-sim-runtime/runs/cw3_eval_first_seed \
  --trials 1

# First pass: one complete 180-step seed for every checkpoint. This uses the
# same output root and protocol as the 40-seed run.
.venv/bin/python sim/scripts/run_contactworld_pi0_v3_matrix.py \
  --trials 40 --gpu 0 --limit-this-run 1

# Full early-stage matrix; resumes at seed 10001 after the first pass.
# Rerun this exact command after any interruption.
.venv/bin/python sim/scripts/run_contactworld_pi0_v3_matrix.py \
  --trials 40 --gpu 0

# After all 13 finish; works earlier with --allow-partial.
.venv/bin/python sim/scripts/analyze_contactworld_pi0_v3_matrix.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
```

The first seed check is optional because the evaluation code was also
exercised locally with an input-only simulator reset, two separate two-step
policy rollouts (map and structured force encoders), and one complete 180-step
map-model rollout. The separate two-step rollouts had bitwise identical
initial RGB, cloud, taxel positions and force fields at the same seed. The
full rollout saved 45 pre-action queries and 180 post-action force fields.
To try just one seed of every checkpoint before the 40-seed run,
use the first-pass command with `--limit-this-run 1`; the subsequent full command
resumes at seed `10001` for those models.

## What is saved

Under `.../contactworld_pi0_v3_13models_40trials_20261003/`:

| File | Purpose |
|---|---|
| `<config>/summary.json` | Resume state, checkpoint metadata hash, fixed protocol, per-seed source-ever/final success, first success and longest success streak. |
| `<config>/seed_<seed>/front_wrist.mp4` | Full dual-camera rollout, including initial frame. |
| `<config>/seed_<seed>/trajectory.json` | Every action, prediction latency at replans, EE/plug/socket poses, source and geometric errors, plug/socket pair contact, gripper width, both pads' activity and force summaries. |
| `<config>/seed_<seed>/inputs.npz` | Every **pre-action** policy query: exact RGB/state, fused cloud, source RGB-D, camera view/projection matrices and cloud pixel provenance, left/right taxel base positions, local and base forces, 16-action predicted chunks; plus every **post-action** bilateral local force grid. All modalities are archived for diagnosis even when a model does not consume them. |
| `<config>/index.html` | Per-model video browser. |
| `paired_diagnostics.csv`, `analysis.json` | Per-seed event metrics and paired comparisons for the planned hypotheses. |
| `case_index.html` | Same-seed videos together, sorted to expose model disagreements. |

`inputs.npz` uses pad order **left, right** and `140` row-major taxels per
pad. For map models, `force_local` is the original `[normal, shear_x,
shear_y]` in simulator units. For base models, `force_base` is signed
`[Fx,Fy,Fz]` in the robot-base frame. The recorded `cloud_xyz` is the fused
4096-point robot-base input, with no hidden plug/socket ground truth. The
checkpoint's normalization and spatial encoder run in the policy server.

## First analysis pass

1. Compare `source_ever`, `source_final`, first success step and longest
   success streak. Source success is the simulator's keypoint threshold, not
   an independent physical insertion certificate. There is no invented
   ten-step hold test in the primary score.
2. For each preplanned contrast (01→02 geometry, 01→03 tactile, 02/03→04
   complementarity, 04→05 contact distribution, 04→06 structured 3D force,
   06→07 grouping, 04→08 coordinate basis, 08/09/10 channel split versus
   token budget, 06→11/12 route, 04→13 token budget), inspect **paired seed
   gains and losses** before reading aggregate rates. Forty seeds are an
   early diagnostic sample; confidence intervals will be wide.
3. For disagreed seeds, align videos and traces by seed and simulator step.
   Mark first close approach, first plug/socket contact, best post-contact
   keypoint error, any improvement after contact, transient source success,
   and changes in EE-to-plug distance. Use the archived raw force grids to
   check whether left/right contact position and shear differ before a
   corrective action. Comparing actions at the same seed and frame also
   helps distinguish different approach plans from contact-phase behavior.
4. Stratify **exploratorily** by initial target offset and initial alignment.
   Large modality differences should be checked for concentration in
   specific difficulty bands; the 40-seed sample does not justify narrow
   claims about each band. A later controlled left/right-offset recovery set
   is needed to test directional tactile correction causally.

The event labels in the analyzer are review aids. Contact-force thresholds,
grip-distance change and pad asymmetry are proxies, not verified explanations
for a failure. The source task's success predicate is the reported outcome.

## Training-neighbor, contact-phase and target-perturbation audits

The train-only nearest-neighbor report is at
`.../contactworld_pi0_v3_13models_40trials_20261003/training_neighbors/index.html`.
It uses the 95 V3 training episodes, not the validation/test episodes. Run:

```bash
.venv/bin/python sim/scripts/analyze_contactworld_training_neighbors.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
.venv/bin/python sim/scripts/render_contactworld_training_neighbors.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
.venv/bin/python sim/scripts/analyze_contactworld_contact_phase.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
.venv/bin/python sim/scripts/plot_contactworld_phase_timeline.py
.venv/bin/python sim/scripts/render_contactworld_contact_phase.py
```

The contact-phase report is under `training_neighbors/phase_audit/index.html`.
Action direction matches the nearest training demonstration most closely during
approach, but starts diverging several steps before measured contact and
diverges further afterward. This also occurs in successful runs. The historical
40-seed matrix remains the primary evaluation; this is a diagnostic analysis.

The isolated causal probe changes only the USB socket actor's XY root position
by ±3 mm after a seeded reset. It uses explicit identical diffusion noise at
each query step within a policy/seed pair. Target motion changes its RGB and
cloud observations; initially both finger force fields are identical. It
saves 180-step videos, full trajectories and query inputs. To resume the
completed 20-trial set or regenerate its report:

```bash
.venv/bin/python sim/scripts/run_contactworld_target_perturb.py
.venv/bin/python sim/scripts/analyze_contactworld_target_perturb.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_target_perturb_20261003
.venv/bin/python sim/scripts/render_contactworld_target_perturb.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_target_perturb_20261003
```

The probe report is at `.../contactworld_target_perturb_20261003/analysis/index.html`.
Its fixed diffusion noise differs from the earlier 40-seed evaluation, so
compare only conditions inside this probe. Twenty trials cover two initial
states, RGB and model 13, and five target positions per model/state. They are
a mechanism check, not a new success-rate estimate.

## Training-neighbor ensembles and rolling observation retrieval

The review page is at
`.../contactworld_pi0_v3_13models_40trials_20261003/training_neighbors/ensemble_audit/index.html`.
It compares the policy with the nearest 1, 3, 5, and 10 training demonstrations,
using uniform and distance weights. The fixed-initial analysis time-aligns
actions and EE/plug trajectories around first contact. The rolling analysis
retrieves again at every saved policy observation (one every four simulation
steps), using either current full geometry or both current camera images
reduced to 16×16 LAB with front-view and task-ROI weighting. The top k
candidates must come from different training
episodes; each contributes the next four executed translation actions.
Random ten and sources fixed at the initial observation are controls.

```bash
OPENBLAS_NUM_THREADS=4 python3 sim/scripts/analyze_contactworld_neighbor_ensemble.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
OPENBLAS_NUM_THREADS=4 python3 sim/scripts/analyze_contactworld_dynamic_neighbors.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
OPENBLAS_NUM_THREADS=2 python3 sim/scripts/analyze_contactworld_perturb_dynamic_neighbors.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_target_perturb_20261003 \
  --library /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/training_neighbors/ensemble_audit/train_frame_library_roi.npz
python3 sim/scripts/render_contactworld_neighbor_ensemble.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003
```

The rolling analysis uses 5,064 valid frames from 95 train episodes, 40 test
initial states, and all 13 models. It compares 5,195 pre-contact and 17,811
post-contact policy observations across 520 trajectories; 7 trajectories had
no measured contact. Initial geometry retrieval is not based on the target
position alone: it also includes plug pose relative to socket and plug pose
relative to gripper. The current-frame image retrieval is a low-resolution
pixel proxy, not the pi0 visual representation.

| Retrieval and phase | Single nearest action cosine | Mean of nearest 10 | Random 10 | Fixed initial 10 sources |
| --- | ---: | ---: | ---: | ---: |
| Geometry, before contact | 0.812 | 0.870 | 0.618 | 0.888 |
| Geometry, after contact | 0.605 | 0.726 | 0.708 | 0.729 |
| Two-camera ROI pixels, before contact | 0.719 | 0.824 | 0.612 | 0.800 |
| Two-camera ROI pixels, after contact | 0.602 | 0.761 | 0.710 | 0.750 |

These are diagnostic action similarities, not insertion success rates. Action
RMS error also decreases for nearest-ten averages. ROI pixel-nearest ten exceeds
random ten after contact by 0.051 cosine (seed-bootstrap 95% interval
0.041–0.061); its benefit over holding the initial ten source identities is
0.012 (0.001–0.023). For geometry after contact, nearest ten versus random
ten is 0.018 (−0.009–0.040), and versus fixed source ten is −0.002
(−0.027–0.017). Thus post-contact action averaging explains much of the
apparent gain; rolling visual matches add a smaller specific gain.

The isolated ±3 mm runs have 16 shifted policy/seed pairs. Their first action
chunk differs from the unshifted run at step 0 by over 0.1 mm RMS in all 16.
For current-frame train retrieval, no pair changes its geometry-nearest source
at the initial frame, while four model pairs (two distinct shifted states)
change their ROI pixel-nearest source immediately. A geometry-nearest source
switches later in all 16 pairs; an ROI pixel-nearest source switches later in
eight of the remaining 12. Later switches may result from already divergent
rollouts. In the earlier *initial demonstration only*
geometry ranking, one of eight unique shifted states changed its top source;
that is a different candidate pool from the rolling all-frame ranking.

This evidence supports the hypothesis that a local mixture of demonstrations
approximates the policy's motion better than one demonstration. It does not
show that pi0 internally retrieves and averages demonstrations. Geometric
retrieval uses simulator ground truth, contact alignment uses the observed
event, and the averaged actions are not guaranteed to be executable as a
closed-loop trajectory. See `pairwise_contrasts.json`,
`dynamic_observations.csv.gz`, and the paired test/train image gallery for
the auditable per-case evidence.

## Overnight learned-neighbor-weight experiment

Start from `/home/sai/zx/openpi-3d-tactile` with one command:

```bash
bash sim/scripts/launch_contactworld_neighbor_night.sh
```

This launches a detached process, prints its PID, and continues after closing
the terminal. The launch log is
`/home/sai/zx/openpi-sim-runtime/runs/contactworld_neighbor_night_20261003.launch.log`.
The final report is
`/home/sai/zx/openpi-sim-runtime/runs/contactworld_neighbor_night_20261003/index.html`.
The full run has **not** been started by the code validation. Its separate
`..._smoke` output is a small engineering check, not a scientific result.

```bash
# Optional preflight; does not train or infer.
bash sim/scripts/launch_contactworld_neighbor_night.sh --check-only

# Main progress (start/end of each stage).
tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_neighbor_night_20261003.launch.log

# Detailed epoch progress after the training stage starts.
tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_neighbor_night_20261003/logs/train.log
```

The launcher uses the existing `.venv`. Frozen feature extraction and the small
weight network explicitly use CPU (8 threads), because this environment's
Torch CUDA kernels do not support the local RTX 5090. Shared pi0 inference
uses the existing JAX GPU path. No environment upgrades, downloads, simulator
rollouts, or pi0 training are performed. GPU execution is sequential, with a
new process for each pi0 checkpoint. Closing the terminal is supported;
suspending or shutting down the computer pauses or stops computation.

The default complete job includes:

1. Cache all 5,064 valid train observation frames from the 95 train episodes.
2. Five outer folds by source episode, with a separate grouped inner validation
   split. Fit S (state), R (RGB+state), RP, RT, and RPT; select the fixed prior
   and learned model epoch using only the inner validation episodes. Refit on
   the 76 outer training episodes and predict the 19 held-out episodes.
3. Fit a final predictor per input variant on all 95 training demonstrations,
   using the median selected epoch count/temperature from the folds.
4. Predict the next four executed actions at every saved policy observation in
   the existing 520 rollouts, retaining source/frame IDs, candidate distances,
   weights, uncertainty, predictions and residuals. Each observation is fed to
   every neighbor input variant for a paired comparison.
5. Select six evenly spaced observations from each RGB and model-13 rollout:
   normally 480 shared observations from 40 seeds and two trajectory origins.
   All 13 pi0 checkpoints infer on these exact observations, with identical
   explicit diffusion noise per observation, producing 6,240 policy queries.
   This is observation pairing, not a new success-rate or counterfactual
   closed-loop evaluation. It compares output commands before simulator action
   clipping, against the neighbor predictor of executed demo commands.
6. Write the report, episode-level bootstrap contrasts, per-frame diagnostics,
   uncertainty bins, per-rollout correspondence, and shared-input policy pairs.

The learned predictor is deliberately restricted:

```text
current observation + candidate observation
    -> shared MLP (64, 32, 1), observation features only
    -> bounded logit correction 2*tanh(MLP)
    -> softmax(similarity_logit + correction)
    -> weighted sum of the candidates' four-step, six-dimensional actions
```

The final MLP layer is initialized to zero, making initial weights exactly
the similarity prior. A KL penalty of 0.01 discourages unnecessary departures
from that prior. Candidate actions enter only the final weighted sum; query
target actions never enter the network. One weight is used per candidate
four-step chunk. The selected ten frames must come from different source
episodes. Training queries exclude **their entire source episode**, including
other frames in that episode. The loss standardizes action components using
training-only statistics and weights the first action by 0.75 and the next
three by 0.25. No predictor creates an action outside the candidates' convex
combination; a low score may therefore reflect candidate coverage limits.

Both `initialized` and `learned` use the same ten candidates and temperature.
The stronger `fixed` comparator chooses k in {1,3,5,10}, equal weights or
distance softmax temperatures {0.5,1,2} on inner validation. Epoch zero is
eligible for selection if weight learning overfits. Additional controls are
one nearest frame, an episode-balanced mean action, and the state-only model.

Feature extraction uses an already-local, frozen ImageNet ResNet18 checkpoint
at `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`, not any of the
ContactWorld-finetuned visual encoders. Each view retains a 4×4 pooled feature
grid, then uses a fixed label-independent projection to 128 dimensions. Both
views total 256 dimensions. Cloud features use continuous bilinear occupancy
on the XY/XZ/YZ planes (64×64 each) in the reviewed robot-base ROI, projected
to 128 dimensions. Tactile features retain finger/taxel ordering and combine
base taxel XYZ, signed-log base forces and signed-log local forces before a
fixed 128-dimensional projection. State is the same 18-dimensional recorded
joint-position/velocity vector. Scaling uses training-fold statistics only.
These descriptors are diagnostic proxies, not the same encoders as the 13 pi0
models. Failure to benefit from a modality here does not prove it is useless.
The point-cloud projection preserves continuous sub-bin changes but is not
an exact reconstruction of the original 4,096-point geometry.

Only the exported LeRobot data and its spatial sidecars are required for
training. Source replay logs are read only for contact-phase **evaluation**
labels; if absent, phase is marked unknown and overall errors remain usable.
No plug/socket ground-truth pose, contact label, time index, source ID or
future state is used as a prediction feature.

Every stage/model has its own log under `logs/`. Successful stages, folds,
features and individual shared pi0 queries are retained. Re-running the
**same command** resumes incomplete work. The output directory records code,
dataset manifest, visual weights, evaluation summaries and checkpoint
manifest hashes. If these change, the runner asks for a new `--output` rather
than mixing incompatible results. A second simultaneous runner for the same
output exits under a file lock. A failed shared pi0 checkpoint does not prevent
the remaining checkpoints and partial report from completing; the launch
log and report identify failed stages.

Useful optional controls:

```bash
# Foreground execution.
.venv/bin/python -u sim/scripts/run_contactworld_neighbor_night.py --device cpu

# Different experiment directory or a reduced input matrix.
bash sim/scripts/launch_contactworld_neighbor_night.sh \
  --output /home/sai/zx/openpi-sim-runtime/runs/contactworld_neighbor_custom \
  --modes R RP RT RPT

# Small, separate engineering check: 20 demos, one outer fold, 2 epochs,
# R and RPT, one recorded trajectory, two shared snapshots, RGB pi0 only.
.venv/bin/python -u sim/scripts/run_contactworld_neighbor_night.py --smoke --device cpu
```

Tomorrow's main files are `analysis.json`, `cv_episode_metrics.csv`,
`cv_frame_diagnostics.csv`, `modality_pairs.csv`, `uncertainty_bins.csv`,
`rollout_correspondence.csv`, `shared_observation_correspondence.csv`, and
`shared_policy_pairs.csv`. Detailed weights and candidate source/frame IDs are
in `cv/*/fold_*/predictions.npz` and `rollout_predictions/*/*.npz`.
