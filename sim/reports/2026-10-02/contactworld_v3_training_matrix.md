# ContactWorld USB V3: first-round matrix

This matrix uses `data/contactworld_usb_v3_bilateral`: 120 accepted replayed
episodes, split by source episode into 95 train, 15 validation and 10 test.
All 13 runs use the same RGB, robot state, executed-action targets, π0 base
weights, action horizon 16, normalization and optimizer schedule. The third
π0 image slot remains masked. The two tactile preview MP4s are rendered from
the force fields for review; **they are never used as tactile RGB training
inputs**. Genuine paired raw tactile RGB and deformation depth are not in V3.
The robot state is 18 values: seven Franka arm positions, two gripper
positions, followed by their nine velocities. The six action targets are
executed end-effector translation and axis-angle rotation displacement
commands; the simulator separately applies a fixed gripper-close target.
They are not six joint-position targets.

## Inputs and tokenizers

- **R:** front and wrist 224×224 RGB enter π0's existing visual encoder.
- **P:** fused front/wrist depth produces the stored 4096 robot-base XYZ points.
  The XHand `StructuredSpatialEncoder` selects 16 FPS centers, gathers 32
  neighbors per center, and emits 16 local plus one global 128D tokens. Point
  RGB is absent (`rgb_valid=false`); RGB comes from the two cameras.
- **Local coupled force maps:** one signed 10×14×3 map per pad, with source
  channels `[normal, shear_x, shear_y]`. A dedicated two-block, four-head
  tactile ViT embeds 35 non-overlapping 2×2 patches and one CLS token per pad.
  It outputs **two tactile tokens**; the same ViT weights are used on left and
  right, with distinct pad embeddings. A projected robot-base pad centroid
  supplies a coarse spatial anchor. Configs 03–04 use this representation.
- **Coarse summary:** for each pad, 13 numbers: signed mean, mean absolute,
  RMS and maximum absolute value of each of the three *local* force channels,
  plus the fraction of taxels with force norm above `1e-4`. A two-layer MLP
  outputs one token per pad, so config 05 has the same tactile token count as
  04. As in 04, the normalized robot-base pad centroid is projected into each
  token, preserving coarse pad location while removing within-pad spatial
  contact distribution. A poor 05
  result is not by itself proof that spatial distribution was the sole cause;
  the smaller input/encoder also changes capacity.
- **Structured 3D force:** each pad has 140 pairs of robot-base taxel XYZ and
  signed base Fx/Fy/Fz, plus force norm and finger ID. The XHand structured
  encoder uses an 8-neighbor taxel graph and produces two local and one
  summary token **per finger**, six in total. Config 07 removes finger IDs and
  pools all 280 taxels in one KNN/query group, with four local queries plus
  two summary queries. Configs 06 and 07 thus both output six tactile tokens,
  including the same four-local/two-summary query budget.
- **Base coupled force maps:** left and right each supply one 10×14×3 map of
  base Fx/Fy/Fz. Config 08 uses exactly the same map ViT architecture and
  two-token budget as 04; its force coordinate basis is the main change.
- **Base split force maps:** config 09 uses separate Fx, Fy and Fz ViTs, each
  shared across the two pads: six single-channel input maps and six output
  tactile tokens. Config 10 is its capacity control: one coupled 3-channel
  ViT per pad with three learned output queries, also six tokens. The coupled
  ViT width is 224; its map-encoder parameter count is 849,088 versus
  811,776 for the three independent 128-wide ViTs (about 4.6% difference).
  Comparing 09 to both 08 and 10 helps separate channel organization from
  extra tokens and capacity.
- **Local-map token-budget control:** config 13 reuses 04's two local
  normal/shear-X/shear-Y maps, patching, two-block 128-wide ViT and suffix
  route, but changes one learned CLS output query per pad to three. It emits
  six tactile tokens instead of two and tests whether 04 compresses each
  pad's contact distribution too aggressively.
- **Scaling:** every force channel is divided by fixed 0.002 simulator units
  without clipping. Across train episodes, nonzero absolute base-force
  components have approximately 0.0016 at P99. These are uncalibrated TacSL
  forces, not Newton measurements.

## Thirteen configurations

Each name below has the `pi0_cw3_usb_` prefix. “Spatial tokens” exclude the
ordinary RGB/language tokens.

| ID | Name suffix | Additional inputs and route | Spatial tokens | Main comparison |
|---:|---|---|---:|---|
| 01 | `rgb` | R only | 0 | visual reference |
| 02 | `fused_pc` | R+P, suffix | 17 | 01: value of local 3D geometry |
| 03 | `rgb_local_map` | R+two local coupled force maps, suffix | 2 | 01: value of full bilateral tactile fields |
| 04 | `fused_pc_local_map` | R+P+two local coupled force maps, suffix | 19 | 02/03: geometry-contact complementarity |
| 05 | `fused_pc_local_summary` | 04 with per-pad 13-value summary | 19 | 04: coarse strength versus contact distribution |
| 06 | `fused_pc_base_structured` | R+P+structured base XYZ/Fx/Fy/Fz, suffix | 23 | 04/08: explicit spatial force representation |
| 07 | `fused_pc_base_pooled` | 06 with pooled fingers, same six tactile tokens | 23 | 06: explicit left/right grouping |
| 08 | `fused_pc_base_map` | 04 with coupled base Fx/Fy/Fz maps | 19 | 04: coordinate basis, same map architecture |
| 09 | `fused_pc_base_map_split` | 08 with three independent single-axis ViTs | 23 | 08/10: channel separation |
| 10 | `fused_pc_base_map_matched` | coupled base maps, six outputs, near-matched map parameters | 23 | 09: token/parameter capacity control |
| 11 | `fused_pc_base_structured_prefix` | identical input/encoder to 06; P and T to VLM prefix | 23 | 06: injection location |
| 12 | `fused_pc_base_structured_split_route` | identical input/encoder to 06; 17 P tokens to prefix, six T tokens to action suffix | 23 | 06/11: modality-specific routing |
| 13 | `fused_pc_local_map_3tok` | 04's local coupled maps with three CLS queries per pad | 23 | 04: tactile token bottleneck |

11 and 12 only change routing relative to 06. For 12, selected tokens are
compacted to 17 prefix and six suffix tokens; the other modality is not
duplicated as masked placeholders. The other spatial configurations inject
their tokens into the action-expert suffix. The map and structured tactile
branches do not occupy the third RGB slot.

The old proposed raw tactile RGB, tactile deformation depth and pretrained
tactile encoder experiments remain outside this round because V3 does not
contain genuine bilateral raw RGB/depth or matching pretrained weights.
Point-cloud noise should first be an evaluation condition applied to these
same checkpoints; a noise-trained model becomes worthwhile if clean-trained
models materially degrade under a fixed perturbation protocol.

## Interpretation and evaluation gates

- The 95 training episodes produce 5,064 valid 16-step windows. With batch 8,
  20,000 steps are about 31.6 passes over these windows. The cosine learning-rate
  decay also runs to 20,000 steps. Keep the 5k/10k/15k and final 19,999-step
  checkpoints and compare paired validation rollouts; the training loss alone
  is not a model-selection criterion. Extend selected models only if their
  validation performance or learning curves justify more updates.
- The 120 exported episodes are successful replays. They do not supply a
  controlled set of left/right offset recoveries. A tactile gain on ordinary
  insertion cannot by itself establish directional jam recovery; that claim
  needs paired offset/contact diagnostics after this first round.
- The existing `eval_contactworld_pi0_matrix.py` is for V1 right-finger and
  1,024-point inputs. It must be adapted to V3 bilateral fields and the same
  fused 4,096-point online capture before comparing these checkpoints. Use
  the same held-out resets, action execution/replanning schedule and source
  success predicate for every configuration, and preserve trajectories,
  contact/force traces and failure cases.
- Config 05 retains the pad centroid but changes a ViT force map to a smaller
  statistics MLP; its comparison with 04 is an informative first pass, not an
  isolated proof of the value of within-pad spatial distribution. Config 07
  removes explicit finger IDs but retains base XYZ, so side can still be
  inferred from position. Config 09/10 matches tactile token count and map
  encoder parameter count to within 5%, not identical computation. Config 13
  retains 04's force input and ViT backbone but increases query/token count.

## Training preparation and launch

`assets/norm/norm_stats.json` was computed from all train-only, 16-step action
windows via `pi0_cw3_usb_01_rgb`. The data directory must be synchronized
alongside the changed code. The server's existing π0 base Orbax `params`
checkpoint is separate from the dataset; set the environment variable if it
is outside the launcher defaults.

```bash
cd /path/to/openpi-3d-tactile
export OPENPI_CONTACTWORLD_BASE_WEIGHTS=/absolute/path/to/pi0_base/params
uv run --no-sync python sim/scripts/launch_contactworld_v2_matrix.py \
  --version v3 --label cw3_round1 --gpus 0,1,2,3,4,5,6,7 --check-only
uv run --no-sync python sim/scripts/launch_contactworld_v2_matrix.py \
  --version v3 --label cw3_round1 --gpus 0,1,2,3,4,5,6,7
```

The launcher verifies all 120 episodes, then audits every config through the
actual normalized LeRobot/OpenPI input path before creating training jobs.
It schedules at most one model per listed GPU. Its log and checkpoint paths
require a new label for each run. Local data and model-input preflight passes;
full base-weight restoration and A800 first-step memory/time still require
the server checkpoint and hardware. The earlier 5-hour, one-token estimate
has not been measured for these local-token models.

`--exclude 01` (or `--exclude pi0_cw3_usb_01_rgb`) omits that configuration
from training; several IDs can follow the flag. `--configs` remains available
to list exactly the desired full config names. The preflight still audits all
V3 input configurations. The historical `pi0_cw_usb_01_rgb` checkpoint has
the same π0 RGB architecture but was trained on 117 original training episodes
with different normalization statistics; V3 uses 95 successful replayed
training episodes and new RGB/state/action rows. For a paired V3 comparison,
train `pi0_cw3_usb_01_rgb` again. Excluding it is a scheduling choice, not a
claim that the historical checkpoint is an equivalent control.

## Audit

`sim/scripts/audit_contactworld_v3_inputs.py` checks train-only selection,
native 18D state and 16×6 actions against their normalized/padded model inputs,
exact saved cloud and both force fields after real transforms, masked third
image slot, encoder shapes and masks, finger/channel isolation, pooled six
tokens, the three-query local-map control, route-invariant encoding of
06/11/12, and actual 17/6 route masks.
Its latest JSON output is `contactworld_v3_input_audit.json`.
