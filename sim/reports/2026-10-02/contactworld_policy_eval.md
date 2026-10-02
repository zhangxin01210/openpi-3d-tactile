# ContactWorld USB direct-policy evaluation

> **2026-10-02 暂停原定 11 模型评测。**[点云采样审查](contactworld_cloud_sampling_audit.md)
> 发现 9 个点云配置共同使用未裁剪的稀疏点云。下方命令只在明确接受这一
> 局限并作探索性评测时使用；启动器现在要求 `--accept-sparse-cloud`。

## Question and scope

Compare eleven trained OpenPI policies on full USB insertion from fresh, paired
random resets. This is a direct-policy task test. It does not reproduce the
ContactWorld paper's CEM goal-reaching score or claim numerical comparability
with Table III.

The published ContactWorld code and paper provide a world-model planning
protocol. As of this review, we have not identified a published π0/VLA result
on the **same ContactWorld USB release** with a standard full-task termination
and action-replanning protocol. The related [ManiFeel benchmark](https://arxiv.org/abs/2505.18472)
does evaluate direct visuotactile policies for USB insertion in the same
Isaac Gym/TacSL task family. Its [public configuration](https://github.com/purdue-mars/manifeel)
uses 50 test cases, an eight-action execution chunk and up to 500 steps.
Differences in task assets, observations, training data and scoring mean its
numbers should not be copied into a ContactWorld comparison. We therefore
declare the following protocol and keep it fixed across modalities.

## Fixed protocol

- Models: `pi0_cw_usb_01_rgb` through `_11_pcnoise`, all step 19999.
- Preliminary pass: 40 seeds `10000..10039` per model. **Recreate the simulator for each trial**,
  then check the complete initial physical state across models by hash. A
  continuous-simulator run showed that the same random seed could leave
  different plug and robot states after prior rollouts, even though the socket
  target matched. That run is archived only for exploratory review and must
  not be used as a strictly paired comparison. Resets use the task's published
  randomization ranges, so this is in-distribution fresh-state evaluation,
  not a separate OOD set.
- One native simulator action step is `4 × 0.016667 s`, about 15 Hz. The video
  is encoded at the same rate. Each policy predicts 16 actions; execute four,
  then query again. Every model uses the same 180-action budget, about 12 s.
  The positive USB demonstration lengths range from 37 to 112 steps, median
  68; 180 permits substantially longer recoveries. A later 8-step/longer-budget
  sensitivity check can separate feedback frequency and time-limit effects.
- The policy receives front and wrist RGB plus joint position/velocity;
  geometry and force are added only when the config requests them. The visual
  point cloud uses the same released-cloud-to-base transform as training. The
  signed right-finger 10×14×3 force field and normalized taxel coordinates
  match the LeRobot sidecar. `_05_ff_summary` computes its summary inside the
  model from that field. `_06_ff_ee3d_proxy` transforms the grid and force by
  the current end-effector pose, matching training. `_07_tacrgb` and
  `_08_tacdepth` receive a third image in addition to the point cloud.
  `_11_pcnoise` receives independent 2 mm Gaussian xyz noise per query,
  matching its training configuration; both clean and perturbed clouds are
  saved. `_09_prefix` and `_10_split` share the full grid inputs of `_04`;
  their routing differs inside the network.
- Run all 180 steps unless a nonfinite state occurs. **No scripted zero-action
  hold and no early stop upon proximity.** The earlier 5+10-step strict hold was
  an exploratory diagnostic invented for the three-video pilot, not a
  ContactWorld or standard VLA success condition.

The environment's `_check_success()` is the mean plug/socket keypoint error
below `close_error_thresh = 0.0079916 m`. We report both **ever within this
threshold** and **within it at the final frame**, plus first success step and
longest consecutive run. This is a simulator proximity label, not an
independent proof of physical insertion. Videos, plug and socket poses, grip
behavior, and contact forces are needed to audit ambiguous successes.

## Diagnostics saved for each rollout

- `trajectory.json`: action at every step, the simulator-clipped action,
  inference latency, keypoint/root/axis errors, source criterion, EE/plug/
  socket poses, joint state, gripper width, EE-to-plug distance, plug/socket
  pairwise contact force, and TacFF activity summaries.
- `inputs.npz`: the exact 16-action prediction at each policy query, query
  frame indices, front/wrist RGB, 18-D state, corrected cloud and raw TacFF.
  It also retains end-effector pose and, for applicable configs, tactile RGB,
  tactile depth, or the clean cloud before added noise.
  Sensors unused by a given policy are retained for diagnosis, never routed
  into its inference request.
- `front_wrist.mp4`: complete rollout at native step timing, including the
  initial frame. `summary.json` records the checkpoint path and metadata hash.
- `paired_diagnostics.csv`, `analysis.json`, `case_index.html` aggregate paired
  outcomes and show the same seed side by side, prioritizing disagreements.
  The analysis includes paired rate differences with bootstrap intervals and
  exploratory breakdowns by socket Y and initial distance. With 40 seeds,
  these strata are hypothesis generators rather than firm subgroup estimates.

After the rollout matrix, `probe_contactworld_modality_sensitivity.py` can
hold RGB/state and the diffusion noise fixed while replacing one live spatial
input with that rollout's first-frame value. It tests whether predicted actions
depend on the added modality. A stale input may lie off the training
distribution, so it does not establish that the modality improves control;
closed-loop interventions on selected paired cases would be a separate step.

The automatic case labels describe observed trajectories (`source_at_end`,
`transient_source`, `near_goal_no_source`, `no_close_approach`). Grip-distance
change and socket contact are **clues**, not causal labels. The EE-to-plug
distance changes substantially even in apparently good rollouts during early
gripper settling, so a fixed change threshold would falsely label slips. We should inspect
paired video and tactile/geometry traces before claiming a modality fixed a
specific failure mechanism.

## Reproduction

```bash
uv run --no-sync python sim/scripts/run_contactworld_pi0_matrix.py \
  --trials 40 --fresh-env-per-trial --isolate-simulator-per-trial \
  --accept-sparse-cloud \
  --output /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_11models_40trials_20261002
uv run --no-sync python sim/scripts/analyze_contactworld_pi0_matrix.py \
  /home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_11models_40trials_20261002
```

The launcher runs one policy/server pair at a time on GPU 0, keeps each
checkpoint under its own result directory, and resumes from the last completed
seed. The paired run must use `--fresh-env-per-trial` and
`--isolate-simulator-per-trial` with a new output root. Isaac Gym crashed when
we attempted to rebuild the simulator repeatedly inside a single process;
one process per seed is the validated isolation boundary.
During a bounded smoke test use `--limit-this-run 5`; rerun without it to
continue. The matrix uses the local checkpoint root
`/home/sai/zsq/FactileLDM/1002_ckpt`. Forty trials form a preliminary pass;
the same seed prefix can later be extended with a new evaluation root or an
explicitly versioned extension protocol. Do not mix the interrupted earlier
100-trial output with this matrix.

The point cloud is not cropped around the USB work area. The source simulator
calls `get_merged_pointcloud_base` with `crop_bounds=None`, samples 1024 points
from the front depth camera, and the converted training sidecar retains those
points. In five online initial frames, 745–937 of 1024 points had Z below
2 cm. This is a material limitation of the existing point-cloud experiment;
cropping only at evaluation time would change the input distribution for the
trained checkpoints. Inspect the [actual input visualizations](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_policy_cloud_20261002/index.html)
before starting the 40-trial matrix. The visualizations overlay simulator
plug/socket poses for diagnosis; those markers are not given to the policy.

Primary source for the original task and planning setup:
[ContactWorld paper](https://arxiv.org/abs/2606.13877),
[public repository](https://github.com/PokuangZhou/ContactWorld).
