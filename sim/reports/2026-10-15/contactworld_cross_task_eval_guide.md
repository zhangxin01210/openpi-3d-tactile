# ContactWorld cross-task checkpoint evaluation

The one-command evaluator covers the 12 checkpoints in
`/home/sai/zsq/FactileLDM/1014_ckpt`: three tasks (peg, barbed flat, valve)
and four matched configurations (RGB, RGB+fused point cloud, model 07, model
10).  Each task/config pair uses the same 40 seed values, 180 control steps,
and four-step replanning schedule as the USB V3 evaluation.

## Preflight

```bash
cd /home/sai/zx/openpi-3d-tactile
bash sim/scripts/launch_contactworld_cross_task_eval.sh --check-only
```

This verifies each converted LeRobot/spatial sidecar, runs the four model input
audits per task, checks all 12 checkpoint metadata files, and does not start
Isaac Gym or policy inference.

## Full evaluation

```bash
bash sim/scripts/launch_contactworld_cross_task_eval.sh
```

The result root defaults to
`runs/contactworld_cross_task_eval_YYYYMMDD`.  A different GPU or checkpoint
location can be selected without editing code:

```bash
CROSS_TASK_EVAL_GPU=0 \
CROSS_TASK_EVAL_CHECKPOINTS=/home/sai/zsq/FactileLDM/1014_ckpt \
bash sim/scripts/launch_contactworld_cross_task_eval.sh
```

Each task/config directory contains `summary.json`, `index.html`,
`policy.log`, `eval.log`, one `seed_*` directory per trial, a synchronized
front/wrist video, `trajectory.json`, and `inputs.npz` with all RGB, cloud,
bilateral tactile and action chunks captured at policy replanning points.
The launcher is resumable: rerunning the identical command continues the
next missing seed and skips completed pairs.

`barbed_flat` and `valve` use seven-dimensional actions; `peg` uses six.  The
server keeps Pi0's internal 32-dimensional padding but sends the task's
post-transform six/seven-dimensional action to the simulator.

