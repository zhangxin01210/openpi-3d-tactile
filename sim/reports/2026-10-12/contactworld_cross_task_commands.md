# ContactWorld cross-task execution

The code leaves USB V3 untouched. Each task receives its own source manifest,
replay archive, LeRobot root, normalisation statistics and checkpoints.

## 1. Download the two archives directly

```bash
cd /home/sai/zx/openpi-3d-tactile
.venv/bin/python sim/scripts/download_contactworld_cross_tasks.py \
  --tasks barbed_flat valve \
  --output /home/sai/zx/openpi-sim-runtime/downloads/contactworld
```

The downloader uses `curl --noproxy '*'`, resumes `.part` files, and does not
unpack data. Peg is already present at
`/home/sai/zx/openpi-sim-runtime/data/contactworld/insertion_peg`.

## 2. One-demo review, one task at a time

```bash
bash sim/scripts/launch_contactworld_cross_task_smoke.sh peg
bash sim/scripts/launch_contactworld_cross_task_smoke.sh barbed_flat \
  /home/sai/zx/openpi-sim-runtime/downloads/contactworld/disassembly_barbed_flat_dataset.tar.gz
bash sim/scripts/launch_contactworld_cross_task_smoke.sh valve \
  /home/sai/zx/openpi-sim-runtime/downloads/contactworld/screwing_valve_dataset.tar.gz
```

Each command produces `.../runs/contactworld_cross_<task>_smoke/review/index.html`.
Review the front/wrist RGB, 4096-point front/fused clouds, both TacFF pads,
base-frame taxel positions and 3D forces, and the preserved executed action
dimension. Do not start the next command until that page is accepted.

## 3. Full conversion after each approved smoke review

```bash
bash sim/scripts/launch_contactworld_cross_task_full.sh peg --approved
bash sim/scripts/launch_contactworld_cross_task_full.sh barbed_flat \
  /home/sai/zx/openpi-sim-runtime/downloads/contactworld/disassembly_barbed_flat_dataset.tar.gz --approved
bash sim/scripts/launch_contactworld_cross_task_full.sh valve \
  /home/sai/zx/openpi-sim-runtime/downloads/contactworld/screwing_valve_dataset.tar.gz --approved
```

This step replays every source episode in the isolated old Isaac Gym runtime,
exports a bilateral LeRobot V3 sidecar, computes `assets/norm/norm_stats.json`,
and runs the exact model-input audit. Upload each completed `data/contactworld_*_v3_bilateral`
directory together with the changed code. No replay cache is needed on the
training server.

## 4. Server preflight and matrix training

```bash
export OPENPI_CONTACTWORLD_BASE_WEIGHTS=/path/to/pi0_base/params
.venv/bin/python sim/scripts/launch_contactworld_cross_task_matrix.py \
  --tasks peg barbed_flat valve --label cw_cross_first \
  --gpus 0,1,2,3,4,5,6,7 --check-only

.venv/bin/python sim/scripts/launch_contactworld_cross_task_matrix.py \
  --tasks peg barbed_flat valve --label cw_cross_first \
  --gpus 0,1,2,3,4,5,6,7
```

For every task the launcher trains four configs from the same original pi0
base weights: RGB; RGB+fused 4096-point cloud; model 07 pooled bilateral
base-coordinate force tokens; and model 10 parameter-matched per-pad TacFF
maps. It first verifies sidecars and runs the actual transform audit. Peg uses
6D actions; Flat Barb and Valve preserve their 7D actions.
