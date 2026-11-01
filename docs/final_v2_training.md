# XHand final V2 training

All three datasets use `spatial/v1`, `object_geometry/v1`, and
`hand_keypoints/v1`. Full requires all three; ST requires spatial only; RGB
requires no sidecar. The dataset QA figures supplied for 1018, 1028 and 1030
are 70/32,343, 100/26,485 and 111/34,963 episodes/frames respectively. Preflight
checks the actual files and indexed joins again; it does not regenerate data.

| Config | Dataset | Base | Spatial tokens |
|---|---|---|---:|
| `pi0_xhand_1028_press_button_rgb_v2` | 1028_press_button | π0 | 0 |
| `pi05_xhand_1028_press_button_rgb_v2` | 1028_press_button | π0.5 | 0 |
| `pi0_xhand_1028_press_button_st_v2` | 1028_press_button | π0 | 58 |
| `pi0_xhand_1028_press_button_full_v2` | 1028_press_button | π0 | 68 |
| `pi0_xhand_1030_yellow_socket_rgb_v2` | 1030_yellow_socket | π0 | 0 |
| `pi05_xhand_1030_yellow_socket_rgb_v2` | 1030_yellow_socket | π0.5 | 0 |
| `pi0_xhand_1030_yellow_socket_st_v2` | 1030_yellow_socket | π0 | 58 |
| `pi0_xhand_1030_yellow_socket_full_v2` | 1030_yellow_socket | π0 | 68 |
| `pi0_xhand_1018_peg_all_st_v2` | 1018_peg_all | π0 | 58 |
| `pi0_xhand_1018_peg_all_full_v2` | 1018_peg_all | π0 | 68 |

The scene branch uses masked 256/24 then 32/16 FPS/KNN with mean/max pooling,
the existing raw-point global token, position and modality embeddings. The
tactile branch retains taxel MLP and per-finger EdgeConv and pools four fixed
link-local geometry regions plus one whole-finger region. HO queries the
manipulated cloud from the 18 exported FK anchors and pools each finger's own
anchors. OO retains 8×8 object regions and five learned queries. HO and OO use
only relative XYZ and distance as relation features; stored metadata is not
modified. Full and ST use the same V2 branches and existing suffix router.

The palm anchor is `palm/frame_origin` on `right_hand_link`. It is a palm frame
reference, not a finger joint or tip, and is excluded from finger token pooling.
The remaining anchors and their associated child or tip links are:

| Finger | Anchor → link |
|---|---|
| thumb | `right_hand_thumb_bend_joint` → `right_hand_thumb_bend_link`; `right_hand_thumb_rota_joint1` → `right_hand_thumb_rota_link1`; `right_hand_thumb_rota_joint2` → `right_hand_thumb_rota_link2`; `right_hand_thumb_rota_tip` → same-named link |
| index | `right_hand_index_bend_joint` → `right_hand_index_bend_link`; `right_hand_index_joint1` → `right_hand_index_rota_link1`; `right_hand_index_joint2` → `right_hand_index_rota_link2`; `right_hand_index_rota_tip` → same-named link |
| middle | `right_hand_mid_joint1` → `right_hand_mid_link1`; `right_hand_mid_joint2` → `right_hand_mid_link2`; `right_hand_mid_tip` → same-named link |
| ring | `right_hand_ring_joint1` → `right_hand_ring_link1`; `right_hand_ring_joint2` → `right_hand_ring_link2`; `right_hand_ring_tip` → same-named link |
| pinky | `right_hand_pinky_joint1` → `right_hand_pinky_link1`; `right_hand_pinky_joint2` → `right_hand_pinky_link2`; `right_hand_pinky_tip` → same-named link |

From the A800 checkout root, set `OPENPI_PI0_BASE_WEIGHTS` and
`OPENPI_PI05_BASE_WEIGHTS` if the defaults under
`/workspace/mnt/sqzhang26/hf_weight/{pi0_base,pi05_base}/params` differ. Set
`OPENPI_1018_FRONT_PROFILE`, `OPENPI_1028_FRONT_PROFILE`, and/or
`OPENPI_1030_FRONT_PROFILE` to each dataset's **original** profile file when
required by its manifest. A file with equivalent JSON but a different SHA256 is
not a replacement. For the three accepted v1 datasets, a missing object
calibration field produces `LEGACY_UNRECORDED`: object calibration provenance
is unknown and has not been verified. Recorded calibration is checked strictly.

Each dataset shares `assets/xhand/<dataset>_unified_v1/norm_stats.json` across
its configurations. The native π0 pipeline uses mean/std; the native π0.5
pipeline uses quantiles from the same statistics. If missing, compute the stats
using the existing script; select the dataset's RGB config, or the 1018 ST
config, for example:

```bash
PYTHONPATH=src .venv/bin/python scripts/compute_norm_stats.py --config-name pi0_xhand_1028_press_button_rgb_v2
```

Run preflight before loading large models, then smoke (short training writes to
a temporary, independent checkpoint directory):

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/preflight_final_v2.py --all-configs --json /tmp/final_v2_preflight.json
PYTHONPATH=src .venv/bin/python scripts/experiments/smoke_final_v2.py --all-configs --check all
```

Use `--config CONFIG --check data|model|train|all` to isolate a configuration.
Review preflight errors per configuration: `--all-configs` continues after
individual failures. The following commands perform preflight again and then use
the original trainer in the requested order:

```bash
bash scripts/experiments/run_final_v2.sh pi0_xhand_1028_press_button_rgb_v2
bash scripts/experiments/run_final_v2.sh pi05_xhand_1028_press_button_rgb_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1028_press_button_st_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1028_press_button_full_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1030_yellow_socket_rgb_v2
bash scripts/experiments/run_final_v2.sh pi05_xhand_1030_yellow_socket_rgb_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1030_yellow_socket_st_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1030_yellow_socket_full_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1018_peg_all_st_v2
bash scripts/experiments/run_final_v2.sh pi0_xhand_1018_peg_all_full_v2
```

`bash scripts/experiments/run_final_v2.sh all` runs that order sequentially.
Resume one existing experiment with `bash scripts/experiments/run_final_v2.sh
CONFIG --resume`. The default experiment name is `final_v2`; override with
`OPENPI_FINAL_V2_EXP_NAME`. The trainer refuses an existing directory unless
resume is requested. Check `logs/final_v2/` for commit, config, sidecars and
weight source, and `checkpoints/CONFIG/EXPERIMENT/` for saved steps.

To avoid repeating a full preflight that has already passed, append
`--skip-preflight` to the training command. Normal DataLoader initialization
and its existing data and Spatial calibration checks still run.

Local checks: V2 module, configuration, simulated sidecar join and router
tests passed (10 tests); the online, configuration, join and relation suite
passed after integration (11 tests); existing object geometry/config tests
passed (7 tests), and CPU online/DataLoader tests passed (6 tests, excluding
the real-data and multiworker cases). An attempted multiworker DataLoader test
failed when the local CUDA backend could not initialize; it is not a passed
check. Configuration preflight resolved all ten configurations locally but
reported missing local data, assets and weights.

Local tests use synthetic inputs and actual repository geometry only. Real
sidecars, calibrated camera metadata, pretrained weights, A800 memory use,
loss/backward/update and checkpoint restoration require cloud verification.
Do not treat configuration registration or local tests as a successful cloud
smoke run. Object pose estimates remain subject to measurement error; HO
distances are not verified physical contact distances.

## 单卡动态训练队列

以下命令在仓库根目录执行；先确认数据、归一化资产和预训练权重在本机就绪。调度器检查所选 GPU 上的计算进程；任何一张已有进程时不会启动。每个训练进程只看到一张 GPU，并在进入训练器之前检查 `jax.device_count() == 1`。队列完成 1028、1030 的八个实验后才启动 1018；任一实验失败时不会越过阶段边界。不同队列之间也会锁定所选 GPU 和实验 checkpoint，外部程序仍需自行避免在调度器运行期间抢占 GPU。

```bash
mkdir -p output/final_v2_runner
nohup .venv/bin/python -u scripts/experiments/queue_final_v2.py run \
  --max-parallel 4 --gpus 0,1,2,3,4,5,6,7 --threads 2 \
  --skip-preflight > output/final_v2_runner/launcher.log 2>&1 < /dev/null &
```

只在已完成所需 Preflight 后使用 `--skip-preflight`；普通 DataLoader 初始化与校验依然执行。要测试 8 并行，在其他条件相同时把 `--max-parallel 4` 改为 `--max-parallel 8`。每进程的 OMP、MKL、OpenBLAS 线程数默认为 2，可用 `--threads` 调整。默认设置 `WANDB_MODE=offline`，如果环境中已显式设置则保留。每模型日志间隔为 100 步；模型配置的 batch size、训练步数、学习率、种子与归一化不变。

```bash
.venv/bin/python scripts/experiments/queue_final_v2.py status
cat output/final_v2_runner/main/runner.log
tail -f output/final_v2_runner/main/models/pi0_xhand_1028_press_button_full_v2.log
.venv/bin/python scripts/experiments/queue_final_v2.py stop
```

`stop` 向该队列发送停止请求，只终止它启动的训练进程组，等待 10 秒后强制结束未退出的子进程。再次运行启动命令时，已完成的最终 checkpoint 自动跳过，有有效 checkpoint 的实验自动恢复；空实验目录从头开始。格式不完整的数字 checkpoint 目录会被标为失败，需人工检查，脚本不会覆盖它。失败任务最终 checkpoint 的状态不会自动视为成功；确认后可用 `--retry-failed` 明确重试。训练日志为追加写入。若修改 `--state-dir`，查询和停止时也须指定同一目录；与正式训练共享 checkpoint 时必须保留状态目录，以保留失败记录。

## 隔离的 4/8 并行吞吐测试

确保正式队列已停止，分别执行以下测试，且两次使用同样数据、线程数及机器负载。每次使用新的空 `--state-dir`，checkpoint 位于该目录下的 `checkpoints/`，不会写入正式 `checkpoints/`。测试只运行 400 步，训练定义其他参数不变；这不是正式训练。也可调大 `--benchmark-steps` 以获得更长的稳定窗口。以下命令假定 Preflight 已完成：

```bash
mkdir -p output/final_v2_benchmarks
nohup .venv/bin/python -u scripts/experiments/queue_final_v2.py run \
  --benchmark --state-dir output/final_v2_benchmarks/b4 \
  --max-parallel 4 --gpus 0,1,2,3,4,5,6,7 --threads 2 --skip-preflight \
  > output/final_v2_benchmarks/b4-launcher.log 2>&1 < /dev/null &
.venv/bin/python scripts/experiments/queue_final_v2.py status --state-dir output/final_v2_benchmarks/b4
# b4 完成后执行：
nohup .venv/bin/python -u scripts/experiments/queue_final_v2.py run \
  --benchmark --state-dir output/final_v2_benchmarks/b8 \
  --max-parallel 8 --gpus 0,1,2,3,4,5,6,7 --threads 2 --skip-preflight \
  > output/final_v2_benchmarks/b8-launcher.log 2>&1 < /dev/null &
.venv/bin/python scripts/experiments/queue_final_v2.py status --state-dir output/final_v2_benchmarks/b8
# 如需停止，在 stop 后指定相同的 --state-dir。
```

完成后比较两个目录的 `throughput.json`：每个模型从至少第 100 步的首条训练日志到最后一条日志计算 steps/s，排除首次 JAX 编译和热身；总 steps/s 为稳定窗口内的总步数除以这些窗口的时间并集，涵盖队列补位和阶段变化。报告也给出各模型稳定窗口内的主机整体 CPU 和对应 GPU 的平均利用率；每 5 秒采样的原始主机 CPU、各 GPU 利用率与显存记录在 `utilization.jsonl`。短任务采样不足时相应平均值为 null，应延长测试而非将其当作零。总吞吐不是 8 卡同时稳定运行时瞬时速率的简单相加，比较时需结合各模型结果与原始采样。
