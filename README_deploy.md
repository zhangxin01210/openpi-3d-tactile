# 部署

在本机仓库根目录执行。八个模型共用服务入口：

| 任务 | MODEL |
|---|---|
| Button RGB | `pi0_xhand_1028_press_button_rgb_v2` |
| Button Pi0.5 RGB | `pi05_xhand_1028_press_button_rgb_v2` |
| Button ST | `pi0_xhand_1028_press_button_st_v2` |
| Button Full | `pi0_xhand_1028_press_button_full_v2` |
| Socket RGB | `pi0_xhand_1030_yellow_socket_rgb_v2` |
| Socket Pi0.5 RGB | `pi05_xhand_1030_yellow_socket_rgb_v2` |
| Socket ST | `pi0_xhand_1030_yellow_socket_st_v2` |
| Socket Full | `pi0_xhand_1030_yellow_socket_full_v2` |

## 服务端（本机）

以下感知路径已在本机确认存在，RGB/ST 不使用额外感知环境。

```bash
MODEL=pi0_xhand_1028_press_button_full_v2

case "$MODEL" in
  pi05_xhand_1030_yellow_socket_rgb_v2) EXP=final_v2_normfix ;;
  *) EXP=final_v2 ;;
esac

PYTHONPATH=src:. \
OPENPI_REPO_ROOT="$PWD" \
OPENPI_PERCEPTION_PYTHON="$PWD/.venv-perception/bin/python" \
OPENPI_SAM2_SOURCE="$HOME/.cache/openpi_deploy/sam2" \
OPENPI_SAM2_CHECKPOINT="$PWD/checkpoints/sam2/sam2.1_hiera_small.pt" \
OPENPI_RGBTRACK_WEIGHTS="$PWD/checkpoints/rgbtrack" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_policy.py \
  --port 8990 \
  policy:checkpoint \
  --policy.config "$MODEL" \
  --policy.dir "checkpoints/$MODEL/$EXP/19999"
```

## 机器人电脑

使用原有机器人环境，在仓库根目录执行；修改服务端 IP 和机器人本地数据目录。

```bash
SERVER_IP=192.168.1.100
DATASET_DIR=/path/to/1028_press_button  # Socket 则换成对应数据目录

sudo -E "$(which python)" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip "$SERVER_IP" --server-port 8990 \
  --dataset-dir "$DATASET_DIR" \
  --spatial-cameras auto --run-mode single \
  --fps 15 --duration 60 \
  --query-frequency 16 --max-action-chunk-size 16 \
  --dry-run --no-home --no-start-hand-align --no-tactile-reset
```

- Prompt、空间相机和 Button/Socket 单参考资产自动选择，无需填写 `--task` 或 `--scene-position`。Peg 仍需人工指定 `--scene-position 1..6`。
- 默认 Front 标定：`configs/ur7e_xhand/front_calibration_candidate.json`。
- 上述命令用于无动作检查，跳过回零、手姿对齐、触觉归零。正式运行去掉 `--dry-run`；三个 `--no-*` 仅在已完成对应初始化时保留，不能直接沿用为默认真机条件。
- Full 感知 worker 在首个有效请求时启动；监听端口成功不代表感知成功，感知同步等待完成后执行动作，不要求达到 15Hz。
