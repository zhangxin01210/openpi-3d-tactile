# UR7e + XHand 部署

以下命令均在仓库根目录执行。服务端的 `CONFIG` 可以只写模型族；任务会由 checkpoint
路径中的任务名和 checkpoint 内的 `norm_stats.json` 严格确认。两者冲突或无法唯一判断时
服务端会拒绝启动，避免误用另一个任务的配置。完整配置名仍然兼容。

## 1106 checkpoint 状态

| 模型族 | 输入 | Button / Socket | 当前状态 |
|---|---|---|---|
| `pi0_xhand_ho_v3_5p` | 物体几何 + 5 个指尖几何中心 | 均有 | 已实际恢复，可部署 |
| `pi0_xhand_pc_v2_tactile_mlp` | Front 点云 + 600 taxel 简单 MLP | 均有 | 已实际恢复，可部署 |
| `pi0_xhand_ho_v4` | 物体几何 + 固定锚点 | 均有 | checkpoint 尚未拉完整，锚点资产也未就绪，不可部署 |

当前 `ho_v4` 两个 checkpoint 均会报 `Truncated Zstd-compressed stream`。
不要根据 `_METADATA` 存在就判断下载完成；必须执行下面的完整恢复检查。

## 1. 检查 checkpoint

修改 `FAMILY` 和 `TASK`。`TASK` 可取 `press_the_button` 或 `yellow_socket`。

```bash
FAMILY=pi0_xhand_ho_v3_5p
TASK=press_the_button
CKPT_ROOT=1106_ckpt
CKPT="$CKPT_ROOT/$FAMILY/$TASK/seed42/19999"

PYTHONPATH=src:. OPENPI_REPO_ROOT="$PWD" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/deploy/verify_checkpoint.py \
  --config "$FAMILY" --checkpoint "$CKPT"
```

只有输出 `PASS` 后才启动服务。下载过程中不要运行服务端。

## 2. 启动服务端

日常部署只修改 `CONFIG` 和 `CKPT`。`CONFIG` 推荐写模型族；checkpoint 换到任意目录时，
只要目录名含任务名，或 checkpoint 内带有该任务的 `norm_stats.json`，都能自动解析。

```bash
CONFIG=pi0_xhand_ho_v3_5p
CKPT=1106_ckpt/pi0_xhand_ho_v3_5p/press_the_button/seed42/19999

PYTHONPATH=src:. OPENPI_REPO_ROOT="$PWD" \
OPENPI_PERCEPTION_PYTHON="$PWD/.venv-perception/bin/python" \
OPENPI_SAM2_SOURCE="$HOME/.cache/openpi_deploy/sam2" \
OPENPI_SAM2_CHECKPOINT="$PWD/checkpoints/sam2/sam2.1_hiera_small.pt" \
OPENPI_RGBTRACK_WEIGHTS="$PWD/checkpoints/rgbtrack" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_policy.py --port 8990 policy:checkpoint \
  --policy.config "$CONFIG" --policy.dir "$CKPT"
```

MLP 模型会忽略未使用的感知变量；HO 模型需要这些变量。监听 `0.0.0.0:8990` 只表示
模型已启动；在线感知 worker 会在第一次有效请求时启动。

## 3. 机器人端先做 dry-run

在机器人电脑的 LeRobot 环境执行，修改服务端 IP 和本地数据集目录：

```bash
SERVER_IP=192.168.10.5
DATASET_DIR=/home/amax/code/lerobot/data/1028_press_button

sudo -E "$(which python)" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip "$SERVER_IP" --server-port 8990 \
  --dataset-dir "$DATASET_DIR" \
  --spatial-cameras auto --run-mode single \
  --fps 15 --duration 60 \
  --query-frequency 16 --max-action-chunk-size 16 \
  --dry-run --no-home --no-start-hand-align --no-tactile-reset
```

确认服务端完成至少一次推理、相机/状态维度正确、输出 action 全部有限后，再去掉
`--dry-run`。正式运行时不要机械保留三个 `--no-*` 参数；只有已人工完成对应初始化时
才可跳过回零、起始手姿对齐或触觉归零。

Prompt、空间相机和 Button/Socket 单参考资产由服务端元数据自动选择。几何模型应发布
`spatial_camera_roles=['front']`，即使模型本身 `spatial_use_visual=False`，因为在线感知仍需
原始 Front RGB-D。

## HO v4

HO v4 还需要：

1. 两个 checkpoint 能通过完整恢复检查；
2. `assets/ho_v4_anchors/<dataset>/{manipulated,target}.json`；
3. 在线几何携带并验证 `surface_points_sha256`。

服务端现已支持第 3 项，并会在恢复大模型前检查第 2 项。本机目前缺少生成锚点所需的
`object_geometry/v1` shard，因此不能绕过验证伪造资产。待完整 sidecar 或已验证锚点从训练机
同步后，再按 `docs/ho_v4_anchors.md` 准备并部署。
