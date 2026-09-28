# OpenPI 3D + Tactile

本仓库基于 [Physical Intelligence 的 OpenPI](https://github.com/Physical-Intelligence/openpi)，增加了 UR7e + XHand 的 RGB-D 三维视觉、五指触觉输入，以及 Structured Spatial Encoder。本文记录本项目常用的训练、远程推理、机器人部署和失败诊断命令；上游通用模型和其他机器人示例见 [`examples/`](examples/) 与 [`docs/`](docs/)。

当前训练链路使用 LeRobot 数据集中的三路 RGB、1972 维原始状态和 18 维绝对关节目标动作。front/left 两路深度经共享的 `SpatialPreprocessor` 生成 4096 个视觉点；五指触觉生成 600 个 taxel 点。模型中的 proprio 是 18 维关节位置，输出动作仍是 18 维绝对目标。训练时读取预先导出的 `spatial/v1`，在线推理时由**模型服务端**根据机器人发送的原始观测执行同一套空间预处理。

以下命令除标注“机器人电脑”的部分外，均在**仓库根目录**执行。示例使用 `pi0_xhand_spatial_structured_suffix` 和 `press_button_4_times_clean`；换实验时，训练配置与服务配置必须一致。

## 环境与路径

训练和模型服务需要 Python 3.11+、项目依赖及可用的 NVIDIA GPU。初次安装可使用：

```bash
cd /path/to/openpi-3d-tactile
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-group rlds
```

激活已安装好的环境后，下面的命令直接使用 `python`。在已有训练环境中不要随手运行普通 `uv run`：它可能同步并重建 `.venv`。如果必须通过 uv 调用已有环境，使用 `uv run --no-sync ...`。`rlds` 依赖组用于其他数据流程，本项目的 LeRobot spatial 训练不需要它。

本项目的数据、配置和输出位置：

| 内容 | 当前示例位置 |
| --- | --- |
| LeRobot 原始数据 | `data/press_button_4_times_clean/`，含 `meta/info.json`、data 和视频 |
| 离线空间数据 | `data/press_button_4_times_clean/spatial/v1/` |
| 相机、手部几何与机器人标定 | `configs/ur7e_xhand/` |
| state/action 归一化统计 | `assets/xhand/press_button_4_times_clean/` |
| 训练 checkpoint | `checkpoints/<config>/<exp-name>/<step>/` |

`data/`、`assets/`、`checkpoints/` 均不随 Git 提交。当前 XHand 配置在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中还包含训练机数据集路径和预训练权重的绝对路径；换机器或数据集时，先修改配置中的 `repo_id`、`spatial.dataset_root`、`weight_loader` 路径。checkpoint 的具体 step 以实际生成的目录为准。

## 训练配置

| 配置名 | 空间编码器 | 空间条件接入位置 | 当前训练方式 |
| --- | --- | --- | --- |
| `pi0_xhand_spatial_structured_prefix` | Structured Spatial Encoder | prefix | LoRA |
| `pi0_xhand_spatial_structured_suffix` | Structured Spatial Encoder | suffix | full |
| `pi0_xhand_spatial_structured_both` | Structured Spatial Encoder | prefix + suffix | LoRA |

仓库还保留 `pi0_xhand_spatial_joint_pointnet_{prefix,suffix,both}` 等对照配置。配置名定义模型结构，**不能用 prefix 配置加载 suffix checkpoint**。

### 1. 导出空间数据

先确认原始数据和 `configs/ur7e_xhand/` 的标定文件齐全。可先做单帧检查，再导出全部 episode：

```bash
python scripts/spatial/export_spatial_derived_dataset.py \
  --dataset data/press_button_4_times_clean \
  --episodes 0 --frames 0 --version v1_smoke \
  --cameras front,left --num-points 4096

python scripts/spatial/export_spatial_derived_dataset.py \
  --dataset data/press_button_4_times_clean \
  --episodes all --frames all --version v1 \
  --cameras front,left --num-points 4096 --shard-size 128
```

导出器默认拒绝覆盖已有的版本目录。已有 `spatial/v1` 时直接进入下一步；改变采样、标定或力定义时应改用新版本，并同步修改训练配置的 `spatial.version`。可选的统计和输入检查：

```bash
python scripts/spatial/compute_spatial_dataset_stats.py \
  --dataset data/press_button_4_times_clean --version v1

python scripts/spatial/smoke_test_xhand_spatial_pipeline.py \
  --config-name pi0_xhand_spatial_structured_suffix --sample-index 0
```

更多数据格式和标定约定见 [`docs/spatial_preprocessing_v1.md`](docs/spatial_preprocessing_v1.md)。

### 2. 计算归一化统计

```bash
python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_spatial_structured_suffix
```

当前三个 Structured 配置共享 `assets/xhand/press_button_4_times_clean/` 下的 state/action 统计。训练需要它；checkpoint 保存时也会把统计复制到自己的 `assets/`，供服务端推理使用。

### 3. 启动训练

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_xhand_spatial_structured_suffix \
  --exp-name xhand_spatial_full_test
```

训练输出位于 `checkpoints/pi0_xhand_spatial_structured_suffix/xhand_spatial_full_test/`。从已有实验继续训练时使用**同一个**配置和实验名：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_xhand_spatial_structured_suffix \
  --exp-name xhand_spatial_full_test --resume
```

`--overwrite` 会覆盖同名实验目录，不要把它写进日常训练命令。更换 prefix/both 时，要同步替换上述统计、检查和训练命令里的配置名。训练配置的 `batch_size` 必须能被可见 GPU 数量整除。

## 远程推理与机器人部署

模型在 GPU 机器上运行 [`scripts/serve_policy.py`](scripts/serve_policy.py)，机器人电脑运行 [`deploy/ur7e_xhand_deploy_3d_tactile_multi.py`](deploy/ur7e_xhand_deploy_3d_tactile_multi.py)。机器人客户端只发送原始 state、三路 RGB、front/left 深度及 prompt；服务端重建点云和触觉空间输入，再返回 action chunk。两端代码需使用匹配版本。

### 1. GPU 机器启动服务端

```bash
cd /path/to/openpi-3d-tactile
OPENPI_REPO_ROOT="$PWD" CUDA_VISIBLE_DEVICES=0 \
python scripts/serve_policy.py \
  --port 8990 \
  policy:checkpoint \
  --policy.config pi0_xhand_spatial_structured_suffix \
  --policy.dir checkpoints/pi0_xhand_spatial_structured_suffix/xhand_spatial_full_test/29999
```

`--policy.dir` 指向包含 `params/` 和 `assets/` 的**具体 step**，上面的 `29999` 只是示例。服务端需要能访问本仓库的 `configs/ur7e_xhand/` 静态资产，`OPENPI_REPO_ROOT` 指向仓库根目录。服务端监听 `0.0.0.0:8990`；机器人电脑应能访问该地址。若 checkpoint 属于 prefix/both，配置名和 checkpoint 路径一起改。

### 2. 机器人电脑检查配置

机器人电脑可以只放 `deploy/` 和与训练数据一致的 `meta/info.json`，不必安装整个 OpenPI。先激活已跑通的 LeRobot/UR7e/XHand 部署环境，再进入脚本所在目录的上一级：

```bash
cd /path/to/deployment
PYTHON_BIN="$(command -v python)"

"$PYTHON_BIN" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --dataset-dir /path/to/press_button_4_times_clean \
  --check-config
```

`--dataset-dir` 指向包含 `meta/info.json` 的目录，用于确定原始 state 和 18 维 action 的**字段顺序**。不要依赖脚本默认的 `grasp_pipette` 目录。配置检查不连接机器人，也不验证现场相机数据。

### 3. 先做 dry run，再正式运行

EtherCAT 手通常需要 root 权限；`sudo -E` 搭配已激活环境的 Python，避免切换到系统 Python。替换服务端 IP 和数据集目录：

```bash
sudo -E "$PYTHON_BIN" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip 192.168.1.100 --server-port 8990 \
  --dataset-dir /path/to/press_button_4_times_clean \
  --prompt "press the button 4 times and put it into the box" \
  --hand-protocol EtherCAT \
  --record-dir ./diagnostic_runs --duration 60 \
  --dry-run --no-home --no-tactile-reset
```

`--dry-run` 不下发策略动作，但默认的 home reset 和触觉清零仍是独立流程；上面两个 `--no-*` 参数使联调时跳过它们。确认服务端返回动作、三路 RGB 和两路深度均正常后，正式运行：

```bash
sudo -E "$PYTHON_BIN" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip 192.168.1.100 --server-port 8990 \
  --dataset-dir /path/to/press_button_4_times_clean \
  --prompt "press the button 4 times and put it into the box" \
  --hand-protocol EtherCAT \
  --record-dir ./diagnostic_runs --duration 60
```

正式运行默认先回零、清零触觉，再等待 Enter 开始。只有在确实需要免确认时才加 `--yes`。多次测试可加 `--run-mode multi --num-runs 0`，每轮按右方向键结束并单独保存诊断目录。机械臂 IP、相机序列号、控制频率等有脚本默认值，平台变更时用 `--help` 查看并覆盖对应参数。

## 失败诊断

启用 `--record-dir` 后，每轮生成 `diagnostic_runs/<时间>_run_<编号>/`，包含 `manifest.json`、`control/*.npz` 和 `inference/query_*_{input,output}.npz`。输入记录保留三路 RGB、两路原始深度、完整 state 和五指触觉；输出记录保留返回的 action chunk、实际使用的 chunk 和耗时；控制记录包含逐帧实际发送动作与 fallback 状态。检查 `manifest.json` 中的 `dropped_records` 和 `writer_errors` 可判断记录是否完整。

在只有 `deploy/` 的机器人电脑上，只要环境有 NumPy、Matplotlib、OpenCV，即可生成汇总图和每次推理图；点云面板会缺失。完整的 4096 点视觉点云及 600 个触觉空间点，需要在有 `src/openpi/spatial` 与标定文件的本仓库中离线重建。

将**整个运行目录**拷回仓库电脑，然后运行：

```bash
# 在仓库电脑执行；替换机器人电脑地址和实际运行目录。
mkdir -p ./diagnostic_runs
scp -r USER@ROBOT_IP:/path/to/diagnostic_runs/20260928_具体时间_run_001 ./diagnostic_runs/

python deploy/render_pi0_diagnostics.py \
  ./diagnostic_runs/20260928_具体时间_run_001 \
  --repo-root "$PWD" --video
```

报告位于该运行目录的 `report/`：`summary.png` 是全程总览，`query_*.png` 是每次推理的输入、触觉、空间点和 chunk 对比，`diagnostic.mp4` 由逐 query 图组成，**不是** 15 Hz 控制帧视频。只需图片时去掉 `--video`。若本机提示缺少 Matplotlib，应切换到装有绘图依赖的分析环境；无需在机器人控制环境里安装整套训练依赖。

## 常见问题

- 服务端无法加载 checkpoint：检查 `--policy.config` 是否与训练配置相同，`--policy.dir` 是否指向具体 step，且该目录有 `params/`、`assets/`。
- 客户端 `--check-config` 报 state 维度不符：检查 `--dataset-dir/meta/info.json` 是否来自训练所用的完整 XHand 数据集；只有前 52 维的 fallback 信息不够生成触觉输入。
- 首次推理缺少 `images` 或 `spatial`：确认 GPU 服务端包含当前 [`spatial_online.py`](src/openpi/policies/spatial_online.py) 与 [`policy_config.py`](src/openpi/policies/policy_config.py) 并已重启。
- 点云诊断显示 `spatial reconstruction unavailable`：检查 `--repo-root`、标定资产与空间预处理依赖；原始 NPZ 数据仍可用于图像、触觉和动作诊断。
- 包管理器访问镜像失败：先检查当前 Python 环境是否已经可运行；不要让 `uv run` 自动重建已用于训练的环境。

通用 OpenPI 的远程推理说明见 [`docs/remote_inference.md`](docs/remote_inference.md)，更多环境和 Docker 说明见 [`docs/docker.md`](docs/docker.md)。
