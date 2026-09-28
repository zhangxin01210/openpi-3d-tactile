# OpenPI 3D + Tactile

本仓库基于 [Physical Intelligence 的 OpenPI](https://github.com/Physical-Intelligence/openpi)，增加了 UR7e + XHand 的 RGB-D 三维视觉、五指触觉输入，以及 Structured Spatial Encoder。本文记录本项目常用的训练、远程推理、机器人部署和失败诊断命令；上游通用模型和其他机器人示例见 [`examples/`](examples/) 与 [`docs/`](docs/)。

当前链路使用 LeRobot 数据集中的三路 RGB、1972 维原始状态和 18 维绝对关节目标动作。旧 `pi0_xhand_spatial_structured_suffix` 保持 front+left 点云不变；新的 front-only 消融组只用 front 深度生成 4096 个视觉点，触觉保持 600 个 taxel 点。三路 RGB 始终供 VLM 使用，left RGB 并没有被移除。训练读取派生 sidecar，在线推理时由**模型服务端**执行相同空间预处理。

以下命令除标注“机器人电脑”的部分外，均在**仓库根目录**执行。训练示例使用 `press_button_4_times_clean`；本机 `data/` 下若只有 `press_button_0`，先替换配置中的数据路径与资产 ID。训练配置与服务配置必须一致。

## 环境与路径

训练和模型服务需要 Python 3.11+、项目依赖及可用的 NVIDIA GPU。初次安装可使用：

```bash
cd /path/to/openpi-3d-tactile
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-group rlds
```

激活已安装好的环境后，下面的命令直接使用 `python`。在已有训练环境中不要随手运行普通 `uv run`：它可能同步并重建 `.venv`。如果必须通过 uv 调用已有环境，使用 `uv run --no-sync ...`。`rlds` 依赖组用于其他数据流程，本项目的 LeRobot spatial 训练不需要它。

离线标注/对照工具额外需要 `pyarrow`、`av`、`opencv-python`、`scipy`、`plotly`；root-to-tip CAD 轮廓还需 `trimesh`、加载 DAE 所需的 `pycollada` 和下述 URDF 网格。缺失时在**单独的分析环境**安装，不必改动机器人控制环境；缺网格会明确报错，不再静默改画骨架。

本项目的数据、配置和输出位置：

| 内容 | 当前示例位置 |
| --- | --- |
| LeRobot 原始数据 | `data/press_button_4_times_clean/`，含 `meta/info.json`、data 和视频 |
| 离线空间数据 | `data/press_button_4_times_clean/spatial/v1_front/`（旧配置仍用 `v1`） |
| 相机、手部几何与机器人标定 | `configs/ur7e_xhand/` |
| state/action 归一化统计 | `assets/xhand/press_button_4_times_clean/` |
| 训练 checkpoint | `checkpoints/<config>/<exp-name>/<step>/` |

`data/`、`assets/`、`checkpoints/` 均不随 Git 提交。当前 XHand 配置在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中还包含训练机数据集路径和预训练权重的绝对路径；换机器或数据集时，先修改配置中的 `repo_id`、`spatial.dataset_root`、`weight_loader` 路径。checkpoint 的具体 step 以实际生成的目录为准。

## 训练配置

| 配置名 | 空间编码器 | 空间条件接入位置 | 当前训练方式 |
| --- | --- | --- | --- |
| `pi0_xhand_spatial_structured_prefix` | Structured Spatial Encoder | prefix | LoRA |
| `pi0_xhand_spatial_structured_suffix` | Structured Spatial Encoder | suffix，front+left 视觉+触觉 | full，旧配置不变 |
| `pi0_xhand_spatial_structured_suffix_front` | Structured Spatial Encoder | suffix，front-only 视觉+触觉 | full，三卡基线 |
| `pi0_xhand_spatial_structured_suffix_front_visual` | Structured Spatial Encoder | suffix，front-only 视觉 | full，消融触觉 |
| `pi0_xhand_spatial_structured_suffix_front_tactile` | Structured Spatial Encoder | suffix，触觉 | full，消融点云 |
| `pi0_xhand_spatial_structured_both` | Structured Spatial Encoder | prefix + suffix | LoRA |

仓库还保留 `pi0_xhand_spatial_joint_pointnet_{prefix,suffix,both}` 等对照配置。配置名定义模型结构，**不能用 prefix 配置加载 suffix checkpoint**。

三个新配置共用 [`config.py`](src/openpi/training/config.py) 中的 `FRONT_SPATIAL_VERSION` 和 `FRONT_CALIBRATION_PROFILE`。默认是未纠正的 `v1_front` 与 `None`；确认要用候选或新拟合的标定后，改成新 sidecar 版本名和相应 profile 路径。旧 suffix 配置不会跟着变。不要覆盖旧 sidecar，也不要在 checkpoint 训练后改动对应 profile 文件。

### 0. 检查 front 标定

先选不同姿态/接触阶段的几帧，以当前本机存在的 `press_button_0` 为例。命令在有完整数据集和 OpenPI 空间依赖的电脑执行：

```bash
PYTHONPATH=src python scripts/spatial/compare_front_calibration.py \
  --dataset data/press_button_0 --episode 0 --frames 0,50,100 \
  --profile configs/ur7e_xhand/front_calibration_candidate.json \
  --output diagnostics/press_button_0_front_compare
```

打开 `diagnostics/press_button_0_front_compare/index.html`，逐帧比较未纠正/候选纠正的 root-to-tip **CAD 网格逐段与累积轮廓** PNG 和交互 3D 网页。轮廓使用原 RGB + 视野外黑色展开画布，与 `0916_marker_anchored_root_to_tip_upper_lag0` 的版式一致。网页同时显示稠密 ROI、**模型实际看到的 4096 点**、触觉点、C0–C8 机械臂/手掌轮廓和五指轮廓；图例可单击开关各层。三维 CAD 是 FK 生成的参考几何，**不是模型额外输入**。`--output` 必须是不存在的目录。点云仍由当前 canonical `SpatialPreprocessor` 生成，稠密与 4096 层使用相同外参。先看图，没问题就不必重新标注。

渲染器优先读取 `assets/root_to_tip/` 中的独立网格包；若不存在，则读取本机忽略的 `3D_tactile/` 目录。只需将旧项目中的以下文件按相对目录复制一次；`assets/` 不提交 Git：

```bash
mkdir -p assets/root_to_tip/pointcloud_delivery/configs
mkdir -p assets/root_to_tip/pointcloud_delivery/diagnostics/ur_description_source/meshes/ur5e
mkdir -p assets/root_to_tip/ur5_xhand
cp 3D_tactile/pointcloud_delivery/configs/ur7e_xhand_verified.urdf assets/root_to_tip/pointcloud_delivery/configs/
cp -a 3D_tactile/pointcloud_delivery/diagnostics/ur_description_source/meshes/ur5e/visual assets/root_to_tip/pointcloud_delivery/diagnostics/ur_description_source/meshes/ur5e/
cp -a 3D_tactile/ur5_xhand/Flange_meshes assets/root_to_tip/ur5_xhand/
cp -a 3D_tactile/ur5_xhand/xhand_meshes assets/root_to_tip/ur5_xhand/
```

如果旧项目不在本机，先从原机器复制上述 4 处到对应的 `assets/root_to_tip/` 相对目录；或者将完整旧项目放在仓库根目录的 `3D_tactile/`。也可用 `--mesh-urdf /path/to/ur7e_xhand_verified.urdf` 指定 URDF，但它引用的相对网格路径必须存在，且 URDF 必须与 `configs/ur7e_xhand/ur7e_xhand_verified.urdf` 一致。这里不需要复制碰撞网格。

要重现 `0916_marker_anchored_root_to_tip_upper_lag0` 那种**使用 marker 标定相机位姿**的投影，单独调用：

```bash
PYTHONPATH=src python scripts/spatial/root_to_tip_front.py \
  --dataset /path/to/0916_camera_upper --episode 0 --frames 309,417,843 \
  --marker-camera-report 3D_tactile/pointcloud_delivery/diagnostics/0916_forearm_marker_continuous_handeye_e2_upperarm_anchor_lag0/report.json \
  --output diagnostics/0916_marker_root_to_tip
```

`--frames report-qc` 可从报告的 `qc_images` 读取帧号。默认读取 `free_camera_T_base_color`；其他键可用 `--camera-report-key`。也可以给上面的 `compare_front_calibration.py` 命令追加 `--marker-camera-report /path/to/report.json`，它会在 baseline/corrected 之外增加 marker 一行；此时 2D 轮廓、3D 稠密云、4096 点、CAD 轮廓均使用报告中的**同一个** front 位姿。这个覆盖只在离线可视化时生效，不会修改训练/部署 profile；不能直接把 marker 页当成当前模型看到的点云。报告只能用于它对应的数据集、相机安装和时间段；换数据集须验证外参是否仍然成立。

单帧点云网页也可直接生成：

```bash
PYTHONPATH=src python scripts/spatial/visualize_spatial_web.py \
  --dataset data/press_button_0 --episode 0 --frame 0 --cameras front \
  --cad-root-to-tip --output diagnostics/frame_000000_cad_cloud.html
```

如网页较大，用 `--max-dense-points-per-camera 5000` 或 `--max-cad-edges-per-group 1500` 缩减**仅用于展示**的点和轮廓线，不影响正式模型输入。

检查点云时，先在**深度相机坐标系**用多个已知尺寸的静态平面/标记物核对深度单位、`depth_intrinsics`、图像分辨率/裁剪及深度对齐方式；再检查 `T_color_depth` 和 RGB 映射；最后用多个姿态、不同深度的已知三维物体校验 `T_base_color`。可记录有效深度覆盖率、平面点到平面距离的 median/p90、CAD 可见表面的深度残差 median/p90，以及独立测量标记的三维误差；按距离、图像区域、姿态分组看系统偏差。不要把所有触觉点到点云的最近邻距离当准确率：不接触时两者本来就有间隔，接触物体也可能被手遮挡。用“真实触觉位置”做绝对误差需要额外独立真值（实测 taxel 几何/接触点或外部追踪），当前状态+触觉值+深度图自身无法给出该真值。

如候选残差在这个数据集上不准，再生成标注页面。点击可辨认的物理关节中心/指尖，不可见或不确定的跳过：

```bash
PYTHONPATH=src python scripts/spatial/front_calibration_workflow.py annotate \
  --dataset data/press_button_0 --episode 0 --frames 0,50,100,150,200,250,300,350 \
  --output diagnostics/press_button_0_annotation
```

打开 `diagnostics/press_button_0_annotation/annotate.html`，点 **Export JSON**，把下载的 `front_annotations.json` 放到同一目录。默认点定义见 [`front_landmarks_example.json`](configs/ur7e_xhand/front_landmarks_example.json)：tip 组只有条件可见的 wrist-3 接口中心，至少要在 6 个姿态中清楚可见，否则请改用已测量的物理标记。其他标记可用 `--landmarks your_landmarks.json` 指定，每项含 `name`、`group` (`base`/`tip`)、`link` 和 `local_xyz_m`。旧项目明确指出“指尖 link 原点”通常不是可直接点击的物理点，**不要拿它冒充指尖标注**；要拟合真正指尖，需先测定可见标记在对应 link 下的坐标。每组至少 6 个清楚的标注，并尽量覆盖不同姿态和图像区域：

```bash
PYTHONPATH=src python scripts/spatial/front_calibration_workflow.py fit \
  --stage base --annotations diagnostics/press_button_0_annotation/front_annotations.json \
  --output diagnostics/press_button_0_annotation/front_base.json

PYTHONPATH=src python scripts/spatial/front_calibration_workflow.py fit \
  --stage tip --annotations diagnostics/press_button_0_annotation/front_annotations.json \
  --base-profile diagnostics/press_button_0_annotation/front_base.json \
  --output diagnostics/press_button_0_annotation/front_fitted.json
```

第一阶段只拟合 base_link 下 front **平移**残差；第二阶段固定外参，只拟合 front **color** 相机的 `fx/fy/cx/cy`。当前工具不拟合旋转、depth 内参或 depth/color 之间的变换。color K 只影响轮廓回投和点云颜色关联，**不会修正由 depth 内参生成的 3D 坐标**；必须结合交互点云和多帧人工判断空间是否可靠。拟合输出与候选 profile 同格式，可替换 `--profile` 重新跑对照；不同数据集分别保存 profile 和 sidecar 版本。

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

# 新三组消融共用这份完整 visual+tactile sidecar。
python scripts/spatial/export_spatial_derived_dataset.py \
  --dataset data/press_button_4_times_clean \
  --episodes all --frames all --version v1_front \
  --cameras front --num-points 4096 --shard-size 128
```

如使用候选/拟合 profile，把 front-only 导出命令改用**新版本名**并追加 `--front-calibration-profile <profile.json>`，然后同步修改 `FRONT_SPATIAL_VERSION` 与 `FRONT_CALIBRATION_PROFILE`。不能让已纠正的 sidecar 配上未纠正的在线服务；训练 loader 会检查角色和标定参数。导出器默认拒绝覆盖已有目录。可选的统计和输入检查：

```bash
python scripts/spatial/compute_spatial_dataset_stats.py \
  --dataset data/press_button_4_times_clean --version v1_front

python scripts/spatial/smoke_test_xhand_spatial_pipeline.py \
  --config-name pi0_xhand_spatial_structured_suffix_front --sample-index 0
```

更多数据格式和标定约定见 [`docs/spatial_preprocessing_v1.md`](docs/spatial_preprocessing_v1.md)。

### 2. 计算归一化统计

```bash
python scripts/compute_norm_stats.py \
  --config-name pi0_xhand_spatial_structured_suffix_front
```

新三组与旧 Structured 配置共享 `assets/xhand/press_button_4_times_clean/` 下的 state/action 统计；已有该数据集的统计时无需重复计算。checkpoint 保存时也会把统计复制到自己的 `assets/`。

### 3. 启动训练

旧 front+left suffix 实验的命令仍是：

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

三卡消融在三个终端分别运行下面三条命令。三组使用同一 front-only sidecar、同一 seed、预训练权重和 full-finetune 超参，只切换结构化空间分支的输入模态：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  pi0_xhand_spatial_structured_suffix_front --exp-name front_both
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  pi0_xhand_spatial_structured_suffix_front_visual --exp-name front_visual
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
  pi0_xhand_spatial_structured_suffix_front_tactile --exp-name front_tactile
```

上述配置继承旧 suffix 中训练机的绝对 `repo_id`/`weight_loader` 路径，换机器要先改。三路 RGB 和 proprio 始终保留，因此消融结论只针对新增的**结构化空间分支**，不是“完全无视觉/无触觉”的机器人。

## 远程推理与机器人部署

模型在 GPU 机器上运行 [`scripts/serve_policy.py`](scripts/serve_policy.py)，机器人电脑运行 [`deploy/ur7e_xhand_deploy_3d_tactile_multi.py`](deploy/ur7e_xhand_deploy_3d_tactile_multi.py)。front-only 客户端只发送原始 state、三路 RGB、front 深度及 prompt；tactile-only 不发送深度。服务端按配置重建启用的空间分支，再返回 action chunk。两端代码需使用匹配版本。

### 1. GPU 机器启动服务端

```bash
cd /path/to/openpi-3d-tactile
OPENPI_REPO_ROOT="$PWD" CUDA_VISIBLE_DEVICES=0 \
python scripts/serve_policy.py \
  --port 8990 \
  policy:checkpoint \
  --policy.config pi0_xhand_spatial_structured_suffix_front \
  --policy.dir checkpoints/pi0_xhand_spatial_structured_suffix_front/front_both/29999
```

`--policy.dir` 指向包含 `params/` 和 `assets/` 的**具体 step**，`29999` 只是示例。服务端需要本仓库的标定/URDF 资产与训练时相同的 profile，`OPENPI_REPO_ROOT` 指向仓库根目录。服务端监听 `0.0.0.0:8990`。visual-only、tactile-only 换相应的 `--policy.config` 和 checkpoint 路径，不要交叉加载。

### 2. 机器人电脑检查配置

机器人电脑可以只放 `deploy/` 和与训练数据一致的 `meta/info.json`，不必安装整个 OpenPI。先激活已跑通的 LeRobot/UR7e/XHand 部署环境，再进入脚本所在目录的上一级：

```bash
cd /path/to/deployment
PYTHON_BIN="$(command -v python)"

"$PYTHON_BIN" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --dataset-dir /path/to/press_button_4_times_clean \
  --spatial-cameras front \
  --check-config
```

`--dataset-dir` 指向包含 `meta/info.json` 的目录，用于确定原始 state 和 18 维 action 的**字段顺序**。不要依赖脚本默认的 `grasp_pipette` 目录。配置检查不连接机器人，也不验证现场相机数据。

### 3. 先做 dry run，再正式运行

EtherCAT 手通常需要 root 权限；`sudo -E` 搭配已激活环境的 Python，避免切换到系统 Python。替换服务端 IP 和数据集目录：

```bash
sudo -E "$PYTHON_BIN" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip 192.168.1.100 --server-port 8990 \
  --dataset-dir /path/to/press_button_4_times_clean \
  --spatial-cameras front \
  --prompt "press the button 4 times and put it into the box" \
  --hand-protocol EtherCAT \
  --record-dir ./diagnostic_runs --duration 60 \
  --dry-run --no-home --no-tactile-reset
```

`--dry-run` 不下发策略动作，但默认的 home reset 和触觉清零仍是独立流程；上面两个 `--no-*` 参数使联调时跳过它们。确认服务端返回动作、三路 RGB 和 front 深度均正常后，正式运行：

```bash
sudo -E "$PYTHON_BIN" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip 192.168.1.100 --server-port 8990 \
  --dataset-dir /path/to/press_button_4_times_clean \
  --spatial-cameras front \
  --prompt "press the button 4 times and put it into the box" \
  --hand-protocol EtherCAT \
  --record-dir ./diagnostic_runs --duration 60
```

正式运行默认先回零、清零触觉，再等待 Enter 开始。只有在确实需要免确认时才加 `--yes`。多次测试可加 `--run-mode multi --num-runs 0`，每轮按右方向键结束并单独保存诊断目录。机械臂 IP、相机序列号、控制频率等有脚本默认值，平台变更时用 `--help` 查看并覆盖对应参数。

双模态/visual-only front 配置使用 `--spatial-cameras front`；tactile-only 使用 `--spatial-cameras none`；旧 suffix 使用默认 `front,left`。新服务端在 metadata 中声明所需角色，客户端会在连接机器人前校验。机器人正式执行前仍须人工确认安全。

## 失败诊断

启用 `--record-dir` 后，每轮生成 `diagnostic_runs/<时间>_run_<编号>/`，包含 `manifest.json`、`control/*.npz` 和 `inference/query_*_{input,output}.npz`。输入记录保留三路 RGB、**按配置发送的**原始深度、完整 state 和五指触觉；输出记录保留返回的 action chunk、实际使用的 chunk 和耗时；控制记录包含逐帧实际发送动作与 fallback 状态。检查 `manifest.json` 中的 `dropped_records` 和 `writer_errors` 可判断记录是否完整。

在只有 `deploy/` 的机器人电脑上，只要环境有 NumPy、Matplotlib、OpenCV，即可生成汇总图和每次推理图；点云面板会缺失。完整的 4096 点视觉点云及 600 个触觉空间点，需要在有 `src/openpi/spatial` 与标定文件的本仓库中离线重建。报告按 manifest 的相机角色重建；如运行使用了校准 profile，默认从服务端 metadata 读取其相对路径，也可用 `--front-calibration-profile` 明确指定。

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
