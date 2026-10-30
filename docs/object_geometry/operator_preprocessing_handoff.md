# 新数据预处理交接（保留已验证的 HOI / RGBTrack / Sidecar 链路）

> **适用范围**：`openpi-3d-tactile` 的漏斗（`funnel`）插入目标烧瓶（`target_flask`）任务；锥形瓶仅是起始支架。首次使用必须先用**一条**新轨迹进行人工审查。文档不是“全自动新物体重建”，也不是实机部署通过证明。
>
> **此次修改不改变** `v1_hand_scene_q2` 68-token 契约（33 场景点云 + 25 空间触觉 + 5 手—漏斗 + 5 漏斗—目标），不覆盖既有 `v1_hand_scene_q2` 或 `v1_hand_scene_dtcr_v1` 数据。

## 0. 先确认哪三件事没有变化

1. **相机/机器人坐标系**：RGB 与 depth 已对齐，front 相机使用经过核验的内外参；不要把未标定的腕部相机加进来。新机位必须重新标定、审查 RGB-D 对齐与 `base_link` 转换。
2. **物体几何**：真实使用的还是同一只漏斗、同一只目标烧瓶吗？如果是，可复用已审核的米制 CAD（`assets/object_geometry/synthware_19_22`）；不需要逐 episode 重建。若更换物体，必须先在 HOI/SAM-3D + MoGe 的**已有工作流**中重建、校正尺度/坐标轴、审核网格，再更新新数据配置的 `assets`。现有 `openpi.object_geometry.cli mesh-build` 和 `mesh-calibrate` 对旧数据仅返回 `skipped_approved_assets`，**不是新的 SAM-3D 推理命令**。
3. **位置分组与目标假设**：`1018_peg_all` 配置只审核了 episode `0–69` 的六组位置，并采用目标烧瓶静止锚点（frame 0），不能直接给新位置/新相机复用模板。新数据建独立配置，指定新的 `episodes.all`、`position_groups`、`initialization_templates`、`paths`、`rgbtrack.scale_manifest`、`assets`、校准 provenance 和 QA 门槛；需要人工审核初始化模板，不能依赖自动分类。

## 1. 环境（云端/真实预处理电脑）

在代码库根目录，以下路径换成实际位置：

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
export REPO="$PWD"
export HOI="/workspace/mnt/sqzhang26/hoi"
export HOI_ROOT="$HOI"
export OPENPI_1018_DATASET_ROOT="$REPO/data/1018_peg_all"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$REPO/.venv/bin/python"
CFG="$REPO/configs/object_geometry/datasets/1018_peg_all/config.json"
EP=23
```

这是**原 1018 数据集演示**。操作真正的新数据时，`OPENPI_1018_DATASET_ROOT` / `CFG` 要指向新建的同构数据目录及其独立配置；不要直接编辑旧配置、旧已确认数据或旧输出版本。`PY`、HOI 的 SAM2 环境和 sam3d/RGBTrack 环境都是已有环境，本次补丁不会安装它们。

## 2. 逐阶段执行（一个代表性 episode）

### A. 从 LeRobot 轨迹到 SAM2 mask（HOI Level-1）

先确认前/左相机图像、状态、触觉和帧 ID 对得上；对齐 depth 到 RGB，并核对 RGB camera intrinsics。HOI Level-1 原有分步操作：

```bash
bash scripts/object_geometry/level1_ep.sh "$EP" prepare
bash scripts/object_geometry/level1_ep.sh "$EP" seed
bash scripts/object_geometry/level1_ep.sh "$EP" annotate
bash scripts/object_geometry/level1_ep.sh "$EP" preview
# 必须人工检查，确认 funnel mask 不含手或初始支架，target_flask 为真正的目标容器
bash scripts/object_geometry/level1_ep.sh "$EP" approve
bash scripts/object_geometry/level1_ep.sh "$EP" track
bash scripts/object_geometry/level1_ep.sh "$EP" review-track
```

**注意**：这只是调用现有 HOI 脚本，不会自动作出人工审批。优先查看初始、第一次接触、抓起、遮挡、插入、释放时的 mask 和帧；检查被支架遮住的漏斗颈、透明烧瓶口、手是否被错误并入掩码。历史的 `level1_batch_preview_all.sh` 固定范围 `0–69`，新数据不要运行它。

如果要为 scene point / spatial tactile 数据构建旧版 HOI 表示，再调用**原有**的两个阶段（不是新实现）：

```bash
bash scripts/object_geometry/level1_ep.sh "$EP" build
bash scripts/object_geometry/level1_ep.sh "$EP" memory
```

已有空间特征须按数据集配置 `paths.spatial_root` 和 `paths.spatial_version` 检查到位；缺失触觉不应静默伪装为有效触觉（旧配置 `qa_thresholds.allow_missing_tactile=false`）。

### B. 确认模型资产（SAM-3D/MoGe 或已审查 CAD）

**同一物体**：复用已校准的 `funnel_F181975.glb`、`flask_F301950.glb`、`dimensions.json`、scale manifest；使用已有 `mesh-approve` 验证 SHA/尺度声明：

```bash
"$PY" -m openpi.object_geometry.cli mesh-approve --config "$CFG" --approved-by "$USER"
```

**不同物体**：需要先通过原 HOI/重建环境中的 **SAM2 mask → SAM-3D + MoGe 单次几何重建 → 人工检验/尺度/坐标轴校正 → 受控 mesh 资产** 流程；项目并没有一个已证实的一键新物体重建命令。请勿在此处直接按旧网格走 RGBTrack；不要逐帧运行 SAM-3D。

### C. 核验 mask 帧数和内容

```bash
"$PY" -m openpi.object_geometry.cli mask-preview --config "$CFG" --episodes "$EP"
# 帧数/命名的快速检查（仍需人工看实际 mask）：
MASK_ROOT="$HOI/outputs/level1_sequence_e${EP}/masks" \
  bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh "$EP" check
```

`mask-preview` 生成的 `mask_review.html` 只是**路径/覆盖率报告**，不是 mask 叠加效果验证；真正视觉检查使用 HOI 的 preview/review-track。**e23 特例**：原配置默认 `E23_MASK_ROOT=../hoi/outputs/level1_sequence_e23_maskfix_v1/masks`，与一般 `level1_sequence_e23/masks` 路径不同。请将这两处 `MASK_ROOT` 设置成**同一份人工确认的 mask**，避免预处理/Sidecar 不一致。

### D. RGBTrack 一次跟踪 + 静止烧瓶先验（默认不生成大 HTML）

原已验证流程保持不变：`rgbtrack_native_upright.py` → `rgbtrack_flask_pivot.py` → `freeze_target_flask_anchor.py` → `summarize_cross_episode_hand_scene.py`。每条轨迹仍有 funnel/target flask 的逐帧 JSONL 及 manifest；目标烧瓶 frame-0 锚定是任务假设，不是测出的逐帧动态位姿。

```bash
TAG="pilot_e${EP}_$(date +%Y%m%d_%H%M%S)"
MASK_ROOT="$HOI/outputs/level1_sequence_e${EP}/masks" \
DATASET="$OPENPI_1018_DATASET_ROOT" \
RENDER_HTML=0 RUN_TAG="$TAG" \
bash scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh "$EP" run

POSE_RUN="$REPO/output/pose_eval/e${EP}_crosscheck_static_target_${TAG}"
cat "$POSE_RUN/crosscheck_summary.json"
```

如果 `EP=23`，请将 `MASK_ROOT` 改为实际已核验的 `maskfix_v1` 路径。**新数据必须额外传入正确的 `CAMERA`、`METRIC`、`FUNNEL`、`FLASK`、`FRAMES`、`DATASET`**，或者优先使用下节基于独立配置的 CLI batch（会传入这些路径）。不能让新数据隐式沿用旧参数。

### E. 需要时单独渲染 HTML（不重新跑 RGBTrack）

```bash
"$PY" scripts/object_geometry/pose_tracking/visualize_rgbtrack_full_episode.py \
  --repo-root "$REPO" \
  --dataset "$OPENPI_1018_DATASET_ROOT" --episode "$EP" \
  --pose-run "$POSE_RUN" \
  --frames-dir "$HOI/data/level1_sequence_e${EP}/frames" \
  --output "$REPO/output/hand_scene_qa/review_e${EP}_${TAG}" \
  --viz-stride 5
```

用于 spot-check 少数轨迹（initial/contact/occlusion/insertion/release/recovery）；**不要为每个 episode 逐帧生成巨型 HTML**。旧可视化脚本可能把文件命名为 `hand_scene_dynamic_e23.html`，即使处理的是别的 episode；按指定输出目录和 manifest 判定，不要只看文件名。单条老式全量可视化仍可 `RENDER_HTML=1` 显式选择。

### F. 按经过审核的 mask/pose 导出 Sidecar

```bash
PILOT_VERSION="v1_hand_scene_operator_pilot_$(date +%Y%m%d)"
"$PY" -m openpi.object_geometry.cli batch --config "$CFG" \
  --episodes "$EP" --no-run-tracking --pose-run "${EP}=${POSE_RUN}" \
  --auto-qa --resume --output-version "$PILOT_VERSION"
"$PY" -m openpi.object_geometry.cli qa --config "$CFG" \
  --episodes "$EP" --output-version "$PILOT_VERSION"
```

- 如 mask 路径与独立配置不一致，同时加 `--mask-root "${EP}=${MASK_ROOT_PATH}"`。批处理需要对应的 spatial/tactile 数据；失败会产生 `scratch/.../batch/batch_report.json` 以及小型 HTML 失败报告；状态可能是 `pending_review`，**不代表可以强行通过**。
- `--resume` 只在配置哈希、mask 内容/位置、pose manifest provenance 一致时复用。重新标注或重新跟踪时，换一个 `output-version`；不要覆盖历史 `v1_hand_scene_q2` 和 `v1_hand_scene_dtcr_v1`。
- 生成的关系 token 基于已审核 mesh + robot FK + taxel/触觉空间数据，沿用训练数据已有实现；本补丁**没有修改**模型融合、token 数量、接触门控或动作训练逻辑。

## 3. 大批量之前的审批门槛

1. 抽样人工检查每个新位置/新机位的 SAM2 mask，尤其首次接触与遮挡。
2. 校准 profile 与 RGB/depth 配准和网格尺度一致；首帧 `T_base_link_object` 合理，漏斗轴不躺倒；烧瓶口/漏斗颈相对几何合理。
3. `crosscheck_summary.json` 的无效追踪区间已检查；`valid` 或 tracker score **不等于**真实位姿精度。
4. 模型资产、静止锚点语义、触觉与状态帧数都一致；侧写 `batch_report.json` 中 QA 阈值满足而不是被跳过。
5. 用一个**新的**输出版本做 pilot，之后才在 config 审核范围内 `batch --episodes <范围> --resume --auto-qa`。全量运行默认 `RENDER_HTML=0`，失败时只生成轻量诊断。

## 4. 改动范围与未解决问题

此次补丁只改：

- `scripts/object_geometry/pose_tracking/run_cross_episode_hand_scene.sh`：参数化数据/标定/网格路径，默认 `RENDER_HTML=0`，总保留 JSON summary；不再人工限制 `EP<=69`（真正的新数据合法性仍由 config/人工审核控制）。
- `scripts/object_geometry/level1_ep.sh` + `tools/hoi_level1_snapshot/scripts/level1_ep.sh`：把外层路径实际传递到原有 HOI 脚本。
- `src/openpi/object_geometry/cli.py` 的 batch 运行环境：传递真实数据集、相机、网格和 scale manifest。对非 `1018_peg_all` 数据集，未配置 `rgbtrack.scale_manifest` 时拒绝启动跟踪，避免隐式使用旧 manifest。

不会自动跑 SAM-3D/MoGe、不会调参优化位姿，不修改旧 dataset config 和既有 Sidecar，不提供未经实机验证的精度保证。**特别需要云端进行 `bash -n`、CLI 帮助/静态测试，再用 1 条新轨迹进行 SAM2 + RGBTrack 真正验证。**
