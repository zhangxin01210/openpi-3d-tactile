# 新任务 RGBTrack：通用双物体位姿跟踪（A800 候选版）

**状态：第一版通用接口，离线测试通过；尚未在 `1028_press_button` 上通过真实 RGBTrack 推理。** 不要在获得 A800 试验结果之前直接全量跑 100 条。旧透明漏斗 `1018_peg_all` 继续使用原来的 upright/静态烧瓶追踪脚本，**不要迁移或覆盖**。

## 0. 环境变量（每个新终端设置一次）

```bash
export REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
export HOI=/workspace/mnt/sqzhang26/hoi
export DATASET=1028_press_button
export RGB_SCRIPT="$REPO/scripts/preprocess/rgbtrack_task.py"
cd "$REPO"
```

此阶段会复用：

- `output/preprocess/$DATASET/sam2/episodes/eXXXXX/masks/<物体名>/00000.png` 等逐帧 Mask；不强制逐条 `video-review`，但要求追踪报告和所有帧都齐全。
- `output/preprocess/$DATASET/geometry/approved/<物体名>.glb`，要求 `human_accepted`、哈希一致、单位米。
- 原始 `data/$DATASET` 的 front RGB 和 depth，以及已验证的 front 标定。
- 现有的 `scripts/object_geometry/pose_tracking/rgbtrack_adapter.py` 与原有 RGBTrack 权重、`mycpp` 等依赖；**没有新的模型或训练。**

**重要**：首次试验请确保前置的相机标定文件仍适用于新数据。照片生成 Mesh 时的手机相机不参与位姿估计；RGBTrack 使用的是机器人 front RGB-D 与内外参。不能因为模型看起来正确就默认位姿准确。

## 1. 安装通用适配器改动（仅在首次整合时）

候选包已经提供 `scripts/preprocess/upgrade_rgbtrack_adapter.py`。先检查补丁是否兼容 A800 上的最新版代码：

```bash
cd "$REPO"
.venv/bin/python scripts/preprocess/upgrade_rgbtrack_adapter.py --repo "$REPO" --check
```

它会显示具体修改的 diff。如果提示 `Adapter changed`，**停止，不要硬套补丁**，需要用云端最新源码定位差异。检查通过再应用：

```bash
.venv/bin/python scripts/preprocess/upgrade_rgbtrack_adapter.py --repo "$REPO" --apply
.venv/bin/python -m py_compile scripts/object_geometry/pose_tracking/rgbtrack_adapter.py
```

补丁会在 `tmp/rgbtrack_adapter_before_generic_20261031.py` 备份原文件。只新增 `--mesh name=approved.glb`、放开物体名字、增加 `--skip-preview-html`，**旧 `--funnel-glb` / `--flask-glb` 等参数仍可使用**，既有旧流程不需要更改。

运行新测试：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_rgbtrack_task_generic.py' -v
```

## 2. 检查 SAM2 和已批准的 Mesh

```bash
.venv/bin/python "$RGB_SCRIPT" --dataset "$DATASET" check --episodes 0,23,50,76,99
```

通过后应打印 `[CHECK OK]`，并显示两物体名称与 Mask 帧数。先确认 camera='front'，不要用未经标定的腕部相机。

## 3. 代表性 Episode 跟踪试验

先打印将执行的命令（不会跑 GPU）：

```bash
.venv/bin/python "$RGB_SCRIPT" --dataset "$DATASET" trial \
  --episodes 0,23,50,76,99 --dry-run
```

确认调用原有 `rgbtrack_adapter.py`、从已批准 GLB 和新 Mask 取数后，实际运行：

```bash
.venv/bin/python "$RGB_SCRIPT" --dataset "$DATASET" trial \
  --episodes 0,23,50,76,99 \
  --front-calibration "$REPO/configs/ur7e_xhand/front_calibration_candidate.json"
```

入口会自动切到已有 HOI 的 `sam3d` Python 环境跑 RGBTrack；无需用户手动切 Python。默认使用 `--mode depth`、front 相机，不套用旧漏斗的 upright 约束，不冻结某个物体。

各 Episode 输出：

```text
output/preprocess/<DATASET>/rgbtrack/episodes/e00023/
├── manifest.json
├── button_poses.jsonl
├── box_poses.jsonl
├── rgb_overlays/...
├── preview.html  # 位姿/来源 JSON 诊断页，不是旧版交互式完整 3D QA
├── tracking_report.json
└── task_input_provenance.json
```

**检查重点**：打开 `rgb_overlays/` 中的抽样投影图片检查首帧及中间帧的 Mesh 位置。当前 `preview.html` 是 JSON 诊断页，**不是**旧任务带灵巧手的交互式 3D 回放；后者尚需单独通用化。首帧 mesh 在 front RGB 上的重投影是否正确；接触/遮挡中是否出现 6D 位姿翻转、平移抖动和追踪无效。有效状态 `valid=True` 不是准确性的证明。已有旧任务 RGBTrack 专用 upright 约束不能直接用于新物体。

## 4. 通过代表性试验后才批量

```bash
.venv/bin/python "$RGB_SCRIPT" --dataset "$DATASET" batch --episodes all
```

已经完成且输入来源一致的 Episode 自动跳过，其余 Episode 不生成大型 HTML，只生成 6D pose JSONL、简要汇总与必要追踪产物。若某条输出不完整，**拒绝覆盖**；要先调查失败原因。

```bash
.venv/bin/python "$RGB_SCRIPT" --dataset "$DATASET" status
```

所有 JSONL 的 `T_base_object` 是基于 front 标定变换到 `base_link` 的物体位姿，单位米。每个对象独立输出，与旧 `funnel_poses.jsonl` 的记录布局兼容，**但下游 Sidecar 的对象角色映射仍需单独适配和验证，不能直接宣布训练端全链路完成**。

### 注意事项

- 当前使用“两个物体独立 RGBTrack”作为通用基线，不预设任意物体是静止的。若某个任务需要静态锚点、运动约束、重注册，等试验观察实际失效模式后再做任务配置化，避免复刻漏斗特判。
- 如果相机被移动，必须更新或验证 front 内外参；不能复用旧标定直接生成 `base_link` 真值。
- 如果 `--check` 发现 Mesh 哈希、SAM2 Mask 帧数不符，必须修前置数据。追踪完成后还保存 Mesh、SAM2 报告及标定的 SHA256，恢复时检查防误复用。
- `--max-frames` 仅用于临时调试，部分结果不会自动当作完整 Episode；不要用在正式试验或全量命令上。
