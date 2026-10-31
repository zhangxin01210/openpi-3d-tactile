# SAM2 两物体预处理：通用新人操作流程

> 适用：非透明、外观较容易匹配的双物体任务。**不替代** `1018_peg_all` 原有的六位置模板流程。所有命令仅涉及 SAM2、Mask 和审核，不触碰 Spatial、3D Mesh、RGBTrack、Sidecar 或原始 LeRobot 文件。
>
> 状态：通用入口已通过离线合成数据测试，**尚需在 A800 运行真实双物体 SAM2 验收**。

## 0. 目录与环境

- 仓库：`/workspace/mnt/sqzhang26/openpi-3d-tactile`
- 数据：`data/1028_press_button`（只读）
- 输出：`output/preprocess/1028_press_button/sam2`（全新目录，不覆盖旧实验和 `hoi/outputs`）
- 通用脚本：`scripts/preprocess/sam2_task.py`
- `prepare` / `video-prepare` 使用仓库的 `.venv/bin/python`；`seed` / `firstframes` / `video-track` 使用 HOI SAM2 Python，**工作目录须为** `$HOI/repos/sam2`。审核命令任一环境均可。
- 两个物体名称由人确认。下文仅以 `button,fixture` **举例**，请替换成此任务实际要建模的两个对象，不要凭空把无关支架标成第二物体。

```bash
export REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
export HOI=/workspace/mnt/sqzhang26/hoi
export TASK_SAM2="$REPO/output/preprocess/1028_press_button/sam2"
export SCRIPT="$REPO/scripts/preprocess/sam2_task.py"
```

## 1. 参考首帧：只需标两个物体，通常各一个正点

在仓库目录使用 OpenPI Python：

```bash
cd "$REPO"
PYTHONPATH=src .venv/bin/python "$SCRIPT" prepare \
  --dataset "$REPO/data/1028_press_button" \
  --out "$TASK_SAM2" --objects button,fixture \
  --camera front --sample-count 27
```

- 提取全部 episode 的**首帧 JPG 缓存**，但先只选择均匀采样的 27 条做 SAM2 首帧测试；不是按任务位置分组。
- 打开 `reference/reference_click.html`，选择两个物体，各点至少一个正点；分割困难可增加正/负点。下载 `prompts.json` 并上传至 `$TASK_SAM2/reference/prompts.json`。
- 这份 HTML 是本地静态文件，无需启动服务器，也不需要输入像素坐标。

## 2. SAM2 参考 Mask + 人工审核（必须）

```bash
cd "$HOI/repos/sam2"
"$HOI/envs/sam2/bin/python" "$SCRIPT" seed --out "$TASK_SAM2"
```

查看 `reference/preview.jpg`：两种颜色分别表示两个物体。若 Mask 不好，重新点击/补负点、替换 `reference/prompts.json`，然后在**审批前**重跑 `seed`。确认两者正确后：

```bash
"$HOI/envs/sam2/bin/python" "$SCRIPT" approve-reference --out "$TASK_SAM2"
```

这会写入 `reference/review.json`，绑定提示点、参考帧、两个 Mask 的哈希。**不会自动批准**。

## 3. 跨 episode 首帧预览：先少量、后全量（两个审核关口）

```bash
"$HOI/envs/sam2/bin/python" "$SCRIPT" firstframes \
  --out "$TASK_SAM2" --episodes sampled --name sampled
```

审核 `firstframes/contact_sheet_sampled.jpg`（多页时会有 `_00.jpg`、`_01.jpg`）。确认两个物体在采样的每条 episode 中均正确，再：

```bash
"$HOI/envs/sam2/bin/python" "$SCRIPT" approve-firstframes \
  --out "$TASK_SAM2" --name sampled
```

如果报告含歧义/交叠警告，但你确实看图确认正确，可以在**人工审核后**添加 `--accept-warnings`；不要为了过脚本而盲目添加。

现在扩展到全部首帧：

```bash
"$HOI/envs/sam2/bin/python" "$SCRIPT" firstframes \
  --out "$TASK_SAM2" --episodes all --name all
```

查看 `firstframes/contact_sheet_all_00.jpg` 等；如果全部确认正确：

```bash
"$HOI/envs/sam2/bin/python" "$SCRIPT" approve-firstframes \
  --out "$TASK_SAM2" --name all
```

新的全量首帧任务会**自动复用已成功生成的采样结果**；不重跑已处理 episode。所有首帧都是独立的 SAM2 推理；不需要预设九宫格。

## 4. 少量视频验证，再批量追踪

先测试 5 条代表 episode；这些 ID 仅是本任务示例，可以修改：

```bash
cd "$REPO"
PYTHONPATH=src .venv/bin/python "$SCRIPT" video-prepare \
  --out "$TASK_SAM2" --episodes 0,23,50,76,99

cd "$HOI/repos/sam2"
"$HOI/envs/sam2/bin/python" "$SCRIPT" video-track \
  --out "$TASK_SAM2" --episodes 0,23,50,76,99
```

打开各 `episodes/eXXXXX/tracking_contact_sheet.jpg`，检查手靠近、接触、按压、遮挡和回撤阶段；确认后：

```bash
"$HOI/envs/sam2/bin/python" "$SCRIPT" video-review \
  --out "$TASK_SAM2" --episodes 0,23,50,76,99
```

如果有自动异常警告，先人工核查；必要时追加 `--accept-warnings` 才允许审批。

5 条通过后，全量操作：

```bash
cd "$REPO"
PYTHONPATH=src .venv/bin/python "$SCRIPT" video-prepare --out "$TASK_SAM2" --episodes all

cd "$HOI/repos/sam2"
"$HOI/envs/sam2/bin/python" "$SCRIPT" video-track --out "$TASK_SAM2" --episodes all
```

抽样查看所有轨迹的 QA 图，并逐批执行 `video-review --episodes <ID或范围>`。对已经审批过的 episode 会保留原审核结果，重复操作时安全跳过。**自动 QA 不是人工批准**。

## 5. 正式目录结构

```text
output/preprocess/<dataset-id>/sam2/
├── manifest.json                         # 任务、两物体、相机、全部 episode/采样列表
├── reference/
│   ├── reference_frame.jpg
│   ├── reference_click.html              # 两物体点击标注
│   ├── prompts.json                      # 由网页下载/上传
│   ├── templates/<object_name>.png       # 人工审核过的参考 Mask
│   ├── preview.jpg
│   ├── candidate_report.json
│   └── review.json                       # 人工批准与哈希
├── firstframes/
│   ├── source/e00000.jpg                  # 全部首帧缓存；不是原始数据修改
│   ├── e00000/<object_name>.png
│   ├── e00000/report.json
│   ├── contact_sheet_sampled.jpg
│   ├── contact_sheet_all_00.jpg ...
│   ├── report_sampled.json / review_sampled.json
│   └── report_all.json / review_all.json
├── episodes/
│   └── e00000/
│       ├── prepare.json
│       ├── masks/
│       │   ├── <object_1>/00000.png, 00001.png, ...
│       │   └── <object_2>/00000.png, 00001.png, ...
│       ├── tracking_contact_sheet.jpg
│       ├── tracking_report.json          # 完整轨迹与 seed 哈希
│       └── review.json                    # 人工批准
└── cache/video_frames/e00000/*.jpg      # SAM2 视频推理临时帧缓存
```

**向后续 RGBTrack 提供的 Mask 根目录**（每条 episode 各不相同）：

`$TASK_SAM2/episodes/e00000/masks`

内部保持以前的 `<object_name>/00000.png` 结构。后续 RGBTrack 仍需改成从当前任务配置读取物体名称，不得把本任务叫作 funnel/target_flask。

## 6. 失败、返工、空间占用

- `status` 查看粗略进度：`python "$SCRIPT" status --out "$TASK_SAM2"`。
- `prepare` 拒绝覆盖任何非空输出目录。已经批准的参考不能直接换提示点；如更换参考/物体，应使用新的 task SAM2 输出版本（例如 `.../sam2_v2`）。
- 自动首帧失败会留下 `firstframes/eXXXXX/` 的**未完成目录**；先检查失败原因，修复后**只对该目录做有意识的移除/重建**，不要删除已经审核的输出。不提供一键删除。
- 视频推理失败可能留下 `episodes/eXXXXX/masks.incomplete/`。这不属于已审核数据；人工检查后只处理该 episode，不可直接当成训练用 Mask。
- `cache/video_frames` 是推理帧缓存，占空间但可重建。确认任务已完成且后续不再复审时，可手工清理此缓存；`episodes/` 和人工审核记录应保留。
- 脚本仅利用图像模板定位：目标外观相似、遮挡、透明或背景混淆会失败；不要用自动分数替代人工看图。
- **旧透明物体** `1018_peg_all` 继续走它已经审核的六位置提示模板、旧 HOI 输出与原来的 Sidecar，完全不需要迁移。

## 7. 真实机测试边界

此次集成只做了离线 API/数据契约测试。尚需在云端真实 SAM2 环境对新任务两物体检查一次参考预览、27 条首帧、至少 3～5 条视频。等这三项通过后，再视情况跑完 100 条。
