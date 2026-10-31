# 3D 重建：手机照片 → 米制 Mesh（新人操作指南）

本工具针对每个任务的**两个关键刚性物体**。每物体一张照片，两个物体共用一套任务级目录。旧任务的透明漏斗流程不改动。本指南默认使用可视化校准页面，**不需要手动编辑 `calibration.json`**。

## 0. 设置环境变量

```bash
export REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
export HOI=/workspace/mnt/sqzhang26/hoi
export DATASET=1028_press_button
export OBJECTS=button,box
export GEOM="$REPO/output/preprocess/$DATASET/geometry"
export GEOM_SCRIPT="$REPO/scripts/preprocess/geometry_task.py"
```

新任务修改 `DATASET` 和 `OBJECTS`。物体名与 SAM2 阶段一致，不要改变文件名或大小写。每开一个新终端需重新设置环境变量。

## 1. 上传照片并得到两张审核通过的照片 Mask

```bash
mkdir -p "$GEOM/photos"
```

上传两张清晰、完整、无遮挡的照片。例如 `$GEOM/photos/button.jpg` 和 `$GEOM/photos/box.jpg`，照片文件名必须对应 `OBJECTS` 中的名字。照片可用 `.jpg/.jpeg/.png`。

```bash
cd "$REPO"
PYTHONPATH=src .venv/bin/python "$GEOM_SCRIPT" prepare \
  --out "$GEOM" --objects "$OBJECTS"
```

下载并打开：

`$GEOM/reference/<物体名>/click.html`

分别在两张图片中点击物体的正点，必要时增加负点。各自下载 `prompts.json`，上传回对应的 `reference/<物体名>/` 目录。

```bash
cd "$HOI/repos/sam2"
"$HOI/envs/sam2/bin/python" "$GEOM_SCRIPT" segment --out "$GEOM"
```

查看各 `reference/<物体名>/preview.jpg`。分割错误先改标注、重新 `segment`；都正确后：

```bash
cd "$REPO"
.venv/bin/python "$GEOM_SCRIPT" approve-masks --out "$GEOM"
```

## 2. 重建与生成可视化校准界面

```bash
cd "$HOI/repos/sam-3d-objects"
"$HOI/envs/sam3d/bin/python" "$GEOM_SCRIPT" reconstruct --out "$GEOM"
```

成功后原始模型保存为：`geometry/reconstruction/<物体名>/raw.glb`。该模型**不是米制，不能直接用于 RGBTrack**。

```bash
cd "$REPO"
.venv/bin/python "$GEOM_SCRIPT" inspect --out "$GEOM"
```

输出单个互动页面：

**`$GEOM/preview/metric_editor.html`**

可下载该 HTML 到本地浏览器打开；页面内置 Plotly 和两份原始 GLB，离线可用。

## 3. 在同一个 HTML 完成方向、尺度和导出

在 `metric_editor.html` 中：

1. 顶部选择第一个物体，拖动右侧三维模型查看形状。
2. 用 **X/Y/Z 旋转滑杆、数字或 ±90° 按钮**将物体转至希望的方向（建议 Z 向上）。这里是旋转模型本身，不是单纯转动相机视角。
3. 在 **X、Y、Z 实际长度**中填写用尺子测得的尺寸，**单位是厘米**。至少填一个方向；选择哪个方向作为等比缩放的基准。若填了多个方向，程序会检查其是否一致。
4. 右侧实时更新 3D 模型，左侧实时显示校准后的三方向尺寸。根据需要设置原点（默认底面中心）。
5. 选择第二个物体，重复以上操作。两个物体的设置会分别保留。
6. 最后点击 **「确认两个物体并导出模型 ZIP」**。浏览器将下载 `geometry_metric_export.zip`，其中包含两个已旋转、已缩放的米制 GLB 和校准记录。

**注意：** 模型外形不对、缺面或比例严重错误时不要通过修改比例硬凑；重新拍摄或使用可靠 CAD。HTML 只做旋转、等比缩放和原点变换。旋转之后的 XYZ 才是你填写实际长度所对应的方向。

## 4. 上传并一键完成云端校准、批准

将刚下载的 `geometry_metric_export.zip` **上传到**：

`$GEOM/geometry_metric_export.zip`

执行一条命令：

```bash
cd "$REPO"
.venv/bin/python "$GEOM_SCRIPT" finalize --out "$GEOM"
```

程序自动：检查 ZIP 来源和原始重建 GLB 的哈希 → 验证实测尺寸误差 → 用云端原始 Mesh 复现校准结果 → 与 HTML 导出的 Mesh 交叉核对 → 将两份正式模型写入 `approved/`。无需手动编辑 JSON、无需再调用 `calibrate` 和 `approve`。

检查结果：

```bash
.venv/bin/python "$GEOM_SCRIPT" status --out "$GEOM"
```

正式输出：

```text
output/preprocess/<DATASET>/geometry/approved/
├── <物体1>.glb
├── <物体2>.glb
├── manifest.json
└── assets_config_snippet.json
```

**后续 RGBTrack 使用 `approved/` 内的 GLB。** 这只是几何（物体局部坐标系），不代表在机器人基座/相机中的 6D 位姿。新任务的 RGBTrack 通用接口还要单独验收。

## 说明 / 常见问题

- **照片 Mask 和视频 Mask 是两套独立结果。** 视频 Mask 在 `sam2/` 下，照片 Mesh 在 `geometry/` 下，二者不会互相覆盖。
- **只填一个尺寸够吗？** 够完成等比缩放。但多填 1～2 个尺寸更容易发现重建几何错误。
- **浏览器能直接写回云端吗？** 不能。HTML 是本地离线文件，所以下载一个 ZIP 并上传回云端；`finalize` 自动生成正式文件。
- **已经跑过 `inspect`，需要重新重建吗？** 不需要。替换新版本后仅重新执行 `inspect`，打开新生成的 `metric_editor.html` 即可。
- **已批准的 Mesh 可以覆盖吗？** 默认拒绝。应保存新版本或显式备份，避免悄悄影响后续追踪。
- **透明或严重反光物体怎么办？** 单张手机照片未必能准确恢复，应该重新拍摄、多视角辅助审核或直接使用 CAD。这不影响已验证的旧透明漏斗任务。
- **技术人员仍然能使用旧命令吗？** 可以。`calibrate` 与 `approve` 保留以兼容已有实验；新人不需要用它们。
