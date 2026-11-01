# RGBTrack 正式数据管线 — 1028_press_button 已验收基线

## 合理边界

- 前提：SAM2 视频 Mask、SAM3D 米制批准 Mesh、前视 D435I SDK 标定与公共深度配准已经生效。
- 动态 `button`：原生 RGBTrack，**不添加毫米级跳变阈值、桌面约束或接触前冻结**。
- 静态 `box`：原有 RGBTrack 252 候选选择中，筛选朝上 + 凸包投影 Mask 匹配，首次通过后固定整条 Episode。
- 凸包 Mask 选择器只对盒状/凸形静态物体验证过。透明/非凸物体（漏斗和烧瓶）不自动使用。
- 统一脚本保留正式入口 `scripts/preprocess/rgbtrack_task.py`，新文件都是该正式入口的内部复用模块。
- `1028_press_button` Episode 0 的旧人工批准输出 `e00000_box_mask_full_v2` 保留不动。正式管线运行输出单独命名，不覆盖旧文件。
- 原有 `rgbtrack/episodes` 老输出不会被隐式替换。通过所有人工验收后，再指定后续训练阶段要读取的批准 run-tag。

## 安装

将 ZIP 解压到仓库 `tmp/`（产生 `tmp/press_button_pipeline_merged_20261031/`），先执行：

```bash
python3 tmp/press_button_pipeline_merged_20261031/install_pipeline.py --repo "$PWD" --check
```

`[PREFLIGHT OK]` 后才能安装：

```bash
python3 tmp/press_button_pipeline_merged_20261031/install_pipeline.py --repo "$PWD" --apply
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_rgbtrack_pipeline_unified.py tests/test_rgbtrack_static_policy_unified.py
```

安装器只接受用户刚上传的云端基线 SHA256 `a0eb323351e94868cd955d0d948e81978b993d393410913cc92ec909a64fcff5`（该文件保留 `.absolute()` 修复），或已安装的新版本；其他版本拒绝覆盖。不会修改 RGB-D 标定文件、SAM2/3、RGBTrack 权重或旧实验结果。自动备份路径打印在安装结果中。

## 先跑已验收任务的 Episode 0（回归）

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
export HOI=/workspace/mnt/sqzhang26/hoi
export REPO=$PWD
export DATASET=1028_press_button

PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
 --dataset "$DATASET" check --episodes 0

PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
 --dataset "$DATASET" trial --episodes 0 \
 --dynamic-object button --static-object box \
 --run-tag press_sdk_static_v1 --review
```

生成：

```
output/preprocess/1028_press_button/rgbtrack/runs/press_sdk_static_v1/episodes/e00000/
  button_poses.jsonl
  box_poses.jsonl
  manifest.json
  task_input_provenance.json
  quality_summary.json
  box_init_candidates.png
  box_init_candidates.json
output/preprocess/1028_press_button/rgbtrack/reviews/press_sdk_static_v1/e00000/review.html
```

`review.html` 与旧版完整 3D 可视化一致。确认首帧 box 不倒置、全程 static_frozen，button 可见阶段正常，遮挡恢复无明显错误；不要把 `valid=true` 当成姿态真值。与先前 `e00000_box_mask_full_v2` 对比。

## 代表 Episode（仅在 E0 人工通过后）

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
 --dataset "$DATASET" trial --episodes 23,50,76,99 \
 --dynamic-object button --static-object box \
 --run-tag press_sdk_static_v1
```

每条完成会生成 `quality_summary.json`：状态分布、有效帧比例和相邻有效帧大跳变；不施加新跳变限制。要看旧版 HTML，用：

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
 --dataset "$DATASET" review --episodes 23,50,76,99 \
 --run-tag press_sdk_static_v1
```

## 全量（必须先人工确认代表 Episode）

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
 --dataset "$DATASET" batch --episodes all \
 --dynamic-object button --static-object box \
 --run-tag press_sdk_static_v1
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
 --dataset "$DATASET" status --run-tag press_sdk_static_v1
```

`batch` 按输入 provenance 完整校验后跳过已经完成且匹配的 episode；遇到部分输出、参数漂移、掩码/标定改动则直接阻止覆盖。批量默认不生成巨大 HTML。

## 回滚

先记录 `--apply` 打印的 `[BACKUP]` 路径，然后：

```bash
python3 tmp/press_button_pipeline_merged_20261031/install_pipeline.py \
 --repo "$PWD" --rollback "<backup目录>"
```

回滚前会核对已安装文件的 SHA256，避免覆盖安装后新修改的代码。

## 尚未完成的事情

RGBTrack 全量验收完成后，需将**批准的 run-tag** 接入后面的原生空间点云、手物关系 / 触觉 token 和 π0 回放。**本包只整合 RGBTrack 正式管线，不声称 token 链路已经完成**；旧的三维位姿/点云副本不会自动成为新训练输入。


## Dynamic motion guard (opt-in experiment)

A two-sided online sanity gate can be enabled on the **dynamic** object only:

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
  --dataset 1028_press_button trial --episodes 0 \
  --dynamic-object button --static-object box \
  --guard-max-step-cm 1.2 --guard-max-turn-deg 35 \
  --run-tag press_motion_guard_v1 --review
```

A candidate exceeding either per-frame threshold relative to the last trusted
observation is discarded as `valid=false, status=motion_rejected_translation`
or `motion_rejected_rotation`. The adapter's causal predictor is rolled back,
RGBTrack is switched to re-registration on the following frame, and the
rejected pose is **never clipped to an apparently plausible value**.
The next registration must also pass the same check against the last trusted pose.
Both threshold values are provisional; compare observed genuine motion before
accepting these parameters. Repeated rejections are a failure, not success:
review `motion_guard_report.json` and 3D HTML. Native low-quality observations
are also rejected to prevent poisoning causal prediction.

This guard is not a proof of physical correctness, and cannot detect every
small systematic drift. Prior accepted runs are not overwritten: always use
new `--run-tag`. Legacy outputs are not automatically training-approved.
