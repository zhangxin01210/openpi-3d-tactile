# Level 1: 多 Episode / 六位置模板复用（v2）

**不改训练代码、不改标定、不覆盖 Episode 0。** 在 A800 运行；先把包中的五个脚本和 `level1_ep.sh` 放进 `/workspace/mnt/sqzhang26/hoi/scripts/`。依赖此前已经放进去的 `level1_object_cloud_probe.py`。

## 关于六个位置

Mask 是具体像素位置，**不能跨位置直接复制**。这里从已验收 Episode 的 **首帧 RGB + 二值 Mask** 截取目标及附近支撑装置，做多尺度模板匹配，仅估计图像平移和尺度，映射以往提示点；**SAM2 重新生成新 Episode 首帧 Mask**，随后单向向未来帧传播。模板匹配可能误匹配同形状的支架：`template_match_report.json` 有分数及空间歧义差值，但**分高也需要看首帧预览**。默认 e0 作为模板；新位置有验收成功的 Episode 后，它会进入模板库。假如某个位置失败，就在该位置代表 Episode 标一次，之后其他同位置 Episode 尽量复用。

## 建议流程（例如 Episode 1、2、3、4、5、6）

**A. 先解码并尝试模板种子**，依次执行（换成真实存在的 Episode ID，不必一定连续）：

```bash
HOI=/workspace/mnt/sqzhang26/hoi
for ep in 1 2 3 4 5 6; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" prepare
  bash "$HOI/scripts/level1_ep.sh" "$ep" seed
  bash "$HOI/scripts/level1_ep.sh" "$ep" annotate
done
```

每个 Episode 生成：`hoi/outputs/level1_sequence_e<ID>/template_seed_preview.jpg`、`template_match_report.json`、`annotate_frame0.html` 和 `seed_prompts.json`。

如 `seed` 提示匹配分数偏低，下载 `annotate_frame0.html` 用浏览器调整点（它预载了建议点），导出的 `seed_prompts.json` 上传覆盖同目录文件。注意：**两种物体都要标，右侧目标烧瓶 ≠ 左侧初始锥形瓶**。

**B. SAM2 首帧分割预览**（这步在 GPU 上，耗时取决于视频长度）：

```bash
for ep in 1 2 3 4 5 6; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" preview
done
```

分别查看 `hoi/outputs/level1_sequence_e<ID>/mask_preview_frame0.jpg`。目标烧瓶/漏斗错分、漏分时，返回该 Episode 的 `annotate_frame0.html` 编辑，上传修正的 `seed_prompts.json`，重新跑 `preview`。已批准的文件会在预览重跑时作废。

**C. 仅对人工确认首帧分割正确的 Episode，执行下面命令：**

```bash
for ep in 1 2 3 4 5 6; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" approve
  bash "$HOI/scripts/level1_ep.sh" "$ep" track
done
```

若 `approve` 因掩码面积警告而报错，应先检查，不要盲目跳过。**警告并不代表错误，也不证明正确。** 使用 `level1_sam2_multi.py approve --episode <ID> --accept-warnings` 显式接受已检查过的警告。

全程追踪后检查 `tracking_contact_sheet.jpg`，尤其接触/遮挡阶段。**跟踪图如有错误，禁止运行训练导出**。

**D. 再人工检查关键帧追踪图。** 不通过的 Episode 暂停，不要送入点云。确认无误后逐条运行：

```bash
for ep in 1 2 3 4 5 6; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" review-track
done
```

有跟踪面积异常警告时，`review-track` 默认拒绝批准；如果看过原始视频和关键帧、确认是遮挡造成的合理变化，可使用 `level1_sam2_multi.py review-track --episode <ID> --accept-warnings`。这仅标记人工验收，并不证明每一帧几何正确。

**E. 生成逐帧候选点云、A/B/C 及汇总：**

```bash
for ep in 1 2 3 4 5 6; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" build
  bash "$HOI/scripts/level1_ep.sh" "$ep" memory
done
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
PYTHONPATH=src .venv/bin/python \
  "$HOI/scripts/level1_multi_summary.py" --episodes 1,2,3,4,5,6
```

每个 Episode 对应：

```
hoi/outputs/level1_sequence_e<ID>/sequence_3d.html
hoi/outputs/level1_sequence_e<ID>/cloud_frames/*.npz
hoi/outputs/level1_sequence_e<ID>/abc_v1/contact_memory_abc.html
hoi/outputs/level1_sequence_e<ID>/abc_v1/frames/*.npz
```

汇总：`hoi/outputs/level1_multi_report/summary.csv` 和 `summary.json`。

**重要限制：** 当前 Level 1 ABC 使用“静止目标烧瓶 + 漏斗抓持时近似刚性”的任务假设。**同一套阈值不因位置而调整**，以测试可迁移性。B 保存的是候选几何（可能含玻璃背后背景），C 是推断的接触锚点；均非真实接触表面。自动 grasp 状态可能在释放后滞留，需看汇总 `grasp_hypothesis_no_active_taxels_frames` 和动态回放。SAM2 本轮仅向前跟踪，以降低离线未来帧泄漏；这不代表已经实现实时部署。不要把同一任务物体的模板初始化当作任意未知物体的通用识别。

## 已知局限

- 多位置时，模板使用仅平移 + 小幅缩放；如果物体/夹具转动较大、透明主体难以区分、光照变动大，可能需要给新位置做一次首帧点选。
- 目前必须检查每个 Episode 的首帧 Mask 和若干时间点跟踪结果（**无需每 Episode 手点**，但需要验收）。新位置稳定后可以减少人工参与。
- 这里不提供自动成功标签。`summary.csv` 中的 `*_roi_clip_fraction` 会指出固定空间 ROI 是否裁掉了新位置的候选点。ROI 不适配时应统一检验/调整，而非逐 Episode 调参。`summary.csv` 是工程诊断，而不是动作成功率或对象几何真值。
- 下游训练前需生成完整 Episode 的训练时序输入（`--stride 1`），并确认在线与离线的输入和延迟保持一致。此轮仅用于 4–6 条 Episode 的验证。
