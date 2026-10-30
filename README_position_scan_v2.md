# 六个烧瓶位置：修正分组与首帧提示点（2026-10-22）

此次两个独立的旧版问题：

1. 原扫描只把 score 与 margin 都过阈值的候选纳入分组；因此59条被错误地从位置聚类中删除。
2. 对透明对象/大背景模板的全局匹配可能锁定机械臂上方的静止结构，尤其 Episode 31–45、62–69。高得分不代表目标正确。

新版用**右下方随目标移动的白色夹具/底座作为图像位置代理**。它不是玻璃表面/真实烧瓶 Mask。
在用户提供的70帧 JPEG 拼图（压缩、有旧扫描十字标记）上已得到六组：0–15、16–30、31–45、46–53、54–61、62–69。真实原始 RGB 上还需要验证。

## A. 上传两个新脚本

保存为：
- `/workspace/mnt/sqzhang26/hoi/scripts/level1_position_scan_v2.py`
- `/workspace/mnt/sqzhang26/hoi/scripts/level1_fixture_seed_v2.py`

保留之前的 `level1_position_scan.py`、`level1_template_init.py`、`level1_ep.sh`，不需修改。

```bash
HOI=/workspace/mnt/sqzhang26/hoi
REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
cd "$REPO"
PYTHONPATH=src "$REPO/.venv/bin/python" "$HOI/scripts/level1_position_scan_v2.py" --episodes 0-69
```

输出：`$HOI/outputs/level1_position_scan_v2/{position_groups.json,episode_positions.csv,representatives.jpg,all_episodes_montage.jpg}`。
**先核实六组是否符合实际**；不要通过任意扩大分组半径强行凑六组。
若自动探测少量帧失败，而首帧拼图确认六段连续，可用：

```bash
PYTHONPATH=src "$REPO/.venv/bin/python" "$HOI/scripts/level1_position_scan_v2.py" \
  --episodes 0-69 --ranges 0-15,16-30,31-45,46-53,54-61,62-69
```

`--ranges` 是按图像检查得到的人工分组先验，不属于自动识别成功的证据。

## B. 每组先抽一条做完整 Mask 检查

推荐六组代表 Episode：`7,23,38,49,58,65`；不需要按顺序做1、2、3……

```bash
HOI=/workspace/mnt/sqzhang26/hoi
REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
for ep in 7 23 38 49 58 65; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" prepare || break
  # 保留已经存在的人工修正种子；不要覆盖。
  if [ ! -f "$HOI/outputs/level1_sequence_e${ep}/seed_prompts.json" ]; then
    bash "$HOI/scripts/level1_ep.sh" "$ep" seed || break
  fi
done
cd "$REPO"
PYTHONPATH=src "$REPO/.venv/bin/python" "$HOI/scripts/level1_fixture_seed_v2.py" \
  --episodes 7,23,38,49,58,65
```

此脚本用 Episode0 的**白色底座中心变化量**平移 Episode0 人工确认过的烧瓶提示点；保留各 Episode 的漏斗提示点，并备份原 `seed_prompts.json`。已有人手工验收的提示点则不会被覆盖。

**检查：** `$HOI/outputs/level1_sequence_e<ID>/fixture_seed_preview.jpg`。
玫红线是翻译后的旧 Mask 外形、绿色是正点、红色是负点、黄色十字是夹具位置代理。该图只是提示点建议，并非 SAM2 输出。

```bash
for ep in 7 23 38 49 58 65; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" annotate || break
done
```

如果点还偏：打开 `annotate_frame0.html` 人工调整正/负点并导出 `seed_prompts.json` 上传覆盖（每个新位置的一条即可）。不要把白色夹具误标为玻璃，也不要误把最初放漏斗的锥形瓶当成目标烧瓶。**漏斗点也需要检查**，因为这个脚本不修复漏斗模板。

## C. 验证 SAM2，再跑全程

```bash
for ep in 7 23 38 49 58 65; do
  bash "$HOI/scripts/level1_ep.sh" "$ep" preview || break
done
```

确认 `mask_preview_frame0.jpg` 中漏斗/目标烧瓶均正确，按之前 README 的 approve → track → review-track → build → memory 继续。
如果首帧看不到漏斗而无法正确正点初始化，先标记为失败样本，不要强行批准。

## 诚实的限制

- 检测到的是白色固定装置的亮块，**不是烧瓶中心**。同位置下图像位移可作为提示点初始化的近似，不能作为三维几何输入。
- 平移从 e0 mask / 点击位置继承。若 e0 的 `target_flask` 本身包含错误对象，本方法会继承错误。必须看 SAM2 预览。
- 本次用拼图像素做过六位置验证，但没有远程运行用户服务器上的70条原始 RGB。
- 六组代表轨迹通过后，可以把这些组与 Episode ID 的对应关系写到处理配置中，整个数据集自动选择相应已验证的首帧模板；不要再做无约束全图匹配。
