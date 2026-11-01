# `1028_press_button` — RGBTrack v1 批量与统一发布

状态：**用户已验收五条代表性 Episode 的可视化效果；透明漏斗任务不在此版本范围内。**

## 输入与冻结策略

- `data/1028_press_button`：原始 LeRobot 数据；SAM2 结果在 `output/preprocess/1028_press_button/sam2/`；米制并已批准的 GLB 在 `geometry/approved/button.glb`、`box.glb`。
- RGB-D：已修正的前置相机 SDK 深度→彩色旋转、出厂 RGB K，和公共 `src/openpi/spatial/rgbd_registration.py`。禁止恢复旧的 RGB 内参修正和单点注册漏洞。
- 动态 `button`：首次注册 +Z 朝上（25°），之后原生 RGBTrack；位移阈值 2.0 cm/帧、旋转阈值 25°/帧、遮挡恢复相对上次可信姿态的总旋转上限 60°。拒绝错误观测，不强行截断位姿。
- 静态 `box`：upright 25°、Mask 凸包投影匹配的首帧初始化；后续静态冻结。
- 这些阈值属于**press_button 的 v1 工作假设**，不自动泛化到其他物体或任务。

## 批量运行（A800，Python `.venv`）

```bash
export REPO=/workspace/mnt/sqzhang26/openpi-3d-tactile
export HOI=/workspace/mnt/sqzhang26/hoi
cd "$REPO"
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
  --dataset 1028_press_button check --episodes all

PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_task.py \
  --dataset 1028_press_button batch --episodes all \
  --dynamic-object button --static-object box \
  --dynamic-init-up-axis +z --dynamic-init-max-tilt-deg 25 \
  --guard-max-step-cm 2.0 --guard-max-turn-deg 25 \
  --guard-recovery-max-turn-deg 60 \
  --run-tag press_button_v1
```

**不要加 `--review` 或 `--max-frames`**。默认不生成交互 HTML，仍逐帧生成完整位姿、质量报告、来源记录。跑过且来源哈希一致的完整 Episode 会跳过；不完整目录不会被自动覆盖，先检查原因再安全移走故障目录。

## 验收与统一路径

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_v1_manage.py \
  --dataset 1028_press_button audit

# 先查看发布预案（不改变任何文件）
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_v1_manage.py \
  --dataset 1028_press_button promote

# 所有 Episode 完整、代表性 HTML 已人工检查后，发布正式入口
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_v1_manage.py \
  --dataset 1028_press_button promote --apply
```

源位姿仍保存在 `rgbtrack/runs/press_button_v1/episodes/eXXXXX/`。正式统一读取路径为 `rgbtrack/episodes/eXXXXX/`（**相对软链接**，不会重复拷贝全部结果），与 `sam2/episodes/eXXXXX/`、`geometry/approved/` 位于同一任务根目录。

`rgbtrack/release_v1.json` 保存本版本配置及全局统计。供下游加载的内容：`button_poses.jsonl`、`box_poses.jsonl`、`quality_summary.json`、`task_input_provenance.json`、`manifest.json`。不要将其硬编码成某个实验 run-tag。

## 处理无效帧：关系 Token 的必要合同

**`valid=false` 不代表物体或关系不存在。** 应保留以下状态：

1. `pose_valid`：该帧是否有经过 QC 的可信视觉位姿。
2. `pose_source`：`observed`、`static_frozen`、`motion_rejected_*`、`occluded_causal_prediction` 等。后两者**不能当成可信物体几何**。
3. `last_valid_age`：距离最近一次真实/可信位姿的帧数，以及连续可见/不可见状态。
4. `relation_valid` 与 `relation_confidence`：手物/物物几何关系是否有可信计算依据；不能把 `valid=false` 的 6D 位姿直接传给 k-NN / 最近点距离算子。
5. 若手部触觉确证稳定抓持，可在**单独的模型和显式低置信标记**下短时沿用“手→物体”的最后可信刚性相对变换（结合 FK）。这只是一种**因果推算**，不是视觉测得真值；握持结束、滑移或长期遮挡时必须失效。仅触觉接触不等于抓握刚性成立。
6. 静态 `box` 的位姿在初始化验证后可继续用于手到 Box、Button 到 Box 距离计算；但 Button 无可信姿态时，Button→Box 距离应 `relation_valid=false`，除非经过独立可信的接触锚定。不要通过未来帧插值伪造在线输入。

遮挡时如果还知道手的 3D 点与五指触觉，可以输出手部/触觉 Token，而不是强迫物体几何 Token 填充错误姿态。无法计算的距离设置掩码，并保证所有数值 finite；不能用无掩码的零距离冒充真实接触。

## 清理已废弃测试输出（仅成功发布 v1 后）

```bash
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_v1_manage.py \
  --dataset 1028_press_button cleanup

# 核对列出的目录确实是历史实验后才执行
PYTHONPATH=src .venv/bin/python scripts/preprocess/rgbtrack_v1_manage.py \
  --dataset 1028_press_button cleanup --apply
```

只清理 **`1028_press_button/rgbtrack/` 目录内部**旧的 `experiments/`、`reviews/`、非 `press_button_v1` 的 `runs/` 子目录。**不删除**原始数据、SAM2 Masks、批准的 GLBs、正式 v1 轨迹、权重，也不自动删除 `tmp/` 下代码。

## 提交 Gitee（仅源代码，不提交权重/数据）

完成 audit/promote 后运行 `scripts/preprocess/publish_press_button_v1_gitee.sh`，脚本会检验远程分支未发生分叉，按精确白名单提交脚本/标定/测试/文档，然后非强制推送到 Gitee main。若远程分支有更新或版本检查失败，停止并人工处理，不执行 `push --force`。
