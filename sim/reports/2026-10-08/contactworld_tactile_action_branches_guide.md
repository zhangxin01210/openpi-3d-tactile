# 双指触觉夹持基线与同初态动作分支

一键后台启动（在仓库根目录）：

```bash
bash sim/scripts/launch_contactworld_tactile_action_branches.sh
```

默认输出：`/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_action_branches_20261008/`。命令立即打印 PID、日志和报告路径。查看进度：

```bash
tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_action_branches_20261008/launch.log
```

出现 `analysis.json` 和 `index.html` 才表示三阶段完成。重新执行同一命令时，已成功保存的分支会复用；改变参数需指定新输出目录，例如 `CW_TACTILE_BRANCH_OUTPUT=/path/to/new_run bash ...`。不要与其他 Isaac Gym 任务同时争用同一张本机 GPU。

## 执行内容

1. **离线夹持基线**：检查 120 条双指示范。每条只在首次 USB—插座接触前寻找连续 4 帧、非零且变化稳定的夹持力图；找不到则明确缺失。局部传感器坐标中保留原始力、基线、差值与最近 3 步变化。按完整示范划分 train/val/test，用轻量线性探针比较接触检测和“下推后受阻/推进”。这只测表示的可读性，不等于 π0 的模态消融。
2. **受控分支**：从 6 类状态各取 3 个不同来源/夹持配置，共 18 个分支点。每次从原示范第一帧重放相同动作前缀，检查重放与已审核轨迹的位置误差，然后执行 7 种候选动作：保持、直接下插、回撤再下插，以及插座局部 ±X/±Y 各移动 2 mm 后下插 6 mm。全部分支共 11 控制步，每个重复 3 次，最多 378 个短试验。若前缀 USB 位置误差超过 0.5 mm、末端超过 1 mm、双指初始力图与原重放明显不符、重复起点散布超过 0.5 mm，或保持动作结果的重复波动超过 0.5 mm，该分支不计入分析。
3. **动作效果与模态信息**：用真实 USB 轴向进展、夹持相对移动、插座接触力评分。按**完整初态组**留出，比较双视图降采样+关节状态+点云固定体素占据/几何统计+已执行动作历史+候选动作，与加入当前双指力、再加入夹持基线差和短时力变化后的预测误差。这里的 18 组仍是探索性样本；如差异很小或不稳定，不能推出触觉无用。

## 报告怎么读

打开 `index.html`。顶部是全量示范上的触觉表示探针；接着按状态展示原始双视图、机械臂 base 下的点云、双指 Fx/Fy/Fz、相对夹持基线的局部力图、7 个动作分支的双视图＋双方六张 Fx/Fy/Fz 动态力图视频和时间曲线。`analysis.json` 保留汇总；`branches/source_.../*.npz` 保留逐步图像、双指力、动作、USB/末端/插座位姿、接触力和效果，便于晚上进一步检验特定假设。视频是本次短分支，不是完整任务成功视频。

仿真精确插座位姿、USB 运动和碰撞力只用于选状态、重放检查及评分；它们不进入诊断预测器。当前试验只能评价这 7 个有限短动作的效果，不能把某个动作宣布为全局最优恢复策略。

最小烟测：

```bash
CW_TACTILE_BRANCH_OUTPUT=/tmp/cw_tactile_branch_smoke bash sim/scripts/launch_contactworld_tactile_action_branches.sh --smoke --repeats 1
```

点云诊断输入使用审核过的同帧 fused 4096 点（机械臂 base 坐标），不是另一次在线重建；分支前缀的位姿门槛限制两者差异。视觉探针采用两视图各约 16×16 的降采样，因此其分数不能解释为原分辨率视觉的性能上限。
