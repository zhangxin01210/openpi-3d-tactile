# ContactWorld 数据覆盖审计与首个闭环基线

日期：2026-10-01。承接[阶段 1–2 验收](stage1_stage2_validation.md)，本轮只做全部发布轨迹的离线审计，以及关节状态＋步数的小模型闭环试验；未开始 OpenPI 多模态训练，也未采集偏移数据。

本轮[运行环境指纹](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/runtime_fingerprint.json)记录 GPU/驱动、隔离环境依赖、资产与 34 个关键源码/脚本的 SHA-256。原 `contactworld-legacy`、`univtac-isaac51` 的 pip freeze 仍与构建前快照逐字一致。

## 结论

1. 354 条发布轨迹中，按当前源码的终点关键点距离判据，USB 有 **145/201** 条、Peg 有 **126/153** 条通过；这不是发布的逐条物理成功标签。数据并非全部可不加筛选地作为成功示范。
2. 分别用终点通过的训练轨迹拟合 q/dq/步数到 6 维动作的 MLP。它们的验证集动作 MSE 优于常数动作，却在 USB 和 Peg 各 **20 个新随机初态中均 0 次**达到源码判据。20 次零成功的单任务 Wilson 95% 区间约为 0–16.1%；这只是一个基线的小样本结果，不能证明任一模态必然有效。
3. 比源码更紧的在线终点代理条件由位置、轴向和 10 步停稳组成。各选一条已成功示范作正对照，USB 与 Peg 都通过全部 10 步；该条件仍未独立验证真实碰撞插入深度。

## 354 条数据的覆盖和质量

[逐轨迹 CSV](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/audit/episodes.csv)与[审计摘要](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/audit/summary.json)直接读取发布 Zarr 中的位姿，按当前任务配置的 4 个轴向关键点、`keypoint_scale=0.5 m`、均距阈值 `7.9916 mm` 重算。终点通过数与既有离线语义审查的 145/126 一致。

| 任务 | 轨迹/帧 | 终点通过源码位姿判据 | 轨迹中任意帧曾通过 | 附加的 3 帧紧位姿代理 | 初始目标 XY 范围 | 初始插件至目标水平距离中位数 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| USB | 201 / 13,671 | 145 | 158 | 142 | X 0.450–0.550 m；Y −0.050–0.050 m | 86.3 mm |
| Peg | 153 / 11,728 | 126 | 137 | 105 | X 0.450–0.550 m；Y −0.050–0.050 m | 87.1 mm |

离线紧位姿代理要求终点通过源码判据，且末 3 帧的插件/孔座根部相距 <2 mm、无符号轴向角 <5°。它仍不是物理成功标签。基于“靠近孔口后抬升 >4 mm，随后下降”的**运动学检索**给出 Peg 31 条候选、USB 0 条；这个检索会漏掉纯横向调整，例如先前人工查看的 USB 111，因此不能据此断言 USB 没有纠偏。Peg 118 的[故事板](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/insertion_peg/episode_0118_story.jpg)可作为一个待人工核对的候选。

## 关节状态＋步数基线

模型是每个任务单独训练的 3 层、每层 256 单元的 MLP。输入只有 9 维关节位置、9 维速度及从 reset 开始的步数/100；没有目标、示范末端轨迹、图像、触觉或点云。输出为 6 维原动作，在线时仅按训练集动作范围限幅。输入/输出均值和方差只由终点通过的**训练划分**计算。当前按完整轨迹哈希划分的数据中，训练示范 USB 117 条、Peg 100 条；验证 18/16 条，测试 10/10 条。

| 任务 | 训练/验证帧数 | 最佳 epoch | 验证动作 MSE | 常数训练均值动作 MSE | 测试动作 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| USB | 7,881 / 1,342 | 55 | 0.0365 | 0.1018 | 0.0430 |
| Peg | 7,745 / 1,109 | 72 | 0.2374 | 0.3781 | 0.2415 |

MSE 为原始动作分量平方误差的平均值，不能换算成插入率。[训练报告与损失曲线数据](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/train/train_report.json)及两个 checkpoint 保存在同目录。

## 全新随机初态闭环

两个任务都使用种子 10000–10019，调用当前任务的 `reset_idx()` 获得新夹持姿态和随机目标，**没有恢复发布示范初态**。USB 每次最多 160 控制步，Peg 最多 240 步。源码判据按轨迹中任意一步、末步分别记录；更紧的在线代理要求源码通过、插件/孔座根部距离 <2 mm、根部相对高度 <5 mm、无符号轴向角 <5° 连续 5 步，随后零动作 10 步仍全部满足。

| 任务 | 源码判据曾通过 | 末步源码通过 | 10 步紧位姿停稳 | 全程最小关键点误差的中位数 / 最好一次 | 源码阈值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| USB | 0/20 | 0/20 | 0/20 | 50.2 / 24.7 mm | 8.0 mm |
| Peg | 0/20 | 0/20 | 0/20 | 49.9 / 29.7 mm | 8.0 mm |

[两条代表性录像](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/index.html)显示第一种子，左前视、右腕视，H.264/yuv420p 且全片解码通过。USB、Peg 的[逐种子结果](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/baseline_summary.json)和每步动作/误差均已保存。USB 评测进程在 16 个种子后收到终止信号，利用逐种子记录继续跑完余下 4 个；最终两任务的种子序列各为连续 20 条，无缺失或重复。

正对照使用**记录末端目标反馈重放**的 USB 1、Peg 1。到达终点后额外施加 10 步零动作，两者每步均符合紧位姿代理；末步根部误差分别约 0.305 / 0.529 mm。见 `runs/contactworld_baseline_20261001/positive_hold_usb1/replay.json` 与 `positive_hold_peg1/replay.json`。正对照只验证该代理对这两条已成功重放可达，不是基线策略的结果。

## 解释边界与下一步

目标 XY 在本任务中独立随机化，而基线输入不含目标位置；其失败符合“需要目标信息”的预期。不过本轮只评测一种模型、一个训练预算及 20 个随机初态；接触动力学偏差、示范质量和动作多峰性也可能影响结果，**不能单独证明视觉、触觉或点云各自有用**。发布数据没有可信的逐帧物理成功标签，紧位姿与停稳仍是代理判据。已按约定停在首个闭环基线节点；建议下一步打通 RGB 策略的同种子评测，再依次加入 TacFF 与点云并比较接触阶段恢复。

复跑入口：

```bash
RUNTIME=/home/sai/zx/openpi-sim-runtime
RUN="$RUNTIME/runs/contactworld_baseline_recheck"
SOURCE="$RUNTIME/third_party/ContactWorld-sm120"

PYTHONPATH=src "$RUNTIME/envs/contactworld-build-sm120/bin/python" \
  sim/scripts/audit_contactworld_episodes.py \
  --data "$RUNTIME/data/contactworld" --source "$SOURCE" --output "$RUN/audit"

bash sim/scripts/run_contactworld_sm120.sh sim/scripts/train_contactworld_proprio_baseline.py \
  --data "$RUNTIME/data/contactworld" --audit "$RUN/audit/episodes.csv" --output "$RUN/train"

# 分批执行；同一输出目录会校验配置并从已保存的种子继续。
bash sim/scripts/run_contactworld_sm120.sh sim/scripts/eval_contactworld_proprio_baseline.py \
  --source "$SOURCE" --task usb --policy "$RUN/train/insertion_usb_proprio.pt" \
  --output "$RUN/eval_usb" --trials 20 --max-steps 160 --seed-base 10000 --limit-this-run 5
bash sim/scripts/run_contactworld_sm120.sh sim/scripts/eval_contactworld_proprio_baseline.py \
  --source "$SOURCE" --task peg --policy "$RUN/train/insertion_peg_proprio.pt" \
  --output "$RUN/eval_peg" --trials 20 --max-steps 240 --seed-base 10000 --limit-this-run 5
# 各评测命令重复调用，直至 summary.json 包含 20 条。

python3 sim/scripts/summarize_contactworld_baseline.py \
  --audit "$RUN/audit/summary.json" --train "$RUN/train/train_report.json" \
  --usb "$RUN/eval_usb/summary.json" --peg "$RUN/eval_peg/summary.json" \
  --output "$RUN/baseline_summary.json"
python3 sim/scripts/render_contactworld_baseline_index.py "$RUN"
```
