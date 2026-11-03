# 新任务 checkpoint 离线部署验证与输入可视化（2026-11-02）

> 后续集成修正：本报告保留当时使用 `baseline_unprofiled` 的测试结果与启动失败记录。
> 用户已确认三个任务共用现有 `front_calibration_candidate.json`，云端两个 Full 已用该文件启动；
> 不再需要寻找其他训练标定文件。当前默认标定、自动 metadata、Button/Socket 初始化与
> Full dry-run 限制见 [统一部署交接](unified_deployment_handoff.md)。原有数值未重测或改写。

8 个 checkpoint 均通过严格参数结构核对并完成真实离线推理，共 178 次，输出均为有限的 `[50,18]` 绝对关节目标。两个任务的实际 SAM2→RGBTrack→512 点物体几何→Full 推理链路均通过。原生 Full 服务启动仍受显式 front 标定 profile 检查阻塞，不能把离线回放通过当作完整服务已可部署。

## 版本与数据

开始检查时工作目录为 `796307f`，未包含 final-v2 编码器与任务配置；该版本下 ST/Full 参数不匹配。随后在隔离 worktree `/tmp/openpi-v2-deploy-20261102` 验证本地已存在的 `gitee/feat/unified-deployment` 提交 `d662da0`。测试期间主目录也已同步到该提交。未把旧版本的参数不匹配判为 checkpoint 损坏。

输入为 `data/1028_press_button` 和 `data/1030_yellow_socket` 的原始 LeRobot 数据。虽然没有空间 sidecar，Parquet 中实际有原始 `uint16` 深度和 1972 维 state；三路 RGB 是 AV1 视频，以 PyAV 读取。没有用合成深度、合成触觉、预计算点云或预计算物体 pose 替代原始数据。

每个 RGB/Pi0.5/ST 模型选择 episode 0、50、99，每个 episode 从开头、中间和最后一个完整 chunk 附近各取连续 3 帧，共 27 帧。Full 额外使用 episode 0 的连续 8 帧，通过实际 SAM2.1 Small、RGBTrack scorer/refiner、审核 mesh、静态/动态初始化和运动保护产生几何，再运行正式 `run_policy_dry_run.py`。两条 Full 回放使用各数据集 `tasks.jsonl` 中的原文 prompt。

六个模型的 27 帧采样与 Full 的 8 帧不是同一评估集合，表中总体误差不能直接用于模型排名。所有数据来自已有数据集，未证明为独立测试集。稀疏采样也不是完整 episode 的因果 EMA 一致性验证。

## 真实推理结果

延迟为策略调用耗时，包含在线空间输入处理（如适用），不包含外部 SAM2/RGBTrack worker。warm 是剔除首次调用后的中位数。首次调用包含 JAX 编译。

| 任务 | 模型 | 帧数 | warm 中位 ms | 首次调用 s | 50步 chunk MAE rad |
|---|---|---:|---:|---:|---:|
| 1028 | pi0_rgb | 27 | 56.6 | 10.57 | 0.05243 |
| 1028 | pi05_rgb | 27 | 63.7 | 10.23 | 0.03031 |
| 1028 | pi0_st | 27 | 91.6 | 23.50 | 0.05145 |
| 1028 | pi0_full | 8 | 93.7 | 24.58 | 0.05878 |
| 1030 | pi0_rgb | 27 | 56.2 | 11.12 | 0.01397 |
| 1030 | pi05_rgb | 27 | 65.0 | 10.72 | 0.01028 |
| 1030 | pi0_st | 27 | 90.0 | 21.45 | 0.01407 |
| 1030 | pi0_full | 8 | 94.2 | 26.56 | 0.01253 |

所有权重均从各自 `19999` 的真实 Orbax 导出加载。参数叶子分别为 Pi0 RGB 60、Pi0.5 RGB 61、ST 103、Full 117；与对应任务配置完全匹配，没有缺失、多余或形状不符。Pi0 使用 mean/std，Pi0.5 使用 quantiles；1030 Pi0.5 使用其 `final_v2_normfix` 内自身统计，没有替换成其他模型统计。

空间输入为场景 `[4096,3]`、触觉 `[600,3]`、Full 的手部 FK `[18,3]`、manipulated/target 各 `[512,3]`。所采样接触帧的触觉 force norm 最大值分别为 44.3847、17.7200（原始传感器尺度，未宣称 N）。Full 直接接收无几何 observation 会正确拒绝；这项负向检查的 FAIL 在各原始 `result.json` 中保留，最终正向结果以 `full_1028/summary.json`、`full_1030/summary.json` 和 `acceptance_summary.json` 为准。

## 可视化与第一帧动作

打开 [可视化总览](../../output/deploy/new_tasks_latest_20261102/visualizations/index.html)。每项任务包含原始三路 RGB、原始/注册深度、物体几何的前相机投影、可旋转场景/触觉/FK/物体点云、触觉接触补充帧、逐关节首步误差和完整 50 步预测曲线。Plotly JS 随文件保存，打开本地 HTML 不依赖 CDN。

第一帧对比固定为两项任务的 episode 0 / frame 0。读取同一次已保存的模型采样，不重新生成更有利的结果。已逐数组确认 Full 和其余模型使用的 RGB、state、native depth 在这一帧完全一致。所有动作都按相同 18 维顺序比较：6 机械臂 + 12 手指；GT 为原始 action，均为绝对关节目标。

| 任务 | 模型 | 首步18维 MAE rad | 首步 arm MAE rad | 首步 hand MAE rad | 首步最大单关节误差 ° | 同次50步 chunk MAE rad |
|---|---|---:|---:|---:|---:|---:|
| 1028 | Pi0 RGB | 0.01461 | 0.00229 | 0.02077 | 4.62 | 0.13471 |
| 1028 | Pi0.5 RGB | 0.01507 | 0.00497 | 0.02012 | 5.55 | 0.06015 |
| 1028 | Pi0 ST | 0.01624 | 0.00405 | 0.02233 | 3.42 | 0.14718 |
| 1028 | Pi0 Full | 0.01347 | 0.00366 | 0.01838 | 2.83 | 0.09304 |
| 1030 | Pi0 RGB | 0.01619 | 0.00466 | 0.02196 | 6.40 | 0.02192 |
| 1030 | Pi0.5 RGB | 0.01581 | 0.00639 | 0.02052 | 5.67 | 0.01338 |
| 1030 | Pi0 ST | 0.01664 | 0.00631 | 0.02180 | 6.99 | 0.02054 |
| 1030 | Pi0 Full | 0.01761 | 0.00642 | 0.02321 | 6.57 | 0.01878 |

首步平均误差约 0.013–0.018 rad（0.77–1.01°），整体较接近，但个别手指关节可差约 7°。1028 的完整 chunk 中，多个模型的若干手指在约 1.5–3 秒更早闭合，明显偏离该 episode 的示教时序；首步接近不能说明完整 chunk 相符。1030 的 50 步误差相对较小。

保持当前实测关节姿态的基线：1028 首步 MAE=0.01106 rad、50步 MAE=0.02569 rad；1030 首步=0.02401 rad、50步=0.04632 rad。1028 首段动作相对静止，hold 基线优于本次模型采样；1030 本次四个模型的整体误差低于 hold。以上只描述这些观察和采样，不能据此推断模型排名或真实成功率。

## 服务启动与标定限制

实际调用原生 `serve_policy.create_policy` 加载两个 Full checkpoint，几何 mesh、release、角色、采样种子和 manifest 校验通过后，均在默认任务配置处停止：`Training config requires an explicit front calibration profile for online geometry`。测试显式设置了现有 perception Python，未伪造训练 profile 来绕过检查。

离线实际感知/策略回放使用任务配置默认 `baseline_unprofiled`（无 profile override）；训练几何 manifest 未记录标定 provenance，无法证明训练和在线几何逐点一致。仍需提供真实训练使用的 profile/确认标定来源，才能完成原生 Full 服务启动验收。本次没有实机运行或发送机器人指令。

## 检查、改动与复现

当前目录修复 `XHandInputs` 直接读取扁平部署 observation 的兼容问题，同时保留 `images/state` 输入。相关已有测试共 22 项通过。三个新增检查/可视化脚本 Ruff 通过。已有 xhand 文件中的中文标点 lint 未整体重写。

```bash
# 六项 raw 路径正向检查，以及两项 Full 缺少几何时的负向检查。
# 后两项拒绝输入，因此本命令整体退出 1 是预期审计结果。
PYTHONPATH=src:. XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  scripts/deploy/test_new_task_checkpoints.py --output-dir output/deploy/new_audit

# 单任务实际感知（输出目录必须未使用；另一个任务同理）
PYTHONPATH=src:. OPENPI_REPO_ROOT="$PWD" .venv-perception/bin/python \
  scripts/deploy/probe_final_v2_perception.py --repo-root "$PWD" \
  --task 1028_press_button --frames 8 --output-dir output/deploy/perception_new_1028

# 根据已经保存的结果重建全部可视化，不重新采样策略
PYTHONPATH=src:. JAX_PLATFORMS=cpu .venv/bin/python \
  scripts/deploy/visualize_new_task_offline.py
```

Full 正向策略回放沿用 `scripts/deploy/run_policy_dry_run.py`：输入上述感知目录下的 `perception/frames`，选择同任务 `*_full_v2` 配置及 checkpoint、传入 `tasks.jsonl` 原文 `--prompt`，并设置 `--gt-dataset`、`--gt-episode 0` 和新的 `--output-dir`。

最终机器可读结果为 [acceptance_summary.json](../../output/deploy/new_tasks_latest_20261102/acceptance_summary.json)；第一帧精确数值位于可视化目录每任务的 `comparison.json`。`full_1028_discarded_wrong_prompt` 是一次已中止的错误 prompt 探索，不参与任何最终指标。最初 `new_tasks_20261102` 目录保留旧提交审计证据；最终结果使用 `new_tasks_latest_20261102`。
