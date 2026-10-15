# ContactWorld 阶段 1–2：固定环境、跨示范重放与训练数据接口

日期：2026-10-01。范围：USB 与 16 mm Peg 两种插入任务；本机 RTX 5090，隔离的 Isaac Gym/TacSL 环境。**本阶段没有训练策略，也没有采集偏移恢复数据。**

## 结论

- 固定的 5 条 USB、3 条 Peg 发布示范都从初态运行到终帧，并通过环境源码 `_check_success()` 的插件与孔座关键点接近判据。重放使用**记录的末端控制目标做位姿反馈**；每步实际执行动作可能不同于发布动作，因此这不是策略成功率，也不是原动作开环成功率。
- 发布数据的前视/腕视 RGB、右侧触觉 RGB/深度/三分量 TacFF、动作与本机同任务观测可以按“第 i 行观测在动作 i 前”比较。恢复发布版 Peg 可视材质，以及只对胶垫和安装座做可视颜色匹配后，首帧 RGB 误差明显下降。在线前视点云与同一步 RGB/depth 投影一致。
- 新的 [PyTorch 数据读取类](../../../../src/openpi/spatial_dataset/contactworld.py) 可从原始 Zarr 按轨迹读取，提供 16 步动作及轨迹尾部有效位；训练/验证/测试以完整轨迹划分，已通过 DataLoader 批次检查。**这只是可供训练使用的数据接口，尚未接入 OpenPI 的训练配置或模型。**

## 环境和资产固定

环境：`/home/sai/zx/openpi-sim-runtime/envs/contactworld-build-sm120`；运行源码：`/home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-sm120`。指纹文件 `runs/contactworld_multireplay_20261001/runtime_fingerprint.json` 记录 GPU/驱动、Python/pip、源码提交及 27 个关键配置、资产和脚本的 SHA-256。两份 Peg 16 mm URDF 已在**隔离运行目录**换回发布版，其 SHA-256 与下载发布资产相同；两份碰撞 OBJ 原已一致。USB 使用发布版插件和孔座网格。胶垫、安装座通过重放参数改可视颜色，不改摩擦、碰撞或控制。原 `contactworld-legacy` 与 `univtac-isaac51` 的 pip freeze 与构建前快照逐字一致。

## 预先固定的 8 条示范

编号在在线运行前固定，且这些示范在离线检查中已通过终点位姿筛选，因此 8/8 **不能外推为数据集成功率**。输入 NPZ、逐文件 SHA-256 和原 Zarr 行范围见 `runs/contactworld_multireplay_20261001/input/manifest.json`；每个 `replay.json` 还记录实际读取的 NPZ SHA-256，以防误用旧结果。最终结果见 `batch_final/batch_result.json`；[8 条同步视频](/home/sai/zx/openpi-sim-runtime/runs/contactworld_multireplay_20261001/batch_final/index.html)左为发布数据、右为在线重放，上为前视、下为腕视。

| 任务 | 编号 | 帧数 | 终点判据 | 插件终点位置误差 (mm) |
| --- | ---: | ---: | --- | ---: |
| USB | 1 | 82 | 通过 | 0.027 |
| USB | 4 | 112 | 通过 | 0.105 |
| USB | 67 | 82 | 通过 | 0.184 |
| USB | 101 | 53 | 通过 | 0.169 |
| USB | 132 | 53 | 通过 | 0.239 |
| Peg | 1 | 131 | 通过 | 0.707 |
| Peg | 33 | 138 | 通过 | 0.192 |
| Peg | 101 | 60 | 通过 | 0.192 |

全部 711 帧视频经 `ffprobe` 核对为 H.264/yuv420p，帧数与示范一致。视频播放帧率 10 fps 仅供观看，**不代表已核实的历史采集频率**。

## 在线数据对齐：验收值和边界

以下 MAE 按逐帧记录的动作前观测计算；RGB 按 0–1，触觉深度为发布数据的原始 0–1 数值，TacFF 保持原始有符号分量。数据表示 USB 382 帧、Peg 329 帧各自的中位数 / 第 95 百分位。逐帧值与本机实际执行动作见各 `replay.json`。

| 字段 | USB 中位 / P95 | Peg 中位 / P95 |
| --- | ---: | ---: |
| 前视 RGB MAE | 0.000059 / 0.00835 | 0.000935 / 0.0142 |
| 腕视 RGB MAE | 0.000316 / 0.00641 | 0.00221 / 0.00578 |
| 右侧触觉 RGB MAE | 0.000851 / 0.0115 | 0.00682 / 0.0121 |
| 右侧触觉深度 MAE | 0.00168 / 0.0333 | 0.0377 / 0.0800 |
| 三分量 TacFF MAE | 0.0000492 / 0.000255 | 0.000245 / 0.000742 |

Peg 的触觉深度和接触阶段位姿仍有偏差，不能称为逐帧精确复现；这与恢复初态会推进一次物理仿真、历史物体速度和接触求解器状态不可恢复相符，但目前不能把偏差唯一归因于其中任一项。终点判据本身是关键点距离阈值，不是对真实插入深度的独立量测。

在**在线重放同一步**，前视 1024 点投影回原生 RGB/depth，8 条的 RGB 平均误差为浮点舍入量级、深度最大差约 `9.3e-8 m`；此检验说明在线取样与投影自洽。历史发布包没有逐帧深度、相机矩阵或校准快照；历史点云至本机基座坐标的校正来自[几何实测](runtime_geometry.json)，可用于当前数据接口，但它仍是根据当前固定前视相机推断的历史坐标约定。腕部 RGB 的历史外参没有同等校准证据，不能宣称历史所有点云具有独立测量的相机真值。

## 训练数据接口

[数据接口审计结果](/home/sai/zx/openpi-sim-runtime/runs/contactworld_multireplay_20261001/training_contract.json)状态为 `passed`：

| 划分 | 轨迹数 | 行数 | USB / Peg 轨迹数 |
| --- | ---: | ---: | ---: |
| 训练 | 282 | 20,268 | 162 / 120 |
| 验证 | 43 | 3,048 | 23 / 20 |
| 测试 | 29 | 2,083 | 16 / 13 |

返回前视/腕视/触觉 RGB、18 维关节位置与速度、校正到本机基座坐标的 `1024×6` XYZRGB、局部 `10×14×3` 有符号 TacFF、`16×6` 发布原动作以及 `action_valid`。轨迹尾部补零并标无效，不跨轨迹读未来帧。`task_id`、`episode_index`、`frame_index` 只用于追溯，训练时不得当作策略观测。动作未经假定的 [-1,1] 归一化：USB/Peg 原始分量绝对值大于 1 的比例分别约 3.19% / 13.85%。TacFF 的三分量正负都实际存在；物理牛顿标定未验证。当前检查覆盖所有动作和每 100 行的力场样本，以及每个划分的样本与 2 条批次；没有逐行重新扫描全部图像。

## 复跑

从项目根目录执行。保留隔离目录中的已核对资产；输出使用新目录，避免误复用旧结果。

```bash
RUNTIME=/home/sai/zx/openpi-sim-runtime
RUN="$RUNTIME/runs/contactworld_stage12_recheck"

"$RUNTIME/envs/contactworld-build-sm120/bin/python" \
  sim/scripts/export_contactworld_replay_subset.py \
  --data "$RUNTIME/data/contactworld" --output "$RUN/input"

python3 sim/scripts/run_contactworld_replay_batch.py \
  --source "$RUNTIME/third_party/ContactWorld-sm120" \
  --input "$RUN/input" --output "$RUN/batch_final" \
  --seed 0 --white-gel --white-mounts

PYTHONPATH=src "$RUNTIME/envs/contactworld-build-sm120/bin/python" \
  sim/scripts/audit_contactworld_training_contract.py \
  --data "$RUNTIME/data/contactworld" \
  --geometry sim/reports/2026-10-01/contactworld/runtime_geometry.json \
  --output "$RUN/training_contract.json"

python3 sim/scripts/render_contactworld_replay_batch.py "$RUN/batch_final"
python3 sim/scripts/freeze_contactworld_runtime.py \
  --runtime "$RUNTIME" --output "$RUN/runtime_fingerprint.json"
```

## 下一阶段的明确缺口

1. 将此数据契约接到 OpenPI 的实际多模态模型与训练配置，确定 RGB/点云/TacFF 编码器、尺度归一化和动作输出；目前尚无训练结果。
2. 建立不读示范末端目标的闭环策略评测，并按随机初态及任务分别报告成功率。需要独立检查视觉、触觉、点云的作用，而非用上述反馈重放数值代替。
3. 若论文结论需要**历史发布数据逐帧准确相机外参或度量级 TacFF**，需取得采集配置/标定证据，或只把有逐帧标定的本机新采数据用于该结论。
4. 用户暂缓的小规模偏移恢复采集继续暂缓；先观察现有示范训练与消融是否已经包含足够的纠偏能力。
