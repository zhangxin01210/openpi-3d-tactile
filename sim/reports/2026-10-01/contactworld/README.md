# ContactWorld USB / Peg 数据审查

> **最新进展（2026-10-01）**：全部发布轨迹的覆盖审计与关节状态＋步数 MLP 的 20×2 次新初态闭环试验已完成；见[基线试验报告](proprio_baseline_pilot.md)。

> **后续进展（2026-10-01）**：隔离环境已完成 5 条 USB、3 条 Peg 的固定示范反馈重放，建立逐帧多模态对齐与 PyTorch 数据接口；详见[阶段 1–2 验收](stage1_stage2_validation.md)。下文是此前离线审查及选型的历史记录。

> **闭环示范重放已修复本条 USB 卡孔问题**：跟踪记录的末端控制目标，在 2 个初始化种子下完整插入，末帧插件误差 0.027/0.016 mm。未修改物理参数；原动作开环复现仍失败，不作为策略成功率。见[修复与复跑说明](replay_review.md)。

> **接触分叉诊断已完成**：已修正初始化重新夹紧，纯视觉改白不影响任何记录的动力学/触觉数组；全程仍在第 64 帧附近接触后分叉。第 75 帧已插入状态的局部恢复可保持，不能算完整成功。当前暂停扩大样本与训练，见[诊断记录](replay_review.md)。

> **最新重放审查**：已纠正 USB 资产来源不一致；示范初态恢复后插件偏差约 0.027 mm，82 步执行完成，但末帧偏差约 16.23 mm、插入判据未通过。当前不训练。见[重放与初始化审查](replay_review.md)。

> **本机 5090 自编译验收已通过（2026-10-01 16:00）**：独立环境中 Torch CUDA、gymtorch、GPU 物理/相机、单 USB reset 与 8 步传感器输出均通过；约 36 分钟达到节点并停止。原环境包快照未变。完整示范重放、在线对齐及训练仍未完成。见[执行记录](source_build_execution.md)。以下为较早记录。

> **最新更新**：用户已接受自编译成本，之前“小适配预算”的限制不再是暂停理由。已完成本机资源核查和[自编译工期估计](source_build_plan.md)，尚未启动长时间编译。以下保留此前选型经过。

> **后续路线决定**：限定范围的兼容排查已结束；发现的 5090 社区方案需要自编译 Torch/torchvision，尚未复现，暂缓本机旧栈。随后 Bench2Dex 三条深度/触觉回放已完成，但选中拼图任务的目标 XY 固定。见[后续选型报告](../bench2dex/README.md)。ContactWorld 数据保留，在线完整任务仍未通过。

数据审查结论：**比之前的 UniVTAC `insert_hole` 更接近需求，值得保留；尚不足以宣布三个模态都必要，或直接开始正式训练。** 已完成下载、离线审计、每任务 10 条完整视频和几何核对。

**后续进展（同日）**：独立旧环境的 GPU PhysX、原生 RGB/depth 与前相机矩阵检查已通过，进一步发现并验证了点云的半像素修正；但测试的 Torch 2.4.1 CUDA 算子无法在 RTX 5090 执行，完整 TacSL 任务与在线接触/动作对齐未完成。见[兼容检查报告](compatibility.md)和[运行时几何契约](runtime_geometry.json)。下文保留离线数据审查时的证据与待办，运行时状态以新报告为准。偏移恢复采集暂缓，未训练。

## 先看这些视频

本机[视频浏览页](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/index.html)包含完整轨迹、分镜、逐帧 CSV，以及跳转到重点帧的按钮。

| 优先查看 | 帧段 | 实际看到的内容 |
| --- | --- | --- |
| [USB 111](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/insertion_usb/episode_0111.mp4) | 41–54 | 高度约停在 18.9 mm，横向误差从 6.72 mm 降至 0.21 mm，随后继续下降；TacFF 同步变化 |
| [Peg 33](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/insertion_peg/episode_0033.mp4) | 68–105 | 多次抬起、调整、再下降；末帧通过当前源码的位姿判据 |
| [USB 0](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/insertion_usb/episode_0000.mp4) | 35–46 | 偏位后抬起约 8.8 mm，同时横移对中 |
| [Peg 0](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/insertion_peg/episode_0000.mp4) | 70–95 | 终帧仍比孔座根部高 16.61 mm，不能默认当成完成插入的正例 |

“高度”均指 `plug_pos.z - socket_pos_gt.z`，不等于距孔口高度。完整解释见 [review_notes.json](review_notes.json)。上述是运动学及外观审查；**首次 plug/socket 接触帧未知**。手指/插件 TacFF 非零不能用作首次碰孔标记。USB 0、111 虽然最终位置接近孔位，末帧姿态仍未通过当前源码阈值，因此未标为严格成功恢复。

## 下载和覆盖范围

| 数据 | 全量条数 / 帧数 | 视频抽样编号 | 导出帧数 |
| --- | --- | --- | ---: |
| USB | 201 / 13,671 | 0, 22, 44, 66, 88, 111, 133, 155, 177, 200 | 668 |
| Peg | 153 / 11,728 | 0, 16, 33, 50, 67, 84, 101, 118, 135, 152 | 894 |

按整个 episode 编号范围均匀抽样，没有按成功或“好看程度”筛选。20 条视频保留全部 1,562 帧，共约 16.4 MiB，AV1 / yuv420p / MP4；均通过帧数核对和 FFmpeg 完整解码。视觉检查使用所有抽样轨迹的分镜及重点片段，全帧数值用于辅助审查；没有声称逐帧人工确认了所有接触。

原始数据位于 `/home/sai/zx/openpi-sim-runtime/data/contactworld/`。两个压缩包约 12.03 GB，均与官方 LFS SHA-256 一致。另下载并核验了约 61.8 MB 的官方 `industreal.tar` 以检查几何。最终完整包由 `hf-mirror.com` 直连下载；最初弃用的代理测试片段约 190 MiB，随后大文件请求均显式 `--noproxy '*'`，没有代理回退。具体版本、字节数、哈希见 [sources.json](sources.json)。

## 对照原始需求

| 需求 | 本次证据 | 判断 |
| --- | --- | --- |
| 目标位置变化、视觉用于定位 | 两任务孔座 x≈0.45–0.55 m、y≈−0.05–0.05 m，覆盖全部九宫格 | 通过位置变化要求 |
| 起点不是已经对齐 | 全量初始 XY 误差中位数 USB 86.3 mm、Peg 87.1 mm；抽样起点也有明显偏位 | 比此前数据更合适；仍然预先抓好插件 |
| 触觉存在 | 每帧右手指 `10×14×3` TacFF，另有 TacRGB/TacDepth | 通过模态存在性检查 |
| 接触后调整 | 有停滞、抬起、横移及再下降片段，Peg 抽样更多 | 有候选；无成对接触标签，尚无严格“视觉相似、动作相反”配对 |
| 准确相机参数和点云 | 已有 `1024×6` 点云，已推导并检查坐标校正；数据本身不保存 K/T 或世界相机 depth | 部分通过；需要运行时确认，腕相机外参未验证 |
| 点云有独立价值 | 单视角三维结构确实存在，USB/Peg 几何不同 | 单任务内未发现几何变化维度；收益仍需消融证明 |
| 灵巧手 | Franka 平行夹爪，插入期间夹爪固定 | 不满足加分项 |
| 现成示范、方便配置 | 示范已落盘；离线读取无需仿真 | 通过离线部分；旧仿真在本机尚未验证 |

按固定 10 cm × 10 cm 区域均分九宫格，行对应 y、列对应 x：

```text
USB             Peg
31 25 17        26 15 11
20 19 22        13 16 18
26 24 17        22 16 16
```

这些统计证明输入场景变化，不证明视觉在信息论上不可替代，也不替代状态/时间基线的闭环测试。

## 相机、点云与触觉核对

实际字段包括 front/wrist RGB（256×256×3）、pointcloud（1024×6 XYZRGB）、右手指 TacRGB（320×240×3）、TacDepth（320×240）、TacFF（10×14×3）、6 维 action、9 维 dof_pos/dof_vel、EE/plug/socket 位姿及 step_idx。没有世界相机 depth、相机标定矩阵、时间戳、任务成功标记、plug/socket 成对接触。仅有 `meta/episode_ends`；属性没有补充采集配置。[逐字段清单](manifest.json)

固定源码前相机配置为 256×256、水平 FOV 75°、位置 `[0.68, 0, 0.15]`、xyzw 四元数 `[-0.258819045, 0, 0.965925826, 0]`。按该实现推得：

```text
K = [[166.8128477, 0, 128],
     [0, 166.8128477, 128],
     [0, 0, 1]]
```

源码 `tacsl_sensors.py` 的点云生成使用正深度，并对行向量乘逆视图矩阵的转置，未转换相机光学轴约定。**发布点云不能按注释直接与基座位姿混用。** 根据该实现推导了固定变换，未用数据拟合参数，也未改写原始数据。每任务 10 条、每条首/中/末帧共 60 帧检查结果：

| 投影方式 | USB RGB 平均绝对误差（0–1） | Peg RGB 平均绝对误差（0–1） |
| --- | ---: | ---: |
| 把原始 XYZ 直接当基座坐标 | 0.4068 | 0.3950 |
| 按源码约定做候选校正 | 0 | 0 |

校正后约 99.80% / 99.85% 点投在图像有效范围，表中颜色误差针对这些有效点。见[投影对照图](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_review/geometry/insertion_usb_0111_middle.jpg)和 [geometry.json](geometry.json)。RGB 重投影一致本身不足以验证外参平移和真实尺寸，因此另做资产核对：

- 使用 **HF 发布资产包**的 `manifeel_usb_socket.obj`。源码补丁目录中的 `USB_socket_1port.obj` 是另一个模型，不能混用。
- 10 条 USB 全帧中，按红色筛出的 29,206 个孔座点，全部位于发布模型包围盒的 1 mm 容差内；最大越界距离约 0.624 mm。
- 白色点高度中位数约 0.545 mm，与桌面附近一致。这是颜色筛选和包围盒检查，尚未做逐面距离或运行时标定认证。
- USB 抽样每帧孔座点数中位数 41、插件点数中位数 59.5；其余多为桌面/机械臂。1024 点全局采样对细孔几何较稀疏，不能保证孔内表面可见。

候选变换及所有统计已保留。**当前视频仍画原始存储点云，并明确标注坐标未验证；正式适配器尚未写入该校正。** 后续运行时检查确认了上述相机与坐标转换，并补充了像素中心修正；最新矩阵应使用 [runtime_geometry.json](runtime_geometry.json)，不要仅使用本次离线的整数像素候选变换。完整状态重放和腕相机验证仍未完成。

TacFF 通道由源码拼成法向、切向 x、切向 y；传感器局部投影方向分别为 −Y、−X、+Z。它来自 SDF 压入和相对切向速度的模拟力模型，不能把三通道直接理解为世界坐标 Fx/Fy/Fz。法向通道实测有正负值，视频采用保留符号的固定色标；剪切方向用箭头表示。力值未做牛顿量纲的独立校准，也不是 plug/socket 接触力。

## 动作、时间与终帧质量

源码控制器把前三维乘 `0.01` 后加到当前末端位置，把后三维乘 `0.05` 作为轴角增量，转换为四元数后左乘当前姿态；旋转不是直接相加的 Euler 角。夹爪保持关闭。数据平移指令可达 ±2，Peg 旋转可达 ±1.5，超过代码注释/动作空间的 ±1；后续适配器不能按注释静默裁剪。[当前上游 VecTask](https://raw.githubusercontent.com/purdue-mars/manifeel-isaacgymenvs/main/isaacgymenvs/tasks/base/vec_task.py)默认 `clipActions=Inf`，但实际采集时是否覆盖配置仍需重放确认。

[论文 Appendix B.2](https://arxiv.org/html/2606.13877v1#A2.SS2)说 10 Hz；固定源码默认 `dt=0.016667`、`controlFrequencyInv=4`，推算约 15 Hz。视频按论文的 10 fps 播放，以帧编号作为可靠索引，不把播放秒数当作已验证物理时间。USB `step_idx` 每条从零递增；**Peg 的 step_idx 全部为零**，需用数组行号和 episode_ends。精确 observation/action 延迟与时间关系未通过重放验证。

按当前源码的四关键点判据（scale=0.5 m、平均距离阈值 7.9916 mm）离线复算终帧：USB 145/201 通过、56 条不通过；Peg 126/153 通过、27 条不通过。结果见 [semantics.json](semantics.json)。这是**当前源码与发布轨迹末帧的对应情况**，不是论文成功率，也不能把所有未通过者都认定为物理插入失败。例如 USB 多数位置已经很接近，姿态仍触发该较长关键点标尺的误差；Peg 0 则有明显未插到底的问题。原数据没有成功标签或精确采集配置，训练前需要明确过滤规则。

`socket_pos(_gt)`、`socket_quat`、`plug_pos/quat` 只用于审查、环境复位和评估；普通策略输入若包含这些真值，会泄漏目标与接触状态。状态基线应使用关节/末端自身状态，并明确是否含时间索引。

## 本轮后的建议与停止点

先看上述 4 条及完整浏览页，判断定位与纠偏复杂度是否达到要求。当前更值得保留 Peg 的反复修正片段，USB 可用于更清楚地观察定位、遮挡和横向调整；不急于增加任务或扩大数据量。

如果外观复杂度认可，下一小步应是有边界的兼容性与数据契约核对：在独立旧环境中只验证 camera/TacFF/reset/step，用同一状态核对已推导点云变换、真实时间步、动作缩放和 plug/socket 成对接触；同时固定发布资产，避免补丁目录模型替换。相机或接触验证失败就停止扩展。完成后再选择少量位姿达标且有纠偏的示范，定义成功标准，做状态/时间基线与视觉/点云/触觉消融。

目前仍未得到“视觉近似相同、接触不同、最优修正方向相反”的严格对照集，单任务形状变化也不足。若这两项是研究核心，仍需在受控环境补采，不能把现有下载包直接包装成已经满足的实验协议。离线审查阶段未安装旧仿真；后续已做有边界的兼容检查，结果见上方新报告。尚未移植、训练或运行闭环策略评估。

## 复现入口

离线环境：`/home/sai/zx/openpi-sim-runtime/envs/contactworld-audit`，Python 3.11，依赖锁定在 [contactworld-audit-requirements.txt](../../../contactworld-audit-requirements.txt)。没有向共享 OpenPI 环境安装依赖。

```bash
CW_RUNTIME=/home/sai/zx/openpi-sim-runtime
CW_PY="$CW_RUNTIME/envs/contactworld-audit/bin/python"

# 固定版本分块下载，始终绕过代理；完成后核验 LFS SHA256
python3 sim/scripts/download_contactworld_audit.py "$CW_RUNTIME/downloads/contactworld"

# 已解压数据的完整视频和数值审查
"$CW_PY" sim/scripts/audit_contactworld.py \
  --data "$CW_RUNTIME/data/contactworld" \
  --output "$CW_RUNTIME/visualizations/contactworld_review" \
  --notes sim/reports/2026-10-01/contactworld/review_notes.json

"$CW_PY" sim/scripts/check_contactworld_geometry.py \
  --data "$CW_RUNTIME/data/contactworld" \
  --output "$CW_RUNTIME/visualizations/contactworld_review/geometry"

"$CW_PY" sim/scripts/check_contactworld_semantics.py \
  --data "$CW_RUNTIME/data/contactworld" \
  --source "$CW_RUNTIME/third_party/ContactWorld" \
  --assets "$CW_RUNTIME/downloads/contactworld/released_assets/industreal" \
  --geometry "$CW_RUNTIME/visualizations/contactworld_review/geometry/geometry.json" \
  --output "$CW_RUNTIME/visualizations/contactworld_review/semantics.json"
```

源码固定在 [8d0d0ed](https://github.com/PokuangZhou/ContactWorld/tree/8d0d0edff44f2bb28b58d96e3cf638088589e789)，数据固定在 [6587189](https://huggingface.co/datasets/Pokuang/ContactWorld/tree/6587189d8e06480e739af5a4004d411c5c07743b)。本报告的实测依据是上述本地 JSON、视频与固定源码；未将论文的中间目标规划分数等同于完整任务成功率。
