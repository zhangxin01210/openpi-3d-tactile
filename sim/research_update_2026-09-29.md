# 仿真环境与数据路线复检

检索日期：2026-09-29。本文复核了先前建议，并补查 UniVTAC、TacEx、Bench2Dex、Tabero、VT-Refine、VTDexManip、ContactWorld、ManiFeel、Taccel 和 Isaac Lab 的公开代码/数据说明。结论是更新首选顺序，而不是直接开始安装。

## 结论

先前“ContactWorld + FORGE”的科学问题拆分仍然成立，但它不应继续作为唯一的本地主线。更适合当前机器和 OpenPI 目标的组合是：

1. **本地主线：UniVTAC 的 Isaac Sim 5.1 分支。** 它直接针对 Isaac Lab + TacEx，发布了 Isaac Sim 5.1 数据，明确支持 RTX 40/50，包含模拟 GelSight Mini、8 个任务、每任务 100 个 Isaac 5.1 HDF5 episodes，并有统一采集、训练和评估入口。
2. **外部数据/表示对照：ContactWorld USB。** 它仍是公开资料里最直接同时提供 point cloud、TacFF 和接触丰富示范的方案之一；但它依赖旧 Isaac Gym/TacSL，且本机 RTX 5090 兼容性未知。因此先做离线数据审计和最小兼容性测试，不把它当作本机仿真底座。
3. **可选 OpenPI 参考：Tabero。** 它已经把 Isaac Lab + Taxim/FOTS 触觉、LIBERO replay 和 π0/π0.5 接口接在一起，代码改动可参考；但公开主任务偏向温和抓取/放置，主要观测是 tactile/force，不能直接替代 point-cloud + 接触后纠偏 benchmark。
4. **可选双臂数据源：Bench2Dex。** 它有 Isaac Sim 5.1 安装说明、26 个任务、约 1.3K teleoperation demonstrations、8 种模态和 tactile replay；但采集依赖 Manus glove，公开说明中 depth 实际关闭以节省存储，点云硬条件不能直接满足。

## 候选比较

| 候选 | 触觉 | 点云/深度 | 已有动作数据 | 闭环任务/评估 | 本机风险 | 判断 |
|---|---|---|---|---|---|---|
| UniVTAC isaac51 | TacEx；GelSight Mini 等，含 RGB/depth/press-depth/marker/pose | 相机和深度可记录，可自行重建 PC；不是现成 PC 字段 | 每任务 100 个 HDF5；自动 scripted + cuRobo，失败 seed 跳过 | 8 任务、采集/评估脚本；已发布 checkpoint 主要对应 4.5 数据 | 中等，需验证本机 5.1 相机和 TacEx/UIPC | **本地主线** |
| ContactWorld | TacFF 10×14×3、TacRGB/TacDepth | 官方 point cloud 1024×6 | USB 等任务公开 demonstrations，SpaceMouse 示范带自然修正 | 官方 world-model/CEM planner，不等价于完整插入成功率 | **高**：旧 Isaac Gym/TacSL + RTX 5090 | **离线/外部对照** |
| Tabero | Taxim/FOTS GelSight、force history/force field | LIBERO 视觉/深度可用性需按其 replay 配置确认 | LIBERO replay 数据；已有 π0 LoRA 和 OpenPI 改造 | 9 个 LIBERO 任务，力/温和性指标 | 中等；Isaac Lab 2.3/Sim 5.1 有 gelpad binding bug，需 minicase workaround | OpenPI 接口参考 |
| Bench2Dex | TacMap/触觉 replay | 6 相机；depth 实际关闭，需自行重建/补录 | **1.3K teleop**，Manus glove；可下载 replay 数据 | 26 个双臂/灵巧手任务，ACT/DP/π0.5/GR00T 管线 | 中等；外设和数据量较大 | 双臂示范备选 |
| ManiFeel | TacSL GelSight/TacFF | 视觉，点云需自己重建 | 公开 USB/排序等 demos | supervised policy benchmark | **很高**：Python 3.8 + 特供 Isaac Gym 二进制 | 不做本地主线 |
| VTDexManip | 触觉视觉 | 主要是旧 Isaac Gym 管线 | 6 个灵巧手 RL 任务 | 有训练/评估脚本 | 高：Ubuntu 20.04/RTX3090/Isaac Gym | 历史参考 |
| VT-Refine | 触觉仿真/assembly | 取决于 ALOHA/SRL 数据配置 | 预训练/微调数据和 Docker | 双臂 assembly | 高，数据和环境链路较重 | 不作为第一轮 |
| Taccel/TacEx 单独使用 | 高吞吐视觉触觉/软体 | 可从相机深度重建 | 主要是 simulator/任务脚本，不是现成 OpenPI 示范集 | 可自行构造任务 | 集成成本高 | 作为后续加速/传感器组件 |

UniVTAC 的关键公开事实来自其 `isaac51` 分支 README 和 Collection 文档：Isaac 5.1 数据与 4.5 不互通；`isaac51/<task>/` 每个任务 100 个 HDF5 episodes；另有 638 个 contact-pretraining episodes；数据可以按 task/version 下载；采集器由 scripted policy + cuRobo 全自动运行，失败 seed 会被跳过。其 HDF5 明确保存触觉 `press_depth`、marker、pose、RGB，并保存 head/wrist RGB；深度是否保存由 observation 配置决定，因此必须实际下载一个 task 后审计，不能把“有触觉”推断成“已有 point cloud”。

ContactWorld 的优点是数据表示和问题贴合：官方仓库说明有 12 个接触任务、PointCloud/TacFF 配置、USB 数据和 checkpoint；但其安装脚本默认 Python 3.8、Isaac Gym TacSL 特供包和 ManiFeel，运行时与本机 Isaac Sim 5.1 并非同一栈。它更适合回答“公开 PC+TacFF 表示是否有信号”，不适合未经验证就作为 RTX 5090 的 simulator base。

Tabero 的优点是已经改造 OpenPI/π0 接口并提供 LeRobot 格式 tactile 数据；它的公开复现说明还明确指出 Isaac Lab 2.3 + Isaac Sim 5.1 在 gelpad 力传感绑定上有 bug，需要绑定 minicase，说明本机版本并非无条件兼容。它适合作为后续接口和 force-aware controller 的代码参考，不能直接证明点云和接触后纠偏已经被覆盖。

Bench2Dex 的价值在于真实 teleoperation 轨迹和双臂任务：README 给出 26 tasks、12 embodiments、约 1.3K demonstrations，并有 origin teleop 数据 replay 后再生成 tactile。它的公开说明同时写明 depth 目前实际关闭以控制存储，因此不能把它当作满足点云硬条件的现成数据集；若以后要做双臂扩展，再考虑下载少量任务。

## 数据如何获得：修正后的建议

不要把“全自动 scripted 数据”和“恢复数据”混为一类。

### 第一批：验证数据管线

从 UniVTAC `isaac51` 只下载 `insert_hole` 或 `insert_HDMI` 一个任务（100 episodes），并选择记录 RGB、depth、tactile RGB、press-depth、marker/pose、joint/action。目标是确认：

- 相机 depth 能否重建到机器人坐标系的点云；
- tactile 的时间戳、sensor pose 和力/压入量是否对齐；
- HDF5 能否转成 OpenPI 的一个 episode/window；
- 自己的 policy 能否在同一个 Isaac 5.1 环境里闭环。

这些 scripted episodes 适合管线 smoke 和 clean baseline。它们不自动证明 tactile 必须有用，因为成功轨迹是由规划器产生的，失败 seed 会被丢弃。

### 第二批：恢复数据

恢复数据不应等待一个“现成完美数据集”。在 UniVTAC 的成功插入轨迹上做可记录、可复现的 late perturbation：接近孔口后才施加横向/旋转残差，保存原始动作、扰动动作、接触事件、深度/点云、触觉和最终结果。使用一个固定的接触反馈控制器或受约束的 scripted recovery controller 产生标签；控制器的目标是“接触后纠偏并成功”，不是提前用真值预对齐。

每条轨迹保留 `clean`、`single-axis-recovery` 和 `compound-recovery` 标签，并记录扰动相对于实际 clearance 的比例。只有通过接触率、恢复率、方向覆盖率和峰值力检查的 bin 才扩大数据量。

### 第三批：自然示范对照

若需要人类自然修正，再下载 ContactWorld USB 或 Bench2Dex 的少量示范做外部对照。ContactWorld 更贴合 PC+TacFF，但需要旧环境；Bench2Dex 更贴合双臂 teleoperation，但点云需要额外补录/重建。两者都不应成为第一轮本地平台安装的阻塞项。

## 更新后的决策门

1. **UniVTAC Gate A：** 在本机 Isaac Sim 5.1/RTX 5090 上启动 `isaac51`，运行一个 insert task，拿到非空 RGB/depth/tactile，并完成 reset/step。
2. **UniVTAC Gate B：** 下载一个 task 的 100 episodes，自动报告 schema、episode 长度、动作、时间频率、depth 可用性、press-depth 非零率、marker/pose 对齐和成功率。
3. **UniVTAC Gate C：** 一个 episode 经 PC/tactile adapter 后通过 OpenPI forward/backward，tiny-set overfit，动作回放方向正确。
4. **Recovery Gate：** late perturbation 后，固定 controller 产生足够多的接触且能恢复；否则停止采集并调难度/控制器，不训练 VLA。
5. **ContactWorld Gate：** 只有当 A–C 通过后，才尝试旧 TacSL 的 GPU/相机兼容性；失败也不影响本地主线。

## 最终判断

我不再建议“先安装 ContactWorld，之后再想本机怎么做触觉”。更合适的是：**本机先用 UniVTAC isaac51 建立可复现的 Isaac Sim 5.1 + TacEx + HDF5 数据闭环；用 ContactWorld USB 作为外部 PC+TacFF 表示和自然修正对照；用 late perturbation 生成真正能检验触觉纠偏的受控数据；Tabero 用作 OpenPI/π0 接口参考。**

这仍保留了前面方案的核心科学设计，但降低了旧 Isaac Gym 在 RTX 5090 上的工程风险，也更直接覆盖你们已有 OpenPI 代码和 Isaac Sim 安装。

## 来源

- [UniVTAC GitHub](https://github.com/univtac/UniVTAC/tree/isaac51)
- [UniVTAC 数据集](https://huggingface.co/datasets/byml/UniVTAC)
- [TacEx](https://github.com/DH-Ng/TacEx)
- [ContactWorld](https://github.com/PokuangZhou/ContactWorld)
- [ContactWorld 数据集](https://huggingface.co/datasets/Pokuang/ContactWorld)
- [Bench2Dex](https://github.com/Bench2Dex/Bench2Dex)
- [Tabero](https://github.com/NathanWu7/Tabero)
- [Tabero-VTLA](https://github.com/NathanWu7/Tabero-VTLA)
- [VT-Refine](https://github.com/NVlabs/vt-refine)
- [VTDexManip](https://github.com/LQTS/VTDexManip)
- [ManiFeel](https://github.com/purdue-mars/manifeel)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
