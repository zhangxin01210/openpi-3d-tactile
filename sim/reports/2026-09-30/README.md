# UniVTAC `insert_hole` 阶段检查

检查日期：2026-09-30。只记录本机已经实测的事实；上游说明不算通过证据。

## 数据门

| 检查 | 结果 |
| --- | --- |
| HDF5 完整性 | 100/100 文件可读，0 个 `.incomplete`，schema 一致；所有数据列的首维帧数与本 episode 的 `step` 数一致 |
| 帧数 | 共 19,151 帧；每条 157–245 帧 |
| 时间步 | 所有相邻记录的 `step` 差均为 2（19,051 对）；上游配置为 120 Hz physics、60 Hz save，因此记录采样预期为 60 Hz |
| 动作 | 文件无独立 action 字段；上游 loader 使用下一行 joint/EEF 作为目标。7 个机械臂关节相邻位姿差范数平均约 0.00233 rad，EEF 平移相邻差平均约 0.000608 m |
| 数值 | `step`、joint、EEF 共计 0 个非有限值 |
| 标签 | `metadata.json` 中 100 条均为 `success`，`source_seed` 分布在 0–761；上游 scripted 收集器只保存成功轨迹，不能用 100/100 推断任务成功率，也不证明有纠偏示范 |
| RGB/触觉 | head、wrist RGB 以及左右 tactile RGB、press depth 均存在；3 条多模态样例视频已导出并解码检查 |
| 点云 | **未通过**：发布 HDF5 不含 head/wrist camera depth；仅有 tactile depth，不能冒充世界相机深度 |

完整逐 episode schema 报告见[`univtac_insert_hole.json`](univtac_insert_hole.json)，其中 head/wrist 的 depth、intrinsic、pose_w_opengl 六个字段全为缺失。样例为 [`episode_0.mp4`](samples/episode_0.mp4)、[`episode_49.mp4`](samples/episode_49.mp4)、[`episode_99.mp4`](samples/episode_99.mp4)。三个视频按同一 HDF5 行读取 head/wrist RGB、左右 tactile RGB 与 press depth，每 2 帧取 1 帧，覆盖完整轨迹；导出工具为 [`visualize_univtac_episode.py`](../../scripts/visualize_univtac_episode.py)。2026-10-01 已把旧样例改为 AV1 MP4，并导出[全部 100 条与本机重采 3 条的视频](../2026-10-01/README.md)。

`press_depth` 在视频第一帧就非零，因为双指夹持物体。因此压入量非零不能独立作为 peg/socket 首次接触事件；恢复统计必须另找环境接触真值或结合相对位姿和接触状态验收。

独立核对 `metadata.json` 中 100 条最终相对位姿：最大 `|x|=0.00143 m`、最大 `|y|=0.000553 m`，`z` 落在 `[-0.0556,-0.0446] m`，均满足上游 `insert_hole.py` 的位置阈值（横向各小于 0.01 m，插入深度大于 0.04 m）。此项仅核对已保存的最终标签；失败 seed 的完整轨迹没有出现在发布 HDF5 中。

## 平台门

专属 Conda 环境 `univtac-isaac51` 已有 Isaac Sim 5.1、PyTorch 2.7.0+cu128、vendored TacEx Python 包及 GCC 12/Ninja/系统 CUDA 12.8 工具链；Isaac Lab 的 core、assets、tasks 三个 editable 包已重绑到 `/home/sai/zx/openpi-sim-runtime/third_party/IsaacLab` 的固定 commit `3c6e67b`，import 路径已核对，`pip check` 无破损依赖。

独立 Isaac Lab 相机 smoke **通过**：[`result.json`](camera_smoke/result.json) 记录 15 个 step 后获得 `1×240×320×3` RGB、`1×240×320×1` depth、100% 有限深度及 3×3 内参；[`rgb.png`](camera_smoke/rgb.png) 中红色方块和地面格线可见，[`depth.npy`](camera_smoke/depth.npy) 的最近值约 3.2 m，对应 4 m 高的顶视相机与 0.8 m 高的方块顶面。加入 OpenGL 世界相机位姿后，[`cloud_stride2.ply`](camera_smoke/cloud_stride2.ply) 重建出 19,200 点，中心像素的世界高度为 0.8000004 m，与方块顶面吻合。脚本为 [`smoke_isaac_camera.py`](../../scripts/smoke_isaac_camera.py)，投影函数为 [`unproject_univtac_depth.py`](../../scripts/unproject_univtac_depth.py)。这证明本机基础 RGB/depth 与坐标换算可用，但还不是 UniVTAC 任务传感器验收。

CUDA 12.6 的 `nvcc --list-gpu-arch` 最高为 `compute_90`，本机 GPU 是 `compute_120`；系统 CUDA 12.8 可列出 `compute_120`，与当前 PyTorch cu128 匹配，因此 UIPC/触觉扩展使用系统 CUDA 12.8 和独立 GCC 12 构建。

UIPC **完整构建已通过**：本地 vcpkg Git registry mirror 使用上游固定的 Microsoft `dd3097e` 与 spiriMirror `7a5a816` baseline，33 个依赖全部安装；`tinygltf` 的官方 tag 新下载字节用项目 overlay 中记录的 SHA-512 验证。环境补充 `cmeel-urdfdom-headers==1.1.1` 后，`urdfdom` 的 CMake targets 能解析。CUDA 12.8、`CMAKE_CUDA_ARCHITECTURES=120`、GCC 12 完成全部 423 个目标，`pyuipc==0.9.0` 与 `tacex-uipc==0.1.0` 已安装，`uipc.Engine` 可导入。构建后的 Python 打包曾因在整个 UniVTAC 仓库执行 `git archive` 卡住；终止这两次归档子进程后，wheel 构建和安装完成。项目补丁 [`univtac_prebuilt_uipc_install.patch`](../../patches/univtac_prebuilt_uipc_install.patch) 可在已生成 `pyuipc` 后只安装 TacEx Python 包，避免重复编译。原构建日志保存在 `/tmp`，主机重启后已被清除。

cuRobo 固定于上游 commit `ebb7170`，五个 CUDA 扩展均可导入，RTX 5090 compute capability 12.0 可见；`pip check` 无破损依赖。

上一轮 UIPC CMake 日志 `/tmp/univtac-uipc-cmake5.log` 显示 vcpkg 下载 `tinygltf` 时 SHA-512 不符，这是源码下载/代理问题；不应把它解释为 RTX 5090 运行失败。

## 本机重采集：三条 depth 轨迹

GPU 空闲后使用 [`collect_univtac_depth_seed.sh`](../../scripts/collect_univtac_depth_seed.sh) 分别采集 seed 0、1、2。HDF5 位于 `/home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth/hdf5/`，共 669 帧（201、237、231），无 `.incomplete`。逐 seed `metadata.json` 均标为 `success`。[`univtac_insert_hole_depth_3seeds.json`](univtac_insert_hole_depth_3seeds.json) 的 `data_gate` 和 `pointcloud_gate` 均通过：逐帧列长度一致，相邻 `step` 差为 2，四路 RGB 每帧非空；两路世界相机 depth、内参、世界位姿逐帧有效。head 每帧有效深度比例至少 34.7%，wrist 为 100%；每条轨迹首/中/末帧均能重建非空世界点云。带双相机 depth 的 seed 0 视频为 [`depth_seed0.mp4`](samples/depth_seed0.mp4)。

上游采集器在 `--start_seed 2 --max_seed 2` 时打印 `1/3 (33.33%)`，其分母是 seed 编号，不是本次尝试次数；`suc_map.txt` 也会被每次独立启动覆盖。**不能引用这行百分比作为任务成功率。**三条已保存轨迹只是 clean scripted 示范和小规模数据链路验收，未包含可验证的受控纠偏。seed 2 第一次运行时主机重启，重跑后成功。

## OpenPI 基础样本接口

[`probe_univtac_sample.py`](../../scripts/probe_univtac_sample.py) 已从 seed 0 第 100 行抽取同步样本：`sim_step=702`，到下一记录间隔 `1/60 s`，head/wrist stride 4 世界点云分别为 2,873/8,160 点。输出保存在 `/home/sai/zx/openpi-sim-runtime/runs/univtac_seed0_row100_sample.{npz,json}`，保留了原 RGB-D、相机几何、双侧 tactile RGB/press depth/marker/pose 与下一记录 joint/EEF。

[`convert_univtac_lerobot.py`](../../scripts/convert_univtac_lerobot.py) 已生成本机 LeRobot v2 基础集 `/home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed`：三条 episode、共 666 个可预测帧，含 head/wrist/双侧 tactile RGB、9D joint state、下一帧 9D joint target 与原始 `step`。抽查每条首尾行，LeRobot `observation.state`、`action`、`source_step` 与 HDF5 完全对应；第 100 行 head RGB 的原始/解码后通道均值近似为 `[210.3,189.6,175.8]`/`[209.4,188.5,174.7]`，AV1 压缩后的平均像素绝对差为 1.71。`action` 是上游 loader 的下一记录目标约定，**不是独立记录的 motor command**。世界 RGB-D 和 tactile press depth 仍以 HDF5 为准；LeRobot 集只完成基础时序和 RGB 接口，不代表点云/触觉已进入 π0 或完成训练。

UniVTAC GelSight 的 `press_depth` 为胶层正压入量（mm），不是三维力向量。当前 XHand structured encoder 的 tactile 合约为 3D taxel position + 3D force，不能将该压入量直接冒充力；后续需定义独立的仿真触觉表示及对应 encoder/transform。

RGB/关节基线的 [`LeRobotUniVTACDataConfig`](../../../src/openpi/training/config.py) 和 [`univtac_policy.py`](../../../src/openpi/policies/univtac_policy.py) 已接入 OpenPI 数据变换。两条真实样本经过 head/wrist RGB、9D joint、4 步动作窗口、7 个机械臂关节 delta 与两个绝对夹爪目标的变换；逆变换恢复原下一帧目标。[`univtac_openpi_rgb_smoke.json`](univtac_openpi_rgb_smoke.json) 记录了随机初始化的 **Pi0 dummy 小模型** 在 CPU 上 batch=2 的 forward/backward：loss 有限、50 个梯度叶均有限。随后 [`overfit_univtac_openpi_rgb.py`](../../scripts/overfit_univtac_openpi_rgb.py) 在 GPU 上固定这两条样本和扩散噪声，100 次更新后 loss 从 1.7574 降到 0.000354（[`univtac_openpi_rgb_tiny_overfit.json`](univtac_openpi_rgb_tiny_overfit.json)）。因此基础 RGB/关节数据和优化路径的小样本过拟合检查通过。**这不是预训练 π0、点云/触觉编码、归一化统计或闭环成功证据。**

随后 [`smoke_univtac_openpi_pointcloud.py`](../../scripts/smoke_univtac_openpi_pointcloud.py) 从同一真实样本的 head/wrist 世界点云各取 1,024 点，作为 visual-only spatial input 注入随机初始化的 structured π0；[`univtac_openpi_pointcloud_smoke.json`](univtac_openpi_pointcloud_smoke.json) 记录 loss 1.342，空间分支 39 个梯度叶中 15 个非零，前向/反向均通过。未启用的 tactile 分支梯度为零属于预期。

现已通过 [`UniVTACVisualDataset`](../../../src/openpi/spatial_dataset/univtac.py) 在 OpenPI dataloader 中按 episode/frame/source `step` 回连原始 HDF5，逐帧用内参与 OpenGL 世界位姿重建 head/wrist 点云，各取 1,024 点。[`univtac_openpi_pointcloud_all_rows.json`](univtac_openpi_pointcloud_all_rows.json) 覆盖全部 666 行：200/236/230，所有行 step 对齐且点云有限；第 100 行与独立探针逐点相同。[`overfit_univtac_openpi_pointcloud.py`](../../scripts/overfit_univtac_openpi_pointcloud.py) 在真实 RGB+点云双样本上做 100 次固定噪声更新，loss 从 1.5928 降到 0.003566（[`univtac_openpi_pointcloud_tiny_overfit.json`](univtac_openpi_pointcloud_tiny_overfit.json)）。这是数据与优化链路验收，不证明点云比 RGB 有益。

双侧 GelSight RGB 现在可按需上下拼成一个保留纵横比的触觉图像，放入 π0 的第三图像槽；`tactile_rgb=false` 时该槽被屏蔽。[`tactile_pair_seed0_row100.png`](samples/tactile_pair_seed0_row100.png) 是实际输入的可视化。触觉信号仍是原始图像，不是力或 XHand taxel。随机小模型在“RGB+GelSight RGB”和“RGB+世界点云+GelSight RGB”两组真实双样本上均完成前向/反向；见 [`univtac_openpi_tactile_image_smoke.json`](univtac_openpi_tactile_image_smoke.json)。四组模态开关现在均可构造：RGB、RGB+PC、RGB+触觉图像、RGB+PC+触觉图像；没有训练四组正式模型，也没有接触后性能比较。

[`compute_univtac_norm_stats.py`](../../scripts/compute_univtac_norm_stats.py) 已遍历 666 行，按 50 步动作窗口统计变换后的 9D state 和动作均值、标准差，写入独立运行目录 `/home/sai/zx/openpi-sim-runtime/assets/univtac_insert_hole_depth_3seed`，保存/读取回环通过；摘要见 [`univtac_openpi_norm_stats.json`](univtac_openpi_norm_stats.json)。仅三条 clean 轨迹，统计量只适合当前小样本管线。

上游回放器用相同配置把 seed 0、1、2 的 HDF5 关节目标逐帧送回 `insert_hole`。[`univtac_insert_hole_depth_replay_3seeds.json`](univtac_insert_hole_depth_replay_3seeds.json) 记录 201/237/231 个目标全部执行；三条在轨迹期间达到成功条件，终态仍成功，关节平均绝对跟踪误差约 `1.53e-5`–`1.84e-5 rad`。每条完整误差见 [`seed0`](univtac_insert_hole_depth_seed0_replay.json)、[`seed1`](univtac_insert_hole_depth_seed1_replay.json)、[`seed2`](univtac_insert_hole_depth_seed2_replay.json)。这是三条**筛选过的 clean 示范回放**，不代表随机初始状态的任务成功率，更不代表学习到的 policy 闭环成功率。

## 实时接口与孔座偏移小试

[`run_univtac_live_trial.py`](../../scripts/run_univtac_live_trial.py) 现在每个控制步从 Isaac 任务读取实时 head/wrist RGB、双侧 GelSight RGB、9D joint，并可选重建两路各 1,024 点的世界点云，经 OpenPI 兼容的 WebSocket msgpack 协议请求动作，再调用任务的 `take_action(qpos)`。当前服务端 [`serve_univtac_recorded_reference.py`](../../scripts/serve_univtac_recorded_reference.py) 按 `source_index` 返回 HDF5 录制目标，**不是学习策略，也不使用观测作决策**。正式在线输入键已与 `UniVTACInputs` 对齐；[`policy_config.py`](../../../src/openpi/policies/policy_config.py) 保留在线点云 `spatial` 键，通过 CPU 上的输入变换检查。两步实际仿真检查发送了每步 2,048 个世界点，服务端响应并执行了两次动作，见 [`univtac_live_contract_pc_seed0_2steps.json`](univtac_live_contract_pc_seed0_2steps.json)。两步检查的终态失败是预期的，不能当作任务失败样本。

同一 seed 0 的完整参考目标试验在第 170 个控制步平移 kinematic 孔座，随后继续发送同一组录制关节目标；汇总见 [`univtac_live_pilot_summary.json`](univtac_live_pilot_summary.json)：

| 孔座 y 偏移 | 201 步执行 | 终态插入 | 扰动后 peg 相对孔座下降 | 右减左 GelSight 最大压入量的后段中位差 |
| --- | --- | --- | ---: | ---: |
| 0 mm | 全部 | 成功 | 32.45 mm | 0.285 mm |
| +1.0 mm | 全部 | 成功 | 32.85 mm | 0.290 mm |
| -0.5 mm | 全部 | 成功 | 32.97 mm | 0.284 mm |
| -1.0 mm | 全部 | 失败 | 0.02 mm | 0.441 mm |

四组的关节平均绝对跟踪误差均约 `1.7e-5`–`2.1e-5 rad`。`-1 mm` 组的 peg 在偏移后几乎停在原高度，而机械臂仍跟踪目标，说明这是一条有用的失败难例；左右压入量差随之增大，但这两路传感器首先感知夹爪与 peg 的接触，**不能据此断言 peg–slot 发生接触或得到接触力**。

随后用正式 `univtac_policy_raw_v1` 输入结构，分别重新运行 clean 和 `-1 mm y` 两条完整轨迹，均逐步发送 2,048 个实时世界点；前者成功，后者失败，见 [`clean_pc_contact`](univtac_live_clean_pc_contact_seed0.json) 与 [`minus1mm_pc_contact`](univtac_live_slot_y_minus1mm_pc_contact_seed0.json)。这两份历史报告的接触探针后来发现混入第三物体参与的原语，**其接触计数已废弃**；原始轨迹的任务结果和点云通信结果仍有效。

接触测量现按更严格的定义重新运行：仅选 UIPC PP/PE/PT/EE 法向接触原语，要求拓扑的**每一个顶点**都属于 peg 或孔座，且两者均出现；PH 隐式平面和混入第三物体的原语全部排除。逐原语梯度索引与拓扑匹配，选定原语在 peg/孔座上的作用反作用残差接近数值零。第 165–169 步，两条轨迹的纯 peg–slot 接触原语中位数都为 27；第 170–200 步，干净组为 25，`-1 mm` 组为 331。UIPC 法向势垒梯度除以物理步长平方所得的**模型内暂定力估计**后段中位数分别约为 3.7 N 和 669 N；它尚未经过独立力传感校准，不能直接规定安全阈值。完整定义、统计和数值检查见 [`univtac_contact_metric_gate_seed0.json`](univtac_contact_metric_gate_seed0.json)，逐步原始数据见 [`clean`](univtac_contact_gradient_clean_seed0_full.json) 与 [`-1 mm`](univtac_contact_gradient_minus1mm_seed0.json)。旧探针报告已标记为被此指标覆盖。

随后在 seed 1、2 各补做一组干净/逆向偏移对照。三条轨迹的扰动索引按录制时 peg 相对孔座的 `z≈83 mm` 选取；偏移方向均沿扰动前横向 `y` 残差的反方向移动孔座，使残差扩大。下表是从严格探针的完整逐步报告重新计算的扰动后至终点中位数，暂定势垒力仅可在相同仿真配置下排序：

| seed / 扰动索引 | 孔座 y 偏移 | 干净 / 偏移终态 | 严格 peg–slot 接触数中位数（干净 / 偏移） | 暂定势垒力中位数（干净 / 偏移） | peg 相对孔座 z 下降（干净 / 偏移） |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 / 170 | −1 mm | 成功 / 失败 | 25 / 331 | 3.7 / 669 N | 33.44 / 0.07 mm |
| 1 / 191 | +1 mm | 成功 / 失败 | 17 / 519 | 2.8 / 432 N | 31.47 / −0.08 mm |
| 2 / 185 | −1 mm | 成功 / 失败 | 35.5 / 358 | 4.1 / 980 N | 31.54 / 0.06 mm |

逐 seed 的五步扰动前/全段扰动后统计、坐标、作用反作用残差及六份原始报告索引见 [`univtac_contact_cross_seed_sweep.json`](univtac_contact_cross_seed_sweep.json)。六次均执行全部录制关节目标，探针原语与梯度拓扑的索引不匹配数均为零。seed 1 的原始报告见 [`clean`](univtac_contact_gradient_clean_seed1.json)、[`+1 mm`](univtac_contact_gradient_slot_y_plus1mm_seed1.json)；seed 2 见 [`clean`](univtac_contact_gradient_clean_seed2.json)、[`−1 mm`](univtac_contact_gradient_slot_y_minus1mm_seed2.json)。扰动时刻只按录制相对高度匹配，尚未逐 seed 确认首次接触时刻；这六条受控回放不能估计任务成功率、恢复率或触觉收益。参考目标完全不使用观测作决策，尚未进行纠偏。

最初的闭环尝试错用了采集模式的 120 Hz 控制，201 步虽执行却未插入；原始回放使用 60 Hz。该诊断保存在 [`univtac_live_timing_mismatch_seed0.json`](univtac_live_timing_mismatch_seed0.json)，正式小试已固定为评估模式的 60 Hz。四条完整物理试验使用了早期嵌套请求结构，由参考服务端消费；之后改成真实 UniVTAC policy 的顶层输入键，并以两步实时试验验收。汇总 JSON 的 `request_schema` 区分这两个版本。

## 下一道门

1. 已在三个 seed 和正/负两个 y 方向复现逆向 `1 mm` 偏移造成的卡滞。下一步应在这些 seed 的邻近时刻和 `0.5–1 mm` 区间定位边界；暂定势垒力可作同配置下的强度排序，安全上限仍需独立校准或经控制器试验约束。
2. 实现使用仿真真值的恢复控制器，验收其在真实接触后能安全纠偏，再扩大采集。学习策略的完整训练与闭环成功率尚未验证。

世界相机点云还需要每帧相机位姿和内参。采集配置要求 `intrinsic`、`pose_w_opengl`，对应的上游最小改动保存在 [`univtac_camera_geometry.patch`](../../patches/univtac_camera_geometry.patch)；三条真实轨迹的所有帧已通过这些字段的审计。

真实任务的自动检查入口为 [`smoke_univtac_insert_hole.py`](../../scripts/smoke_univtac_insert_hole.py)。seed 0 的完整 reset 与三步观测 **通过**，结果与输出图见 [`insert_hole_smoke/result.json`](insert_hole_smoke/result.json)：head/wrist RGB 均为 270×480，head 有效世界深度约 34.7%（0.50–1.48 m），wrist 有效深度 100%（0.082–0.278 m）；两侧触觉 RGB 均为 240×320，最大正压入量分别约 0.450 和 0.739 mm。两路内参和 OpenGL 世界位姿有限，按 [`unproject_univtac_depth.py`](../../scripts/unproject_univtac_depth.py) 输出的彩色点云分别有 2,842 与 8,160 点。head 点云的世界 z 范围约 -0.002–0.540 m，wrist 为 0.585–0.781 m，和相机所见工作台/手中物体的高度相符。对应的 RGB PNG、深度 NPY、触觉 PNG/压入量 NPY 和 PLY 均在 [`insert_hole_smoke`](insert_hole_smoke) 目录。这个检查证实任务传感器可运行，但只是一帧观测，不能替代逐帧 HDF5 审计。

默认 Isaac Lab PhysX GPU 缓冲区曾在场景构造时报 CUDA allocation error；单环境容量调小后完成构造和 reset。容量已写进重采集配置，并由 [`univtac_physx_config_overrides.patch`](../../patches/univtac_physx_config_overrides.patch) 接入上游配置解析器。首次正式采集尝试在 cuRobo 规划时显存不足：已有的 `scripts/serve_policy.py` 进程占用约 24.03 GiB，仿真进程占用约 7.27 GiB，剩余 2.06 MiB；错误摘录见 [`depth_collect_attempt_0_error.txt`](depth_collect_attempt_0_error.txt)。该次没有生成 HDF5，不是独占 GPU 下的模拟器失败；GPU 空闲后采集成功。[`collect_univtac_depth_seed.sh`](../../scripts/collect_univtac_depth_seed.sh) 在启动前检查至少 12,000 MiB 空闲显存和是否已有同 seed 文件。
