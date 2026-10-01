# PC + Tactile 仿真平台：初步方案与本机核查

> **OpenPI USB 训练准备（2026-10-01）**：已导出 LeRobot v2.1 数据包 `data/contactworld_usb_positive_all/`（145 条通过位姿筛选的 USB 轨迹；训练仅读取 117 条训练轨迹的完整 16 步窗口），配置了 11 组单任务实验与并行启动脚本。已完成逐集文件核验、真实批次读取和小模型的前向/反向检查；尚未在 A800 上实际启动训练。同步清单和命令见[训练交接说明](reports/2026-10-01/contactworld/openpi_usb_training_matrix.md)。

> **最新检查点（2026-10-01）**：ContactWorld 全部 354 条 USB/Peg 发布轨迹已审计；终点通过当前源码位姿判据的为 145/201、126/153。仅用关节位置、速度和步数训练的 MLP 在两个任务各 20 个新随机初态上均 0 次达到源码判据；成功重放各 1 条的 10 步停稳正对照通过。见[数据审计与闭环基线报告](reports/2026-10-01/contactworld/proprio_baseline_pilot.md)和[两条基线录像](/home/sai/zx/openpi-sim-runtime/runs/contactworld_baseline_20261001/index.html)。尚未开始 OpenPI 多模态训练；下方横幅是历史节点。

> **最新验收节点（2026-10-01）**：ContactWorld 隔离环境与 USB/Peg 发布资产已固定；预先选定的 5 条 USB、3 条 Peg 完整示范通过源码终点判据，RGB/触觉/在线点云逐帧审计和 PyTorch 训练数据读取接口已完成。重放依赖记录的末端目标，不是策略成功率；历史逐帧相机标定及真实训练仍待后续。见[阶段 1–2 验收报告](reports/2026-10-01/contactworld/stage1_stage2_validation.md)和[8 条同步视频](/home/sai/zx/openpi-sim-runtime/runs/contactworld_multireplay_20261001/batch_final/index.html)。以下各阶段横幅保留为历史记录。

> **闭环示范重放已修复本条 USB 卡孔问题**：跟踪记录的末端控制目标，在 2 个初始化种子下完整插入，末帧插件误差 0.027/0.016 mm。未修改物理参数；原动作开环复现仍失败，不作为策略成功率。见[修复与复跑说明](reports/2026-10-01/contactworld/replay_review.md)。

> **接触分叉诊断已完成**：已修正初始化重新夹紧，纯视觉改白不影响任何记录的动力学/触觉数组；全程仍在第 64 帧附近接触后分叉。第 75 帧已插入状态的局部恢复可保持，不能算完整成功。当前暂停扩大样本与训练，见[诊断记录](reports/2026-10-01/contactworld/replay_review.md)。

> **最新重放审查**：已纠正 USB 资产来源不一致；示范初态恢复后插件偏差约 0.027 mm，82 步执行完成，但末帧偏差约 16.23 mm、插入判据未通过。当前不训练。见[重放与初始化审查](reports/2026-10-01/contactworld/replay_review.md)。

> **ContactWorld 本机自编译与最小 USB 验收已通过（2026-10-01 16:00）**：独立环境中 Torch CUDA、gymtorch、GPU 物理/相机、单 USB reset 与 8 步传感器输出均通过；约 36 分钟达到节点并停止。原环境包快照未变。完整示范重放、在线对齐及训练仍未完成。见[执行记录](reports/2026-10-01/contactworld/source_build_execution.md)。以下为较早记录。

> **最新用户偏好（同日后续）**：ContactWorld USB/Peg 的任务最符合需求，恢复为优先路线；可接受几个小时的自编译尝试，不接受无期限适配。建议首轮以 4 小时为停止检查点，实际编译尚未开始，不能保证完整 TacSL 在该时限内通过。Bench2Dex 双臂拼图因任务复杂度暂缓推进。见[时间估计与尝试边界](reports/2026-10-01/contactworld/source_build_plan.md)。以下为较早阶段记录。

> **用户更新标准后的决定（2026-10-01）**：初始物体随机化已满足当前视觉需求，Bench2Dex 任务 73 恢复为近期主候选；三条样本的位置/朝向变化与官方低成功率已核对，下一步验证动态控制闭环和训练数据契约。ContactWorld 自编译成本已获接受，完成资源核查，估计通常 1–3 天，尚未启动长时间编译。见[Bench2Dex 重新审查](reports/2026-10-01/bench2dex/reconsideration.md)和[ContactWorld 工期估计](reports/2026-10-01/contactworld/source_build_plan.md)。TacEx 回到备用，偏移恢复采集仍暂缓。以下是同日较早决定的历史记录。

> **最新选型节点（2026-10-01，同日后续）**：ContactWorld 限定范围排查结束，暂缓本机自编译旧栈。Bench2Dex 在当前 5090 上完成任务 73 的三条 RGB/depth/相机参数/10 指 TacMap 回放，共 2452 帧；胸前点云平面一致性通过，但任务装配目标 XY 固定、标签存在矛盾，暂不作为正式训练任务。建议在限定工程预算下转向 TacEx 自定义随机化任务设计，Bench2Dex 保留为灵巧手备选。详见[本轮结论与复跑说明](reports/2026-10-01/bench2dex/README.md)及[三条同步视频](/home/sai/zx/openpi-sim-runtime/visualizations/bench2dex_review/index.html)。偏移恢复采集和正式训练仍未开始。下文为此前记录。

> **当前阶段（2026-10-01）**：已完成 ContactWorld USB / Peg 的数据选型审查：354 条下载数据、20 条完整同步视频，见[数据审查报告](reports/2026-10-01/contactworld/README.md)。随后建立独立旧环境：GPU PhysX、原生 RGB/depth、前相机矩阵和点云校正已通过；**Torch 2.4.1 CUDA 算子在 RTX 5090 上报 no kernel image，完整 TacSL/reset/step 与在线数据对齐尚未完成**。已按约定停在兼容性节点，见[兼容检查与新发现的半像素偏差](reports/2026-10-01/contactworld/compatibility.md)。偏移恢复采集暂缓，未开始正式训练。现有 UniVTAC 运行平台保留作工程验证，实测历史见 [`reports/2026-09-30/README.md`](reports/2026-09-30/README.md)；下方早期路线内容保留为历史记录。

100 条下载版和 3 条本机重采轨迹的全量可播放视频见 [`2026-10-01 视频检查入口`](reports/2026-10-01/README.md)。本机浏览页位于 `/home/sai/zx/openpi-sim-runtime/visualizations/univtac_insert_hole/index.html`。

本轮可复查入口：

```bash
# 已下载数据的离线审计和同步视频（专属环境）
/home/sai/miniconda3/envs/univtac-isaac51/bin/python sim/scripts/audit_univtac_hdf5.py \
  /home/sai/zx/openpi-sim-runtime/data/isaac51/insert_hole \
  --output sim/reports/2026-09-30/univtac_insert_hole.json
/home/sai/miniconda3/envs/univtac-isaac51/bin/python sim/scripts/visualize_univtac_episode.py \
  /home/sai/zx/openpi-sim-runtime/data/isaac51/insert_hole/hdf5/0.hdf5 \
  sim/reports/2026-09-30/samples/episode_0.mp4 --stride 2

# 全量视频：100 条下载版 + 3 条本机重采，60 fps AV1，逐条解码验收
/home/sai/miniconda3/envs/univtac-isaac51/bin/python \
  sim/scripts/export_univtac_review_videos.py

# Isaac Lab 基础 RGB/depth smoke（不依赖 TacEx/UIPC）
OMNI_KIT_ACCEPT_EULA=YES /home/sai/miniconda3/envs/univtac-isaac51/bin/python \
  sim/scripts/smoke_isaac_camera.py --headless \
  --output-dir /home/sai/zx/openpi-sim-runtime/runs/isaac_camera_smoke

# UIPC、TacEx、cuRobo 安装完成后的真实任务检查
OMNI_KIT_ACCEPT_EULA=YES /home/sai/miniconda3/envs/univtac-isaac51/bin/python \
  sim/scripts/smoke_univtac_insert_hole.py --headless \
  --output-dir /home/sai/zx/openpi-sim-runtime/runs/insert_hole_smoke

# 已采集 seed 0、1、2；其他 seed 可在 GPU 显存空闲时同法采集
# bash sim/scripts/collect_univtac_depth_seed.sh <seed>

# 审计已采集的 RGB-D/tactile HDF5 和逐帧相机几何
/home/sai/miniconda3/envs/univtac-isaac51/bin/python sim/scripts/audit_univtac_hdf5.py \
  /home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth \
  --output sim/reports/2026-09-30/univtac_insert_hole_depth_3seeds.json

# 从一帧核对 RGB-D / 世界点云 / GelSight / 下一帧目标的时间契约
/home/sai/miniconda3/envs/univtac-isaac51/bin/python sim/scripts/probe_univtac_sample.py \
  /home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth/hdf5/0.hdf5 \
  --row 100 --output /home/sai/zx/openpi-sim-runtime/runs/univtac_seed0_row100_sample.npz

# 转成 OpenPI 可读取的 LeRobot 基础 RGB/关节数据；输出目录须不存在
/home/sai/openpi_chem/openpi/.venv/bin/python sim/scripts/convert_univtac_lerobot.py \
  /home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth \
  --output /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed

# CPU 上用两条真实样本验收 OpenPI RGB/关节动作变换和小模型梯度
JAX_PLATFORMS=cpu /home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/smoke_univtac_openpi_rgb.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  --output sim/reports/2026-09-30/univtac_openpi_rgb_smoke.json

# GPU 上固定两条样本与扩散噪声，验证小模型可以过拟合基础 RGB/关节数据
/home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/overfit_univtac_openpi_rgb.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  --steps 100 --output sim/reports/2026-09-30/univtac_openpi_rgb_tiny_overfit.json

# 一个真实世界点云样本进入 visual-only structured π0 的梯度检查
/home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/smoke_univtac_openpi_pointcloud.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  /home/sai/zx/openpi-sim-runtime/runs/univtac_seed0_row100_sample.npz \
  --output sim/reports/2026-09-30/univtac_openpi_pointcloud_smoke.json

# 全 666 行逐帧点云 join 验收与两样本优化检查
JAX_PLATFORMS=cpu /home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/audit_univtac_openpi_pointcloud.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  --output sim/reports/2026-09-30/univtac_openpi_pointcloud_all_rows.json
/home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/overfit_univtac_openpi_pointcloud.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  --steps 100 --output sim/reports/2026-09-30/univtac_openpi_pointcloud_tiny_overfit.json

# 两路 GelSight RGB 作为触觉图像的模态检查
/home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/smoke_univtac_modalities.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  --output sim/reports/2026-09-30/univtac_openpi_tactile_image_smoke.json

# 当前小样本 state/action 归一化统计；目标 assets 目录须不存在
JAX_PLATFORMS=cpu /home/sai/openpi_chem/openpi/.venv/bin/python \
  sim/scripts/compute_univtac_norm_stats.py \
  /home/sai/zx/openpi-sim-runtime/data_lerobot/univtac_insert_hole_depth_3seed \
  --assets-dir /home/sai/zx/openpi-sim-runtime/assets/univtac_insert_hole_depth_3seed \
  --report sim/reports/2026-09-30/univtac_openpi_norm_stats.json

# 实时观测/动作 WebSocket 小试：在另一个终端启动录制目标参考服务端
/home/sai/miniconda3/envs/univtac-isaac51/bin/python \
  sim/scripts/serve_univtac_recorded_reference.py \
  /home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth/hdf5/0.hdf5 \
  --port 8765
# 在 UniVTAC 源码目录运行 Isaac；服务端只是接口参考，并非学习策略
cd /home/sai/zx/openpi-sim-runtime/third_party/UniVTAC-full
OMNI_KIT_ACCEPT_EULA=YES /home/sai/miniconda3/envs/univtac-isaac51/bin/python \
  /home/sai/zx/openpi-3d-tactile/sim/scripts/run_univtac_live_trial.py \
  --headless \
  --hdf5 /home/sai/zx/openpi-sim-runtime/data_depth/insert_hole/univtac_insert_hole_depth/hdf5/0.hdf5 \
  --provider websocket --websocket-uri ws://127.0.0.1:8765 \
  --offset-mm 0 -1 --perturb-index 170 --include-pointcloud --probe-contact \
  --output /home/sai/zx/openpi-sim-runtime/runs/live_trial_slot_y_minus1mm.json

# 在 UniVTAC 源码目录运行，回放已采集的三条 clean 示范
# OMNI_KIT_ACCEPT_EULA=YES /home/sai/miniconda3/envs/univtac-isaac51/bin/python \
#   scripts/replay.py insert_hole \
#   /home/sai/zx/openpi-3d-tactile/sim/configs/univtac_insert_hole_depth.yml \
#   --headless --seeds 0 1 2
```

重采集世界相机 depth 时，先在固定 `d541e5568227ca3b66104d294f63c80acad7c52c` 的 UniVTAC 源码上应用 [`univtac_camera_geometry.patch`](patches/univtac_camera_geometry.patch) 与 [`univtac_physx_config_overrides.patch`](patches/univtac_physx_config_overrides.patch)，再使用 [`univtac_insert_hole_depth.yml`](configs/univtac_insert_hole_depth.yml)。它同时保存 head/wrist 的 RGB、depth、内参和 OpenGL 世界位姿，并为单环境设置已通过 smoke 的 PhysX 容量。TacEx/UIPC 构建用 [`configure_univtac_uipc_sm120.sh`](scripts/configure_univtac_uipc_sm120.sh)；本机 vcpkg registry 网络异常时，可应用 [`univtac_vcpkg_registry_mirror.patch`](patches/univtac_vcpkg_registry_mirror.patch) 并设置 `OPENPI_VCPKG_REGISTRY_MIRROR` 指向含上游两个固定 baseline 的本地 Git mirror。已编译 `pyuipc` 后，可应用 [`univtac_prebuilt_uipc_install.patch`](patches/univtac_prebuilt_uipc_install.patch)，用 `OPENPI_USE_PREBUILT_UIPC=1 pip install --no-build-isolation --no-deps -e third_party/TacEx/source/tacex_uipc` 安装 Python 包。真实任务 reset/step、四路视觉/触觉 smoke 和三条逐帧 depth HDF5 审计均已通过；666 帧 LeRobot 基础集和全部 666 行点云加载已核对。RGB、RGB+点云两组随机小模型通过双样本过拟合，RGB+触觉图像及全模态图像/点云组合通过前向反向检查。三条 clean 关节目标在模拟器回放均完成插入。实时参考服务端闭环与三个 seed 的孔座偏移接触小试见[阶段报告](reports/2026-09-30/README.md)；下一道门是扰动边界细化、恢复控制器和学习策略闭环评估。

核查日期：2026-09-29。状态：完成环境盘点、上游源码/下载清单核对及一次现有 Isaac 环境复测；尚未完成平台安装、USB 数据下载、任务验收或 OpenPI 仿真适配。

本方案参考用户提供的 `deep-research-report(1).md` 和 `msg.txt`，其中的安装命令、实验建议和旧结论视为待核实的参考材料。当前交付是用户要求的、基于本机情况的初步方案。

## 建议

保留 ContactWorld USB 作为首个现成数据验证任务；复用现有 Isaac Sim 5.1 的安装经验和软件包，建立本项目独立的 Isaac Lab/FORGE 平台。先把 ContactWorld 在 RTX 5090 上的兼容性作为独立验收项，不能因为本机装了 Isaac Sim 就认定 ContactWorld 可以直接运行。

两条路线按顺序投入：先完成基础平台检查和 USB 数据审计；ContactWorld 闭环通过后再安排第一轮训练；FORGE 的受控纠偏采集作为后续机制实验。若 ContactWorld 的旧二进制确实不能在本机运行，保留离线数据路线，将闭环评估迁往兼容 GPU 主机，或把本机 FORGE 提升为主线。移植 ContactWorld 到 Isaac Lab 是另外一项工程，不能默认视为官方 benchmark 的等价实现。

每个阶段记录：验证问题、通过条件、失败证据、下一步。数据量、环境数量和训练时长本身不作为进展指标。

## 已核查的本机情况

| 项目 | 事实 | 对方案的影响 |
| --- | --- | --- |
| GPU | RTX 5090，32607 MiB，compute capability 12.0；驱动 580.173.02 | Isaac Gym/TacSL 旧二进制与 GPU 扩展需要单独实测；容器不能消除 GPU 架构不兼容 |
| 系统 | Ubuntu 24.04.3，内核 7.0.0-34-generic；Ryzen 9 9950X，249 GiB 内存 | 足以开展本地仿真、数据处理和小规模评估；模型与渲染同时运行的显存仍须测量 |
| 内部磁盘 | `/home/sai` 所在 ext4 分区剩余约 109 GiB | 首轮仅 USB；复制环境、解压、派生数据和 checkpoint 均须计入空间峰值 |
| 外置磁盘 | 当前挂载的 NTFS3 卷剩余约 440 GiB、1.6 TiB、1.7 TiB | 可作为后续资产存储候选；本次未选择或写入任何外置卷，不把环境安装在 NTFS 上 |
| 现有 Isaac 环境 | `isaacsim45` 约 15 GiB；`isaacsim51` 约 17 GiB；`gpt-policy-isaac51` 约 20 GiB | 优先参考最后一个已适配 5090 的环境，而非重新改动全局驱动/CUDA |
| GPT policy 环境 | Python 3.11.15；Isaac Sim 5.1.0.0；Torch 2.7.0+cu128；NumPy 1.26.0 | `pip check` 和本次 CUDA 矩阵运算通过 |
| Isaac Lab | `/home/sai/zx/llm as policy/third_party/IsaacLab`，官方 v2.3.0，commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77` | 源码工作树检查时干净；环境中为 editable 安装，单纯复制 conda 环境仍会指向别人源码 |
| FORGE | 本机源码注册了 `Isaac-Forge-PegInsert-Direct-v0` 等任务 | 任务尚未在本次核查中启动，不代表资产、控制、接触或 teacher 已通过 |
| 触觉能力 | 该版本源码未找到 `VisuoTactileSensor` / TacSL 实现；FORGE 有 F/T 路径 | 净力/力矩仅作为初期接触信息对照，不能当作空间 taxel 阵列已经实现 |
| OpenPI 环境 | `.venv -> /home/sai/openpi_chem/openpi/.venv`；editable `openpi` 指向本代码库 | 是共享环境，直接 `uv sync` / `pip install` 可能影响另一个项目；新建专属环境 |
| OpenPI 依赖 | Python >=3.11，Torch 2.7.1，JAX/JAXlib 0.5.3，NumPy 1.26.4（现有环境） | 训练、转换/审计、仿真分别管理依赖；本次未验收 JAX/模型 GPU 推理 |
| Git | 最终复查本地 main 与通过 origin SSH 实时查询的远程 main 均为 `165493a54636160b0afd40660d2fb16554667a37` | 当前没有 main 提交差异需要合并；不能覆盖已有未提交内容 |

初次检查时已有本地修改：`.gitignore`；未跟踪的 `diagnostic_runs.zip`、`diagnostic_runs/`、`diagnostics.zip`、`src/openpi/policies/xhand_policy_test.py`。核查期间仓库从 `7d0dcd1` 更新到 `165493a`，最终检查 `.gitignore` 已无本地差异，以上未跟踪文件仍在；本次未执行 commit、pull 或更改这些文件。此处 Git 结论是核查时刻快照，正式编码前仍需重新核对。

### 实际启动复测：相机检查未通过

原项目 2026-09-23 保存的基础检查和 RoboDojo 检查为通过。本次用同一 Python 和原检查脚本重新运行，输出、Kit 配置与缓存放在独立临时目录：

```text
/tmp/openpi-sim-audit-20260929-6zCGEW/isaac_smoke/
```

本次结果：CUDA 矩阵计算通过；SimulationApp 启动；Isaac Lab controller 导入通过；200 次 world step 后 `camera.get_rgba()` 返回 shape `(0,)`，未满足 `(240, 320, 4)`。日志同时出现 viewport 等待超时。根因尚未确定，不能仅据此归因于显卡、驱动或 cold cache。原脚本在该断言之后才验收物理位置和 reset，因此本次不能把这两项记为通过。

机器可读结果见 [reports/2026-09-29/isaac_smoke.json](reports/2026-09-29/isaac_smoke.json)。原始日志位于上述临时目录，可能随系统清理而消失。此次进程返回码为 0，但结果 JSON 为 failed；以后验收必须同时检查结构化结果和进程状态。现有脚本的退出处理不能单独作为通过依据。

第一项实现工作是在本项目隔离副本中定位 headless 相机初始化/渲染问题。参考原项目已记录的 `isaaclab.python.kit` experience 配置，加入 RGB、depth、时间戳递增、相机更新与物理步同步检查；保留失败日志。不能直接把旧成功记录升级成本次验收结果。

## 可以怎样复用而不影响其他项目

1. 现有环境仅用于诊断和作为复制来源，后续包安装只进入新 prefix。使用 conda `--clone ... --copy` 或独立安装，避免用共享可写 hardlink 作为隔离手段。
2. 给 Isaac Lab 建立独立、固定 commit 的源码副本，在新环境内重新安装其 editable 包。检查 `.pth`、`direct_url.json` 和实际 import 路径，消除对 `llm as policy` 源码的隐式依赖。
3. Kit 配置、日志、缓存、CUDA/Warp 编译缓存和运行输出均使用本项目路径。既有缓存可复制经过检查的内容；不采用可写链接共享别人的 cache。
4. OpenPI 使用新的 Python 3.11 环境，例如显式 `UV_PROJECT_ENVIRONMENT=<专属路径>`，不替换原 `.venv` 链接。根据本机 GPU 验证锁定依赖；任何为 5090 增加的差异记录在机器运行约束中。
5. ContactWorld 使用自己的 legacy Python 环境。官方安装脚本默认 Python 3.8、NumPy 1.23.3，涉及 Isaac Gym TacSL 特供包、ManiFeel 和代码替换；外层还可能 `pip install --user gdown`。实施前将这些操作限定到独立目录和环境，并固定全部依赖源码版本。
6. 配置/脚本和 manifest 纳入 Git；模型、原始数据、环境和日志放在 Git 外。源仓库、数据与 Isaac/TacSL 组件分别按其许可提供安装入口，不默认把第三方二进制重新打包发布。

建议目录职责如下；除本次文档和清单外，其余均为待实施结构：

```text
/home/sai/zx/openpi-3d-tactile/
  sim/                         # 可分享的入口、配置、审计/转换/eval 工具、文档
    README.md
    sources.initial.json
    reports/2026-09-29/
    scripts/                   # 后续增加 bootstrap / audit / smoke / eval
    configs/                   # 后续增加 USB 和 FORGE 的任务配置
  src/openpi/                  # 后续增加正式 dataset/policy/config 接口

/home/sai/zx/openpi-sim-runtime/ # 建议的独立运行目录，尚未创建
  envs/{isaac,contactworld,openpi,audit}/
  third_party/{IsaacLab,ContactWorld}/
  cache/
  runs/

OPENPI_SIM_DATA_ROOT/           # 数据位置可配置，避免机器绝对路径进入训练配置
  downloads/
  raw/contactworld/insertion_usb/
  derived/
  checkpoints/
```

初期预计独立 Isaac 副本约 20 GiB，OpenPI 环境按目前约 8 GiB 作参考；USB 已知压缩包合计约 8.16 GB。ContactWorld 环境、解压体积、派生数据及渲染缓存尚未测量，因此这不是总空间承诺。下载前检查目标卷，预留解压峰值和系统盘余量，再决定哪些资产放外置盘。

## 第一批获取的资源

上游代码：[PokuangZhou/ContactWorld](https://github.com/PokuangZhou/ContactWorld)，核查版本 `8d0d0edff44f2bb28b58d96e3cf638088589e789`。

官方数据：[Pokuang/ContactWorld](https://huggingface.co/datasets/Pokuang/ContactWorld)，核查 revision `6587189d8e06480e739af5a4004d411c5c07743b`。

| 文件 | 远程标注字节数 | 用途 |
| --- | ---: | --- |
| `releases/insertion_usb_dataset.tar.gz` | 6,585,153,409 | USB 原始数据，约 6.59 GB / 6.13 GiB |
| `releases/insertion_usb_ckpt.tar.gz` | 1,508,068,239 | 官方模型集合，约 1.51 GB / 1.40 GiB；先验收 PC 与 PC+TacFF |
| `assets/industreal.tar` | 61,839,360 | 官方仿真资产 |

固定 revision 的 URL、预期 SHA-256 和字节数见 [sources.initial.json](sources.initial.json)。这些是远程清单信息，文件尚未下载，也未做本地哈希验证。Isaac Gym TacSL 特供包还需单独获取、核对来源和许可；本机有限深度目录搜索未发现现成 ContactWorld/IsaacGym/TacSL 安装，不能据此断言整机绝对不存在。

原报告给出的 201 episodes、13,671 steps、10 Hz、1024×6 XYZRGB 和 10×14×3 TacFF 作为审计预期，不写成已实际读到的 archive schema。官方 loader 注释中有 `state`，也不能证明发布包具备所需的全部本体状态或 taxel 位姿。

## 执行阶段与验收条件

| 阶段 | 要消除的不确定性 | 通过条件及产物 | 失败后的处理 |
| --- | --- | --- | --- |
| P0 隔离平台 | 现有底座在独立环境是否可重建、可渲染 | 专属环境、固定源码、无旧 editable 引用；CUDA、RGB/depth、重力碰撞、reset、时间同步通过；保存环境/版本报告 | 分别归类 dependency、render、physics、asset 问题，先修底座 |
| P1 USB 数据 | 原始数据是否完整、同步，是否真的包含接触/纠偏 | 下载哈希正确；遍历所有 episode；导出 schema、NaN/range、动作统计、接触统计和至少 3 条同步多模态样例 | 缺失字段或异常先记录和定位，不能凭论文补造 state |
| P2 ContactWorld 本机兼容 | 旧 Isaac Gym/TacSL 是否能在 5090 上工作 | 原生 import→GPU physics→camera→TacFF→USB reset/step→官方 PC/PC+TacFF CEM 闭环依次通过 | 若确定是不可修复的 GPU 二进制问题，转兼容主机或本机 FORGE；离线审计可继续 |
| P3 OpenPI 适配 | 新动作与模态是否保持物理语义、确实能训练 | 一个样本及 batch forward/backward；动作 roundtrip 与仿真方向测试；tiny-set overfit；保存/加载 ckpt 和一次推理；在线/离线同一预处理 | 区分坐标/时间/动作/模型问题，再决定是否放大训练 |
| P4 远程训练与回测 | 模型是否在相同实验条件下使用 PC 和触觉 | 首轮 USB 四组消融；记录 commit、split、归一化、seed；回传 checkpoint 后配对初始状态闭环评估和视频 | 先查接触子集、观测利用率和失败类别，不直接追加数据或改任务 |
| P5 FORGE 受控纠偏 | 是否能证明接触后的快速纠偏作用 | clean/recovery/shift 测试集；teacher 可恢复；先 F/T 对照，再真实空间 TacFF；最终做接触前 PC/接触后触觉干预 | teacher 不会恢复则停止扩大采集；无接触信息收益先审任务和学习问题 |

P1 的下载/离线审计可与 P2 兼容性排查交错推进。正式训练的前提是至少有一条经过验收、可用来闭环评估的路径，避免训练结束后才发现本机跑不了 benchmark。

## 适配时必须明确的语义

- OpenPI 现有 XHand 路径是 1972-D 原始 state、18-D proprio/绝对关节动作和 XHand taxel 几何。新增仿真 embodiment/data 接口，复用 spatial encoder，不把 Franka 数据硬塞进 XHand 索引。
- ContactWorld 的动作维度、单位、控制坐标、旋转约定、scaling/clipping、固定夹爪以及 observation/action 时间关系，以发布数据和控制代码交叉确认。报告所述 6-D relative Cartesian 暂作预期。
- 本机 FORGE 原生 `action_space=7`，源码中前 6 维为相对固定物体的目标表示，第 7 维为 success prediction；它不是可直接复用的 ContactWorld 6-D delta action。两个环境可以共享数据接口，但须各自实现动作 adapter。
- taxel grid reshape 不等于空间触觉：必须有传感器布局、局部 taxel 坐标、传感器/机器人位姿和力向量坐标变换，才能生成与 PC 共坐标系的 `xyz_m/force`。缺少几何时单列 sensor-local 表示实验，不能伪称已完成 metric spatial fusion。
- 两个手指的 TacFF 数量、顺序、force 单位和符号均需审计；不能预先认定总共只有 140 taxels，也不能把 gripper 的持续夹持力直接当作 plug-socket 首次接触。
- 缺失 camera slot 使用 mask，不复制图像制造额外视角；缺失 proprio 使用明确的无状态配置/掩码，不能借用 simulator 的隐藏 socket pose 填充。
- 训练与测试按 episode 划分，先划分再取窗口；动作 chunk 不跨 episode；归一化仅使用训练集；RGB/PC/TacFF 必须对应同一观测时刻。
- 原生 FORGE actor observation 已含相对物体状态与 force。teacher 可用 privileged state；四组 π0 student 的输入须独立列白名单，避免 RGB baseline 实际拿到 force 或物体真值。
- 第一轮建议按数据原始控制频率运行；若核实为 10 Hz，先每次执行 1–2 个动作再观测。分别记录模型预测 horizon 与实际开环执行长度，避免长 action chunk 掩盖触觉反应。离线动作 loss 不能代替闭环成功率。

## 实验与训练交接

首轮 USB 为 RGB、RGB+PC、RGB+TacFF、RGB+PC+TacFF。同 episodes/split、初始化、训练预算和评估初始状态，单 development seed 先排查问题，有稳定信号后再多 seed。每组的实际输入 mask 写入运行清单。

官方 ContactWorld checkpoint 是 JEPA world model + CEM planner，不是可直接作为 π0 teacher 的行为策略。当前官方 `eval_planner.py` 的成功判据比较目标帧的 plug 与 EE 位姿，目标有 12/24/36/48 步偏移；该 goal-reaching success 不能直接叫作“完整 USB 插入成功率”。官方结果用于环境校准，π0 的完整任务成功和严格物理插入判据另行实现、验证和报告。

本地交付训练包至少包含：Git commit/patch、环境 lock、dataset revision 与 hash、episode split、schema 版本、动作语义、归一化、配置与命令、tiny-set 检查证据。远程训练结束回传 checkpoint 时同时回传这些元数据，避免模型和预处理版本不一致。

回测至少记录完整任务成功率、首次接触误差、接触后恢复成功率、纠偏延迟（仿真 steps/seconds）以及峰值接触力。推理 wall time 独立报告；暂停仿真等待策略的评估不能称为实时 10 Hz。所有依赖隐藏物体真值的指标只在评估侧计算；原始数据缺字段时不虚构离线恢复指标。

FORGE 后续使用接近插孔后的小幅 residual perturbation，按真实间隙标定难度，先证明 teacher 能恢复再扩大采集。30/50/20 等训练分布只是报告中的候选值，最终由 pilot 的接触率、恢复率和方向覆盖率决定。F/T 对照能检验接触信息的作用，但满足空间触觉硬要求仍需后续 taxel normal/shear 传感与几何验收。

## 当前下一步

第一批实现集中于 P0–P2：建立专属运行目录及环境副本，解决相机复测失败，锁定并下载 USB 文件，生成完整数据审计，同时验证 Isaac Gym/TacSL 的 5090 兼容性。完成这一批后再据事实确定 P3 的具体 schema 和训练配置。预计排障成本最高的部分是旧 Gym 二进制兼容、空间 taxel 位姿恢复、FORGE teacher 与同步控制；下载、哈希、格式转换和报告生成可脚本化。
