# 限时选型复核：ContactWorld → Bench2Dex

> **最新修订**：用户确认“初始物体随机化即可满足视觉用途”，并接受 ContactWorld 自编译成本。下面的路线决定保留作历史；当前以[重新审查结论](reconsideration.md)为准：Bench2Dex 恢复为近期主候选，任务 73 的初态变化已量化，官方成功率较低，值得继续；TacEx 回到备用。ContactWorld 见[自编译工期估计](../contactworld/source_build_plan.md)。

2026-10-01。范围：ContactWorld 小范围兼容性排查；遇到需要维护自编译框架的路线后，转向 Bench2Dex 的一个任务、三条示范。到此停止，没有启动偏移恢复采集或训练。

## 路线结论

| 路线 | 本轮证据 | 当前决定 |
| --- | --- | --- |
| ContactWorld USB/Peg | 旧 Torch 在 5090 上的 CUDA 算子失败；找到的社区成功方案需要自编译 Torch/torchvision，尚未复现 | 暂停本机旧栈适配，保留已下载数据。兼容机器或愿意承担自编译维护时再恢复 |
| Bench2Dex | 当前 5090 完成三条双手拼图示范的 RGB、depth、相机参数和 TacMap 回放 | 环境可用，作为灵巧手备选；选中的任务 73 不直接进入正式训练 |
| 现有 TacEx | 此前已验证本机运行和传感器通路；任务设计、合格示范仍待完成 | 在限定工程预算下，建议下一阶段主线采用自定义随机化接触任务；先设计验收协议，再决定采集 |

Bench2Dex 任务 73 的装配目标 XY 固定，未满足用户的目标位置变化要求。这里只审查了一个任务的三条数据，不能据此断言整个 benchmark 不适合。仍未证明视觉、点云、触觉各自对成功率有必要贡献。TacEx 自定义协议需要自己采集示范，不能直接套用官方任务成绩。

## ContactWorld 排查边界

原生 GPU PhysX / RGB-D 曾通过，但 Python 3.8 + Torch 2.4.1+cu121 的 CUDA 运算失败；已安装 `libtorch_cuda.so` 没有可列出的 PTX，包内只有 gymtorch 桥接源码，没有 `gym_38` 绑定源码。同一台机器上现代 Torch 2.7.0+cu128 的 CUDA 运算正常。

- [社区 5090 成功报告](https://github.com/escontra/gauss_gym/issues/13#issuecomment-4914476549)使用 Python 3.8、CUDA 12.8、sm120 自编译 Torch 2.4.1，并重建 torchvision 等。本轮没有复现，不能当成本项目的已验证安装配方；它超出约定的小适配范围。
- [ManiFeel Isaac Lab PR 16](https://github.com/purdue-mars/manifeel/pull/16)已关闭、未合并，实际测试任务为 Lift-Cube，不能代替 USB/Peg/TacSL 迁移。
- 结论是暂缓这一依赖路线，**并非证明 5090 永远无法运行 ContactWorld**。详见[排查证据](../contactworld/compatibility_followup.json)与[原始兼容报告](../contactworld/compatibility.md)。

## Bench2Dex 实际跑通的内容

任务 `73_jigsaw_puzzle_assembly`，KUKA 双臂 + Sharpa 双手。从 50 条不同 SHA 的轨迹中均匀选取 0、24、49；目录虽有 100 个文件，`_1` 文件与对应原文件 SHA 相同。

| 示范 | 全部帧数 | 回位前帧数 | 回位前有效且有限的动作帧 | 当前源码终态判据 |
| --- | ---: | ---: | ---: | --- |
| 000000 | 815 | 783 | 782 | 通过 |
| 000024 | 841 | 812 | 811 | 通过 |
| 000049 | 796 | 767 | 766 | 通过 |
| 总计 | 2452 | 2362 | 2359 | 仅这三条样本 |

全部 58 个关节匹配。每条均输出胸前和右腕 RGB、深度、相机参数及逐帧外参，以及双手共 10 个指尖的 TacMap、法线方向距离和 contact mask。全序列检查显示每个距离通道均为有限数，且各自存在非零帧。

**这是依据记录关节/物体状态进行的运动学回放。** 回放过程会刷新场景以查询触觉，但没有验证动作闭环控制、接触动力学恢复或学习策略成功率。不能把三条终态通过写成 benchmark 的 100% 成功率。

### 环境处理

- 复用 `univtac-isaac51`：Isaac Sim 5.1、Torch 2.7.0+cu128、RTX 5090。
- 已安装 Isaac Lab 2.3.0 缺少所需 multi-mesh ray caster 数据模块，首次导入失败。
- 另下载 Isaac Lab 2.3.2 源码，仅通过每个回放进程的 `PYTHONPATH` 选择它；未替换环境内安装，也未修改 Bench2Dex 源码。
- 三次完整回放退出码均为 0；日志保留了上游少量 visual prim 引用警告，未完成所有机器人视觉网格的正确性检查。
- 本轮源码、资产、样本全部直连下载，无代理。48 个选定资产共 62,385,361 bytes；选中三条原始轨迹合计 11,702,428 bytes。初始下载过的两份额外样本保留，未纳入三条审查统计。

固定版本与哈希见 [sources.json](sources.json)，资产列表见 [asset_manifest.json](asset_manifest.json)。来源为 [Bench2Dex 官方源码](https://github.com/Bench2Dex/Bench2Dex)、[官方遥操作数据](https://modelscope.cn/datasets/Bench2Dex/teleopdata)、[官方资产](https://modelscope.cn/datasets/Bench2Dex/Bench2Dex)。

### 深度与点云

开启官方回放的 `--enable-depth` 后，两台相机均实际产生深度，而非仅存在配置字段。胸前针孔相机使用运行时 K、逐帧外参反投影，坐标轴为相机体坐标 `(+X forward, +Y left, +Z up)`，由光学坐标转换为 `[z,-x,-y]`，像素取中心 `(u+0.5,v+0.5)`。

每条的首、中、尾帧都导出了彩色世界点云。筛选桌面附近点后，与已知桌面高度的绝对残差中位数约为 8.1e-8～1.7e-7 m。这只说明简单平面的几何实现一致；筛选窗口为 15 mm，不能解释为整个场景、整个相机模型达到该标定精度。三条共九帧检查见 [geometry_checks.json](geometry_checks.json)。

右腕为鱼眼模型，图像有效区域外存在 mask。已保存其深度、模型参数与外参，但未独立验证鱼眼三维重建，不能套用胸前针孔公式。

## 数据和任务的实质问题

1. **装配目标 XY 固定。** 任务 YAML 的两块白色锚点设置 `freeze_pose: true`，成功判据写死其余四块的世界 XY 目标。三条样本锚点 XY 范围均为 0，Z 范围约 1.46 mm。待抓物初态、相机、桌高变化不等于装配目的地变化。任务 86 源码也有类似冻结锚点，但未回放它的数据。
2. **触觉是几何压入代理。** TacMap 和法线方向距离来自 ray casting，不是法向力＋双切向力。可以研究空间接触图，但不能宣称有三维力测量，也未验证触觉能区分左右偏位的恢复决策。
3. **元数据有矛盾。** 三条均写 `collection_mode=scripted_hold`、`demo_eligible=false`、`success=false`，但 action/source 主要为 teleop，且可见操作与装配。instruction 还描述八块 Tetris 和拆开，实际任务为六块拼图。不能直接用这些字段作为训练语言或成败标签。
4. **当前源码重算只证明较宽松终态。** 使用原记录的物体速度，三条回位前最后一帧都通过。判据为 XY 20 mm、相对高度 10 mm、速度阈值，无姿态检查，不能据此标成精密三维插入成功。通过帧数分别为 33、14、22，最长连续通过帧数 18、10、20。
5. **动作和回位需要筛选。** 每条首行动作为 NaN，已标 `action_valid=false`；没有发现 NaN 被标为有效。视频对回位尾段作了标记。2359 只是过滤后的候选动作帧数，尚未验证动作时序、语义和闭环执行，不能视为训练适配已完成。

原始数据没有改写；详见 [raw_sample_audit.json](raw_sample_audit.json) 和 [replay_audit.json](replay_audit.json)。

## 可视化与复跑

[本机三条同步视频浏览页](/home/sai/zx/openpi-sim-runtime/visualizations/bench2dex_review/index.html)：胸前 RGB、腕部 RGB、胸前深度、10 指 TacMap，另附分镜、PLY 和几何检查。视频为 20 fps、1280×960、AV1/yuv420p MP4；三条均完成 FFmpeg 全片解码，已查看分镜。

单条回放命令（在固定版本的 Bench2Dex 源码目录执行，换编号可回放其余两条；输出指向新目录以保留本轮证据）：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH=/home/sai/zx/openpi-sim-runtime/third_party/IsaacLab-2.3.2/source/isaaclab \
/home/sai/miniconda3/envs/univtac-isaac51/bin/python replay.py \
  --headless --log-level INFO \
  --hdf5 /home/sai/zx/openpi-sim-runtime/data/bench2dex_review/dataset/73_jigsaw_puzzle_assembly/origin-generalization/episode_000000.hdf5 \
  --enable-rgb --enable-depth --enable-tactile --restore-generalization \
  --cameras cam_chest cam_wrist_right \
  --output /home/sai/zx/openpi-sim-runtime/runs/bench2dex_review_repeat/episode_000000/replay.hdf5
```

仓库内新增三个脚本：[按 SHA 去重抽样下载](../../../scripts/download_bench2dex_review.py)、[原始/回放审计](../../../scripts/audit_bench2dex_review.py)、[同步视频与点云导出](../../../scripts/export_bench2dex_review.py)。脚本的 `--help` 给出参数。本轮日志及完整回放 HDF5 位于 `/home/sai/zx/openpi-sim-runtime/runs/bench2dex_review/episode_*/`。

## 继续或更换的条件

- ContactWorld：有兼容 GPU 环境，或明确接受维护自编译依赖后，再验证完整 TacSL/reset/step 与在线对齐。当前停止重复安装尝试。
- Bench2Dex：找到目标位姿确实随机、接触调整有证据的现成任务，或接受修改协议并重采示范，才进入训练准备。当前只保留已打通的回放底座；不因一个任务而否定全部任务，也不扩大到全量下载。
- TacEx：若坚持当前电脑、限定适配预算、明确要求目标随机化，优先设计自定义插接任务：定位过程完整、目标九宫格/姿态变化、几何变化可控，再审查少量合格示范。新协议下的恢复采集仍按用户要求暂缓。

下一节点应是“选定任务协议并确认合格示范来源”，而不是继续扩充安装数量。正式训练、关节状态/时间基线、多模态消融、闭环成功率测试均未开始。
