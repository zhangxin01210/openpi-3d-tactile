# ContactWorld 5090 自编译：执行记录

开始：2026-10-01 15:24（Asia/Shanghai）。用户授权开始，首轮约 4 小时为进度检查点；若卡点仍有明确进展，可以适当延长，并汇报依据。Bench2Dex 暂留备用。

## 关键节点与当前状态

| 节点 | 原计划 | 当前实测 |
| --- | --- | --- |
| A 独立环境、源码、补丁 | 累计 20–45 分钟 | 约 16 分钟完成 |
| B Torch 编译与 CUDA | 累计 1–3 小时 | 15:56 打包完成，约 15:59 CUDA/cuDNN 验收通过 |
| C gymtorch / GPU 张量 | B 后 15–40 分钟 | 约 15:59 通过，含 GPU 物理和相机张量 |
| D 单 USB reset/step、相机和 TacFF | C 后 30–90 分钟 | 16:00 前通过 reset_idx、8 步零动作和传感器输出 |

这些时间是未实测阶段的规划范围，不是成功保证。运行中状态以 `/home/sai/zx/openpi-sim-runtime/runs/contactworld_build_5090/status.json` 和日志为准。

## 隔离范围

- 新环境：`/home/sai/zx/openpi-sim-runtime/envs/contactworld-build-sm120`，由旧环境 `conda --clone --copy` 得到。Torch 文件 inode 不同已核对。
- 源码：`third_party/pytorch-cw-sm120`，Torch v2.4.1，提交 `ee1b6804381c57161c477caa380a840a84167676`。
- 任务：`third_party/ContactWorld-sm120`，独立复制 ContactWorld、ManiFeel 和 IsaacGymEnvs，再按官方替换脚本应用补丁。原工作目录保留。
- 准备过私有 cuDNN 9.7，但实际编译头文件优先读到系统 CUDA 目录中的 9.21.1；运行探针暴露版本不一致后，启动脚本改为加载已有系统 cuDNN 9.21.1。没有修改系统库。后续构建入口也已统一到该版本。
- 原环境 pip 快照：运行目录的 `legacy_before.txt`、`isaac51_before.txt`。
- 下载均直连；没有将之前单个安装包的代理授权扩展到本轮。

## 已解决的准备问题

1. Conda 离线缓存不足：改为独立复制既有旧环境，没有修改原环境。
2. Git 子模块传输停滞：改用固定 gitlink 提交的官方 codeload 压缩包；嵌套 gitlink 从发布者 API 解析，保存提交号和 SHA256。Eigen 的 GitLab archive 返回 403，但直接获取同一提交成功。Gloo 的 `.gitmodules` 有失效条目，按该提交 Git tree 的实际 gitlink 处理。
3. 首轮 CMake 自动选中系统 cuDNN / GCC 13：在早期编译阶段停止，固定 cuDNN ROOT 与 GCC 11 CUDA 主机编译器后重新配置。保留已完成的对象文件复用，原配置与日志也保留。

构建设置：仅 sm120，24 个编译 job，关闭测试、分布式、FlashAttention、mem-efficient attention、MKLDNN 和部分非必要 CPU 优化组件。该构建先用于仿真兼容验证，不能等同完整训练环境的性能配置。

## 脚本与证据

- [最小架构补丁](../../../scripts/patch_contactworld_torch_sm120.py)
- [构建入口](../../../scripts/build_contactworld_torch_sm120.sh)
- [固定子模块下载](../../../scripts/fetch_torch_submodule_archives.py)
- [Torch/Gym GPU 物理与相机张量探针](../../../scripts/probe_contactworld_gpu_interop.py)
- [单 USB 探针](../../../scripts/probe_contactworld_usb.py)：只加载 USB 类，跳过无关任务注册；不修改任务物理实现，不训练或采集偏移恢复轨迹。
- [4090 服务器只读检查](../../../scripts/inspect_contactworld_host.py)：`python3 inspect_contactworld_host.py`，不安装依赖、不要求 root。缺少 vulkaninfo 本身不代表 Vulkan 不可用，仍需相机实测。

日志：`/home/sai/zx/openpi-sim-runtime/runs/contactworld_build_5090/`；主要文件 `torch_build.log`、`torch_sm120.patch`、`source_archives_final.log`。子模块清单：`downloads/contactworld/torch_archives/manifest.json`。

## 与 4090 的比较

用户已有服务器，但驱动和权限尚不确定。若 Linux 驱动、用户级环境安装、网络和无头图形渲染均可用，尝试现成旧 Torch wheel 通常更快，也减少自编译维护；这是一项有条件的工程判断，尚未在该服务器实测。PyTorch 官方提供 [2.4.1 等旧版安装入口](https://pytorch.org/get-started/previous-versions/)，但相机/TacSL 仍需验收。若服务器缺少图形驱动、容器 GPU 图形权限或安装权限，等待管理员的时间可能超过本机编译。

完整任务通过前，不宣称环境已经可训练或可采集。

## 16:00 验收结果与停止节点

从 15:24 准备到单 USB 探针通过，约 **36 分钟**。Torch 编译/打包约 16 分钟（含一次早期重配），实际明显短于预估。到达约定关键节点后停止，不启动训练或偏移采集。

- `gpu_interop_final/probe.json`：Torch 2.4.1+cu128sm120、CUDA 12.8、cuDNN 9.21.1、sm120；CUDA 基础算子、逆矩阵和卷积通过。gymtorch 编译导入成功，GPU PhysX 方块落地 z=0.052 m，RGB/depth GPU 张量有效。
- `usb_probe_calib/probe.json`：官方 USB 类初始化、reset_idx、8 步零动作成功，探针用时约 5.5 秒。front/wrist 各 256×256 RGB；pointcloud 为 1024×6；相机 view/projection 矩阵和原生深度保存于 `sensors.npz`。
- `tactile_force_field_right` 有限且非零，8 步最大绝对值约 0.00108–0.00109（保持源码原值，尚未独立核定力单位/标定）。这仅证明输出链路可执行，不证明左右接触辨识或恢复能力。
- 补齐稀疏检出遗漏的官方 `gelsight_r15_data/bg.jpg` 和 `polycalib.npz`，约 2.3 MB，直连获取固定 HEAD 对应文件。
- 最小 Torch 构建没有 CPU LAPACK。GPU 逆矩阵使用 NumPy CPU 结果做独立参照后通过；未伪装成完整 CPU 线代支持。正式训练前仍需单独核定依赖和构建功能。
- 原 `contactworld-legacy` 和 `univtac-isaac51` 的前后 pip freeze 均逐字一致；仅新环境安装了自编译 Torch/torchvision。Gym 自动写入用户 SDF 缓存，未更改原环境包。
- wheel 路径、大小与 SHA256 见运行目录 `wheel_receipts.json`；早期失败日志保留，成功证据以上述 final/calib 探针为准。

后续应验证一条完整官方示范的动作重放与多模态时序/几何对齐，并补充持续运行检查。当前 smoke 未量化插入成功率，未验证数据集与在线 TacFF 数值一致性，未完成左右偏移恢复验收；不应直接开始正式训练。

## 后续修正：USB 资产来源

重放审查发现此前隔离 USB 任务加载的 mesh 与数据发布包不一致。核心 Torch/Gym 兼容测试仍成立；USB smoke 仅证明旧资产任务能运行，不能证明数据任务已复现。已更正隔离资产，初始夹持与轨迹显著改善，但完整插入仍未通过。见 [重放审查](replay_review.md)。
