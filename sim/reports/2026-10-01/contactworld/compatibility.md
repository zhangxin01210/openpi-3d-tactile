# ContactWorld：旧环境兼容与在线几何检查

> **同日后续**：小范围兼容排查已经结束，证据见 [compatibility_followup.json](compatibility_followup.json)。社区自编译方案超出此次适配预算，暂缓本机旧栈；后续已完成 Bench2Dex 三条 RGB-D/触觉回放，见[选型结论](../bench2dex/README.md)。下文保留首次兼容测试的实测记录。

2026-10-01。本轮执行用户批准的前两步；偏移恢复采集暂缓。

**结论：前两步尚未全部完成。旧 Isaac Gym 的 GPU 物理和原生 RGB/depth 已通过，但测试的 PyTorch CUDA 栈无法在 RTX 5090 上执行，完整 TacSL 任务因此停在依赖门槛。前相机与点云转换获得了运行时验证，并发现、量化了半像素偏差。**

## 已通过与未通过

| 检查 | 实测结果 |
| --- | --- |
| 独立旧环境 | Python 3.8.20；没有向共享训练/Isaac Sim 环境安装依赖 |
| 原生 GPU PhysX | 120 步落箱测试通过；箱中心 z=0.05199987 m，预期落在平面上 |
| CPU PhysX 对照 | 120 步通过；箱中心 z=0.05199997 m |
| 原生 RGB/depth | 两种物理配置均输出有效图像、深度；这是 CPU 读回接口 |
| 前相机运行时矩阵 | 与固定源码的相机位置、姿态、FOV 推导吻合 |
| 点云坐标转换 | 坐标轴、矩阵转置和平移修正通过原生相机对照；另修正像素中心约定 |
| 发布数据抽样 | USB/Peg 各 30 帧，共 60 帧，经运行时矩阵反投影后，有效点的 RGB 误差为 0 |
| Torch CUDA | **失败**：`torch.arange(9, device='cuda:0')` 即报 `no kernel image is available for execution on the device` |
| gymtorch / GPU 相机张量 / SDF / TacFF | 未进入验证；原生 API 中存在 SDF 接口，不代表其计算已通过 |
| USB/Peg 完整 reset/step | 未运行 |
| 数据状态恢复、动作时序、plug/socket 接触、成功判定 | 在线部分未完成；仍只有此前的离线/源码证据 |
| 腕相机外参 | 未验证 |

机器：RTX 5090，sm_120，驱动 580.173.02。安装包只有 `gym_36.so`、`gym_37.so`、`gym_38.so`，其 setup 声明 Python `>=3.6,<3.9`。测试的 Python 3.8 环境安装了 `torch 2.4.1+cu121`，运行时列出的支持架构到 sm_90，实际 CUDA 算子失败。故障发生在任务导入之前，不能据此宣称 Isaac Gym 原生物理不支持本机，也不能宣称已经验证完整任务。

这里证明的是**这一套已测试的旧依赖组合被阻塞**，没有穷举其他 wheel、重编译或跨 Python 进程方案。根据此前约定，本轮在明确的兼容障碍处停止，没有展开长期移植。CPU PhysX 能运行也不能替代原任务依赖的 GPU 相机张量和触觉 SDF 验证。

## 前相机与点云：新确认的细节

直接调用官方包的 `get_camera_view_matrix` / `get_camera_proj_matrix`，使用源码的 256×256、FOV 75°、位置 `[0.68,0,0.15]` 和姿态：

- 实际 fx=fy=166.81285095 px；此前推导值 166.81284772 px。
- 逆视图旋转与推导值最大差约 1.56e-8；平移最大差约 7.15e-9 m。
- 此前坐标轴/转置/平移校正与运行时的整数像素反投影最大差约 6.61e-7 m。
- **原代码用 `(u,v)`，实际深度采样对应 `(u+0.5,v+0.5)`。** 仅修坐标系仍保留半像素几何偏差。

使用已知箱体和平面、限制深度 <1 m 的 49,920 个点做距离检查：整数像素反投影的表面误差 P99 为 2.433 mm，使用像素中心后降为约 0.00128 mm。此数字是简单已知几何的实现一致性检查，不是整个数据集的标定精度承诺。原先未考虑像素中心的失败结果也保留在 `compatibility/front_camera/`。

将运行时矩阵及像素中心修正用于发布数据的 60 帧：

| 指标 | USB | Peg |
| --- | ---: | ---: |
| 有效点 RGB MAE 最大值 | 0 | 0 |
| 投影回原数组索引的最大误差 | 9.60e-5 px | 9.05e-5 px |
| 加入半像素修正带来的点位移中位数 | 0.801 mm | 0.876 mm |
| 点位移 P99 | 3.239 mm | 3.327 mm |

可复用的 `world = A @ stored_xyz + t` 矩阵、内参、抽样行号均保存在 [runtime_geometry.json](runtime_geometry.json)。适用范围是**固定前相机、当前源码的错误转换方式、基座与世界坐标重合**；不能直接套到腕相机或移动基座。精确采集时配置未随数据保存，完整状态重放仍未完成。原始数据和视频未改写，正式训练适配器尚未接入该转换。

## 下载、环境和证据

- 用户明确允许代理下载的 Isaac Gym/TacSL 包：268,908,854 bytes，下载完成、tar 可完整解压。本地 SHA-256 为 `532517b65aaa98141f8fed501c703990120b2f4c85315e092ace1c92dd2e7edf`；没有发布者校验和可核对。见 [下载收据](compatibility/isaacgym_package_receipt.json)。
- Conda 基础环境、Torch/CUDA 等大依赖均显式清除代理变量后从清华镜像直连。IsaacGymEnvs 使用稀疏代码检出，未下载其完整资产目录。
- 环境：`/home/sai/zx/openpi-sim-runtime/envs/contactworld-legacy`。已安装版本见 [依赖快照](../../../contactworld-legacy-requirements.txt)，该文件是失败组合的记录，不是 5090 推荐配置。
- 原生包：`/home/sai/zx/openpi-sim-runtime/third_party/IsaacGym_Preview_TacSL_Package/isaacgym`。
- 源码：ContactWorld `8d0d0edff44f2bb28b58d96e3cf638088589e789`；IsaacGymEnvs `1a61da683cf1485f3307684c740771ea5e842b39`。尚未安装完整任务依赖或应用任务补丁。
- [汇总 JSON](compatibility.json)、[GPU 原生探针](compatibility/native_gpu/probe.json)、[相机几何探针](compatibility/front_camera_centers/probe.json)、[Torch 故障日志](compatibility/torch/run.log)。
- [前相机测试图](/home/sai/zx/openpi-sim-runtime/runs/contactworld_compat/front_camera_centers/rgb.png)；原始矩阵、深度和点云在同目录的 `camera.npz` / `geometry.npz`。

所有测试进程已结束，GPU 显存回到约 16 MiB。没有偏移恢复采集、训练或数据覆盖。

## 复现

从仓库根目录运行。前两个命令使用旧 Python，第三个使用已有离线审查环境。

```bash
CW_RUNTIME=/home/sai/zx/openpi-sim-runtime
CW_LEGACY_PY="$CW_RUNTIME/envs/contactworld-legacy/bin/python"
export PYTHONPATH="$CW_RUNTIME/third_party/IsaacGym_Preview_TacSL_Package/isaacgym/python"
export LD_LIBRARY_PATH="$CW_RUNTIME/envs/contactworld-legacy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -c 0

timeout 90s "$CW_LEGACY_PY" sim/scripts/probe_contactworld_native.py \
  --physics gpu --contactworld-camera \
  --output "$CW_RUNTIME/runs/contactworld_compat/recheck_front"

# 本机已知会失败；换兼容机器后先过此门槛。
MAX_JOBS=2 timeout 180s "$CW_LEGACY_PY" sim/scripts/probe_contactworld_torch.py \
  --output "$CW_RUNTIME/runs/contactworld_compat/recheck_torch/probe.json"

"$CW_RUNTIME/envs/contactworld-audit/bin/python" \
  sim/scripts/check_contactworld_runtime_geometry.py \
  --camera "$CW_RUNTIME/runs/contactworld_compat/front_camera_centers/camera.npz" \
  --data "$CW_RUNTIME/data/contactworld" \
  --output "$CW_RUNTIME/runs/contactworld_compat/recheck_geometry.json"
```

## 下一步的边界

优先在能够运行这套旧 CUDA/PyTorch 依赖的 GPU 环境上先跑两个探针，通过后继续单环境 USB/Peg 的 TacFF、reset/step 及同状态数据对齐。不能把换机器视作已经验证成功。若必须留在当前 5090，下一步需要明确评估新版 Python 绑定、源码构建或进程桥接等额外工程；它们超出本轮最小兼容检查，也没有现成可用的结论。

恢复在线工作后，仍需固定 HF 发布资产，核对腕相机、实际控制周期、动作缩放和裁剪、状态恢复误差及 plug/socket 真接触。完成这些之前，暂不开展偏移恢复采集或正式训练。
