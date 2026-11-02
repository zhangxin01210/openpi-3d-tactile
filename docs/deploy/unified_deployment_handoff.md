# π₀ 在线部署交接（离线开发结果）

GPU 服务端仍用原生命令：

```bash
OPENPI_REPO_ROOT="$PWD" OPENPI_PERCEPTION_PYTHON=/path/to/perception-venv/bin/python \
  OPENPI_1028_FRONT_PROFILE=/path/to/training-front-profile.json \
  python scripts/serve_policy.py --port 8990 policy:checkpoint \
  --policy.config pi0_xhand_1028_press_button_full_v2 --policy.dir /path/to/checkpoint/step
```

这只是训练配置的示例，不能证明对应权重存在。`OPENPI_PERCEPTION_PYTHON` 指向已有 SAM2/RGBTrack 环境；训练配置用已有的 `OPENPI_1028_FRONT_PROFILE` 等变量确定前相机配置。RGB/ST 模型仍走原服务流程，无需感知 worker。机器人端使用原脚本：Button/Socket 的 full 模型沿现有 history 协议使用已有 `--scene-position 1`；Peg 旧路径继续使用已有六位置选择。客户端 `--dry-run` 仍可能回零和手部对齐，测试不能运行机器人客户端主程序。

## 代码路径与训练契约

对于 final-v2 full 模型，`serve_policy.py` 通过训练配置的 dataset root 查找训练 Object Geometry v1 manifest 和 `output/preprocess/<dataset>/geometry/approved/manifest.json`、`output/preprocess/<dataset>/rgbtrack/release_v1.json`；校验 mesh、release 和几何 manifest 哈希、role、采样种子、512 点及 base_link，复用训练端 calibration 检查。历史记录未包含标定时沿训练端记录为 LEGACY_UNRECORDED，**不代表标定一致**。缺少资产或不一致时服务拒绝启动。根据几何 manifest 标记的旧 q2 迁移来源，Peg 使用原有 Peg 感知及六位置路径。

已有 `HandScenePerceptionBridge` 在客户端每次推理请求前发布因果帧、等待现有 `run_online_perception.py` 的同帧几何并调用原策略。新任务 v1 路径读取已审核 geometry 与训练 RGBTrack release；复用 SAM2 单参考、SAM2 video-memory、`rgbtrack_adapter`、训练静态、动态初始化与运动保护；按 release 的 dynamic/static 角色映射至 manipulated/target，复用训练 exporter 的 metric mesh 采样与数组转换（512 点、invalid mask/age/source）。RGB-D 使用训练 adapter 的注册函数并检查 K 和 base 外参；在线 spatial、触觉和 18 点 FK 使用项目公共实现。该路径只支持训练 exporter 当前的 depth/static_upright_convex_mask_v1 策略，其他策略明确拒绝。

训练仓库仅定义十个候选配置，未能确认其中八个 checkpoint 的实际对应关系。原生 loader 按选择的配置从 checkpoint 读取其 asset ID 的 Norm Stats；仍须核实每个 checkpoint 和配置确属同一次训练。

## 缺少的实际文件与验证

此开发机只有 `assets/perception/sam2_references/<task>/v1` 和 Peg metric GLB。Button/Socket 缺少以下训练产物：

- `output/preprocess/1028_press_button/geometry/approved/{manifest.json,button.glb,box.glb}` 和 `output/preprocess/1028_press_button/rgbtrack/release_v1.json`；
- `output/preprocess/1030_yellow_socket/geometry/approved/{manifest.json,red.glb,yellow.glb}` 和 `output/preprocess/1030_yellow_socket/rgbtrack/release_v1.json`；
- 三个任务对应的 `data/<task>/object_geometry/v1/manifest.json`、真实训练相机 profile、checkpoint 及其 norm stats，以及 SAM2 和 RGBTrack 权重/环境。若 Peg sidecar 声明 q2 迁移来源，需保留旧配置、metric scale 文件和匹配的旧运行环境。

新增 CPU 测试以真实 worker 函数、训练几何转换、已有桥接、实际 WebSocket 服务器及机器人客户端类、真实 spatial/tactile/FK 组装完成 Button/Socket 观测到模拟 action 的往返。Mock 为 SAM2 图像预测与 video stream、RGBTrack 与静态 hook、RGB-D 注册、mesh 采样及模型推理；没有启动机器人。旧 Peg 桥接与单元测试继续运行。

**尚未达到可以实机运行的状态。**可在部署电脑准备以上资产并进行离线验收：先确认八个 checkpoint 配置及 Norm Stats，验证依赖和训练资产哈希；用真实 episode 比对 mask、RGB-D 注册、base_link 的 512 点和有效性/age、600 点触觉、18 点 FK、模型输入与 action；再做不发送动作的相机和服务检查。真实权重加载、SAM2、RGBTrack、数据逐帧一致性、实际帧率和硬件均为 NOT TESTED。旧 Peg 路径和新任务路径都须现场验证，尤其是透明物体、低置信度拒绝和静态目标冻结；现场人员完成急停、限速限力、工作空间与回零检查后才可考虑动作。
