# 统一部署交接：d662da0 后的最小集成修复

继续使用 `scripts/serve_policy.py → HandScenePerceptionBridge → run_online_perception.py`
和机器人端 `deploy/ur7e_xhand_deploy_3d_tactile_multi.py`。未增加部署入口，未修改
模型结构、checkpoint、Norm Stats、SAM2/RGBTrack 算法或底层机器人控制。

## Front 标定与输入模式

Peg、Button、Socket 共用现有文件
`configs/ur7e_xhand/front_calibration_candidate.json`，现在也是三个 Final-v2 任务的
默认值。无需寻找/生成其他任务专用标定。保留原有 `OPENPI_1018_FRONT_PROFILE`、
`OPENPI_1028_FRONT_PROFILE`、`OPENPI_1030_FRONT_PROFILE` 显式覆盖；如 shell 留有旧值，
应清除或确认它仍指向该共享文件。文件存在性、几何/release/mesh 哈希、role、512 点、
采样种子及已有标定 provenance 检查均保留。历史 manifest 缺少 provenance 时仍显示
`LEGACY_UNRECORDED`，不伪造或重写训练记录；这不表示需要重新标定。

`d662da0` 的 `policy_config._policy_metadata` 已经从 **created data config** 及模型的
`use_visual/use_tactile` 生成 metadata，而不是直接发布训练配置的原始字典。
此前认为 Final-v2 必须手工指定相机的结论混淆了这两层。客户端应使用
`--spatial-cameras auto`（默认值）：

| 模型 | 实际训练 `camera_roles` / 采集空间深度 | 空间视觉 | 空间触觉 |
|---|---|---|---|
| Pi0/Pi0.5 RGB | 无 spatial / 无 | false | false |
| Final-v2 ST | `('front',)` / Front | true | true |
| Final-v2 Full | `('front',)` / Front | true | true |

三路普通 RGB 图像槽始终保留。其他旧配置若实际训练角色为 front+left，继续按其
配置发布，不能为了统一命令把它们硬改为 Front。

任务文本优先级为：客户端显式 `--prompt/--task` > 服务端 `--default-prompt`（如有）
或模型对应数据集真实训练文本 > 机器人本地数据集 `meta/tasks.jsonl` 单任务文本。
多任务或缺失文本而无法唯一确定时要求显式覆盖，不再静默使用 Peg 文本。

## 初始化状态：Button/Socket 自动，Peg 当前仍需人工选择

Button/Socket 的任务由模型训练配置确定，几何配置必须与模型任务一致。现有 worker
读取各自 `assets/perception/sam2_references/<dataset>/v1` 的 e00000 单参考资产，
执行原有 SAM2 初始化和 RGBTrack。机器人端不再需要 `--scene-position 1`；
桥接内部兼容历史 session key，不把占位号当作场景分类。旧客户端传 1 仍兼容，
其他六场景编号对这两个任务无效。

Peg 的六参考及人工覆盖路径保留：`--scene-position 1..6` 对应 E0/E23/E38/E49/E58/E65。
已有 `qa_peg_six_reference_selection.extract_features/pair_score` 和
`position_initialization.initialize_masks` 的自动选择实现没有被重写。
用户提供的既有离线验证为 70/70、排除参考样本后 64/64；本轮不重新运行。
**真实桥接仍发送人工 `--position-template-episode`，没有完成自动接线。**
缺少的是：把在线首帧交给现有六参考选择路径，并传递经确认的
`max_reference_score/min_reference_margin` 阈值配置与选择结果。
70/70 分类正确率不等于这些线上拒绝阈值已给定。本轮不猜阈值、不重建资产，
因此 Peg 仍要求人工场景号，不宣称已自动化。

## Button Full：云端现有路径启动命令

在现有仓库根目录执行。以下资源路径由云端部署事实提供，使用原有环境和权重：

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile

PYTHONPATH=src:. \
OPENPI_REPO_ROOT="$PWD" \
OPENPI_PERCEPTION_PYTHON=/workspace/mnt/sqzhang26/hoi/envs/sam3d/bin/python \
OPENPI_SAM2_SOURCE=/workspace/mnt/sqzhang26/hoi/repos/sam2 \
OPENPI_SAM2_CHECKPOINT=/workspace/mnt/sqzhang26/hoi/weights/sam2/sam2.1_hiera_small.pt \
OPENPI_RGBTRACK_WEIGHTS="$PWD/checkpoints/rgbtrack" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_policy.py \
  --port 8990 \
  policy:checkpoint \
  --policy.config pi0_xhand_1028_press_button_full_v2 \
  --policy.dir checkpoints/pi0_xhand_1028_press_button_full_v2/final_v2/19999
```

共享 Front profile 已有默认值，无需在命令中另设标定路径。SAM2 source 也兼容原有
`OPENPI_SAM2_ROOT`；若两个变量同时设置，以 `OPENPI_SAM2_SOURCE` 为准。
服务端将这些实际路径传给桥接，再原样传入 worker 的 `--sam2-source`、
`--sam2-checkpoint`、`--weights-root`。保留 perception Python 的虚拟环境入口路径，
不 resolve 到环境外的基础 Python。子进程固定 JAX CPU，Torch 仍使用 GPU。

机器人端的真实服务器 IP 和本机数据目录仍需操作员提供。在机器人已有 LeRobot 环境中：

```bash
# SERVER_IP 和 ROBOT_DATASET_DIR 分别设置为现场推理机 IP、机器人端 Button 数据目录。
sudo -E "$(which python)" deploy/ur7e_xhand_deploy_3d_tactile_multi.py \
  --server-ip "$SERVER_IP" --server-port 8990 \
  --dataset-dir "$ROBOT_DATASET_DIR" \
  --spatial-cameras auto \
  --dry-run --no-home --no-start-hand-align --no-tactile-reset \
  --run-mode single --duration 30 \
  --query-frequency 16 --max-action-chunk-size 16 \
  --record-dir ./diagnostic_runs
```

无需手工 task 文本或 scene-position。机器人电脑的机械臂 IP、相机序列号、
EtherCAT 网卡等已有设备参数若与脚本默认值不同，仍需沿用现场已确认的覆盖值。

## worker、实时性与运动限制

worker 是 **首次收到有效 history 请求时才启动**。WebSocket 开始监听只代表模型与
服务配置可建立，不能作为 SAM2/RGBTrack 已运行的证据。启动实际 worker 时校验
源码/权重路径，打印真实 Python、SAM2 和 RGBTrack 路径；出现
`[PERCEPTION] ... completed geometry frame=...` 且获得同帧策略响应，才说明该帧
真正走过感知链。失败/超时继续上抛，并保留 session `perception.log`。

现有阻塞式 backlog→感知→策略链保留，**尚未证明满足 15Hz 闭环**。
`--fps 15` 只是目标调度频率，`--query-frequency` 也不是实时性保证。

Full 使用现有同步阻塞链：等待感知和策略响应，再按原有控制循环执行动作。
不以 15Hz 或实时性验证作为动作门禁；所有模型均可通过去掉 `--dry-run` 发送动作。
`hand_scene_actuation_allowed=true` 表示协议允许执行，不代表已完成现场验证。
保留原有 `robot.send_action`、关节限制、触觉阈值/重规划、推理后接触复查、
`SafetyTrip` 停止、键盘退出及底层 UR7e/XHand 控制。

`--dry-run` 本身仍可能执行默认回零和起始手姿对齐；无策略动作检查同时使用
`--no-home --no-start-hand-align`。是否归零触觉由已有 `--no-tactile-reset` 控制。
现场仍需确认相机/标定应用、Peg 人工场景、感知首帧效果、实际延迟/积压、
已有接触阈值、硬件急停和停止行为。软件触觉检查不能替代硬件保护。

## 本轮验证范围

一次有效的无动作接口检查已通过：轻量配置/metadata/客户端输入核对，加一个 Button Full
录制首帧通过现有 WebSocket 客户端、真实 worker 与 Full 策略的往返，返回有限的 `[50,18]`。
不连接机器人，不重跑八模型，不重处理训练数据，不扩展 Mock 测试框架。
具体结果记录于 `output/deploy/final_integration_20261102/interface_check.json`。
临时 loopback listener 的第一次搭建遗漏原 `run()` 初始化锁，在进入策略/worker 前中止；
修正检查夹具后才执行上述唯一真实帧，前次记录单独保留，没有掩盖为 worker 成功。
本机使用已有本地 perception 资源进行检查；云端绝对路径的实际 worker 运行仍以
云端首个 history 请求及其感知日志为准，不能把路径传递核对称为远程运行成功。

后续修正：移除了新增的 Full 强制 dry-run 门禁；仅核对输入校验与代码语法，未重跑模型或连接机器人。
