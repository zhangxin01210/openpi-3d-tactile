# ContactWorld USB 新点云：五条示范的视觉验收节点

**状态：仅完成采集与审查，等待用户看图确认；尚未转换完整训练集，尚未训练。**

更新：原 1024 点窄 ROI 缺失桌面和可见机械臂区域；4096 点宽 ROI 和触觉坐标审计见 [contactworld_base_cloud_tactile_audit.md](contactworld_base_cloud_tactile_audit.md)。本页窄 ROI 不再是推荐训练候选。

## 观看入口

- [五条完整对照视频、关键帧、可旋转 3D 点云](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_dense_dual_pilot_20261002/index.html)
- [前视裁剪方案的独立页面](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_dense_pilot_20261002/index.html)
- [双视角逐条数值汇总](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_dense_dual_pilot_20261002/summary.json)

主页面每条的四栏视频依次为发布 RGB、同帧重放 RGB、旧整场点云在 RGB 上的位置、前视裁剪后点云的位置。第二段视频显示腕视 RGB 与分配给腕视的 512 点。关键帧包含深度和三维图，首/中/末帧可旋转。三维图提供按高度着色和仅供诊断的 RGB 着色；**模型候选输入仍只有 XYZ**。

## 数据和协议

重放固定的 USB 发布示范 1、4、67、101、132，共 382 个动作前帧。每帧保存前视/腕视 RGB、两路原生 `256×256` 深度、逐帧相机矩阵、18 维关节状态、TacFF、插件/孔座位姿（仅诊断）、发布动作和反馈重放**实际执行动作**。输出在独立目录 `runs/contactworld_dense_dual_pilot_20261002/`，未覆盖原始发布数据、原 LeRobot 数据或旧 checkpoint。

固定基座坐标裁剪范围：`x∈[0.25,0.62] m, y∈[-0.20,0.20] m, z∈[0.01,0.30] m`。它在运行时不读取插件/孔座真值。两种候选均固定 1024 点：

1. **前视裁剪**：先从前视全分辨率深度重建几何，再在该范围内无放回抽 1024 点。
2. **双视角合并**：前视抽 512 点，腕视抽 512 点；腕视使用该帧动态 view/projection 矩阵变换到同一基座坐标，并打乱点顺序。相机来源 ID 和 RGB 只作诊断，不进入目前设想的模型输入。

采样随机数由示范编号和帧号固定，重复采集可复现。若决定正式训练，还需将同一协议落实到在线推理。

## 验证结果与限制

| 项目 | 五条示范结果 |
| --- | --- |
| 完整重放 | 5/5 终帧通过环境原判据；这不是策略成功率 |
| 前视裁剪候选点数 | 全部 382 帧不少于 1542 点 |
| 腕视裁剪候选点数 | 全部 382 帧不少于 18257 点 |
| 固定裁剪覆盖 | 在这五条中，前视红色像素对应的有效深度点覆盖率最低 95.5%；蓝色最低 100%；颜色阈值不等于物体分割真值 |
| 原点云 / 前视裁剪 / 双视角合并 | 红色像素对应点数中位数：27 / 153 / 97；蓝色：48 / 311 / 230.5；白色：841 / 1 / 15 |
| 相机与深度回查 | 382 帧采样点回投影最大误差约 0.00026 像素，深度最大误差 `7.5e-8 m`；三帧抽查的前视/腕视共享桌面高度中位数最大差 `1.5e-6 m` |
| 发布/重放前视 RGB MAE | 中位数 `5.9e-5`、P95 `0.00835`、最大 `0.0251`（0–1 范围） |
| 发布/重放插件位置差 | 中位数 `0.034 mm`、P95 `2.45 mm`、最大 `5.88 mm` |

前视裁剪在某些初帧仍看不到被夹爪遮挡的插件。例如示范 1 首帧，旧点云、前视裁剪、双视角合并分别有 0、0、137 个点投到蓝色像素。双视角能补上这部分可见几何，但也增加了腕视相机和跨视角融合的工程范围。请先看实际 3D 图再确定采用哪版。

**训练数据不能只把新点云替换进旧 LeRobot。**深度来自重新执行的轨迹，重放 RGB、触觉、关节状态和动作与发布包有小幅差异；示范 4 的单帧最大插件位置差约 5.88 mm，反馈动作与原动作的六维 L2 差最大 0.443。新点云须与同一重放帧的其他观测及实际执行动作共同形成新数据集，并保留来源示范 ID 与 train/val/test 轨迹划分。因此，若进入训练，RGB、RGB+TacFF、RGB+PC、RGB+PC+TacFF 四个首批模型都应在同一新数据集上重训，不能拿旧 `01/03` 直接对比新 `02/04`。

本次只覆盖五条已知正例。发布训练轨迹的 7881 帧中，插件和孔座位姿中心都在固定裁剪范围内；这不是对其他轨迹的可见性、点数或教师动作质量的完整验收。用户确认视觉效果后，下一步才是完整数据重采和逐轨迹审计。

## 复现

从仓库根目录执行；输出目录应使用新路径，避免覆盖现有审查结果。

```bash
python3 sim/scripts/run_contactworld_replay_batch.py \
  --source /home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-sm120 \
  --input /home/sai/zx/openpi-sim-runtime/runs/contactworld_multireplay_20261001/input \
  --output /home/sai/zx/openpi-sim-runtime/runs/contactworld_dense_dual_pilot_20261002 \
  --seed 0 --white-gel --white-mounts --usb-only --capture-dense

.venv/bin/python sim/scripts/render_contactworld_dense_pilot.py \
  --runs /home/sai/zx/openpi-sim-runtime/runs/contactworld_dense_dual_pilot_20261002 \
  --output /home/sai/zx/openpi-sim-runtime/visualizations/contactworld_dense_dual_pilot_20261002
```
