# ContactWorld USB：4096 点 base 点云与触觉坐标审计

状态：**五条示范的视觉候选，等待人工确认；没有转换完整训练集，也没有用这些候选训练模型。**

## 看图

- [五条 USB 的同帧 RGB／旧 1024／新 4096 视频、关键帧和可旋转点云](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_base_cloud_4096_20261002/index.html)
- [同帧点云和真实仿真触觉采样点位置、base 力向量](/home/sai/zx/openpi-sim-runtime/visualizations/contactworld_tactile_base_probe_20261002/index.html)

新点云从先前保存的前视和腕视 `256×256` 全分辨率深度重建，取 base ROI `x∈[-0.20,0.65] m, y∈[-0.35,0.35] m, z∈[-0.005,0.35] m`，再做 5 mm voxel representative 和 4096 点采样。ROI 不读插件或插座真值。它保留相机看得到的机械臂、桌面、USB 与插座，并裁掉远处棋盘背景；两个相机都看不到的机械臂表面仍不可能通过深度重建。三种候选都只用真实深度点：

1. `uniform`：所有 ROI voxel 按 Morton stride 采 4096 点，与现有 `press_0828_17` 的 5 mm voxel / Morton 采样形式一致。
2. `surface-balanced`：保留所有高于桌面 3 mm 的 voxel，再用桌面 voxel 补齐 4096；这使机械臂和目标轮廓更清楚，但改变了桌面／非桌面采样比例。这不是物体分割，也不使用真值标签。
3. `fused surface-balanced`：前视和每帧移动的腕视分别用当帧 view/projection 矩阵回投到 base，合并后按同一 ROI、voxel 和非桌面优先采样。重叠 voxel 优先保留前视真实点。它更接近 `press_0828_17` 的双相机处理，但需要训练和在线推理一致使用腕视深度；当前发布数据没有该深度，需要重放采集。

全部五条、382 帧的前视 ROI 均有至少 8139 个 voxel。`uniform` 的非桌面点中位数按示范约为 412–744；`surface-balanced` 为约 883–1802。前两种新 4096 点共 3,129,344 个 XYZ 对前视深度回投影：最大像素误差 `4.43e-5`，最大深度误差 `5.96e-8 m`。融合候选的 1,564,672 点按来源相机逐帧回投影：最大像素误差 `2.11e-4`，最大深度误差 `1.19e-7 m`。这些是几何对应检查，不等于训练质量验收。

## base 坐标结论

先前的相机公式先得到世界坐标；`DenseCapture.add()` 在**每个捕获帧**检查 Franka base 的世界位置为 `(0,0,0)`、四元数为 `(0,0,0,1)`，否则报错。因此在这些 USB 场景中，世界 XYZ 与机器人 base XYZ 数值完全相同。之前称其为 base 是因为此处两个坐标系重合；通用实现不应依赖这个条件，未来若 base 移动必须显式做 world→base 转换并保存逐帧 base 位姿。先前 1024 点稀疏和窄 ROI 的问题是**采样与裁剪**，不是本场景中的世界/base 数值错位。

原 `press_0828_17` 的 `spatial/v1` manifest 明确是 `base_link`，4096 点；这里的候选也按机械臂 base 解释，与其坐标语义一致。两套任务的 ROI 数值无需相同。

## TacFF 坐标结论

用户指出的问题成立。已训练的 ContactWorld `03/04` 等结构化力分支，原始三通道是**右侧触觉传感器局部坐标**，但 `ContactWorldLeRobotSpatial` 给它配的 `xyz_m` 是 `[-1,1]` 无量纲平面网格，不是实际触觉采样点位置。故不能把这些旧 checkpoint 的触觉输入说成与点云共 base 的空间三维力。它们可以测试“加入局部接触分布是否有帮助”，但不能测试“物理共定位的视觉点云与触觉力融合”。`05_ff_summary` 是局部力统计量，不主张空间共定位；`06_ff_ee3d_proxy` 的 EE 平面只是近似，不能冒充实际触觉几何。

找到仿真源代码中的实际触觉采样点位置／姿态：`TactileFieldSensor.tactile_pos_world` 与 `tactile_quat_world`。源代码将 world 力投到三个存储通道 `[normal, shear_x, shear_y]=[-F_local_y,-F_local_x,F_local_z]`。新增 `contactworld_tactile_base.py` 逆映射局部向量，经逐点四元数转 world，再经 Franka base 逆变换，输出同帧 `[140,3]` `tactile_xyz_base` 和 `tactile_force_base`。独立重新执行 USB 示范 1 的 82 帧，确认数据键和形状，模拟器全程通过；与前一次捕获的 RGB、深度、局部力、EE 位姿逐元素一致。投影图中采样点落在右侧夹爪区域。源数值是仿真内部力尺度，未证实为 N。

发布的原始 Zarr 没有逐帧触觉采样点世界姿态，不能从旧力网格和 EE 姿态**准确**补回这些 base 力和位置。若采用物理共坐标的触觉输入，必须在同帧仿真重放采集上述采样点数据，并让训练和在线推理使用同一转换函数。

## 后续验收边界

用户先看上述 4096 点预览，选择前视 `uniform`、前视 `surface-balanced` 或双视角 `fused surface-balanced`，也可以指出仍缺失的可见区域。确认后才能冻结 ROI／采样协议、采集完整同帧 RGB-D／状态／触觉／实际执行动作，并重训需要公平比较的配置。旧 `02/04/09/10/11` 等使用稀疏 1024 点的结果不能当作新点云效果；旧 `03/04` 等不能当作已完成 base 坐标对齐的三维触觉实验。

## 复现

```bash
PYTHONPATH=src:sim/scripts .venv/bin/python sim/scripts/preview_contactworld_base_cloud_4096.py
PYTHONPATH=src:sim/scripts .venv/bin/python sim/scripts/render_contactworld_tactile_base_probe.py
```

触觉探针的原始重放位于 `/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_base_probe_20261002/insertion_usb_episode_001/`；`dense_capture.npz` 包含逐帧触觉采样点的 base XYZ 和力。它由以下命令生成：

```bash
bash sim/scripts/run_contactworld_sm120.sh sim/scripts/probe_contactworld_usb.py \
  --task usb \
  --source /home/sai/zx/openpi-sim-runtime/third_party/ContactWorld-sm120 \
  --output /home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_base_probe_20261002/insertion_usb_episode_001 \
  --demo /home/sai/zx/openpi-sim-runtime/runs/contactworld_multireplay_20261001/input/insertion_usb_episode_001.npz \
  --seed 0 --pose-feedback --capture-dense --white-gel --white-mounts
```
