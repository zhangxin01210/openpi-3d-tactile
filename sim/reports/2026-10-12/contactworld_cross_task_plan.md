# ContactWorld 07/10 跨任务验证计划

## 目标与控制变量

目标不是证明“加入更多输入总会更好”，而是检验 USB 上表现较好的两种触觉表示，在不同接触机制中是否仍能稳定增加价值。

每个新任务都从同一个 π0 base 权重独立训练四个配置，不继承 USB checkpoint：

| ID | 输入 | 作用 |
|---:|---|---|
| 01 | 双视图 RGB | 视觉基线 |
| 02 | RGB + 4096 点 fused base-frame 点云 | 隔离点云收益 |
| 07 | 02 + 双指 base XYZ/Fx/Fy/Fz 结构化编码，左右合池化，6 个触觉 token | 检验无手指标识的空间力集合 |
| 10 | 02 + 左右各一张 base Fx/Fy/Fz 耦合力图，每指 3 个查询，6 个触觉 token | 检验保留指垫拓扑的耦合力图 |

主要对照是 07−02、10−02 和 07−10。只比较 07/10 与 RGB 会把点云收益误算成触觉收益。

## 第一轮任务组合：三个新任务

| 任务 | 类型与作用 | 发布规模 | 本轮定位 |
|---|---|---:|---|
| `insertion_peg` | 与 USB 同属插入，但几何、间隙和接触分布不同；仍为 6D 末端动作 | 153 条、11,728 帧；压缩包约 5.44 GB | 近域复现与流水线首测 |
| `disassembly_barbed_flat` | 从约束结构中拔出，运动方向和成功机制与插入相反；动作含第 7 维夹爪控制 | 200 条、18,177 帧；约 7.33 GB | 跨接触机制验证 |
| `screwing_valve` | 持续旋转、长时接触和夹持控制；动作含第 7 维夹爪控制 | 102 条、16,307 帧；约 7.66 GB | 跨运动形式与长时控制验证 |

这三个任务形成“近域插入—反向拆卸—持续旋转”三层证据。第一轮共 3×4=12 个模型，在 8 张 A800 上是两波训练。

`insertion_power` 暂不放入第一轮。它仍是插入任务且数据包约 12.6 GB，新增信息与 Peg 部分重叠。如果结果表现为“Peg 有效、拆卸和旋拧无效”，再加入 Power Plug，用来判断方法只在插入家族有效，还是只在 USB/Peg 的特定几何上有效。Nut/Bulb 数据更大、轨迹更长且无初态随机化；探索/分拣任务又混入搜索和光照问题，不适合作为第一轮结构验证。

## 阶段 1：下载和原始数据审计

官方数据仓库固定到与现有 USB/Peg 相同的 revision。实现时增加可恢复、分块、SHA256 校验且默认绕过代理的下载器：

```bash
python sim/scripts/download_contactworld_cross_tasks.py \
  /home/sai/zx/openpi-sim-runtime/downloads/contactworld \
  --tasks disassembly_barbed_flat screwing_valve \
  --workers 12
```

Peg 已经下载、校验并解压在本机，不重复下载。新增下载量约 15 GB。下载器必须先取得并固定官方 LFS/Xet 文件大小和 SHA256，不能只依赖 HTTP 成功。官方发布页：<https://huggingface.co/datasets/Pokuang/ContactWorld/tree/main/releases>。

随后由准备脚本无覆盖地解压到独立目录，并审计每个 Zarr 的：episode 数、总帧数、RGB/状态/动作/TacFF 形状、NaN、episode 边界及动作维数。原始发布数据只有右指 TacFF，且保存的 1024 点点云不能替代已经审核过的 4096 点 RGB-D 点云，因此不能直接转换后训练 07/10。

## 阶段 2：每个任务只重放一条，先验收再全量

在本机隔离的 ContactWorld 运行环境中扩展现有补丁：

1. 为 Peg、Flat Barb、Valve 各自启用左右指 TacFF，并让左右传感器使用独立的物体采样点缓存。
2. 同帧记录 front/wrist RGB-D、相机内外参、机器人 base pose、18 维 q/dq、实际执行动作、双指 taxel base 坐标与 base 三维力。
3. Peg 保持 6D 动作；Flat Barb 和 Valve 完整保留第 7 维夹爪动作。当前 `ContactWorldOutputs` 的 `[..., :6]` 必须改为任务指定维数，不能静默丢弃夹爪控制。
4. 每个任务先重放一条完整示范，生成审核网页：双 RGB、密集深度点云、裁剪后的 4096 fused 点云、左右六张 Fx/Fy/Fz 图、状态与动作曲线、源码/重放终态成功判据。
5. 为每个任务单独确定 base-frame ROI。USB 的 ROI 不直接复用；目标是保留机械臂、被操作物和目标结构的可见表面，再做 5 mm voxel 与 surface-balanced 4096 点采样。

只有三条示范都通过以下门槛才全量运行：画面与动作逐帧同步；左右触觉方向和位置正确；点云与 RGB 对齐；7D 动作未被截断；重放终态与任务成功语义一致。

## 阶段 3：本机全量重放与 LeRobot 转换

全量过程在本机完成，因为双指触觉和 4096 点点云必须从旧 Isaac Gym 环境重新采集；训练服务器不需要安装该环境。

对每个任务：

1. 根据固定 source episode ID 建立可追溯的 train/val/test 划分。
2. 顺序重放全部示范，保留成功复现及失败清单，不用调摩擦等方式逐条追成功。
3. 导出独立的 `data/contactworld_<task>_v3_bilateral`：
   - LeRobot 视频：front、wrist；
   - `observation.state`：9 个 q + 9 个 dq；
   - `action`：Peg 6D，Flat/Valve 7D；
   - spatial sidecar：4096 fused base XYZ、左右各 140 个 taxel XYZ/force、左右 10×14×3 local/base 力图、逐帧相机和坐标变换；
   - 训练集独立计算的 normalization stats、source receipt、SHA256 与 split manifest。
4. 触觉预览 MP4 只用于人工审查，不进入模型 RGB 槽。
5. 自动验证 H.264/yuv420p、帧数、finite、坐标范围、左右非重复、动作/状态维数和训练窗口数。

USB 全量重放 201 条约用 53 分钟，转换和审计约十余分钟。Peg 预计 1 小时量级；Flat 和 Valve 轨迹更长且是首次适配，分别预留 2–4 小时。首次任务补丁与单条验收比批量计算更可能耗时。

## 阶段 4：训练代码改造

把 USB 专用逻辑改成任务规格驱动，同时保持已有 USB 配置名和行为不变：

- `ContactWorldTaskSpec` 保存 dataset root、指令文本、动作维数、仿真类、最大步数和成功判据。
- 数据 loader 从 manifest 读取 task/action_dim，不再把路径和 USB 指令写死。
- `ContactWorldInputs/Outputs` 支持 6D/7D 动作，并在形状不匹配时直接报错。
- 生成 12 个配置：`pi0_cw3_{peg,barbed_flat,valve}_{01,02,07,10}_...`；07/10 的编码器、token 数、路由、LoRA 与 USB 完全一致。
- 所有配置使用同一 π0 base 初始化、seed、batch size、action horizon 和 20k cosine schedule；保存 5k/10k/15k/20k。主结果用统一的 20k checkpoint。若同一任务四个配置的训练/验证曲线在 20k 都未稳定，再把该任务四个配置一起续训到 30k，不能只延长落后的模型。
- 新增 dataset/model-input audit，抽取同一帧经过 loader、归一化和 tokenizer 后的 RGB、点云、触觉以及动作目标，训练前生成可视化验收页。

## 阶段 5：上传与一键训练

服务器只需同步：

1. 修改后的代码仓库；
2. 三个最终 `data/contactworld_<task>_v3_bilateral/` 目录，包括 `assets/norm` 和 `spatial/manifest.json`。

服务器不需要原始 tar、Zarr、重放缓存、ContactWorld/Isaac Gym 资产或官方 world-model checkpoint。π0 base 权重沿用当前服务器已有路径。

最终训练入口设计为：

```bash
export OPENPI_CONTACTWORLD_BASE_WEIGHTS=/path/to/pi0_base/params

python sim/scripts/launch_contactworld_cross_task_matrix.py \
  --tasks peg,barbed_flat,valve \
  --configs 01,02,07,10 \
  --label cw_cross_v1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --check-only

python sim/scripts/launch_contactworld_cross_task_matrix.py \
  --tasks peg,barbed_flat,valve \
  --configs 01,02,07,10 \
  --label cw_cross_v1 \
  --gpus 0,1,2,3,4,5,6,7
```

launcher 在启动前完成全量数据验证、模型输入 smoke、权重检查、配置去重与输出目录防覆盖；运行中自动补位 GPU，并为每个任务/配置单独记录日志。按此前每模型约 5 小时估算，12 个模型在 8 卡上约 10 小时；Valve/Flat 的 7D 动作和更长数据可能使总时长落在 10–15 小时。

## 阶段 6：配对闭环评测与判定

每个任务先固定 40 个测试初态，4 个模型共享同一批初态、控制频率、重规划间隔和时限，总计 480 次 rollout。保存双视图、双指力、点云、动作、物体/目标真值、接触与夹持相对运动，便于之后解释失败。

主比较按任务分别报告：

- 02−01：点云增益；
- 07−02、10−02：触觉的增量价值；
- 07−10：两种触觉结构谁更适合该接触机制；
- 曾成功、最终成功、任务相关的稳定成功、首次成功时间以及失败类型。

不同任务不把所有 rollout 混成一个比例，而是先做同 seed 配对，再对三个任务做等权 macro average 和按任务 bootstrap 区间。40 条是筛选阶段；出现稳定信号后，冻结评测设置，把相关模型补到每任务 100 个初态。

“稳定有效”的最低证据标准预先定为：07 或 10 相对 02 在至少两个新任务上有同方向改善，三任务等权平均为正，并且没有一个任务出现明显灾难性退化。若只在 Peg 上有效，应准确表述为插入家族内复现；若拆卸或旋拧失败，则根据失败类型决定增加 Power Plug、改进时间历史，还是处理 7D 夹爪控制，而不是直接汇总成一个平均数。

## 实施节点

1. 完成通用下载/审计脚本并由用户运行约 15 GB 下载。
2. 完成三任务各一条双指+点云重放；停下人工审核。
3. 审核通过后本机全量重放、转换、验证；停在可上传数据集。
4. 完成 12 个训练配置、输入审计和一键 launcher；用户同步后先跑 `--check-only`。
5. 训练结束后再完成三任务闭环评测脚本。

