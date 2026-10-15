# 固定初态示范：连续轨迹对比（当前默认入口）

打开[简洁对比页](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/index.html)，刷新即可看到新版。40 个初态、每个初态 13 个配置，左侧测试双视图，右侧固定训练示范双视图；相同初态的全部配置共享一条训练示范。共引用 35 条不同训练示范，训练视频保留原始逐帧顺序，来源不跳变。

- 默认从起点按控制步／帧索引同步。训练采集时的真实帧率未知，不能解释成物理时间对齐。
- 可切换“首次插座接触对齐”，0 为各自首次插件—插座合力范数超过 0.1 的动作后观测帧；负数表示接触前。7 次测试没有接触锚点，在此模式暂停并标注，不伪造接触时间。
- 先结束的一侧保持末帧，未开始的一侧保持首帧；不伸缩轨迹长度。
- 每个初态的进度条、逐步按钮和播放控制同步 13 个配置；展开初态时加载该组视频，收起初态会释放视频资源；再次展开会重新加载。
- “初态最近”沿用已有训练覆盖分析的几何距离（目标位置、插件相对目标位姿、夹持相对位姿），不保证视觉最像。页面显示具体毫米与角度差；只从训练划分选择。
- 触觉、点云按需打开下方诊断链接；旧动态近邻页面也给左侧测试轨迹显示阶段色带与当前数值。
- 两侧视频下方新增逐帧阶段色带，可点击色带跳转。阶段依据已保存的 U 盘—插座接触合力、XY 偏差、Z 高度、轴角和环境成功判据；无需重新仿真。手指触觉从夹持开始就可能活跃，不能用它直接认定已碰到插座。

色带区分未接触、接触后暂时分离、孔口对齐中、近轴待下降、正在轴向下降、下降后停顿、目标高度附近，以及环境目标判据通过。近轴操作阈值为 XY ≤ 2 mm 且轴角 ≤ 5°；“正在下降”要求当前接触期间最近三帧 Z 降低至少 0.5 mm；“下降后停顿”要求首次接触以来累计下降至少 3 mm；“目标高度附近”要求近轴且 |Z|≤5 mm。这些是轨迹诊断标签，数值在每帧显示，不是策略输入，也不单独证明纠偏意图或可靠插入。2026-10-04 更正：旧标签只看最近三帧，会将几乎插完但已停住的案例误称“未见下插”。

生成／更新简洁页（复用已经生成的训练视频）：

```bash
python3 sim/scripts/render_contactworld_fixed_initial.py
```

源码：[生成器](/home/sai/zx/openpi-3d-tactile/sim/scripts/render_contactworld_fixed_initial.py)、[页面模板](/home/sai/zx/openpi-3d-tactile/sim/scripts/contactworld_fixed_initial.html)。[清单](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/fixed_initial_manifest.json)记录逐例参考来源、接触锚点与视频路径。[校验结果](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/fixed_initial_verification.json)记录视频检查结果。网页通过 JavaScript 语法与 DOM 模拟交互检查；又在本机 Chrome 143 以实际本地文件方式验证：测试与训练视频从文件路径加载、跳转到第 50 帧、收起再展开后正常恢复。

以下为保留的完整传感器／动态近邻诊断页说明。旧生成器检测到新版清单后会更新 `dynamic_nearest.html`，不覆盖默认简洁页。

---

# 40 个初态 × 13 配置：多模态与动态训练近邻回放

网页位置：[完整对比页](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/dynamic_nearest.html)。可以直接在本机浏览器打开；不需要联网或安装服务。

已生成全部 **520 个案例 / 1,040 段对比视频**（每案例两种检索）。[校验结果](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/verification.json)通过：全部视频的编码、尺寸、帧数、时长正确；46,800 组训练 f/f+1 来源检查通过；独立重算 117 个视觉最近邻一致；18 段视频的首尾帧抽样解码通过。

每个初态可展开 13 个配置，从上到下排列。一段对比视频的三列分别为：

1. 此配置当前的真实模型观测。
2. 从实际训练划分检索的最近观测帧 f。
3. 同一条训练示范的真实 f+1 帧，展示示范下一步发生什么。

三列均包含前视＋腕视 RGB、左右指分别展示的 base Fx/Fy/Fz、局部 C0/C1/C2、固定机械臂 base 视角下的 4096 点云与两指 taxel 位置。当前动作的 dXYZ 也显示在画面里。近邻每次重新检索，视频明确标出 source、LeRobot episode、frame；来源切换时会提示 SOURCE SWITCH。这是随测试观测变化的训练帧拼接，不是一条真实连续的训练轨迹。

## 推荐操作

- 先展开一个初态，在顶部用“同步进度条”让 13 个配置停在相同控制步；再向下逐个比较。
- 用“上一观测／下一观测”检查接触附近。每个配置也可以单独播放或步进。
- 同步播放时，使用该初态顶部的同步暂停按钮；若希望自由拖动单个视频，先暂停同步。
- 默认“视觉近邻”只用双相机冻结 ResNet18 特征。另一个选项“输入模态诊断近邻”使用 RGB＋状态以及配置对应的 P/T 固定诊断特征。后者不是各 π0 的内部编码器，也不完全模拟 summary/map 等表示差异。
- 每个配置下方可展开原始完整帧率 RGB 视频。对比视频和原视频可分别打开为独立 MP4。
- 可以先看 [10005](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/dynamic_nearest.html#seed10005)、[10008](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/dynamic_nearest.html#seed10008)、[10014](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/dynamic_nearest.html#seed10014)、[10034](/home/sai/zx/openpi-sim-runtime/runs/contactworld_pi0_v3_13models_40trials_20261003/all_sensor_comparison/dynamic_nearest.html#seed10034)，与先前分析的案例对应。

## 时间与可视化约定

完整 P/T 模型观测只在每四个控制步保存一次，即约每 0.267 秒一次。对比视频按这些快照更新并保持到下一次观测，没有合成中间点云。MP4 使用 H.264、yuv420p、15 fps；重复帧仅用于保持真实时间长度和播放器兼容性。所有 520 次评测的原始完整 RGB 视频仍可查看。

两指力分量使用共同固定的有符号色阶：蓝负、白零、红正。每个坐标系的每个分量，色阶来自所有测试与训练力场的抽样绝对值 P99.5，并在所有帧、两指、各配置之间固定。超色阶的比例会显示 clip%，真实最大绝对数值保留。不同轴的色阶可能不同，请结合数值比较；这是合成 TacFF 单位，不能直接解释为牛顿。

点云显示全部 4096 个 XYZ，使用所有案例共享的投影视口，保留米制几何比例，没有逐帧居中或缩放。颜色表示 base 高度，仅为展示；策略并未输入这种颜色。01/03 没有点云输入，01/02 没有触觉输入，但画面仍展示保存的传感器信息用于诊断，并在顶部标明 not input。触觉图展示输入前的原始力场，不是网络 token；summary 配置实际会进一步汇总。

检索仅在 95 条实际训练示范、5064 个有效观测帧内进行；f+1 只在搜索完成后用于展示，不进入相似度计算。视觉近邻可以与人认为最像的图像不同，最近的定义是已说明的特征距离。两种检索的距离数值不可直接相互比较。每条视频旁的 matches.json 保留全部逐帧来源。

## 生成与校验

源码：[生成器](/home/sai/zx/openpi-3d-tactile/sim/scripts/render_contactworld_compare_all.py)、[网页模板](/home/sai/zx/openpi-3d-tactile/sim/scripts/contactworld_compare_all.html)。输出与原始评测文件分开保存，支持从已完成案例继续生成。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 sim/scripts/render_contactworld_compare_all.py --workers 4
python3 sim/scripts/render_contactworld_compare_all.py --verify-only
```

仅更新网页不重新生成视频：

```bash
python3 sim/scripts/render_contactworld_compare_all.py --page-only
```

校验报告写入输出目录的 verification.json，检查 40 个配对初态、13 个配置、全部 MP4 编码/尺寸/帧数/时长、训练划分与 f/f+1 来源，另独立重算部分视觉近邻并解码部分视频的首尾帧。当前浏览器连接没有可用实例；页面 JavaScript 通过语法检查，导航、同步定位、检索切换与播放控制用 DOM 模拟执行检查，不替代真实浏览器测试。
