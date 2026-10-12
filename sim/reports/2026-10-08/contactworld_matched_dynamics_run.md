# ContactWorld USB 配对动态分析：一键运行

在仓库 `/home/sai/zx/openpi-3d-tactile` 根目录执行：

```bash
bash sim/scripts/launch_contactworld_matched_dynamics.sh
```

该命令在后台运行并立即返回 PID。默认结果写到：

```text
/home/sai/zx/openpi-sim-runtime/runs/contactworld_matched_dynamics_20261008/
```

查看进度：

```bash
tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_matched_dynamics_20261008/launch.log
```

日志出现 `DONE` 后打开该目录的 `index.html`。如果中断，可原样重跑命令；已完成的配对构建和统计阶段会复用，已生成的视频也会保留。可用 `CW_MATCHED_OUTPUT=/your/path` 调整输出目录；不同数据源或烟测要使用单独目录。

本流程复用以下现有结果，不启动 Isaac Gym、不训练 π0、不下载数据或占用 GPU：

- `data/contactworld_usb_v3_bilateral`：双视图和双指触觉的 LeRobot 视频。
- `/home/sai/zx/openpi-sim-runtime/runs/contactworld_history_audit_20261008`：已完成的全量轨迹缓存和状态历史匹配。
- `/home/sai/zx/openpi-sim-runtime/runs/contactworld_sensor_discovery_20261004`：训练集力图固定色标，以及此前只使用策略可获得输入的留出预测。

报告先显示每种动作类型在完整示范中有多少片段，再显示近插口已接触状态的匹配覆盖。类别允许重叠，连续片段要求至少两个控制步；动作类别是事后分析标签，不参与选择最近邻。

报告中的 **A** 按插件相对位姿、夹持、速度和前三步动作匹配，然后比较当前示范动作。**B** 还匹配当前执行指令，再比较下一步物体运动与插件在夹爪坐标系内的相对位移。两者只在配对后读取触觉与结果。严格／放宽门槛和每项实际差异均保留。插座接触合力只显示为事后核查项。

每个展示案例有同步的两条 H.264 视频：第一条为双方前视、腕视；第二条为双方左／右指 base 坐标 Fx/Fy/Fz，共十二张动态力图。蓝负、白零、红正；所有片段采用训练集固定 P99.5 色标。视频按相对控制步对齐，提供同步播放、逐帧拖动和跳到配对帧。下方曲线显示力、执行动作和实际物体位移。

统计部分按完整查询示范留出，正则系数在剩余示范内部选择。报告六项目标：当前动作 XY、Z、旋转；给定相近当前动作的下一步物体 XY、Z 和夹持相对位移。比较匹配后剩余几何/历史差异，再加入双视图 RGB、双指粗略触觉统计、完整触觉图，以及 RGB＋触觉。另复用早期的测试预测，按动作类别分别展示关节状态、RGB 和触觉在**实际策略可获得输入**下的表现。

主要文件：

```text
index.html                            网页首页与同步案例
build/coverage.json                   类别覆盖与配对门槛
build/pairs_A.json                    A 的所有配对及差异
build/pairs_B.json                    B 的所有配对及差异
analysis/metrics.json                 分层误差、示范级区间、有符号力相关
analysis/*_predictions.json           每条查询帧的留出预测
cases/cases.json                      展示案例来源及门槛
cases/case_*_rgb.mp4                  原始双视图对照
cases/case_*_force.mp4                双方六分量动态触觉对照
figures/action_coverage.png           动作类型和配对覆盖
figures/signed_force_associations.png 有符号 Fx/Fy/Fz 与动作／位移的相关
```

这是一项离线观测分析。“动作不同”“触觉不同”和配对预测收益都不能单独证明触觉导致了纠偏，也不能回答从同一初态是否能够直接下插。此前的测试集已被多轮检查，因此本报告的统计结果应先用于定位机制和设计后续受控试验。

本机已用独立临时目录完成端到端烟测：90 个查询帧中 A 配对 82、B 配对 28；六项留出预测、同步页面和两条各 25 帧的 H.264/yuv420p 视频均成功生成。烟测数量仅验证流程，不用于研究结论。
