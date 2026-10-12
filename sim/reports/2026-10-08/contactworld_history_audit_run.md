# ContactWorld USB 动态状态与触觉历史审查

## 一键启动

在 `/home/sai/zx/openpi-3d-tactile` 仓库根目录执行：

```bash
bash sim/scripts/launch_contactworld_history_audit.sh
```

该命令在后台运行，返回 PID、日志和报告路径。默认输出：

```text
/home/sai/zx/openpi-sim-runtime/runs/contactworld_history_audit_20261008/
```

查看进度：

```bash
tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_history_audit_20261008/launch.log
```

看到 `DONE` 后打开该目录的 `index.html`。重复执行会复用已完成的 `prepare`、`fit`、`match` 阶段；报告会重新生成。运行期间不要移动原始数据或前一次 sensor-discovery 缓存。脚本不启动 Isaac Gym、不下载文件、不占用 GPU，不改动先前的报告。

先前的缓存应位于 `/home/sai/zx/openpi-sim-runtime/runs/contactworld_sensor_discovery_20261004`；双指 LeRobot 数据应位于 `data/contactworld_usb_v3_bilateral`。这两个路径也可以用 Python 脚本的 `--prior`、`--dataset` 调整。输出位置可用 `CW_HISTORY_OUTPUT` 环境变量调整，且需要对目标目录有写权限。每次只允许一个进程写同一输出目录。

## 报告如何读

1. **匹配覆盖**：先看严格和放宽门槛各能匹配多少帧。匹配只使用接触前已知的几何、夹持、速度及前三步动作；插座合力只在匹配后检查差异，不参与检索或门槛。若覆盖不足，不能把最近例子当作“相同状态”。所有候选与门槛在 `match/matches.json`。
2. **案例短片**：上排为独立验证／测试示范，下排为训练示范。每排依次是前视、腕视、左指、右指。中心帧前后三步保持连续，不逐帧更换邻居。旁边六通道触觉图是左右指在机器人 base 下的 Fx/Fy/Fz，采用固定训练集色标；曲线显示接触前后动作、力和位置。差异表列出每个匹配变量与门槛。展示案例在通过门槛后才按触觉差异选择，供观察范围，**不是代表性抽样**。
3. **动作与运动表**：所有探针按完整示范分 train/val/test，正则系数只用 val 选择，表中为 10 条留出 test 示范的接触帧。`action` 目标为当前示范六维动作，不输入当前动作；`motion1` 给定当前动作预测下一步物体运动；`motion4` 离线给定完整四步动作序列，预测四步物体运动。后一项用于物理响应诊断，未来动作并非实际策略可用的输入。`H_geometry_history` 对 `H_T`、`H_RGB` 等的差值和示范级自助采样区间见 `paired_intervals.json`。
4. **时间曲线**：每条示范分别计算触觉与滑动、物体竖直运动、不同方向动作的滞后相关，再汇总中位数与四分位范围。正偏移代表触觉时刻之后的变量。相关峰值不能直接解释为传感器延迟；还需结合动作历史探针和案例曲线。

匹配使用仿真真值，仅为分析工具，不是 π0 的输入。门槛是预先固定的探索性数值，报告同时展示严格／放宽覆盖和未通过项。单纯观察数据不能判断“此刻能否直接下插”，因为缺少从相同动态初态施加同一动作的反事实试验。

## 测过的最小流程

小样本命令：

```bash
CW_HISTORY_OUTPUT=/tmp/cw_history_audit_smoke_verified_20261008 \
  bash sim/scripts/launch_contactworld_history_audit.sh --smoke --max-pairs 3
```

已在本机跑通数据准备、18 个轻量预测探针、状态匹配、3 个案例、H.264/yuv420p 视频、匹配后触觉差异与动作差异散点图和 HTML 报告。烟测只使用 6/2/2 条 train/val/test 示范，不用于科学结论。它的 82 个候选查询帧中，严格历史匹配为 0，放宽历史匹配为 16；全量覆盖需等正式运行后再判断。
