# 双指触觉短动作分支：修订版运行说明

首轮运行的复核见 [contactworld_tactile_action_branches_audit.md](contactworld_tactile_action_branches_audit.md)。首轮 378 条分支仅 98 条通过，完整七动作对照只有 3 个初态；原跨状态模态预测分数不可用于结论。

在仓库根目录一键后台运行：

```bash
bash sim/scripts/launch_contactworld_tactile_action_branches_v3.sh
```

输出为 `/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_action_branches_v3_20261008/`。脚本打印 PID、`launch.log` 和 `index.html` 路径；`analysis.json` 出现代表三个阶段完成。不要让两轮 Isaac Gym 试验同时使用同一张 GPU。新版本使用独立输出目录，首轮数据不覆盖。

修订点：

- 六种物理状态各预选 4 个不同示范/夹持配置，共 24 个候选初态。
- 每个动作争取 3 次有效重复，最多试 5 次。只有至少 2 次稳定保持，并且某候选动作至少 2 次从相近起点执行，才算该动作在该状态可比较。
- 起点严重位姿失配仍拒绝。原示范与在线重放的双指力图差异单独记录；此外检查不同重放之间的位姿、姿态、触觉和保持动作结果，避免把不一样的起点误称为同状态。
- 只有七种动作都至少有 2 次有效重复的初态进入模态预测。如果完整初态不足 6 组，网页明确不输出跨初态模态分数。
- 视频包含同步双视图和双指六张 base 坐标 Fx/Fy/Fz 动态力图，即使第 0 次试验被拒绝，后续首次有效重复也会生成视频。

运行前代码检查已通过；单独复测中，首轮漏掉的自由空间状态 source 6 恢复了全部 21 条稳定分支，接触状态 source 130 在允许少数重放离群值后取得七种动作各至少 2 次有效重复。完整 v3 尚未执行，具体覆盖量需要看最终 `analysis.json`。
