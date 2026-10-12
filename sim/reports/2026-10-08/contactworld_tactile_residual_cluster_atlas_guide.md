# ContactWorld USB 双指触觉残差聚类

## 一键运行

在仓库根目录执行：

    bash sim/scripts/launch_contactworld_tactile_residual_cluster_atlas.sh

脚本在后台运行，不占用当前终端。输出目录：
/home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_residual_clusters_20261008/

查看进度：

    tail -f /home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_residual_clusters_20261008/launch.log

完成以该目录的 complete.json 和日志末尾 DONE 为准；打开 index.html 查看报告。
重跑并保留旧结果时，将 CW_TACTILE_RESIDUAL_OUTPUT 设为新目录。
需固定类别数时，在启动命令末尾加 --k 10；默认比较 K=2,3,...,18 后自动选择。

## 输入与坐标

覆盖 ContactWorld USB v3 双指数据的整段 120 条示范，不按插座接触标签筛帧。
每条示范分别在首次插座接触前找 4 帧稳定夹持窗口；
窗口必须有非零且稳定的夹持力。
以每个 taxel 局部三通道的窗口中位力作基线，
先在传感器局部坐标逐帧相减，再依当前帧 taxel 姿态
和机器人 base 姿态旋转到机器人 base 下的 Fx/Fy/Fz。
120 条示范均找到基线。完整预检核对了 8,347 帧，
局部到 base 的原始力转换与存储的 base 力完全一致（最大绝对差为 0）。

每帧的聚类输入为左右指各 140 个 taxel 的三分量残差，共 840 个有符号值。
六张图按训练集每指每分量的残差绝对值 99.5% 分位固定缩放，
极值截断到 ±3。没有逐帧归一化或 PCA。
接触、动作、USB 位姿、成功与否均不参加聚类和 K 选择。
验证与测试按整条示范隔离；训练、验证、测试分别是 95、15、10 条。

候选 K 用验证集 silhouette、类内 RMS、重启稳定性 ARI 和最小类覆盖共同检查。
自动规则先要求训练与验证中每类至少约 0.5% 的帧、ARI 至少 0.6，
随后选验证 silhouette 最好者；0.005 以内选较小 K。
若没有候选满足门槛，会逐级放宽；完整比较在 k_selection.json。
类别数少仅表示压缩更强，不保证每类内部更精确。

## 网页怎么读

每个类别从上往下排列，放六张类中心残差图，
再放多个来自不同示范的代表帧和连续短片。
代表帧由六张图中最大的归一化 RMS 选出，
避免其中一张差很多却仍被称为相似。
每例并排显示原始 base 力图、投到当前 base 的稳定夹持基线、
用于聚类的残差图。原始与基线共用一套颜色标尺；
残差图跨所有类别共用另一套颜色标尺。

视频包含前视、腕视、六张动态残差图与原始双指触觉预览。
按钮可按各自代表帧同步播放，滑杆可共同定位到代表帧前后。
画面同时标出代表帧类别和当前帧类别。
短片中的类别可能变化，不能把整段片子当作代表帧所属类别。
USB 接触、当前动作、下一步实际位移仅作事后说明。

报告提供类间距离、动作关联、测试示范逐帧类别时间轴，
以及 frame_assignments.csv、representatives.json、
cluster_summary.json、baseline_audit.json 和 cluster_model.npz。

本实验是描述性分析。同一残差类别可能对应不同动作和接触状态；
它不能单独证明触觉能选择纠偏方向。
要评价控制价值，应结合相同初态的受控动作结果。



## 全部成员与 K 敏感性

精选首页的 5 条视频是每类靠近中心、来自不同示范的帧，
优先取测试 3 条、验证 1 条、训练 1 条；不是类内差异最大的 5 条。
首页现在提供 all_members.html 链接。该浏览页收录 8,347 帧，
可按划分、接触、source 过滤，按中心距离、六图最大误差
或示范时间排序，分页逐帧查看；最多选择 4 帧同步对齐。
选中后双视图、双指原始触觉视频以及六张残差图按当前帧更新。
全部 120 条示范的四路原视频只复制一次到 member_media，
总计约 79 MB；残差图压缩在 all_members_data.js 中，约 10 MB。
因此页面与 member_media 目录一起移动即可离线查看。

本次自动 K=5 的验证 silhouette 为 0.3127；
K=6 为 0.3172，但按预设的 0.005 近似并列规则选较小的 5。
K=7 的 silhouette 为 0.3078、ARI 为 0.857；
K=9 为 0.3059、ARI 为 0.804。K 越大，类内 RMS 通常越低，
这不能单独证明 K 更合理。K=7 测试集最小类只有 11 帧，
K=9 最小类只有 7 帧；这些细类的动作比例容易受少数示范影响。
为了视觉审核，另存两份固定 K 报告，均有全成员页：

- /home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_residual_clusters_k7_20261008/index.html
- /home/sai/zx/openpi-sim-runtime/runs/contactworld_tactile_residual_clusters_k9_20261008/index.html

不同 K 的类别编号不能直接对应。可先比较中心图与边缘成员，
再判断新增类别是否把你关心的不同接触／动作过程分开。

