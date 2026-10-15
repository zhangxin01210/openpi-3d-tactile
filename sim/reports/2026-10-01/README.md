# UniVTAC `insert_hole` 全量视频检查入口

已把本机现有的 100 条下载版 HDF5 与 3 条本机重采 HDF5 全部转换为同步多模态视频。输出目录为 `/home/sai/zx/openpi-sim-runtime/visualizations/univtac_insert_hole/`，打开其中的 `index.html` 可按缩略图浏览和播放全部 103 条；`manifest.json` 逐条记录源文件、帧数、时长、编码和解码检查。视频与原始 HDF5 分开放置，避免把约 45 MiB 的派生视频写入代码仓库。

| 数据 | 条数 | 源帧数 | 视频画面 |
| --- | ---: | ---: | --- |
| 下载版 `insert_hole` | 100 | 19,151 | head/wrist RGB、左右 GelSight RGB、左右 `press_depth` |
| 本机重采 depth 版 | 3 | 669 | 上述六格，另加 head/wrist 世界相机深度 |

导出时保留全部 60 Hz 帧；输出为 MP4 内 AV1 Main、`yuv420p`，与用户提供的可播放参考视频采用相同 codec/pixel format。103 条视频均由 FFmpeg 完整解码，帧数与 HDF5 `step` 长度相等。另已把 [`2026-09-30/samples`](../2026-09-30/samples) 里的四条旧 `mp4v` 样例改为 AV1。这里的“通过”指文件结构及本机 FFmpeg 解码；具体 IDE 播放器能否打开，仍以实际打开检查为准。

复现命令：

```bash
/home/sai/miniconda3/envs/univtac-isaac51/bin/python sim/scripts/export_univtac_review_videos.py
```

脚本逐条验收、生成缩略图，并可在中断后重跑；已有视频会先按编码、像素格式、尺寸和帧数检查再复用。可用 `--output-dir` 指定其他目录，用 `--encoder libx264` 输出 H.264，或用 `--encoder libsvtav1` 在没有 NVIDIA AV1 编码器的机器上生成 AV1。

**评估时需要注意的已知缺口：**下载版的 100 条 HDF5 不含 head/wrist 世界相机 depth、内参及世界位姿，视频不能为它们补出点云；只有本机重采的 3 条包含这些字段。`press_depth` 是手指夹持物体时的胶层压入量，不能单凭非零值认定 peg 与孔座接触，也不能当成三维接触力。下载版仅保存上游 scripted 成功轨迹，不能从 100/100 视频推断任务成功率或纠偏能力。视频只覆盖图像/深度外观，动作、位姿、marker 和数值异常应继续查看[逐 episode 数据审计](../2026-09-30/univtac_insert_hole.json)及[阶段报告](../2026-09-30/README.md)。
