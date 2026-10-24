# RGBTrack 集成约定

RGBTrack 以 Git submodule 形式固定在 `third_party/RGBTrack`。主仓库只记录
submodule URL 和 gitlink，不复制第三方源码到主仓库历史，也不在运行时自动
更新第三方版本。

当前固定 commit：

```text
23e43f832b82f4813d6ab72773779bf8d2ee4d83
```

## 其他电脑初始化

已有主仓库 checkout 时，只初始化这个 submodule：

```bash
git submodule sync -- third_party/RGBTrack
git submodule update --init --checkout third_party/RGBTrack
git -C third_party/RGBTrack rev-parse HEAD
```

输出应为上面的固定 commit。不要使用
`git submodule update --remote`，也不要在主仓库中提交 RGBTrack 工作树里的
普通文件；主仓库应只出现 `third_party/RGBTrack` 的 gitlink 变化。

RGBTrack 自身声明的嵌套依赖不会由本约定自动初始化；本次整理不安装 Python
环境、不下载模型权重，具体实验依赖由后续任务单独处理。

## 本仓库目录约定

- 自有 RGBTrack 适配代码：`scripts/object_geometry/pose_tracking/`
- RGBTrack 模型权重：`checkpoints/rgbtrack/`
- pose evaluation 数据和物体 GLB：`data/pose_eval/`
- pose evaluation 输出：`output/pose_eval/`

后三类目录是电脑本地数据/产物，已加入 `.gitignore`，不会进入 Git。正常的
Python、配置和文档文件仍可被跟踪；`configs/` 本身不再被整体忽略，只保留
生成压缩包的忽略规则。
