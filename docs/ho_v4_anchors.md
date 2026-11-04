# HO V4 固定锚点资产

HO V4 的两个任务需要在训练和在线服务启动前准备四个锚点 JSON。它们由已批准的 GLB、RGBTrack release 和既有 `object_geometry/v1` 生成；生成器不会改写原始数据或 Object Geometry。

在包含对应数据、`output/preprocess/<dataset>/rgbtrack/release_v1.json`、`output/preprocess/<dataset>/geometry/approved/` 以及 `data/<dataset>/object_geometry/v1/` 的云端项目根目录执行：

```bash
cd /workspace/mnt/sqzhang26/openpi-3d-tactile
JAX_PLATFORMS=cpu uv run --no-sync --with 'trimesh==4.6.8' python scripts/preprocess/build_ho_v4_anchors.py
```

脚本复用现有 RGBTrack exporter 的 `load_surface_points()`：种子为 `20261101 + release.objects 中的物体序号`。它先用 PCA 包围盒八角点选择不同表面点，再用确定性 FPS 补足 16 点。生成前会验证原始 release、已批准 Mesh 和 Object Geometry Manifest 的角色、哈希与采样来源；还会读取有效的现有 sidecar 帧，检查 512 点的刚体距离指纹与重采样点序一致。若旧 sidecar 的采样实现不可重现，会停止并报告错误。重复执行时，已有 JSON 不一致则拒绝覆盖。

输出：

```text
assets/ho_v4_anchors/{1028_press_button,1030_yellow_socket}/{manipulated,target}.json
output/analysis/ho_v4_anchor_review/{1028_press_button,1030_yellow_socket}_{manipulated,target}.html
```

JSON 同时保留已验证的 512 点局部几何、16 个固定索引与坐标，以及采样数组、GLB、release、批准清单和采样代码哈希。训练及服务只读取 JSON，不重新运行 PCA 或 FPS，也不依赖 trimesh；但仍需要可读取的已批准 GLB、release、批准清单和 sidecar Manifest，用于核对资产与当前来源一致。每帧验证输入点的刚体距离指纹后，将固定索引传入模型。全部无效的物体保持既有 Mask；部分有效但所有固定锚点均无效的物体无法校验点身份，会明确报错。

在线调用方需要在现有 `object_geometry` 字典中提供 `surface_points_sha256` 字段，值为包含 `manipulated` 和 `target` 的字典，两项分别等于对应 JSON 的 `surface_points_sha256`。服务端还会检查输入 512 点的顺序和刚体几何；缺失指纹、几何不匹配或缺少 Object Geometry 时拒绝推理。现有部署客户端目前不提供 Object Geometry，尚不能直接用于 HO V4。
