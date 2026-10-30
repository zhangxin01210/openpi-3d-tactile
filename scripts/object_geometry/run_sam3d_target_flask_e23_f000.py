from pathlib import Path

import numpy as np
import torch
from PIL import Image
from hydra.utils import instantiate
from omegaconf import OmegaConf

HOI = Path("/workspace/mnt/sqzhang26/hoi")
SAM3D = HOI / "repos/sam-3d-objects"
INPUT = Path("/workspace/mnt/sqzhang26/openpi-3d-tactile/data/pose_eval/sam3d_inputs/e23")
OUTPUT = Path("/workspace/mnt/sqzhang26/openpi-3d-tactile/output/pose_eval/meshes/target_flask_e23_f000")
OUTPUT.mkdir(parents=True, exist_ok=True)
if (OUTPUT / "flask.glb").exists():
    raise FileExistsError("拒绝覆盖已有的 flask.glb")

config_path = SAM3D / "checkpoints/hf/pipeline.yaml"
moge_path = HOI / "weights/moge-vitl/model.pt"

assert moge_path.is_file(), f"MoGe 权重不存在：{moge_path}"

config = OmegaConf.load(config_path)
config.workspace_dir = str(config_path.parent)
config.rendering_engine = "pytorch3d"
config.compile_model = False

# 将 Hugging Face 模型名替换为已下载的本地文件
config.depth_model.model.pretrained_model_name_or_path = str(moge_path)

rgb = np.asarray(
    Image.open(INPUT / "target_flask_f000_rgb.png").convert("RGB")
).copy()

mask = np.asarray(
    Image.open(INPUT / "target_flask_f000_mask.png").convert("L")
)

mask = ((mask > 127).astype(np.uint8) * 255)

assert rgb.shape[:2] == mask.shape
assert mask.max() > 0, "Mask 是空的"

print("输入图像：", rgb.shape, flush=True)
print("Mask 像素：", int((mask > 0).sum()), flush=True)

print("\n[1/3] 加载 SAM 3D + MoGe...", flush=True)
pipeline = instantiate(config)

print("\n[2/3] 开始目标烧瓶三维重建...", flush=True)
result = pipeline.run(
    rgb,
    mask=mask,
    seed=42,
    stage1_only=False,
    with_mesh_postprocess=False,
    with_texture_baking=False,
    with_layout_postprocess=False,
    use_vertex_color=True,
)

print("\n[3/3] 导出结果...", flush=True)

glb = result.get("glb")
if glb is not None:
    path = OUTPUT / "flask.glb"
    glb.export(str(path))
    print("Mesh:", path, flush=True)
else:
    print("警告：没有返回 GLB", flush=True)

gs = result.get("gs")
if gs is not None:
    path = OUTPUT / "flask_splat.ply"
    gs.save_ply(str(path))
    print("Gaussian Splat:", path, flush=True)

print("\n输出键：", sorted(result.keys()), flush=True)
print("✅ 推理完成！", flush=True)
