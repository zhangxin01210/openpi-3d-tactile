import runpy
from pathlib import Path
import torch

HOI = Path("/workspace/mnt/sqzhang26/hoi")

DINO_REPO = Path(
    "/root/.cache/torch/hub/facebookresearch_dinov2_main"
)

DINO_WEIGHTS = Path(
    "/root/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
)

assert (DINO_REPO / "hubconf.py").is_file(), "DINOv2 源码缓存不存在"
assert DINO_WEIGHTS.is_file(), "DINOv2 权重缓存不存在"

print("[OK] DINOv2 本地源码和权重均已找到", flush=True)

original_load = torch.hub.load

def load_from_local(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2":
        print(
            f"[LOCAL DINO] {model}：跳过 GitHub，直接读取本地缓存",
            flush=True,
        )
        repo_or_dir = str(DINO_REPO)
        kwargs["source"] = "local"

    return original_load(repo_or_dir, model, *args, **kwargs)

torch.hub.load = load_from_local

runpy.run_path(
    str(Path(__file__).with_name("run_sam3d_target_flask_e23_f000.py")),
    run_name="__main__",
)
