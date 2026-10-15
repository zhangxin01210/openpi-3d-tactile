#!/usr/bin/env python3
"""Fail early on legacy Torch CUDA incompatibility before importing whole tasks."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "python": sys.version, "stages": []}

    def stage(name, **values):
        report["stages"].append({"stage": name, **values})
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(name, values, flush=True)

    try:
        # Gym must be imported before torch.
        from isaacgym import gymapi
        import torch
        stage("imports", torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
              torch_arch_list=torch.cuda.get_arch_list())
        stage("cuda_elementwise_begin")
        x = torch.arange(9, dtype=torch.float32, device="cuda:0").reshape(3,3)
        y = torch.sin(x) + x.square()
        torch.cuda.synchronize()
        if not torch.allclose(y.cpu(), torch.sin(x.cpu())+x.cpu().square()):
            raise RuntimeError("GPU elementwise numerical mismatch")
        stage("cuda_elementwise_pass")
        a = x @ x.T + torch.eye(3, device="cuda:0")
        result = torch.linalg.inv(a)
        torch.cuda.synchronize()
        if not torch.allclose(result.cpu(), torch.linalg.inv(a.cpu()), atol=1e-4):
            raise RuntimeError("GPU inverse numerical mismatch")
        stage("cuda_linear_algebra_pass")
        from isaacgym import gymtorch
        stage("gymtorch_import_pass")
        report["status"] = "passed"
        stage("completed")
    except Exception as exc:
        report["status"] = "failed"
        stage("exception", type=type(exc).__name__, message=str(exc))
        raise


if __name__ == "__main__":
    main()
