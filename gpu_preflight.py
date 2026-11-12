"""Verify the selected GPU environment before launching a reconstruction benchmark."""

import argparse
import importlib
import inspect
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def check_environment(fastgs_dir):
    fastgs_dir = Path(fastgs_dir).resolve()
    sys.path.insert(0, str(fastgs_dir))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; activate the GPU FastGS environment or use --python")
    torch.cuda.set_device(0)
    fastgs = importlib.import_module("train")
    if Path(fastgs.__file__).resolve() != fastgs_dir / "train.py":
        raise RuntimeError("Imported train module is not from the selected FastGS checkout")
    for name in ("Scene", "GaussianModel", "render_fastgs", "fast_ssim", "compute_gaussian_score_fastgs",
                 "sampling_cameras", "training_report", "prepare_output_and_logger", "safe_state"):
        if not hasattr(fastgs, name):
            raise RuntimeError(f"Incompatible FastGS checkout: train.{name} is unavailable")
    parameters = inspect.signature(fastgs.GaussianModel.densify_and_prune_fastgs).parameters
    if not {"importance_score", "pruning_score", "radii"}.issubset(parameters):
        raise RuntimeError("FastGS densification API differs from the renderer adapter")
    from simple_knn._C import distCUDA2
    points = torch.rand((32, 3), device="cuda")
    if not torch.isfinite(distCUDA2(points)).all():
        raise RuntimeError("simple-knn CUDA smoke check produced nonfinite distances")
    image = torch.rand((1, 3, 32, 32), device="cuda", requires_grad=True)
    fastgs.fast_ssim(image, image.detach()).backward()
    torch.cuda.synchronize()
    return {"gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(), "python_executable": sys.executable,
            "hostname": platform.node(), "fastgs_dir": str(fastgs_dir),
            "checks": ["CUDA", "FastGS imports/API", "simple-knn CUDA kernel", "fused SSIM forward/backward"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        environment = check_environment(args.fastgs_dir)
    except (ImportError, RuntimeError, OSError) as error:
        parser.exit(1, f"GPU preflight failed: {error}\nCheck the selected Python environment and the device's FastGS extension builds.\n")
    text = json.dumps(environment, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
