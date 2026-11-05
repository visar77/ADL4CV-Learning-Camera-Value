"""Render FastGS test views and store per-view metrics in results.json."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
FASTGS_DIR = ROOT.parent / "FastGS"
if str(FASTGS_DIR) not in sys.path:
    sys.path.insert(0, str(FASTGS_DIR))


def evaluate(reconstruction_dir, result_path, iteration):
    with result_path.open(encoding="utf-8") as file:
        result = json.load(file)
    dataset_dir = Path(result["dataset_path"])
    global_manifest = dataset_dir / "global_transforms" / "transforms_train.json"
    with global_manifest.open(encoding="utf-8") as file:
        global_train = json.load(file)
    train_frames = [
        global_train["frames"][index] for index in result["train_lattice_indices"]
    ]
    train_manifest = dataset_dir / "transforms_train.json"
    with train_manifest.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "camera_angle_x": global_train["camera_angle_x"],
                "frames": train_frames,
            },
            file,
            indent=2,
        )

    try:
        subprocess.run(
            [
                sys.executable,
                str(FASTGS_DIR / "render.py"),
                "--model_path",
                str(reconstruction_dir),
                "--iteration",
                str(iteration),
                "--skip_train",
            ],
            cwd=FASTGS_DIR,
            check=True,
        )

        from lpipsPyTorch import lpips
        from utils.image_utils import psnr
        from utils.loss_utils import ssim

        test_dir = reconstruction_dir / "test" / f"ours_{iteration}"
        render_dir = test_dir / "renders"
        ground_truth_dir = test_dir / "gt"
        per_view = {}
        for render_path in sorted(render_dir.glob("*.png")):
            gt_path = ground_truth_dir / render_path.name
            render = tf.to_tensor(Image.open(render_path)).unsqueeze(0)[:, :3].cuda()
            ground_truth = tf.to_tensor(Image.open(gt_path)).unsqueeze(0)[:, :3].cuda()
            per_view[render_path.stem] = {
                "PSNR": float(psnr(render, ground_truth)),
                "SSIM": float(ssim(render, ground_truth)),
                "LPIPS": float(lpips(render, ground_truth, net_type="vgg")),
            }

        averages = {
            metric: sum(view[metric] for view in per_view.values()) / len(per_view)
            for metric in ("PSNR", "SSIM", "LPIPS")
        }
        result["metrics"] = {"averages": averages, "per_view": per_view}
        result["metrics_iteration"] = iteration
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
    finally:
        train_manifest.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", default="gamecube")
    parser.add_argument("--experiment-id")
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    result_root = ROOT / "results" / args.object_id
    paths = (
        [result_root / args.experiment_id / "results.json"]
        if args.experiment_id
        else sorted(result_root.glob("*/results.json"))
    )
    for result_path in paths:
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
        evaluate(
            Path(result["reconstruction_path"]),
            result_path,
            args.iteration,
        )


if __name__ == "__main__":
    main()
