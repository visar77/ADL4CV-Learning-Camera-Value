"""Renderer-owned adaptation entry point using an unmodified FastGS checkout.

The training loop lives here so initialization and adaptation policies do not
require patches, monkey-patches, or a feature branch in the FastGS repository.
CUDA/native dependencies are imported only when training starts.
"""

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

from fastgs_schedule import add_adaptation_arguments, configure_adaptation


ROOT = Path(__file__).resolve().parent


def initialized_model_class(gaussian_model_class, initial_ply):
    """Use Scene's initialization hook to avoid random Gaussian/KNN creation."""
    class InitializedGaussianModel(gaussian_model_class):
        def create_from_pcd(self, point_cloud, spatial_lr_scale):
            print(f"Initializing Gaussians from {initial_ply}", flush=True)
            self.load_ply(str(initial_ply))
            self.spatial_lr_scale = spatial_lr_scale
            self.max_radii2D = self.get_xyz.new_zeros(self.get_xyz.shape[0])

    return InitializedGaussianModel


def adaptation_optimizer_step(gaussians, iteration, final=False):
    """Keep adaptation updates independent of FastGS's late scratch schedule."""
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    if iteration % 16 == 0 or final:
        gaussians.shoptimizer.step()
        gaussians.shoptimizer.zero_grad(set_to_none=True)


def train_adaptation(dataset, opt, pipe, args, fastgs):
    """Adapt one base independently, using the existing FastGS CUDA operations."""
    torch = fastgs.torch
    start = time.perf_counter()
    writer = fastgs.prepare_output_and_logger(dataset)
    model_class = initialized_model_class(fastgs.GaussianModel, args.init_ply)
    gaussians = model_class(dataset.sh_degree, opt.optimizer_type)
    scene = fastgs.Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.],
                              dtype=torch.float32, device="cuda")
    bg = torch.rand((3,), device="cuda") if opt.random_background else background
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError("Adaptation requires at least one training camera")
    stack = []
    progress = fastgs.tqdm(total=opt.iterations, desc="Adapting base")
    ema_loss = 0.0
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    try:
        for iteration in range(1, opt.iterations + 1):
            event_start.record()
            gaussians.update_learning_rate(iteration)
            # load_ply already enables the full SH degree; no degree reset.
            if not stack:
                stack = cameras.copy()
            camera = stack.pop(random.randrange(len(stack)))
            package = fastgs.render_fastgs(camera, gaussians, pipe, bg, opt.mult)
            image = package["render"]
            gt = camera.original_image.cuda()
            l1 = fastgs.l1_loss(image, gt)
            ssim = fastgs.fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * l1 + opt.lambda_dssim * (1.0 - ssim)
            loss.backward()
            with torch.no_grad():
                if iteration < opt.densify_until_iter:
                    visible = package["visibility_filter"]
                    radii = package["radii"]
                    gaussians.max_radii2D[visible] = torch.maximum(
                        gaussians.max_radii2D[visible], radii[visible])
                    gaussians.add_densification_stats(package["viewspace_points"], visible)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        selected = fastgs.sampling_cameras(cameras.copy())
                        importance, pruning = fastgs.compute_gaussian_score_fastgs(
                            selected, gaussians, pipe, bg, opt, DENSIFY=True)
                        gaussians.densify_and_prune_fastgs(
                            max_screen_size=None, min_opacity=0.005,
                            extent=scene.cameras_extent, radii=radii, args=opt,
                            importance_score=importance, pruning_score=pruning)
                # Do not reset trained opacity or invoke late scratch pruning.
                adaptation_optimizer_step(gaussians, iteration, final=iteration == opt.iterations)
                if iteration in args.save_iterations:
                    scene.save(iteration)
                event_end.record()
                torch.cuda.synchronize()
                elapsed_ms = event_start.elapsed_time(event_end)
                if iteration % 100 == 0 or iteration in args.test_iterations:
                    fastgs.training_report(writer, iteration, l1, loss, fastgs.l1_loss,
                                           elapsed_ms, args.test_iterations, scene,
                                           fastgs.render_fastgs, (pipe, background, opt.mult))
                if iteration % 10 == 0 or iteration == opt.iterations:
                    ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                    progress.set_postfix(Loss=f"{ema_loss:.7f}")
                    progress.update(iteration - progress.n)
        summary = {"iterations": opt.iterations,
                   "gaussian_count": gaussians.get_xyz.shape[0],
                   "wall_time_seconds": time.perf_counter() - start,
                   "adaptation": vars(opt)}
        (Path(dataset.model_path) / "adaptation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Adaptation complete: {summary['wall_time_seconds']:.2f}s", flush=True)
    finally:
        progress.close()
        if writer:
            writer.close()


def main(argv=None):
    # Locate FastGS's Python modules before loading its lightweight argument
    # definitions. --help does not load CUDA extensions.
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    location, _ = preliminary.parse_known_args(argv)
    fastgs_dir = location.fastgs_dir.expanduser().resolve()
    if not (fastgs_dir / "train.py").is_file():
        preliminary.error(f"FastGS checkout not found: {fastgs_dir}")
    sys.path.insert(0, str(fastgs_dir))
    from arguments import ModelParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    add_adaptation_arguments(parser)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.init_ply:
        parser.error("init_ply is required; use FastGS/train.py for scratch training")
    if args.optimizer_type != "default":
        parser.error("The adaptation adapter currently supports optimizer_type=default")
    args.init_ply = str(Path(args.init_ply).expanduser().resolve())
    try:
        opt = configure_adaptation(optimization.extract(args), args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    fastgs = importlib.import_module("train")
    fastgs.safe_state(args.quiet)
    train_adaptation(model.extract(args), opt, pipeline.extract(args), args, fastgs)


if __name__ == "__main__":
    main()
