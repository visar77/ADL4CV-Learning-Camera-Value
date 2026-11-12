"""CPU checks for renderer-owned warm-start state, schedules, and optimization."""

import copy
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastgs_adapter import initialized_model_class, adaptation_optimizer_step, train_adaptation
from fastgs_schedule import add_adaptation_arguments, configure_adaptation


def optimization(iterations=3000):
    return SimpleNamespace(
        iterations=iterations, position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_max_steps=30000, lowfeature_lr=0.0025, highfeature_lr=0.005,
        opacity_lr=0.025, scaling_lr=0.005, rotation_lr=0.001,
        densify_from_iter=500, densify_until_iter=15000, densification_interval=100)


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ply = self.root / "base.ply"
        self.ply.touch()
        self.parser = ArgumentParser()
        self.parser.add_argument("--iterations", type=int, default=3000)
        add_adaptation_arguments(self.parser)

    def configuration(self, *extra):
        args = self.parser.parse_args(["--init_ply", str(self.ply), *extra])
        return configure_adaptation(optimization(args.iterations), args)

    def test_schedule_is_local_and_scaled_without_mutating_scratch_defaults(self):
        opt = self.configuration()
        self.assertEqual(opt.position_lr_max_steps, 3000)
        self.assertEqual(opt.densify_until_iter, 1500)
        self.assertEqual(opt.densification_interval, 250)
        self.assertAlmostEqual(opt.position_lr_init, 0.000016)
        self.assertAlmostEqual(opt.opacity_lr, 0.0025)
        args = self.parser.parse_args([])
        scratch = optimization()
        before = copy.deepcopy(vars(scratch))
        configure_adaptation(scratch, args)
        self.assertEqual(vars(scratch), before)

    def test_densification_can_be_disabled_or_shortened(self):
        self.assertEqual(self.configuration("--adaptation_densify_until_iter", "0").densify_until_iter, 0)
        opt = self.configuration("--adaptation_densify_from_iter", "10",
                                 "--adaptation_densify_until_iter", "1000",
                                 "--adaptation_densification_interval", "50")
        self.assertEqual((opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval), (10, 1000, 50))
        self.assertEqual(self.configuration("--iterations", "2").densify_until_iter, 0)

    def test_rejects_invalid_adaptation_configuration(self):
        for extra in (["--adaptation_lr_scale", "nan"], ["--adaptation_lr_scale", "0"],
                      ["--adaptation_densification_interval", "0"],
                      ["--adaptation_densify_until_iter", "4000"],
                      ["--adaptation_densify_until_iter", "50"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.configuration(*extra)
        with self.assertRaises(FileNotFoundError):
            self.configuration("--init_ply", str(self.root / "missing.ply"))
        args = self.parser.parse_args(["--init_ply", str(self.ply)])
        args.start_checkpoint = "checkpoint.pth"
        with self.assertRaises(ValueError):
            configure_adaptation(optimization(), args)

    def test_initialization_hook_loads_ply_and_sets_training_buffers(self):
        class GaussianModel:
            def create_from_pcd(self, point_cloud, spatial_lr_scale):
                raise AssertionError("Random Gaussian/KNN initialization must be skipped")

            def load_ply(self, path):
                self.loaded_path = path
                self.get_xyz = torch.ones((7, 3))
                self.active_sh_degree = 3

        model_class = initialized_model_class(GaussianModel, self.ply)
        model = model_class()
        model.create_from_pcd(object(), 2.5)
        self.assertEqual(model.loaded_path, str(self.ply))
        self.assertEqual(model.spatial_lr_scale, 2.5)
        self.assertEqual(model.active_sh_degree, 3)
        self.assertEqual(tuple(model.max_radii2D.shape), (7,))
        self.assertEqual(model.max_radii2D.sum().item(), 0)
        self.assertIsNot(model_class.create_from_pcd, GaussianModel.create_from_pcd)

    def test_adaptation_updates_every_step_and_flushes_final_sh_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        sh_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        model = SimpleNamespace(optimizer=torch.optim.Adam([parameter], lr=0.1),
                                shoptimizer=torch.optim.Adam([sh_parameter], lr=0.1))
        parameter.grad = torch.ones_like(parameter)
        sh_parameter.grad = torch.ones_like(sh_parameter)
        adaptation_optimizer_step(model, 30001)
        self.assertLess(parameter.item(), 1.0)
        self.assertEqual(sh_parameter.item(), 1.0)
        self.assertIsNotNone(sh_parameter.grad)
        parameter.grad = torch.ones_like(parameter)
        adaptation_optimizer_step(model, 30002, final=True)
        self.assertLess(sh_parameter.item(), 1.0)
        self.assertIsNone(parameter.grad)
        self.assertIsNone(sh_parameter.grad)

    def test_training_loop_updates_final_step_before_save_and_uses_local_densification(self):
        models = []
        saved = []

        class Model:
            def __init__(self, *args):
                self.value = torch.nn.Parameter(torch.tensor(1.0))
                self.sh = torch.nn.Parameter(torch.tensor(0.0))
                self.get_xyz = torch.ones((1, 3))
                self.densify_and_prune_fastgs = Mock()
                self.add_densification_stats = Mock()
                self.update_learning_rate = Mock()
                models.append(self)

            def load_ply(self, path):
                self.initialization = path

            def training_setup(self, opt):
                self.optimizer = torch.optim.SGD([self.value], lr=0.1)
                self.shoptimizer = torch.optim.SGD([self.sh], lr=0.1)

        class Scene:
            cameras_extent = 1.0

            def __init__(self, dataset, model):
                self.gaussians = model
                model.create_from_pcd(None, self.cameras_extent)

            def getTrainCameras(self):
                return [SimpleNamespace(original_image=torch.zeros((3, 4, 4)))]

            def save(self, iteration):
                saved.append((iteration, self.gaussians.value.item()))

        class Progress:
            n = 0

            def update(self, count):
                self.n += count

            def set_postfix(self, **kwargs):
                pass

            def close(self):
                pass

        event = SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 1.0)
        cpu_torch = SimpleNamespace(
            tensor=lambda value, **kwargs: torch.tensor(value, dtype=kwargs.get("dtype")),
            float32=torch.float32, no_grad=torch.no_grad, maximum=torch.maximum,
            cuda=SimpleNamespace(Event=lambda **kwargs: event, synchronize=lambda: None))
        render = lambda camera, model, *args: {
            "render": model.value.expand((3, 4, 4)), "visibility_filter": torch.tensor([True]),
            "radii": torch.tensor([1.0]), "viewspace_points": model.value}
        runtime = SimpleNamespace(
            torch=cpu_torch, prepare_output_and_logger=lambda dataset: None,
            GaussianModel=Model, Scene=Scene, tqdm=lambda **kwargs: Progress(), render_fastgs=render,
            l1_loss=lambda image, gt: (image - gt).abs().mean(),
            fast_ssim=lambda image, gt: torch.tensor(1.0), sampling_cameras=lambda cameras: cameras,
            compute_gaussian_score_fastgs=lambda *args, **kwargs: (None, None), training_report=Mock())
        dataset = SimpleNamespace(model_path=str(self.root), sh_degree=3, white_background=False)
        opt = optimization(5)
        opt.optimizer_type = "default"
        opt.random_background = False
        opt.mult = 0.5
        opt.lambda_dssim = 0.0
        opt.densify_from_iter = 0
        opt.densify_until_iter = 4
        opt.densification_interval = 2
        args = SimpleNamespace(init_ply=str(self.ply), save_iterations=[5], test_iterations=[])
        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
            train_adaptation(dataset, opt, SimpleNamespace(), args, runtime)
        self.assertEqual(models[0].update_learning_rate.call_count, 5)
        models[0].densify_and_prune_fastgs.assert_called_once()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0], 5)
        self.assertAlmostEqual(saved[0][1], 0.5, places=5)
        runtime.training_report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
