import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

import FisherRF_baseline as baseline


class ScoringTests(unittest.TestCase):
    def test_information_accumulation_and_candidate_independence(self):
        diagonals = [[0., 2.], [0., 3.], [1., 0.], [0., 10.]]
        calls = []
        def information(i):
            calls.append(i)
            return torch.tensor(diagonals[i])
        scores, _ = baseline.acquisition_scores([0, 1], [2, 3], information, iter([1, 2, 3]).__next__)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertAlmostEqual(scores[2], 1e6)
        self.assertAlmostEqual(scores[3], 10 / (5 + 1e-6), places=6)

    def test_actions_relocated_indices_aliases_and_signed_gain(self):
        runs = {name: dict(status="complete", train_lattice_indices=views)
                for name, views in dict(base=[8, 3], a=[3, 1, 8], b=[8, 3, 1],
                                        c=[8, 3, 7], duplicate=[3, 8]).items()}
        base, aliases = baseline.actions(runs)
        self.assertEqual(base, [8, 3])
        self.assertEqual(aliases, {1: ["a", "b"], 7: ["c"]})
        metrics = {k: {"averages": {"PSNR": v}} for k, v in dict(base=20, a=17, b=21, c=22).items()}
        report = baseline.evaluate({7: 1., 1: 1.}, aliases, metrics)
        self.assertEqual(report["selected_camera"], 1)
        self.assertEqual(report["selected_gain_db"], -1.)
        self.assertEqual(report["regret_db"], 3.)
        runs["bad"] = dict(status="complete", train_lattice_indices=[8, 5])
        with self.assertRaises(ValueError):
            baseline.actions(runs)

    @unittest.skipUnless(torch.cuda.is_available() and os.getenv("FISHERRF_RUN_ROOT"),
                         "needs CUDA, built FisherRF, and the four-object run")
    def test_real_crate_render_and_upstream_selector(self):
        from PIL import Image
        import numpy as np
        sys.path.insert(0, str(os.getenv("FISHERRF_ROOT") or baseline.DEFAULT_FISHERRF))
        from active.H_reg import HRegSelector
        from gaussian_renderer import render

        root = Path(os.environ["FISHERRF_RUN_ROOT"])
        context = Path("Crate/random_3_seed_77")
        folder = root / "reconstructions" / context / "base"
        transforms = root / "views/Crate/global_transforms"
        model, bg = baseline.load_base(folder, 30000)
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        def camera(manifest, i):
            return baseline.make_camera(manifest["frames"][i], manifest["camera_angle_x"], 512, 512)
        test_camera = camera(baseline.read_json(transforms / "transforms_test.json"), 0)
        with torch.no_grad():
            prediction = render(test_camera, model, pipe, bg)["render"].clamp(0, 1).cpu()
        reference = Image.open(folder / "test/ours_30000/renders/00000.png").convert("RGB")
        expected = torch.from_numpy(np.array(reference)).permute(2, 0, 1) / 255.
        self.assertLess((expected - prediction).abs().mean().item(), 2 / 255)

        result = baseline.read_json(root / "results" / context / "results.json")
        base, candidates = baseline.actions(result["reconstructions"])
        manifest = baseline.read_json(transforms / "transforms_train.json")
        cameras = {i: camera(manifest, i) for i in [*base, *candidates]}
        scores, _ = baseline.acquisition_scores(base, candidates, lambda i: baseline.diagonal(
            cameras[i], model, pipe, bg), baseline.cuda_clock)
        # HRegSelector uses the optimizer only to clear gradients and serialize capture().
        model.optimizer = torch.optim.SGD([model._xyz, model._features_dc, model._features_rest,
                                           model._scaling, model._rotation, model._opacity], lr=0.)
        scene = SimpleNamespace(get_candidate_set=lambda: list(candidates),
                                getTrainCameras=lambda: [cameras[i] for i in base],
                                getCandidateCameras=lambda: [cameras[i] for i in candidates])
        args = SimpleNamespace(seed=0, reg_lambda=1e-6, I_test=False, I_acq_reg=False,
                               filter_out_grad=["rotation"])
        chosen, = HRegSelector(args).nbvs(model, scene, 1, pipe, bg, lambda: False)
        torch.testing.assert_close(torch.tensor(scores[chosen]), torch.tensor(max(scores.values())),
                                   rtol=1e-5, atol=1e-7)
