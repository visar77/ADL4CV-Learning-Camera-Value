"""Opt-in CUDA integration test: FASTGS_RUN_GPU_TESTS=1 python -m unittest discover -s tests -p test_fastgs_adapter_gpu.py -v"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RENDERER_ROOT = Path(__file__).resolve().parents[1]
FASTGS_ROOT = Path(os.environ.get("FASTGS_DIR", RENDERER_ROOT.parent / "FastGS")).resolve()


@unittest.skipUnless(os.environ.get("FASTGS_RUN_GPU_TESTS") == "1", "Set FASTGS_RUN_GPU_TESTS=1 on a CUDA device")
class GPUAdaptationTests(unittest.TestCase):
    def test_matched_suite_trains_evaluates_and_compares_all_three_initializations(self):
        import numpy as np
        import torch
        from PIL import Image
        sys.path.insert(0, str(FASTGS_ROOT))
        from scene.dataset_readers import storePly
        sys.path.insert(0, str(RENDERER_ROOT))
        from sampling import fibonacci_lattice

        self.assertTrue(torch.cuda.is_available(), "The GPU smoke test requires CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "views" / "smoke object"
            manifests = source / "global_transforms"
            manifests.mkdir(parents=True)
            all_frames = []
            for i, direction in enumerate(fibonacci_lattice(12)):
                back = np.array(direction)
                right = np.cross(np.array([0., 1., 0.]), back)
                right /= np.linalg.norm(right)
                matrix = np.eye(4)
                matrix[:3, :3] = np.stack((right, np.cross(back, right), back), axis=1)
                matrix[:3, 3] = 2 * back
                Image.new("RGB", (64, 64), (80 + i * 3, 100, 120)).save(source / f"{i}.png")
                all_frames.append({"file_path": str(i), "transform_matrix": matrix.tolist()})
            for split, frames in (("train", all_frames[:8]), ("test", all_frames[8:])):
                (manifests / f"transforms_{split}.json").write_text(json.dumps({"camera_angle_x": 0.8, "frames": frames}))
            xyz = np.random.default_rng(77).uniform(-0.4, 0.4, (64, 3)).astype(np.float32)
            storePly(str(source / "points3d.ply"), xyz, np.full((64, 3), 100, dtype=np.uint8))
            experiments = root / "experiments"
            experiments.mkdir()
            (experiments / "context.json").write_text(json.dumps({
                "id": "context", "train": {"strategy": "furthest", "count": 2, "seed": 77}}))
            configuration = {
                "schema_version": 1, "reference": "fresh", "base_iterations": 8,
                "candidates": [{"strategy": "blindspot"}, {"strategy": "random", "count": 1}],
                "adaptation_defaults": {"densify_from_iter": 0, "densify_until_iter": 4,
                                         "densification_interval": 2},
                "methods": [{"id": "fresh", "initialization": "scratch", "iterations": 8},
                            {"id": "warm", "initialization": "base", "iterations": 8},
                            {"id": "extra", "initialization": "base", "iterations": 8, "extra_gaussians": 16}]}
            config = root / "config.json"
            config.write_text(json.dumps(configuration))
            output = root / "benchmark"
            subprocess.run([
                sys.executable, str(RENDERER_ROOT / "benchmark_reconstructions.py"),
                "--config", str(config), "--output-root", str(output),
                "--views-root", str(source.parent), "--experiments-dir", str(experiments),
                "--fastgs-dir", str(FASTGS_ROOT),
            ], cwd=RENDERER_ROOT, check=True, timeout=600)
            report = json.loads((output / "report/comparison.json").read_text())
            self.assertEqual(set(report["overall"]), {"warm", "extra"})
            for item in report["comparisons"]:
                self.assertGreater(len(item["rows"]), 0)
                self.assertGreater(item["summary"]["reference_candidate_seconds"], 0)
                self.assertGreater(item["summary"]["method_candidate_seconds"], 0)
                self.assertTrue(all(row["timing_environment_verified"] for row in item["rows"]))
            extra = json.loads((output / "results/extra/smoke object/context/results.json").read_text())
            preparation = extra["initialization"]["preparation"]
            self.assertEqual(preparation["extra_gaussians"], 16)
            self.assertEqual(preparation["initial_gaussians"], preparation["original_gaussians"] + 16)

    def test_ply_initialization_training_densification_and_final_save(self):
        import numpy as np
        import torch
        from PIL import Image
        from plyfile import PlyData
        sys.path.insert(0, str(FASTGS_ROOT))
        from scene import GaussianModel
        from scene.dataset_readers import storePly

        self.assertTrue(torch.cuda.is_available(), "The GPU smoke test requires CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "views"
            source.mkdir()
            frames = []
            for i, eye in enumerate(((0, 0, 2), (2, 0, 0), (0, 2, 0))):
                eye = np.array(eye, dtype=float)
                back = eye / np.linalg.norm(eye)
                up = np.array((0, 1, 0) if i != 2 else (0, 0, 1), dtype=float)
                right = np.cross(up, back)
                right /= np.linalg.norm(right)
                matrix = np.eye(4)
                matrix[:3, :3] = np.stack((right, np.cross(back, right), back), axis=1)
                matrix[:3, 3] = eye
                Image.new("RGB", (32, 32), (80 + i * 20, 100, 120)).save(source / f"{i}.png")
                frames.append({"file_path": str(i), "transform_matrix": matrix.tolist()})
            for name, selected in (("train", frames), ("test", frames[:1])):
                (source / f"transforms_{name}.json").write_text(json.dumps({"camera_angle_x": 0.8, "frames": selected}))
            xyz = np.random.default_rng(77).uniform(-0.4, 0.4, (64, 3)).astype(np.float32)
            storePly(str(source / "points3d.ply"), xyz, np.full((64, 3), 100, dtype=np.uint8))
            model = GaussianModel(3)
            for name, values in {
                "_xyz": torch.from_numpy(xyz), "_features_dc": torch.zeros(64, 1, 3),
                "_features_rest": torch.zeros(64, 15, 3), "_opacity": torch.zeros(64, 1),
                "_scaling": torch.full((64, 3), -2.5),
                "_rotation": torch.tensor([[1., 0., 0., 0.]]).repeat(64, 1),
            }.items():
                setattr(model, name, torch.nn.Parameter(values))
            initial = root / "initial.ply"
            model.save_ply(str(initial))
            output = root / "adapted"
            subprocess.run([
                sys.executable, str(RENDERER_ROOT / "fastgs_adapter.py"),
                "--fastgs-dir", str(FASTGS_ROOT),
                "--source_path", str(source), "--model_path", str(output), "--eval",
                "--init_ply", str(initial), "--iterations", "8", "--test_iterations",
                "--save_iterations", "8", "--adaptation_densify_from_iter", "0",
                "--adaptation_densify_until_iter", "6", "--adaptation_densification_interval", "2",
            ], cwd=FASTGS_ROOT, check=True, timeout=180)
            final = output / "point_cloud/iteration_8/point_cloud.ply"
            self.assertTrue(final.is_file())
            vertices = PlyData.read(final)["vertex"].data
            self.assertGreater(len(vertices), 0)
            self.assertTrue(all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names))
            self.assertTrue(np.any(vertices["f_dc_0"] != 0), "Adaptation must update the loaded Gaussian features")
            self.assertTrue(np.all(PlyData.read(initial)["vertex"].data["f_dc_0"] == 0),
                            "The shared initialization must remain unchanged")


if __name__ == "__main__":
    unittest.main()
