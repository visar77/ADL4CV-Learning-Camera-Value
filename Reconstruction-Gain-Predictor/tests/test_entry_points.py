"""Integration boundaries affected by moving source files into a package."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest

from gain_predictor import GainEstimator, candidate_utility
from gain_predictor.artifacts import PROJECT_ROOT
from gain_predictor.checkpoints import load_frozen_backbone
from gain_predictor.data.manifest import load_manifest
from gain_predictor.feature_cache import implementation_identity
from gain_predictor.geometry import CAMERA_VERSION
from gain_predictor.models.lvsm import LVSM_CHECKPOINT_NAME
from tests.test_research_pipeline import TinyBackbone, example
import cache_features
import prepare_data


class EntryPointTests(unittest.TestCase):
    def test_commands_work_outside_the_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("prepare_data", "cache_features", "train", "compare", "evaluate", "predict"):
                with self.subTest(command=name):
                    result = subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / f"{name}.py"), "--help"],
                        cwd=directory, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
        self.assertTrue(callable(GainEstimator))
        self.assertTrue(callable(candidate_utility))

    def test_legacy_image_paths_still_resolve_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = replace(example(root), context_image_paths=("data/views/old.png",))
            manifest = root / "legacy.jsonl"
            manifest.write_text(json.dumps(sample.to_json()) + "\n")
            self.assertEqual(load_manifest(manifest)[0].image_paths,
                             (PROJECT_ROOT / "data/views/old.png",))
            manifest.write_text(json.dumps(replace(sample, path_base="manifest").to_json()) + "\n")
            self.assertEqual(load_manifest(manifest)[0].image_paths,
                             (root / "data/views/old.png",))

    def test_default_backbone_weights_remain_at_project_root(self):
        expected_path = PROJECT_ROOT / LVSM_CHECKPOINT_NAME
        factory = Mock(side_effect=lambda *args: TinyBackbone())
        spec = {"name": "tiny", "checkpoint_sha256": "test", "image_size": 8,
                "latent_tokens": 4, "precision": "fp32", "camera_version": CAMERA_VERSION,
                "implementation": implementation_identity()}
        with patch.dict("gain_predictor.checkpoints.BACKBONES", {"tiny": factory}), \
                patch("gain_predictor.checkpoints.file_sha256", return_value="test"):
            load_frozen_backbone({"backbone_spec": spec}, device="cpu")
        self.assertEqual(factory.call_args.args[0], expected_path)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            sample = example(root)
            manifest = root / "labels.jsonl"
            manifest.write_text(json.dumps(sample.to_json()) + "\n")
            with patch.dict(cache_features.BACKBONES, {"tiny": factory}), \
                    patch.object(cache_features, "file_sha256", return_value="test"), \
                    patch.object(cache_features, "extract_features", return_value=root / "features.pt") as extract:
                cache_features.main(["--dataset", str(manifest), "--output", str(root / "cache"),
                                     "--backbone", "tiny", "--device", "cpu"])
            self.assertEqual(extract.call_args.args[-1], expected_path)

    def test_prepare_command_upgrades_calibration_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            sample = replace(example(root, gain=-0.1), context_intrinsics=(), candidate_intrinsics=None)
            (root / "transforms_train.json").write_text(json.dumps({
                "camera_convention": "opencv", "camera_angle_x": 0.8,
                "frames": [{"file_path": "view0.png", "transform_matrix":
                            [sample.candidate_pose[i:i + 4] for i in range(0, 16, 4)]}],
            }))
            manifest = root / "legacy.jsonl"
            original = json.dumps(sample.to_json()) + "\n"
            manifest.write_text(original)
            prepare_data.main(["--upgrade", str(manifest)])
            upgraded = load_manifest(manifest)[0]
            self.assertEqual(upgraded.psnr_gain, -0.1)
            self.assertEqual(upgraded.context_poses, sample.context_poses)
            self.assertTrue(upgraded.context_intrinsics)
            self.assertTrue(upgraded.candidate_intrinsics)
            self.assertEqual(manifest.with_suffix(".jsonl.bak").read_text(), original)


if __name__ == "__main__":
    unittest.main()
