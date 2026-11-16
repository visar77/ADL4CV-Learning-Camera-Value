import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dataset import load_manifest, relativize_manifest, collate_gain_samples
from experiments import make_splits, restored_split
from features import FeatureStore, extract_features, file_sha256
from paths import relative_path
from predictor import GainEstimator, _intrinsics
from runtime import load_run, score_samples
from train import parse_args, train
from test_research_pipeline import TinyBackbone, example


class PortabilityAndInferenceTests(unittest.TestCase):
    def test_manifest_cache_run_and_direct_inference_after_relocation(self):
        import inference
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            parent = Path(directory)
            original = parent / "original"
            original.mkdir()
            samples = [example(original, 0, q, .1*q) for q in range(3)]
            manifest = original / "data" / "labels.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(''.join(json.dumps(s.to_json())+'\n' for s in samples))
            before_labels = [s.psnr_gain for s in samples]
            relativize_manifest(manifest)
            raw_records = [json.loads(line) for line in manifest.read_text().splitlines()]
            for row in raw_records:
                self.assertFalse(Path(row["source_results"]).is_absolute())
                self.assertTrue(all(not Path(p).is_absolute() for p in row["context_image_paths"]))
            samples = load_manifest(manifest)
            self.assertEqual([s.psnr_gain for s in samples], before_labels)
            original_splits = make_splits(samples, mode="overfit")[1]
            weights = original / "weights.pt"
            weights.write_bytes(b"test backbone identity")
            cache = extract_features(samples, original/"cache", TinyBackbone,
                                     {"name":"tiny", "latent_tokens":4, "checkpoint_sha256":file_sha256(weights)},
                                     image_size=8, backbone_checkpoint=weights)
            config = parse_args(["--dataset",str(manifest),"--features",str(cache),"--model","cached",
                                 "--mode","overfit","--epochs","5","--output",str(original/"run"/"head.pt"),
                                 "--log-dir",str(original/"run"/"logs")])
            train(config)
            persisted = torch.load(config.output, weights_only=True)
            for name in ("dataset_path", "features_path", "backbone_checkpoint"):
                self.assertFalse(Path(persisted[name]).is_absolute())
            ck, head = load_run(config.output)
            expected = score_samples(ck, head, samples)
            moved = parent/"moved"
            shutil.copytree(original, moved)
            shutil.rmtree(original)
            unrelated = parent/"unrelated"; unrelated.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(unrelated)
                ck, head = load_run(moved/"run"/"head.pt")
                samples = load_manifest(Path(ck["dataset_path"]))
                self.assertEqual(make_splits(samples, mode="overfit")[1], original_splits)
                self.assertEqual(len(restored_split(samples, ck, "train")), 3)
                self.assertTrue(FeatureStore(ck["features_path"]).backbone_path.is_file())
                torch.testing.assert_close(score_samples(ck, head, samples), expected)
                poses = [s.candidate_pose for s in samples]
                first = samples[0]
                estimator = GainEstimator(moved/"run"/"head.pt", backbone=TinyBackbone(), candidate_batch_size=2)
                direct = estimator.predict(first.image_paths, first.context_poses, poses,
                                           context_intrinsics=first.context_intrinsics,
                                           candidate_intrinsics=[s.candidate_intrinsics for s in samples],
                                           camera_convention="opencv")
                torch.testing.assert_close(direct, expected.flatten())
                # Live inference must not read the training manifest or feature cache.
                Path(ck["dataset_path"]).unlink()
                shutil.rmtree(moved/"cache")
                request = {"camera_convention":"opencv", "intrinsics":[1.,1.,.5,.5],
                           "context":[{"image":relative_path(first.image_paths[0],moved),
                                       "pose":first.context_poses[0]}],
                           "candidates":[{"id":str(i),"pose":s.candidate_pose} for i,s in enumerate(samples)]}
                request_path = moved/"request.json"
                request_path.write_text(json.dumps(request))
                with patch.dict("runtime.BACKBONES", {"tiny":lambda *args: TinyBackbone()}):
                    rows = inference.main(["--checkpoint",str(moved/"run"/"head.pt"),
                                           "--input",str(request_path),"--candidate-batch-size","2"])
                torch.testing.assert_close(torch.tensor([row["predicted_gain"] for row in rows]), expected.flatten())
                self.assertTrue(all("ground_truth_relative_gain" not in row for row in rows))
            finally:
                os.chdir(old_cwd)

    def test_missing_migration_target_explains_optional_step(self):
        import preprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "table.jsonl"
            existing.write_text("existing manifest must not be modified")
            missing = root / "table_v2.jsonl"
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                preprocess.main(["--relativize", str(missing)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("need no migration", error.getvalue())
            self.assertIn(str(existing), error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())
            self.assertFalse(missing.exists())
            self.assertEqual(existing.read_text(), "existing manifest must not be modified")

    def test_pixel_intrinsics_and_image_input_forms(self):
        sizes = [(12,16)]
        normalized = _intrinsics([[12.,0.,6.],[0.,16.,8.],[0.,0.,1.]], sizes)
        torch.testing.assert_close(normalized, torch.tensor([[1.,1.,.5,.5]]))
        with self.assertRaises(ValueError):
            _intrinsics(None, sizes)
        with self.assertRaises(ValueError):
            _intrinsics(None, sizes, fov=180)
        from predictor import _context_images
        import numpy as np
        pixels = np.full((16,12,3), 64, dtype=np.uint8)
        a, _ = _context_images([pixels], 8)
        b, _ = _context_images(torch.from_numpy(pixels).permute(2,0,1), 8)
        c, _ = _context_images([pixels.astype(np.float32)/255], 8)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a,c)


if __name__ == "__main__":
    unittest.main()
