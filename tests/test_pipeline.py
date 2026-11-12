import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import collate_gain_samples, load_manifest
from model import ContextPoseCNN, _normalize_lvsm_rgb


class PipelineTests(unittest.TestCase):
    def test_jsonl_loader_and_model_owned_image_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "view.png"
            Image.new("RGB", (10, 12), color=(255, 0, 0)).save(image)
            manifest = root / "dataset.jsonl"
            record = {
                "scene_id": "scene",
                "experiment_id": "experiment",
                "candidate_id": "candidate",
                "context_image_paths": [str(image)],
                "context_poses": [[1.0] * 16],
                "candidate_pose": [1.0] * 16,
                "psnr_gain": 0.1,
                "base_psnr": 20.0,
                "candidate_psnr": 22.0,
                "source_results": "results.json",
            }
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            samples = load_manifest(manifest)
            batch = collate_gain_samples(samples)
            prediction = ContextPoseCNN(image_size=8)(
                batch["images"],
                batch["context_poses"],
                batch["candidate_pose"],
                batch["mask"],
            )
            self.assertEqual(batch["images"].dtype, torch.uint8)
            self.assertEqual(tuple(prediction.shape), (1, 1))

    def test_collate_resizes_images_to_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "view.png"
            Image.new("RGB", (10, 12), color=(255, 0, 0)).save(image)
            manifest = root / "dataset.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "scene_id": "scene",
                        "experiment_id": "experiment",
                        "candidate_id": "candidate",
                        "context_image_paths": [str(image)],
                        "context_poses": [[1.0] * 16],
                        "candidate_pose": [1.0] * 16,
                        "psnr_gain": 0.1,
                        "source_results": "results.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            batch = collate_gain_samples(load_manifest(manifest), image_size=8)
            self.assertEqual(tuple(batch["images"].shape), (1, 1, 3, 8, 8))

    def test_lvsm_input_rgb_is_normalized_once(self) -> None:
        images = torch.tensor([[[[[1.0]], [[0.0]], [[0.5]]]]])
        normalized = _normalize_lvsm_rgb(images)
        self.assertAlmostEqual(float(normalized[0, 0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(normalized[0, 0, 1, 0, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()
