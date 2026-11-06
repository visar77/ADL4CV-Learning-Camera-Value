import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
import train
from model import LVSMBackbone


class MockGainPredictor(nn.Module):
    """Small injectable model used to exercise the training orchestration."""

    def __init__(self, input_dim: int = len(train.FEATURE_NAMES), **_: object) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(train.FEATURE_NAMES):
            raise AssertionError(f"unexpected feature shape: {tuple(features.shape)}")
        return self.output(features)


class MockWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, dict[str, float], int]] = []
        self.closed = False

    def add_scalars(self, tag: str, scalars: dict[str, float], step: int) -> None:
        self.scalars.append((tag, scalars, step))

    def close(self) -> None:
        self.closed = True


def write_result(
    path: Path,
    object_id: str,
    experiment_id: str,
    psnr: float,
    indices: list[int] | None = None,
    lpips: float = 0.2,
) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "object_id": object_id,
                "experiment_id": experiment_id,
                "train_lattice_indices": indices or [0, 3, 8, 12],
                "metrics": {"averages": {"PSNR": psnr, "SSIM": 0.8, "LPIPS": lpips}},
            }
        ),
        encoding="utf-8",
    )


class TrainPipelineTests(unittest.TestCase):
    def test_lvsm_backbone_is_frozen_and_head_can_load_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "backbone.pt"
            # Build the fixture state without invoking a download.
            raw = {
                "encoder.0.weight": torch.randn(256, len(train.FEATURE_NAMES)),
                "encoder.0.bias": torch.randn(256),
                "encoder.2.weight": torch.randn(128, 256),
                "encoder.2.bias": torch.randn(128),
            }
            torch.save(raw, checkpoint)
            loaded = LVSMBackbone(
                len(train.FEATURE_NAMES), checkpoint=checkpoint, url=None
            )

            self.assertTrue(
                all(not parameter.requires_grad for parameter in loaded.parameters())
            )
            self.assertEqual(
                loaded(torch.zeros(1, len(train.FEATURE_NAMES))).shape, (1, 128)
            )

    def test_record_loader_skips_unlabelled_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_result(
                root / "object-a" / "exp-a" / "results.json", "a", "exp-a", 25.0
            )
            incomplete = root / "object-b" / "exp-b" / "results.json"
            incomplete.parent.mkdir(parents=True)
            incomplete.write_text(
                json.dumps({"object_id": "b", "train_lattice_indices": [1]}),
                encoding="utf-8",
            )

            records = train.load_records(root, "PSNR")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].object_id, "a")
            self.assertEqual(records[0].target, 25.0)

    def test_gain_labels_are_derived_from_nested_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_result(
                root / "base" / "results.json",
                "object-a",
                "base",
                20.0,
                indices=[0, 3],
                lpips=0.30,
            )
            write_result(
                root / "expanded" / "results.json",
                "object-a",
                "expanded",
                22.5,
                indices=[0, 3, 8],
                lpips=0.20,
            )

            psnr_gain = train.load_records(root, "PSNR", mode="gain")
            lpips_gain = train.load_records(root, "LPIPS", mode="gain")

            self.assertEqual(len(psnr_gain), 1)
            self.assertEqual(psnr_gain[0].train_lattice_indices, (8,))
            self.assertEqual(psnr_gain[0].target, 2.5 / 20.0)
            self.assertAlmostEqual(lpips_gain[0].target, 0.1 / 0.3)

    def test_record_selection_supports_sample_subset_and_object(self) -> None:
        records = [
            train.ResultRecord("a", "one", (1,), 1.0),
            train.ResultRecord("a", "two", (2,), 2.0),
            train.ResultRecord("b", "three", (3,), 3.0),
        ]

        self.assertEqual(len(train.select_records(records, object_id="a")), 2)
        self.assertEqual(
            train.select_records(records, sample_index=1)[0].experiment_id, "two"
        )
        self.assertEqual(len(train.select_records(records, max_samples=2)), 2)

    def test_mock_model_and_input_run_training_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for object_index in range(4):
                for experiment_index in range(2):
                    write_result(
                        root
                        / "results"
                        / f"object-{object_index}"
                        / f"exp-{experiment_index}"
                        / "results.json",
                        f"object-{object_index}",
                        f"exp-{experiment_index}",
                        20.0 + object_index + experiment_index,
                    )

            output = root / "checkpoint.pt"
            log_directory = root / "runs"
            writer = MockWriter()
            arguments = Namespace(
                data=root / "results",
                target="PSNR",
                mode="quality",
                output=output,
                log_dir=log_directory,
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                validation_fraction=0.25,
                seed=42,
                device="cpu",
                checkpoint_every=1,
                backbone_checkpoint=None,
                backbone_url=None,
            )

            with (
                patch.object(train, "GainPredictor", MockGainPredictor),
                patch.object(train, "_writer", return_value=writer),
            ):
                checkpoint_path = train.train(arguments)

            self.assertEqual(checkpoint_path, output)
            self.assertTrue(output.is_file())
            checkpoint = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["target_metric"], "PSNR")
            self.assertEqual(tuple(checkpoint["feature_names"]), train.FEATURE_NAMES)
            self.assertEqual(len(writer.scalars), 2)
            self.assertTrue(output.with_name("checkpoint.epoch-1.pt").is_file())
            self.assertTrue(output.with_name("checkpoint.epoch-2.pt").is_file())
            self.assertEqual(
                {name for name in writer.scalars[0][1]}, {"train", "validation"}
            )
            self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
