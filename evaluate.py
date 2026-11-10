"""Evaluate a checkpoint on a chosen scene split without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import GainDataset, collate_gain_samples, load_manifest
from model import ContextPoseCNN, LVSMGainPredictor
from train import _split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--lvsm-checkpoint", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    image_size = int(checkpoint.get("image_size", 128))
    model_type = checkpoint.get("model_type")
    if model_type == "toy":
        model = ContextPoseCNN(image_size=image_size)
    elif model_type == "lvsm":
        if args.lvsm_checkpoint is None:
            raise ValueError("--lvsm-checkpoint is required for an LVSM checkpoint.")
        model = LVSMGainPredictor(args.lvsm_checkpoint, image_size=image_size)
    else:
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}")
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    samples = load_manifest(args.dataset)
    stored = checkpoint.get("split")
    if stored:
        assignments = {
            name: set(stored.get(name, [])) for name in ("train", "validation", "test")
        }
        groups = tuple(
            [sample for sample in samples if sample.scene_id in assignments[name]]
            for name in ("train", "validation", "test")
        )
    else:
        groups = _split(
            samples,
            checkpoint.get("seed", 42),
            args.split_file,
            args.validation_fraction,
            args.test_fraction,
        )
    split_samples = {"train": groups[0], "validation": groups[1], "test": groups[2]}[
        args.split
    ]
    loader = DataLoader(
        GainDataset(split_samples),
        batch_size=8,
        collate_fn=lambda b: collate_gain_samples(b, image_size),
    )
    criterion = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["images"].to(args.device),
                batch["context_poses"].to(args.device),
                batch["candidate_pose"].to(args.device),
                batch["mask"].to(args.device),
            )
            target = (
                (batch["targets"] - checkpoint["target_mean"])
                / checkpoint["target_std"]
            ).to(args.device)
            total += criterion(prediction, target).item() * len(batch["samples"])
    print(f"split: {args.split}")
    print(f"samples: {len(split_samples)}")
    print(f"normalized_mse: {total / len(split_samples):.6f}")


if __name__ == "__main__":
    main()
