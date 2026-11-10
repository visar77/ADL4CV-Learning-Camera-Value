"""Train the PSNR gain predictor on a preprocessed dataset."""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    GainDataset,
    GainSample,
    collate_gain_samples,
    load_processed_dataset,
)
from model import ContextPoseCNN, LVSMGainPredictor


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install TensorBoard with: pip install tensorboard"
        ) from error
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def _split(
    samples: list[GainSample], seed: int
) -> tuple[list[GainSample], list[GainSample]]:
    scenes = sorted({sample.scene_id for sample in samples})
    if len(scenes) < 2:
        return samples, samples
    random.Random(seed).shuffle(scenes)
    validation_scenes = set(scenes[: max(1, round(len(scenes) * 0.2))])
    return (
        [sample for sample in samples if sample.scene_id not in validation_scenes],
        [sample for sample in samples if sample.scene_id in validation_scenes],
    )


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    metadata, samples = load_processed_dataset(args.dataset)
    if args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num-samples must be positive.")
        if args.num_samples > len(samples):
            raise ValueError(
                f"--num-samples {args.num_samples} exceeds dataset size {len(samples)}."
            )
        samples = samples[: args.num_samples]
    train_samples, validation_samples = _split(samples, args.seed)
    targets = torch.tensor([sample.psnr_gain for sample in train_samples])
    target_mean = float(targets.mean())
    target_std = max(float(targets.std(unbiased=False)), 1e-6)
    train_loader = DataLoader(
        GainDataset(train_samples),
        batch_size=min(args.batch_size, len(train_samples)),
        shuffle=True,
        collate_fn=lambda batch: collate_gain_samples(batch, args.image_size),
    )
    validation_loader = DataLoader(
        GainDataset(validation_samples),
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_gain_samples(batch, args.image_size),
    )
    if args.model == "toy":
        model = ContextPoseCNN(image_size=args.image_size)
    else:
        if args.lvsm_checkpoint is None:
            raise ValueError(
                "Normal training requires --lvsm-checkpoint. "
                "Use debug_train.py for the toy CPU model."
            )
        model = LVSMGainPredictor(args.lvsm_checkpoint, image_size=args.image_size)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    criterion = nn.MSELoss()
    writer = _writer(args.log_dir)
    best_loss = math.inf
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            prediction = model(
                batch["images"].to(args.device),
                batch["context_poses"].to(args.device),
                batch["candidate_pose"].to(args.device),
                batch["mask"].to(args.device),
            )
            target = ((batch["targets"] - target_mean) / target_std).to(args.device)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch["samples"])
        train_loss /= len(train_loader.dataset)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                prediction = model(
                    batch["images"].to(args.device),
                    batch["context_poses"].to(args.device),
                    batch["candidate_pose"].to(args.device),
                    batch["mask"].to(args.device),
                )
                target = ((batch["targets"] - target_mean) / target_std).to(args.device)
                validation_loss += criterion(prediction, target).item() * len(
                    batch["samples"]
                )
        validation_loss /= len(validation_loader.dataset)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", validation_loss, epoch)
        print(
            f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": args.model,
                    "target_metric": "PSNR",
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "dataset_format": metadata["format"],
                    "image_size": args.image_size,
                    "candidate_rgb_used": False,
                    "num_samples": len(samples),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                args.output,
            )
    writer.close()
    print(f"Saved checkpoint to {args.output}")
    return args.output


def parse_args(
    argv: list[str] | None = None, defaults: dict[str, object] | None = None
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=("lvsm", "toy"), default="lvsm")
    parser.add_argument("--lvsm-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-name", default="psnr_lvsm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Use the first N records from the processed dataset.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    timestamp = _timestamp()
    if args.log_dir is None:
        args.log_dir = Path("runs") / args.model_name / timestamp
    if args.output is None:
        args.output = args.checkpoint_root / args.model_name / timestamp / "model.pt"
    return args


def main(
    argv: list[str] | None = None, defaults: dict[str, object] | None = None
) -> Path:
    return train(parse_args(argv, defaults))


if __name__ == "__main__":
    main()
