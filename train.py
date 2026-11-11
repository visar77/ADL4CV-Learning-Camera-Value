"""Train the PSNR gain predictor on a transparent JSONL dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    DEFAULT_IMAGE_SIZE,
    GainDataset,
    GainSample,
    collate_gain_samples,
    load_manifest,
    sample_key,
)
from model import (
    LVSM_CHECKPOINT_NAME,
    LVSM_CHECKPOINT_URL,
    ContextPoseCNN,
    LVSMGainPredictor,
)


def _writer(path: Path):
    from torch.utils.tensorboard import SummaryWriter

    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def _select_subset(samples: list[GainSample], count: int | None) -> list[GainSample]:
    if count is None:
        return samples
    if count < 1 or count > len(samples):
        raise ValueError(f"--num-samples must be between 1 and {len(samples)}.")
    grouped: dict[str, list[GainSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.scene_id, []).append(sample)
    selected: list[GainSample] = []
    while len(selected) < count:
        for scene in sorted(grouped):
            if grouped[scene]:
                selected.append(grouped[scene].pop(0))
                if len(selected) == count:
                    return selected
    return selected


def _split(
    samples: list[GainSample],
    seed: int,
    split_file: Path | None,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[GainSample], list[GainSample], list[GainSample], dict[str, list[str]]]:
    scenes = sorted({sample.scene_id for sample in samples})
    if split_file:
        raw = json.loads(split_file.read_text(encoding="utf-8"))
        assignments = {
            name: set(raw.get(name, [])) for name in ("train", "validation", "test")
        }
        all_assigned = set().union(*assignments.values())
        if all_assigned - set(scenes) or len(all_assigned) != sum(
            map(len, assignments.values())
        ):
            raise ValueError("Split file contains unknown or duplicate scene IDs.")
        if set(scenes) - all_assigned:
            raise ValueError(
                f"Split file omits scenes: {sorted(set(scenes) - all_assigned)}"
            )
    elif len(scenes) == 1:
        assignments = {name: set(scenes) for name in ("train", "validation", "test")}
    else:
        shuffled = scenes[:]
        random.Random(seed).shuffle(shuffled)
        test_count = max(1, round(len(scenes) * test_fraction))
        validation_count = max(1, round(len(scenes) * validation_fraction))
        while test_count + validation_count >= len(scenes):
            if test_count > validation_count:
                test_count -= 1
            else:
                validation_count -= 1
        test_start = len(scenes) - test_count
        validation_start = test_start - validation_count
        assignments = {
            "train": set(shuffled[:validation_start]),
            "validation": set(shuffled[validation_start:test_start]),
            "test": set(shuffled[test_start:]),
        }
    split_samples = tuple(
        [sample for sample in samples if sample.scene_id in assignments[name]]
        for name in ("train", "validation", "test")
    )
    if not split_samples[0]:
        raise ValueError("The training split contains no samples.")
    return (
        *split_samples,
        {name: sorted(values) for name, values in assignments.items()},
    )


def _evaluate(model, loader, mean: float, std: float, device: str, criterion) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["images"].to(device),
                batch["context_poses"].to(device),
                batch["candidate_pose"].to(device),
                batch["mask"].to(device),
            )
            target = ((batch["targets"] - mean) / std).to(device)
            total += criterion(prediction, target).item() * len(batch["samples"])
    return total / len(loader.dataset)


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    samples = load_manifest(args.dataset)
    samples = _select_subset(samples, args.num_samples)
    train_samples, validation_samples, test_samples, split = _split(
        samples,
        args.seed,
        args.split_file,
        args.validation_fraction,
        args.test_fraction,
    )
    print("training configuration:")
    print(f"  model: {args.model}")
    print(f"  dataset: {args.dataset}")
    print(f"  selected samples: {len(samples)}")
    print(
        "  split samples: "
        f"train={len(train_samples)}, "
        f"validation={len(validation_samples)}, "
        f"test={len(test_samples)}"
    )
    print(f"  scenes: {split}")
    print(f"  image size: {args.image_size}x{args.image_size}")
    if args.model == "lvsm":
        print(f"  LVSM latent tokens: {args.lvsm_latent_tokens}")
    print(f"  batch size: {args.batch_size}")
    print(f"  learning rate: {args.learning_rate:g}")
    print(f"  scheduler: {args.scheduler}")
    if args.scheduler == "plateau":
        print(
            "  scheduler settings: "
            f"factor={args.scheduler_factor:g}, "
            f"patience={args.scheduler_patience}, "
            f"min_lr={args.min_learning_rate:g}"
        )
    elif args.scheduler == "cosine":
        print(f"  scheduler settings: min_lr={args.min_learning_rate:g}")
    print(
        "  early stopping: "
        + (
            f"patience={args.early_stopping_patience}"
            if args.early_stopping_patience
            else "disabled"
        )
    )
    if len(set(split["train"]) | set(split["validation"]) | set(split["test"])) == 1:
        print(
            "  note: one scene is reused for all splits; "
            "this is a pipeline/overfitting smoke test, not generalization."
        )
    targets = torch.tensor([sample.psnr_gain for sample in train_samples])
    target_mean = float(targets.mean())
    target_std = max(float(targets.std(unbiased=False)), 1e-6)
    collate = lambda batch: collate_gain_samples(batch, args.image_size)
    train_counts = {
        scene: sum(s.scene_id == scene for s in train_samples)
        for scene in split["train"]
    }
    weights = torch.tensor([1.0 / train_counts[s.scene_id] for s in train_samples])
    train_loader = DataLoader(
        GainDataset(train_samples),
        batch_size=min(args.batch_size, len(train_samples)),
        sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        GainDataset(validation_samples), batch_size=args.batch_size, collate_fn=collate
    )
    test_loader = DataLoader(
        GainDataset(test_samples), batch_size=args.batch_size, collate_fn=collate
    )
    if args.model == "toy":
        model = ContextPoseCNN(image_size=args.image_size)
    else:
        if args.lvsm_checkpoint is None:
            raise ValueError("Normal training requires --lvsm-checkpoint.")
        model = LVSMGainPredictor(
            args.lvsm_checkpoint,
            image_size=args.image_size,
            latent_tokens=args.lvsm_latent_tokens,
            checkpoint_url=args.lvsm_url,
        )
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    criterion = nn.MSELoss()
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_learning_rate,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
        )
    else:
        scheduler = None
    writer = _writer(args.log_dir)
    best_loss = math.inf
    epochs_without_improvement = 0
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
        validation_loss = _evaluate(
            model, validation_loader, target_mean, target_std, args.device, criterion
        )
        test_loss = _evaluate(
            model, test_loader, target_mean, target_std, args.device, criterion
        )
        if scheduler is not None:
            previous_learning_rate = optimizer.param_groups[0]["lr"]
            if args.scheduler == "plateau":
                scheduler.step(validation_loss)
            else:
                scheduler.step()
            learning_rate = optimizer.param_groups[0]["lr"]
            if learning_rate != previous_learning_rate:
                print(
                    f"  scheduler: learning rate "
                    f"{previous_learning_rate:.3g} -> {learning_rate:.3g}"
                )
        else:
            learning_rate = optimizer.param_groups[0]["lr"]
        writer.add_scalars(
            "loss",
            {"train": train_loss, "validation": validation_loss, "test": test_loss},
            epoch,
        )
        writer.add_scalar("learning_rate", learning_rate, epoch)
        print(
            f"epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} test_loss={test_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": args.model,
                    "target_metric": "PSNR",
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "image_size": args.image_size,
                    "lvsm_latent_tokens": args.lvsm_latent_tokens,
                    "candidate_rgb_used": False,
                    "num_samples": len(samples),
                    "sample_ids": [sample_key(sample) for sample in samples],
                    "split": split,
                    "seed": args.seed,
                    "validation_fraction": args.validation_fraction,
                    "test_fraction": args.test_fraction,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "test_loss": test_loss,
                    "scheduler": args.scheduler,
                    "early_stopping_patience": args.early_stopping_patience,
                },
                args.output,
            )
            print(
                f"  checkpoint: saved new best validation loss "
                f"{validation_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience is not None
                and args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"early stopping after {epoch} epochs "
                    f"(patience={args.early_stopping_patience})"
                )
                break
    writer.close()
    print(f"Saved checkpoint to {args.output}")
    return args.output


def parse_args(argv=None, defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=("lvsm", "toy"), default="lvsm")
    parser.add_argument(
        "--lvsm-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / LVSM_CHECKPOINT_NAME,
    )
    parser.add_argument("--lvsm-url", default=LVSM_CHECKPOINT_URL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-name", default="psnr_lvsm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--lvsm-latent-tokens", type=int, default=3072)
    parser.add_argument(
        "--scheduler", choices=("none", "plateau", "cosine"), default="plateau"
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0 <= args.validation_fraction < 1 or not 0 <= args.test_fraction < 1:
        parser.error("split fractions must be in [0, 1)")
    if args.image_size < 1:
        parser.error("--image-size must be positive")
    if not 1 <= args.lvsm_latent_tokens <= 3072:
        parser.error("--lvsm-latent-tokens must be between 1 and 3072")
    if args.scheduler_factor <= 0 or args.scheduler_factor >= 1:
        parser.error("--scheduler-factor must be between 0 and 1")
    if args.scheduler_patience < 0 or args.min_learning_rate < 0:
        parser.error(
            "scheduler patience and minimum learning rate must be non-negative"
        )
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    args.log_dir = args.log_dir or Path("runs") / args.model_name / timestamp
    args.output = (
        args.output or args.checkpoint_root / args.model_name / timestamp / "model.pt"
    )
    return args


def main(argv=None, defaults=None):
    return train(parse_args(argv, defaults))


if __name__ == "__main__":
    main()
