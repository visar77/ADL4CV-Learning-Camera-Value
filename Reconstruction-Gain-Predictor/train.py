"""Train a baseline, CNN, or gain head on a labelled manifest."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse

from gain_predictor.devices import resolve_device
from gain_predictor.models import MODELS
from gain_predictor.models.heads import DEFAULT_HEAD_WIDTH, FEATURE_HEADS
from gain_predictor.training import train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="lvsm")
    parser.add_argument("--mode", choices=("overfit", "context", "scene"), default="context")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--scene-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--duplicates", choices=("mean", "keep"), default="mean")
    parser.add_argument("--target", choices=("delta-db", "relative"), default="delta-db",
                        help="Signed PSNR change in dB (default), or the previous relative gain")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=1, help="Maximum validation batches per epoch")
    parser.add_argument("--validation-window", type=int, default=10, help="Number of recent batches in the rolling mean")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--head", dest="head_type", choices=FEATURE_HEADS, default="mlp",
                        help="Cached/LVSM gain head (default: mlp)")
    parser.add_argument("--head-hidden-dim", type=int,
                        help="Cached/LVSM gain-head width (default: 64 for mlp, 32 for coverage/pose)")
    parser.add_argument("--head-hidden-layers", type=int, default=1,
                        help="Number of hidden layers in the cached/LVSM gain head (default: 1)")
    parser.add_argument("--ridge", type=float, default=1.)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--balance-scenes", action="store_true")
    parser.add_argument("--scheduler", choices=("none", "plateau", "cosine"), default="plateau")
    parser.add_argument("--scheduler-factor", type=float, default=.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.head_hidden_dim is None:
        args.head_hidden_dim = DEFAULT_HEAD_WIDTH[args.head_type]
    if min(args.validation_batch_size, args.validation_batches, args.validation_window) < 1:
        parser.error("Validation batch size, batch budget, and window must be positive")
    args.device = resolve_device(args.device)
    if min(args.epochs, args.batch_size, args.image_size, args.log_every) < 1:
        parser.error("epochs, batch size, image size and log frequency must be positive")
    if min(args.head_hidden_dim, args.head_hidden_layers) < 1:
        parser.error("Gain-head hidden width and layer count must be positive")
    if args.num_workers < 0 or args.ridge < 0 or args.weight_decay < 0 or args.learning_rate <= 0:
        parser.error("Invalid worker count or optimization parameters")
    if not 0 <= args.dropout < 1 or not 0 < args.scheduler_factor < 1:
        parser.error("Invalid dropout or scheduler factor")
    if min(args.scheduler_patience, args.early_stopping_patience, args.min_learning_rate, args.min_delta) < 0:
        parser.error("Patience, minimum learning rate and minimum delta must be nonnegative")
    if args.model == "toy" and args.balance_scenes:
        parser.error("--balance-scenes is currently supported for cached-feature training only")
    name = args.model_name or f"psnr_{args.model}_{args.mode}"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    args.log_dir = args.log_dir or Path("runs") / name / timestamp
    args.output = args.output or args.checkpoint_root / name / timestamp / "model.pt"
    return args


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
