"""Create a self-contained PSNR training dataset from renderer outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import preprocess_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data/results"))
    parser.add_argument("--views-root", type=Path, default=Path("data/views"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/psnr.jsonl")
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.image_size < 1:
        parser.error("--image-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    samples = preprocess_samples(
        args.results_root,
        args.views_root,
        args.output,
        image_size=args.image_size,
        limit=args.limit,
    )
    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
