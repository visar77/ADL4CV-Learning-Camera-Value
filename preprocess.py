"""Create a self-contained PSNR training dataset from renderer outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import preprocess_samples, relativize_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data/results"))
    parser.add_argument("--views-root", type=Path, default=Path("data/views"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/psnr.jsonl")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--relativize", type=Path, help="Migrate an existing manifest in place without changing labels")
    parser.add_argument("--duplicates", choices=("mean", "keep"), default="mean")
    args = parser.parse_args(argv)
    if args.relativize:
        if not args.relativize.is_file():
            existing = sorted(args.relativize.parent.glob("*.jsonl"))
            available = f" Available manifests: {', '.join(str(p) for p in existing[:5])}." if existing else ""
            parser.error(
                f"Manifest not found: {args.relativize}. --relativize only converts an existing "
                "older manifest; it does not create one. Newly generated manifests already use "
                f"relative paths and need no migration.{available}"
            )
        samples = relativize_manifest(args.relativize)
        print(f"Made {len(samples)} records portable in {args.relativize}")
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    samples = preprocess_samples(
        args.results_root,
        args.views_root,
        args.output,
        limit=args.limit,
        duplicates=args.duplicates,
    )
    print(f"Saved {len(samples)} samples to {args.output}")
    print(f"Audit: {args.output.with_suffix('.audit.json')}")


if __name__ == "__main__":
    main()
