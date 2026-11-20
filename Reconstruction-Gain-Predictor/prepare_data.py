"""Build labelled manifests from renderer scores and camera/image metadata."""

from __future__ import annotations

from pathlib import Path
import argparse

from gain_predictor.data.manifest import build_manifest, relativize_manifest, upgrade_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("data/results"))
    parser.add_argument("--views-root", type=Path, help="Rebuild default: data/views; upgrade default: infer beside existing images")
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/psnr.jsonl")
    )
    parser.add_argument("--limit", type=int)
    migration = parser.add_mutually_exclusive_group()
    migration.add_argument("--relativize", type=Path, help="Change paths only in an existing manifest")
    migration.add_argument("--upgrade", type=Path, help="Add calibration from existing cameras; preserve records/labels and create a backup")
    parser.add_argument("--duplicates", choices=("mean", "keep"), default="mean")
    args = parser.parse_args(argv)
    if args.relativize or args.upgrade:
        target = args.relativize or args.upgrade
        if not target.is_file():
            existing = sorted(target.parent.glob("*.jsonl"))
            available = f" Available manifests: {', '.join(str(p) for p in existing[:5])}." if existing else ""
            parser.error(
                f"Manifest not found: {target}. Migration only converts an existing "
                "older manifest; it does not create one. Newly generated manifests already use "
                f"relative paths and need no migration.{available}"
            )
        try:
            samples = (upgrade_manifest(target, views_root=args.views_root) if args.upgrade
                       else relativize_manifest(target))
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        if args.upgrade:
            print(f"Upgraded {len(samples)} records with calibrated intrinsics in {target}; labels preserved")
        else:
            print(f"Made {len(samples)} records portable in {target}")
            missing = sum(not s.context_intrinsics or s.candidate_intrinsics is None for s in samples)
            if missing:
                print(f"{missing} records still lack intrinsics. Paths-only migration does not add calibration. "
                      f"Run: python prepare_data.py --upgrade {target}")
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        samples = build_manifest(
            args.results_root,
            args.views_root or Path("data/views"),
            args.output,
            limit=args.limit,
            duplicates=args.duplicates,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Saved {len(samples)} samples to {args.output}")
    print(f"Audit: {args.output.with_suffix('.audit.json')}")


if __name__ == "__main__":
    main()
