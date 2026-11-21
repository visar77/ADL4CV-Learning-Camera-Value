"""Build labelled manifests from renderer scores and camera/image metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from gain_predictor.data.manifest import build_manifest, relativize_manifest, upgrade_manifest


def _read_config(path: Path, parser: argparse.ArgumentParser) -> tuple[dict, Path]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"Cannot read config {path}: {error}")
    if not isinstance(config, dict):
        parser.error(f"Config must contain a JSON object: {path}")
    return config, path.resolve().parent


def _config_path(value, config_dir: Path, parser: argparse.ArgumentParser, name: str) -> Path:
    if not isinstance(value, str) or not value:
        parser.error(f"Config field {name!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def _paths(value, config_dir: Path, parser: argparse.ArgumentParser, name: str) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    if not values:
        parser.error(f"Config field {name!r} must contain at least one path.")
    return [_config_path(item, config_dir, parser, name) for item in values]


def _output_filename(object_ids: list[str], metric: str) -> str:
    slug = "_".join(re.sub(r"[^A-Za-z0-9._-]+", "-", object_id).strip("-._").lower()
                    for object_id in sorted(object_ids))
    if len(slug) > 160:
        digest = hashlib.sha256("\0".join(sorted(object_ids)).encode()).hexdigest()[:12]
        slug = f"{slug[:140]}_{digest}"
    return f"{slug}_{metric.lower()}.jsonl"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("preprocess_config.json"))
    parser.add_argument(
        "--object-id", "--object-ids", nargs="+", dest="object_ids",
        help="Object IDs to combine in one manifest (alias: --object-ids)",
    )
    parser.add_argument("--results-root", type=Path, nargs="+")
    parser.add_argument("--views-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path, help="Override the manifest output path")
    parser.add_argument("--output-name", "--output_name", dest="output_name",
                        help="Manifest filename inside output-root")
    parser.add_argument("--metric", choices=("PSNR", "SSIM", "LPIPS"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--relativize", type=Path, help="Change paths only in an existing manifest")
    parser.add_argument("--upgrade", type=Path, help="Add calibration to an existing manifest")
    parser.add_argument("--duplicates", choices=("mean", "keep"))
    args = parser.parse_args(argv)

    config, config_dir = _read_config(args.config, parser)
    supported = {
        "results-root", "views-root", "output-root", "output-name", "object-ids",
        "metric", "limit", "relativize", "upgrade", "duplicates",
    }
    unknown = sorted(set(config) - supported)
    if unknown:
        parser.error(f"Unsupported config field(s): {', '.join(unknown)}")

    def setting(cli_value, key, default=None):
        return cli_value if cli_value is not None else config.get(key, default)

    results_roots = (
        args.results_root if args.results_root is not None
        else _paths(config.get("results-root"), config_dir, parser, "results-root")
    )
    views_value = args.views_root if args.views_root is not None else config.get("views-root")
    output_root_value = args.output_root if args.output_root is not None else config.get("output-root")
    views_root = (args.views_root if args.views_root is not None
                  else _config_path(views_value, config_dir, parser, "views-root"))
    output_root = (args.output_root if args.output_root is not None
                   else _config_path(output_root_value, config_dir, parser, "output-root"))
    metric = setting(args.metric, "metric", "PSNR")
    if metric not in {"PSNR", "SSIM", "LPIPS"}:
        parser.error("Config field 'metric' must be PSNR, SSIM, or LPIPS.")
    limit = setting(args.limit, "limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        parser.error("limit must be a positive integer or null.")
    duplicates = setting(args.duplicates, "duplicates", "mean")
    if duplicates not in {"mean", "keep"}:
        parser.error("duplicates must be 'mean' or 'keep'.")

    relativize = args.relativize if args.relativize is not None else config.get("relativize")
    upgrade = args.upgrade if args.upgrade is not None else config.get("upgrade")
    if relativize and upgrade:
        parser.error("Choose only one of relativize or upgrade.")
    if relativize or upgrade:
        target = (relativize if isinstance(relativize, Path) else
                  _config_path(relativize, config_dir, parser, "relativize")) if relativize else (
                  upgrade if isinstance(upgrade, Path) else
                  _config_path(upgrade, config_dir, parser, "upgrade"))
        if not target.is_file():
            existing = sorted(target.parent.glob("*.jsonl"))
            available = f" Available manifests: {', '.join(str(path) for path in existing[:5])}." if existing else ""
            parser.error(
                f"Manifest not found: {target}. Migration only converts an existing manifest; "
                f"newly generated manifests already use relative paths.{available}"
            )
        try:
            samples = (upgrade_manifest(target, views_root=views_root) if upgrade
                       else relativize_manifest(target))
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        if upgrade:
            print(f"Upgraded {len(samples)} records with calibrated intrinsics in {target}; labels preserved")
        else:
            print(f"Made {len(samples)} records portable in {target}")
            missing = sum(not sample.context_intrinsics or sample.candidate_intrinsics is None
                          for sample in samples)
            if missing:
                print(f"{missing} records still lack intrinsics. Paths-only migration does not add calibration.")
        return

    object_ids = args.object_ids if args.object_ids is not None else config.get("object-ids", [])
    if isinstance(object_ids, str):
        object_ids = [object_ids]
    if (not isinstance(object_ids, list) or not object_ids
            or any(not isinstance(object_id, str) or not object_id.strip() for object_id in object_ids)):
        parser.error("Provide one or more object IDs with --object-id/--object-ids or in config.")
    object_ids = list(dict.fromkeys(object_id.strip() for object_id in object_ids))

    output_name = args.output_name or config.get("output-name")
    if output_name is not None and (
        not isinstance(output_name, str) or not output_name or Path(output_name).name != output_name
    ):
        parser.error("output-name must be a filename, not a path.")
    if output_name is not None:
        if Path(output_name).suffix and Path(output_name).suffix.lower() != ".jsonl":
            parser.error("output-name must have a .jsonl extension.")
        if not Path(output_name).suffix:
            output_name += ".jsonl"
    output = args.output or output_root / (output_name or _output_filename(object_ids, metric))

    try:
        samples = build_manifest(
            results_roots,
            views_root,
            output,
            metric=metric,
            object_ids=object_ids,
            limit=limit,
            duplicates=duplicates,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"Saved {len(samples)} {metric} samples for {', '.join(object_ids)} to {output}")
    print(f"Audit: {output.with_suffix('.audit.json')}")


if __name__ == "__main__":
    main()
