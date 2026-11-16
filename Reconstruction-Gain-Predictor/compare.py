"""Run cheap baselines and optional cached LVSM on exactly the same split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import parse_args, train
from runtime import MODELS
from paths import relative_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["mean", "view-count", "geometry", "toy"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    results = {}
    splits = None
    for name in args.models:
        config = parse_args(remaining + ["--model", name, "--output", str(args.output_dir / name / "model.pt"),
                                         "--log-dir", str(args.output_dir / name / "logs")])
        path = train(config)
        report = json.loads(path.with_suffix(".report.json").read_text())
        if splits is not None and splits != report["splits"]:
            raise ValueError("Comparison runs produced different splits.")
        splits = report["splits"]
        results[name] = {"checkpoint": relative_path(path, args.output_dir), "metrics": report["metrics"],
                         "elapsed_seconds": report["elapsed_seconds"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "comparison.json"
    destination.write_text(json.dumps({"splits": splits, "models": results}, indent=2, allow_nan=False))
    print(f"Comparison saved to {destination}")
    return destination


if __name__ == "__main__":
    main()
