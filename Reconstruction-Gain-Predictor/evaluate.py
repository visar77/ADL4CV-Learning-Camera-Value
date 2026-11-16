"""Evaluate exact saved splits using regression and candidate-selection metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import load_manifest
from experiments import restored_split, selection_metrics
from runtime import load_run, score_samples
from devices import resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--lvsm-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.device = resolve_device(args.device)
    checkpoint, model = load_run(args.checkpoint, args.device)
    samples = restored_split(load_manifest(args.dataset or Path(checkpoint["dataset_path"])), checkpoint, args.split)
    if not samples:
        raise ValueError(f"The {args.split} split is empty in {checkpoint['mode']} mode. "
                         "Use --split train to inspect an overfitting run.")
    prediction = score_samples(checkpoint, model, samples, args.features, args.device, args.lvsm_checkpoint)
    report = {"mode": checkpoint["mode"], "split": args.split,
              **selection_metrics(samples, prediction, checkpoint["target"])}
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return report


if __name__ == "__main__":
    main()
