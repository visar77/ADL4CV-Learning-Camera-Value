"""Train a reconstruction-quality or reconstruction-gain predictor.

The renderer writes one ``results.json`` per object/experiment.  This script
uses the recorded train-view subset as input and the evaluated average PSNR,
SSIM, or LPIPS as the target. In gain mode, labels are computed dynamically by
comparing experiments for the same object; no gain labels are stored.

Example:
    python train.py --data ../ObjaverseRenderer/results --target PSNR
    python train.py --data ../ObjaverseRenderer/results --mode gain --target PSNR

TensorBoard is used for metrics when installed::

    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from model import GainPredictor

METRICS = ("PSNR", "SSIM", "LPIPS")
FEATURE_NAMES = (
    "train_view_count",
    "index_min",
    "index_max",
    "index_mean",
    "index_std",
    "index_span",
    "index_density",
)


@dataclass(frozen=True)
class ResultRecord:
    """The subset of a renderer result needed by the predictor."""

    object_id: str
    experiment_id: str
    train_lattice_indices: tuple[int, ...]
    target: float


@dataclass(frozen=True)
class _MetricRecord:
    object_id: str
    experiment_id: str
    train_lattice_indices: tuple[int, ...]
    metrics: dict[str, float]


def _records_from_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        yield from payload
    else:
        raise ValueError(f"{path} must contain a JSON object or list of objects")


def _load_metric_records(data_path: Path) -> list[_MetricRecord]:
    """Load records containing finite reconstruction metric averages."""
    paths = (
        [data_path] if data_path.is_file() else sorted(data_path.rglob("results.json"))
    )
    if not paths:
        raise FileNotFoundError(f"No results.json files found under {data_path}")

    records: list[_MetricRecord] = []
    skipped = 0
    for path in paths:
        for raw in _records_from_json(path):
            try:
                averages = raw["metrics"]["averages"]
                indices = tuple(int(index) for index in raw["train_lattice_indices"])
                metrics = {metric: float(averages[metric]) for metric in METRICS}
                if not indices or not all(
                    math.isfinite(value) for value in metrics.values()
                ):
                    raise ValueError
                records.append(
                    _MetricRecord(
                        object_id=str(raw["object_id"]),
                        experiment_id=str(raw.get("experiment_id", path.parent.name)),
                        train_lattice_indices=indices,
                        metrics=metrics,
                    )
                )
            except (KeyError, TypeError, ValueError):
                skipped += 1

    if skipped:
        print(f"Skipped {skipped} incomplete or malformed record(s).")
    return records


def load_records(
    data_path: Path, target_metric: str, mode: str = "quality"
) -> list[ResultRecord]:
    """Load quality records or derive gain records from experiment pairs.

    Gain mode selects, for each experiment, the largest strict subset of its
    training views from the same object as its baseline. The feature indices
    are the newly added views and the target is the metric improvement.
    """
    if target_metric not in METRICS:
        raise ValueError(f"Unknown target metric: {target_metric}")
    if mode not in {"quality", "gain"}:
        raise ValueError(f"Unknown training mode: {mode}")

    metric_records = _load_metric_records(data_path)
    if mode == "quality":
        records = [
            ResultRecord(
                object_id=record.object_id,
                experiment_id=record.experiment_id,
                train_lattice_indices=record.train_lattice_indices,
                target=record.metrics[target_metric],
            )
            for record in metric_records
        ]
    else:
        records = _derive_gain_records(metric_records, target_metric)

    if not records:
        raise ValueError(
            f"No usable {mode} records found under {data_path}. "
            "Gain mode requires at least two experiments for an object where "
            "one training-view set is a strict subset of the other."
        )
    return records


def _derive_gain_records(
    metric_records: list[_MetricRecord], target_metric: str
) -> list[ResultRecord]:
    by_object: dict[str, list[_MetricRecord]] = {}
    for record in metric_records:
        by_object.setdefault(record.object_id, []).append(record)

    gains: list[ResultRecord] = []
    for object_records in by_object.values():
        for larger in object_records:
            larger_set = set(larger.train_lattice_indices)
            baselines = [
                candidate
                for candidate in object_records
                if set(candidate.train_lattice_indices) < larger_set
            ]
            if not baselines:
                continue
            baseline = max(
                baselines,
                key=lambda candidate: (
                    len(candidate.train_lattice_indices),
                    candidate.experiment_id,
                ),
            )
            added_indices = tuple(
                sorted(larger_set - set(baseline.train_lattice_indices))
            )
            if target_metric == "LPIPS":
                delta = baseline.metrics[target_metric] - larger.metrics[target_metric]
            else:
                delta = larger.metrics[target_metric] - baseline.metrics[target_metric]
            baseline_scale = max(abs(baseline.metrics[target_metric]), 1e-6)
            target = delta / baseline_scale
            gains.append(
                ResultRecord(
                    object_id=larger.object_id,
                    experiment_id=f"{larger.experiment_id}-vs-{baseline.experiment_id}",
                    train_lattice_indices=added_indices,
                    target=target,
                )
            )
    return gains


def record_features(record: ResultRecord) -> list[float]:
    indices = torch.tensor(record.train_lattice_indices, dtype=torch.float32)
    count = float(len(indices))
    minimum = float(indices.min())
    maximum = float(indices.max())
    mean = float(indices.mean())
    std = float(indices.std(unbiased=False))
    span = maximum - minimum
    density = count / (maximum + 1.0)
    return [count, minimum, maximum, mean, std, span, density]


class ResultDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        records: list[ResultRecord],
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.features = torch.tensor(
            [record_features(record) for record in records], dtype=torch.float32
        )
        self.features = (self.features - feature_mean) / feature_std
        self.targets = torch.tensor(
            [(record.target - target_mean) / target_std for record in records],
            dtype=torch.float32,
        ).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]


def split_by_object(
    records: list[ResultRecord], validation_fraction: float, seed: int
) -> tuple[list[ResultRecord], list[ResultRecord]]:
    groups: dict[str, list[ResultRecord]] = {}
    for record in records:
        groups.setdefault(record.object_id, []).append(record)
    object_ids = list(groups)
    random.Random(seed).shuffle(object_ids)
    validation_count = max(1, round(len(object_ids) * validation_fraction))
    if len(object_ids) > 1:
        validation_count = min(validation_count, len(object_ids) - 1)
    validation_ids = set(object_ids[:validation_count])
    validation = [record for record in records if record.object_id in validation_ids]
    training = [record for record in records if record.object_id not in validation_ids]
    return training, validation or training


def _writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install it with: pip install tensorboard"
        ) from error
    return SummaryWriter(log_dir=str(log_dir))


def select_records(
    records: list[ResultRecord],
    object_id: str | None = None,
    sample_index: int | None = None,
    max_samples: int | None = None,
) -> list[ResultRecord]:
    selected = [
        record
        for record in records
        if object_id is None or record.object_id == object_id
    ]
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(selected):
            raise IndexError(
                f"--sample-index {sample_index} is outside 0..{len(selected) - 1}"
            )
        selected = [selected[sample_index]]
    elif max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError("Record filters selected no training samples.")
    return selected


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.data, args.target, getattr(args, "mode", "quality"))
    records = select_records(
        records,
        getattr(args, "object_id", None),
        getattr(args, "sample_index", None),
        getattr(args, "max_samples", None),
    )
    train_records, validation_records = split_by_object(
        records, args.validation_fraction, args.seed
    )

    train_features = torch.tensor(
        [record_features(record) for record in train_records], dtype=torch.float32
    )
    feature_mean = train_features.mean(0)
    feature_std = train_features.std(0, unbiased=False).clamp_min(1e-6)
    target_mean = sum(record.target for record in train_records) / len(train_records)
    target_std = math.sqrt(
        sum((record.target - target_mean) ** 2 for record in train_records)
        / len(train_records)
    )
    target_std = max(target_std, 1e-6)

    train_loader = DataLoader(
        ResultDataset(
            train_records, feature_mean, feature_std, target_mean, target_std
        ),
        batch_size=min(args.batch_size, len(train_records)),
        shuffle=True,
    )
    validation_loader = DataLoader(
        ResultDataset(
            validation_records, feature_mean, feature_std, target_mean, target_std
        ),
        batch_size=args.batch_size,
    )
    device = torch.device(args.device)
    model = GainPredictor(
        input_dim=len(FEATURE_NAMES),
        backbone_checkpoint=getattr(args, "backbone_checkpoint", None),
        backbone_url=getattr(args, "backbone_url", None),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    writer = _writer(args.log_dir)

    best_validation_loss = math.inf
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path, epoch: int, validation_loss: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "target_metric": args.target,
                "mode": getattr(args, "mode", "quality"),
                "feature_names": FEATURE_NAMES,
                "epoch": epoch,
                "validation_loss": validation_loss,
            },
            path,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(features)
        training_loss /= len(train_loader.dataset)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for features, targets in validation_loader:
                validation_loss += criterion(
                    model(features.to(device)), targets.to(device)
                ).item() * len(features)
        validation_loss /= len(validation_loader.dataset)
        writer.add_scalars(
            "loss", {"train": training_loss, "validation": validation_loss}, epoch
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={training_loss:.4f} validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(args.output, epoch, validation_loss)
        if epoch % getattr(args, "checkpoint_every", 10) == 0:
            periodic_path = args.output.with_name(
                f"{args.output.stem}.epoch-{epoch}{args.output.suffix}"
            )
            save_checkpoint(periodic_path, epoch, validation_loss)
    writer.close()
    print(f"Saved best checkpoint to {args.output}")
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="results.json or results directory"
    )
    parser.add_argument("--target", choices=METRICS, default="PSNR")
    parser.add_argument(
        "--mode",
        choices=("quality", "gain"),
        default="quality",
        help="Predict an absolute metric or dynamically derived experiment gain",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/gain_predictor.pt")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/gain_predictor"))
    parser.add_argument(
        "--backbone-checkpoint", type=Path, default=Path("checkpoints/lvsm_backbone.pt")
    )
    parser.add_argument("--backbone-url", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--object-id")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.epochs < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("--epochs, --batch-size, and --checkpoint-every must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
