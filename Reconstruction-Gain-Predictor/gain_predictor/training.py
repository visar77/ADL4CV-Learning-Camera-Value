"""Train small gain predictors with explicit overfit, context, or scene splits."""
from __future__ import annotations

from collections import Counter, deque
import json
import math
import time

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch

from gain_predictor.artifacts import atomic_save, relative_path
from gain_predictor.data.manifest import load_manifest
from gain_predictor.data.records import audit_samples, sample_key, target_values
from gain_predictor.data.splits import make_splits, observation_signatures, select_samples
from gain_predictor.devices import resolve_device
from gain_predictor.feature_cache import FeatureStore
from gain_predictor.geometry import COVERAGE_DIM
from gain_predictor.metrics import selection_metrics
from gain_predictor.models import create_model
from gain_predictor.scoring import (
    forward_toy,
    head_matrix,
    input_matrix,
    normalize_matrix,
    predict_normalized,
    toy_loader,
)


class RollingValidation:
    """Bounded validation work; sample-weighted MSE over recent batches.

    The iterator persists across epochs and restarts only after every validation
    batch has been visited. Its independent RNG never perturbs training sampling.
    The estimate includes predictions from recent model states by design.
    """
    def __init__(self, loader, predict_batch, window=10, batches_per_update=1):
        if not len(loader) or min(window, batches_per_update) < 1:
            raise ValueError("Rolling validation needs nonempty data and positive limits.")
        self.loader = loader
        self.predict_batch = predict_batch
        self.observations = deque(maxlen=window)
        self.batches_per_update = min(batches_per_update, len(loader))
        self.iterator = None
        self.samples_seen = 0

    @torch.no_grad()
    def update(self, model):
        model.eval()
        samples = 0
        last_loss = 0.
        for _ in range(self.batches_per_update):
            if self.iterator is None:
                self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            prediction, target = self.predict_batch(model, batch)
            squared_error = (prediction - target).square()
            if not torch.isfinite(squared_error).all():
                raise ValueError("Non-finite validation loss.")
            count = len(target)
            error_sum = squared_error.sum().item()
            self.observations.append((error_sum, count))
            last_loss = error_sum / count
            samples += count
        self.samples_seen += samples
        window_samples = sum(count for _, count in self.observations)
        return {"loss": sum(error for error, _ in self.observations) / window_samples,
                "last_batch_loss": last_loss, "samples": samples,
                "batches": self.batches_per_update, "window_samples": window_samples,
                "window_batches": len(self.observations), "samples_seen": self.samples_seen,
                "completed_sweeps": self.samples_seen // len(self.loader.dataset)}


def train(args):
    """Fit a model and write its best checkpoint and split reports."""
    args.device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    raw = load_manifest(args.dataset)
    samples = select_samples(raw, args.num_samples, args.scene_id, args.experiment_id, args.duplicates)
    groups, splits = make_splits(samples, args.mode, args.seed, args.validation_fraction,
                                 args.test_fraction, args.split_file)
    targets = {name: target_values(group, args.target) for name, group in groups.items()}
    mean = targets["train"].mean().item()
    std = max(targets["train"].std(unbiased=False).item(), 1e-6)
    config = {"format_version": 2, "model_type": args.model, "mode": args.mode,
              "target": args.target, "target_mean": mean, "target_std": std,
              "image_size": args.image_size, "batch_size": args.batch_size,
              "num_workers": args.num_workers, "dropout": args.dropout,
              "head_hidden_dim": args.head_hidden_dim,
              "head_hidden_layers": args.head_hidden_layers,
              "seed": args.seed, "splits": splits, "duplicates": args.duplicates,
              "candidate_rgb_used": False,
              "selection": {"scene_id": args.scene_id, "experiment_id": args.experiment_id},
              "sample_signatures": observation_signatures(samples), "signature_mode": "content-v1",
              "artifact_paths": "checkpoint-relative",
              "dataset_path": relative_path(args.dataset, args.output.parent), "learning_rate": args.learning_rate,
              "scheduler": args.scheduler, "ridge": args.ridge,
              "validation_batch_size": args.validation_batch_size,
              "validation_batches": args.validation_batches, "validation_window": args.validation_window,
              "validation_fraction": args.validation_fraction, "test_fraction": args.test_fraction}
    matrices = {}
    if args.model in {"cached", "lvsm"}:
        if not args.features:
            raise ValueError("Cached head training requires --features. Run cache_features.py once first.")
        store = FeatureStore(args.features)
        # Hash source images and materialize features once, never inside the epoch loop.
        embeddings = store.matrix(samples)
        config.update(head_type=args.head_type, embedding_dim=embeddings.shape[1])
        all_features = head_matrix(samples, embeddings, config)
        indices = {sample_key(s): i for i, s in enumerate(samples)}
        matrices = {name: all_features[[indices[sample_key(s)] for s in group]]
                    for name, group in groups.items()}
        config.update(features_path=relative_path(args.features, args.output.parent), feature_identity=store.identity,
                      backbone_spec=store.spec, image_size=store.spec["image_size"])
        if store.backbone_path is not None:
            config["backbone_checkpoint"] = relative_path(store.backbone_path, args.output.parent)
    elif args.model != "toy":
        matrices = {name: input_matrix(group, config) if group else None for name, group in groups.items()}
    if matrices:
        train_inputs = matrices["train"]
        if config.get("head_type") == "pose":
            # Contexts can have different view counts at inference. Only the
            # fixed-width embedding uses fitted feature statistics.
            train_inputs = train_inputs[:, :config["embedding_dim"]]
        config["input_dim"] = train_inputs.shape[1]
        config["input_mean"] = train_inputs.mean(0)
        config["input_std"] = train_inputs.std(0, unbiased=False).clamp_min(1e-5)
        if config.get("head_type") == "coverage":
            # These four features have fixed units, even if a training fold
            # contains only one view count or a constant angular separation.
            config["input_mean"][-COVERAGE_DIM:] = 0
            config["input_std"][-COVERAGE_DIM:] = 1
    model = create_model(config).to(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.log_dir))
    counts = {name: len(group) for name, group in groups.items()}
    print(f"model={args.model} mode={args.mode} device={args.device} target={args.target} splits={counts}", flush=True)
    audit = audit_samples(samples)
    start = time.perf_counter()
    monitor_name = "validation" if groups["validation"] else "train"
    if monitor_name == "train":
        print("Monitoring fit on the training samples; no held-out validation estimate.", flush=True)

    def evaluate(name):
        normalized = predict_normalized(model, groups[name], config, args.device, matrices.get(name))
        truth = (targets[name] - mean) / std
        loss = float((normalized - truth).square().mean())
        prediction = normalized * std + mean
        return loss, prediction

    is_baseline = args.model in {"mean", "view-count", "geometry"}
    if is_baseline:
        monitor_kind = "full_loss"
    elif groups["validation"]:
        monitor_kind = "rolling_validation"
    else:
        monitor_kind = "training_epoch_loss"

    def save(epoch, monitor):
        atomic_save({**config, "model": model.state_dict(), "epoch": epoch,
                     "monitor": monitor_name, "monitor_loss": monitor,
                     "monitor_kind": monitor_kind}, args.output)

    try:
        history = []
        if is_baseline:
            _fit_ridge(model, matrices["train"], targets["train"], config, args)
            monitor, _ = evaluate(monitor_name)
            save(0, monitor)
        else:
            history = _fit_neural(model, args, config, groups, matrices, targets,
                                  monitor_name, save, writer)
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        reports = {}
        # Test is evaluated once, after restoring the best selected checkpoint.
        for name, group in groups.items():
            if not group:
                reports[name] = None
                continue
            loss, prediction = evaluate(name)
            reports[name] = {**selection_metrics(group, prediction, args.target), "normalized_mse": loss}
        report = {"model": args.model, "mode": args.mode, "target": args.target,
                  "checkpoint": args.output.name, "splits": splits,
                  "audit": audit, "source_audit": audit_samples(raw),
                  "best_epoch": checkpoint["epoch"], "monitor": monitor_name,
                  "monitor_kind": checkpoint["monitor_kind"],
                  "validation_config": {"batch_size": args.validation_batch_size, "batches_per_epoch": args.validation_batches,
                                        "window_batches": args.validation_window},
                  "elapsed_seconds": time.perf_counter()-start, "metrics": reports, "history": history}
        args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"Saved {args.output}; report: {args.output.with_suffix('.report.json')}", flush=True)
    finally:
        writer.close()
    return args.output

@torch.no_grad()
def _fit_ridge(model, matrix, targets, config, args):
    x = normalize_matrix(matrix, config).double()
    y = ((targets - config["target_mean"]) / config["target_std"]).double()
    design = torch.cat((torch.ones(len(x), 1, dtype=x.dtype), x), dim=1)
    penalty = torch.eye(design.shape[1], dtype=x.dtype) * args.ridge
    penalty[0, 0] = 0  # Leave the intercept unpenalized.
    weights = torch.linalg.lstsq(design.T @ design + penalty, design.T @ y).solution
    model.bias.copy_(weights[0].float().to(args.device))
    model.weight.copy_(weights[1:].T.float().to(args.device))


def _fit_neural(model, args, config, groups, matrices, targets, monitor_name, save, writer):
    mean, std = config["target_mean"], config["target_std"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_factor,
                                                              patience=args.scheduler_patience,
                                                              min_lr=args.min_learning_rate)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_learning_rate)
    else:
        scheduler = None

    if args.model == "toy":
        def make_loader(split, batch_size, generator=None):
            return toy_loader(groups[split], {**config, "batch_size": batch_size},
                              shuffle=True, generator=generator, device=args.device, target=args.target)

        def predict_batch(model, batch):
            prediction = forward_toy(model, batch, args.device)
            return prediction, ((batch["targets"] - mean) / std).to(args.device)
    else:
        def make_loader(split, batch_size, generator=None):
            x = normalize_matrix(matrices[split], config)
            y = (targets[split] - mean) / std
            sampler = None
            if split == "train" and args.balance_scenes:
                scenes = [sample.scene_id for sample in groups[split]]
                counts = Counter(scenes)
                sampler = WeightedRandomSampler([1 / counts[scene] for scene in scenes],
                                                len(scenes), replacement=True)
            return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                              shuffle=sampler is None, sampler=sampler, generator=generator)

        def predict_batch(model, batch):
            inputs, target = (value.to(args.device) for value in batch)
            return model(inputs), target

    loader = make_loader("train", args.batch_size)
    rolling = None
    if groups["validation"]:
        generator = torch.Generator().manual_seed(args.seed + 104729)
        validation = make_loader("validation", args.validation_batch_size, generator)
        rolling = RollingValidation(validation, predict_batch, args.validation_window,
                                     args.validation_batches)
        print(f"Validation: up to {args.validation_batches} batches of {args.validation_batch_size} "
              f"per epoch; rolling window {args.validation_window} batches", flush=True)

    history = []
    best_loss = math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = predict_batch(model, batch)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
        train_loss = total / len(groups["train"])
        validation_start = time.perf_counter()
        validation = rolling.update(model) if rolling is not None else None
        monitor = validation["loss"] if validation is not None else train_loss
        validation_seconds = time.perf_counter() - validation_start
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(monitor)
            else:
                scheduler.step()
        entry = {"epoch": epoch, "train_loss": train_loss, "monitor_loss": monitor,
                 "seconds": time.perf_counter()-epoch_start,
                 "learning_rate": optimizer.param_groups[0]["lr"],
                 "validation": validation, "validation_seconds": validation_seconds}
        history.append(entry)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar(f"loss/{monitor_name}_rolling" if rolling is not None else "loss/train_monitor", monitor, epoch)
        writer.add_scalar("time/validation_seconds", validation_seconds, epoch)
        if validation is not None:
            writer.add_scalar("validation/samples_seen", validation["samples_seen"], epoch)
            writer.add_scalar("validation/window_samples", validation["window_samples"], epoch)
        writer.add_scalar("time/epoch_seconds", entry["seconds"], epoch)
        writer.add_scalar("learning_rate", entry["learning_rate"], epoch)
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"epoch {epoch}/{args.epochs} train={train_loss:.6f} {monitor_name}={monitor:.6f}"
                  f"{' (rolling)' if rolling is not None else ''}", flush=True)
        if monitor < best_loss - args.min_delta:
            best_loss, best_epoch = monitor, epoch
            save(epoch, monitor)
        elif args.early_stopping_patience and epoch-best_epoch >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break
    return history
