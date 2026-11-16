"""Shared model construction and predictions for training, evaluation and inference."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import GainDataset, collate_gain_samples, collate_cameras
from experiments import pose_features
from features import FeatureStore, device_batch, file_sha256, compatible_implementation
from geometry import CAMERA_VERSION
from model import BACKBONES, ContextPoseCNN, FeatureGainHead, LVSM_CHECKPOINT_NAME
from paths import resolve_reference
from devices import resolve_device, compute_precision, autocast_for

MODELS = ("mean", "view-count", "geometry", "toy", "cached", "lvsm")


def create_model(config):
    kind = config["model_type"]
    if kind == "toy":
        return ContextPoseCNN(image_size=config["image_size"], dropout=config.get("dropout", 0.))
    if kind in {"cached", "lvsm"}:
        return FeatureGainHead(config["input_dim"], config.get("dropout", 0.),
                               config.get("head_hidden_dim", 64),
                               config.get("head_hidden_layers", 1))
    if kind in {"mean", "view-count", "geometry"}:
        return nn.Linear(config["input_dim"], 1)
    raise ValueError(f"Unknown model type {kind}")


def load_frozen_backbone(config, checkpoint_path=None, device="auto"):
    device = resolve_device(device)
    spec = config["backbone_spec"]
    if not compatible_implementation(spec):
        raise ValueError("Checkpoint uses a different camera/input implementation.")
    if spec["name"] not in BACKBONES:
        raise ValueError(f"Register the {spec['name']} backbone before online inference.")
    reference = checkpoint_path or config.get("backbone_checkpoint") or Path(__file__).parent / LVSM_CHECKPOINT_NAME
    reference = Path(reference)
    if reference.is_file() and file_sha256(reference) != spec["checkpoint_sha256"]:
        raise ValueError("Backbone checkpoint hash does not match the trained head.")
    backbone = BACKBONES[spec["name"]](reference, spec["image_size"], spec["latent_tokens"]).to(device).eval()
    if file_sha256(reference) != spec["checkpoint_sha256"]:
        raise ValueError("Downloaded backbone does not match this head; provide its original checkpoint.")
    backbone.compute_precision = compute_precision(spec["precision"], device)
    return backbone


def input_matrix(samples, config, features=None, backbone_checkpoint=None, device="auto"):
    device = resolve_device(device)
    kind = config["model_type"]
    if kind == "mean":
        return torch.zeros(len(samples), 1)
    if kind == "view-count":
        return pose_features(samples)[:, :1]
    if kind == "geometry":
        return pose_features(samples)
    if kind not in {"cached", "lvsm"}:
        return None
    if backbone_checkpoint is not None:
        spec = config["backbone_spec"]
        backbone = load_frozen_backbone(config, backbone_checkpoint, device)
        values = []
        contexts = {}
        from features import context_identity, image_hashes
        hashes = image_hashes(samples)
        with torch.no_grad():
            for sample in samples:
                key = context_identity(sample, hashes)
                with autocast_for(backbone.compute_precision, device):
                    if key not in contexts:
                        batch = device_batch(collate_gain_samples([sample], spec["image_size"]), device)
                        contexts[key] = backbone.encode_context(batch["images"], batch["context_poses"],
                                                                batch["mask"], batch["context_intrinsics"])
                    batch = device_batch(collate_cameras([sample]), device)
                    values.append(backbone.decode_candidates(contexts[key], batch["candidate_pose"][:, None],
                                                             batch["candidate_intrinsics"][:, None])[0, 0].float().cpu())
        return torch.stack(values)
    path = features or config.get("features_path")
    if not path or not Path(path).is_file():
        reference = config.get("backbone_checkpoint") or Path(__file__).parent / LVSM_CHECKPOINT_NAME
        return input_matrix(samples, config, backbone_checkpoint=reference, device=device)
    store = FeatureStore(path)
    if store.identity != config.get("feature_identity", store.identity):
        raise ValueError("Feature-cache backbone/configuration does not match this head.")
    return store.matrix(samples)


def normalize_matrix(matrix, config):
    return (matrix - config["input_mean"].to(matrix.device)) / config["input_std"].to(matrix.device)


def toy_loader(samples, config, shuffle=False, generator=None, device="auto"):
    device = resolve_device(device)
    return DataLoader(GainDataset(samples), batch_size=config["batch_size"], shuffle=shuffle, generator=generator,
                      num_workers=config.get("num_workers", 0),
                      persistent_workers=config.get("num_workers", 0) > 0, pin_memory=torch.device(device).type == "cuda",
                      collate_fn=partial(collate_gain_samples, image_size=config["image_size"]))


def forward_toy(model, batch, device):
    batch = device_batch(batch, device)
    return model(batch["images"], batch["context_poses"], batch["candidate_pose"], batch["mask"])


def predict_normalized(model, samples, config, device="auto", matrix=None):
    device = resolve_device(next(model.parameters()).device if device is None or str(device) == "auto" else device)
    if not samples:
        return torch.empty(0, 1)
    model.eval()
    values = []
    with torch.no_grad():
        if config["model_type"] == "toy":
            for batch in toy_loader(samples, config, device=device):
                values.append(forward_toy(model, batch, device).cpu())
        else:
            matrix = normalize_matrix(matrix, config)
            for chunk in matrix.split(config["batch_size"]):
                values.append(model(chunk.to(device)).cpu())
    return torch.cat(values)


def load_run(path, device="auto"):
    device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Legacy checkpoints use the old camera/mask pipeline; retrain with the current manifest.")
    if checkpoint.get("artifact_paths") == "checkpoint-relative":
        for name in ("dataset_path", "features_path", "backbone_checkpoint"):
            if checkpoint.get(name):
                checkpoint[name] = str(resolve_reference(checkpoint[name], Path(path).resolve().parent))
    model = create_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model.eval()


def score_samples(checkpoint, model, samples, features=None, device="auto", backbone_checkpoint=None):
    device = resolve_device(next(model.parameters()).device if device is None or str(device) == "auto" else device)
    matrix = input_matrix(samples, checkpoint, features, backbone_checkpoint, device)
    normalized = predict_normalized(model, samples, checkpoint, device, matrix)
    return normalized * checkpoint["target_std"] + checkpoint["target_mean"]
