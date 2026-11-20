"""Prepare model inputs and score labelled records for training and evaluation.

For new images/cameras, use the public GainEstimator API in inference.py.
"""
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader
import torch

from gain_predictor.artifacts import image_hashes
from gain_predictor.checkpoints import load_frozen_backbone
from gain_predictor.data.batches import (
    collate_cameras,
    collate_gain_samples,
    device_batch,
    pose_features,
)
from gain_predictor.devices import autocast_for, resolve_device
from gain_predictor.feature_cache import FeatureStore, context_identity
from gain_predictor.models.heads import feature_head_inputs


def head_matrix(samples, embeddings, config):
    head_type = config.get("head_type", "mlp")
    if head_type == "mlp":
        return embeddings
    batch = collate_cameras(samples)
    return feature_head_inputs(head_type, embeddings, batch["context_poses"],
                               batch["candidate_pose"], batch["mask"])


def input_matrix(samples, config, features=None, backbone_checkpoint=None, device="auto"):
    """Build the inputs for a linear baseline or a frozen feature head."""
    kind = config["model_type"]
    if kind == "mean":
        return torch.zeros(len(samples), 1)
    if kind == "view-count":
        return pose_features(samples)[:, :1]
    if kind == "geometry":
        return pose_features(samples)
    if kind not in {"cached", "lvsm"}:
        return None
    path = features or config.get("features_path")
    if backbone_checkpoint is None and path and Path(path).is_file():
        store = FeatureStore(path)
        if store.identity != config.get("feature_identity", store.identity):
            raise ValueError("Feature-cache backbone/configuration does not match this head.")
        embeddings = store.matrix(samples)
    else:
        embeddings = _encode_samples(samples, config, backbone_checkpoint, resolve_device(device))
    return head_matrix(samples, embeddings, config)


@torch.no_grad()
def _encode_samples(samples, config, backbone_checkpoint, device):
    backbone = load_frozen_backbone(config, backbone_checkpoint, device)
    image_size = config["backbone_spec"]["image_size"]
    hashes = image_hashes(samples)
    contexts = {}
    values = []
    with autocast_for(backbone.compute_precision, device):
        for sample in samples:
            key = context_identity(sample, hashes)
            if key in contexts:
                batch = device_batch(collate_cameras([sample]), device)
            else:
                batch = device_batch(collate_gain_samples([sample], image_size), device)
                contexts[key] = backbone.encode_context(
                    batch["images"], batch["context_poses"], batch["mask"], batch["context_intrinsics"],
                )
            embedding = backbone.decode_candidates(
                contexts[key], batch["candidate_pose"][:, None], batch["candidate_intrinsics"][:, None],
            )
            values.append(embedding[0, 0].float().cpu())
    return torch.stack(values)


def normalize_matrix(matrix, config):
    mean = config["input_mean"].to(matrix.device)
    std = config["input_std"].to(matrix.device)
    if config.get("head_type") == "pose":
        width = config["embedding_dim"]
        embedding = (matrix[:, :width] - mean) / std
        return torch.cat((embedding, matrix[:, width:]), dim=1)
    return (matrix - mean) / std


def toy_loader(samples, config, shuffle=False, generator=None, device="auto", *, target=None):
    device = resolve_device(device)
    return DataLoader(samples, batch_size=config["batch_size"], shuffle=shuffle, generator=generator,
                      num_workers=config.get("num_workers", 0),
                      persistent_workers=config.get("num_workers", 0) > 0, pin_memory=torch.device(device).type == "cuda",
                      collate_fn=partial(collate_gain_samples, image_size=config["image_size"], target=target))


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


def score_samples(checkpoint, model, samples, features=None, device="auto", backbone_checkpoint=None):
    device = resolve_device(next(model.parameters()).device if device is None or str(device) == "auto" else device)
    matrix = input_matrix(samples, checkpoint, features, backbone_checkpoint, device)
    normalized = predict_normalized(model, samples, checkpoint, device, matrix)
    return normalized * checkpoint["target_std"] + checkpoint["target_mean"]
