"""Load trained gain models and their matching frozen backbone weights."""
from pathlib import Path

import torch

from gain_predictor.artifacts import PROJECT_ROOT, file_sha256, resolve_reference
from gain_predictor.devices import compute_precision, resolve_device
from gain_predictor.feature_cache import compatible_implementation
from gain_predictor.models import BACKBONES, create_model
from gain_predictor.models.lvsm import LVSM_CHECKPOINT_NAME


def load_frozen_backbone(config, checkpoint_path=None, device="auto"):
    device = resolve_device(device)
    spec = config["backbone_spec"]
    if not compatible_implementation(spec):
        raise ValueError("Checkpoint uses a different camera/input implementation.")
    if spec["name"] not in BACKBONES:
        raise ValueError(f"Register the {spec['name']} backbone before online inference.")
    reference = checkpoint_path or config.get("backbone_checkpoint") or PROJECT_ROOT / LVSM_CHECKPOINT_NAME
    reference = Path(reference)
    if reference.is_file() and file_sha256(reference) != spec["checkpoint_sha256"]:
        raise ValueError("Backbone checkpoint hash does not match the trained head.")
    backbone = BACKBONES[spec["name"]](reference, spec["image_size"], spec["latent_tokens"]).to(device).eval()
    if file_sha256(reference) != spec["checkpoint_sha256"]:
        raise ValueError("Downloaded backbone does not match this head; provide its original checkpoint.")
    backbone.compute_precision = compute_precision(spec["precision"], device)
    return backbone


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
