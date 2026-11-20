"""Construct trainable models and register frozen feature backbones."""
from torch import nn

from .cnn import ContextPoseCNN
from .heads import DEFAULT_HEAD_WIDTH, FEATURE_HEADS
from .lvsm import LVSMFeatureBackbone

MODELS = ("mean", "view-count", "geometry", "toy", "cached", "lvsm")
BACKBONES = {"lvsm": LVSMFeatureBackbone}


def register_backbone(name: str, factory):
    """Register a FeatureBackbone factory without changing the feature trainer."""
    if name in BACKBONES:
        raise ValueError(f"Backbone already registered: {name}")
    BACKBONES[name] = factory


def create_model(config):
    kind = config["model_type"]
    if kind == "toy":
        return ContextPoseCNN(image_size=config["image_size"], dropout=config.get("dropout", 0.))
    if kind in {"cached", "lvsm"}:
        head_type = config.get("head_type", "mlp")
        return FEATURE_HEADS[head_type](config["input_dim"], config.get("dropout", 0.),
                                        config.get("head_hidden_dim", DEFAULT_HEAD_WIDTH[head_type]),
                                        config.get("head_hidden_layers", 1))
    if kind in {"mean", "view-count", "geometry"}:
        return nn.Linear(config["input_dim"], 1)
    raise ValueError(f"Unknown model type {kind}")
