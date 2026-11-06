"""Model components for reconstruction-gain prediction.

``LVSMBackbone`` is the integration boundary for the eventual LVSM encoder.
The current checkpoint-compatible adapter consumes the metadata feature vector
because renderer records do not yet contain a batched image-view tensor.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch import Tensor, nn


class LVSMBackbone(nn.Module):
    """Frozen LVSM adapter with a downloadable checkpoint.

    The small adapter is deliberately checkpoint-compatible and replaceable:
    when the real LVSM encoder/input representation is available, only this
    class needs to change. A checkpoint or URL must be supplied for a normal
    training run; tests inject a mock predictor before this class is created.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 128,
        checkpoint: Path | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
        )
        self._load_checkpoint(checkpoint, url)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def _load_checkpoint(self, checkpoint: Path | None, url: str | None) -> None:
        if checkpoint is None:
            raise ValueError("A backbone checkpoint path is required.")
        if not checkpoint.is_file():
            if not url:
                raise FileNotFoundError(
                    f"Backbone checkpoint not found at {checkpoint}. "
                    "Provide --backbone-url to download it."
                )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.load_state_dict(state)

    def forward(self, features: Tensor) -> Tensor:
        return self.encoder(features)


class GainPredictor(nn.Module):
    """Frozen LVSM backbone followed by a trainable single-value MLP head."""

    def __init__(
        self,
        input_dim: int,
        backbone_checkpoint: Path | None,
        backbone_url: str | None,
        backbone_feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = LVSMBackbone(
            input_dim=input_dim,
            feature_dim=backbone_feature_dim,
            checkpoint=backbone_checkpoint,
            url=backbone_url,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(backbone_feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            encoded = self.backbone(features)
        return self.regression_head(encoded)
