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
from torch.nn.functional import interpolate


def prepare_images(images: Tensor, image_size: int) -> Tensor:
    """Convert RGB tensors from uint8/0..255 into normalized model input."""
    images = images.float()
    if images.max() > 1:
        images = images / 255.0
    batch, views, channels, height, width = images.shape
    images = interpolate(
        images.reshape(batch * views, channels, height, width),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return images.reshape(batch, views, channels, image_size, image_size)


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


class ContextPoseCNN(nn.Module):
    """CPU toy encoder with an LVSM-shaped context/query boundary.

    The public forward contract is context images and poses plus one candidate
    pose. No candidate image is accepted, preventing inference-time leakage.
    """

    def __init__(
        self, pose_dim: int = 16, context_dim: int = 64, image_size: int = 128
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, context_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.context_pose = nn.Sequential(nn.Linear(pose_dim, context_dim), nn.GELU())
        self.candidate_pose = nn.Sequential(nn.Linear(pose_dim, context_dim), nn.GELU())
        self.regression_head = nn.Sequential(
            nn.Linear(context_dim * 3, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        images: Tensor,
        context_poses: Tensor,
        candidate_pose: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if images.ndim != 5 or context_poses.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "Expected images [B,N,3,H,W], poses [B,N,P], and mask [B,N]"
            )
        if (
            images.shape[:2] != context_poses.shape[:2]
            or images.shape[:2] != mask.shape
        ):
            raise ValueError("Context image, pose, and mask dimensions must agree")
        images = prepare_images(images, self.image_size)
        batch, views, channels, height, width = images.shape
        tokens = self.image_encoder(
            images.reshape(batch * views, channels, height, width)
        )
        tokens = tokens.flatten(1).reshape(batch, views, -1)
        tokens = tokens + self.context_pose(context_poses)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        query = self.candidate_pose(candidate_pose)
        return self.regression_head(torch.cat((pooled, query, pooled * query), dim=-1))


class LVSMGainPredictor(nn.Module):
    """Trainable gain head around a downloaded LVSM context/query module.

    The serialized LVSM module must implement
    ``forward(images, context_poses, candidate_pose, mask)`` and return one
    embedding per batch item.
    """

    def __init__(
        self, checkpoint: Path, feature_dim: int = 128, image_size: int = 128
    ) -> None:
        super().__init__()
        self.image_size = image_size
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded, nn.Module):
            raise ValueError(
                "LVSM checkpoint must be a serialized torch.nn.Module. "
                "A raw state_dict is not enough to determine the LVSM input contract."
            )
        self.lvsm = loaded
        for parameter in self.lvsm.parameters():
            parameter.requires_grad = False
        self.lvsm.eval()
        self.regression_head = nn.Sequential(
            nn.LazyLinear(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        images: Tensor,
        context_poses: Tensor,
        candidate_pose: Tensor,
        mask: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            encoded = self.lvsm(
                prepare_images(images, self.image_size),
                context_poses,
                candidate_pose,
                mask,
            )
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        if encoded.ndim != 2:
            raise ValueError("LVSM must return an embedding shaped [batch, features].")
        return self.regression_head(encoded)
