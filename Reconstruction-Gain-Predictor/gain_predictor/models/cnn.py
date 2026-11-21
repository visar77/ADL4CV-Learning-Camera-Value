"""Small trainable RGB-and-camera CNN for smoke tests and baseline experiments."""
from torch import Tensor, nn
import torch

from gain_predictor.data.batches import prepare_images


class ContextPoseCNN(nn.Module):
    """CPU toy encoder with an LVSM-shaped context/query boundary.

    The public forward contract is context images and poses plus one candidate
    pose. No candidate image is accepted, preventing inference-time leakage.
    """

    def __init__(
        self, pose_dim: int = 16, context_dim: int = 64, image_size: int = 32, dropout: float = 0.0
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
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        images: Tensor,
        context_poses: Tensor,
        candidate_pose: Tensor,
        mask: Tensor,
        **unused,
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
