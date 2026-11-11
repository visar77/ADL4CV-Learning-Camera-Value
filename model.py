"""Model components for reconstruction-gain prediction."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.functional import interpolate

DEFAULT_IMAGE_SIZE = 512
LVSM_PATCH_SIZE = 8
LVSM_CHECKPOINT_NAME = "lvsm_scene_encoder_decoder_res512.pt"
LVSM_CHECKPOINT_URL = (
    "https://huggingface.co/coast01/LVSM/resolve/main/"
    "lvsm_scene_encoder_decoder_res512.pt"
)


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
        self, pose_dim: int = 16, context_dim: int = 64, image_size: int = 32
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


class _RMSNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, values: Tensor) -> Tensor:
        normalized = values * torch.rsqrt(
            values.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5
        )
        return normalized.type_as(values) * self.weight.type_as(values)


class _LVSMBlock(nn.Module):
    def __init__(self, dimension: int = 768, head_dimension: int = 64) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, bias=False)
        self.attn = nn.Module()
        self.attn.to_qkv = nn.Linear(dimension, 3 * dimension, bias=False)
        self.attn.fc = nn.Linear(dimension, dimension, bias=False)
        self.attn.q_norm = _RMSNorm(head_dimension)
        self.attn.k_norm = _RMSNorm(head_dimension)
        self.norm2 = nn.LayerNorm(dimension, bias=False)
        self.mlp = nn.Module()
        self.mlp.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension, bias=False),
            nn.GELU(),
            nn.Linear(4 * dimension, dimension, bias=False),
            nn.Dropout(0.0),
        )
        self.heads = dimension // head_dimension
        self.head_dimension = head_dimension

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm1(values)
        query, key, value = self.attn.to_qkv(normalized).chunk(3, dim=-1)
        shape = (*query.shape[:2], self.heads, self.head_dimension)
        query = self.attn.q_norm(query.reshape(shape))
        key = self.attn.k_norm(key.reshape(shape))
        value = value.reshape(shape)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        values = values + self.attn.fc(attended.reshape_as(values))
        return values + self.mlp.mlp(self.norm2(values))


class _LVSMSceneEncoder(nn.Module):
    """Minimal inference/training adapter for the official LVSM state dict."""

    def __init__(self, latent_tokens: int = 3072) -> None:
        super().__init__()
        self.n_light_field_latent = nn.Parameter(torch.empty(3072, 768))
        self.latent_tokens = latent_tokens
        self.image_tokenizer = nn.Sequential(
            nn.Identity(), nn.Linear(576, 768, bias=False)
        )
        self.target_pose_tokenizer = nn.Sequential(
            nn.Identity(), nn.Linear(384, 768, bias=False)
        )
        self.transformer_encoder = nn.ModuleList([_LVSMBlock() for _ in range(12)])
        self.transformer_decoder = nn.ModuleList([_LVSMBlock() for _ in range(12)])
        self.transformer_input_layernorm_decoder = nn.LayerNorm(768, bias=False)

    def load_checkpoint(self, checkpoint: Path, url: str = LVSM_CHECKPOINT_URL) -> None:
        if not checkpoint.is_file():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name(checkpoint.name + ".part")
            print(f"Downloading LVSM backbone to {checkpoint}...")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(checkpoint)
            print("Finished downloading LVSM backbone.")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("model"), dict):
            raise ValueError("LVSM checkpoint must contain a 'model' state dict.")
        state = {
            key: value
            for key, value in loaded["model"].items()
            if key in self.state_dict()
        }
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(key.startswith("loss_computer.") for key in missing):
            raise ValueError("LVSM checkpoint contains an incompatible model state.")
        required = {"n_light_field_latent", "image_tokenizer.1.weight"}
        if not required.issubset(state):
            raise ValueError("LVSM checkpoint is missing scene encoder weights.")

    @staticmethod
    def _rays(poses: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
        device = poses.device
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        directions = torch.stack((x, y, torch.ones_like(x)), dim=-1)
        rotation = poses[..., :3, :3]
        origins = poses[..., :3, 3]
        ray_directions = torch.einsum("bvij,hwj->bvihw", rotation, directions)
        ray_origins = origins[..., :, None, None].expand_as(ray_directions)
        return ray_origins, torch.nn.functional.normalize(ray_directions, dim=2)

    @staticmethod
    def _pose_condition(ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
        return torch.cat(
            (torch.cross(ray_origins, ray_directions, dim=2), ray_directions), dim=2
        )

    @staticmethod
    def _tokenize(tokenizer: nn.Module, values: Tensor) -> Tensor:
        batch, views, channels, height, width = values.shape
        patch = LVSM_PATCH_SIZE
        values = values.reshape(
            batch * views,
            channels,
            height // patch,
            patch,
            width // patch,
            patch,
        ).permute(0, 2, 4, 3, 5, 1)
        return tokenizer[1](values.reshape(batch * views, -1, channels * patch * patch))

    def forward(
        self,
        images: Tensor,
        context_poses: Tensor,
        candidate_pose: Tensor,
        mask: Tensor,
    ) -> Tensor:
        batch, views, _, height, width = images.shape
        if height % LVSM_PATCH_SIZE or width % LVSM_PATCH_SIZE:
            raise ValueError("LVSM image dimensions must be divisible by 8.")
        context_poses = context_poses.reshape(batch, views, 4, 4)
        candidate_pose = candidate_pose.reshape(batch, 1, 4, 4)
        ray_o, ray_d = self._rays(context_poses, height, width)
        posed_images = torch.cat(
            (images.float() / 255.0 * 2.0 - 1.0, self._pose_condition(ray_o, ray_d)),
            dim=2,
        )
        image_tokens = self._tokenize(self.image_tokenizer, posed_images)
        image_tokens = image_tokens.reshape(batch, views * image_tokens.shape[1], -1)
        patch_count = image_tokens.shape[1] // views
        image_mask = mask.repeat_interleave(patch_count, dim=1).unsqueeze(-1)
        image_tokens = image_tokens * image_mask.to(image_tokens.dtype)
        latent = (
            self.n_light_field_latent[: self.latent_tokens]
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        encoded = torch.cat((latent, image_tokens), dim=1)
        for layer in self.transformer_encoder:
            encoded = layer(encoded)
        latent = encoded[:, : self.latent_tokens]

        target_o, target_d = self._rays(candidate_pose, height, width)
        target_condition = self._pose_condition(target_o, target_d)
        target_tokens = self._tokenize(self.target_pose_tokenizer, target_condition)
        decoder_input = torch.cat((target_tokens, latent), dim=1)
        decoder_input = self.transformer_input_layernorm_decoder(decoder_input)
        for layer in self.transformer_decoder:
            decoder_input = layer(decoder_input)
        return decoder_input[:, : target_tokens.shape[1]].mean(dim=1)


class LVSMGainPredictor(nn.Module):
    """Trainable gain head around a downloaded LVSM context/query module.

    The checkpoint is the official LVSM training checkpoint containing a
    ``model`` state dict. The LVSM scene encoder is frozen and its
    candidate-conditioned token embedding feeds the trainable gain head.
    """

    def __init__(
        self,
        checkpoint: Path,
        feature_dim: int = 128,
        image_size: int = DEFAULT_IMAGE_SIZE,
        latent_tokens: int = 3072,
        checkpoint_url: str = LVSM_CHECKPOINT_URL,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        if latent_tokens < 1 or latent_tokens > 3072:
            raise ValueError("latent_tokens must be between 1 and 3072.")
        self.lvsm = _LVSMSceneEncoder(latent_tokens=latent_tokens)
        self.lvsm.load_checkpoint(checkpoint, checkpoint_url)
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
