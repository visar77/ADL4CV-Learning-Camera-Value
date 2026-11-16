"""Model components for reconstruction-gain prediction."""

from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod

from geometry import apply_frame, context_frame, rays
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
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    elif images.is_floating_point():
        images = images.float()  # Floating input contract: [0, 1].
    else:
        raise TypeError("Images must be uint8 [0,255] or floating point [0,1].")
    batch, views, channels, height, width = images.shape
    if (height, width) != (image_size, image_size):
        images = interpolate(
            images.reshape(batch * views, channels, height, width),
            size=(image_size, image_size), mode="bilinear", align_corners=False,
        ).reshape(batch, views, channels, image_size, image_size)
    return images


def _normalize_lvsm_rgb(images: Tensor) -> Tensor:
    return images.float() * 2.0 - 1.0


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

    def forward(self, values: Tensor, valid_tokens: Tensor | None = None) -> Tensor:
        normalized = self.norm1(values)
        query, key, value = self.attn.to_qkv(normalized).chunk(3, dim=-1)
        shape = (*query.shape[:2], self.heads, self.head_dimension)
        query = self.attn.q_norm(query.reshape(shape))
        key = self.attn.k_norm(key.reshape(shape))
        value = value.reshape(shape)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=None if valid_tokens is None else valid_tokens[:, None, None, :],
        ).transpose(1, 2)
        values = values + self.attn.fc(attended.reshape_as(values))
        return values + self.mlp.mlp(self.norm2(values))


class _LVSMSceneEncoder(nn.Module):
    """Minimal inference/training adapter for the official LVSM state dict."""

    def __init__(self, latent_tokens: int = 3072, dimension: int = 768,
                 head_dimension: int = 64, encoder_depth: int = 12, decoder_depth: int = 12,
                 latent_capacity: int = 3072) -> None:
        super().__init__()
        self.n_light_field_latent = nn.Parameter(torch.empty(latent_capacity, dimension))
        nn.init.normal_(self.n_light_field_latent, std=.02)
        self.latent_tokens = latent_tokens
        self.image_tokenizer = nn.Sequential(
            nn.Identity(), nn.Linear(576, dimension, bias=False)
        )
        self.target_pose_tokenizer = nn.Sequential(
            nn.Identity(), nn.Linear(384, dimension, bias=False)
        )
        self.transformer_encoder = nn.ModuleList([_LVSMBlock(dimension, head_dimension) for _ in range(encoder_depth)])
        self.transformer_decoder = nn.ModuleList([_LVSMBlock(dimension, head_dimension) for _ in range(decoder_depth)])
        self.transformer_input_layernorm_decoder = nn.LayerNorm(dimension, bias=False)

    def load_checkpoint(self, checkpoint: Path, url: str = LVSM_CHECKPOINT_URL) -> None:
        if not checkpoint.is_file():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name(checkpoint.name + ".part")
            print(f"Downloading LVSM backbone to {checkpoint}...")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(checkpoint)
            print("Finished downloading LVSM backbone.")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("model"), dict):
            raise ValueError("LVSM checkpoint must contain a 'model' state dict.")
        source = loaded["model"]
        expected = self.state_dict()
        missing = sorted(expected.keys() - source.keys())
        unexpected = sorted(k for k in source.keys() - expected.keys()
                            if not k.startswith(("image_token_decoder.", "loss_computer.")))
        mismatched = [k for k in expected.keys() & source.keys()
                      if expected[k].shape != source[k].shape]
        if missing or unexpected or mismatched:
            raise ValueError(f"Incompatible LVSM checkpoint: missing={missing}, "
                             f"unexpected={unexpected}, shape_mismatch={mismatched}")
        self.load_state_dict({k: source[k] for k in expected}, strict=True, assign=True)

    _rays = staticmethod(rays)

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

    def encode_context(self, images, context_poses, mask, context_intrinsics):
        batch, views, _, height, width = images.shape
        if height % LVSM_PATCH_SIZE or width % LVSM_PATCH_SIZE:
            raise ValueError("LVSM image dimensions must be divisible by 8.")
        context_poses = context_poses.reshape(batch, views, 4, 4)
        frame, scale = context_frame(context_poses, mask)
        context_poses = apply_frame(context_poses, frame, scale)
        intrinsics = torch.where(mask[..., None], context_intrinsics,
                                 context_intrinsics.new_tensor([1, 1, 0.5, 0.5]))
        ray_o, ray_d = rays(context_poses, intrinsics, height, width)
        posed = torch.cat((_normalize_lvsm_rgb(images), self._pose_condition(ray_o, ray_d)), dim=2)
        tokens = self._tokenize(self.image_tokenizer, posed)
        patches = tokens.shape[1]
        tokens = tokens.reshape(batch, views * patches, -1)
        latent = self.n_light_field_latent[:self.latent_tokens].unsqueeze(0).expand(batch, -1, -1)
        valid = torch.cat((torch.ones(batch, self.latent_tokens, device=mask.device, dtype=torch.bool),
                           mask.repeat_interleave(patches, dim=1)), dim=1)
        encoded = torch.cat((latent, tokens), dim=1)
        for layer in self.transformer_encoder:
            encoded = layer(encoded, valid)
        return {"latents": encoded[:, :self.latent_tokens], "frame": frame, "scale": scale}

    def decode_candidates(self, context, candidate_pose, candidate_intrinsics, height, width):
        batch, candidates = candidate_pose.shape[:2]
        poses = apply_frame(candidate_pose.reshape(batch, candidates, 4, 4),
                            context["frame"], context["scale"])
        ray_o, ray_d = rays(poses, candidate_intrinsics, height, width)
        tokens = self._tokenize(self.target_pose_tokenizer, self._pose_condition(ray_o, ray_d))
        latent = context["latents"][:, None].expand(-1, candidates, -1, -1).reshape(
            batch * candidates, self.latent_tokens, -1)
        values = self.transformer_input_layernorm_decoder(torch.cat((tokens, latent), dim=1))
        for layer in self.transformer_decoder:
            values = layer(values)
        return values[:, :tokens.shape[1]].mean(dim=1).reshape(batch, candidates, -1)

    def forward(self, images, context_poses, candidate_pose, mask,
                context_intrinsics, candidate_intrinsics):
        context = self.encode_context(images, context_poses, mask, context_intrinsics)
        return self.decode_candidates(context, candidate_pose[:, None],
                                      candidate_intrinsics[:, None], *images.shape[-2:])[:, 0]


class FeatureBackbone(nn.Module, ABC):
    """Replaceable frozen extractor. Cameras use OpenCV c2w convention.

    Context state is opaque to callers. Candidate poses are [B,Q,16] and
    normalized intrinsics are [B,Q,4]; decoded features are [B,Q,D].
    """
    feature_dim: int

    @abstractmethod
    def encode_context(self, images, context_poses, mask, context_intrinsics):
        raise NotImplementedError

    @abstractmethod
    def decode_candidates(self, context, candidate_poses, candidate_intrinsics):
        raise NotImplementedError


class LVSMFeatureBackbone(FeatureBackbone):
    feature_dim = 768

    def __init__(self, checkpoint: Path, image_size=DEFAULT_IMAGE_SIZE,
                 latent_tokens=3072, checkpoint_url=LVSM_CHECKPOINT_URL):
        super().__init__()
        if image_size < 8 or image_size % 8 or not 1 <= latent_tokens <= 3072:
            raise ValueError("LVSM requires an image size divisible by 8 and 1..3072 latents.")
        self.image_size = image_size
        with torch.device("meta"):
            self.encoder = _LVSMSceneEncoder(latent_tokens)
        self.encoder.load_checkpoint(Path(checkpoint), checkpoint_url)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def encode_context(self, images, context_poses, mask, context_intrinsics):
        if context_intrinsics is None:
            raise ValueError("Missing intrinsics. Regenerate the manifest with preprocess.py.")
        with torch.no_grad():
            return self.encoder.encode_context(prepare_images(images, self.image_size),
                                               context_poses, mask, context_intrinsics)

    def decode_candidates(self, context, candidate_poses, candidate_intrinsics):
        with torch.no_grad():
            return self.encoder.decode_candidates(context, candidate_poses, candidate_intrinsics,
                                                   self.image_size, self.image_size)


BACKBONES = {"lvsm": LVSMFeatureBackbone}


def register_backbone(name: str, factory):
    """Register a FeatureBackbone factory without changing the feature trainer."""
    if name in BACKBONES:
        raise ValueError(f"Backbone already registered: {name}")
    BACKBONES[name] = factory


class FeatureGainHead(nn.Sequential):
    def __init__(self, feature_dim: int, dropout: float = 0.0,
                 hidden_dim: int = 64, hidden_layers: int = 1):
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("Gain-head width and hidden-layer count must be positive.")
        layers = []
        input_dim = feature_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        super().__init__(*layers)


class LVSMGainPredictor(nn.Module):
    """Online wrapper; training uses the same head on cached features."""
    def __init__(self, checkpoint: Path, feature_dim=768, image_size=DEFAULT_IMAGE_SIZE,
                 latent_tokens=3072, checkpoint_url=LVSM_CHECKPOINT_URL, dropout=0.0,
                 head_hidden_dim=64, head_hidden_layers=1):
        super().__init__()
        self.backbone = LVSMFeatureBackbone(checkpoint, image_size, latent_tokens, checkpoint_url)
        self.regression_head = FeatureGainHead(self.backbone.feature_dim, dropout,
                                               head_hidden_dim, head_hidden_layers)

    def forward(self, images, context_poses, candidate_pose, mask,
                context_intrinsics=None, candidate_intrinsics=None):
        context = self.backbone.encode_context(images, context_poses, mask, context_intrinsics)
        encoded = self.backbone.decode_candidates(context, candidate_pose[:, None], candidate_intrinsics[:, None])[:, 0]
        return self.regression_head(encoded)
