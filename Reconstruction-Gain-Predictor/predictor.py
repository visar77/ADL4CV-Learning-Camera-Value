"""Direct gain inference from ordinary images and cameras; no manifests or labels."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geometry import geometry_features, to_opencv
from model import feature_head_inputs
from runtime import load_frozen_backbone, load_run, normalize_matrix
from devices import resolve_device, compute_precision, autocast_for


def candidate_utility(predicted_gain: torch.Tensor) -> torch.Tensor:
    """Predicted value when keeping the base is available at zero gain."""
    return predicted_gain.clamp_min(0)


def _context_images(images, image_size):
    if isinstance(images, (str, Path, Image.Image)) or getattr(images, "ndim", None) == 3:
        images = [images]
    tensors, sizes = [], []
    for image in images:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert("RGB")
        elif not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().permute(1, 2, 0).numpy()
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] not in {3, 4}:
                raise ValueError("Images must be RGB/RGBA arrays [H,W,C] or tensors [C,H,W].")
            if np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError("Floating RGB input must be finite and in [0,1].")
                image = np.round(image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError("Images must use uint8 [0,255] or floating point [0,1].")
            image = Image.fromarray(image)
        sizes.append(image.size)
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        tensors.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
    if not tensors:
        raise ValueError("At least one context image is required.")
    return torch.stack(tensors)[None], sizes


def _poses(values, expected=None):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0 or values.numel() % 16:
        raise ValueError("Camera poses must be 4x4 camera-to-world matrices.")
    values = values.reshape(-1, 4, 4)
    if not torch.isfinite(values).all() or (expected is not None and len(values) != expected):
        raise ValueError("Camera poses must be finite and match the image count.")
    if not torch.allclose(values[:, 3], torch.tensor([0., 0., 0., 1.]).expand(len(values), -1), atol=1e-5):
        raise ValueError("Camera-to-world poses must have bottom row [0,0,0,1].")
    return values


def _intrinsics(values, sizes, fov=None):
    count = len(sizes)
    dimensions = torch.tensor(sizes, dtype=torch.float32)
    if values is None:
        if fov is None or not math.isfinite(fov) or not 0 < fov < math.pi:
            raise ValueError("Provide intrinsics or camera_angle_x in radians (between 0 and pi).")
        fx = 1 / (2 * math.tan(fov / 2))
        result = torch.tensor([fx, fx, .5, .5]).repeat(count, 1)
        result[:, 1] *= dimensions[:, 0] / dimensions[:, 1]
    else:
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.shape == (3, 3):
            values = values[None].expand(count, -1, -1)
        elif values.shape == (4,):
            values = values[None].expand(count, -1)
        if values.shape == (count, 3, 3):
            if not torch.allclose(values[:, 0, 1], torch.zeros(count), atol=1e-6):
                raise ValueError("Skewed intrinsic matrices are not supported.")
            result = torch.stack((values[:, 0, 0] / dimensions[:, 0], values[:, 1, 1] / dimensions[:, 1],
                                  values[:, 0, 2] / dimensions[:, 0], values[:, 1, 2] / dimensions[:, 1]), -1)
        elif values.shape == (count, 4):
            result = values.clone()
        else:
            raise ValueError("Intrinsics must be pixel K [3,3]/[N,3,3] or normalized [4]/[N,4].")
    if not torch.isfinite(result).all() or (result[:, :2] <= 0).any():
        raise ValueError("Intrinsics must be finite with positive focal lengths.")
    return result


class GainEstimator:
    """Load a trained predictor once and score new candidate cameras.

    RGB loading/resizing, calibration conversion, ray generation and scene
    normalization happen inside predict(). No labels, PSNR, manifest or feature
    extraction command is required. Backbones are loaded lazily and reused.
    """
    def __init__(self, checkpoint, *, backbone_checkpoint=None, device="auto",
                 candidate_batch_size=1, backbone=None):
        if candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive.")
        device = resolve_device(device)
        self.config, self.head = load_run(checkpoint, device)
        self.device = device
        self.backbone_checkpoint = backbone_checkpoint
        self.candidate_batch_size = candidate_batch_size
        self.backbone = backbone.to(device).eval().requires_grad_(False) if backbone is not None else None

    def predict(self, context_images, context_poses, candidate_poses, *, context_intrinsics=None,
                candidate_intrinsics=None, camera_angle_x=None, candidate_image_size=None,
                camera_convention="blender"):
        """Return one gain per candidate, in input order, as a CPU tensor.

        Poses: [N,4,4] and [Q,4,4] camera-to-world matrices (or flattened).
        Intrinsics: normalized fx/W,fy/H,cx/W,cy/H or pixel K matrices. Pixel K
        uses original image sizes; candidate_image_size=(width,height) defaults
        to the first context image's size. A common horizontal FOV in radians is
        convenient when all images use the same camera. No candidate RGB is used.
        """
        images, sizes = _context_images(context_images, self.config["image_size"])
        context = to_opencv(_poses(context_poses, len(sizes)), camera_convention)
        candidates = to_opencv(_poses(candidate_poses), camera_convention)
        count = len(candidates)
        if candidate_image_size is None:
            candidate_image_size = sizes[0]
        if len(candidate_image_size) != 2 or min(candidate_image_size) <= 0:
            raise ValueError("candidate_image_size must be a positive (width,height) pair.")
        kind = self.config["model_type"]
        mask = torch.ones(1, len(sizes), dtype=torch.bool, device=self.device)
        poses = context.flatten(1)[None].to(self.device)
        queries = candidates.flatten(1)[None].to(self.device)
        values = []
        with torch.no_grad():
            if kind in {"cached", "lvsm"}:
                ck = _intrinsics(context_intrinsics, sizes, camera_angle_x)
                if candidate_intrinsics is None and camera_angle_x is None:
                    if not torch.allclose(ck, ck[:1].expand_as(ck)):
                        raise ValueError("Provide candidate_intrinsics when context camera intrinsics differ.")
                    candidate_intrinsics = ck[0]
                qk = _intrinsics(candidate_intrinsics, [candidate_image_size] * count, camera_angle_x)
                if self.backbone is None:
                    self.backbone = load_frozen_backbone(self.config, self.backbone_checkpoint, self.device)
                if self.backbone.feature_dim != self.config.get("embedding_dim", self.config["input_dim"]):
                    raise ValueError("Backbone feature width does not match the trained head.")
                precision = self.config["backbone_spec"].get("precision", "fp32")
                precision = getattr(self.backbone, "compute_precision", None) or compute_precision(precision, self.device)
                with autocast_for(precision, self.device):
                    state = self.backbone.encode_context(images.to(self.device), poses, mask, ck[None].to(self.device))
                    for start in range(0, count, self.candidate_batch_size):
                        q = queries[:, start:start+self.candidate_batch_size]
                        features = self.backbone.decode_candidates(
                            state, q,
                            qk[None, start:start+self.candidate_batch_size].to(self.device))[0].float()
                        # Head weights and cached feature normalization are FP32.
                        with torch.autocast(torch.device(self.device).type, enabled=False):
                            features = feature_head_inputs(self.config.get("head_type", "mlp"), features,
                                                           poses.expand(q.size(1), -1, -1), q[0],
                                                           mask.expand(q.size(1), -1))
                            normalized = normalize_matrix(features.cpu(), self.config).to(self.device)
                            values.append(self.head(normalized).cpu())
            elif kind == "toy":
                for start in range(0, count, self.candidate_batch_size):
                    q = queries[0, start:start+self.candidate_batch_size]
                    values.append(self.head(images.to(self.device).expand(len(q), -1, -1, -1, -1),
                                            poses.expand(len(q), -1, -1), q,
                                            mask.expand(len(q), -1)).cpu())
            else:
                features = geometry_features(poses.expand(count, -1, -1), queries[0], mask.expand(count, -1))
                if kind == "mean":
                    features = torch.zeros(count, 1)
                elif kind == "view-count":
                    features = features[:, :1]
                normalized = normalize_matrix(features.cpu(), self.config).to(self.device)
                values.append(self.head(normalized).cpu())
        normalized = torch.cat(values).flatten()
        result = normalized * self.config["target_std"] + self.config["target_mean"]
        if not torch.isfinite(result).all():
            raise ValueError("Predictor returned a non-finite gain.")
        return result

    def predict_utility(self, context_images, context_poses, candidate_poses, **camera_options):
        """Report nonnegative candidate utilities in the checkpoint's gain units."""
        gains = self.predict(context_images, context_poses, candidate_poses, **camera_options)
        return candidate_utility(gains)
