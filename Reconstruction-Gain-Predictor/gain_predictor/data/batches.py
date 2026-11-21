"""Load RGB tensors and pad variable-length camera sets for neural networks."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.nn.functional import interpolate
import numpy as np
import torch

from gain_predictor.geometry import geometry_features, to_opencv

from .records import target_values

DEFAULT_IMAGE_SIZE = 512


def collate_cameras(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    views = max(len(s.context_poses) for s in samples)
    poses = torch.eye(4).flatten().repeat(len(samples), views, 1)
    mask = torch.zeros(len(samples), views, dtype=torch.bool)
    intrinsics = torch.tensor([1., 1., .5, .5]).repeat(len(samples), views, 1)
    query_intrinsics = torch.tensor([1., 1., .5, .5]).repeat(len(samples), 1)
    queries = []
    calibrated = True
    for i, sample in enumerate(samples):
        count = len(sample.context_poses)
        poses[i, :count] = to_opencv(torch.tensor(sample.context_poses).reshape(-1, 4, 4),
                                     sample.camera_convention).flatten(1)
        queries.append(to_opencv(torch.tensor(sample.candidate_pose).reshape(4, 4),
                                 sample.camera_convention).flatten())
        mask[i, :count] = True
        if sample.context_intrinsics and sample.candidate_intrinsics:
            intrinsics[i, :count] = torch.tensor(sample.context_intrinsics)
            query_intrinsics[i] = torch.tensor(sample.candidate_intrinsics)
        else:
            calibrated = False
    return {"context_poses": poses, "candidate_pose": torch.stack(queries), "mask": mask,
            "context_intrinsics": intrinsics if calibrated else None,
            "candidate_intrinsics": query_intrinsics if calibrated else None}


@lru_cache(maxsize=64)
def _resized_image(path, image_size, modified_ns, file_size):
    """Bounded per-worker cache; changed files get a fresh cache key."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def collate_gain_samples(samples, image_size=DEFAULT_IMAGE_SIZE, *, target=None):
    if image_size < 1:
        raise ValueError("image_size must be positive")
    batch = collate_cameras(samples)
    images = torch.zeros(len(samples), batch["mask"].shape[1], 3, image_size, image_size, dtype=torch.uint8)
    for i, sample in enumerate(samples):
        for j, path in enumerate(sample.image_paths):
            stat = Path(path).stat()
            images[i, j] = _resized_image(str(path), image_size, stat.st_mtime_ns, stat.st_size)
    batch["images"] = images
    if target is not None:
        batch["targets"] = target_values(samples, target)
    return batch


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


def device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def pose_features(samples):
    batch = collate_cameras(samples)
    return geometry_features(batch["context_poses"], batch["candidate_pose"], batch["mask"])
