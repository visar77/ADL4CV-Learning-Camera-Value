"""Calibrated OpenCV cameras and context-only scene normalization."""
from __future__ import annotations

import torch
from torch import Tensor

CAMERA_VERSION = "opencv-pixel-centers-context-frame-v1"


def to_opencv(poses: Tensor, convention: str) -> Tensor:
    if convention not in {"blender", "opencv"}:
        raise ValueError(f"Unknown camera convention: {convention}")
    result = poses.clone()
    if convention == "blender":
        result[..., :3, 1:3] *= -1
    return result


def context_frame(poses: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Use first valid orientation, mean context center, and context extent only.

    Avoid averaging directions: cameras surrounding an object can cancel out.
    The fallback unit scale also supports a single-view overfitting experiment.
    """
    if not mask.any(dim=1).all():
        raise ValueError("Every context must contain at least one valid camera.")
    weights = mask.to(poses.dtype)
    center = (poses[..., :3, 3] * weights[..., None]).sum(1) / weights.sum(1)[:, None]
    first = mask.long().argmax(dim=1)
    rotation = poses[torch.arange(len(poses), device=poses.device), first, :3, :3].transpose(-1, -2)
    frame = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(len(poses), 1, 1)
    frame[:, :3, :3] = rotation
    frame[:, :3, 3] = -(rotation @ center[..., None]).squeeze(-1)
    relative = frame[:, None] @ poses
    extent = (relative[..., :3, 3].abs() * weights[..., None]).amax(dim=(1, 2))
    scale = torch.where(extent > 1e-6, 1.35 * extent, torch.ones_like(extent))
    return frame, scale


def apply_frame(poses: Tensor, frame: Tensor, scale: Tensor) -> Tensor:
    result = frame[:, None] @ poses
    result[..., :3, 3] /= scale[:, None, None]
    return result


def rays(poses: Tensor, intrinsics: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Intrinsics are normalized (fx/W, fy/H, cx/W, cy/H), after resize/crop."""
    if intrinsics.shape != (*poses.shape[:2], 4):
        raise ValueError("Calibrated intrinsics [batch, views, 4] are required.")
    if not torch.isfinite(intrinsics).all() or (intrinsics[..., :2] <= 0).any():
        raise ValueError("Intrinsics must be finite with positive focal lengths.")
    y, x = torch.meshgrid(
        (torch.arange(height, device=poses.device, dtype=poses.dtype) + 0.5) / height,
        (torch.arange(width, device=poses.device, dtype=poses.dtype) + 0.5) / width,
        indexing="ij",
    )
    fx, fy, cx, cy = intrinsics.unbind(-1)
    dx = (x - cx[..., None, None]) / fx[..., None, None]
    dy = (y - cy[..., None, None]) / fy[..., None, None]
    directions = torch.stack((dx, dy, torch.ones_like(dx)), dim=2)
    directions = torch.einsum("bvij,bvjhw->bvihw", poses[..., :3, :3], directions)
    directions = torch.nn.functional.normalize(directions, dim=2)
    origins = poses[..., :3, 3, None, None].expand_as(directions)
    return origins, directions


def geometry_features(poses: Tensor, candidate: Tensor, mask: Tensor) -> Tensor:
    """Pose-only features; no measured base/test quality is used as input."""
    poses = poses.reshape(*mask.shape, 4, 4)
    candidate = candidate.reshape(-1, 1, 4, 4)
    frame, scale = context_frame(poses, mask)
    poses = apply_frame(poses, frame, scale)
    candidate = apply_frame(candidate, frame, scale)
    distance = (poses[..., :3, 3] - candidate[..., :3, 3]).norm(dim=-1)
    cosine = (poses[..., :3, 2] * candidate[..., :3, 2]).sum(-1).clamp(-1, 1)
    angle = torch.acos(cosine) / torch.pi
    weights = mask.to(poses.dtype)
    count = weights.sum(1)
    def stats(values):
        return (values.masked_fill(~mask, float("inf")).amin(1),
                (values * weights).sum(1) / count,
                values.masked_fill(~mask, float("-inf")).amax(1))
    return torch.stack((count.log(), *stats(distance), *stats(angle),
                        *candidate[:, 0, :3, 3].unbind(-1)), dim=1)
