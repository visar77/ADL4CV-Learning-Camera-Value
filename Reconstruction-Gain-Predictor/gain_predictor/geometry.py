"""Calibrated OpenCV cameras and context-only scene normalization."""
from __future__ import annotations

from torch import Tensor
import torch

CAMERA_VERSION = "opencv-pixel-centers-context-frame-v1"
COVERAGE_DIM = 4
RELATIVE_POSE_DIM = 9  # Three offsets and the first two columns of a rotation.


def relative_camera_poses(poses: Tensor, candidate: Tensor, mask: Tensor) -> Tensor:
    """Each context camera in candidate coordinates, plus a trailing validity flag."""
    context = poses.reshape(*mask.shape, 4, 4)
    query = candidate.reshape(-1, 1, 4, 4)
    count = mask.sum(1, keepdim=True)
    if not count.all():
        raise ValueError("Every context must contain at least one valid camera.")

    centers = context[..., :3, 3]
    # Subtract one camera first so coincident centers have exactly zero spread.
    first = mask.long().argmax(1)
    offsets = centers - centers[torch.arange(len(centers), device=poses.device), first][:, None]
    centroid = offsets.masked_fill(~mask[..., None], 0).sum(1) / count
    squared_radius = (offsets - centroid[:, None]).square().sum(-1)
    spread = (squared_radius.masked_fill(~mask, 0).sum(1, keepdim=True) / count).sqrt()
    scale = torch.where(spread > 1e-6, spread, torch.ones_like(spread))

    world_to_query = query[..., :3, :3].transpose(-1, -2)
    displacement = (world_to_query @ (centers - query[..., :3, 3]).unsqueeze(-1)).squeeze(-1)
    translation = torch.asinh(displacement / scale[..., None])
    axes = world_to_query @ context[..., :3, :2]
    relation = torch.cat((translation, axes.transpose(-1, -2).flatten(-2)), dim=-1)
    return torch.cat((relation.masked_fill(~mask[..., None], 0), mask[..., None].to(poses.dtype)), dim=-1)


def camera_coverage(poses: Tensor, candidate: Tensor, mask: Tensor) -> Tensor:
    """Log context count and nearest, mean, farthest angular separations.

    On the renderer's centered orbit, squared chord distance between optical
    axes, divided by four, is 0 for the same direction and 1 for opposite views.
    """
    axes = poses.reshape(*mask.shape, 4, 4)[..., :3, 2]
    query = candidate.reshape(-1, 1, 4, 4)[..., :3, 2]
    separation = (axes - query).square().sum(-1) / 4
    count = mask.sum(1)
    if not count.all():
        raise ValueError("Every context must contain at least one valid camera.")
    return torch.stack((count.to(poses.dtype).log(),
                        separation.masked_fill(~mask, torch.inf).amin(1),
                        separation.masked_fill(~mask, 0).sum(1) / count,
                        separation.masked_fill(~mask, -torch.inf).amax(1)), dim=1)


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
