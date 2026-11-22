"""Render FastGS checkpoints from the pipeline overview camera."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

FASTGS = Path(__file__).resolve().parents[3] / "FastGS"
sys.path.insert(0, str(FASTGS))
from gaussian_renderer import render_fastgs
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix

WIDTH, HEIGHT = 1600, 1200
# Blender's 0.8-strength world background under the Standard view transform.
OVERVIEW_BACKGROUND = 231 / 255


def camera_from_blender(metadata, matrix, width, height, width_in_radii):
    c2w = np.asarray(matrix, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda").T
    fovx = 2 * math.atan(width_in_radii / 20)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)
    znear, zfar = 0.01, max(100.0, metadata["bounding_radius"] * 30)
    projection = getProjectionMatrix(znear, zfar, fovx, fovy).T.cuda()
    return MiniCam(width, height, fovy, fovx, znear, zfar, view, view @ projection)


def overview_camera(metadata, width_in_radii):
    center = np.asarray(metadata["center"], dtype=np.float64)
    radius = metadata["bounding_radius"]
    direction = np.array([0.42, -0.83, 0.37], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, up, direction))
    c2w[:3, 3] = center + direction * radius * 20
    return camera_from_blender(metadata, c2w, WIDTH, HEIGHT, width_in_radii)


def load_gaussians(checkpoint):
    properties = PlyData.read(str(checkpoint))["vertex"].properties
    count = sum(prop.name.startswith("f_rest_") for prop in properties)
    degree = math.isqrt(count // 3 + 1) - 1
    if count != 3 * ((degree + 1) ** 2 - 1):
        raise ValueError(f"Invalid spherical harmonic properties in {checkpoint}")
    gaussians = GaussianModel(degree)
    gaussians.load_ply(str(checkpoint))
    return gaussians


def render_transparent(camera, gaussians, pipe, black, white):
    dark = render_fastgs(camera, gaussians, pipe, black, 0.5)["render"]
    light = render_fastgs(camera, gaussians, pipe, white, 0.5)["render"]
    alpha = (1 - (light - dark).mean(dim=0, keepdim=True)).clamp(0, 1)
    rgb = torch.where(alpha > 1e-6, dark / alpha.clamp_min(1e-6), 0).clamp(0, 1)
    return torch.cat((rgb, alpha), dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transparent-output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    gaussians = load_gaussians(args.checkpoint)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    white = torch.ones(3, device="cuda")
    black = torch.zeros(3, device="cuda")
    background = torch.full((3,), OVERVIEW_BACKGROUND, device="cuda")
    with torch.no_grad():
        close = render_fastgs(overview_camera(metadata, 3.4), gaussians, pipe, background, 0.5)["render"]
        wide_camera = overview_camera(metadata, 8.5)
        rgba = render_transparent(wide_camera, gaussians, pipe, black, white)
    for image, path in ((close, args.output), (rgba, args.transparent_output)):
        pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        Image.fromarray(pixels).save(path)


if __name__ == "__main__":
    main()
