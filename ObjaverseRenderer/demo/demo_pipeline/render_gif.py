"""Render a 360-degree FastGS turntable using Blender camera poses."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

FRAMES = 60
WIDTH, HEIGHT = 800, 600
DURATION_MS = 80


def save_gif(frames, output):
    with tempfile.TemporaryDirectory(prefix="pipeline-gif-", dir=output.parent) as temp_dir:
        gif = Path(temp_dir) / "turntable.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=DURATION_MS,
                       loop=0, disposal=2)
        gif.replace(output)
    print(f"Saved {output}", flush=True)


def save_gif_from_files(paths, output):
    from PIL import Image

    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").quantize(colors=256))
    save_gif(frames, output)


def write_camera_poses(metadata_path, output):
    import bpy
    from mathutils import Vector

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    center = Vector(metadata["center"])
    direction = Vector((0.42, -0.83, 0.37)).normalized()
    distance = metadata["bounding_radius"] * 20
    camera = bpy.data.objects.new("Turntable camera", bpy.data.cameras.new("Turntable camera"))
    bpy.context.scene.collection.objects.link(camera)
    poses = []
    for index in range(FRAMES):
        angle = 2 * math.pi * index / FRAMES
        x = direction.x * math.cos(angle) - direction.y * math.sin(angle)
        y = direction.x * math.sin(angle) + direction.y * math.cos(angle)
        camera.location = center + Vector((x, y, direction.z)) * distance
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        poses.append([list(row) for row in camera.matrix_world])
    output.write_text(json.dumps(poses), encoding="utf-8")


def render_gif(args):
    import numpy as np
    import torch
    from PIL import Image
    from render_gaussians import (
        OVERVIEW_BACKGROUND, camera_from_blender, load_gaussians, render_fastgs, render_transparent,
    )

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pipeline-gif-", dir=args.output.parent) as temp_dir:
        poses_path = Path(temp_dir) / "poses.json"
        subprocess.run([
            args.blender, "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", str(Path(__file__).resolve()), "--", "--camera-poses",
            "--metadata", str(args.metadata), "--output", str(poses_path),
        ], check=True)
        poses = json.loads(poses_path.read_text(encoding="utf-8"))
        if len(poses) != FRAMES:
            raise RuntimeError(f"Expected {FRAMES} camera poses, got {len(poses)}")
        gaussians = load_gaussians(args.checkpoint)
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        background = torch.full((3,), OVERVIEW_BACKGROUND, device="cuda")
        white = torch.ones(3, device="cuda") if args.transparent_frames else None
        black = torch.zeros(3, device="cuda") if args.transparent_frames else None
        frames = []
        with torch.no_grad():
            for index, pose in enumerate(poses):
                camera = camera_from_blender(metadata, pose, WIDTH, HEIGHT, 3.4)
                image = render_fastgs(camera, gaussians, pipe, background, 0.5)["render"]
                pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
                frames.append(Image.fromarray(pixels).quantize(colors=256))
                if args.transparent_frames:
                    camera = camera_from_blender(metadata, pose, WIDTH, HEIGHT, 8.5)
                    image = render_transparent(camera, gaussians, pipe, black, white)
                    pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
                    Image.fromarray(pixels).save(args.transparent_frames / f"{index:03d}.png")
        save_gif(frames, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--blender")
    parser.add_argument("--camera-poses", action="store_true")
    parser.add_argument("--transparent-frames", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.camera_poses:
        if args.transparent_frames:
            parser.error("--transparent-frames cannot be used with --camera-poses")
        write_camera_poses(args.metadata, args.output)
    else:
        if not args.checkpoint or not args.blender:
            parser.error("--checkpoint and --blender are required for GIF rendering")
        render_gif(args)


if __name__ == "__main__":
    main()
