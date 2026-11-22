"""Render the full lattice and standard test cameras without a floor."""

import argparse
import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blender_worker import configure_camera, configure_render, import_model, randomize_lighting, render_frame, write_transforms
from sampling import fibonacci_lattice, test_view_directions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--glb-path", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--lattice-size", required=True, type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    object_dir = args.run_dir / "views" / args.object_id
    global_dir = object_dir / "global_transforms"
    global_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    center, radius = import_model(str(args.glb_path))
    import random
    random.seed(77)
    randomize_lighting()
    camera, distance, fov = configure_camera(scene, center, radius, {
        "horizontal_fov_deg": 50, "camera_distance_scale": 1.15,
    })
    configure_render(scene, {"image_width": 512, "image_height": 512, "use_gpu": True})
    if scene.cycles.device != "GPU":
        raise RuntimeError("A CUDA GPU is required for the pipeline demo")
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(object_dir / "rendered_scene.blend"))
    for split, directions in (("train", fibonacci_lattice(args.lattice_size)), ("test", test_view_directions())):
        image_dir = object_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, direction in enumerate(directions):
            frame = render_frame(scene, camera, center, distance, direction, str(image_dir / f"{index:03d}.png"))
            frame["file_path"] = f"images/{split}/{index:03d}"
            frames.append(frame)
        write_transforms(str(global_dir / f"transforms_{split}.json"), fov, frames)
    metadata = {
        "object_id": args.object_id, "lattice_size": args.lattice_size,
        "trajectory_elevation_deg": 20.0, "center": list(center),
        "bounding_radius": radius, "camera_distance": distance,
        "image_width": 512, "image_height": 512,
    }
    (object_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
