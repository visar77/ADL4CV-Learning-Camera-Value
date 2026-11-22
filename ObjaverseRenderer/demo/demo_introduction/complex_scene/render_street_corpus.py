"""Render the free street camera views from the saved preview scene."""

import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

COMPLEX = Path(__file__).resolve().parent
sys.path.insert(0, str(COMPLEX))
from street_common import load_street_config, run_dir
from blender_worker import configure_render, render_frame, write_transforms


def bounds(objects):
    corners = [obj.matrix_world @ Vector(corner) for obj in objects if obj.type == "MESH" for corner in obj.bound_box]
    if not corners:
        raise ValueError("Vespa contains no meshes")
    return (Vector([min(point[i] for point in corners) for i in range(3)]),
            Vector([max(point[i] for point in corners) for i in range(3)]))


def main():
    config = load_street_config()
    scene = bpy.context.scene
    root = scene.objects.get(config["target_root"])
    if root is None or scene.camera is None:
        raise RuntimeError("The street scene is missing its Vespa root or preview camera")
    lights = [obj for obj in scene.objects if obj.type == "LIGHT"]
    if len(lights) != 1 or lights[0].data.type != "SUN":
        raise RuntimeError("The street scene must have exactly one sun light")
    if "Street" not in scene.objects or not any(obj.name.startswith("Opposite street ") for obj in scene.objects):
        raise RuntimeError("The source scene must contain both street facades")
    vespa = root.children_recursive
    minimum, maximum = bounds(vespa)
    center = (minimum + maximum) / 2
    radius = (maximum - minimum).length / 2
    object_dir = run_dir(config["object_id"]) / "views" / config["object_id"]
    global_dir = object_dir / "global_transforms"
    if global_dir.exists():
        raise FileExistsError(f"Street camera manifests already exist: {global_dir}")
    global_dir.mkdir(parents=True)

    scene["demo_object_id"] = config["object_id"]
    configure_render(scene, {"image_width": 800, "image_height": 600, "use_gpu": True})
    if scene.cycles.device != "GPU":
        raise RuntimeError("A CUDA GPU is required for the street demo")
    scene.cycles.samples = 32
    scene.render.film_transparent = False
    camera = scene.camera
    camera.data.sensor_fit = "HORIZONTAL"
    fov = camera.data.angle_x
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(object_dir / "grounded_scene.blend"))

    def render_set(split, views):
        directory = object_dir / "images" / split
        directory.mkdir(parents=True)
        frames = []
        for view in views:
            image = directory / f"{view['id']}.png"
            start = time.monotonic()
            if "rotation_degrees" in view:
                camera.location = view["position"]
                camera.rotation_mode = "XYZ"
                camera.rotation_euler = tuple(math.radians(value) for value in view["rotation_degrees"])
                bpy.context.view_layer.update()
                scene.render.filepath = str(image)
                bpy.ops.render.render(write_still=True)
                frame = {"transform_matrix": [list(row) for row in camera.matrix_world]}
            else:
                target = Vector(view["target"])
                offset = Vector(view["position"]) - target
                frame = render_frame(scene, camera, target, offset.length, offset, str(image))
            elapsed = time.monotonic() - start
            print(f"Street view {view['id']} rendered in {elapsed:.1f} seconds", flush=True)
            frame["file_path"] = f"images/{split}/{view['id']}"
            frame["view_id"] = view["id"]
            frames.append(frame)
        return frames

    train_frames = render_set("train", config["base_views"] + config["candidate_views"])
    test_frames = render_set("test", config["test_views"])
    write_transforms(str(global_dir / "transforms_train.json"), fov, train_frames)
    write_transforms(str(global_dir / "transforms_test.json"), fov, test_frames)
    metadata = {
        "object_id": config["object_id"],
        "center": list(center),
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "ground_z": 0,
        "bounding_radius": radius,
        "render_engine": scene.render.engine,
        "image_width": scene.render.resolution_x,
        "image_height": scene.render.resolution_y,
    }
    (object_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
