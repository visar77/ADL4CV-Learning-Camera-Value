"""Make an editable overview using the demo's actual cameras and input images."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

COMPLEX = Path(__file__).resolve().parent
sys.path.insert(0, str(COMPLEX))
from street_common import load_street_config, run_dir
from blender_worker import configure_render


def emission_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1)
    mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return mat


def image_material(name, path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path), check_existing=True)
    image.image.pack()
    uv = nodes.new("ShaderNodeUVMap")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(uv.outputs["UV"], separate.inputs[0])
    flip = nodes.new("ShaderNodeMath")
    flip.operation = "SUBTRACT"
    flip.inputs[0].default_value = 1
    links.new(separate.outputs["X"], flip.inputs[1])
    flipped = nodes.new("ShaderNodeCombineXYZ")
    links.new(flip.outputs[0], flipped.inputs["X"])
    links.new(separate.outputs["Y"], flipped.inputs["Y"])
    backface = nodes.new("ShaderNodeNewGeometry")
    uv_choice = nodes.new("ShaderNodeMixRGB")
    links.new(backface.outputs["Backfacing"], uv_choice.inputs[0])
    links.new(uv.outputs["UV"], uv_choice.inputs[1])
    links.new(flipped.outputs[0], uv_choice.inputs[2])
    links.new(uv_choice.outputs[0], image.inputs["Vector"])
    links.new(image.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs["Surface"])
    return mat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    config = load_street_config()
    run_root = run_dir(config["object_id"])
    output_dir = args.output_dir.resolve() if args.output_dir else run_root
    output_dir.mkdir(parents=True, exist_ok=True)
    object_dir = run_root / "views" / config["object_id"]
    metadata = json.loads((object_dir / "metadata.json").read_text(encoding="utf-8"))
    transforms = json.loads((object_dir / "global_transforms" / "transforms_train.json").read_text(encoding="utf-8"))
    frames = {frame["view_id"]: frame for frame in transforms["frames"]}
    scene = bpy.context.scene
    if "Street" not in scene.objects or scene.get("demo_object_id") != config["object_id"]:
        raise RuntimeError("Open this object's demo grounded_scene.blend first")
    center = Vector(metadata["center"])
    radius = metadata["bounding_radius"]

    collection = bpy.data.collections.new("Camera overview")
    scene.collection.children.link(collection)
    blue = emission_material("Seen - blue", (0.012, 0.20, 0.8))
    red = emission_material("Candidate - red", (0.85, 0.035, 0.025))
    unknown = emission_material("Unknown image background", (1, 0.92, 0.92))

    old_camera = scene.camera
    overview_data = bpy.data.cameras.new("Slide overview")
    overview = bpy.data.objects.new("Slide overview", overview_data)
    collection.objects.link(overview)
    overview.location = config["overview"]["position"]
    overview.rotation_mode = "XYZ"
    overview.rotation_euler = tuple(math.radians(value) for value in config["overview"]["rotation_degrees"])
    overview_data.type = "PERSP"
    overview_data.sensor_fit = "HORIZONTAL"
    overview_data.sensor_width = 36
    overview_data.lens = config["overview"]["lens_mm"]
    overview_data.clip_end = 100
    scene.camera = overview
    bpy.data.objects.remove(old_camera, do_unlink=True)
    camera_rotation = overview.rotation_euler.to_matrix()
    direction = camera_rotation @ Vector((0, 0, 1))
    fov = transforms["camera_angle_x"]
    aspect = metadata["image_width"] / metadata["image_height"]

    def add_camera(index, known):
        _, number = index.rsplit("-", 1)
        label = f'{"Captured" if known else "Candidate"} {int(number)}'
        mat = blue if known else red
        frame = frames[index]
        pose = Matrix(frame["transform_matrix"])
        position = pose.translation
        distance = (position - center).length
        depth = distance * 0.24
        width = depth * math.tan(fov / 2)
        height = width / aspect

        data = bpy.data.cameras.new(label)
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36
        data.lens = data.sensor_width / (2 * math.tan(fov / 2))
        data.display_size = depth
        camera = bpy.data.objects.new(label, data)
        collection.objects.link(camera)
        camera.matrix_world = pose
        camera.hide_set(True)

        curve = bpy.data.curves.new(f"{label} frustum", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius * 0.009
        curve.bevel_resolution = 2
        frame_corners = [
            Vector((x * width, y * height, -depth))
            for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        edges = [
            *((Vector((0, 0, 0)), corner) for corner in frame_corners),
            *((frame_corners[i], frame_corners[(i + 1) % 4]) for i in range(4)),
        ]
        for start, end in edges:
            spline = curve.splines.new("POLY")
            spline.points.add(1)
            spline.points[0].co = (*start, 1)
            spline.points[1].co = (*end, 1)
        frustum = bpy.data.objects.new(f"{label} frustum", curve)
        collection.objects.link(frustum)
        frustum.matrix_world = pose
        curve.materials.append(mat)

        width *= 0.94
        height *= 0.94
        plane = bpy.data.meshes.new(f"{label} image plane")
        plane.from_pydata(
            [(-width, -height, -depth), (width, -height, -depth),
             (width, height, -depth), (-width, height, -depth)],
            [], [(0, 1, 2, 3)],
        )
        plane.uv_layers.new(name="UVMap")
        for loop, uv in zip(plane.uv_layers.active.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
            loop.uv = uv
        image_obj = bpy.data.objects.new(f"{label} image", plane)
        collection.objects.link(image_obj)
        image_obj.matrix_world = pose
        if known:
            image_path = object_dir / f"{frame['file_path']}.png"
            plane.materials.append(image_material(f"{label} image", image_path))
        else:
            plane.materials.append(unknown)
            for side in (-1, 1):
                text = bpy.data.curves.new(f"{label} question {side}", "FONT")
                text.body = "?"
                text.align_x = "CENTER"
                text.align_y = "CENTER"
                text.size = height * 1.5
                text.font = bpy.data.fonts.load(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                check_existing=True,
                )
                text.materials.append(red)
                question = bpy.data.objects.new(f"{label} question {side}", text)
                collection.objects.link(question)
                offset = Matrix.Translation(Vector((0, 0, -depth + side * radius * 0.003)))
                rotation = Matrix.Rotation(math.pi, 4, "Y") if side < 0 else Matrix.Identity(4)
                question.matrix_world = pose @ offset @ rotation

        projected = camera_rotation.transposed() @ (position - center)
        radial = Vector((projected.x, projected.y, 0))
        if radial.length < 1e-6:
            radial = Vector((0, 1, 0))
        radial.normalize()
        text = bpy.data.curves.new(f"{label} label", "FONT")
        text.body = label
        text.align_x = "CENTER"
        text.align_y = "CENTER"
        camera_depth = -(camera_rotation.transposed() @ (position - overview.location)).z
        text.size = radius * 0.29 * camera_depth / 10 * 0.4
        text.materials.append(mat)
        obj = bpy.data.objects.new(f"{label} label", text)
        collection.objects.link(obj)
        label_position = position + camera_rotation @ (radial * radius * 0.2) + direction * radius * 0.4
        obj.location = overview.location + (label_position - overview.location) * 0.4
        if index == "candidate-01":
            obj.location += camera_rotation @ Vector((0.1, -0.05, 0))
        obj.rotation_euler = overview.rotation_euler

    base = [view["id"] for view in config["base_views"]]
    candidates = [view["id"] for view in config["candidate_views"]]
    for index in base:
        add_camera(index, True)
    for index in candidates:
        add_camera(index, False)
    for obj in collection.objects:
        obj.visible_shadow = False

    configure_render(scene, {"image_width": 1600, "image_height": 1200, "use_gpu": True})
    if scene.cycles.device != "GPU":
        raise RuntimeError("A CUDA GPU is required for the street overview")
    scene.cycles.samples = 32
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.render.filepath = str(output_dir / "camera_overview.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "camera_overview.blend"))
    start = time.monotonic()
    bpy.ops.render.render(write_still=True)
    elapsed = time.monotonic() - start
    print(f"Overview rendered in {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
