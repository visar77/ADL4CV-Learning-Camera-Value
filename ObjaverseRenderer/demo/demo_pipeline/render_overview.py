"""Render the object, lattice, experiment, and held-out camera overviews."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blender_worker import configure_render
from demo.demo_pipeline.render_gif import FRAMES, HEIGHT, WIDTH


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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--reconstruction-image", type=Path)
    parser.add_argument("--test-renders", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gif-frames-dir", type=Path)
    parser.add_argument("--reconstruction-frames", type=Path)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    reconstruction = args.reconstruction_image is not None
    if reconstruction != (args.test_renders is not None) or reconstruction != (args.output is not None):
        parser.error("--reconstruction-image, --test-renders and --output must be supplied together")
    if bool(args.reconstruction_frames) != bool(reconstruction and args.gif_frames_dir):
        parser.error("--reconstruction-frames requires a reconstruction and --gif-frames-dir")
    if args.skip_images and not args.gif_frames_dir:
        parser.error("--skip-images requires --gif-frames-dir")
    object_dir = args.run_dir / "views" / args.object_id
    metadata = json.loads((object_dir / "metadata.json").read_text(encoding="utf-8"))
    selection = json.loads((args.run_dir / "selection.json").read_text(encoding="utf-8"))
    train = json.loads((object_dir / "global_transforms" / "transforms_train.json").read_text(encoding="utf-8"))
    test = json.loads((object_dir / "global_transforms" / "transforms_test.json").read_text(encoding="utf-8"))
    center = Vector(metadata["center"])
    radius = metadata["bounding_radius"]
    scene = bpy.context.scene
    configure_render(scene, {"image_width": 1600, "image_height": 1200, "use_gpu": True})
    if scene.cycles.device != "GPU":
        raise RuntimeError("A CUDA GPU is required for the pipeline demo")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (1, 1, 1, 1)
    background.inputs["Strength"].default_value = 0.8
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1200
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"

    direction = Vector((0.42, -0.83, 0.37)).normalized()
    overview_data = bpy.data.cameras.new("Pipeline overview")
    overview = bpy.data.objects.new("Pipeline overview", overview_data)
    scene.collection.objects.link(overview)
    overview.location = center + direction * radius * 10
    overview.rotation_euler = (center - overview.location).to_track_quat("-Z", "Y").to_euler()
    overview_data.type = "ORTHO"
    overview_data.clip_end = radius * 30
    scene.camera = overview
    if reconstruction:
        for obj in scene.objects:
            if obj.type not in {"CAMERA", "LIGHT"}:
                obj.hide_render = True
        overview_data.type = "PERSP"
        overview_data.sensor_fit = "HORIZONTAL"
        overview_data.angle = 2 * math.atan(8.5 / 20)
    blue = emission_material("Pipeline blue", (0.012, 0.20, 0.8))
    red = emission_material("Pipeline red", (0.85, 0.035, 0.025))
    green = emission_material("Pipeline green", (0.015, 0.6, 0.12))

    def render(number, frames):
        overview.location = center + direction * radius * 10
        overview.rotation_euler = (center - overview.location).to_track_quat("-Z", "Y").to_euler()
        labels = []
        collection = bpy.data.collections.new(f"Pipeline {number} cameras")
        scene.collection.children.link(collection)
        if reconstruction:
            width = radius * 8.5
            height = width * scene.render.resolution_y / scene.render.resolution_x
            mesh = bpy.data.meshes.new("Gaussian overview")
            mesh.from_pydata([(-width / 2, -height / 2, 0), (width / 2, -height / 2, 0),
                              (width / 2, height / 2, 0), (-width / 2, height / 2, 0)],
                             [], [(0, 1, 2, 3)])
            mesh.uv_layers.new(name="UVMap")
            for loop, uv in zip(mesh.uv_layers.active.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
                loop.uv = uv
            material = bpy.data.materials.new("Gaussian overview")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            nodes.clear()
            image = nodes.new("ShaderNodeTexImage")
            image.image = bpy.data.images.load(str(args.reconstruction_image), check_existing=True)
            emission = nodes.new("ShaderNodeEmission")
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            mix = nodes.new("ShaderNodeMixShader")
            output = nodes.new("ShaderNodeOutputMaterial")
            links = material.node_tree.links
            links.new(image.outputs["Color"], emission.inputs["Color"])
            links.new(image.outputs["Alpha"], mix.inputs[0])
            links.new(transparent.outputs[0], mix.inputs[1])
            links.new(emission.outputs[0], mix.inputs[2])
            links.new(mix.outputs[0], output.inputs["Surface"])
            mesh.materials.append(material)
            reconstruction_plane = bpy.data.objects.new("Gaussian overview", mesh)
            collection.objects.link(reconstruction_plane)
            reconstruction_plane.location = center
            reconstruction_plane.rotation_euler = overview.rotation_euler
            reconstruction_plane.visible_shadow = False
            reconstruction_texture = image
        if number == 2:
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=64, ring_count=32, radius=metadata["camera_distance"], location=center,
            )
            sphere = bpy.context.object
            sphere.name = "Sampling sphere"
            for owner in list(sphere.users_collection):
                owner.objects.unlink(sphere)
            collection.objects.link(sphere)
            sphere.visible_shadow = False
            material = bpy.data.materials.new("Translucent sampling sphere")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            nodes.clear()
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (0.35, 0.6, 0.95, 1)
            mix = nodes.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = 0.06
            output = nodes.new("ShaderNodeOutputMaterial")
            links = material.node_tree.links
            links.new(transparent.outputs[0], mix.inputs[1])
            links.new(emission.outputs[0], mix.inputs[2])
            links.new(mix.outputs[0], output.inputs["Surface"])
            sphere.data.materials.append(material)
            for polygon in sphere.data.polygons:
                polygon.use_smooth = True
        if number == 4:
            corners = [Matrix(frame["transform_matrix"]).translation for _, frame, _ in frames]
            cube = bpy.data.curves.new("Test camera cube", "CURVE")
            cube.dimensions = "3D"
            cube.bevel_depth = radius * 0.006
            cube.bevel_resolution = 2
            for i, start in enumerate(corners):
                for end in corners[i + 1:]:
                    differences = sum(abs(start[axis] - end[axis]) > radius * 1e-4 for axis in range(3))
                    if differences != 1:
                        continue
                    spline = cube.splines.new("POLY")
                    spline.points.add(1)
                    spline.points[0].co = (*start, 1)
                    spline.points[1].co = (*end, 1)
            cube.materials.append(green)
            cube_obj = bpy.data.objects.new("Test camera cube", cube)
            collection.objects.link(cube_obj)
            cube_obj.visible_shadow = False
        for index, (label, frame, material) in enumerate(frames):
            pose = Matrix(frame["transform_matrix"])
            depth = metadata["camera_distance"] * (0.19 if number == 2 else 0.23)
            half_width = depth * math.tan(train["camera_angle_x"] / 2)
            half_height = half_width * metadata["image_height"] / metadata["image_width"]
            corners = [Vector((x * half_width, y * half_height, -depth))
                       for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            curve = bpy.data.curves.new(f"{label} frustum", "CURVE")
            curve.dimensions = "3D"
            curve.bevel_depth = radius * 0.008
            curve.bevel_resolution = 2
            edges = [(Vector((0, 0, 0)), corner) for corner in corners]
            edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
            for start, end in edges:
                spline = curve.splines.new("POLY")
                spline.points.add(1)
                spline.points[0].co = (*start, 1)
                spline.points[1].co = (*end, 1)
            curve.materials.append(material)
            obj = bpy.data.objects.new(f"{label} frustum", curve)
            collection.objects.link(obj)
            obj.matrix_world = pose
            obj.visible_shadow = False
            mesh = bpy.data.meshes.new(f"{label} image")
            mesh.from_pydata([tuple(corner * 0.97 + Vector((0, 0, -depth * 0.03)))
                              for corner in corners], [], [(0, 1, 2, 3)])
            mesh.uv_layers.new(name="UVMap")
            for loop, uv in zip(mesh.uv_layers.active.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
                loop.uv = uv
            image_path = (args.test_renders / f"{index:05d}.png" if reconstruction
                          else object_dir / f"{frame['file_path']}.png")
            mesh.materials.append(image_material(f"{label} image", image_path))
            plane = bpy.data.objects.new(f"{label} image", mesh)
            collection.objects.link(plane)
            plane.matrix_world = pose
            plane.visible_shadow = False
            if number == 3 and label.startswith("Candidate "):
                text = bpy.data.curves.new(f"{label} label", "FONT")
                text.body = label.removeprefix("Candidate ")
                text.align_x = "CENTER"
                text.align_y = "CENTER"
                text.size = radius * 0.17
                text.materials.append(material)
                text_obj = bpy.data.objects.new(f"{label} label", text)
                collection.objects.link(text_obj)
                text_obj.visible_shadow = False
                labels.append((text_obj, pose.translation.copy()))

        def update_labels(camera_direction):
            rotation = overview.rotation_euler.to_matrix()
            for text_obj, position in labels:
                projected = rotation.transposed() @ (position - center)
                radial = Vector((projected.x, projected.y, 0))
                if radial.length < 1e-6:
                    radial = Vector((0, 1, 0))
                radial.normalize()
                text_obj.location = position + rotation @ (radial * radius * 0.45) + camera_direction * radius * 0.5
                text_obj.rotation_euler = overview.rotation_euler

        overview_data.ortho_scale = radius * (3.4 if number == 1 else 8.5)
        update_labels(direction)
        if not args.skip_images:
            scene.render.resolution_x, scene.render.resolution_y = 1600, 1200
            scene.render.filepath = str(args.output if reconstruction else args.run_dir / f"pipeline_{number}.png")
            bpy.ops.render.render(write_still=True)
        if args.gif_frames_dir:
            scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
            for index in range(FRAMES):
                angle = 2 * math.pi * index / FRAMES
                camera_direction = Vector((direction.x * math.cos(angle) - direction.y * math.sin(angle),
                                           direction.x * math.sin(angle) + direction.y * math.cos(angle),
                                           direction.z))
                overview.location = center + camera_direction * radius * 10
                overview.rotation_euler = (center - overview.location).to_track_quat("-Z", "Y").to_euler()
                update_labels(camera_direction)
                if reconstruction:
                    reconstruction_plane.rotation_euler = overview.rotation_euler
                    previous_image = reconstruction_texture.image
                    reconstruction_texture.image = bpy.data.images.load(
                        str(args.reconstruction_frames / f"{index:03d}.png"))
                    if previous_image.users == 0:
                        bpy.data.images.remove(previous_image)
                scene.render.filepath = str(args.gif_frames_dir / f"pipeline_{number}_{index:03d}.png")
                bpy.ops.render.render(write_still=True)
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)

    train_frames = train["frames"]
    if reconstruction:
        render(4, [(f"Test {i:03d}", frame, green) for i, frame in enumerate(test["frames"])])
        return
    render(1, [])
    render(2, [(f"Lattice {i:03d}", frame, blue) for i, frame in enumerate(train_frames)])
    render(3, ([(f"Seen {i:03d}", train_frames[i], blue) for i in selection["base"]]
               + [(f"Candidate {name}", train_frames[i], red) for name, i in selection["candidates"].items()]))
    render(4, [(f"Test {i:03d}", frame, green) for i, frame in enumerate(test["frames"])])


if __name__ == "__main__":
    main()
