"""
Run this file from PyCharm with regular Python.
It launches Blender, imports one GLB, and renders its camera views.
"""

import glob
import json
import math
import os
import shutil
import subprocess
import sys


# ========================= USER SETTINGS =========================

GLB_PATH = r"C:\Users\lumi_vi\PycharmProjects\ObjaverseRenderer\objects\gamecube.glb"
OUTPUT_DIR = r"C:\Users\lumi_vi\PycharmProjects\ObjaverseRenderer\views"

# Set this to your blender.exe path, or leave as None for auto-detection.
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
# Example for auto-detection:
# BLENDER_EXE = None
SAVE_BLEND_FILE = True
OPEN_SAVED_BLEND = True

TRAIN_COUNT = 10
TEST_COUNT = 8

# Camera distances from the center of the GLB, in Blender units.
TRAIN_RADIUS = 5.0
TEST_RADIUS = 5.0
SPECIAL_RADIUS = None  # None uses TRAIN_RADIUS

# Direction of the extra training camera.
SPECIAL_AZIMUTH_DEG = 45.0
SPECIAL_ELEVATION_DEG = 20.0

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
HORIZONTAL_FOV_DEG = 50.0

# Lighting scales with the model's bounding-box size.
KEY_LIGHT_ENERGY = 1500.0
FILL_LIGHT_ENERGY = 800.0
RIM_LIGHT_ENERGY = 1200.0

# ================================================================


def find_blender():
    if BLENDER_EXE:
        path = os.path.abspath(BLENDER_EXE)
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Blender not found at: {path}")

    found = shutil.which("blender")
    if found:
        return found

    # Try the usual Windows installation folder.
    matches = glob.glob(
        r"C:\Program Files\Blender Foundation\Blender *\blender.exe"
    )
    if matches:
        return sorted(matches)[-1]

    raise FileNotFoundError(
        "Could not find blender.exe. Set BLENDER_EXE in the USER SETTINGS block."
    )


def fibonacci_sphere(count):
    """Generate approximately even directions on a unit sphere."""
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions = []

    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * i
        directions.append((r * math.cos(phi), r * math.sin(phi), z))

    return directions


def get_test_directions(count):
    """
    First 8 positions: above, below, four horizontal cardinal directions,
    and two horizontal diagonals. Additional cameras use sphere sampling.
    """
    diagonal = 1.0 / math.sqrt(2.0)
    directions = [
        (0, 0, 1),                  # above
        (0, 0, -1),                 # below
        (1, 0, 0),                  # +X
        (-1, 0, 0),                 # -X
        (0, 1, 0),                  # +Y
        (0, -1, 0),                 # -Y
        (diagonal, diagonal, 0),
        (-diagonal, -diagonal, 0),
    ]

    if count <= len(directions):
        return directions[:count]

    return directions + fibonacci_sphere(count - len(directions))


def run_external_python():
    if not os.path.isfile(GLB_PATH):
        raise FileNotFoundError(f"GLB file not found: {GLB_PATH}")
    if not GLB_PATH.lower().endswith(".glb"):
        raise ValueError("GLB_PATH must point to a .glb file.")
    if TRAIN_COUNT < 1 or TEST_COUNT < 1:
        raise ValueError("TRAIN_COUNT and TEST_COUNT must be at least 1.")
    if TRAIN_RADIUS <= 0 or TEST_RADIUS <= 0:
        raise ValueError("TRAIN_RADIUS and TEST_RADIUS must be greater than 0.")
    if not 0 < HORIZONTAL_FOV_DEG < 180:
        raise ValueError("HORIZONTAL_FOV_DEG must be between 0 and 180.")

    blender = find_blender()
    script_path = os.path.abspath(__file__)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The config is passed through Blender's "--" script-argument separator.
    config = {
        "glb_path": os.path.abspath(GLB_PATH),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "train_count": TRAIN_COUNT,
        "test_count": TEST_COUNT,
        "train_radius": TRAIN_RADIUS,
        "test_radius": TEST_RADIUS,
        "special_radius": SPECIAL_RADIUS,
        "special_azimuth_deg": SPECIAL_AZIMUTH_DEG,
        "special_elevation_deg": SPECIAL_ELEVATION_DEG,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
        "key_light_energy": KEY_LIGHT_ENERGY,
        "fill_light_energy": FILL_LIGHT_ENERGY,
        "rim_light_energy": RIM_LIGHT_ENERGY,
        "blend_path": os.path.join(os.path.abspath(OUTPUT_DIR), "rendered_scene.blend"),
        "save_blend_file": SAVE_BLEND_FILE,

    }

    command = [
        blender,
        "--background",
        "--python", script_path,
        "--",
        "--worker-config",
        json.dumps(config),
    ]

    print(f"Launching Blender: {blender}")
    subprocess.run(command, check=True)

    if OPEN_SAVED_BLEND and SAVE_BLEND_FILE:
        blend_path = os.path.join(
            os.path.abspath(OUTPUT_DIR), "rendered_scene.blend"
        )
        subprocess.Popen([blender, blend_path])


def run_inside_blender(config):
    import bpy
    from mathutils import Vector
    scene = bpy.context.scene
    glb_path = config["glb_path"]
    output_dir = config["output_dir"]

    # Remove the default scene objects, then import the GLB.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.import_scene.gltf(filepath=glb_path)

    mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects were imported from: {glb_path}")

    bpy.context.view_layer.update()

    # Find the world-space bounding box covering all imported mesh parts.
    corners = []
    for obj in mesh_objects:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)

    bbox_min = Vector((
        min(v.x for v in corners),
        min(v.y for v in corners),
        min(v.z for v in corners),
    ))
    bbox_max = Vector((
        max(v.x for v in corners),
        max(v.y for v in corners),
        max(v.z for v in corners),
    ))
    center = (bbox_min + bbox_max) / 2.0
    model_size = max((bbox_max - bbox_min).length, 0.1)

    # Create a simple studio world.
    world = bpy.data.worlds.new("DatasetWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.18, 0.18, 0.18, 1.0
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
    scene.world = world

    def add_area_light(name, offset, energy, size_factor):
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy * model_size * model_size
        light_data.shape = "DISK"
        light_data.size = size_factor * model_size

        light_obj = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light_obj)
        light_obj.location = center + Vector(offset) * model_size
        light_obj.rotation_euler = (center - light_obj.location).to_track_quat(
            "-Z", "Y"
        ).to_euler()
        return light_obj

    add_area_light(
        "KeyLight", (2.5, -3.0, 3.0),
        config["key_light_energy"], 2.0
    )
    add_area_light(
        "FillLight", (-3.0, -1.5, 1.0),
        config["fill_light_energy"], 2.5
    )
    add_area_light(
        "RimLight", (0.0, 3.0, 2.5),
        config["rim_light_energy"], 2.0
    )

    # Create the render camera.
    camera_data = bpy.data.cameras.new("DatasetCamera")
    camera_obj = bpy.data.objects.new("DatasetCamera", camera_data)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    fov_x = math.radians(config["horizontal_fov_deg"])
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = camera_data.sensor_width / (2.0 * math.tan(fov_x / 2.0))
    camera_data.clip_start = 0.01
    camera_data.clip_end = max(
        config["train_radius"],
        config["test_radius"],
        model_size,
    ) * 20.0

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = config["image_width"]
    scene.render.resolution_y = config["image_height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.refresh_devices()

    for device in preferences.devices:
        print("Cycles device:", device.name, device.type, device.use)
        device.use = device.type == "CUDA"

    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "test"), exist_ok=True)

    def render_frame(direction, radius, absolute_path, relative_path):
        direction = Vector(direction).normalized()
        camera_obj.location = center + direction * radius
        camera_obj.rotation_euler = (
            center - camera_obj.location
        ).to_track_quat("-Z", "Y").to_euler()

        bpy.context.view_layer.update()
        scene.render.filepath = absolute_path
        bpy.ops.render.render(write_still=True)

        return {
            "file_path": relative_path.replace("\\", "/"),
            "transform_matrix": [
                [float(value) for value in row]
                for row in camera_obj.matrix_world
            ],
        }

    train_count = config["train_count"]
    test_count = config["test_count"]
    train_radius = config["train_radius"]
    test_radius = config["test_radius"]
    special_radius = config["special_radius"]
    if special_radius is None:
        special_radius = train_radius

    train_frames = []
    for i, direction in enumerate(fibonacci_sphere(train_count)):
        filename = f"{i:03d}.png"
        train_frames.append(render_frame(
            direction,
            train_radius,
            os.path.join(output_dir, "images", "train", filename),
            f"./images/train/{filename}",
        ))

    az = math.radians(config["special_azimuth_deg"])
    el = math.radians(config["special_elevation_deg"])
    special_direction = (
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    )

    special_filename = f"{train_count:03d}.png"
    special_frame = render_frame(
        special_direction,
        special_radius,
        os.path.join(output_dir, "images", "train", special_filename),
        f"./images/train/{special_filename}",
    )

    test_frames = []
    for i, direction in enumerate(get_test_directions(test_count)):
        filename = f"{i:03d}.png"
        test_frames.append(render_frame(
            direction,
            test_radius,
            os.path.join(output_dir, "images", "test", filename),
            f"./images/test/{filename}",
        ))

    def write_transforms(filename, frames):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"camera_angle_x": fov_x, "frames": frames},
                f,
                indent=2,
            )

    write_transforms("transforms_train1.json", train_frames)
    write_transforms("transforms_train2.json", train_frames + [special_frame])
    write_transforms("transforms_test.json", test_frames)

    print(f"Finished rendering. Output folder: {output_dir}")
    if config["save_blend_file"]:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])


if __name__ == "__main__":
    # When launched by ordinary Python, start Blender.
    if "--worker-config" not in sys.argv:
        run_external_python()
    else:
        # When Blender launches this same file, read the JSON after "--".
        separator = sys.argv.index("--")
        worker_args = sys.argv[separator + 1:]
        config_index = worker_args.index("--worker-config")
        config_json = worker_args[config_index + 1:]
        config = json.loads(" ".join(config_json))
        run_inside_blender(config)
