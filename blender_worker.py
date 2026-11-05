"""Blender worker launched by renderer.py."""

import json
import math
import os
import random
import sys


def get_config_path():
    """Read the config path passed after Blender's '--' separator."""
    if "--" not in sys.argv:
        raise RuntimeError("Expected Blender arguments after '--'.")

    args = sys.argv[sys.argv.index("--") + 1 :]
    if "--config" not in args:
        raise RuntimeError("Missing '--config <path>' argument.")

    return args[args.index("--config") + 1]


def safe_name(value):
    """Make a string safe to use as a folder name."""
    result = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in str(value)
    )
    return result.strip("._") or "object"


def load_config():
    config_path = get_config_path()
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def import_model(glb_path):
    """Import the GLB and return its world-space center and bounding radius."""
    import bpy
    from mathutils import Vector

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.import_scene.gltf(filepath=glb_path)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from: {glb_path}")

    bpy.context.view_layer.update()

    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in mesh_objects
        for corner in obj.bound_box
    ]

    minimum = Vector(
        (
            min(point.x for point in corners),
            min(point.y for point in corners),
            min(point.z for point in corners),
        )
    )
    maximum = Vector(
        (
            max(point.x for point in corners),
            max(point.y for point in corners),
            max(point.z for point in corners),
        )
    )

    center = (minimum + maximum) / 2.0
    bounding_radius = max((maximum - minimum).length / 2.0, 0.05)
    return center, bounding_radius


def configure_hdri(scene, hdri_dir, seed, strength_min=0.7, strength_max=1.3):
    """Choose and orient one HDRI per object; keep it static across its views."""
    import bpy

    if not os.path.isdir(hdri_dir):
        raise FileNotFoundError(f"HDRI directory not found: {hdri_dir}")

    hdri_files = sorted(
        os.path.join(hdri_dir, filename)
        for filename in os.listdir(hdri_dir)
        if filename.lower().endswith((".hdr", ".exr"))
    )
    if not hdri_files:
        raise FileNotFoundError(f"No .hdr or .exr files found in: {hdri_dir}")

    rng = random.Random(seed)
    hdri_path = rng.choice(hdri_files)
    rotation_z = rng.uniform(0.0, 2.0 * math.pi)
    strength = rng.uniform(strength_min, strength_max)

    world = bpy.data.worlds.new("RandomizedStaticHDRI")
    world.use_nodes = True

    nodes = world.node_tree.nodes
    nodes.clear()

    coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    environment = nodes.new("ShaderNodeTexEnvironment")
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")

    environment.image = bpy.data.images.load(hdri_path, check_existing=True)
    mapping.inputs["Rotation"].default_value[2] = rotation_z
    background.inputs["Strength"].default_value = strength

    world.node_tree.links.new(
        coordinates.outputs["Generated"],
        mapping.inputs["Vector"],
    )
    world.node_tree.links.new(
        mapping.outputs["Vector"],
        environment.inputs["Vector"],
    )
    world.node_tree.links.new(
        environment.outputs["Color"],
        background.inputs["Color"],
    )
    world.node_tree.links.new(
        background.outputs["Background"],
        output.inputs["Surface"],
    )

    scene.world = world

    print(
        f"HDRI: {os.path.basename(hdri_path)}, "
        f"Z rotation: {rotation_z:.3f} rad, "
        f"strength: {strength:.3f}"
    )
    return os.path.basename(hdri_path)


def configure_objaverse_area_light(scene, center, model_size):
    """Single large area light, following the Objaverse renderer setup."""
    import bpy
    from mathutils import Vector

    light_data = bpy.data.lights.new(
        name="ObjaverseArea",
        type="AREA",
    )
    light_data.energy = 30000.0
    light_data.shape = "DISK"

    # The reference renderer scales its area light to 100.
    light_data.size = max(100.0 * model_size, 1.0)

    light_obj = bpy.data.objects.new("ObjaverseArea", light_data)
    scene.collection.objects.link(light_obj)

    # Offset relative to the model, rather than assuming it is centered at origin.
    light_obj.location = center + Vector((0.0, 0.0, 0.5 * model_size))
    light_obj.rotation_euler = (0.0, 0.0, 0.0)

    return "objaverse_area"


def configure_camera(scene, center, bounding_radius, config):
    """Create a camera and calculate a conservative distance to fit the model."""
    import bpy

    camera_data = bpy.data.cameras.new("DatasetCamera")
    camera = bpy.data.objects.new("DatasetCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    horizontal_fov = math.radians(config["horizontal_fov_deg"])
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = camera_data.sensor_width / (2.0 * math.tan(horizontal_fov / 2.0))
    camera_data.clip_start = 0.01

    distance_scale = config.get("camera_distance_scale", 1.15)
    distance = (
        bounding_radius / max(math.sin(horizontal_fov / 2.0), 0.01) * distance_scale
    )
    camera_data.clip_end = max(distance + bounding_radius * 10.0, 100.0)

    return camera, distance, horizontal_fov


def configure_render(scene, config):
    import bpy

    """Set up Cycles rendering."""
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(config["image_width"])
    scene.render.resolution_y = int(config["image_height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    scene.cycles.samples = int(config.get("samples", 32))
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"

    # GPU is optional. CPU remains the fallback if a usable device isn't found.
    if config.get("use_gpu", False):
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.refresh_devices()

        for device in preferences.devices:
            device.use = device.type == "OPTIX"
            print("Cycles device:", device.name, device.type, device.use)

        scene.cycles.device = "GPU"
        scene.cycles.use_denoising = True

        try:
            scene.cycles.denoiser = "OPTIX"
        except (AttributeError, TypeError):
            print("OptiX denoising setting unavailable; using Blender's default.")

    # Use Blender's standard modern display transform when available.
    scene.view_settings.view_transform = "AgX"


def render_frame(scene, camera, center, distance, direction, image_path):
    """Render one view and return its camera transform record."""
    import bpy
    from mathutils import Vector

    direction = Vector(direction).normalized()
    camera.location = center + direction * distance
    camera.rotation_euler = (
        (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    )

    bpy.context.view_layer.update()
    scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)

    return {
        # FastGS appends ".png" to paths from the transforms file.
        "file_path": os.path.splitext(os.path.basename(image_path))[0],
        "transform_matrix": [
            [float(value) for value in row] for row in camera.matrix_world
        ],
    }


def write_transforms(path, horizontal_fov, frames):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "camera_angle_x": horizontal_fov,
                "frames": frames,
            },
            file,
            indent=2,
        )


def main():
    # The worker script and sampling.py should be beside one another.
    worker_dir = os.path.dirname(os.path.abspath(__file__))
    if worker_dir not in sys.path:
        sys.path.insert(0, worker_dir)

    import bpy
    from sampling import fibonacci_lattice

    config = load_config()

    glb_path = os.path.abspath(config["glb_path"])
    object_id = safe_name(config["object_id"])
    object_dir = os.path.abspath(config["object_dir"])

    if not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    os.makedirs(object_dir, exist_ok=True)
    scene = bpy.context.scene
    center, bounding_radius = import_model(glb_path)

    seed = int(config.get("seed", 0))
    lighting_mode = config.get("lighting_mode", "area").lower()

    if lighting_mode == "hdri":
        lighting_name = configure_hdri(
            scene,
            config["hdri_dir"],
            seed=seed,
            strength_min=float(config.get("hdri_strength_min", 0.7)),
            strength_max=float(config.get("hdri_strength_max", 1.3)),
        )
    elif lighting_mode == "area":
        lighting_name = configure_objaverse_area_light(
            scene,
            center=center,
            model_size=max(bounding_radius * 2.0, 0.1),
        )
    else:
        raise ValueError(
            f"Unknown lighting_mode {lighting_mode!r}; use 'area' or 'hdri'."
        )
    camera, distance, horizontal_fov = configure_camera(
        scene,
        center,
        bounding_radius,
        config,
    )
    configure_render(scene, config)

    lattice_size = int(config["lattice_size"])
    lattice = fibonacci_lattice(lattice_size)
    trajectory_elevation = float(config.get("trajectory_elevation_deg", 20.0))

    image_train_dir = os.path.join(object_dir, "images", "train")
    image_test_dir = os.path.join(object_dir, "images", "test")
    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_test_dir, exist_ok=True)

    metadata = {
        "object_id": object_id,
        "glb_path": glb_path,
        "lighting": lighting_name,
        "lattice_size": lattice_size,
    }

    # Render every lattice direction exactly once. Subset manifests below
    # reference these shared images instead of copying or rerendering them.
    all_train_frames = []
    for lattice_index, direction in enumerate(lattice):
        image_name = f"{lattice_index:03d}.png"
        image_path = os.path.join(image_train_dir, image_name)
        frame = render_frame(
            scene,
            camera,
            center,
            distance,
            direction,
            image_path,
        )
        frame["file_path"] = os.path.join(
            "images",
            "train",
            os.path.splitext(image_name)[0],
        ).replace("\\", "/")
        all_train_frames.append(frame)

    global_transforms_dir = os.path.join(object_dir, "global_transforms")
    os.makedirs(global_transforms_dir, exist_ok=True)
    write_transforms(
        os.path.join(global_transforms_dir, "transforms_train.json"),
        horizontal_fov,
        all_train_frames,
    )

    test_count = int(config.get("test_count", 8))
    if test_count < 1:
        raise ValueError("test_count must be at least 1")
    test_frames = []
    for image_number, direction in enumerate(fibonacci_lattice(test_count)):
        image_name = f"{image_number:03d}.png"
        image_path = os.path.join(image_test_dir, image_name)
        frame = render_frame(
            scene,
            camera,
            center,
            distance,
            direction,
            image_path,
        )
        frame["file_path"] = os.path.join(
            "images",
            "test",
            os.path.splitext(image_name)[0],
        ).replace("\\", "/")
        test_frames.append(frame)

    write_transforms(
        os.path.join(global_transforms_dir, "transforms_test.json"),
        horizontal_fov,
        test_frames,
    )

    with open(
        os.path.join(object_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    if config.get("save_blend_file", False):
        blend_path = os.path.join(object_dir, "rendered_scene.blend")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print(f"Finished rendering object '{object_id}' to: {object_dir}")


if __name__ == "__main__":
    main()
