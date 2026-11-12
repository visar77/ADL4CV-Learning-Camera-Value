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
        config = json.load(file)
    config["_directory"] = os.path.dirname(os.path.abspath(config_path))
    return config


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
    bounding_radius = (maximum - minimum).length / 2.0
    if not math.isfinite(bounding_radius) or bounding_radius <= 0:
        raise ValueError("Imported object has empty or invalid bounds")
    return center, bounding_radius


def _create_light(
    name, light_type, location, rotation, energy, use_shadow=False, specular_factor=1.0
):
    """Create a light as in the official Objaverse-XL script."""
    import bpy

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting():
    """Sample the official Objaverse-XL rig once per object."""
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )
    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


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
    camera_data.clip_start = max(bounding_radius * 0.001, 1e-7)

    distance_scale = config.get("camera_distance_scale", 1.15)
    distance = (
        bounding_radius / max(math.sin(horizontal_fov / 2.0), 0.01) * distance_scale
    )
    camera_data.clip_end = max(distance + bounding_radius * 10.0, 100.0)

    return camera, distance, horizontal_fov


def configure_render(scene, config):
    """Use the Cycles settings from the supplied Objaverse-XL script."""
    import bpy

    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(config["image_width"])
    scene.render.resolution_y = int(config["image_height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"

    # GPU is optional. CPU remains the fallback if a usable device isn't found.
    if config.get("use_gpu", False):
        preferences = bpy.context.preferences.addons["cycles"].preferences
        usable_gpu = False
        try:
            preferences.compute_device_type = "CUDA"
            preferences.refresh_devices()
            for device in preferences.devices:
                device.use = device.type == "CUDA"
                usable_gpu |= device.use
                print("Cycles device:", device.name, device.type, device.use)
        except (TypeError, RuntimeError) as error:
            print(f"CUDA unavailable; using CPU: {error}")

        scene.cycles.device = "GPU" if usable_gpu else "CPU"


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
    from sampling import fibonacci_lattice, test_view_directions

    config = load_config()
    config_dir = config["_directory"]

    glb_path = os.path.abspath(os.path.join(config_dir, config["glb_path"]))
    object_id = safe_name(config["object_id"])
    object_dir = os.path.abspath(os.path.join(config_dir, config["object_dir"]))

    if not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    os.makedirs(object_dir, exist_ok=True)
    scene = bpy.context.scene
    center, bounding_radius = import_model(glb_path)

    lights = randomize_lighting()
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
        "glb_path": os.path.relpath(glb_path, object_dir).replace("\\", "/"),
        "lighting": "objaverse_xl",
        "light_energies": {name: light.data.energy for name, light in lights.items()},
        "lattice_size": lattice_size,
        "trajectory_elevation_deg": trajectory_elevation,
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

    test_frames = []
    for image_number, direction in enumerate(test_view_directions()):
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
