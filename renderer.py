import argparse
import json
import os
import shutil
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")

    for key in ("objects_root", "output_root"):
        config[key] = os.path.abspath(os.path.join(SCRIPT_DIR, config[key]))
    if config["blender_exe"] is not None:
        config["blender_exe"] = os.path.abspath(
            os.path.join(SCRIPT_DIR, config["blender_exe"])
        )
    return config


CONFIG = load_config()


def find_blender():
    blender_exe = CONFIG["blender_exe"]
    if blender_exe:
        if os.path.isfile(blender_exe):
            return blender_exe
        raise FileNotFoundError(f"Blender not found: {blender_exe}")

    found = shutil.which("blender")
    if found:
        return found

    raise FileNotFoundError(
        "Set blender_exe to your Blender executable in config.json."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render an object's view corpus.")
    parser.add_argument("--glb-path", help="Path to the source GLB.")
    parser.add_argument("--object-id", required=True, help="Output directory name.")
    parser.add_argument(
        "--output-root",
        default=CONFIG["output_root"],
        help="Corpus root directory (defaults to config.json).",
    )
    args = parser.parse_args(argv)

    glb_path = os.path.abspath(
        args.glb_path
        or os.path.join(CONFIG["objects_root"], f"{args.object_id}.glb")
    )
    if not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB not found: {glb_path}")
    if not glb_path.lower().endswith(".glb"):
        raise ValueError("--glb-path must point to a .glb file.")

    object_id = args.object_id
    if object_id in (".", "..") or os.path.basename(object_id) != object_id:
        parser.error("Object ID must be a single directory name")
    object_dir = os.path.abspath(os.path.join(args.output_root, object_id))
    os.makedirs(object_dir, exist_ok=True)
    config_dir = os.path.abspath(object_dir)

    config = {
        "glb_path": os.path.relpath(glb_path, config_dir),
        "object_id": object_id,
        "object_dir": ".",
        "image_width": CONFIG["image_width"],
        "image_height": CONFIG["image_height"],
        "horizontal_fov_deg": CONFIG["horizontal_fov_deg"],
        "trajectory_elevation_deg": CONFIG["trajectory_elevation_deg"],
        "camera_distance_scale": CONFIG["camera_distance_scale"],
        "save_blend_file": CONFIG["save_blend_file"],
        "use_gpu": CONFIG["use_gpu"],
        "lattice_size": CONFIG["lattice_size"],
    }

    config_path = os.path.join(object_dir, "_worker_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    command = [
        find_blender(),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        os.path.join(SCRIPT_DIR, "blender_worker.py"),
        "--",
        "--config",
        config_path,
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
