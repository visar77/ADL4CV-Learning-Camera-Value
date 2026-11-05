import json
import os
import shutil
import subprocess
import argparse

# ---------------------------- Settings ----------------------------

OBJECTS_ROOT = os.path.join(os.path.dirname(__file__), "objects")
GLB_PATH = os.path.join(OBJECTS_ROOT, "gamecube.glb")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "views")
HDRI_DIR = os.path.join(os.path.dirname(__file__), "hdris")
BLENDER_EXE = None

OBJECT_ID = None  # None uses the GLB filename
WORKER = os.path.join(os.path.dirname(__file__), "blender_worker.py")

TEST_COUNT = 8
SEED = 77
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LATTICE_SIZE = 100
HORIZONTAL_FOV_DEG = 50.0
TRAJECTORY_ELEVATION_DEG = 20.0
CAMERA_DISTANCE_SCALE = 1.15  # normalized to object dimension
LIGHTING_MODE = "area"  # "area" or "hdri"

SAVE_BLEND_FILE = True
USE_GPU = False  # CPU is the safe default; enable GPU in the worker if configured.

# ------------------------------------------------------------------


def find_blender():
    if BLENDER_EXE:
        if os.path.isfile(BLENDER_EXE):
            return BLENDER_EXE
        raise FileNotFoundError(f"Blender not found: {BLENDER_EXE}")

    found = shutil.which("blender")
    if found:
        return found

    raise FileNotFoundError("Set BLENDER_EXE to your blender.exe path.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb-path", default=GLB_PATH)
    parser.add_argument("--object-id")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    glb_path = os.path.abspath(args.glb_path)
    if not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB not found: {glb_path}")
    if not glb_path.lower().endswith(".glb"):
        raise ValueError("GLB_PATH must point to a .glb file.")

    object_id = (
        args.object_id or OBJECT_ID or os.path.splitext(os.path.basename(glb_path))[0]
    )
    object_dir = os.path.join(args.output_root, object_id)
    os.makedirs(object_dir, exist_ok=True)
    config_dir = os.path.abspath(object_dir)

    config = {
        "glb_path": os.path.relpath(glb_path, config_dir),
        "object_id": object_id,
        "object_dir": ".",
        "hdri_dir": os.path.relpath(HDRI_DIR, config_dir),
        "test_count": TEST_COUNT,
        "seed": SEED,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
        "trajectory_elevation_deg": TRAJECTORY_ELEVATION_DEG,
        "camera_distance_scale": CAMERA_DISTANCE_SCALE,
        "save_blend_file": SAVE_BLEND_FILE,
        "use_gpu": USE_GPU,
        "lighting_mode": LIGHTING_MODE,
        "lattice_size": LATTICE_SIZE,
    }

    config_path = os.path.join(object_dir, "_worker_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    command = [
        find_blender(),
        "--background",
        "--python",
        WORKER,
        "--",
        "--config",
        config_path,
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
