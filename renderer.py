import json
import os
import shutil
import subprocess

# ---------------------------- Settings ----------------------------

GLB_PATH = r"C:\Users\lumi_vi\PycharmProjects\ObjaverseRenderer\objects\gamecube.glb"
OUTPUT_ROOT = r"C:\Users\lumi_vi\PycharmProjects\ObjaverseRenderer\views"
HDRI_DIR = r"C:\Users\lumi_vi\PycharmProjects\ObjaverseRenderer\hdris"
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

OBJECT_ID = None  # None uses the GLB filename
WORKER = os.path.join(os.path.dirname(__file__), "blender_worker.py")

EXPERIMENTS = [
    {"strategy": "furthest", "count": 16, "candidate_strategy": "furthest"},
    {"strategy": "random", "count": 8, "candidate_strategy": "random"},
]

TEST_COUNT = 8
SEED = 77
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LATTICE_SIZE = 200
HORIZONTAL_FOV_DEG = 50.0
TRAJECTORY_ELEVATION_DEG = 20.0
CAMERA_DISTANCE_SCALE = 1.15 # normalized to object dimension
LIGHTING_MODE = "area"  # "area" or "hdri"

SAVE_BLEND_FILE = True
USE_GPU = True  # CPU is the safe default; enable GPU in the worker if configured.

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
    if not os.path.isfile(GLB_PATH):
        raise FileNotFoundError(f"GLB not found: {GLB_PATH}")
    if not GLB_PATH.lower().endswith(".glb"):
        raise ValueError("GLB_PATH must point to a .glb file.")


    object_id = OBJECT_ID or os.path.splitext(os.path.basename(GLB_PATH))[0]
    object_dir = os.path.join(OUTPUT_ROOT, object_id)
    os.makedirs(object_dir, exist_ok=True)

    config = {
        "glb_path": os.path.abspath(GLB_PATH),
        "object_id": object_id,
        "object_dir": os.path.abspath(object_dir),
        "hdri_dir": os.path.abspath(HDRI_DIR),
        "experiments": EXPERIMENTS,
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
        "--python", WORKER,
        "--",
        "--config", config_path,
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
