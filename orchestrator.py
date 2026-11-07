"""Render a shared view corpus and train FastGS for configured experiments."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from sampling import fibonacci_lattice, select_indices

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_ROOT = ROOT / "views"
OBJECTS_ROOT = ROOT / "objects"
RECONSTRUCTIONS_ROOT = ROOT / "reconstructions"
RESULTS_ROOT = ROOT / "results"


def load_experiment(path):
    with path.open(encoding="utf-8") as file:
        experiment = json.load(file)
    experiment.setdefault("id", path.stem)
    return experiment


def write_manifest(path, camera_angle_x, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"camera_angle_x": camera_angle_x, "frames": frames}, file, indent=2)


def make_dataset(object_dir, experiment, lattice):
    global_dir = object_dir / "global_transforms"
    with (global_dir / "transforms_train.json").open(encoding="utf-8") as file:
        global_train = json.load(file)
    metadata_path = object_dir / "metadata.json"
    trajectory_elevation = 20.0
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        trajectory_elevation = float(
            metadata.get("trajectory_elevation_deg", trajectory_elevation)
        )

    train_config = experiment["train"]
    train_indices = select_indices(
        train_config["strategy"],
        int(train_config["count"]),
        lattice,
        seed=int(train_config["seed"]),
        trajectory_elevation_deg=trajectory_elevation,
    )
    frames = []
    for index in train_indices:
        frame = dict(global_train["frames"][index])
        frames.append(frame)

    write_manifest(
        object_dir / "transforms_train.json",
        global_train["camera_angle_x"],
        frames,
    )
    return train_indices, [frame["file_path"] for frame in frames]


def reconstruction_complete(path):
    return (path / "point_cloud" / "iteration_30000" / "point_cloud.ply").is_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id", action="append")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    args = parser.parse_args()

    object_ids = args.object_id or sorted(
        path.stem for path in OBJECTS_ROOT.glob("*.glb")
    )
    if not object_ids:
        raise FileNotFoundError(f"No .glb files found in {OBJECTS_ROOT}")

    experiment_files = sorted(EXPERIMENTS_DIR.glob("*.json"))
    for object_id in object_ids:
        object_dir = OUTPUT_ROOT / object_id
        glb_path = OBJECTS_ROOT / f"{object_id}.glb"
        if not glb_path.is_file():
            raise FileNotFoundError(f"Object GLB not found: {glb_path}")
        global_dir = object_dir / "global_transforms"
        if not args.skip_render and (
            args.rerender
            or not (global_dir / "transforms_train.json").is_file()
            or not (global_dir / "transforms_test.json").is_file()
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "renderer.py"),
                    "--glb-path",
                    str(glb_path),
                    "--object-id",
                    object_id,
                ],
                check=True,
            )

        with (global_dir / "transforms_train.json").open(encoding="utf-8") as file:
            global_train = json.load(file)
        with (global_dir / "transforms_test.json").open(encoding="utf-8") as file:
            global_test = json.load(file)
        write_manifest(
            object_dir / "transforms_test.json",
            global_test["camera_angle_x"],
            global_test["frames"],
        )
        lattice = fibonacci_lattice(len(global_train["frames"]))

        for experiment_path in experiment_files:
            experiment = load_experiment(experiment_path)
            train_indices, train_image_paths = make_dataset(
                object_dir,
                experiment,
                lattice,
            )
            reconstruction_dir = RECONSTRUCTIONS_ROOT / object_id / experiment["id"]
            result_path = RESULTS_ROOT / object_id / experiment["id"] / "results.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                "object_id": object_id,
                "experiment_id": experiment["id"],
                "dataset_path": str(object_dir),
                "reconstruction_path": str(reconstruction_dir),
                "train_lattice_indices": train_indices,
                "train_image_paths": train_image_paths,
                "train_manifest": str(object_dir / "transforms_train.json"),
                "test_manifest": str(object_dir / "transforms_test.json"),
                "status": (
                    "complete"
                    if reconstruction_complete(reconstruction_dir)
                    else "pending"
                ),
                "metrics": {},
            }
            if result_path.is_file():
                with result_path.open(encoding="utf-8") as file:
                    previous = json.load(file)
                record["metrics"] = previous.get("metrics", {})
                record["metrics_iteration"] = previous.get("metrics_iteration")
            with result_path.open("w", encoding="utf-8") as file:
                json.dump(record, file, indent=2)

            try:
                if not args.force_retrain and reconstruction_complete(
                    reconstruction_dir
                ):
                    continue
                record["status"] = "running"
                with result_path.open("w", encoding="utf-8") as file:
                    json.dump(record, file, indent=2)
                reconstruction_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(args.fastgs_dir / "train.py"),
                        "--source_path",
                        str(object_dir),
                        "--model_path",
                        str(reconstruction_dir),
                        "--eval",
                        "--densification_interval",
                        "500",
                        "--optimizer_type",
                        "default",
                        "--test_iterations",
                        "30000",
                        "--save_iterations",
                        "30000",
                        "--grad_abs_thresh",
                        "0.0012",
                    ],
                    cwd=args.fastgs_dir,
                    check=True,
                )
                record["status"] = "complete"
                with result_path.open("w", encoding="utf-8") as file:
                    json.dump(record, file, indent=2)
            except Exception:
                record["status"] = "failed"
                with result_path.open("w", encoding="utf-8") as file:
                    json.dump(record, file, indent=2)
                if not args.continue_on_error:
                    raise
            finally:
                (object_dir / "transforms_train.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
