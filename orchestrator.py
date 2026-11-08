"""Render a shared view corpus and train FastGS for configured experiments."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from sampling import (
    candidate_configurations,
    fibonacci_lattice,
    select_candidate_indices,
    select_indices,
)

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

    return (
        train_indices,
        [frame["file_path"] for frame in frames],
        trajectory_elevation,
    )


def write_training_manifest(object_dir, global_train, train_indices):
    frames = [dict(global_train["frames"][index]) for index in train_indices]
    write_manifest(
        object_dir / "transforms_train.json",
        global_train["camera_angle_x"],
        frames,
    )


def reconstruction_complete(path):
    return (path / "point_cloud" / "iteration_30000" / "point_cloud.ply").is_file()


def train_reconstruction(fastgs_dir, object_dir, reconstruction_dir):
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(fastgs_dir / "train.py"),
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
        cwd=fastgs_dir,
        check=True,
    )


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
            train_config = experiment["train"]
            (
                train_indices,
                train_image_paths,
                trajectory_elevation,
            ) = make_dataset(object_dir, experiment, lattice)
            base_reconstruction_dir = (
                RECONSTRUCTIONS_ROOT / object_id / experiment["id"] / "base"
            )
            result_path = RESULTS_ROOT / object_id / experiment["id"] / "results.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)

            reconstruction_records = {
                "base": {
                    "reconstruction_path": str(base_reconstruction_dir),
                    "train_lattice_indices": train_indices,
                    "train_image_paths": train_image_paths,
                    "status": (
                        "complete"
                        if reconstruction_complete(base_reconstruction_dir)
                        else "pending"
                    ),
                }
            }
            candidate_runs = []
            for strategy, count in candidate_configurations(
                experiment.get("candidates")
            ):
                candidate_indices = select_candidate_indices(
                    strategy=strategy,
                    count=count,
                    lattice=lattice,
                    training_indices=train_indices,
                    seed=int(train_config.get("seed", 0)),
                    train_strategy=train_config["strategy"],
                    trajectory_elevation_deg=trajectory_elevation,
                )
                for index, candidate_index in enumerate(candidate_indices, start=1):
                    candidate_key = f"{strategy}-{count}-{index}"
                    candidate_train_indices = (
                        train_indices
                        if strategy == "redundancy"
                        else train_indices + [candidate_index]
                    )
                    candidate_path = (
                        RECONSTRUCTIONS_ROOT
                        / object_id
                        / experiment["id"]
                        / candidate_key
                    )
                    if strategy != "redundancy":
                        candidate_runs.append(candidate_key)
                    reconstruction_records[candidate_key] = {
                        "candidate_lattice_index": candidate_index,
                        "reconstruction_path": (
                            str(base_reconstruction_dir)
                            if strategy == "redundancy"
                            else str(candidate_path)
                        ),
                        "train_lattice_indices": candidate_train_indices,
                        "train_image_paths": [
                            global_train["frames"][i]["file_path"]
                            for i in candidate_train_indices
                        ],
                        "status": (
                            "redundant"
                            if strategy == "redundancy"
                            else (
                                "complete"
                                if reconstruction_complete(candidate_path)
                                else "pending"
                            )
                        ),
                    }

            record = {
                "object_id": object_id,
                "experiment_id": experiment["id"],
                "dataset_path": str(object_dir),
                "reconstruction_path": str(base_reconstruction_dir),
                "train_lattice_indices": train_indices,
                "train_image_paths": train_image_paths,
                "train_manifest": str(object_dir / "transforms_train.json"),
                "test_manifest": str(object_dir / "transforms_test.json"),
                "reconstructions": reconstruction_records,
                "status": "pending",
                "metrics": {
                    name: {}
                    for name in ("base", *candidate_runs)
                },
            }
            if result_path.is_file():
                with result_path.open(encoding="utf-8") as file:
                    previous = json.load(file)
                previous_metrics = previous.get("metrics", {})
                if "averages" in previous_metrics and "per_view" in previous_metrics:
                    previous_metrics = {"base": previous_metrics}
                for name in record["metrics"]:
                    record["metrics"][name] = previous_metrics.get(name, {})
                record["metrics_iteration"] = previous.get("metrics_iteration")

            try:
                for name, reconstruction in reconstruction_records.items():
                    if reconstruction["status"] == "redundant":
                        continue
                    reconstruction_dir = Path(reconstruction["reconstruction_path"])
                    if not args.force_retrain and reconstruction_complete(
                        reconstruction_dir
                    ):
                        continue

                    reconstruction["status"] = "running"
                    record["status"] = "running"
                    with result_path.open("w", encoding="utf-8") as file:
                        json.dump(record, file, indent=2)

                    if name == "base":
                        indices = train_indices
                    else:
                        indices = reconstruction["train_lattice_indices"]
                    write_training_manifest(object_dir, global_train, indices)
                    try:
                        train_reconstruction(
                            args.fastgs_dir,
                            object_dir,
                            reconstruction_dir,
                        )
                        reconstruction["status"] = "complete"
                    finally:
                        (object_dir / "transforms_train.json").unlink(missing_ok=True)

                    with result_path.open("w", encoding="utf-8") as file:
                        json.dump(record, file, indent=2)

                record["status"] = (
                    "complete"
                    if all(
                        item["status"] in ("complete", "redundant")
                        for item in reconstruction_records.values()
                    )
                    else "pending"
                )
                with result_path.open("w", encoding="utf-8") as file:
                    json.dump(record, file, indent=2)
            except Exception:
                record["status"] = "failed"
                for reconstruction in reconstruction_records.values():
                    if reconstruction["status"] == "running":
                        reconstruction["status"] = "failed"
                with result_path.open("w", encoding="utf-8") as file:
                    json.dump(record, file, indent=2)
                if not args.continue_on_error:
                    raise
            finally:
                (object_dir / "transforms_train.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
