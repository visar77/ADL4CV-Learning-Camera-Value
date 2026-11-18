"""Render datasets and train selected FastGS reconstructions."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys

from sampling import (
    candidate_configurations,
    fibonacci_lattice,
    select_candidate_indices,
    select_indices,
)
from training_fastgs import CONFIG as TRAINING_CONFIG


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "orchestrator_config.json"
TRAINER = ROOT / "training_fastgs.py"
RENDERER = ROOT / "renderer.py"
EVALUATOR = ROOT / "evaluate_metrics.py"
PYTHON = TRAINING_CONFIG["python"]


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Orchestrator config must be a JSON object: {CONFIG_PATH}")

    for key in (
        "objects_root",
        "views_root",
        "experiments_dir",
        "reconstructions_root",
        "results_root",
    ):
        config[key] = (ROOT / config[key]).resolve()
    if config.get("base_cloud_ply") is not None:
        config["base_cloud_ply"] = (ROOT / config["base_cloud_ply"]).resolve()
    return config


CONFIG = load_config()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", "--object_id")
    parser.add_argument("--experiment-id", "--experiment_id")
    parser.add_argument("--candidate-id", "--candidate_id")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument(
        "--use-candidate-base",
        "--use_candidate_base",
        nargs="?",
        const="base-3k",
        choices=("base-3k",),
        help="Train/use the shared 3k base for candidate reconstructions.",
    )
    parser.add_argument(
        "--do-not-evaluate",
        action="store_true",
        help="Skip automatic evaluation after reconstructions.",
    )
    parser.add_argument("--redo", action="store_true", help="Redo metric evaluation.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to other reconstructions after all retries for a job fail.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries after the initial attempt for FastGS/evaluation failures.",
    )
    return parser


def load_experiments(requested_id=None):
    experiments = {}
    for path in sorted(CONFIG["experiments_dir"].glob("*.json")):
        with path.open(encoding="utf-8") as file:
            experiment = json.load(file)
        if not isinstance(experiment, dict) or not isinstance(
            experiment.get("train"), dict
        ):
            raise ValueError(f"Invalid experiment definition: {path}")
        identifier = experiment.setdefault("id", path.stem)
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in (".", "..")
            or Path(identifier).name != identifier
        ):
            raise ValueError(f"Invalid experiment ID in {path}: {identifier!r}")
        if identifier in experiments:
            raise ValueError(f"Duplicate experiment ID: {identifier}")
        experiments[identifier] = experiment

    if requested_id is not None:
        if requested_id not in experiments:
            raise ValueError(f"Unknown experiment ID: {requested_id}")
        return [experiments[requested_id]]
    if not experiments:
        raise FileNotFoundError(f"No experiments found in {CONFIG['experiments_dir']}")
    return list(experiments.values())


def discover_objects(requested_id=None):
    if requested_id is not None:
        if not requested_id or Path(requested_id).name != requested_id:
            raise ValueError("Object ID must be a single directory name")
        return [requested_id]
    return sorted(path.stem for path in CONFIG["objects_root"].glob("*.glb"))


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def ensure_views(object_id, rerender):
    object_dir = CONFIG["views_root"] / object_id
    global_dir = object_dir / "global_transforms"
    train_manifest = global_dir / "transforms_train.json"
    test_manifest = global_dir / "transforms_test.json"
    if rerender or not (train_manifest.is_file() and test_manifest.is_file()):
        glb_path = CONFIG["objects_root"] / f"{object_id}.glb"
        if not glb_path.is_file():
            raise FileNotFoundError(f"GLB not found for {object_id}: {glb_path}")
        subprocess.run(
            [
                sys.executable,
                str(RENDERER),
                "--glb-path",
                str(glb_path),
                "--object-id",
                object_id,
                "--output-root",
                str(CONFIG["views_root"]),
            ],
            check=True,
        )
    if not (train_manifest.is_file() and test_manifest.is_file()):
        raise FileNotFoundError(f"Rendered train/test manifests not found: {global_dir}")
    return object_dir, load_json(train_manifest)


@contextmanager
def training_manifests(object_dir, global_train, global_test, indices):
    paths = [object_dir / f"transforms_{split}.json" for split in ("train", "test")]
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        train = {
            "camera_angle_x": global_train["camera_angle_x"],
            "frames": [dict(global_train["frames"][index]) for index in indices],
        }
        paths[0].write_text(json.dumps(train, indent=2), encoding="utf-8")
        paths[1].write_text(json.dumps(global_test, indent=2), encoding="utf-8")
        yield
    finally:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def checkpoint_path(model_dir, iterations):
    return model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"


def safely_remove_output(path):
    if path.is_symlink():
        raise ValueError(f"Refusing to remove symlink output: {path}")
    path = path.resolve()
    root = CONFIG["reconstructions_root"].resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Refusing to remove output outside reconstruction root: {path}")
    if path.exists():
        shutil.rmtree(path)


def run_training(object_dir, model_dir, indices, iterations, init_ply, retrain):
    expected = checkpoint_path(model_dir, iterations)
    if expected.is_file() and not retrain:
        return expected

    if model_dir.exists():
        safely_remove_output(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    arguments = [
        PYTHON,
        str(TRAINER),
        "--source-path",
        str(object_dir),
        "--model-path",
        str(model_dir),
        "--iterations",
        str(iterations),
        "--fastgs-dir",
        str(TRAINING_CONFIG["fastgs_dir"]),
    ]
    if init_ply is not None:
        arguments.extend(["--init-ply", str(init_ply)])
    with training_manifests(
        object_dir,
        load_json(object_dir / "global_transforms/transforms_train.json"),
        load_json(object_dir / "global_transforms/transforms_test.json"),
        indices,
    ):
        subprocess.run(arguments, check=True)
    if not expected.is_file():
        raise RuntimeError(f"FastGS did not write the expected checkpoint: {expected}")
    return expected


def run_with_retries(operation, retries, description):
    for attempt in range(retries + 1):
        try:
            return operation()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            if attempt == retries:
                raise
            print(
                f"{error.__class__.__name__} during {description} "
                f"(attempt {attempt + 1}/{retries + 1}): {error}; retrying"
            )


def make_candidate_jobs(experiment, lattice, base_indices, trajectory_elevation):
    train = experiment["train"]
    seed = int(train.get("seed", 0))
    configurations = candidate_configurations(experiment.get("candidates"))
    jobs = []
    seen = set()
    for strategy, count in configurations:
        indices = select_candidate_indices(
            strategy,
            count,
            lattice,
            base_indices,
            seed=seed,
            train_strategy=train["strategy"],
            trajectory_elevation_deg=trajectory_elevation,
        )
        for action, candidate_index in enumerate(indices, start=1):
            candidate_id = f"{strategy}-{action}"
            if candidate_id in seen:
                raise ValueError(
                    f"Candidate IDs are ambiguous in {experiment['id']}: {candidate_id}"
                )
            seen.add(candidate_id)
            candidate_train_indices = (
                list(base_indices)
                if strategy == "redundancy"
                else [*base_indices, candidate_index]
            )
            jobs.append(
                {
                    "id": candidate_id,
                    "strategy": strategy,
                    "train_indices": candidate_train_indices,
                    "status": "redundant" if strategy == "redundancy" else "pending",
                }
            )
    return jobs


def run_experiment(object_id, object_dir, global_train, experiment, args):
    experiment_id = experiment["id"]
    base_dir = CONFIG["reconstructions_root"] / object_id / experiment_id
    result_path = CONFIG["results_root"] / object_id / experiment_id / "results.json"
    metadata_path = object_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    lattice_size = int(metadata.get("lattice_size", len(global_train["frames"])))
    if lattice_size != len(global_train["frames"]):
        raise ValueError(
            f"Lattice metadata ({lattice_size}) does not match rendered train frames "
            f"({len(global_train['frames'])}) for {object_id}"
        )
    lattice = fibonacci_lattice(lattice_size)
    train = experiment["train"]
    seed = int(train.get("seed", 0))
    trajectory_elevation = float(metadata.get("trajectory_elevation_deg", 20.0))
    base_indices = select_indices(
        train["strategy"],
        int(train["count"]),
        lattice,
        seed=seed,
        trajectory_elevation_deg=trajectory_elevation,
    )
    candidates = make_candidate_jobs(
        experiment, lattice, base_indices, trajectory_elevation
    )
    if args.candidate_id is not None:
        candidate_ids = {candidate["id"] for candidate in candidates}
        if args.candidate_id not in candidate_ids:
            raise ValueError(
                f"Unknown candidate ID {args.candidate_id!r} in {object_id}/{experiment_id}; "
                f"available: {', '.join(sorted(candidate_ids))}"
            )
        candidates = [
            item for item in candidates if item["id"] == args.candidate_id
        ]

    base_iterations = int(TRAINING_CONFIG["base_iterations"])
    base_model_dir = base_dir / "base"
    base_ply = checkpoint_path(base_model_dir, base_iterations)
    uses_internal_base = (
        args.use_candidate_base is not None and CONFIG["base_cloud_ply"] is None
    )
    run_base = (
        args.candidate_id is None
        or any(candidate["status"] == "redundant" for candidate in candidates)
        or (uses_internal_base and not base_ply.is_file())
    )
    result = {
        "object_id": object_id,
        "experiment_id": experiment_id,
        "dataset_path": str(object_dir),
        "reconstructions": {},
    }
    if result_path.is_file():
        previous = load_json(result_path)
        if (
            previous.get("object_id") == object_id
            and previous.get("experiment_id") == experiment_id
        ):
            result["reconstructions"].update(previous.get("reconstructions", {}))
    if run_base:
        base_record = {
            "status": "running",
            "iterations": base_iterations,
            "reconstruction_path": str(base_model_dir),
            "train_lattice_indices": base_indices,
        }
        result["reconstructions"]["base"] = base_record
        save_json(result_path, result)
        try:
            checkpoint = run_with_retries(
                lambda: run_training(
                    object_dir,
                    base_model_dir,
                    base_indices,
                    base_iterations,
                    None,
                    args.retrain if args.candidate_id is None else False,
                ),
                args.retries,
                "base reconstruction",
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            base_record.update(status="failed", error=str(error))
            if args.use_candidate_base is not None and uses_internal_base:
                result.update(
                    status="skipped",
                    skip_reason=(
                        "Candidate-base experiment skipped because the base "
                        f"reconstruction failed: {error}"
                    ),
                )
                save_json(result_path, result)
                print(
                    f"Skipping experiment {object_id}/{experiment_id}: "
                    f"candidate base reconstruction failed: {error}"
                )
                return
            save_json(result_path, result)
            if not args.continue_on_error:
                raise
            print(f"Continuing after base reconstruction failed: {error}")
        else:
            base_record.update(
                status="complete",
                checkpoint_path=str(checkpoint),
            )
            base_record.pop("error", None)
            save_json(result_path, result)
            if not args.do_not_evaluate:
                evaluate_reconstruction_with_retries(
                    result_path, result, "base", args
                )
    if "base" not in result["reconstructions"] and (
        run_base or (uses_internal_base and base_ply.is_file())
    ):
        result["reconstructions"]["base"] = {
            "status": "complete",
            "iterations": base_iterations,
            "reconstruction_path": str(base_model_dir),
            "checkpoint_path": str(base_ply),
            "train_lattice_indices": base_indices,
        }
        save_json(result_path, result)
        if not run_base and uses_internal_base and not args.do_not_evaluate:
            evaluate_reconstruction_with_retries(
                result_path, result, "base", args
            )
    elif not run_base and uses_internal_base and base_ply.is_file():
        base_record = result["reconstructions"]["base"]
        base_record.update(
            status="complete",
            iterations=base_iterations,
            reconstruction_path=str(base_model_dir),
            checkpoint_path=str(base_ply),
            train_lattice_indices=base_indices,
        )
        base_record.pop("error", None)
        save_json(result_path, result)
        if not args.do_not_evaluate:
            evaluate_reconstruction_with_retries(
                result_path, result, "base", args
            )

    candidate_base_ply = None
    candidate_iterations = int(TRAINING_CONFIG["candidate_iterations"])
    if args.use_candidate_base:
        candidate_iterations = int(TRAINING_CONFIG["candidate_base_iterations"])
        if CONFIG["base_cloud_ply"] is not None:
            candidate_base_ply = CONFIG["base_cloud_ply"]
            if not candidate_base_ply.is_file():
                raise FileNotFoundError(
                    f"Configured base cloud not found: {candidate_base_ply}"
                )
        else:
            if not base_ply.is_file():
                if args.continue_on_error and result["reconstructions"].get(
                    "base", {}
                ).get("status") == "failed":
                    candidate_base_ply = None
                else:
                    raise RuntimeError(f"Base reconstruction is missing its PLY: {base_ply}")
            else:
                candidate_base_ply = base_ply

    save_json(result_path, result)

    for candidate in candidates:
        candidate_id = candidate["id"]
        if candidate["status"] == "redundant":
            result["reconstructions"][candidate_id] = {
                "status": "redundant",
                "iterations": base_iterations,
                "reconstruction_path": str(base_model_dir),
                "train_lattice_indices": candidate["train_indices"],
            }
            save_json(result_path, result)
            continue

        model_dir = base_dir / candidate_id
        if model_dir == base_model_dir:
            raise ValueError(f"Candidate ID conflicts with reserved path: {candidate_id}")
        record = {
            "status": "running",
            "iterations": candidate_iterations,
            "reconstruction_path": str(model_dir),
            "train_lattice_indices": candidate["train_indices"],
        }
        result["reconstructions"][candidate_id] = record
        save_json(result_path, result)
        if args.use_candidate_base and uses_internal_base and candidate_base_ply is None:
            error = "Candidate base reconstruction failed; its PLY is unavailable."
            record.update(status="failed", error=error)
            save_json(result_path, result)
            continue
        try:
            checkpoint = run_with_retries(
                lambda: run_training(
                    object_dir,
                    model_dir,
                    candidate["train_indices"],
                    candidate_iterations,
                    candidate_base_ply,
                    args.retrain,
                ),
                args.retries,
                f"{candidate_id} reconstruction",
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            record.update(status="failed", error=str(error))
            save_json(result_path, result)
            if not args.continue_on_error:
                raise
            print(f"Continuing after {candidate_id} reconstruction failed: {error}")
            continue
        record.update(status="complete", checkpoint_path=str(checkpoint))
        record.pop("error", None)
        save_json(result_path, result)
        if not args.do_not_evaluate:
            evaluate_reconstruction_with_retries(
                result_path, result, candidate_id, args
            )


def evaluate_one(result_path, result, reconstruction_id, redo):
    command = [
        PYTHON,
        str(EVALUATOR),
        "--results-root",
        str(CONFIG["results_root"]),
        "--fastgs-dir",
        str(TRAINING_CONFIG["fastgs_dir"]),
        "--python",
        PYTHON,
        "--object-id",
        result["object_id"],
        "--experiment-id",
        result["experiment_id"],
        "--reconstruction-id",
        reconstruction_id,
    ]
    if redo:
        command.append("--redo")
    subprocess.run(command, check=True)
    updated = load_json(result_path)
    metrics = updated.get("metrics", {}).get(reconstruction_id)
    if not (
        isinstance(metrics, dict)
        and "averages" in metrics
        and "per_view" in metrics
    ):
        raise RuntimeError(
            f"Metric evaluation did not complete for {result['object_id']}/"
            f"{result['experiment_id']}/{reconstruction_id}"
        )
    result.clear()
    result.update(updated)


def evaluate_reconstruction_with_retries(result_path, result, name, args):
    record = result["reconstructions"][name]
    record["evaluation_status"] = "running"
    record.pop("evaluation_error", None)
    save_json(result_path, result)
    try:
        run_with_retries(
            lambda: evaluate_one(result_path, result, name, args.redo),
            args.retries,
            f"{name} metric evaluation",
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        record = result["reconstructions"][name]
        record.update(evaluation_status="failed", evaluation_error=str(error))
        save_json(result_path, result)
        if not args.continue_on_error:
            raise
        print(f"Continuing after {name} metric evaluation failed: {error}")
    else:
        record = result["reconstructions"][name]
        record["evaluation_status"] = "complete"
        record.pop("evaluation_error", None)
        save_json(result_path, result)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    object_ids = discover_objects(args.object_id)
    if not object_ids:
        raise FileNotFoundError(f"No GLBs found in {CONFIG['objects_root']}")
    experiments = load_experiments(args.experiment_id)

    for object_id in object_ids:
        object_dir, global_train = ensure_views(object_id, args.rerender)
        for experiment in experiments:
            run_experiment(object_id, object_dir, global_train, experiment, args)

if __name__ == "__main__":
    main()
