"""Render FastGS test views and store per-view metrics in results.json."""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FASTGS_DIR = ROOT.parent / "FastGS"
DEFAULT_ITERATION = 30000
if str(FASTGS_DIR) not in sys.path:
    sys.path.insert(0, str(FASTGS_DIR))


def evaluate_reconstruction(reconstruction_dir, dataset_dir, train_indices, iteration):
    global_manifest = dataset_dir / "global_transforms" / "transforms_train.json"
    with global_manifest.open(encoding="utf-8") as file:
        global_train = json.load(file)
    train_frames = [
        global_train["frames"][index] for index in train_indices
    ]
    train_manifest = dataset_dir / "transforms_train.json"
    with train_manifest.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "camera_angle_x": global_train["camera_angle_x"],
                "frames": train_frames,
            },
            file,
            indent=2,
        )

    try:
        subprocess.run(
            [
                sys.executable,
                str(FASTGS_DIR / "render.py"),
                "--model_path",
                str(reconstruction_dir),
                "--iteration",
                str(iteration),
                "--skip_train",
            ],
            cwd=FASTGS_DIR,
            check=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_dir = Path(temp_dir)
            (metrics_dir / "test").symlink_to(
                (reconstruction_dir / "test").resolve(),
                target_is_directory=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(FASTGS_DIR / "metrics.py"),
                    "--model_paths",
                    str(metrics_dir),
                ],
                cwd=FASTGS_DIR,
                check=True,
            )
            with (metrics_dir / "results.json").open(encoding="utf-8") as file:
                averages = json.load(file)[f"ours_{iteration}"]
            with (metrics_dir / "per_view.json").open(encoding="utf-8") as file:
                per_view_by_metric = json.load(file)[f"ours_{iteration}"]

        per_view = {}
        for metric, scores in per_view_by_metric.items():
            for name, score in scores.items():
                per_view.setdefault(Path(name).stem, {})[metric] = score
        return {"averages": averages, "per_view": per_view}
    finally:
        train_manifest.unlink(missing_ok=True)


def load_reconstructions(result):
    if "reconstructions" in result:
        return result["reconstructions"]
    return {
        "base": {
            "reconstruction_path": result["reconstruction_path"],
            "train_lattice_indices": result["train_lattice_indices"],
        }
    }


def has_metrics_for_iteration(result, metrics, name, iteration):
    subset_metrics = metrics.get(name)
    if not (
        isinstance(subset_metrics, dict)
        and "averages" in subset_metrics
        and "per_view" in subset_metrics
    ):
        return False

    metrics_iterations = result.get("metrics_iterations", {})
    recorded_iteration = metrics_iterations.get(name)
    if recorded_iteration is None:
        recorded_iteration = result.get("metrics_iteration")
    if recorded_iteration is None:
        recorded_iteration = DEFAULT_ITERATION
    return recorded_iteration == iteration


def copy_redundancy_metrics(result, metrics, reconstructions, iteration):
    redundancy_names = [
        name for name in reconstructions if name.startswith("redundancy-")
    ]
    if not has_metrics_for_iteration(result, metrics, "base", iteration):
        for name in redundancy_names:
            metrics.pop(name, None)
            result.get("metrics_iterations", {}).pop(name, None)
        return

    for name in redundancy_names:
        metrics[name] = copy.deepcopy(metrics["base"])
        result.setdefault("metrics_iterations", {})[name] = iteration


def save_result(result_path, result):
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    results_root = ROOT / "results"
    paths = sorted(results_root.glob("*/*/results.json"))
    if not paths:
        print(
            f"No experiment results found under {results_root}. "
            "Run orchestrator.py first."
        )
        return

    if args.object_id:
        paths = [path for path in paths if path.parent.parent.name == args.object_id]
    if args.experiment_id:
        paths = [path for path in paths if path.parent.name == args.experiment_id]
    if not paths:
        filters = []
        if args.object_id:
            filters.append(f"object ID {args.object_id!r}")
        if args.experiment_id:
            filters.append(f"experiment ID {args.experiment_id!r}")
        print(f"No experiment results match {' and '.join(filters)}.")
        return

    for result_path in paths:
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
        dataset_dir = Path(result["dataset_path"])
        reconstructions = load_reconstructions(result)
        metrics = result.setdefault("metrics", {})
        if "averages" in metrics and "per_view" in metrics:
            metrics = {"base": metrics}
            result["metrics"] = metrics

        for name, reconstruction in reconstructions.items():
            if name.startswith("redundancy-"):
                continue

            if not args.redo and has_metrics_for_iteration(
                result, metrics, name, args.iteration
            ):
                print(
                    f"Skipping {result.get('object_id', result_path.parent.parent.name)}"
                    f"/{result.get('experiment_id', result_path.parent.name)}"
                    f"/{name}: metrics already complete"
                )
                continue

            reconstruction_dir = Path(reconstruction["reconstruction_path"])
            checkpoint = (
                reconstruction_dir
                / "point_cloud"
                / f"iteration_{args.iteration}"
                / "point_cloud.ply"
            )
            if not checkpoint.is_file():
                print(
                    f"Skipping {result.get('object_id', result_path.parent.parent.name)}"
                    f"/{result.get('experiment_id', result_path.parent.name)}"
                    f"/{name}: can't find checkpoint {checkpoint}"
                )
                continue

            metrics[name] = evaluate_reconstruction(
                reconstruction_dir,
                dataset_dir,
                reconstruction["train_lattice_indices"],
                args.iteration,
            )
            result.setdefault("metrics_iterations", {})[name] = args.iteration
            result["metrics_iteration"] = args.iteration
            save_result(result_path, result)

        copy_redundancy_metrics(result, metrics, reconstructions, args.iteration)
        save_result(result_path, result)


if __name__ == "__main__":
    main()
