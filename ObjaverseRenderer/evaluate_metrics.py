"""Render FastGS test views and store per-view metrics in results.json."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from gaussian_initialization import sha256

ROOT = Path(__file__).resolve().parent
FASTGS_DIR = ROOT.parent / "FastGS"
DEFAULT_ITERATION = 30000
if str(FASTGS_DIR) not in sys.path:
    sys.path.insert(0, str(FASTGS_DIR))


def evaluate_reconstruction(reconstruction_dir, dataset_dir, train_indices, iteration,
                            fastgs_dir=FASTGS_DIR, python=sys.executable):
    global_manifest = dataset_dir / "global_transforms" / "transforms_train.json"
    with global_manifest.open(encoding="utf-8") as file:
        global_train = json.load(file)
    train_frames = [
        global_train["frames"][index] for index in train_indices
    ]
    train_manifest = dataset_dir / "transforms_train.json"
    old_train = train_manifest.read_bytes() if train_manifest.exists() else None
    test_manifest = dataset_dir / "transforms_test.json"
    old_test = test_manifest.read_bytes() if test_manifest.exists() else None
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
        test_manifest.write_bytes((dataset_dir / "global_transforms/transforms_test.json").read_bytes())
        subprocess.run(
            [
                python,
                str(fastgs_dir / "render.py"),
                "--model_path",
                str(reconstruction_dir),
                "--iteration",
                str(iteration),
                "--source_path",
                str(dataset_dir),
                "--eval",
                "--skip_train",
            ],
            cwd=fastgs_dir,
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
                    python,
                    str(fastgs_dir / "metrics.py"),
                    "--model_paths",
                    str(metrics_dir),
                ],
                cwd=fastgs_dir,
                check=True,
            )
            if not (metrics_dir / "results.json").is_file() or not (metrics_dir / "per_view.json").is_file():
                raise RuntimeError("FastGS metrics.py did not produce metrics; inspect its preceding CUDA/LPIPS error output")
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
        for path, previous in ((train_manifest, old_train), (test_manifest, old_test)):
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)


def evaluation_fingerprint(dataset_dir):
    """Identify actual held-out cameras and image bytes, independent of device paths."""
    manifest = json.loads((dataset_dir / "global_transforms/transforms_test.json").read_text())
    if not manifest.get("frames"):
        raise ValueError(f"Empty test set: {dataset_dir}")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
    for frame in manifest["frames"]:
        image = dataset_dir / frame["file_path"]
        if not image.suffix:
            image = image.with_suffix(".png")
        digest.update(sha256(image).encode())
    return digest.hexdigest()


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


def reconstruction_iteration(reconstruction, override=None):
    return override if override is not None else reconstruction.get("iterations", DEFAULT_ITERATION)


def copy_redundancy_metrics(result, metrics, reconstructions, iteration=None):
    iteration = reconstruction_iteration(reconstructions.get("base", {}), iteration)
    redundancy_names = [
        name for name, item in reconstructions.items()
        if name.startswith("redundancy-") or item.get("status") == "redundant"
    ]
    if not has_metrics_for_iteration(result, metrics, "base", iteration):
        for name in redundancy_names:
            metrics.pop(name, None)
            result.get("metrics_iterations", {}).pop(name, None)
        return

    for name in redundancy_names:
        metrics[name] = copy.deepcopy(metrics["base"])
        result.setdefault("metrics_iterations", {})[name] = iteration
        if "base" in result.get("metrics_provenance", {}):
            result["metrics_provenance"][name] = copy.deepcopy(result["metrics_provenance"]["base"])


def save_result(result_path, result):
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def main(argv=None, shared_cache=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--reconstruction-id")
    parser.add_argument("--iteration", type=int,
                        help="Override every checkpoint iteration; default uses each reconstruction's recorded budget.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--views-root", type=Path, help="Override dataset paths after transferring to another device.")
    parser.add_argument("--reconstructions-root", type=Path,
                        help="Override the run's reconstruction root after transferring to another device.")
    parser.add_argument("--fastgs-dir", type=Path, default=FASTGS_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args(argv)
    if args.iteration is not None and args.iteration < 1:
        parser.error("iteration must be positive")
    args.fastgs_dir = args.fastgs_dir.expanduser().resolve()

    results_root = args.results_root.expanduser().resolve()
    paths = sorted(results_root.rglob("results.json"))
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
    if args.reconstruction_id:
        matched_paths = []
        for path in paths:
            with path.open(encoding="utf-8") as file:
                result = json.load(file)
            if args.reconstruction_id in load_reconstructions(result):
                matched_paths.append(path)
        paths = matched_paths
    if not paths:
        filters = []
        if args.object_id:
            filters.append(f"object ID {args.object_id!r}")
        if args.experiment_id:
            filters.append(f"experiment ID {args.experiment_id!r}")
        if args.reconstruction_id:
            filters.append(f"reconstruction ID {args.reconstruction_id!r}")
        print(f"No experiment results match {' and '.join(filters)}.")
        return

    evaluation_cache = shared_cache if shared_cache is not None else {}
    for result_path in paths:
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
        dataset_dir = Path(result["dataset_path"])
        if args.views_root:
            dataset_dir = args.views_root.expanduser().resolve() / result["object_id"]
        reconstructions = load_reconstructions(result)
        if args.reconstruction_id:
            if args.reconstruction_id not in reconstructions:
                raise ValueError(
                    f"Reconstruction {args.reconstruction_id!r} not found in {result_path}"
                )
            reconstructions = {
                args.reconstruction_id: reconstructions[args.reconstruction_id]
            }
        metrics = result.setdefault("metrics", {})
        if "averages" in metrics and "per_view" in metrics:
            metrics = {"base": metrics}
            result["metrics"] = metrics

        test_fingerprint = evaluation_fingerprint(dataset_dir)
        train_corpus = json.loads((dataset_dir / "global_transforms/transforms_train.json").read_text())
        result.setdefault("metrics_provenance", {})
        for name, reconstruction in reconstructions.items():
            if name.startswith("redundancy-") or reconstruction.get("status") == "redundant":
                continue
            if reconstruction.get("status") not in (None, "complete"):
                print(f"Skipping {name}: reconstruction is not complete")
                continue
            iteration = reconstruction_iteration(reconstruction, args.iteration)
            reconstruction_dir = Path(reconstruction["reconstruction_path"])
            if args.reconstructions_root:
                reconstruction_dir = (args.reconstructions_root.expanduser().resolve()
                                      / result["object_id"] / result["experiment_id"]
                                      / reconstruction_dir.name)
            checkpoint = (
                reconstruction_dir
                / "point_cloud"
                / f"iteration_{iteration}"
                / "point_cloud.ply"
            )
            if not checkpoint.is_file():
                print(
                    f"Skipping {result.get('object_id', result_path.parent.parent.name)}"
                    f"/{result.get('experiment_id', result_path.parent.name)}"
                    f"/{name}: can't find checkpoint {checkpoint}"
                )
                continue
            training_frames = [train_corpus["frames"][i] for i in reconstruction["train_lattice_indices"]]
            provenance = {"iteration": iteration, "checkpoint_sha256": sha256(checkpoint),
                          "test_data_sha256": test_fingerprint,
                          "training_cameras_sha256": hashlib.sha256(
                              json.dumps({"camera_angle_x": train_corpus["camera_angle_x"],
                                          "frames": training_frames}, sort_keys=True).encode()).hexdigest(),
                          "evaluator_sha256": sha256(Path(__file__))}
            cache_key = (str(reconstruction_dir.resolve()), json.dumps(provenance, sort_keys=True))
            if (not args.redo and has_metrics_for_iteration(result, metrics, name, iteration)
                    and result["metrics_provenance"].get(name) == provenance):
                print(f"Skipping {result.get('object_id')}/{result.get('experiment_id')}/{name}: metrics already complete")
                evaluation_cache[cache_key] = metrics[name]
                continue
            if cache_key not in evaluation_cache:
                evaluation_cache[cache_key] = evaluate_reconstruction(
                    reconstruction_dir, dataset_dir, reconstruction["train_lattice_indices"],
                    iteration, fastgs_dir=args.fastgs_dir, python=args.python)
            metrics[name] = copy.deepcopy(evaluation_cache[cache_key])
            result["metrics_provenance"][name] = provenance
            result.setdefault("metrics_iterations", {})[name] = iteration
            # Keep the legacy scalar only when every evaluated budget agrees.
            iterations = set(result["metrics_iterations"].values())
            if len(iterations) == 1:
                result["metrics_iteration"] = iteration
            else:
                result.pop("metrics_iteration", None)
            save_result(result_path, result)

        copy_redundancy_metrics(result, metrics, reconstructions, args.iteration)
        save_result(result_path, result)


if __name__ == "__main__":
    main()
