"""Plan independent candidate reconstructions and execute resumable experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gaussian_initialization import prepare_initialization
from reconstruction_scripts.datasets import make_dataset
from reconstruction_scripts.io import file_sha256, load_json, save_json
from reconstruction_scripts.training import (
    point_cloud_path,
    reconstruction_complete,
    train_reconstruction,
    training_spec,
)
from sampling import candidate_configurations, select_candidate_indices


@dataclass
class BaseInitialization:
    original_path: Optional[Path] = None
    original_sha: Optional[str] = None
    ply: Optional[Path] = None
    ply_sha: Optional[str] = None
    metadata: Optional[dict] = None


def prepare_base(
    args, object_id, object_dir, global_train, train_indices, experiment_id
):
    """Train/reuse the shared base and optionally derive an augmented initializer."""
    base = BaseInitialization()
    warm = args.candidate_initialization == "base"
    if not (warm or args.shared_base):
        return base
    base.original_path = (
        args.base_reconstructions_root / object_id / experiment_id / "base"
    )
    destination = args.reconstructions_root / object_id / experiment_id / "base"
    if warm and base.original_path == destination:
        raise ValueError(
            "Shared initializer and run directory must have different output paths"
        )
    spec = training_spec(args, global_train, train_indices, args.base_iterations)
    train_reconstruction(
        args,
        object_dir,
        base.original_path,
        global_train,
        train_indices,
        spec,
        force=args.force_base_retrain,
    )
    original_ply = point_cloud_path(base.original_path, args.base_iterations)
    base.original_sha = (
        file_sha256(original_ply) if original_ply.is_file() else "pending-dry-run"
    )
    if not warm:
        return base
    base.ply, base.ply_sha = original_ply, base.original_sha
    preparation = None
    if args.extra_gaussians:
        base.ply, preparation = prepare_initialization(
            original_ply,
            args.initializations_root / object_id / experiment_id,
            args.extra_gaussians,
            args.extra_gaussian_seed,
            args.extra_gaussian_jitter,
            args.extra_gaussian_opacity,
            args.extra_gaussian_scale,
            args.dry_run,
            args.python,
        )
        base.ply_sha = preparation["ply_sha256"]
    base.metadata = {
        "reconstruction_path": str(base.original_path),
        "iteration": args.base_iterations,
        "original_base_sha256": base.original_sha,
        "ply_path": str(base.ply),
        "ply_sha256": base.ply_sha,
        "preparation": preparation,
    }
    return base


def make_reconstruction(path, indices, global_train, iterations):
    return {
        "reconstruction_path": str(path),
        "train_lattice_indices": indices,
        "train_image_paths": [global_train["frames"][i]["file_path"] for i in indices],
        "iterations": iterations,
        "status": "pending",
    }


def plan_reconstructions(
    args, destination, global_train, lattice, experiment, selection, base
):
    train_indices, _, elevation = selection
    warm = args.candidate_initialization == "base"
    paired_control = warm and args.gain_reference == "equal-budget-base-control"
    base_budget = args.candidate_iterations if paired_control else args.base_iterations
    base_path = (
        destination / "base"
        if paired_control or base.original_path is None
        else base.original_path
    )
    reconstructions = {
        "base": make_reconstruction(base_path, train_indices, global_train, base_budget)
    }
    if warm and args.base_control and not paired_control:
        reconstructions["base_control"] = make_reconstruction(
            destination / "base_control",
            train_indices,
            global_train,
            args.candidate_iterations,
        )
    owners = {}
    for strategy, count in candidate_configurations(experiment.get("candidates")):
        candidates = select_candidate_indices(
            strategy,
            count,
            lattice,
            train_indices,
            seed=int(experiment["train"].get("seed", 0)),
            train_strategy=experiment["train"]["strategy"],
            trajectory_elevation_deg=elevation,
        )
        for number, index in enumerate(candidates, 1):
            redundant = strategy == "redundancy"
            if (
                not redundant
                and index not in owners
                and args.candidate_limit is not None
                and len(owners) >= args.candidate_limit
            ):
                continue
            name = f"{strategy}-{count}-{number}"
            indices = train_indices if redundant else train_indices + [index]
            reconstruction = make_reconstruction(
                destination / name, indices, global_train, args.candidate_iterations
            )
            reconstruction["candidate_lattice_index"] = index
            if redundant:
                reconstruction.update(
                    {
                        "reconstruction_path": str(base_path),
                        "iterations": base_budget,
                        "status": "redundant",
                        "alias_of": "base",
                    }
                )
            elif index in owners:
                owner = owners[index]
                reconstruction.update(
                    {
                        "alias_of": owner,
                        "reconstruction_path": reconstructions[owner][
                            "reconstruction_path"
                        ],
                    }
                )
            else:
                owners[index] = name
            reconstructions[name] = reconstruction
    for name, reconstruction in reconstructions.items():
        alias = reconstruction.get("alias_of")
        if alias:
            reconstruction["training"] = reconstructions[alias]["training"]
            continue
        initial_sha = (
            base.ply_sha if warm and (name != "base" or paired_control) else None
        )
        reconstruction["training"] = training_spec(
            args,
            global_train,
            reconstruction["train_lattice_indices"],
            reconstruction["iterations"],
            initial_sha,
        )
    return reconstructions


def preserve_metrics(record, previous, name, reconstruction, warm):
    """Keep cached metrics only when the recorded training specification matches."""
    old = previous.get("reconstructions", {}).get(name, {})
    same_training = old.get("training") == reconstruction["training"] or (
        not old.get("training")
        and not warm
        and reconstruction["iterations"] == 30000
        and old.get("train_lattice_indices") == reconstruction["train_lattice_indices"]
    )
    if not same_training:
        return
    metrics = previous.get("metrics", {})
    if "averages" in metrics:
        metrics = {"base": metrics}
    record["metrics"][name] = metrics.get(name, {})
    if name in previous.get("metrics_provenance", {}):
        record["metrics_provenance"][name] = previous["metrics_provenance"][name]
    iteration = previous.get("metrics_iterations", {}).get(
        name, previous.get("metrics_iteration")
    )
    if iteration is not None:
        record["metrics_iterations"][name] = iteration


def execute_reconstructions(args, object_dir, global_train, result_path, record, base):
    previous = load_json(result_path) if result_path.is_file() else {}
    warm = args.candidate_initialization == "base"
    paired_control = warm and args.gain_reference == "equal-budget-base-control"
    reconstructions = record["reconstructions"]
    try:
        for name, reconstruction in reconstructions.items():
            alias = reconstruction.get("alias_of")
            if alias:
                owner = reconstructions[alias]
                if reconstruction["status"] != "redundant":
                    reconstruction["status"] = owner["status"]
                if "timing" in owner:
                    reconstruction["timing"] = owner["timing"]
                continue
            path = Path(reconstruction["reconstruction_path"])
            spec = reconstruction["training"]
            shared_base = (
                name == "base" and base.original_path is not None and not paired_control
            )
            force = args.force_retrain and not shared_base
            if (
                reconstruction_complete(path, reconstruction["iterations"], spec)
                and not force
            ):
                preserve_metrics(record, previous, name, reconstruction, warm)
            reconstruction["status"] = record["status"] = "running"
            if not args.dry_run:
                save_json(result_path, record)
            train_reconstruction(
                args,
                object_dir,
                path,
                global_train,
                reconstruction["train_lattice_indices"],
                spec,
                initial_ply=base.ply if spec["initialization"] == "base" else None,
                force=force,
            )
            receipt = path / "training_run.json"
            if receipt.is_file():
                reconstruction["timing"] = load_json(receipt)
            if name == "base" and not paired_control:
                ply = point_cloud_path(path, reconstruction["iterations"])
                record["protocol"]["original_base_sha256"] = (
                    file_sha256(ply) if ply.is_file() else base.original_sha
                )
            reconstruction["status"] = "pending" if args.dry_run else "complete"
        record["status"] = "pending" if args.dry_run else "complete"
    except BaseException:
        record["status"] = "failed"
        for reconstruction in reconstructions.values():
            if reconstruction["status"] == "running":
                reconstruction["status"] = "failed"
        raise
    finally:
        if not args.dry_run:
            save_json(result_path, record)


def process_experiment(args, object_id, object_dir, global_train, lattice, experiment):
    selection = make_dataset(object_dir, global_train, experiment, lattice)
    train_indices, train_paths, _ = selection
    experiment_id = experiment["id"]
    destination = args.reconstructions_root / object_id / experiment_id
    result_path = args.results_root / object_id / experiment_id / "results.json"
    base = prepare_base(
        args, object_id, object_dir, global_train, train_indices, experiment_id
    )
    reconstructions = plan_reconstructions(
        args, destination, global_train, lattice, experiment, selection, base
    )
    paired_control = (
        args.candidate_initialization == "base"
        and args.gain_reference == "equal-budget-base-control"
    )
    record = {
        "object_id": object_id,
        "experiment_id": experiment_id,
        "dataset_path": str(object_dir),
        "reconstruction_path": reconstructions["base"]["reconstruction_path"],
        "train_lattice_indices": train_indices,
        "train_image_paths": train_paths,
        "train_manifest": str(object_dir / "transforms_train.json"),
        "test_manifest": str(object_dir / "transforms_test.json"),
        "protocol": {
            "candidate_initialization": args.candidate_initialization,
            "gain_reference": (
                "equal_budget_base_control" if paired_control else "original_base"
            ),
            "original_base_sha256": base.original_sha,
            "extra_gaussians": args.extra_gaussians,
            "input_data_sha256": getattr(args, "data_fingerprint", None),
            "base_iterations": args.base_iterations,
            "candidate_iterations": args.candidate_iterations,
        },
        "initialization": base.metadata,
        "reconstructions": reconstructions,
        "status": "pending",
        "metrics": {},
        "metrics_iterations": {},
        "metrics_provenance": {},
    }
    execute_reconstructions(args, object_dir, global_train, result_path, record, base)
    return record
