"""Render shared views and train scratch or base-initialized FastGS experiments."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from gaussian_initialization import prepare_initialization, validate_options

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
ADAPTER_SCRIPT = ROOT / "fastgs_adapter.py"


def load_experiment(path):
    with path.open(encoding="utf-8") as file:
        experiment = json.load(file)
    experiment.setdefault("id", path.stem)
    return experiment


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_manifest(path, camera_angle_x, frames):
    save_json(path, {"camera_angle_x": camera_angle_x, "frames": frames})


def make_dataset(object_dir, experiment, lattice):
    global_dir = object_dir / "global_transforms"
    global_train = load_json(global_dir / "transforms_train.json")
    metadata_path = object_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    trajectory_elevation = float(metadata.get("trajectory_elevation_deg", 20.0))
    config = experiment["train"]
    indices = select_indices(
        config["strategy"], int(config["count"]), lattice,
        seed=int(config["seed"]), trajectory_elevation_deg=trajectory_elevation,
    )
    return indices, [global_train["frames"][i]["file_path"] for i in indices], trajectory_elevation


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_training_manifest(object_dir, global_train, train_indices):
    write_manifest(object_dir / "transforms_train.json", global_train["camera_angle_x"],
                   [dict(global_train["frames"][i]) for i in train_indices])


def point_cloud_path(path, iterations):
    return path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"


def reconstruction_complete(path, iterations=30000, spec=None):
    if not point_cloud_path(path, iterations).is_file():
        return False
    if spec is None:
        return True
    receipt = path / "training_run.json"
    if receipt.is_file():
        run = load_json(receipt)
        return run.get("status") == "complete" and run.get("spec") == spec
    # Existing pipeline outputs have no receipt. Only accept them as legacy
    # 30k scratch runs, never as evidence that a warm-start run finished.
    return spec["initialization"] == "scratch" and iterations == 30000


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_spec(args, global_train, indices, iterations, initial_sha256=None):
    frames = [global_train["frames"][i] for i in indices]
    manifest = {"camera_angle_x": global_train["camera_angle_x"], "frames": frames}
    spec = {
        "version": 3,
        "initialization": "base" if initial_sha256 else "scratch",
        "iterations": iterations,
        "train_lattice_indices": indices,
        "train_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }
    if getattr(args, "fastgs_sha256", None):
        spec["fastgs_sha256"] = args.fastgs_sha256
    if getattr(args, "data_fingerprint", None):
        spec["input_data_sha256"] = args.data_fingerprint
    if initial_sha256:
        spec.update({
            "adapter_sha256": hashlib.sha256(
                ADAPTER_SCRIPT.read_bytes() + (ROOT / "fastgs_schedule.py").read_bytes()).hexdigest(),
            "initialization_sha256": initial_sha256,
            "adaptation_lr_scale": args.adaptation_lr_scale,
            "adaptation_densify_from_iter": args.adaptation_densify_from_iter,
            "adaptation_densify_until_iter": args.adaptation_densify_until_iter,
            "adaptation_densification_interval": args.adaptation_densification_interval,
        })
    return spec


def training_command(args, object_dir, reconstruction_dir, iterations, initial_ply=None):
    command = [
        args.python, str(ADAPTER_SCRIPT if initial_ply else args.fastgs_dir / "train.py"),
        "--source_path", str(object_dir), "--model_path", str(reconstruction_dir),
        "--eval", "--iterations", str(iterations),
        "--densification_interval", "500", "--optimizer_type", "default",
        "--save_iterations", str(iterations), "--grad_abs_thresh", "0.0012",
        # Evaluation is separate by default, avoiding duplicate LPIPS/rendering.
        "--test_iterations",
    ]
    if args.evaluate_during_training:
        command.append(str(iterations))
    elif not initial_ply:
        # Unmodified FastGS requires at least one value here. No positive
        # iteration matches this sentinel, so its evaluation remains disabled.
        command.append("-1")
    if initial_ply:
        command.extend([
            "--fastgs-dir", str(args.fastgs_dir),
            "--init_ply", str(initial_ply),
            "--adaptation_lr_scale", str(args.adaptation_lr_scale),
            "--adaptation_densify_from_iter", str(args.adaptation_densify_from_iter),
            "--adaptation_densify_until_iter", str(args.adaptation_densify_until_iter),
            "--adaptation_densification_interval", str(args.adaptation_densification_interval),
        ])
    return command


def train_reconstruction(args, object_dir, reconstruction_dir, global_train, indices,
                         spec, initial_ply=None, force=False):
    command = training_command(args, object_dir, reconstruction_dir, spec["iterations"], initial_ply)
    if not force and reconstruction_complete(reconstruction_dir, spec["iterations"], spec):
        print(f"Skipping complete reconstruction: {reconstruction_dir}", flush=True)
        return
    if args.dry_run:
        if not hasattr(args, "planned_reconstructions"):
            args.planned_reconstructions = set()
        key = (str(reconstruction_dir.resolve()), json.dumps(spec, sort_keys=True))
        if key in args.planned_reconstructions:
            return
        args.planned_reconstructions.add(key)
        print(json.dumps({"command": command, "training": spec}, indent=2))
        return
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = reconstruction_dir / "training_run.json"
    receipt = {"status": "running", "spec": spec, "command": command,
               "environment": getattr(args, "runtime_environment", None),
               "timing_definition": "subprocess wall time including startup, loading, training and saving; excludes evaluation"}
    save_json(receipt_path, receipt)
    manifest_path = object_dir / "transforms_train.json"
    old_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
    test_manifest_path = object_dir / "transforms_test.json"
    old_test_manifest = test_manifest_path.read_bytes() if test_manifest_path.is_file() else None
    start = time.perf_counter()
    final_ply = point_cloud_path(reconstruction_dir, spec["iterations"])
    before = final_ply.stat().st_mtime_ns if final_ply.is_file() else None
    try:
        write_training_manifest(object_dir, global_train, indices)
        test_manifest_path.write_bytes((object_dir / "global_transforms/transforms_test.json").read_bytes())
        start = time.perf_counter()
        subprocess.run(command, cwd=args.fastgs_dir, check=True)
        if not final_ply.is_file() or final_ply.stat().st_mtime_ns == before:
            raise RuntimeError(f"FastGS exited without its expected final PLY: {reconstruction_dir}")
        receipt["status"] = "complete"
    except Exception:
        receipt["status"] = "failed"
        raise
    finally:
        receipt["wall_time_seconds"] = time.perf_counter() - start
        save_json(receipt_path, receipt)
        for path, previous in ((manifest_path, old_manifest), (test_manifest_path, old_test_manifest)):
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", action="append")
    parser.add_argument("--experiment-id", action="append", help="Limit to one or more experiment IDs.")
    parser.add_argument("--candidate-limit", type=int, help="Limit non-redundant candidates per context for smoke tests.")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-base-retrain", action="store_true",
                        help="Also rebuild the shared scratch initializer in base mode.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    parser.add_argument("--python", default=sys.executable, help="Python executable in the GPU FastGS environment.")
    parser.add_argument("--views-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--objects-root", type=Path, default=OBJECTS_ROOT)
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--reconstructions-root", type=Path, default=RECONSTRUCTIONS_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--base-reconstructions-root", type=Path,
                        help="Existing scratch reconstructions, arranged as OBJECT/EXPERIMENT/base/.")
    parser.add_argument("--shared-base", action="store_true", help="Use the shared base root for scratch baselines too.")
    parser.add_argument("--gain-reference", choices=("original-base", "equal-budget-base-control"),
                        default="original-base")
    parser.add_argument("--base-control", action="store_true", help="Also adapt base views alone as a separate control.")
    parser.add_argument("--extra-gaussians", type=int, default=0)
    parser.add_argument("--extra-gaussian-seed", type=int, default=77)
    parser.add_argument("--extra-gaussian-jitter", type=float, default=0.5)
    parser.add_argument("--extra-gaussian-opacity", type=float, default=0.02)
    parser.add_argument("--extra-gaussian-scale", type=float, default=0.5)
    parser.add_argument("--initializations-root", type=Path,
                        help="Cache for derived PLYs; defaults beside reconstruction run directories.")
    parser.add_argument("--run-name", help="Separate output namespace; defaults to base_init_<budget> in base mode.")
    parser.add_argument("--candidate-initialization", choices=("scratch", "base"), default="scratch")
    parser.add_argument("--base-iterations", type=int, default=30000)
    parser.add_argument("--candidate-iterations", type=int,
                        help="Defaults to 3000 in base mode, otherwise base-iterations.")
    parser.add_argument("--adaptation-lr-scale", type=float, default=0.1)
    parser.add_argument("--adaptation-densify-from-iter", type=int, default=100)
    parser.add_argument("--adaptation-densify-until-iter", type=int,
                        help="Exclusive end, defaults to half the candidate budget; 0 disables densification.")
    parser.add_argument("--adaptation-densification-interval", type=int, default=250)
    parser.add_argument("--evaluate-during-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print pending commands without writing or training.")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_options(args.extra_gaussians, args.extra_gaussian_jitter,
                         args.extra_gaussian_opacity, args.extra_gaussian_scale)
    except ValueError as error:
        parser.error(str(error))
    if args.extra_gaussian_seed < 0:
        parser.error("extra-gaussian-seed must be nonnegative")
    if args.extra_gaussians and args.candidate_initialization != "base":
        parser.error("extra-gaussians requires candidate-initialization=base")
    if args.gain_reference == "equal-budget-base-control" and args.candidate_initialization != "base":
        parser.error("equal-budget-base-control requires candidate-initialization=base")
    if args.candidate_iterations is None:
        args.candidate_iterations = 3000 if args.candidate_initialization == "base" else args.base_iterations
    if args.base_iterations < 1 or args.candidate_iterations < 1:
        parser.error("iteration budgets must be positive")
    if args.candidate_limit is not None and args.candidate_limit < 0:
        parser.error("candidate-limit must be nonnegative")
    if not math.isfinite(args.adaptation_lr_scale) or args.adaptation_lr_scale <= 0:
        parser.error("adaptation-lr-scale must be finite and positive")
    if args.adaptation_densification_interval < 1 or args.adaptation_densify_from_iter < 0:
        parser.error("invalid adaptation densification interval/start")
    if args.adaptation_densify_until_iter is None:
        args.adaptation_densify_until_iter = args.candidate_iterations // 2
        if args.adaptation_densify_until_iter <= args.adaptation_densify_from_iter:
            args.adaptation_densify_until_iter = 0
    end = args.adaptation_densify_until_iter
    if end < 0 or end > args.candidate_iterations or (end and end <= args.adaptation_densify_from_iter):
        parser.error("adaptation densification end must be 0 or between start and candidate-iterations")
    if args.candidate_initialization == "base" and args.run_name is None:
        args.run_name = f"base_init_{args.candidate_iterations}" + (f"_extra_{args.extra_gaussians}" if args.extra_gaussians else "")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        parser.error("run-name must be a single directory name")
    if args.candidate_initialization == "base" and not args.run_name:
        parser.error("base initialization requires a nonempty run-name")
    for name in ("fastgs_dir", "views_root", "objects_root", "experiments_dir",
                 "reconstructions_root", "results_root", "base_reconstructions_root", "initializations_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.base_reconstructions_root = args.base_reconstructions_root or args.reconstructions_root
    args.initializations_root = args.initializations_root or args.reconstructions_root / "_initializations"
    if args.run_name:
        args.reconstructions_root /= args.run_name
        args.results_root /= args.run_name
    return args


def make_reconstruction(path, indices, global_train, iterations):
    return {"reconstruction_path": str(path), "train_lattice_indices": indices,
            "train_image_paths": [global_train["frames"][i]["file_path"] for i in indices],
            "iterations": iterations, "status": "pending"}


def process_experiment(args, object_id, object_dir, global_train, lattice, experiment):
    train_indices, train_paths, elevation = make_dataset(object_dir, experiment, lattice)
    experiment_id = experiment["id"]
    destination = args.reconstructions_root / object_id / experiment_id
    result_path = args.results_root / object_id / experiment_id / "results.json"
    warm = args.candidate_initialization == "base"
    initial_ply = None
    initial_sha = None
    initialization = None
    original_base = None
    original_sha = None
    initial_spec = training_spec(args, global_train, train_indices, args.base_iterations)
    if warm or args.shared_base:
        original_base = args.base_reconstructions_root / object_id / experiment_id / "base"
        if warm and original_base == destination / "base":
            raise ValueError("Shared initializer and run directory must have different output paths")
        train_reconstruction(args, object_dir, original_base, global_train, train_indices,
                             initial_spec, force=args.force_base_retrain)
        original_ply = point_cloud_path(original_base, args.base_iterations)
        original_sha = file_sha256(original_ply) if original_ply.is_file() else "pending-dry-run"
        if warm:
            initial_ply = original_ply
            initial_sha = original_sha
            preparation = None
            if args.extra_gaussians:
                initial_ply, preparation = prepare_initialization(
                    original_ply, args.initializations_root / object_id / experiment_id,
                    args.extra_gaussians, args.extra_gaussian_seed, args.extra_gaussian_jitter,
                    args.extra_gaussian_opacity, args.extra_gaussian_scale, args.dry_run, args.python)
                initial_sha = preparation["ply_sha256"]
            initialization = {"reconstruction_path": str(original_base), "iteration": args.base_iterations,
                              "original_base_sha256": original_sha, "ply_path": str(initial_ply),
                              "ply_sha256": initial_sha, "preparation": preparation}

    paired_control = warm and args.gain_reference == "equal-budget-base-control"
    base_budget = args.candidate_iterations if paired_control else args.base_iterations
    base_path = destination / "base" if paired_control or original_base is None else original_base
    reconstructions = {"base": make_reconstruction(base_path, train_indices, global_train, base_budget)}
    if warm and args.base_control and not paired_control:
        reconstructions["base_control"] = make_reconstruction(
            destination / "base_control", train_indices, global_train, args.candidate_iterations)
    unique_candidates = {}
    candidate_count = 0
    for strategy, count in candidate_configurations(experiment.get("candidates")):
        candidates = select_candidate_indices(
            strategy, count, lattice, train_indices, seed=int(experiment["train"].get("seed", 0)),
            train_strategy=experiment["train"]["strategy"], trajectory_elevation_deg=elevation)
        for number, index in enumerate(candidates, 1):
            redundant = strategy == "redundancy"
            if not redundant and args.candidate_limit is not None and candidate_count >= args.candidate_limit:
                continue
            name = f"{strategy}-{count}-{number}"
            indices = train_indices if redundant else train_indices + [index]
            reconstruction = make_reconstruction(destination / name, indices, global_train, args.candidate_iterations)
            reconstruction["candidate_lattice_index"] = index
            if redundant:
                reconstruction.update({"reconstruction_path": str(base_path),
                                       "iterations": base_budget, "status": "redundant", "alias_of": "base"})
            else:
                candidate_count += 1
                if index in unique_candidates:
                    owner = unique_candidates[index]
                    reconstruction.update({"alias_of": owner,
                                           "reconstruction_path": reconstructions[owner]["reconstruction_path"]})
                else:
                    unique_candidates[index] = name
            reconstructions[name] = reconstruction

    record = {
        "object_id": object_id, "experiment_id": experiment_id,
        "dataset_path": str(object_dir), "reconstruction_path": str(base_path),
        "train_lattice_indices": train_indices, "train_image_paths": train_paths,
        "train_manifest": str(object_dir / "transforms_train.json"),
        "test_manifest": str(object_dir / "transforms_test.json"),
        "protocol": {"candidate_initialization": args.candidate_initialization,
                     "gain_reference": "equal_budget_base_control" if paired_control else "original_base",
                     "original_base_sha256": original_sha,
                     "extra_gaussians": args.extra_gaussians,
                     "input_data_sha256": getattr(args, "data_fingerprint", None),
                     "base_iterations": args.base_iterations, "candidate_iterations": args.candidate_iterations},
        "initialization": initialization,
        "reconstructions": reconstructions, "status": "pending", "metrics": {}, "metrics_iterations": {},
        "metrics_provenance": {},
    }
    previous = load_json(result_path) if result_path.is_file() else {}
    previous_metrics = previous.get("metrics", {})
    if "averages" in previous_metrics:
        previous_metrics = {"base": previous_metrics}
    for name, reconstruction in reconstructions.items():
        reconstruction["training"] = training_spec(
            args, global_train, reconstruction["train_lattice_indices"], reconstruction["iterations"],
            initial_sha if warm and (name != "base" or paired_control) and reconstruction.get("alias_of") != "base" else None)
        if reconstruction.get("alias_of"):
            reconstruction["training"] = reconstructions[reconstruction["alias_of"]]["training"]
    try:
        for name, reconstruction in reconstructions.items():
            alias = reconstruction.get("alias_of")
            path = Path(reconstruction["reconstruction_path"])
            spec = reconstruction["training"]
            if alias:
                if reconstruction["status"] != "redundant":
                    reconstruction["status"] = reconstructions[alias]["status"]
                if "timing" in reconstructions[alias]:
                    reconstruction["timing"] = reconstructions[alias]["timing"]
                continue
            complete = reconstruction_complete(path, reconstruction["iterations"], spec)
            old = previous.get("reconstructions", {}).get(name, {})
            same_training = old.get("training") == spec or (
                not old.get("training") and not warm and reconstruction["iterations"] == 30000
                and old.get("train_lattice_indices") == reconstruction["train_lattice_indices"])
            if complete and (not args.force_retrain or (name == "base" and original_base is not None and not paired_control)) and same_training:
                record["metrics"][name] = previous_metrics.get(name, {})
                if name in previous.get("metrics_provenance", {}):
                    record["metrics_provenance"][name] = previous["metrics_provenance"][name]
                old_iteration = previous.get("metrics_iterations", {}).get(name, previous.get("metrics_iteration"))
                if old_iteration is not None:
                    record["metrics_iterations"][name] = old_iteration
            reconstruction["status"] = "running"
            record["status"] = "running"
            if not args.dry_run:
                save_json(result_path, record)
            train_reconstruction(args, object_dir, path, global_train,
                                 reconstruction["train_lattice_indices"], spec,
                                 initial_ply=initial_ply if spec["initialization"] == "base" else None,
                                 force=args.force_retrain and not (name == "base" and original_base is not None and not paired_control))
            receipt = path / "training_run.json"
            if receipt.is_file():
                reconstruction["timing"] = load_json(receipt)
            if name == "base" and not paired_control:
                ply = point_cloud_path(path, reconstruction["iterations"])
                record["protocol"]["original_base_sha256"] = file_sha256(ply) if ply.is_file() else original_sha
            reconstruction["status"] = "complete" if not args.dry_run else "pending"
        record["status"] = "complete" if not args.dry_run else "pending"
    except Exception:
        record["status"] = "failed"
        for reconstruction in reconstructions.values():
            if reconstruction["status"] == "running":
                reconstruction["status"] = "failed"
        raise
    finally:
        if not args.dry_run:
            save_json(result_path, record)


def main(argv=None):
    args = parse_args(argv)
    print(f"Reconstructions: {args.reconstructions_root}\nResults: {args.results_root}", flush=True)
    if args.object_id:
        object_ids = args.object_id
    elif args.skip_render:
        object_ids = sorted(path.name for path in args.views_root.iterdir()
                            if (path / "global_transforms" / "transforms_train.json").is_file())
    else:
        object_ids = sorted(path.stem for path in args.objects_root.glob("*.glb"))
    if not object_ids:
        raise FileNotFoundError("No objects found; specify --object-id and check --views-root/--objects-root")
    experiments = [load_experiment(path) for path in sorted(args.experiments_dir.glob("*.json"))]
    if args.experiment_id:
        unknown = set(args.experiment_id) - {e["id"] for e in experiments}
        if unknown:
            raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
        experiments = [e for e in experiments if e["id"] in args.experiment_id]
    if not experiments:
        raise FileNotFoundError(f"No experiments found in {args.experiments_dir}")
    for object_id in object_ids:
        object_dir = args.views_root / object_id
        global_dir = object_dir / "global_transforms"
        if not args.skip_render and (args.rerender or not (global_dir / "transforms_train.json").is_file()
                                     or not (global_dir / "transforms_test.json").is_file()):
            if args.dry_run:
                raise ValueError("Render views before a dry run, or use --skip-render with an existing corpus")
            glb_path = args.objects_root / f"{object_id}.glb"
            if not glb_path.is_file():
                raise FileNotFoundError(glb_path)
            subprocess.run([sys.executable, str(ROOT / "renderer.py"), "--glb-path", str(glb_path),
                            "--object-id", object_id, "--output-root", str(args.views_root)], check=True)
        global_train = load_json(global_dir / "transforms_train.json")
        global_test = load_json(global_dir / "transforms_test.json")
        if not args.dry_run:
            write_manifest(object_dir / "transforms_test.json", global_test["camera_angle_x"], global_test["frames"])
        lattice = fibonacci_lattice(len(global_train["frames"]))
        for experiment in experiments:
            try:
                process_experiment(args, object_id, object_dir, global_train, lattice, experiment)
            except Exception as error:
                print(f"Failed {object_id}/{experiment['id']}: {error}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise


if __name__ == "__main__":
    main()
