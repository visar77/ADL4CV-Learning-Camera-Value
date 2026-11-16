"""Compare fresh candidate training with 3k-step adaptation from a shared base."""

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

RENDERER_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = Path(__file__).resolve().parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_CONFIG = BENCHMARKS_ROOT / f"{SCRIPT_STEM}.json"
OUTPUT_ROOT = BENCHMARKS_ROOT / SCRIPT_STEM

for path in (BENCHMARKS_ROOT, RENDERER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_reconstructions as comparison
import evaluate_metrics as evaluation
import orchestrator
import training_fastgs
from gaussian_initialization import sha256
from sampling import candidate_configurations, fibonacci_lattice


REFERENCE = "fresh"
ADAPTED = "base-cloud-3k"


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Benchmark config must be a schema_version 1 JSON object")
    allowed = {
        "schema_version",
        "base_iterations",
        "fresh_candidate_iterations",
        "adapted_candidate_iterations",
        "adaptation",
        "candidates",
        "objects",
        "experiments",
    }
    if config.keys() - allowed:
        raise ValueError(f"Unknown benchmark config keys: {sorted(config.keys() - allowed)}")
    for key in (
        "base_iterations",
        "fresh_candidate_iterations",
        "adapted_candidate_iterations",
    ):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config.get("adaptation", {}), dict):
        raise ValueError("adaptation must be a JSON object")
    adaptation = config.get("adaptation", {})
    adaptation_options = {
        "lr_scale",
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
    }
    if adaptation.keys() - adaptation_options:
        raise ValueError(
            f"Unknown adaptation settings: {sorted(adaptation.keys() - adaptation_options)}"
        )
    if "lr_scale" in adaptation and (
        not isinstance(adaptation["lr_scale"], (int, float))
        or isinstance(adaptation["lr_scale"], bool)
        or not math.isfinite(adaptation["lr_scale"])
        or adaptation["lr_scale"] <= 0
    ):
        raise ValueError("adaptation.lr_scale must be finite and positive")
    for key in ("densify_from_iter", "densify_until_iter", "densification_interval"):
        value = adaptation.get(key)
        if key == "densify_until_iter" and value is None:
            continue
        if key in adaptation and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"adaptation.{key} must be a nonnegative integer")
    if adaptation.get("densification_interval", 1) == 0:
        raise ValueError("adaptation.densification_interval must be positive")
    if adaptation.get("densify_until_iter") is not None:
        densify_until = adaptation["densify_until_iter"]
        densify_from = adaptation.get("densify_from_iter", 100)
        if densify_until > config["adapted_candidate_iterations"]:
            raise ValueError("adaptation.densify_until_iter exceeds the adaptation budget")
        if densify_until and densify_from >= densify_until:
            raise ValueError(
                "adaptation.densify_from_iter must be less than densify_until_iter"
            )
    candidates = candidate_configurations(config.get("candidates"))
    if not candidates or any(strategy == "redundancy" for strategy, _ in candidates):
        raise ValueError("candidates must contain non-redundant candidate strategies")
    for key in ("objects", "experiments"):
        values = config.get(key)
        if values is not None and (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError(f"{key} must be a nonempty list of IDs when specified")
    return config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--views-root", type=Path, default=orchestrator.CONFIG["views_root"])
    parser.add_argument(
        "--experiments-dir", type=Path, default=orchestrator.CONFIG["experiments_dir"]
    )
    parser.add_argument("--fastgs-dir", type=Path, default=training_fastgs.CONFIG["fastgs_dir"])
    parser.add_argument("--python", default=training_fastgs.CONFIG["python"])
    parser.add_argument("--object-id", "--object-ids", action="append", dest="object_ids")
    parser.add_argument(
        "--experiment-id", "--experiment-ids", action="append", dest="experiment_ids"
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _selected_ids(requested, configured, available, label):
    values = requested if requested is not None else configured
    selected = list(dict.fromkeys(values if values is not None else available))
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown {label} IDs: {sorted(unknown)}")
    return selected


def _image_path(object_dir, frame):
    image = object_dir / frame["file_path"]
    return image if image.suffix else image.with_suffix(".png")


def input_fingerprint(object_dir, train, test):
    digest = hashlib.sha256(
        json.dumps({"train": train, "test": test}, sort_keys=True).encode()
    )
    paths = sorted(
        {
            _image_path(object_dir, frame)
            for manifest in (train, test)
            for frame in manifest["frames"]
        }
    )
    for image in paths:
        if not image.is_file():
            raise FileNotFoundError(f"Benchmark input image is missing: {image}")
        digest.update(image.relative_to(object_dir).as_posix().encode())
        digest.update(bytes.fromhex(sha256(image)))
    return digest.hexdigest()


def _load_contexts(args, config):
    views_root = args.views_root.expanduser().resolve()
    experiments_dir = args.experiments_dir.expanduser().resolve()
    view_objects = sorted(
        path.name
        for path in views_root.iterdir()
        if path.is_dir()
        and (path / "global_transforms/transforms_train.json").is_file()
        and (path / "global_transforms/transforms_test.json").is_file()
    ) if views_root.is_dir() else []
    object_ids = _selected_ids(args.object_ids, config.get("objects"), view_objects, "object")

    previous_experiments = orchestrator.CONFIG["experiments_dir"]
    orchestrator.CONFIG["experiments_dir"] = experiments_dir
    try:
        all_experiments = orchestrator.load_experiments()
    finally:
        orchestrator.CONFIG["experiments_dir"] = previous_experiments
    available_experiments = [item["id"] for item in all_experiments]
    experiment_ids = _selected_ids(
        args.experiment_ids, config.get("experiments"), available_experiments, "experiment"
    )
    by_id = {item["id"]: item for item in all_experiments}
    contexts = []
    for object_id in object_ids:
        if not object_id or Path(object_id).name != object_id:
            raise ValueError(f"Invalid object ID: {object_id!r}")
        object_dir = views_root / object_id
        train = orchestrator.load_json(
            object_dir / "global_transforms/transforms_train.json"
        )
        test = orchestrator.load_json(
            object_dir / "global_transforms/transforms_test.json"
        )
        metadata_path = object_dir / "metadata.json"
        metadata = orchestrator.load_json(metadata_path) if metadata_path.is_file() else {}
        if int(metadata.get("lattice_size", len(train["frames"]))) != len(train["frames"]):
            raise ValueError(f"Lattice metadata does not match train frames for {object_id}")
        fingerprint = input_fingerprint(object_dir, train, test)
        for experiment_id in experiment_ids:
            experiment = copy.deepcopy(by_id[experiment_id])
            experiment["candidates"] = copy.deepcopy(config["candidates"])
            experiment_sha256 = hashlib.sha256(
                json.dumps(
                    {
                        "experiment": experiment,
                        "trajectory_elevation_deg": metadata.get(
                            "trajectory_elevation_deg", 20.0
                        ),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            contexts.append(
                {
                    "object_id": object_id,
                    "object_dir": object_dir,
                    "train": train,
                    "experiment": experiment,
                    "trajectory_elevation": float(
                        metadata.get("trajectory_elevation_deg", 20.0)
                    ),
                    "input_data_sha256": fingerprint,
                    "experiment_sha256": experiment_sha256,
                }
            )
    if not contexts:
        raise ValueError("No matching object/experiment contexts were found")
    return contexts


def _train_model(
    args,
    object_dir,
    global_train,
    train_indices,
    model_dir,
    iterations,
    init_ply=None,
    adaptation=None,
    retrain=False,
):
    expected = orchestrator.checkpoint_path(model_dir, iterations)
    if expected.is_file() and not retrain:
        receipt_path = model_dir / "training_run.json"
        receipt = orchestrator.load_json(receipt_path) if receipt_path.is_file() else None
        return expected, receipt

    model_root = model_dir.parents[2]
    resolved_model = model_dir.resolve()
    resolved_root = model_root.resolve()
    if resolved_model == resolved_root or resolved_root not in resolved_model.parents:
        raise ValueError(f"Refusing to replace model output outside {resolved_root}: {model_dir}")
    if model_dir.is_symlink():
        raise ValueError(f"Refusing to replace symlink model output: {model_dir}")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(training_fastgs.ROOT / "training_fastgs.py"),
        "--source-path",
        str(object_dir),
        "--model-path",
        str(model_dir),
        "--iterations",
        str(iterations),
        "--fastgs-dir",
        str(args.fastgs_dir),
        "--python",
        args.python,
    ]
    if init_ply is not None:
        command.extend(["--init-ply", str(init_ply)])
        settings = adaptation or {}
        for key, option, default_key in (
            ("lr_scale", "--adaptation-lr-scale", "adaptation_lr_scale"),
            ("densify_from_iter", "--adaptation-densify-from-iter", "adaptation_densify_from_iter"),
            ("densification_interval", "--adaptation-densification-interval", "adaptation_densification_interval"),
        ):
            value = settings.get(key, training_fastgs.CONFIG[default_key])
            command.extend([option, str(value)])
        until = settings.get(
            "densify_until_iter",
            training_fastgs.CONFIG.get("adaptation_densify_until_iter"),
        )
        if until is not None:
            command.extend(["--adaptation-densify-until-iter", str(until)])

    global_test = orchestrator.load_json(
        object_dir / "global_transforms/transforms_test.json"
    )
    with orchestrator.training_manifests(
        object_dir, global_train, global_test, train_indices
    ):
        start = time.perf_counter()
        subprocess.run(command, check=True)
    if not expected.is_file():
        raise RuntimeError(f"FastGS did not write the expected checkpoint: {expected}")
    receipt = {
        "status": "complete",
        "wall_time_seconds": time.perf_counter() - start,
    }
    orchestrator.save_json(model_dir / "training_run.json", receipt)
    return expected, receipt


def _load_result(path, context, base_indices, base_sha256, retrain):
    protocol = {
        "gain_reference": "original_base",
        "original_base_sha256": base_sha256,
        "input_data_sha256": context["input_data_sha256"],
        "experiment_sha256": context["experiment_sha256"],
    }
    if path.is_file() and not retrain:
        result = orchestrator.load_json(path)
        if result.get("protocol") != protocol:
            raise ValueError(
                f"Saved benchmark inputs changed for "
                f"{context['object_id']}/{context['experiment']['id']}; "
                "use --retrain or a clean output directory"
            )
    else:
        result = {
            "object_id": context["object_id"],
            "experiment_id": context["experiment"]["id"],
            "dataset_path": str(context["object_dir"]),
            "train_lattice_indices": base_indices,
            "reconstructions": {},
            "protocol": protocol,
        }
    result["train_lattice_indices"] = base_indices
    result["protocol"] = protocol
    return result


def _base_indices(context):
    metadata_path = context["object_dir"] / "metadata.json"
    metadata = orchestrator.load_json(metadata_path) if metadata_path.is_file() else {}
    lattice = fibonacci_lattice(len(context["train"]["frames"]))
    train = context["experiment"]["train"]
    return orchestrator.select_indices(
        train["strategy"],
        int(train["count"]),
        lattice,
        seed=int(train.get("seed", 0)),
        trajectory_elevation_deg=context["trajectory_elevation"],
    )


def _run_context(args, config, context):
    object_id = context["object_id"]
    experiment = context["experiment"]
    base_indices = _base_indices(context)
    key = (object_id, experiment["id"])
    method_records = {}
    method_roots = {
        REFERENCE: OUTPUT_ROOT / "methods" / REFERENCE,
        ADAPTED: OUTPUT_ROOT / "methods" / ADAPTED,
    }
    base_model_dir = (
        method_roots[REFERENCE] / "reconstructions"
        / object_id / experiment["id"] / "base"
    )
    base_ply, base_timing = _train_model(
        args,
        context["object_dir"],
        context["train"],
        base_indices,
        base_model_dir,
        config["base_iterations"],
        retrain=args.retrain,
    )
    base_sha256 = sha256(base_ply)
    for method, method_root in method_roots.items():
        reconstruction_root = method_root / "reconstructions"
        result_path = (
            method_root / "results" / object_id / experiment["id"] / "results.json"
        )
        result = _load_result(
            result_path, context, base_indices, base_sha256, args.retrain
        )
        result["reconstructions"]["base"] = {
            "status": "complete",
            "iterations": config["base_iterations"],
            "reconstruction_path": str(base_model_dir),
            "checkpoint_path": str(base_ply),
            "train_lattice_indices": base_indices,
            "timing": base_timing
            or result["reconstructions"].get("base", {}).get("timing"),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        orchestrator.save_json(result_path, result)
        method_records[method] = (result_path, result, reconstruction_root)

    candidate_jobs = orchestrator.make_candidate_jobs(
        experiment,
        fibonacci_lattice(len(context["train"]["frames"])),
        base_indices,
        context["trajectory_elevation"],
    )
    base_ply = method_records[REFERENCE][1]["reconstructions"]["base"]["checkpoint_path"]
    for candidate in candidate_jobs:
        candidate_id = candidate["id"]
        for method, method_root in method_roots.items():
            result_path, result, reconstruction_root = method_records[method]
            model_dir = reconstruction_root / object_id / experiment["id"] / candidate_id
            init_ply = Path(base_ply) if method == ADAPTED else None
            iterations = (
                config["adapted_candidate_iterations"]
                if method == ADAPTED
                else config["fresh_candidate_iterations"]
            )
            checkpoint, timing = _train_model(
                args,
                context["object_dir"],
                context["train"],
                candidate["train_indices"],
                model_dir,
                iterations,
                init_ply=init_ply,
                adaptation=config.get("adaptation") if init_ply is not None else None,
                retrain=args.retrain,
            )
            previous = result["reconstructions"].get(candidate_id, {})
            result["reconstructions"][candidate_id] = {
                "status": "complete",
                "iterations": iterations,
                "reconstruction_path": str(model_dir),
                "checkpoint_path": str(checkpoint),
                "train_lattice_indices": candidate["train_indices"],
                "timing": timing or previous.get("timing"),
            }
            orchestrator.save_json(result_path, result)
    return key


def _environment(args):
    return {
        "fastgs_dir": str(args.fastgs_dir.expanduser().resolve()),
        "python": args.python,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve()
    args.views_root = args.views_root.expanduser().resolve()
    args.experiments_dir = args.experiments_dir.expanduser().resolve()
    args.fastgs_dir = args.fastgs_dir.expanduser().resolve()
    try:
        config = load_config(args.config)
        contexts = _load_contexts(args, config)
        plan = {
            "configuration": config,
            "contexts": [
                {
                    "object_id": context["object_id"],
                    "experiment_id": context["experiment"]["id"],
                    "candidate_count": len(
                        orchestrator.make_candidate_jobs(
                            context["experiment"],
                            fibonacci_lattice(len(context["train"]["frames"])),
                            _base_indices(context),
                            context["trajectory_elevation"],
                        )
                    ),
                }
                for context in contexts
            ],
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return plan
        train_script = args.fastgs_dir / "train.py"
        if not train_script.is_file():
            raise FileNotFoundError(f"FastGS train.py not found: {train_script}")
        snapshot_path = OUTPUT_ROOT / "benchmark_config.json"
        if snapshot_path.is_file():
            previous = orchestrator.load_json(snapshot_path)
            if previous != config:
                raise ValueError(
                    f"Configuration differs from the saved run in {OUTPUT_ROOT}; "
                    "use a new script/config name for different settings"
                )
        orchestrator.save_json(snapshot_path, config)
        orchestrator.save_json(
            OUTPUT_ROOT / "benchmark_manifest.json",
            {**plan, "environment": _environment(args)},
        )
        context_keys = set()
        for context in contexts:
            context_keys.add(_run_context(args, config, context))

        for method in (REFERENCE, ADAPTED):
            method_root = OUTPUT_ROOT / "methods" / method
            evaluation.main(
                [
                    "--results-root",
                    str(method_root / "results"),
                    "--views-root",
                    str(args.views_root),
                    "--fastgs-dir",
                    str(args.fastgs_dir),
                    "--python",
                    args.python,
                ]
            )
        report = comparison.build_report(
            OUTPUT_ROOT / "methods",
            REFERENCE,
            [ADAPTED],
            context_keys=context_keys,
        )
        print(comparison.write_report(report, OUTPUT_ROOT / "report"), end="")
        return report
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
