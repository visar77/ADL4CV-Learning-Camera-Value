"""Run matched scratch/warm/extra-Gaussian experiments and write comparison tables."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import compare_reconstructions as comparison
import evaluate_metrics as evaluation
import orchestrator as pipeline
from sampling import candidate_configurations, fibonacci_lattice

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "benchmark_configs/base_initialization.json"
METHOD_OPTIONS = {"id", "initialization", "iterations", "lr_scale", "densify_from_iter", "densify_until_iter",
                  "densification_interval", "extra_gaussians", "extra_gaussian_seed", "extra_gaussian_jitter",
                  "extra_gaussian_opacity", "extra_gaussian_scale"}


def load_config(path):
    config = pipeline.load_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("Benchmark config schema_version must be 1")
    unknown = config.keys() - {"schema_version", "reference", "base_iterations", "candidates", "adaptation_defaults", "methods"}
    if unknown:
        raise ValueError(f"Unknown benchmark configuration keys: {sorted(unknown)}")
    if not isinstance(config.get("base_iterations"), int) or config["base_iterations"] < 1:
        raise ValueError("base_iterations must be a positive integer")
    candidate_configurations(config.get("candidates"))
    if not config.get("candidates"):
        raise ValueError("Benchmark candidates must be explicitly specified")
    defaults = config.get("adaptation_defaults", {})
    if defaults.keys() - {"lr_scale", "densify_from_iter", "densify_until_iter", "densification_interval"}:
        raise ValueError("Unknown adaptation_defaults setting")
    methods = config.get("methods", [])
    identifiers = set()
    for method in methods:
        if method.keys() - METHOD_OPTIONS:
            raise ValueError(f"Unknown method settings: {sorted(method.keys() - METHOD_OPTIONS)}")
        name = method.get("id", "")
        if not name or Path(name).name != name or name in (".", "..") or name in identifiers:
            raise ValueError(f"Invalid or duplicate method ID: {name}")
        identifiers.add(name)
        if method.get("initialization") not in ("scratch", "base"):
            raise ValueError(f"Unknown initialization for {name}")
        if not isinstance(method.get("iterations"), int) or method["iterations"] < 1:
            raise ValueError(f"iterations must be a positive integer for {name}")
        if method.get("extra_gaussians", 0) and method["initialization"] != "base":
            raise ValueError(f"Extra Gaussians require base initialization: {name}")
    reference = next((m for m in methods if m["id"] == config.get("reference")), None)
    if reference is None or reference["initialization"] != "scratch":
        raise ValueError("reference must name a scratch method")
    if len(methods) < 2:
        raise ValueError("A benchmark needs a reference and at least one comparison method")
    return config


def input_fingerprint(object_dir, train, test):
    digest = hashlib.sha256(json.dumps({"train": train, "test": test}, sort_keys=True).encode())
    paths = sorted({frame["file_path"] for manifest in (train, test) for frame in manifest["frames"]})
    for relative in paths:
        image = object_dir / relative
        if not image.suffix:
            image = image.with_suffix(".png")
        digest.update(pipeline.file_sha256(image).encode())
    return digest.hexdigest()


def fastgs_fingerprint(path):
    digest = hashlib.sha256()
    for relative in ("train.py", "scene/gaussian_model.py", "scene/__init__.py", "utils/fast_utils.py",
                     "arguments/__init__.py", "gaussian_renderer/__init__.py"):
        file = path / relative
        if not file.is_file():
            raise FileNotFoundError(f"FastGS file is missing: {file}")
        digest.update(relative.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def method_arguments(args, config, method):
    options = {**config.get("adaptation_defaults", {}), **method}
    command = ["--skip-render", "--shared-base", "--gain-reference", "original-base",
               "--fastgs-dir", str(args.fastgs_dir), "--python", args.python,
               "--views-root", str(args.views_root), "--base-reconstructions-root", str(args.base_reconstructions_root),
               "--reconstructions-root", str(args.output_root / "reconstructions"),
               "--results-root", str(args.output_root / "results"),
               "--initializations-root", str(args.output_root / "initializations"),
               "--run-name", method["id"], "--candidate-initialization", method["initialization"],
               "--base-iterations", str(config["base_iterations"]), "--candidate-iterations", str(method["iterations"])]
    for name, flag in {"lr_scale": "--adaptation-lr-scale", "densify_from_iter": "--adaptation-densify-from-iter",
                       "densify_until_iter": "--adaptation-densify-until-iter", "densification_interval": "--adaptation-densification-interval",
                       **{key: "--" + key.replace("_", "-") for key in METHOD_OPTIONS if key.startswith("extra_gaussian")}}.items():
        if name in options:
            command.extend([flag, str(options[name])])
    if args.candidate_limit is not None:
        command.extend(["--candidate-limit", str(args.candidate_limit)])
    for attr in ("dry_run", "force_retrain", "base_control"):
        if getattr(args, attr):
            command.append("--" + attr.replace("_", "-"))
    # A shared base is forced at most once per context, by the runner.
    return pipeline.parse_args(command)


def select_contexts(args, config):
    object_ids = args.object_id or sorted(p.name for p in args.views_root.iterdir()
                                         if (p / "global_transforms/transforms_train.json").is_file())
    experiments = [pipeline.load_experiment(p) for p in sorted(args.experiments_dir.glob("*.json"))]
    if args.experiment_id:
        missing = set(args.experiment_id) - {e["id"] for e in experiments}
        if missing:
            raise ValueError(f"Unknown experiment IDs: {sorted(missing)}")
        experiments = [e for e in experiments if e["id"] in args.experiment_id]
    contexts = []
    for object_id in object_ids:
        object_dir = args.views_root / object_id
        train = pipeline.load_json(object_dir / "global_transforms/transforms_train.json")
        test = pipeline.load_json(object_dir / "global_transforms/transforms_test.json")
        fingerprint = input_fingerprint(object_dir, train, test)
        for original in experiments:
            experiment = copy.deepcopy(original)
            experiment["candidates"] = config["candidates"]
            contexts.append((object_id, object_dir, train, test, experiment, fingerprint))
    if args.max_contexts:
        contexts = contexts[:args.max_contexts]
    if not contexts:
        raise ValueError("No matching contexts; check --views-root, --object-id, --experiments-dir and --experiment-id")
    return contexts


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmarks/base_initialization")
    parser.add_argument("--views-root", type=Path, default=pipeline.OUTPUT_ROOT)
    parser.add_argument("--experiments-dir", type=Path, default=pipeline.EXPERIMENTS_DIR)
    parser.add_argument("--base-reconstructions-root", type=Path, help="Existing shared OBJECT/EXPERIMENT/base outputs; otherwise use output-root/shared_base")
    parser.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--object-id", action="append")
    parser.add_argument("--experiment-id", action="append")
    parser.add_argument("--method", action="append", help="Select method IDs; the reference is always included")
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--base-control", action="store_true", help="Also train a separate base-only adaptation for each warm method")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-base-retrain", action="store_true")
    parser.add_argument("--redo-metrics", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--gpu-smoke-test", action="store_true", help="Also run the self-contained CUDA adaptation smoke test first")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("config", "output_root", "views_root", "experiments_dir", "base_reconstructions_root", "fastgs_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.base_reconstructions_root = args.base_reconstructions_root or args.output_root / "shared_base"
    if args.candidate_limit is not None and args.candidate_limit < 1:
        parser.error("candidate-limit must be positive for a comparison")
    if args.max_contexts is not None and args.max_contexts < 1:
        parser.error("max-contexts must be positive")
    try:
        previous_manifest_path = args.output_root / "benchmark_manifest.json"
        previous_manifest = pipeline.load_json(previous_manifest_path) if previous_manifest_path.is_file() else {}
        stored_config = args.output_root / "benchmark_config.json"
        if (args.report_only or args.evaluate_only) and args.config == DEFAULT_CONFIG and stored_config.is_file():
            args.config = stored_config
        config = load_config(args.config)
        methods = config["methods"]
        if args.report_only and args.method is None and previous_manifest.get("selected_methods"):
            args.method = previous_manifest["selected_methods"]
        if args.method:
            unknown = set(args.method) - {m["id"] for m in methods}
            if unknown:
                raise ValueError(f"Unknown method IDs: {sorted(unknown)}")
            methods = [m for m in methods if m["id"] in set(args.method) | {config["reference"]}]
        if len(methods) < 2:
            raise ValueError("Select at least one method besides the reference")
        # Validate every method before starting any GPU work.
        for method in methods:
            method_arguments(args, config, method)
        if args.report_only:
            context_keys = {(c["object_id"], c["experiment_id"]) for c in previous_manifest.get("contexts", [])} or None
            report = comparison.build_report(args.output_root / "results", config["reference"],
                                              [m["id"] for m in methods], object_ids=args.object_id,
                                              experiment_ids=args.experiment_id, context_keys=context_keys)
            print(comparison.write_report(report, args.output_root / "report"), end="")
            return report
        contexts = select_contexts(args, config)
        fastgs_sha = fastgs_fingerprint(args.fastgs_dir)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    environment = None
    if not args.dry_run and not args.skip_preflight:
        environment_file = args.output_root / "environment.json"
        subprocess.run([args.python, str(ROOT / "gpu_preflight.py"), "--fastgs-dir", str(args.fastgs_dir),
                        "--output", str(environment_file)], check=True)
        environment = pipeline.load_json(environment_file)
    if args.gpu_smoke_test and not args.dry_run:
        subprocess.run([args.python, "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                        "-p", "test_fastgs_adapter_gpu.py", "-v"], cwd=ROOT,
                       env={**os.environ, "FASTGS_RUN_GPU_TESTS": "1", "FASTGS_DIR": str(args.fastgs_dir)}, check=True)
    snapshot = {"configuration": config, "selected_methods": [m["id"] for m in methods],
                "contexts": [{"object_id": c[0], "experiment_id": c[4]["id"], "input_data_sha256": c[5]}
                             for c in contexts], "fastgs_sha256": fastgs_sha,
                "environment": environment, "views_root": str(args.views_root)}
    if not args.dry_run:
        pipeline.save_json(args.output_root / "benchmark_config.json", config)
        pipeline.save_json(args.output_root / "benchmark_manifest.json", snapshot)
    if not args.evaluate_only:
        planned = set()
        for object_id, object_dir, train, test, experiment, fingerprint in contexts:
            for method_index, method in enumerate(methods):
                print(f"Benchmark: {object_id}/{experiment['id']}/{method['id']}", flush=True)
                run_args = method_arguments(args, config, method)
                run_args.runtime_environment = environment
                run_args.fastgs_sha256 = fastgs_sha
                run_args.data_fingerprint = fingerprint
                run_args.planned_reconstructions = planned
                run_args.force_base_retrain = args.force_base_retrain and method_index == 0
                pipeline.process_experiment(run_args, object_id, object_dir, train,
                                            fibonacci_lattice(len(train["frames"])), experiment)
    if args.dry_run or args.train_only:
        return snapshot
    cache = {}
    for method in methods:
        command = ["--results-root", str(args.output_root / "results" / method["id"]),
                   "--views-root", str(args.views_root), "--fastgs-dir", str(args.fastgs_dir), "--python", args.python]
        # Evaluation filters are applied one exact context at a time.
        if args.redo_metrics:
            command.append("--redo")
        for object_id, _, _, _, experiment, _ in contexts:
            evaluation.main(command + ["--object-id", object_id, "--experiment-id", experiment["id"]], shared_cache=cache)
    report = comparison.build_report(args.output_root / "results", config["reference"], [m["id"] for m in methods],
                                      context_keys={(c[0], c[4]["id"]) for c in contexts})
    print(comparison.write_report(report, args.output_root / "report"), end="")
    return report


if __name__ == "__main__":
    main()
