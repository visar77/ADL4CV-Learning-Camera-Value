"""Benchmark fresh-30k and base-cloud-3k across an object/experiment matrix."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import compare_reconstructions as comparison
import evaluate_metrics as evaluation
from benchmark_config import (
    DEFAULT_CONFIG,
    load_config,
    method_arguments,
    select_methods,
)
from benchmark_contexts import fastgs_fingerprint, saved_context_keys, select_contexts
from reconstruction_scripts.config import EXPERIMENTS_DIR, OUTPUT_ROOT
from reconstruction_scripts.experiments import process_experiment
from reconstruction_scripts.io import ROOT, load_json, save_json
from sampling import fibonacci_lattice


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "benchmarks/base_initialization"
    )
    parser.add_argument("--views-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument(
        "--base-reconstructions-root",
        type=Path,
        help="Existing OBJECT/EXPERIMENT/base outputs; defaults to output-root/shared_base",
    )
    parser.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--object-id",
        "--object-ids",
        dest="object_id",
        action="extend",
        nargs="+",
        help="Object IDs; accepts a list or repeated flags. Defaults to config objects or all rendered objects.",
    )
    parser.add_argument(
        "--experiment-id",
        "--experiment-ids",
        dest="experiment_id",
        action="extend",
        nargs="+",
        help="Experiment IDs; accepts a list or repeated flags. Defaults to config experiments or all definitions.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Select method IDs; the reference is always included",
    )
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument(
        "--base-control",
        action="store_true",
        help="Also train a separate base-only adaptation for each warm method",
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-base-retrain", action="store_true")
    parser.add_argument("--redo-metrics", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help="Run the self-contained CUDA integration tests first",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "config",
        "output_root",
        "views_root",
        "experiments_dir",
        "base_reconstructions_root",
        "fastgs_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.base_reconstructions_root = (
        args.base_reconstructions_root or args.output_root / "shared_base"
    )
    for name in ("candidate_limit", "max_contexts"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    return parser, args


def prepare_environment(args):
    environment = None
    if not args.dry_run and not args.skip_preflight:
        environment_file = args.output_root / "environment.json"
        subprocess.run(
            [
                args.python,
                str(ROOT / "gpu_preflight.py"),
                "--fastgs-dir",
                str(args.fastgs_dir),
                "--output",
                str(environment_file),
            ],
            check=True,
        )
        environment = load_json(environment_file)
    if args.gpu_smoke_test and not args.dry_run:
        subprocess.run(
            [
                args.python,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(ROOT / "tests"),
                "-p",
                "test_fastgs_adapter_gpu.py",
                "-v",
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "FASTGS_RUN_GPU_TESTS": "1",
                "FASTGS_DIR": str(args.fastgs_dir),
            },
            check=True,
        )
    return environment


def train_contexts(args, config, methods, contexts, fastgs_sha, environment):
    planned = set()
    for context_index, context in enumerate(contexts, 1):
        lattice = fibonacci_lattice(len(context.train["frames"]))
        for method_index, method in enumerate(methods):
            print(
                f"Benchmark [{context_index}/{len(contexts)}]: "
                f"{context.object_id}/{context.experiment['id']}/{method['id']}",
                flush=True,
            )
            options = method_arguments(args, config, method)
            options.runtime_environment = environment
            options.fastgs_sha256 = fastgs_sha
            options.data_fingerprint = context.fingerprint
            options.planned_reconstructions = planned
            # Force each shared base only once, before the first method uses it.
            options.force_base_retrain = args.force_base_retrain and method_index == 0
            process_experiment(
                options,
                context.object_id,
                context.object_dir,
                context.train,
                lattice,
                context.experiment,
            )


def evaluate_contexts(args, methods, contexts):
    cache = {}
    for method in methods:
        command = [
            "--results-root",
            str(args.output_root / "results" / method["id"]),
            "--views-root",
            str(args.views_root),
            "--fastgs-dir",
            str(args.fastgs_dir),
            "--python",
            args.python,
        ]
        if args.redo_metrics:
            command.append("--redo")
        for context in contexts:
            evaluation.main(
                command
                + [
                    "--object-id",
                    context.object_id,
                    "--experiment-id",
                    context.experiment["id"],
                ],
                shared_cache=cache,
            )


def write_comparison(args, config, methods, context_keys):
    report = comparison.build_report(
        args.output_root / "results",
        config["reference"],
        [method["id"] for method in methods],
        object_ids=args.object_id,
        experiment_ids=args.experiment_id,
        context_keys=context_keys,
    )
    print(comparison.write_report(report, args.output_root / "report"), end="")
    return report


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        manifest_path = args.output_root / "benchmark_manifest.json"
        previous = load_json(manifest_path) if manifest_path.is_file() else {}
        stored_config = args.output_root / "benchmark_config.json"
        replay = args.report_only or args.evaluate_only
        # Report/evaluation replay the recorded selection, including irregular
        # subsets produced by max-contexts, rather than expanding it to a product.
        if replay and args.config == DEFAULT_CONFIG and stored_config.is_file():
            args.config = stored_config
        config = load_config(args.config)
        requested = args.method
        if replay and requested is None and previous.get("selected_methods"):
            requested = previous["selected_methods"]
        methods = select_methods(config, requested)
        for method in methods:
            method_arguments(args, config, method)
        keys = (
            saved_context_keys(previous, args.object_id, args.experiment_id)
            if replay
            else None
        )
        if args.report_only:
            if keys is None:
                args.object_id = (
                    args.object_id
                    if args.object_id is not None
                    else config.get("objects")
                )
                args.experiment_id = (
                    args.experiment_id
                    if args.experiment_id is not None
                    else config.get("experiments")
                )
            return write_comparison(args, config, methods, keys)
        if (
            args.evaluate_only
            and args.views_root == OUTPUT_ROOT
            and previous.get("views_root")
        ):
            args.views_root = Path(previous["views_root"])
        contexts = select_contexts(args, config, context_keys=keys)
        if args.evaluate_only:
            recorded_inputs = {
                (context["object_id"], context["experiment_id"]): context.get(
                    "input_data_sha256"
                )
                for context in previous.get("contexts", [])
            }
            for context in contexts:
                recorded = recorded_inputs.get(context.key)
                if recorded and recorded != context.fingerprint:
                    raise ValueError(
                        f"Inputs changed for {context.object_id}/{context.experiment['id']}; "
                        "rerun training instead of --evaluate-only"
                    )
        fastgs_sha = fastgs_fingerprint(args.fastgs_dir)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(
        f"Selected {len(contexts)} object/experiment contexts, {len(methods)} methods",
        flush=True,
    )
    environment = prepare_environment(args)
    snapshot = {
        "configuration": config,
        "selected_methods": [method["id"] for method in methods],
        "contexts": [context.manifest_entry() for context in contexts],
        "fastgs_sha256": fastgs_sha,
        "environment": environment,
        "views_root": str(args.views_root),
    }
    if not args.dry_run and not args.evaluate_only:
        save_json(args.output_root / "benchmark_config.json", config)
        save_json(manifest_path, snapshot)
    if not args.evaluate_only:
        train_contexts(args, config, methods, contexts, fastgs_sha, environment)
    if args.dry_run or args.train_only:
        return snapshot
    evaluate_contexts(args, methods, contexts)
    return write_comparison(
        args, config, methods, {context.key for context in contexts}
    )


if __name__ == "__main__":
    main()
