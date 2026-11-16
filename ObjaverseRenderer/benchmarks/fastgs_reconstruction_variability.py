"""Measure run-to-run FastGS variation for selected reconstructions."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from statistics import mean, pstdev

BENCHMARKS_ROOT = Path(__file__).resolve().parent
RENDERER_ROOT = BENCHMARKS_ROOT.parent
SCRIPT_STEM = Path(__file__).stem
DEFAULT_CONFIG = BENCHMARKS_ROOT / f"{SCRIPT_STEM}.json"
OUTPUT_ROOT = BENCHMARKS_ROOT / SCRIPT_STEM

for path in (BENCHMARKS_ROOT, RENDERER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import comparison_base_cloud_and_fresh_candidates as shared
import evaluate_metrics as evaluation
import orchestrator
import training_fastgs
from sampling import fibonacci_lattice

FRESH = "fresh-30k"
BASE_CLOUD = "base-cloud-3k"
BASE_ITERATIONS = 30000
FRESH_ITERATIONS = 30000
ADAPTED_ITERATIONS = 3000
METHODS = (FRESH, BASE_CLOUD)
METRICS = ("PSNR", "SSIM", "LPIPS")


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Benchmark config must be a schema_version 1 JSON object")
    allowed = {
        "schema_version",
        "repetitions",
        "objects",
        "experiments",
        "candidate_ids",
    }
    if config.keys() - allowed:
        raise ValueError(f"Unknown benchmark config keys: {sorted(config.keys() - allowed)}")
    repetitions = config.get("repetitions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    for key in ("objects", "experiments", "candidate_ids"):
        values = config.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{key} must be a nonempty list of unique strings")
    if any(Path(value).name != value for value in config["objects"]):
        raise ValueError("Object IDs must be single directory names")
    if any(Path(value).name != value for value in config["experiments"]):
        raise ValueError("Experiment IDs must be single directory names")
    if any(Path(value).name != value for value in config["candidate_ids"]):
        raise ValueError("Candidate IDs must be single directory names")
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
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Unique output subdirectory name; defaults to UTC time.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_contexts(args, config):
    previous_experiments = orchestrator.CONFIG["experiments_dir"]
    orchestrator.CONFIG["experiments_dir"] = args.experiments_dir
    try:
        experiments = {
            experiment["id"]: experiment
            for experiment in orchestrator.load_experiments()
        }
    finally:
        orchestrator.CONFIG["experiments_dir"] = previous_experiments
    unknown_experiments = set(config["experiments"]) - experiments.keys()
    if unknown_experiments:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown_experiments)}")

    contexts = []
    for object_id in config["objects"]:
        object_dir = args.views_root / object_id
        train_path = object_dir / "global_transforms/transforms_train.json"
        test_path = object_dir / "global_transforms/transforms_test.json"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(f"Missing global train/test transforms for {object_id}")
        train = orchestrator.load_json(train_path)
        test = orchestrator.load_json(test_path)
        metadata_path = object_dir / "metadata.json"
        metadata = orchestrator.load_json(metadata_path) if metadata_path.is_file() else {}
        if int(metadata.get("lattice_size", len(train["frames"]))) != len(train["frames"]):
            raise ValueError(f"Lattice metadata does not match train frames for {object_id}")
        trajectory_elevation = float(metadata.get("trajectory_elevation_deg", 20.0))
        lattice = fibonacci_lattice(len(train["frames"]))

        for experiment_id in config["experiments"]:
            experiment = experiments[experiment_id]
            base_indices = orchestrator.select_indices(
                experiment["train"]["strategy"],
                int(experiment["train"]["count"]),
                lattice,
                seed=int(experiment["train"].get("seed", 0)),
                trajectory_elevation_deg=trajectory_elevation,
            )
            candidates = orchestrator.make_candidate_jobs(
                experiment, lattice, base_indices, trajectory_elevation
            )
            available = {candidate["id"]: candidate for candidate in candidates}
            requested = set(config["candidate_ids"]) - {"base"}
            unknown = requested - available.keys()
            if unknown:
                raise ValueError(
                    f"Unknown candidate IDs for {object_id}/{experiment_id}: {sorted(unknown)}"
                )
            redundant = [
                candidate_id
                for candidate_id in requested
                if available[candidate_id]["status"] == "redundant"
            ]
            if redundant:
                raise ValueError(
                    f"Redundancy controls are not reconstructions: {sorted(redundant)}"
                )
            contexts.append(
                {
                    "object_id": object_id,
                    "object_dir": object_dir,
                    "train": train,
                    "test": test,
                    "experiment_id": experiment_id,
                    "experiment": experiment,
                    "base_indices": base_indices,
                    "candidates": available,
                    "input_data_sha256": shared.input_fingerprint(
                        object_dir, train, test
                    ),
                }
            )
    return contexts


def gaussian_count(checkpoint):
    count = None
    with checkpoint.open("rb") as ply:
        while True:
            line = ply.readline()
            if not line:
                break
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"Invalid PLY header in {checkpoint}") from error
            if text == "end_header":
                break
            parts = text.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                try:
                    count = int(parts[2])
                except ValueError as error:
                    raise ValueError(f"Invalid vertex count in {checkpoint}") from error
    if count is None or count < 0:
        raise ValueError(f"PLY header has no valid vertex count: {checkpoint}")
    return count


def _stats(values):
    return {
        "count": len(values),
        "mean": mean(values),
        "population_std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _record_reconstruction(
    result,
    result_path,
    method,
    context,
    candidate_id,
    repetition,
    stage,
    name,
    model_dir,
    checkpoint,
    timing,
):
    if stage == "base_initializer" or candidate_id == "base":
        iterations = BASE_ITERATIONS
    elif method == FRESH:
        iterations = FRESH_ITERATIONS
    else:
        iterations = ADAPTED_ITERATIONS
    result["reconstructions"][name] = {
        "status": "complete",
        "iterations": iterations,
        "reconstruction_path": str(model_dir),
        "checkpoint_path": str(checkpoint),
        "train_lattice_indices": (
            context["base_indices"]
            if stage == "base_initializer" or candidate_id == "base"
            else context["candidates"][candidate_id]["train_indices"]
        ),
        "timing": timing,
    }
    orchestrator.save_json(result_path, result)
    return {
        "method": method,
        "object_id": context["object_id"],
        "experiment_id": context["experiment_id"],
        "candidate_id": candidate_id,
        "repetition": repetition,
        "stage": stage,
        "name": name,
        "reconstruction_path": str(model_dir),
        "checkpoint_path": str(checkpoint),
        "gaussian_count": gaussian_count(checkpoint),
        "training_wall_seconds": timing.get("wall_time_seconds"),
        "result_path": str(result_path),
    }


def _run_reconstruction(args, run_root, context, method, candidate_id, repetition):
    candidate = context["candidates"].get(candidate_id)
    train_indices = (
        context["base_indices"]
        if candidate_id == "base"
        else candidate["train_indices"]
    )
    repeat_name = f"repetition-{repetition:04d}"
    result_path = (
        run_root / "methods" / method / "results"
        / context["object_id"] / context["experiment_id"]
        / repeat_name / candidate_id / "results.json"
    )
    result = {
        "object_id": context["object_id"],
        "experiment_id": context["experiment_id"],
        "dataset_path": str(context["object_dir"]),
        "train_lattice_indices": context["base_indices"],
        "reconstructions": {},
        "metrics": {},
        "protocol": {
            "candidate_id": candidate_id,
            "repetition": repetition,
            "input_data_sha256": context["input_data_sha256"],
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    method_root = run_root / "methods" / method
    reconstruction_root = method_root / "reconstructions"
    target_folder = (
        reconstruction_root / context["object_id"] / context["experiment_id"]
        / repeat_name / candidate_id
    )

    if method == BASE_CLOUD and candidate_id != "base":
        model_dir = target_folder / "adapted"
        checkpoint, timing = shared._train_model(
            args,
            context["object_dir"],
            context["train"],
            train_indices,
            model_dir,
            ADAPTED_ITERATIONS,
            init_ply=context["shared_base"]["checkpoint_path"],
            adaptation={},
        )
    else:
        model_dir = target_folder / ("base" if candidate_id == "base" else "fresh")
        iterations = BASE_ITERATIONS if candidate_id == "base" else FRESH_ITERATIONS
        checkpoint, timing = shared._train_model(
            args,
            context["object_dir"],
            context["train"],
            train_indices,
            model_dir,
            iterations,
        )

    records.append(
        _record_reconstruction(
            result, result_path, method, context, candidate_id, repetition,
            "final", candidate_id, model_dir, checkpoint, timing,
        )
    )
    return result_path, records


def _run_shared_base(args, run_root, context):
    method_root = run_root / "methods" / BASE_CLOUD
    model_dir = (
        method_root / "reconstructions" / context["object_id"]
        / context["experiment_id"] / "shared-base"
    )
    result_path = (
        method_root / "results" / context["object_id"]
        / context["experiment_id"] / "shared-base" / "results.json"
    )
    checkpoint, timing = shared._train_model(
        args,
        context["object_dir"],
        context["train"],
        context["base_indices"],
        model_dir,
        BASE_ITERATIONS,
    )
    result = {
        "object_id": context["object_id"],
        "experiment_id": context["experiment_id"],
        "dataset_path": str(context["object_dir"]),
        "train_lattice_indices": context["base_indices"],
        "reconstructions": {},
        "metrics": {},
        "protocol": {
            "candidate_id": "base",
            "repetition": 1,
            "input_data_sha256": context["input_data_sha256"],
            "shared_base": True,
        },
    }
    record = _record_reconstruction(
        result,
        result_path,
        BASE_CLOUD,
        context,
        "base",
        1,
        "final",
        "base",
        model_dir,
        checkpoint,
        timing,
    )
    context["shared_base"] = {
        "checkpoint_path": str(checkpoint),
        "reconstruction_path": str(model_dir),
        "result_path": str(result_path),
    }
    return {
        "result_path": str(result_path),
        "records": [record],
    }


def _collect_records(run_plans):
    records = []
    for plan in run_plans:
        result = orchestrator.load_json(Path(plan["result_path"]))
        for record in plan["records"]:
            metrics = result.get("metrics", {}).get(record["name"])
            if not isinstance(metrics, dict) or not isinstance(metrics.get("averages"), dict):
                raise ValueError(
                    f"Metrics were not produced for {record['method']}/"
                    f"{record['object_id']}/{record['experiment_id']}/"
                    f"{record['candidate_id']}/{record['repetition']}/{record['name']}"
                )
            averages = metrics["averages"]
            values = {}
            for metric in METRICS:
                if metric not in averages:
                    raise ValueError(f"Missing {metric} metrics for {record['name']}")
                value = float(averages[metric])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {metric} metrics for {record['name']}")
                values[metric] = value
            record["metrics"] = values
            records.append(record)
    return records


def _summaries(records):
    groups = defaultdict(list)
    for record in records:
        key = (
            record["method"], record["object_id"], record["experiment_id"],
            record["candidate_id"], record["stage"],
        )
        groups[key].append(record)
    summaries = []
    for key, items in sorted(groups.items()):
        method, object_id, experiment_id, candidate_id, stage = key
        summaries.append(
            {
                "method": method,
                "object_id": object_id,
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "stage": stage,
                "repetitions": len(items),
                "gaussian_count": _stats([item["gaussian_count"] for item in items]),
                "training_wall_seconds": (
                    _stats([item["training_wall_seconds"] for item in items])
                    if all(item["training_wall_seconds"] is not None for item in items)
                    else None
                ),
                "metrics": {
                    metric: _stats([item["metrics"][metric] for item in items])
                    for metric in METRICS
                },
            }
        )
    return summaries


def _text_report(report):
    lines = [
        "FastGS reconstruction variability",
        f"Repetitions per selected object/experiment/candidate: {report['configuration']['repetitions']}",
        "Fresh targets use 30,000 iterations; base-cloud targets use a newly trained 30,000-iteration base",
        "once per object/experiment, then reuse it for each 3,000-iteration adaptation.",
        "Gaussian count and metrics are summarized with population standard deviation.",
        "",
    ]
    for item in report["summaries"]:
        lines.append(
            f"{item['method']} | {item['object_id']} | {item['experiment_id']} | "
            f"{item['candidate_id']} | {item['stage']} | n={item['repetitions']}"
        )
        gaussian = item["gaussian_count"]
        lines.append(
            f"  Gaussians: mean {gaussian['mean']:.2f}, std {gaussian['population_std']:.2f}, "
            f"min {gaussian['min']}, max {gaussian['max']}"
        )
        for metric in METRICS:
            values = item["metrics"][metric]
            lines.append(
                f"  {metric}: mean {values['mean']:.6f}, std {values['population_std']:.6f}, "
                f"min {values['min']:.6f}, max {values['max']:.6f}"
            )
        times = item["training_wall_seconds"]
        if times:
            lines.append(
                f"  Training seconds: mean {times['mean']:.2f}, "
                f"std {times['population_std']:.2f}"
            )
    return "\n".join(lines) + "\n"


def _run_id(value):
    value = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if Path(value).name != value or value in (".", ".."):
        raise ValueError("run-id must be a single directory name")
    return value


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.config = args.config.expanduser().resolve()
    args.views_root = args.views_root.expanduser().resolve()
    args.experiments_dir = args.experiments_dir.expanduser().resolve()
    args.fastgs_dir = args.fastgs_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    try:
        config = load_config(args.config)
        contexts = _load_contexts(args, config)
        plan = {
            "configuration": config,
            "reconstruction_targets": [
                {
                    "object_id": context["object_id"],
                    "experiment_id": context["experiment_id"],
                    "candidate_id": candidate_id,
                    "repetitions": config["repetitions"],
                }
                for context in contexts
                for candidate_id in config["candidate_ids"]
            ],
            "training_runs_by_method": {
                FRESH: config["repetitions"]
                * len(contexts)
                * len(config["candidate_ids"]),
                BASE_CLOUD: len(contexts)
                + sum(
                    config["repetitions"]
                    for _context in contexts
                    for candidate_id in config["candidate_ids"]
                    if candidate_id != "base"
                ),
            },
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return plan

        train_script = args.fastgs_dir / "train.py"
        if not train_script.is_file():
            raise FileNotFoundError(f"FastGS train.py not found: {train_script}")
        run_id = _run_id(args.run_id)
        run_root = args.output_root / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        orchestrator.save_json(run_root / "benchmark_config.json", config)
        orchestrator.save_json(
            run_root / "benchmark_manifest.json",
            {**plan, "run_id": run_id, "methods": list(METHODS)},
        )

        run_plans = []
        for context in contexts:
            shared_base_plan = _run_shared_base(args, run_root, context)
            run_plans.append(shared_base_plan)
            for candidate_id in config["candidate_ids"]:
                for repetition in range(1, config["repetitions"] + 1):
                    for method in METHODS:
                        if method == BASE_CLOUD and candidate_id == "base":
                            continue
                        result_path, records = _run_reconstruction(
                            args, run_root, context, method, candidate_id,
                            repetition,
                        )
                        run_plans.append(
                            {"result_path": str(result_path), "records": records}
                        )

        for method in METHODS:
            evaluation.main(
                [
                    "--results-root",
                    str(run_root / "methods" / method / "results"),
                    "--views-root",
                    str(args.views_root),
                    "--fastgs-dir",
                    str(args.fastgs_dir),
                    "--python",
                    args.python,
                ]
            )
        records = _collect_records(run_plans)
        report = {
            "schema_version": 1,
            "run_id": run_id,
            "configuration": config,
            "methods": list(METHODS),
            "records": records,
            "summaries": _summaries(records),
        }
        orchestrator.save_json(run_root / "report.json", report)
        text = _text_report(report)
        (run_root / "report.txt").write_text(text, encoding="utf-8")
        print(text, end="")
        return report
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
