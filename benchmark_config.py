"""Validate benchmark presets and construct reconstruction options."""

from pathlib import Path

from reconstruction.config import build_parser, resolve_args
from reconstruction.io import ROOT, load_json
from sampling import candidate_configurations

DEFAULT_CONFIG = ROOT / "benchmark_configs/base_initialization.json"
METHOD_OPTIONS = {
    "id",
    "initialization",
    "iterations",
    "lr_scale",
    "densify_from_iter",
    "densify_until_iter",
    "densification_interval",
    "extra_gaussians",
    "extra_gaussian_seed",
    "extra_gaussian_jitter",
    "extra_gaussian_opacity",
    "extra_gaussian_scale",
}


def load_config(path):
    config = load_json(path)
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a JSON object")
    if config.get("schema_version") != 1:
        raise ValueError("Benchmark config schema_version must be 1")
    unknown = config.keys() - {
        "schema_version",
        "reference",
        "base_iterations",
        "candidates",
        "adaptation_defaults",
        "methods",
        "objects",
        "experiments",
    }
    if unknown:
        raise ValueError(f"Unknown benchmark configuration keys: {sorted(unknown)}")
    for key in ("objects", "experiments"):
        if key in config and (
            not isinstance(config[key], list)
            or not config[key]
            or any(
                not isinstance(value, str) or not value.strip() for value in config[key]
            )
        ):
            raise ValueError(f"{key} must be a nonempty list of IDs")
    if (
        not isinstance(config.get("base_iterations"), int)
        or config["base_iterations"] < 1
    ):
        raise ValueError("base_iterations must be a positive integer")
    candidate_configurations(config.get("candidates"))
    if not config.get("candidates"):
        raise ValueError("Benchmark candidates must be explicitly specified")
    defaults = config.get("adaptation_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("adaptation_defaults must be a JSON object")
    if defaults.keys() - {
        "lr_scale",
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
    }:
        raise ValueError("Unknown adaptation_defaults setting")
    methods = config.get("methods", [])
    if not isinstance(methods, list) or any(
        not isinstance(method, dict) for method in methods
    ):
        raise ValueError("methods must be a list of method objects")
    identifiers = set()
    for method in methods:
        if method.keys() - METHOD_OPTIONS:
            raise ValueError(
                f"Unknown method settings: {sorted(method.keys() - METHOD_OPTIONS)}"
            )
        name = method.get("id", "")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
            or name in identifiers
        ):
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
        raise ValueError(
            "A benchmark needs a reference and at least one comparison method"
        )
    return config


def select_methods(config, requested=None):
    """Always include the fresh reference, before any warm-start methods."""
    available = {method["id"] for method in config["methods"]}
    if requested is not None:
        unknown = set(requested) - available
        if unknown:
            raise ValueError(f"Unknown method IDs: {sorted(unknown)}")
    selected = (
        available if requested is None else set(requested) | {config["reference"]}
    )
    methods = [method for method in config["methods"] if method["id"] in selected]
    if len(methods) < 2:
        raise ValueError("Select at least one method besides the reference")
    return sorted(methods, key=lambda method: method["id"] != config["reference"])


def method_arguments(args, config, method):
    """Build validated options directly, without converting them through CLI strings."""
    options = build_parser().parse_args([])
    overrides = {
        "skip_render": True,
        "shared_base": True,
        "gain_reference": "original-base",
        "fastgs_dir": args.fastgs_dir,
        "python": args.python,
        "views_root": args.views_root,
        "base_reconstructions_root": args.base_reconstructions_root,
        "reconstructions_root": args.output_root / "reconstructions",
        "results_root": args.output_root / "results",
        "initializations_root": args.output_root / "initializations",
        "run_name": method["id"],
        "candidate_initialization": method["initialization"],
        "base_iterations": config["base_iterations"],
        "candidate_iterations": method["iterations"],
        "candidate_limit": args.candidate_limit,
        "dry_run": args.dry_run,
        "force_retrain": args.force_retrain,
        "base_control": args.base_control,
    }
    adaptation = {
        "lr_scale",
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
    }
    for name, value in {**config.get("adaptation_defaults", {}), **method}.items():
        if name in adaptation:
            overrides["adaptation_" + name] = value
        elif name.startswith("extra_gaussian"):
            overrides[name] = value
    for name, value in overrides.items():
        setattr(options, name, value)
    return resolve_args(options)
