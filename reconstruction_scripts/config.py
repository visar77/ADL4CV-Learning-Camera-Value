"""Command-line configuration for standalone reconstruction runs."""

import argparse
import json
import math
import sys
from pathlib import Path

from gaussian_initialization import validate_options
from reconstruction_scripts.io import ROOT

EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_ROOT = ROOT / "views"
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
NULLABLE_OPTIONS = {
    "object_id",
    "experiment_id",
    "candidate_limit",
    "python",
    "base_reconstructions_root",
    "initializations_root",
    "run_name",
    "candidate_iterations",
    "adaptation_densify_until_iter",
}


def build_parser(defaults=None):
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON overrides for reconstruction_scripts/config.json; "
        "relative paths are based on that file.",
    )
    parser.add_argument(
        "--object-id",
        "--object-ids",
        dest="object_id",
        action="extend",
        nargs="+",
        help="Object IDs; accepts a list or repeated flags.",
    )
    parser.add_argument(
        "--experiment-id",
        "--experiment-ids",
        dest="experiment_id",
        action="extend",
        nargs="+",
        help="Experiment IDs; accepts a list or repeated flags.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        help="Limit non-redundant candidates per context for smoke tests.",
    )
    parser.add_argument("--skip-render", action=argparse.BooleanOptionalAction)
    parser.add_argument("--rerender", action=argparse.BooleanOptionalAction)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--force-base-retrain",
        action=argparse.BooleanOptionalAction,
        help="Also rebuild the shared scratch initializer in base mode.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fastgs-dir", type=Path)
    parser.add_argument(
        "--python",
        help="Python executable in the GPU FastGS environment.",
    )
    parser.add_argument("--views-root", type=Path)
    parser.add_argument("--objects-root", type=Path)
    parser.add_argument("--experiments-dir", type=Path)
    parser.add_argument("--reconstructions-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--base-reconstructions-root",
        type=Path,
        help="Existing scratch reconstructions, arranged as OBJECT/EXPERIMENT/base/.",
    )
    parser.add_argument(
        "--shared-base",
        action=argparse.BooleanOptionalAction,
        help="Use the shared base root for scratch baselines too.",
    )
    parser.add_argument(
        "--gain-reference",
        choices=("original-base", "equal-budget-base-control"),
    )
    parser.add_argument(
        "--base-control",
        action=argparse.BooleanOptionalAction,
        help="Also adapt base views alone as a separate control.",
    )
    parser.add_argument("--extra-gaussians", type=int)
    parser.add_argument("--extra-gaussian-seed", type=int)
    parser.add_argument("--extra-gaussian-jitter", type=float)
    parser.add_argument("--extra-gaussian-opacity", type=float)
    parser.add_argument("--extra-gaussian-scale", type=float)
    parser.add_argument(
        "--initializations-root",
        type=Path,
        help="Cache for derived PLYs; defaults beside reconstruction run directories.",
    )
    parser.add_argument(
        "--run-name",
        help="Separate output namespace; defaults to base_init_<budget> in base mode.",
    )
    parser.add_argument("--candidate-initialization", choices=("scratch", "base"))
    parser.add_argument("--base-iterations", type=int)
    parser.add_argument(
        "--candidate-iterations",
        type=int,
        help="Defaults to config base_candidate_iterations in base mode, "
        "otherwise base-iterations.",
    )
    parser.add_argument("--adaptation-lr-scale", type=float)
    parser.add_argument("--adaptation-densify-from-iter", type=int)
    parser.add_argument(
        "--adaptation-densify-until-iter",
        type=int,
        help="Exclusive end, defaults to half the candidate budget; 0 disables densification.",
    )
    parser.add_argument("--adaptation-densification-interval", type=int)
    parser.add_argument(
        "--evaluate-during-training", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        help="Print pending commands without writing or training.",
    )
    parser.set_defaults(**(load_defaults(parser) if defaults is None else defaults))
    return parser


def load_defaults(parser, config_path=None):
    """Merge JSON defaults, validating types before passing them to argparse."""

    def read(path):
        try:
            with path.open(encoding="utf-8") as stream:
                values = json.load(stream)
        except (OSError, ValueError) as error:
            parser.error(f"Cannot load config {path}: {error}")
        if not isinstance(values, dict):
            parser.error(f"Config {path} must be a JSON object")
        return values

    shipped = read(DEFAULT_CONFIG)
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in ("help", "config")
    }
    allowed = actions.keys() | {"base_candidate_iterations"}
    if shipped.keys() != allowed:
        parser.error(f"Default config {DEFAULT_CONFIG} must define {sorted(allowed)}")
    config_path = Path(config_path or DEFAULT_CONFIG).expanduser().resolve()
    layers = [(DEFAULT_CONFIG, shipped)]
    if config_path != DEFAULT_CONFIG:
        layers.append((config_path, read(config_path)))
    defaults = {}
    for path, values in layers:
        unknown = values.keys() - allowed
        if unknown:
            parser.error(f"Unknown config keys in {path}: {sorted(unknown)}")
        for name, value in values.items():
            action = actions.get(name)
            if value is None and name in NULLABLE_OPTIONS:
                defaults[name] = None
                continue
            if action is None or action.type is int:
                valid = type(value) is int
            elif action.type is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            elif isinstance(action, argparse.BooleanOptionalAction):
                valid = type(value) is bool
            elif action.nargs == "+":
                valid = (
                    isinstance(value, list)
                    and bool(value)
                    and all(isinstance(item, str) and item.strip() for item in value)
                )
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid or (action and action.choices and value not in action.choices):
                parser.error(f"Invalid config value for {name} in {path}: {value!r}")
            if action and action.type is Path:
                value = Path(value).expanduser()
                if not value.is_absolute():
                    value = path.parent / value
                value = value.resolve()
            defaults[name] = value
    if defaults["base_candidate_iterations"] < 1:
        parser.error("base_candidate_iterations must be positive")
    if defaults["python"] is None:
        defaults["python"] = sys.executable
    return defaults


def parse_args(argv=None):
    # Parse only supplied flags, so list arguments replace configured lists too.
    parser = build_parser(defaults={})
    supplied = parser.parse_args(argv)
    config_path = getattr(supplied, "config", DEFAULT_CONFIG)
    defaults = load_defaults(parser, config_path)
    args = argparse.Namespace(**(defaults | vars(supplied)))
    return resolve_args(args, parser)


def resolve_args(args, parser=None):
    """Validate CLI or programmatically constructed options and resolve paths."""
    parser = parser or build_parser(defaults={})
    try:
        validate_options(
            args.extra_gaussians,
            args.extra_gaussian_jitter,
            args.extra_gaussian_opacity,
            args.extra_gaussian_scale,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.extra_gaussian_seed < 0:
        parser.error("extra-gaussian-seed must be nonnegative")
    if args.extra_gaussians and args.candidate_initialization != "base":
        parser.error("extra-gaussians requires candidate-initialization=base")
    if (
        args.gain_reference == "equal-budget-base-control"
        and args.candidate_initialization != "base"
    ):
        parser.error("equal-budget-base-control requires candidate-initialization=base")
    if args.candidate_iterations is None:
        args.candidate_iterations = (
            args.base_candidate_iterations
            if args.candidate_initialization == "base"
            else args.base_iterations
        )
    if args.base_iterations < 1 or args.candidate_iterations < 1:
        parser.error("iteration budgets must be positive")
    if args.candidate_limit is not None and args.candidate_limit < 0:
        parser.error("candidate-limit must be nonnegative")
    if not math.isfinite(args.adaptation_lr_scale) or args.adaptation_lr_scale <= 0:
        parser.error("adaptation-lr-scale must be finite and positive")
    if (
        args.adaptation_densification_interval < 1
        or args.adaptation_densify_from_iter < 0
    ):
        parser.error("invalid adaptation densification interval/start")
    if args.adaptation_densify_until_iter is None:
        args.adaptation_densify_until_iter = args.candidate_iterations // 2
        if args.adaptation_densify_until_iter <= args.adaptation_densify_from_iter:
            args.adaptation_densify_until_iter = 0
    end = args.adaptation_densify_until_iter
    if (
        end < 0
        or end > args.candidate_iterations
        or (end and end <= args.adaptation_densify_from_iter)
    ):
        parser.error(
            "adaptation densification end must be 0 or between start and candidate-iterations"
        )
    if args.candidate_initialization == "base" and args.run_name is None:
        args.run_name = f"base_init_{args.candidate_iterations}" + (
            f"_extra_{args.extra_gaussians}" if args.extra_gaussians else ""
        )
    if args.run_name and (
        Path(args.run_name).name != args.run_name or args.run_name in (".", "..")
    ):
        parser.error("run-name must be a single directory name")
    if args.candidate_initialization == "base" and not args.run_name:
        parser.error("base initialization requires a nonempty run-name")
    for name in (
        "fastgs_dir",
        "views_root",
        "objects_root",
        "experiments_dir",
        "reconstructions_root",
        "results_root",
        "base_reconstructions_root",
        "initializations_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.base_reconstructions_root = (
        args.base_reconstructions_root or args.reconstructions_root
    )
    args.initializations_root = (
        args.initializations_root or args.reconstructions_root / "_initializations"
    )
    if args.run_name:
        args.reconstructions_root /= args.run_name
        args.results_root /= args.run_name
    return args
