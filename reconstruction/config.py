"""Command-line configuration for standalone reconstruction runs."""

import argparse
import math
import sys
from pathlib import Path

from gaussian_initialization import validate_options
from reconstruction.io import ROOT

EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_ROOT = ROOT / "views"
OBJECTS_ROOT = ROOT / "objects"
RECONSTRUCTIONS_ROOT = ROOT / "reconstructions"
RESULTS_ROOT = ROOT / "results"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--force-base-retrain",
        action="store_true",
        help="Also rebuild the shared scratch initializer in base mode.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fastgs-dir", type=Path, default=ROOT.parent / "FastGS")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable in the GPU FastGS environment.",
    )
    parser.add_argument("--views-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--objects-root", type=Path, default=OBJECTS_ROOT)
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument(
        "--reconstructions-root", type=Path, default=RECONSTRUCTIONS_ROOT
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--base-reconstructions-root",
        type=Path,
        help="Existing scratch reconstructions, arranged as OBJECT/EXPERIMENT/base/.",
    )
    parser.add_argument(
        "--shared-base",
        action="store_true",
        help="Use the shared base root for scratch baselines too.",
    )
    parser.add_argument(
        "--gain-reference",
        choices=("original-base", "equal-budget-base-control"),
        default="original-base",
    )
    parser.add_argument(
        "--base-control",
        action="store_true",
        help="Also adapt base views alone as a separate control.",
    )
    parser.add_argument("--extra-gaussians", type=int, default=0)
    parser.add_argument("--extra-gaussian-seed", type=int, default=77)
    parser.add_argument("--extra-gaussian-jitter", type=float, default=0.5)
    parser.add_argument("--extra-gaussian-opacity", type=float, default=0.02)
    parser.add_argument("--extra-gaussian-scale", type=float, default=0.5)
    parser.add_argument(
        "--initializations-root",
        type=Path,
        help="Cache for derived PLYs; defaults beside reconstruction run directories.",
    )
    parser.add_argument(
        "--run-name",
        help="Separate output namespace; defaults to base_init_<budget> in base mode.",
    )
    parser.add_argument(
        "--candidate-initialization", choices=("scratch", "base"), default="scratch"
    )
    parser.add_argument("--base-iterations", type=int, default=30000)
    parser.add_argument(
        "--candidate-iterations",
        type=int,
        help="Defaults to 3000 in base mode, otherwise base-iterations.",
    )
    parser.add_argument("--adaptation-lr-scale", type=float, default=0.1)
    parser.add_argument("--adaptation-densify-from-iter", type=int, default=100)
    parser.add_argument(
        "--adaptation-densify-until-iter",
        type=int,
        help="Exclusive end, defaults to half the candidate budget; 0 disables densification.",
    )
    parser.add_argument("--adaptation-densification-interval", type=int, default=250)
    parser.add_argument("--evaluate-during-training", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pending commands without writing or training.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    return resolve_args(parser.parse_args(argv), parser)


def resolve_args(args, parser=None):
    """Validate CLI or programmatically constructed options and resolve paths."""
    parser = parser or build_parser()
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
            3000 if args.candidate_initialization == "base" else args.base_iterations
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
