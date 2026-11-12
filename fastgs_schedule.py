"""Warm-start schedules, kept independent of CUDA for inspection and testing."""

import math
from pathlib import Path


def add_adaptation_arguments(parser):
    group = parser.add_argument_group("Base reconstruction adaptation")
    group.add_argument("--init_ply", type=str,
                       help="Initialize from a trained Gaussian PLY with a fresh optimizer.")
    group.add_argument("--adaptation_lr_scale", type=float, default=0.1)
    group.add_argument("--adaptation_densify_from_iter", type=int, default=100)
    group.add_argument("--adaptation_densify_until_iter", type=int,
                       help="Exclusive end; defaults to half the adaptation budget. Use 0 to disable.")
    group.add_argument("--adaptation_densification_interval", type=int, default=250)


def configure_adaptation(opt, args):
    """Use local iteration numbers and a shortened position-LR schedule."""
    if opt.iterations < 1:
        raise ValueError("iterations must be positive")
    if not args.init_ply:
        return opt
    if getattr(args, "start_checkpoint", None):
        raise ValueError("init_ply and start_checkpoint are mutually exclusive")
    if not Path(args.init_ply).is_file():
        raise FileNotFoundError(f"Initialization PLY not found: {args.init_ply}")
    scale = args.adaptation_lr_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("adaptation_lr_scale must be finite and positive")
    start = args.adaptation_densify_from_iter
    end = args.adaptation_densify_until_iter
    if end is None:
        end = opt.iterations // 2
    if start < 0 or end < 0 or end > opt.iterations:
        raise ValueError("adaptation densification bounds must be between 0 and iterations")
    if end and start >= end:
        # Very short smoke runs can simply skip densification with the defaults.
        if args.adaptation_densify_until_iter is not None:
            raise ValueError("adaptation densify_from_iter must be less than densify_until_iter")
        end = 0
    if args.adaptation_densification_interval < 1:
        raise ValueError("adaptation_densification_interval must be positive")
    for name in ("position_lr_init", "position_lr_final", "lowfeature_lr",
                 "highfeature_lr", "opacity_lr", "scaling_lr", "rotation_lr"):
        setattr(opt, name, getattr(opt, name) * scale)
    opt.position_lr_max_steps = opt.iterations
    opt.densify_from_iter = start
    opt.densify_until_iter = end
    opt.densification_interval = args.adaptation_densification_interval
    return opt
