"""Render shared views and run independent FastGS reconstruction experiments."""

import subprocess
import sys

from reconstruction_scripts.config import parse_args
from reconstruction_scripts.datasets import discover_objects, load_experiments
from reconstruction_scripts.experiments import process_experiment
from reconstruction_scripts.io import ROOT, load_json
from sampling import fibonacci_lattice


def prepare_object(args, object_id):
    """Render missing views, then load the immutable global camera corpus."""
    object_dir = args.views_root / object_id
    global_dir = object_dir / "global_transforms"
    missing = any(
        not (global_dir / f"transforms_{split}.json").is_file()
        for split in ("train", "test")
    )
    if not args.skip_render and (args.rerender or missing):
        if args.dry_run:
            raise ValueError(
                "Render views before a dry run, or use --skip-render with an existing corpus"
            )
        glb_path = args.objects_root / f"{object_id}.glb"
        if not glb_path.is_file():
            raise FileNotFoundError(glb_path)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "renderer.py"),
                "--glb-path",
                str(glb_path),
                "--object-id",
                object_id,
                "--output-root",
                str(args.views_root),
            ],
            check=True,
        )
    train = load_json(global_dir / "transforms_train.json")
    load_json(global_dir / "transforms_test.json")
    return object_dir, train


def main(argv=None):
    args = parse_args(argv)
    print(
        f"Reconstructions: {args.reconstructions_root}\nResults: {args.results_root}",
        flush=True,
    )
    if args.object_id is not None:
        # Explicit IDs may refer to rendered corpora even when the GLBs are absent.
        object_ids = list(dict.fromkeys(args.object_id))
        if any(not name or name in (".", "..") or "/" in name for name in object_ids):
            raise ValueError("Object IDs must be single directory names")
    else:
        directory = args.views_root if args.skip_render else args.objects_root
        object_ids = discover_objects(directory, rendered=args.skip_render)
    if not object_ids:
        raise FileNotFoundError(
            "No objects found; specify --object-id and check --views-root/--objects-root"
        )
    experiments = load_experiments(args.experiments_dir, args.experiment_id)
    for object_id in object_ids:
        object_dir, train = prepare_object(args, object_id)
        lattice = fibonacci_lattice(len(train["frames"]))
        for experiment in experiments:
            try:
                process_experiment(
                    args, object_id, object_dir, train, lattice, experiment
                )
            except Exception as error:
                print(
                    f"Failed {object_id}/{experiment['id']}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    raise


if __name__ == "__main__":
    main()
