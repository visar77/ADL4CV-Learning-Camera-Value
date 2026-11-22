"""Render and reconstruct an isolated, object-configurable pipeline demo."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent.parent
sys.path.insert(0, str(ROOT))

import orchestrator
from demo.demo_pipeline.initialization import prepare_points
from demo.demo_pipeline.render_gif import FRAMES, save_gif_from_files
from gaussian_initialization import sha256
from renderer import find_blender


def demo_training_config(config, defaults):
    settings = config["training"]
    keys = {
        "base_iterations", "candidate_iterations",
        "adaptation_lr_scale", "adaptation_densify_from_iter", "adaptation_densify_until_iter",
        "adaptation_densification_interval",
    }
   
    for key in ("base_iterations", "candidate_iterations", "adaptation_densification_interval"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    scale = settings["adaptation_lr_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("training.adaptation_lr_scale must be a finite positive number")
    start = settings["adaptation_densify_from_iter"]
    end = settings["adaptation_densify_until_iter"]
    if type(start) is not int or start < 0:
        raise ValueError("training.adaptation_densify_from_iter must be a non-negative integer")
    if end is not None and (type(end) is not int or end < 0 or end > settings["candidate_iterations"]
                            or (end > 0 and start >= end)):
        raise ValueError("Invalid training adaptation densification bounds")
    training = {**defaults, **{key: value for key, value in settings.items() if key != "candidate_iterations"}}
    training["candidate_base_iterations"] = settings["candidate_iterations"]
    return training


def select_jobs(experiment, object_id, object_dir, train, candidate_ids):
    base, available = orchestrator.plan_candidates(object_id, object_dir, train, [experiment])[experiment["id"]]
    jobs_by_id = {job["id"]: job for job in available}
    if len(candidate_ids) != 3 or len(set(candidate_ids)) != 3 or any(item not in jobs_by_id for item in candidate_ids):
        raise ValueError("Configure exactly three distinct existing candidate IDs")
    jobs = [jobs_by_id[item] for item in candidate_ids]
    if any(job["strategy"] == "redundancy" for job in jobs):
        raise ValueError("Visualization requires three non-redundant candidate cameras")
    candidate_indices = [job["train_indices"][-1] for job in jobs]
    if len(set(candidate_indices)) != 3:
        raise ValueError("The selected candidate jobs must use three different cameras")
    return base, jobs, {"base": base, "candidates": dict(zip(candidate_ids, candidate_indices))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--glb-path", type=Path)
    parser.add_argument("--overview-candidate", default="blindspot-1",
                        help="Candidate reconstruction to show in the overview images")
    args = parser.parse_args()
    if not args.object_id.strip() or args.object_id in (".", "..") or "/" in args.object_id or "\\" in args.object_id:
        parser.error("Object ID must be a nonempty single directory name")
    glb = (args.glb_path or ROOT / "objects" / f"{args.object_id}.glb").expanduser().resolve()
    if not glb.is_file() or glb.suffix.lower() != ".glb":
        raise FileNotFoundError(f"GLB not found: {glb}")
    config = json.loads((DEMO / "config.json").read_text(encoding="utf-8"))
    run_id = config["run_id"]
    if not isinstance(run_id, str) or not run_id.strip() or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a nonempty single directory name")
    training = demo_training_config(config, json.loads((ROOT / "training_config.json").read_text(encoding="utf-8")))
    experiment = orchestrator.load_experiments(config["experiment_id"])[0]
    object_runs = DEMO / "runs" / args.object_id
    directory = object_runs / run_id
    if object_runs.is_symlink() or directory.is_symlink():
        raise ValueError(f"Run directory must not be a symlink: {directory}")
    spec = {
        "object_id": args.object_id, "glb": str(glb), "glb_sha256": sha256(glb),
        "config": {key: value for key, value in config.items() if key != "training"},
        "experiment": experiment, "training": training,
        "rendering": "cycles-cuda",
    }
    spec_path = directory / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(f"Run inputs changed; choose a new run_id in config.json: {directory}")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Nonempty run has no specification: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    blender = find_blender()
    object_dir = directory / "views" / args.object_id
    manifests = object_dir / "global_transforms"
    metadata_path = object_dir / "metadata.json"
    if not metadata_path.is_file():
        if (directory / "reconstructions").exists() or (directory / "results").exists():
            raise RuntimeError("Incomplete corpus has reconstruction data; refusing to overwrite it")
        subprocess.run([
            blender, "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", str(DEMO / "render_corpus.py"), "--",
            "--object-id", args.object_id, "--glb-path", str(glb),
            "--run-dir", str(directory), "--lattice-size", str(config["lattice_size"]),
        ], check=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train = json.loads((manifests / "transforms_train.json").read_text(encoding="utf-8"))
    test = json.loads((manifests / "transforms_test.json").read_text(encoding="utf-8"))
    if metadata["object_id"] != args.object_id or metadata["lattice_size"] != config["lattice_size"]:
        raise ValueError("Corpus metadata does not match the run")
    if len(train["frames"]) != config["lattice_size"] or len(test["frames"]) != 8:
        raise ValueError("Incomplete camera manifests")
    for frame in train["frames"] + test["frames"]:
        image = (object_dir / f"{frame['file_path']}.png").resolve()
        if not image.is_relative_to(object_dir.resolve()) or not image.is_file():
            raise FileNotFoundError(f"Invalid or missing corpus image: {image}")

    ids = config["candidate_ids"]
    if args.overview_candidate not in ids:
        raise ValueError(f"Overview candidate {args.overview_candidate!r} is not in config candidate_ids")
    base, jobs, selection = select_jobs(experiment, args.object_id, object_dir, train, ids)
    selection_path = directory / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    images = [directory / f"pipeline_{i}.png" for i in range(1, 5)]
    gifs = [directory / f"pipeline_{i}.gif" for i in range(1, 5)]
    if not all(path.is_file() for path in images) or not all(path.is_file() for path in gifs):
        with tempfile.TemporaryDirectory(prefix="pipeline-overview-", dir=directory) as temp_dir:
            command = [
                blender, "--background", str(object_dir / "rendered_scene.blend"),
                "--python-exit-code", "1", "--python", str(DEMO / "render_overview.py"),
                "--", "--run-dir", str(directory), "--object-id", args.object_id,
            ]
            if not all(path.is_file() for path in gifs):
                command.extend(["--gif-frames-dir", temp_dir])
                if all(path.is_file() for path in images):
                    command.append("--skip-images")
            subprocess.run(command, check=True)
            for number, gif in enumerate(gifs, 1):
                if not gif.is_file():
                    paths = [Path(temp_dir) / f"pipeline_{number}_{index:03d}.png" for index in range(FRAMES)]
                    save_gif_from_files(paths, gif)
    if not all(path.is_file() for path in images):
        raise RuntimeError("Overview rendering did not produce all four images")
    if not all(path.is_file() for path in gifs):
        raise RuntimeError("Overview rendering did not produce all four GIFs")

    prepare_points(object_dir, metadata, directory)
    orchestrator.CONFIG.update({
        "run_root": directory, "views_root": directory / "views",
        "reconstructions_root": directory / "reconstructions",
        "results_root": directory / "results", "base_cloud_ply": None,
    })
    orchestrator.PYTHON = sys.executable
    orchestrator.TRAINING_CONFIG.update({key: value for key, value in training.items()
                                         if key not in ("fastgs_dir", "python")})
    orchestrator.TRAINER = DEMO / "train_demo.py"
    options = orchestrator.build_parser().parse_args(["--use-candidate-base", "--retries", "0"])
    previous_spec = os.environ.get("PIPELINE_DEMO_RUN_SPEC")
    os.environ["PIPELINE_DEMO_RUN_SPEC"] = str(spec_path)
    try:
        orchestrator.run_experiment(args.object_id, object_dir, experiment["id"], base, jobs, options)
    finally:
        if previous_spec is None:
            del os.environ["PIPELINE_DEMO_RUN_SPEC"]
        else:
            os.environ["PIPELINE_DEMO_RUN_SPEC"] = previous_spec
    result_path = directory / "results" / args.object_id / experiment["id"] / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for name in ["base", *ids]:
        if result["reconstructions"].get(name, {}).get("status") != "complete" or name not in result.get("metrics", {}):
            raise RuntimeError(f"Reconstruction or evaluation incomplete: {name}")
    with tempfile.TemporaryDirectory(prefix="pipeline-reconstruction-") as temp_dir:
        for label, name in (("base", "base"), ("candidate", args.overview_candidate)):
            record = result["reconstructions"][name]
            model_dir = Path(record["reconstruction_path"])
            checkpoint = Path(record["checkpoint_path"])
            test_dir = model_dir / "test" / f"ours_{record['iterations']}" / "renders"
            for index in range(len(test["frames"])):
                if not (test_dir / f"{index:05d}.png").is_file():
                    raise FileNotFoundError(f"Missing evaluated test render: {test_dir / f'{index:05d}.png'}")
            transparent = Path(temp_dir) / f"{label}.png"
            subprocess.run([
                sys.executable, str(DEMO / "render_gaussians.py"),
                "--metadata", str(metadata_path), "--checkpoint", str(checkpoint),
                "--output", str(directory / f"pipeline_reconstruct_{label}.png"),
                "--transparent-output", str(transparent),
            ], check=True)
            transparent_frames = Path(temp_dir) / f"{label}-transparent"
            evaluation_frames = Path(temp_dir) / f"{label}-evaluation"
            transparent_frames.mkdir()
            evaluation_frames.mkdir()
            subprocess.run([
                sys.executable, str(DEMO / "render_gif.py"),
                "--metadata", str(metadata_path), "--checkpoint", str(checkpoint),
                "--blender", blender, "--transparent-frames", str(transparent_frames),
                "--output", str(directory / f"pipeline_reconstruct_{label}.gif"),
            ], check=True)
            subprocess.run([
                blender, "--background", str(object_dir / "rendered_scene.blend"),
                "--python-exit-code", "1", "--python", str(DEMO / "render_overview.py"),
                "--", "--run-dir", str(directory), "--object-id", args.object_id,
                "--reconstruction-image", str(transparent), "--test-renders", str(test_dir),
                "--reconstruction-frames", str(transparent_frames),
                "--gif-frames-dir", str(evaluation_frames),
                "--output", str(directory / f"pipeline_reconstruct_evaluate_{label}.png"),
            ], check=True)
            save_gif_from_files(
                [evaluation_frames / f"pipeline_4_{index:03d}.png" for index in range(FRAMES)],
                directory / f"pipeline_reconstruct_evaluate_{label}.gif",
            )
    print(f"Pipeline demo complete: {directory}", flush=True)


if __name__ == "__main__":
    main()
