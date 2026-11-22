"""Render and reconstruct an isolated, object-configurable pipeline demo."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent.parent
sys.path.insert(0, str(ROOT))

import orchestrator
from demo.demo_pipeline.initialization import prepare_points
from gaussian_initialization import sha256
from renderer import find_blender


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
    args = parser.parse_args()
    if not args.object_id.strip() or args.object_id in (".", "..") or "/" in args.object_id or "\\" in args.object_id:
        parser.error("Object ID must be a nonempty single directory name")
    glb = (args.glb_path or ROOT / "objects" / f"{args.object_id}.glb").expanduser().resolve()
    if not glb.is_file() or glb.suffix.lower() != ".glb":
        raise FileNotFoundError(f"GLB not found: {glb}")
    config = json.loads((DEMO / "config.json").read_text(encoding="utf-8"))
    experiment = orchestrator.load_experiments(config["experiment_id"])[0]
    directory = DEMO / "runs" / args.object_id
    if directory.is_symlink():
        raise ValueError(f"Run directory must not be a symlink: {directory}")
    spec = {
        "object_id": args.object_id, "glb": str(glb), "glb_sha256": sha256(glb),
        "config": config, "experiment": experiment,
        "training": json.loads((ROOT / "training_config.json").read_text(encoding="utf-8")),
        "rendering": "cycles-cuda",
    }
    spec_path = directory / "run_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(f"Run inputs changed; choose a new object ID: {directory}")
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
    base, jobs, selection = select_jobs(experiment, args.object_id, object_dir, train, ids)
    selection_path = directory / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    images = [directory / f"pipeline_{i}.png" for i in range(1, 5)]
    if not all(path.is_file() for path in images):
        subprocess.run([
            blender, "--background", str(object_dir / "rendered_scene.blend"),
            "--python-exit-code", "1", "--python", str(DEMO / "render_overview.py"),
            "--", "--run-dir", str(directory), "--object-id", args.object_id,
        ], check=True)
    if not all(path.is_file() for path in images):
        raise RuntimeError("Overview rendering did not produce all four images")

    prepare_points(object_dir, metadata, directory)
    orchestrator.CONFIG.update({
        "run_root": directory, "views_root": directory / "views",
        "reconstructions_root": directory / "reconstructions",
        "results_root": directory / "results", "base_cloud_ply": None,
    })
    orchestrator.PYTHON = sys.executable
    options = orchestrator.build_parser().parse_args(["--use-candidate-base", "--retries", "0"])
    orchestrator.run_experiment(args.object_id, object_dir, experiment["id"], base, jobs, options)
    result_path = directory / "results" / args.object_id / experiment["id"] / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for name in ["base", *ids]:
        if result["reconstructions"].get(name, {}).get("status") != "complete" or name not in result.get("metrics", {}):
            raise RuntimeError(f"Reconstruction or evaluation incomplete: {name}")
    print(f"Pipeline demo complete: {directory}", flush=True)


if __name__ == "__main__":
    main()
