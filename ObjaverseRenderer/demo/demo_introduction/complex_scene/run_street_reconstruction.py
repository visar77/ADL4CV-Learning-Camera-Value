"""Reconstruct the street demo with its four base and two candidate views."""

import json
import math
import sys

import numpy as np
from plyfile import PlyData, PlyElement

from street_common import ROOT, load_street_config, run_dir, street_run_spec
import orchestrator


def prepare_points(object_dir, metadata, run_root):
    path = object_dir / "points3d.ply"
    if path.exists():
        return
    if (run_root / "reconstructions").exists() or (run_root / "results").exists():
        raise FileNotFoundError("Cannot replace the initialization of an existing demo run")
    center = np.asarray(metadata["center"])
    radius = metadata["bounding_radius"]
    xyz = np.random.default_rng(77).uniform(-1.5 * radius, 1.5 * radius, size=(100_000, 3)) + center
    vertices = np.empty(100_000, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, axis]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0
    for name in ("red", "green", "blue"):
        vertices[name] = 128
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def validate_existing_run(result, base, jobs):
    expected = {"base": base, **{job["id"]: job["train_indices"] for job in jobs}}
    for name, record in result["reconstructions"].items():
        if name not in expected or record.get("train_lattice_indices") != expected[name]:
            raise ValueError(f"Existing reconstruction {name} has a different camera selection")


def rotation_matrix(degrees):
    x, y, z = np.radians(degrees)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def validate_manifest(manifest, views, object_dir):
    frames = manifest["frames"]
    if [frame["view_id"] for frame in frames] != [view["id"] for view in views]:
        raise ValueError("Street manifest camera IDs do not match the configured views")
    for frame, view in zip(frames, views):
        pose = np.asarray(frame["transform_matrix"], dtype=float)
        position = np.asarray(view["position"], dtype=float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[:3, 3], position, atol=1e-5)
        ):
            raise ValueError(f"Camera pose does not match street view {view['id']}")
        if "rotation_degrees" in view:
            matches = np.allclose(pose[:3, :3], rotation_matrix(view["rotation_degrees"]), atol=1e-5)
        else:
            offset = position - np.asarray(view["target"], dtype=float)
            matches = np.allclose(pose[:3, 2], offset / np.linalg.norm(offset), atol=1e-5)
        if not matches or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f"Camera pose does not match street view {view['id']}")
        image = (object_dir / f"{frame['file_path']}.png").resolve()
        if not image.is_relative_to(object_dir.resolve()) or not image.is_file():
            raise FileNotFoundError(f"Missing or invalid image for street view {view['id']}: {image}")


def main():
    config = load_street_config()
    run_root = run_dir(config["object_id"])
    spec_path = run_root / "run_spec.json"
    if not spec_path.is_file() or json.loads(spec_path.read_text(encoding="utf-8")) != street_run_spec(config):
        raise ValueError("Street run specification is missing or changed; render a new run")
    object_dir = run_root / "views" / config["object_id"]
    metadata = json.loads((object_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["object_id"] != config["object_id"]:
        raise ValueError("Corpus object ID does not match the configured street run")
    global_dir = object_dir / "global_transforms"
    train = json.loads((global_dir / "transforms_train.json").read_text(encoding="utf-8"))
    test = json.loads((global_dir / "transforms_test.json").read_text(encoding="utf-8"))
    validate_manifest(train, config["base_views"] + config["candidate_views"], object_dir)
    validate_manifest(test, config["test_views"], object_dir)
    if train["camera_angle_x"] != test["camera_angle_x"]:
        raise ValueError("Street training and test camera fields of view differ")
    prepare_points(object_dir, metadata, run_root)

    frame_index = {frame["view_id"]: i for i, frame in enumerate(train["frames"])}
    base = [frame_index[view["id"]] for view in config["base_views"]]
    jobs = [
        {
            "id": view["id"],
            "strategy": "selected",
            "train_indices": [*base, frame_index[view["id"]]],
            "status": "pending",
        }
        for view in config["candidate_views"]
    ]
    result_path = run_root / "results" / config["object_id"] / config["experiment_id"] / "results.json"
    if result_path.exists():
        validate_existing_run(json.loads(result_path.read_text(encoding="utf-8")), base, jobs)
    orchestrator.CONFIG.update({
        "run_root": run_root,
        "objects_root": ROOT / "objects",
        "views_root": run_root / "views",
        "reconstructions_root": run_root / "reconstructions",
        "results_root": run_root / "results",
        "base_cloud_ply": None,
    })
    orchestrator.PYTHON = sys.executable
    args = orchestrator.build_parser().parse_args(["--use-candidate-base", "--retries", "0"])
    orchestrator.run_experiment(config["object_id"], object_dir, config["experiment_id"], base, jobs, args)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    base_psnr = result["metrics"]["base"]["averages"]["PSNR"]
    if not math.isfinite(base_psnr):
        raise ValueError("Nonfinite base PSNR")
    print(f"Base mean test PSNR: {base_psnr:.3f} dB")
    for job in jobs:
        score = result["metrics"][job["id"]]["averages"]["PSNR"]
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite PSNR for {job['id']}")
        print(f"{job['id']}: {score:.3f} dB (gain {score - base_psnr:+.3f} dB)")


if __name__ == "__main__":
    main()
