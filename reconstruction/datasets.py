"""Select experiment views and temporarily expose FastGS dataset manifests."""

from contextlib import contextmanager

from reconstruction.io import load_json, save_json
from sampling import select_indices


def load_experiment(path):
    experiment = load_json(path)
    experiment.setdefault("id", path.stem)
    return experiment


def write_manifest(path, camera_angle_x, frames):
    save_json(path, {"camera_angle_x": camera_angle_x, "frames": frames})


def make_dataset(object_dir, global_train, experiment, lattice):
    metadata_path = object_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    trajectory_elevation = float(metadata.get("trajectory_elevation_deg", 20.0))
    config = experiment["train"]
    indices = select_indices(
        config["strategy"],
        int(config["count"]),
        lattice,
        seed=int(config["seed"]),
        trajectory_elevation_deg=trajectory_elevation,
    )
    return (
        indices,
        [global_train["frames"][i]["file_path"] for i in indices],
        trajectory_elevation,
    )


def write_training_manifest(object_dir, global_train, train_indices):
    write_manifest(
        object_dir / "transforms_train.json",
        global_train["camera_angle_x"],
        [dict(global_train["frames"][i]) for i in train_indices],
    )


@contextmanager
def training_manifests(object_dir, global_train, train_indices):
    """Restore both root manifests even if training or manifest setup fails."""
    paths = [object_dir / f"transforms_{split}.json" for split in ("train", "test")]
    previous = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        write_training_manifest(object_dir, global_train, train_indices)
        paths[1].write_bytes(
            (object_dir / "global_transforms/transforms_test.json").read_bytes()
        )
        yield
    finally:
        for path, content in previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def select_ids(requested, available, kind):
    """Validate explicit selections and keep their first occurrence in order."""
    available = list(available)
    if requested is None:
        return sorted(available)
    selected = list(dict.fromkeys(requested))
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown {kind} IDs: {sorted(unknown)}")
    return selected


def load_experiments(directory, requested=None):
    experiments = {}
    for path in sorted(directory.glob("*.json")):
        experiment = load_experiment(path)
        identifier = experiment["id"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in (".", "..")
            or "/" in identifier
        ):
            raise ValueError(f"Invalid experiment ID in {path}: {identifier!r}")
        if identifier in experiments:
            raise ValueError(f"Duplicate experiment ID: {identifier}")
        experiments[identifier] = experiment
    identifiers = select_ids(requested, experiments, "experiment")
    if not identifiers:
        raise ValueError(f"No experiments found in {directory}")
    return [experiments[identifier] for identifier in identifiers]


def discover_objects(directory, rendered=True):
    if rendered:
        return sorted(
            path.name
            for path in directory.iterdir()
            if (path / "global_transforms/transforms_train.json").is_file()
        )
    return sorted(path.stem for path in directory.glob("*.glb"))
