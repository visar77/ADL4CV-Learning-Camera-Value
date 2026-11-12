"""Select object/experiment matrices and fingerprint their immutable inputs."""

import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path

from reconstruction.datasets import discover_objects, load_experiments, select_ids
from reconstruction.io import file_sha256, load_json


@dataclass
class BenchmarkContext:
    object_id: str
    object_dir: Path
    train: dict
    experiment: dict
    fingerprint: str

    @property
    def key(self):
        return self.object_id, self.experiment["id"]

    def manifest_entry(self):
        return {"object_id": self.object_id, "experiment_id": self.experiment["id"],
                "input_data_sha256": self.fingerprint}


def input_fingerprint(object_dir, train, test):
    digest = hashlib.sha256(json.dumps({"train": train, "test": test}, sort_keys=True).encode())
    paths = sorted({frame["file_path"] for manifest in (train, test) for frame in manifest["frames"]})
    for relative in paths:
        image = object_dir / relative
        if not image.suffix:
            image = image.with_suffix(".png")
        digest.update(file_sha256(image).encode())
    return digest.hexdigest()


def fastgs_fingerprint(path):
    digest = hashlib.sha256()
    for relative in ("train.py", "scene/gaussian_model.py", "scene/__init__.py", "utils/fast_utils.py",
                     "arguments/__init__.py", "gaussian_renderer/__init__.py"):
        file = path / relative
        if not file.is_file():
            raise FileNotFoundError(f"FastGS file is missing: {file}")
        digest.update(relative.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def saved_context_keys(manifest, object_ids=None, experiment_ids=None):
    """Replay exactly the recorded contexts, optionally narrowed by CLI filters."""
    contexts = manifest.get("contexts", [])
    if not contexts:
        return None
    selected_objects = select_ids(object_ids, {c["object_id"] for c in contexts}, "object")
    selected_experiments = select_ids(experiment_ids, {c["experiment_id"] for c in contexts}, "experiment")
    keys = {(c["object_id"], c["experiment_id"]) for c in contexts
            if c["object_id"] in selected_objects and c["experiment_id"] in selected_experiments}
    if not keys:
        raise ValueError("No saved contexts match the selected objects and experiments")
    return keys


def select_contexts(args, config, context_keys=None):
    """Select the Cartesian product, or replay exact saved pairs for evaluation."""
    requested_objects = args.object_id if args.object_id is not None else config.get("objects")
    requested_experiments = args.experiment_id if args.experiment_id is not None else config.get("experiments")
    if context_keys is not None:
        requested_objects = sorted({key[0] for key in context_keys})
        requested_experiments = sorted({key[1] for key in context_keys})
    object_ids = select_ids(requested_objects, discover_objects(args.views_root), "object")
    experiments = load_experiments(args.experiments_dir, requested_experiments)
    pairs = ((object_id, experiment) for object_id, experiment in product(object_ids, experiments)
             if context_keys is None or (object_id, experiment["id"]) in context_keys)
    # Limit before fingerprinting, so unused corpora are neither loaded nor hashed.
    pairs = islice(pairs, args.max_contexts) if args.max_contexts is not None else pairs
    contexts, corpora = [], {}
    for object_id, original in pairs:
        if object_id not in corpora:
            object_dir = args.views_root / object_id
            train = load_json(object_dir / "global_transforms/transforms_train.json")
            test = load_json(object_dir / "global_transforms/transforms_test.json")
            corpora[object_id] = object_dir, train, input_fingerprint(object_dir, train, test)
        object_dir, train, fingerprint = corpora[object_id]
        experiment = copy.deepcopy(original)
        experiment["candidates"] = copy.deepcopy(config["candidates"])
        contexts.append(BenchmarkContext(object_id, object_dir, train, experiment, fingerprint))
    if not contexts:
        raise ValueError("No matching contexts; check --views-root, --object-ids, --experiments-dir and --experiment-ids")
    return contexts
