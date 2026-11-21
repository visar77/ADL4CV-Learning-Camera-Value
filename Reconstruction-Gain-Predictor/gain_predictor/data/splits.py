"""Select records and restore reproducible, disjoint context or scene splits."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import random

from .records import context_key, deduplicate_samples, digest, sample_key
from gain_predictor.artifacts import image_hashes


def select_samples(samples, count=None, scene_id=None, experiment_id=None, duplicates="mean"):
    samples = [s for s in samples if (scene_id is None or s.scene_id == scene_id)
               and (experiment_id is None or s.experiment_id == experiment_id)]
    if duplicates == "mean":
        samples = deduplicate_samples(samples)
    if not samples:
        raise ValueError("No samples match the selection.")
    if count is None:
        return samples
    if not 1 <= count <= len(samples):
        raise ValueError(f"--num-samples must be between 1 and {len(samples)}.")
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.scene_id].append(sample)
    selected = []
    while len(selected) < count:
        for scene in sorted(grouped):
            if grouped[scene]:
                selected.append(grouped[scene].pop(0))
                if len(selected) == count:
                    return selected
    return selected


def make_splits(samples, mode="context", seed=42, validation_fraction=.1,
                test_fraction=.1, split_file=None):
    if mode not in {"overfit", "context", "scene"}:
        raise ValueError(f"Unknown experiment mode {mode}")
    if mode == "overfit":
        if split_file:
            raise ValueError("--split-file cannot be combined with overfit mode.")
        # There is one explicitly labelled fit report, no pretend validation/test.
        groups = {"train": samples, "validation": [], "test": []}
        assignments = {"train": ["overfit"], "validation": [], "test": []}
    else:
        if not (0 <= validation_fraction < 1 and 0 <= test_fraction < 1
                and validation_fraction + test_fraction < 1):
            raise ValueError("Split fractions must be nonnegative and sum to less than 1.")
        unit = (lambda s: s.scene_id) if mode == "scene" else context_key
        units = sorted({unit(s) for s in samples})
        if split_file:
            raw = json.loads(Path(split_file).read_text())
            assignments = {name: list(raw.get(name, [])) for name in ("train", "validation", "test")}
            flat = [v for values in assignments.values() for v in values]
            if len(flat) != len(set(flat)) or set(flat) != set(units):
                raise ValueError("Split file must assign each scene/context unit exactly once.")
        else:
            random.Random(seed).shuffle(units)
            nv = max(1, round(len(units) * validation_fraction)) if validation_fraction else 0
            nt = max(1, round(len(units) * test_fraction)) if test_fraction else 0
            if nv + nt >= len(units):
                raise ValueError(f"Not enough {mode} groups for requested splits ({len(units)}). "
                                 "Use --mode overfit, reduce split fractions, or --test-fraction 0.")
            assignments = {"train": units[:len(units)-nv-nt],
                           "validation": units[len(units)-nv-nt:len(units)-nt] if nv else [],
                           "test": units[len(units)-nt:] if nt else []}
        groups = {name: [s for s in samples if unit(s) in values] for name, values in assignments.items()}
        # Physical contexts must not cross boundaries even if scene/experiment IDs differ.
        context_sets = {name: {context_key(s) for s in group} for name, group in groups.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if context_sets[a] & context_sets[b]:
                raise ValueError("A physical context appears in multiple splits.")
    if not groups["train"]:
        raise ValueError("Training split is empty.")
    metadata = {"mode": mode, "units": assignments,
                "sample_ids": {name: [sample_key(s) for s in group] for name, group in groups.items()}}
    return groups, metadata


def observation_signatures(samples):
    """Validate observations and image contents without binding them to a machine path."""
    hashes = image_hashes(samples)
    signatures = {}
    for sample in samples:
        record = sample.to_json()
        record.pop("context_image_paths")
        record.pop("source_results")
        record.pop("path_base", None)
        record["image_contents"] = [hashes[str(p)] for p in sample.image_paths]
        signatures[sample_key(sample)] = digest(record)
    return signatures


def restored_split(samples, checkpoint, split):
    """Restore exact observations (including duplicate averaging) from a run."""
    selection = checkpoint.get("selection", {})
    samples = select_samples(samples, scene_id=selection.get("scene_id"),
                             experiment_id=selection.get("experiment_id"),
                             duplicates=checkpoint.get("duplicates", "mean"))
    by_id = {sample_key(s): s for s in samples}
    ids = checkpoint["splits"]["sample_ids"][split]
    if any(i not in by_id for i in ids):
        raise ValueError("Manifest is missing samples saved in this run's split.")
    selected = [by_id[i] for i in ids]
    # IDs are insufficient: labels and cameras can change while their names stay fixed.
    expected = checkpoint.get("sample_signatures", {})
    actual = (observation_signatures(selected) if checkpoint.get("signature_mode") == "content-v1"
              else {sample_key(s): digest(s.to_json()) for s in selected})
    if any(expected.get(key) != value for key, value in actual.items()):
        raise ValueError("Manifest observations changed since training; use the original manifest.")
    return selected
