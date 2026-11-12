"""Selection, explicit experiment splits, and candidate-selection metrics."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from dataset import (GainSample, audit_samples, context_key, deduplicate_samples,
                     input_key, load_manifest, sample_key, collate_cameras)
from geometry import geometry_features


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
    from features import image_hashes
    from dataset import digest
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
    from dataset import digest
    actual = (observation_signatures(selected) if checkpoint.get("signature_mode") == "content-v1"
              else {sample_key(s): digest(s.to_json()) for s in selected})
    if any(expected.get(key) != value for key, value in actual.items()):
        raise ValueError("Manifest observations changed since training; use the original manifest.")
    return selected


def target_values(samples, target):
    if target == "relative":
        return torch.tensor([[s.psnr_gain] for s in samples], dtype=torch.float32)
    if any(s.base_psnr is None or s.candidate_psnr is None for s in samples):
        raise ValueError("Delta-dB targets require base_psnr and candidate_psnr.")
    return torch.tensor([[s.candidate_psnr - s.base_psnr] for s in samples], dtype=torch.float32)


def pose_features(samples):
    batch = collate_cameras(samples)
    return geometry_features(batch["context_poses"], batch["candidate_pose"], batch["mask"])


def selection_metrics(samples, predictions, target="relative"):
    if not samples:
        return None
    predicted = torch.as_tensor(predictions, dtype=torch.float64).flatten()
    actual = target_values(samples, target).double().flatten()
    if len(predicted) != len(samples) or not torch.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match sample count.")
    error = predicted - actual
    result = {"samples": len(samples), "mae": error.abs().mean().item(),
              "rmse": error.square().mean().sqrt().item(), "target": target}
    db_scales = torch.tensor([abs(s.base_psnr) if s.base_psnr is not None else float("nan") for s in samples])
    if target == "delta-db":
        result["mae_db"] = result["mae"]
    elif torch.isfinite(db_scales).all():
        result["mae_db"] = (error.abs() * db_scales).mean().item()
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[context_key(sample)].append(i)
    novelty = pose_features(samples)[:, 1].double()  # max-min positional novelty, like renderer blindspot.
    per_context = []
    def selected_gain(scores, gains):
        # Expected utility under uniform tie breaking; a constant predictor equals random.
        best = torch.isclose(scores, scores.max(), rtol=1e-7, atol=1e-9)
        return gains[best].mean().item()
    for key, indices in groups.items():
        # Repeated reconstructions are observations, not extra capture actions.
        actions = defaultdict(list)
        for i in indices:
            actions[input_key(samples[i])].append(i)
        ids = list(actions.values())
        gains = torch.tensor([actual[v].mean().item() for v in ids], dtype=torch.float64)
        scores = torch.tensor([predicted[v].mean().item() for v in ids], dtype=torch.float64)
        blindspot = torch.tensor([novelty[v].mean().item() for v in ids], dtype=torch.float64)
        picked = selected_gain(scores, gains)
        pair_scores = []
        for i in range(len(gains)):
            for j in range(i):
                delta = float(gains[i] - gains[j])
                if abs(delta) <= 1e-9:
                    continue
                diff = float(scores[i] - scores[j])
                pair_scores.append(.5 if abs(diff) <= 1e-9 else float(delta * diff > 0))
        per_context.append({"context": key, "scene": samples[indices[0]].scene_id,
                            "candidates": len(gains), "selected_gain": picked,
                            "oracle_gain": gains.max().item(), "regret": max(0., gains.max().item()-picked),
                            "random_gain": gains.mean().item(),
                            "blindspot_gain": selected_gain(blindspot, gains),
                            "pairwise_accuracy": sum(pair_scores)/len(pair_scores) if pair_scores else None})
    # Single-action contexts carry no ranking evidence.
    eligible = [g for g in per_context if g["candidates"] >= 2]
    names = ("selected_gain", "oracle_gain", "regret", "random_gain", "blindspot_gain", "pairwise_accuracy")
    scenes = defaultdict(list)
    for group in eligible:
        scenes[group["scene"]].append(group)
    scene_metrics = {}
    for scene, values in scenes.items():
        scene_metrics[scene] = {name: sum(g[name] for g in values if g[name] is not None) /
                              sum(g[name] is not None for g in values)
                              if any(g[name] is not None for g in values) else None for name in names}
    result["ranking_contexts"] = len(eligible)
    result["ranking"] = {name: sum(v[name] for v in scene_metrics.values() if v[name] is not None) /
                         sum(v[name] is not None for v in scene_metrics.values())
                         if any(v[name] is not None for v in scene_metrics.values()) else None for name in names}
    result["per_scene"] = scene_metrics
    result["per_context"] = per_context
    return result
