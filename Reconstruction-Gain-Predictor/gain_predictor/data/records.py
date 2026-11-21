"""Manifest records, physical camera identities, labels, and duplicate audits."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import torch

from gain_predictor.artifacts import resolve_reference


DEFAULT_TARGET = "delta-db"


@dataclass(frozen=True)
class GainSample:
    scene_id: str
    experiment_id: str
    candidate_id: str
    context_image_paths: tuple[str, ...]
    context_poses: tuple[tuple[float, ...], ...]
    candidate_pose: tuple[float, ...]
    psnr_gain: float
    source_results: str
    base_psnr: float | None = None
    candidate_psnr: float | None = None
    context_intrinsics: tuple[tuple[float, ...], ...] = ()
    candidate_intrinsics: tuple[float, ...] | None = None
    camera_convention: str = "blender"
    label_kind: str = "retrained"
    label_sources: tuple[str, ...] = ()
    observed_gains: tuple[float, ...] = ()
    path_base: str | None = None
    path_root: Path | None = field(default=None, compare=False, repr=False)
    metric: str = "PSNR"
    base_metric: float | None = None
    candidate_metric: float | None = None

    @property
    def image_paths(self) -> tuple[Path, ...]:
        root = self.path_root or Path.cwd()
        return tuple(resolve_reference(path, root) for path in self.context_image_paths)

    def to_json(self) -> dict[str, Any]:
        record = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "path_root"}
        if self.path_base is None:
            record.pop("path_base")  # Preserve signatures of legacy records.
        return record


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_key(sample: GainSample) -> str:
    return f"{sample.scene_id}\x1f{sample.experiment_id}\x1f{sample.candidate_id}"


def context_key(sample: GainSample) -> str:
    """Physical context group, independent of experiment alias and view order."""
    intrinsics = sample.context_intrinsics or (None,) * len(sample.context_poses)
    views = sorted(zip(sample.context_image_paths, sample.context_poses, intrinsics))
    return digest([sample.scene_id, sample.camera_convention, views])


def input_key(sample: GainSample) -> str:
    return digest([context_key(sample), sample.candidate_pose, sample.candidate_intrinsics])


def audit_samples(samples: list[GainSample]) -> dict:
    groups = defaultdict(list)
    for sample in samples:
        groups[input_key(sample)].append(sample)
    duplicates = []
    for key, group in groups.items():
        observations = [v for s in group for v in (s.observed_gains or (s.psnr_gain,))]
        if len(observations) > 1:
            duplicates.append({"input_key": key, "sample_ids": [sample_key(s) for s in group],
                               "observations": len(observations),
                               "gain_range": max(observations) - min(observations),
                               "gain_std": float(np.std(observations))})
    return {"records": len(samples), "scenes": len({s.scene_id for s in samples}),
            "contexts": len({context_key(s) for s in samples}), "unique_inputs": len(groups),
            "redundancy_records": sum(s.label_kind == "redundancy_control" for s in samples),
            "negative_gains": sum(s.psnr_gain < 0 for s in samples), "duplicates": duplicates}


def deduplicate_samples(samples: list[GainSample]) -> list[GainSample]:
    """Average repeated observations, retaining their labels and provenance."""
    groups = defaultdict(list)
    for sample in samples:
        groups[input_key(sample)].append(sample)
    result = []
    for group in groups.values():
        observations = tuple(v for s in group for v in (s.observed_gains or (s.psnr_gain,)))
        sources = tuple(v for s in group for v in (s.label_sources or (sample_key(s),)))
        candidate_values = [s.candidate_psnr for s in group]
        candidate = (sum(candidate_values) / len(candidate_values)
                     if all(v is not None for v in candidate_values) else None)
        base_metric_values = [s.base_metric for s in group]
        base_metric = (sum(base_metric_values) / len(base_metric_values)
                       if all(v is not None for v in base_metric_values) else None)
        candidate_metric_values = [s.candidate_metric for s in group]
        candidate_metric = (sum(candidate_metric_values) / len(candidate_metric_values)
                            if all(v is not None for v in candidate_metric_values) else None)
        result.append(replace(group[0], psnr_gain=sum(observations) / len(observations),
                              candidate_psnr=candidate, base_metric=base_metric,
                              candidate_metric=candidate_metric, label_sources=sources,
                              observed_gains=observations))
    return result


def require_calibration(samples, manifest=None):
    missing = sum(not s.context_intrinsics or s.candidate_intrinsics is None for s in samples)
    if missing:
        name = str(manifest) if manifest is not None else "<existing-manifest.jsonl>"
        raise ValueError(
            f"{missing}/{len(samples)} records lack calibrated intrinsics. "
            f"Upgrade the existing manifest with: python prepare_data.py --upgrade {name}. "
            "This uses the camera metadata beside its existing images and preserves labels; "
            "--relativize only changes paths. Use --views-root to specify another existing camera-metadata root."
        )


def target_values(samples, target):
    if target == "relative":
        values = []
        for sample in samples:
            if sample.metric == "PSNR":
                values.append([sample.psnr_gain])
                continue
            if sample.base_metric is None or sample.candidate_metric is None:
                raise ValueError(f"Missing {sample.metric} scores for {sample.scene_id}/"
                                 f"{sample.experiment_id}/{sample.candidate_id}.")
            if abs(sample.base_metric) < 1e-6:
                raise ValueError(f"Cannot calculate relative {sample.metric} gain from a zero base score "
                                 f"for {sample.scene_id}/{sample.experiment_id}/{sample.candidate_id}.")
            direction = -1 if sample.metric == "LPIPS" else 1
            values.append([direction * (sample.candidate_metric - sample.base_metric)
                           / abs(sample.base_metric)])
        return torch.tensor(values, dtype=torch.float32)
    values = []
    for sample in samples:
        if sample.metric != "PSNR":
            if sample.base_metric is None or sample.candidate_metric is None:
                raise ValueError(f"Missing {sample.metric} scores for {sample.scene_id}/"
                                 f"{sample.experiment_id}/{sample.candidate_id}.")
            direction = -1 if sample.metric == "LPIPS" else 1
            values.append([direction * (sample.candidate_metric - sample.base_metric)])
            continue
        base = sample.base_psnr
        if base is None:
            raise ValueError(
                f"Cannot derive a dB gain for {sample.scene_id}/{sample.experiment_id}/{sample.candidate_id}: "
                "the manifest record has no base_psnr. Use --target relative to train from the "
                "existing labels, or recover the base PSNR from saved reconstruction results."
            )
        if sample.candidate_psnr is None:
            # Older records can still carry the relative gain and the base score.
            gain_db = sample.psnr_gain * abs(base)
        else:
            gain_db = sample.candidate_psnr - base
        values.append([gain_db])
    return torch.tensor(values, dtype=torch.float32)
