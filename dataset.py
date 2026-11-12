"""Calibrated manifests, label audits, and variable-view RGB loading."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, fields, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from geometry import to_opencv
from paths import relative_path, resolve_reference

POSE_DIM = 16
DEFAULT_IMAGE_SIZE = 512


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
        result.append(replace(group[0], psnr_gain=sum(observations) / len(observations),
                              candidate_psnr=candidate, label_sources=sources,
                              observed_gains=observations))
    return result


def _records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    return payload if isinstance(payload, list) else [payload]


def _intrinsics(payload, frame, image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    values = {**payload, **frame}
    if "fl_x" in values:
        fx = float(values["fl_x"])
        fy = float(values.get("fl_y", fx))
    elif "camera_angle_x" in values:
        fx = width / (2 * math.tan(float(values["camera_angle_x"]) / 2))
        fy = (height / (2 * math.tan(float(values["camera_angle_y"]) / 2))
              if "camera_angle_y" in values else fx)
    else:
        raise ValueError(f"Missing calibrated intrinsics for {image_path}")
    return (fx / width, fy / height, float(values.get("cx", width / 2)) / width,
            float(values.get("cy", height / 2)) / height)


def build_manifest(results_root: Path, views_root: Path, output: Path, *, limit=None,
                   duplicates="mean") -> list[GainSample]:
    samples = []
    for results_path in sorted(results_root.rglob("results.json")):
        for raw in _records(results_path):
            scene_id = str(raw["object_id"])
            view_dir = views_root / scene_id
            # Lattice indices always refer to the global manifest, never a temporary subset.
            transforms_path = view_dir / "global_transforms" / "transforms_train.json"
            if not transforms_path.is_file():
                transforms_path = view_dir / "transforms_train.json"
            payload = json.loads(transforms_path.read_text())
            frames = payload["frames"]
            def camera(index):
                frame = frames[int(index)]
                image_path = view_dir / frame["file_path"]
                if not image_path.suffix:
                    image_path = image_path.with_suffix(".png")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                pose = tuple(float(v) for row in frame["transform_matrix"] for v in row)
                return relative_path(image_path, output.parent), pose, _intrinsics(payload, frame, image_path)
            context = [camera(i) for i in raw["train_lattice_indices"]]
            base = float(raw["metrics"]["base"]["averages"]["PSNR"])
            if not math.isfinite(base) or abs(base) < 1e-6:
                raise ValueError(f"Invalid base PSNR in {results_path}")
            for candidate_id, candidate in raw.get("reconstructions", {}).items():
                if candidate_id == "base" or candidate.get("status") not in {"complete", "redundant"}:
                    continue
                metric = raw["metrics"].get(candidate_id, {}).get("averages", {})
                index = candidate.get("candidate_lattice_index")
                if index is None or "PSNR" not in metric:
                    continue
                iterations = raw.get("metrics_iterations", {})
                base_iteration = iterations.get("base", raw.get("metrics_iteration"))
                candidate_iteration = iterations.get(candidate_id, raw.get("metrics_iteration"))
                if base_iteration != candidate_iteration:
                    raise ValueError(f"Mixed metric iterations in {results_path}: {candidate_id}")
                _, pose, intrinsics = camera(index)
                value = float(metric["PSNR"])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite candidate PSNR in {results_path}")
                sample = GainSample(scene_id, str(raw["experiment_id"]), str(candidate_id),
                                    tuple(c[0] for c in context), tuple(c[1] for c in context),
                                    pose, (value - base) / abs(base), relative_path(results_path, output.parent),
                                    base, value, tuple(c[2] for c in context), intrinsics,
                                    payload.get("camera_convention", "blender"),
                                    "redundancy_control" if candidate.get("status") == "redundant" else "retrained",
                                    path_base="manifest", path_root=output.resolve().parent)
                samples.append(sample)
    if not samples:
        raise ValueError("No complete PSNR candidate samples found.")
    report = audit_samples(samples)
    if duplicates == "mean":
        samples = deduplicate_samples(samples)
    elif duplicates != "keep":
        raise ValueError("duplicates must be mean or keep")
    if limit is not None:
        samples = samples[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(s.to_json()) + "\n" for s in samples))
    report["written_records"] = len(samples)
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
    return samples


def load_manifest(path: Path) -> list[GainSample]:
    samples = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            for key in ("context_poses", "context_intrinsics"):
                if key in raw:
                    raw[key] = tuple(tuple(float(v) for v in row) for row in raw[key])
            for key in ("candidate_pose", "candidate_intrinsics", "context_image_paths",
                        "label_sources", "observed_gains"):
                if raw.get(key) is not None:
                    raw[key] = tuple(raw[key])
            root = path.resolve().parent if raw.get("path_base") == "manifest" else Path(__file__).resolve().parent
            sample = GainSample(**raw, path_root=root)
            if sample.path_base not in {None, "manifest"}:
                raise ValueError("path_base must be manifest")
            if not sample.context_poses or len(sample.context_image_paths) != len(sample.context_poses):
                raise ValueError("context image/pose lengths differ")
            poses = (*sample.context_poses, sample.candidate_pose)
            if any(len(p) != 16 or not all(math.isfinite(v) for v in p) for p in poses):
                raise ValueError("poses must be finite flattened 4x4 matrices")
            if sample.camera_convention not in {"blender", "opencv"}:
                raise ValueError("camera_convention must be blender or opencv")
            if sample.context_intrinsics and len(sample.context_intrinsics) != len(sample.context_poses):
                raise ValueError("context intrinsics/pose lengths differ")
            intrinsics = (*sample.context_intrinsics, *((sample.candidate_intrinsics,) if sample.candidate_intrinsics else ()))
            if any(len(k) != 4 or not all(math.isfinite(v) for v in k) or min(k[:2]) <= 0 for k in intrinsics):
                raise ValueError("intrinsics must contain finite fx,fy,cx,cy with positive focal lengths")
            values = (sample.psnr_gain, *sample.observed_gains,
                      *((sample.base_psnr,) if sample.base_psnr is not None else ()),
                      *((sample.candidate_psnr,) if sample.candidate_psnr is not None else ()))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("labels must be finite")
            samples.append(sample)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid manifest record at line {line_number}: {error}") from error
    if not samples:
        raise ValueError(f"No records found in {path}")
    if len({sample_key(s) for s in samples}) != len(samples):
        raise ValueError("Manifest contains duplicate sample IDs.")
    return samples


class GainDataset(Dataset):
    def __init__(self, samples, image_size=DEFAULT_IMAGE_SIZE):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_cameras(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    views = max(len(s.context_poses) for s in samples)
    poses = torch.eye(4).flatten().repeat(len(samples), views, 1)
    mask = torch.zeros(len(samples), views, dtype=torch.bool)
    intrinsics = torch.tensor([1., 1., .5, .5]).repeat(len(samples), views, 1)
    query_intrinsics = torch.tensor([1., 1., .5, .5]).repeat(len(samples), 1)
    queries = []
    calibrated = True
    for i, sample in enumerate(samples):
        count = len(sample.context_poses)
        poses[i, :count] = to_opencv(torch.tensor(sample.context_poses).reshape(-1, 4, 4),
                                     sample.camera_convention).flatten(1)
        queries.append(to_opencv(torch.tensor(sample.candidate_pose).reshape(4, 4),
                                 sample.camera_convention).flatten())
        mask[i, :count] = True
        if sample.context_intrinsics and sample.candidate_intrinsics:
            intrinsics[i, :count] = torch.tensor(sample.context_intrinsics)
            query_intrinsics[i] = torch.tensor(sample.candidate_intrinsics)
        else:
            calibrated = False
    return {"context_poses": poses, "candidate_pose": torch.stack(queries), "mask": mask,
            "context_intrinsics": intrinsics if calibrated else None,
            "candidate_intrinsics": query_intrinsics if calibrated else None,
            "targets": torch.tensor([[s.psnr_gain] for s in samples]), "samples": samples}


@lru_cache(maxsize=64)
def _resized_image(path, image_size, modified_ns, file_size):
    """Bounded per-worker cache; changed files get a fresh cache key."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def collate_gain_samples(samples, image_size=DEFAULT_IMAGE_SIZE):
    if image_size < 1:
        raise ValueError("image_size must be positive")
    batch = collate_cameras(samples)
    images = torch.zeros(len(samples), batch["mask"].shape[1], 3, image_size, image_size, dtype=torch.uint8)
    for i, sample in enumerate(samples):
        for j, path in enumerate(sample.image_paths):
            stat = Path(path).stat()
            images[i, j] = _resized_image(str(path), image_size, stat.st_mtime_ns, stat.st_size)
    batch["images"] = images
    return batch


# Compatibility aliases for callers of the original manifest API.
ProcessedGainDataset = GainDataset
collate_processed_samples = collate_gain_samples


def preprocess_samples(results_root, views_root, output, *, image_size=DEFAULT_IMAGE_SIZE,
                       limit=None, duplicates="mean"):
    return build_manifest(results_root, views_root, output, limit=limit, duplicates=duplicates)


def load_processed_dataset(path):
    return {"format": "reconstruction_gain_psnr_jsonl_v2"}, load_manifest(path)


def relativize_manifest(path: Path):
    """Migrate paths in place, preserving labels, observations and provenance IDs."""
    samples = load_manifest(path)
    migrated = []
    for sample in samples:
        for image in sample.image_paths:
            if not image.is_file():
                raise FileNotFoundError(f"Cannot migrate missing image {image}. Regenerate from local renderer data.")
        source = resolve_reference(sample.source_results, sample.path_root)
        migrated.append(replace(sample,
                                context_image_paths=tuple(relative_path(p, path.parent) for p in sample.image_paths),
                                source_results=relative_path(source, path.parent),
                                path_base="manifest", path_root=path.resolve().parent))
    temporary = path.with_name(path.name + ".part")
    temporary.write_text("".join(json.dumps(s.to_json()) + "\n" for s in migrated))
    temporary.replace(path)
    path.with_suffix(".audit.json").write_text(json.dumps(audit_samples(migrated), indent=2))
    return migrated
