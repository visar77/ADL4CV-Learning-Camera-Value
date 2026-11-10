"""Canonical renderer manifest generation and variable-view loading."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def sample_key(sample: GainSample) -> str:
    return f"{sample.scene_id}\x1f{sample.experiment_id}\x1f{sample.candidate_id}"


def _frames(path: Path) -> dict[int, tuple[str, tuple[float, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, tuple[str, ...]] = {}
    for index, frame in enumerate(payload["frames"]):
        matrix = tuple(
            float(value) for row in frame["transform_matrix"] for value in row
        )
        if len(matrix) != POSE_DIM or not all(math.isfinite(value) for value in matrix):
            raise ValueError(f"Invalid camera pose at frame {index} in {path}")
        result[index] = (str(frame["file_path"]), matrix)
    return result


def _records(path: Path) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        yield from payload
    else:
        raise ValueError(f"{path} must contain a JSON object or list")


def build_manifest(
    results_root: Path,
    views_root: Path,
    output: Path,
    *,
    limit: int | None = None,
) -> list[GainSample]:
    """Create PSNR-only samples without touching reconstruction assets."""
    samples: list[GainSample] = []
    for results_path in sorted(results_root.rglob("results.json")):
        for raw in _records(results_path):
            scene_id = str(raw["object_id"])
            experiment_id = str(raw["experiment_id"])
            base = raw["metrics"]["base"]["averages"]["PSNR"]
            indices = tuple(int(index) for index in raw["train_lattice_indices"])
            view_dir = views_root / scene_id
            transform_path = view_dir / "transforms_train.json"
            if not transform_path.is_file():
                transform_path = (
                    view_dir / "global_transforms" / "transforms_train.json"
                )
            transforms = _frames(transform_path)
            context_paths: list[str] = []
            context_poses: list[tuple[float, ...]] = []
            for index in indices:
                file_path, pose = transforms[index]
                image_path = view_dir / f"{file_path}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                context_paths.append(str(image_path))
                context_poses.append(pose)
            for candidate_id, candidate in raw.get("reconstructions", {}).items():
                if candidate_id == "base" or candidate.get("status") != "complete":
                    continue
                metric = raw["metrics"].get(candidate_id, {}).get("averages", {})
                candidate_index = candidate.get("candidate_lattice_index")
                if candidate_index is None or "PSNR" not in metric:
                    continue
                _, candidate_pose = transforms[int(candidate_index)]
                base_psnr = float(base)
                if not math.isfinite(base_psnr) or abs(base_psnr) < 1e-6:
                    raise ValueError(f"Invalid base PSNR in {results_path}")
                gain = (float(metric["PSNR"]) - base_psnr) / abs(base_psnr)
                if not math.isfinite(gain):
                    raise ValueError(f"Non-finite PSNR gain in {results_path}")
                samples.append(
                    GainSample(
                        scene_id=scene_id,
                        experiment_id=experiment_id,
                        candidate_id=str(candidate_id),
                        context_image_paths=tuple(context_paths),
                        context_poses=tuple(context_poses),
                        candidate_pose=candidate_pose,
                        psnr_gain=gain,
                        source_results=str(results_path),
                        base_psnr=base_psnr,
                        candidate_psnr=float(metric["PSNR"]),
                    )
                )
                if limit is not None and len(samples) >= limit:
                    break
            if limit is not None and len(samples) >= limit:
                break
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError("No complete PSNR candidate samples found.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(sample.to_json()) for sample in samples) + "\n",
        encoding="utf-8",
    )
    return samples


def load_manifest(path: Path) -> list[GainSample]:
    samples: list[GainSample] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            sample = GainSample(
                scene_id=str(raw["scene_id"]),
                experiment_id=str(raw["experiment_id"]),
                candidate_id=str(raw["candidate_id"]),
                context_image_paths=tuple(
                    str(value) for value in raw["context_image_paths"]
                ),
                context_poses=tuple(
                    tuple(float(value) for value in pose)
                    for pose in raw["context_poses"]
                ),
                candidate_pose=tuple(float(value) for value in raw["candidate_pose"]),
                psnr_gain=float(raw["psnr_gain"]),
                source_results=str(raw["source_results"]),
                base_psnr=(
                    float(raw["base_psnr"])
                    if raw.get("base_psnr") is not None
                    else None
                ),
                candidate_psnr=(
                    float(raw["candidate_psnr"])
                    if raw.get("candidate_psnr") is not None
                    else None
                ),
            )
            if not sample.context_image_paths or len(sample.context_image_paths) != len(
                sample.context_poses
            ):
                raise ValueError("context image/pose lengths differ")
            if (
                any(len(pose) != POSE_DIM for pose in sample.context_poses)
                or len(sample.candidate_pose) != POSE_DIM
            ):
                raise ValueError("poses must be flattened 4x4 matrices")
            if not math.isfinite(sample.psnr_gain):
                raise ValueError("psnr_gain must be finite")
            samples.append(sample)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid manifest record at line {line_number}: {error}"
            ) from error
    if not samples:
        raise ValueError(f"No records found in {path}")
    return samples


class GainDataset(Dataset[GainSample]):
    def __init__(
        self, samples: list[GainSample], image_size: int = DEFAULT_IMAGE_SIZE
    ) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GainSample:
        return self.samples[index]


@dataclass(frozen=True)
class ProcessedSample:
    scene_id: str
    experiment_id: str
    candidate_id: str
    images: Tensor
    context_poses: Tensor
    candidate_pose: Tensor
    psnr_gain: float
    base_psnr: float
    candidate_psnr: float


def preprocess_samples(
    results_root: Path,
    views_root: Path,
    output: Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    limit: int | None = None,
) -> list[GainSample]:
    """Write transparent JSONL records; images remain ordinary RGB files."""
    del image_size
    return build_manifest(results_root, views_root, output, limit=limit)


def load_processed_dataset(path: Path) -> tuple[dict[str, Any], list[GainSample]]:
    samples = load_manifest(path)
    return {"format": "reconstruction_gain_psnr_jsonl_v1"}, samples


class ProcessedGainDataset(Dataset[GainSample]):
    def __init__(self, samples: list[GainSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GainSample:
        return self.samples[index]


def collate_processed_samples(
    samples: list[GainSample],
) -> dict[str, Tensor | list[GainSample]]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_views = max(len(sample.context_image_paths) for sample in samples)
    batch_size = len(samples)
    loaded_images: list[list[Tensor]] = []
    max_height = max_width = 0
    for sample in samples:
        current: list[Tensor] = []
        for path in sample.context_image_paths:
            with Image.open(path) as image:
                tensor = torch.from_numpy(
                    np.array(image.convert("RGB"), copy=True)
                ).permute(2, 0, 1)
            current.append(tensor)
            max_height = max(max_height, tensor.shape[1])
            max_width = max(max_width, tensor.shape[2])
        loaded_images.append(current)
    images = torch.zeros(
        len(samples), max_views, 3, max_height, max_width, dtype=torch.uint8
    )
    poses = torch.zeros(batch_size, max_views, POSE_DIM)
    mask = torch.zeros(batch_size, max_views, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = len(sample.context_image_paths)
        for view_index, image in enumerate(loaded_images[index]):
            images[index, view_index, :, : image.shape[1], : image.shape[2]] = image
        poses[index, :count] = torch.tensor(sample.context_poses)
        mask[index, :count] = True
    return {
        "images": images,
        "context_poses": poses,
        "candidate_pose": torch.tensor([sample.candidate_pose for sample in samples]),
        "targets": torch.tensor([[sample.psnr_gain] for sample in samples]),
        "samples": samples,
        "mask": mask,
    }


def collate_gain_samples(
    samples: list[GainSample], image_size: int = DEFAULT_IMAGE_SIZE
) -> dict[str, Tensor | list[GainSample]]:
    del image_size
    return collate_processed_samples(samples)
