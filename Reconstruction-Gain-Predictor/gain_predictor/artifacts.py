"""Portable paths, content hashes, and atomic CPU-tensor saves."""
from __future__ import annotations

from pathlib import Path
import hashlib
import os

import torch

# Keep legacy relative paths and the default LVSM weights at the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def relative_path(path, directory):
    return Path(os.path.relpath(Path(path).resolve(), Path(directory).resolve())).as_posix()


def resolve_reference(path, directory):
    path = Path(path)
    return path if path.is_absolute() else Path(directory) / path


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def image_hashes(samples):
    return {str(path): file_sha256(path) for path in sorted({p for s in samples for p in s.image_paths})}


def cpu_artifact(value):
    """Serialize tensors on CPU, including nested model state and normalization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_artifact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_artifact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_artifact(item) for item in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    torch.save(cpu_artifact(value), temporary)
    temporary.replace(path)
