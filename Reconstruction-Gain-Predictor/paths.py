"""Portable artifact references: paths are relative to the file owning them."""
from __future__ import annotations

import os
from pathlib import Path


def relative_path(path, directory):
    return Path(os.path.relpath(Path(path).resolve(), Path(directory).resolve())).as_posix()


def resolve_reference(path, directory):
    path = Path(path)
    return path if path.is_absolute() else Path(directory) / path
