"""Runtime device/precision selection, independent of serialized artifacts."""
from __future__ import annotations

import contextlib
import sys
import torch


def resolve_device(value="auto"):
    if value is None or str(value) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda, and cuda:<index>.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and working driver, or use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable.")
    return str(device)


def compute_precision(requested, device):
    if requested not in {"fp32", "bf16"}:
        raise ValueError("Precision must be fp32 or bf16.")
    if requested == "bf16" and torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported():
        print("BF16 is unavailable on this GPU; using FP32 (small numerical differences are possible).", file=sys.stderr, flush=True)
        return "fp32"
    return requested


def autocast_for(precision, device):
    return (torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
            if precision == "bf16" else contextlib.nullcontext())
