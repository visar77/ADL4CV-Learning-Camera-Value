"""Create reproducible Gaussian exploration points without changing the base PLY."""

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_options(count, jitter, opacity, scale):
    if count < 0:
        raise ValueError("extra-gaussians must be nonnegative")
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError("extra-gaussian-jitter must be finite and nonnegative")
    if not math.isfinite(opacity) or not 0 < opacity < 1:
        raise ValueError("extra-gaussian-opacity must be between 0 and 1")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("extra-gaussian-scale must be finite and positive")


def augment_ply(source, destination, count, seed=77, jitter=0.5, opacity=0.02, scale=0.5):
    """Add points sampled in the local covariance frames of random base splats.

    Base rows are preserved exactly. Extra points inherit appearance/orientation,
    use smaller spatial scales and low opacity, and do not use candidate/test RGB.
    """
    import numpy as np
    from plyfile import PlyData, PlyElement

    validate_options(count, jitter, opacity, scale)
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("The augmented initialization must not overwrite the base PLY")
    ply = PlyData.read(str(source))
    base = ply["vertex"].data
    required = {"x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)),
                *(f"rot_{i}" for i in range(4)), *(f"f_dc_{i}" for i in range(3))}
    missing = required - set(base.dtype.names or ())
    if missing or not len(base):
        raise ValueError(f"Expected a nonempty trained Gaussian PLY; missing fields: {sorted(missing)}")
    for name in required:
        if not np.isfinite(base[name]).all():
            raise ValueError(f"Nonfinite Gaussian field: {name}")
    generator = np.random.default_rng(seed)
    extra = base[generator.integers(0, len(base), size=count)].copy()
    if count:
        scales = np.exp(np.column_stack([extra[f"scale_{i}"] for i in range(3)]).astype(np.float64))
        if not np.isfinite(scales).all():
            raise ValueError("Invalid Gaussian spatial scales")
        quaternion = np.column_stack([extra[f"rot_{i}"] for i in range(4)]).astype(np.float64)
        lengths = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(lengths < 1e-12):
            raise ValueError("Invalid zero-length Gaussian rotation")
        quaternion /= lengths
        w, x, y, z = quaternion.T
        rotation = np.stack([
            1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
            2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
            2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y),
        ], axis=1).reshape(-1, 3, 3)
        local = generator.normal(size=(count, 3)) * scales * jitter
        offsets = np.einsum("nij,nj->ni", rotation, local)
        for i, name in enumerate(("x", "y", "z")):
            extra[name] += offsets[:, i]
            extra[f"scale_{i}"] += math.log(scale)
        extra["opacity"] = math.log(opacity / (1 - opacity))
    vertices = np.concatenate((base, extra))
    elements = [PlyElement.describe(vertices, "vertex") if element.name == "vertex" else element
                for element in ply.elements]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.ply")
    PlyData(elements, text=ply.text, byte_order=ply.byte_order,
            comments=ply.comments, obj_info=ply.obj_info).write(str(temporary))
    temporary.replace(destination)
    return {"original_gaussians": len(base), "extra_gaussians": count,
            "initial_gaussians": len(vertices)}


def prepare_initialization(source, root, count, seed=77, jitter=0.5, opacity=0.02,
                           scale=0.5, dry_run=False, python=None):
    """Content-address the derived PLY so every candidate/budget can share it."""
    source = Path(source)
    validate_options(count, jitter, opacity, scale)
    source_hash = sha256(source) if source.is_file() else "pending-dry-run"
    spec = {"base_sha256": source_hash, "count": count, "seed": seed, "jitter": jitter,
            "opacity": opacity, "scale": scale, "implementation_sha256": sha256(Path(__file__))}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    output = Path(root) / key / "point_cloud.ply"
    receipt_path = output.parent / "initialization.json"
    if output.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("spec") == spec and receipt.get("ply_sha256") == sha256(output):
            return output, receipt
    if dry_run:
        print(json.dumps({"prepare_initialization": str(output), "spec": spec}, indent=2))
        return output, {"spec": spec, "ply_sha256": "pending-dry-run", "wall_time_seconds": None}
    start = time.perf_counter()
    if python:
        result = subprocess.run([
            python, str(Path(__file__).resolve()), "--source", str(source), "--output", str(output),
            "--count", str(count), "--seed", str(seed), "--jitter", str(jitter),
            "--opacity", str(opacity), "--scale", str(scale),
        ], check=True, stdout=subprocess.PIPE, text=True)
        counts = json.loads(result.stdout)
    else:
        counts = augment_ply(source, output, count, seed, jitter, opacity, scale)
    receipt = {"spec": spec, **counts, "ply_sha256": sha256(output),
               "wall_time_seconds": time.perf_counter() - start}
    temporary = receipt_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    temporary.replace(receipt_path)
    return output, receipt


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--opacity", type=float, default=0.02)
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args(argv)
    print(json.dumps(augment_ply(args.source, args.output, args.count, args.seed,
                                args.jitter, args.opacity, args.scale)))


if __name__ == "__main__":
    main()
