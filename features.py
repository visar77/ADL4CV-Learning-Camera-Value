"""Content-addressed frozen feature extraction, independent of the gain head."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from dataset import collate_cameras, collate_gain_samples, digest, load_manifest
from experiments import select_samples
from geometry import CAMERA_VERSION
from model import BACKBONES, LVSM_CHECKPOINT_NAME, LVSM_CHECKPOINT_URL
from paths import relative_path, resolve_reference
from devices import resolve_device, compute_precision, autocast_for

CACHE_VERSION = 1


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def image_hashes(samples):
    return {str(path): file_sha256(path) for path in sorted({p for s in samples for p in s.image_paths})}


def context_identity(sample, hashes):
    # Order matters for the first-camera canonical orientation.
    return digest([sample.camera_convention, sample.context_poses, sample.context_intrinsics,
                   [hashes[str(p)] for p in sample.image_paths]])


def feature_identity(sample, hashes):
    return digest([context_identity(sample, hashes), sample.candidate_pose, sample.candidate_intrinsics])


# Explicit input semantics version; file locations and CLI edits do not change the mathematics.
# Bump this when resizing, camera normalization, tokenization or backbone computation changes.
INPUT_IMPLEMENTATION = "calibrated-rgb-context-only-lvsm-v1"
LEGACY_IMPLEMENTATIONS = {"78f0c62c6de61dd77ab0be5576d547e7aa74cad8c93323e7557c8d7c350afcd0"}


def implementation_identity():
    return digest({"input_implementation": INPUT_IMPLEMENTATION, "camera_version": CAMERA_VERSION})


def compatible_implementation(spec):
    return (spec.get("implementation") in {implementation_identity(), *LEGACY_IMPLEMENTATIONS}
            and spec.get("camera_version") == CAMERA_VERSION)


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


def device_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def extract_features(samples, output, backbone_factory, backbone_spec, image_size=512,
                     candidate_batch_size=1, device="auto", precision="fp32", backbone_checkpoint=None):
    """Encode each context once; resume context/candidate files without loading a model.

    A factory can create any FeatureBackbone, including an adapter registered by
    another module. The spec must identify its checkpoint and configuration.
    """
    if candidate_batch_size < 1 or image_size < 1:
        raise ValueError("Image size and candidate batch size must be positive.")
    device = resolve_device(device)
    precision = compute_precision(precision, device)
    overall_start = time.perf_counter()
    if any(not s.context_intrinsics or s.candidate_intrinsics is None for s in samples):
        raise ValueError("Feature extraction requires calibrated intrinsics. Re-run preprocess.py.")
    root = Path(output)
    if root.suffix == ".pt":
        bundle_path, cache_root = root, root.with_suffix("")
    else:
        bundle_path, cache_root = root / "features.pt", root
    print(f"features: hashing source images for {len(samples)} records ({device}, {precision})", flush=True)
    hash_start = time.perf_counter()
    hashes = image_hashes(samples)
    hash_seconds = time.perf_counter() - hash_start
    spec = {**backbone_spec, "image_size": image_size, "precision": precision,
            "camera_version": CAMERA_VERSION, "implementation": implementation_identity(),
            "cache_version": CACHE_VERSION}
    identity = digest(spec)
    namespace = cache_root / identity
    groups = defaultdict(dict)
    for sample in samples:
        groups[context_identity(sample, hashes)][feature_identity(sample, hashes)] = sample
    backbone = None
    features = {}
    timing = {"load_seconds": 0., "encode_seconds": 0., "decode_seconds": 0.,
              "encoded_contexts": 0, "decoded_candidates": 0, "reused_candidates": 0,
              "hash_seconds": hash_seconds, "backbone_init_seconds": 0.}
    for number, (context_id, candidates) in enumerate(groups.items(), 1):
        pending = []
        for key, sample in candidates.items():
            path = namespace / "candidates" / f"{key}.pt"
            if path.is_file():
                features[key] = torch.load(path, map_location="cpu", weights_only=True)
                timing["reused_candidates"] += 1
            else:
                pending.append((key, sample))
        if not pending:
            continue
        if backbone is None:
            print("features: loading frozen backbone weights (once)", flush=True)
            init_start = time.perf_counter()
            backbone = backbone_factory().to(device).eval().requires_grad_(False)
            _sync(device)
            timing["backbone_init_seconds"] += time.perf_counter() - init_start
        context_path = namespace / "contexts" / f"{context_id}.pt"
        with torch.inference_mode(), autocast_for(precision, device):
            if context_path.is_file():
                context = device_batch(torch.load(context_path, map_location="cpu", weights_only=True), device)
            else:
                start = time.perf_counter()
                batch = device_batch(collate_gain_samples([pending[0][1]], image_size), device)
                _sync(device)
                timing["load_seconds"] += time.perf_counter() - start
                views = len(pending[0][1].context_poses)
                detail = (f"; {backbone_spec['latent_tokens'] + views*(image_size//8)**2:,} encoder tokens"
                          if backbone_spec.get("name") == "lvsm" else "")
                print(f"features: encoding context {number}/{len(groups)} ({views} views{detail})", flush=True)
                start = time.perf_counter()
                context = backbone.encode_context(batch["images"], batch["context_poses"],
                                                  batch["mask"], batch["context_intrinsics"])
                _sync(device)
                timing["encode_seconds"] += time.perf_counter() - start
                timing["encoded_contexts"] += 1
                atomic_save({k: v.detach().cpu() for k, v in context.items()}, context_path)
            print(f"features: decoding {len(pending)} uncached candidates for context {number}/{len(groups)} "
                  f"(batch size {candidate_batch_size})", flush=True)
            for offset in range(0, len(pending), candidate_batch_size):
                chunk = pending[offset:offset + candidate_batch_size]
                cameras = device_batch(collate_cameras([s for _, s in chunk]), device)
                start = time.perf_counter()
                values = backbone.decode_candidates(context, cameras["candidate_pose"][None],
                                                      cameras["candidate_intrinsics"][None])[0]
                _sync(device)
                timing["decode_seconds"] += time.perf_counter() - start
                if values.ndim != 2 or len(values) != len(chunk) or not torch.isfinite(values).all():
                    raise ValueError("Backbone must return finite features [B,Q,D].")
                for (key, _), value in zip(chunk, values):
                    features[key] = value.detach().float().cpu()
                    atomic_save(features[key], namespace / "candidates" / f"{key}.pt")
                timing["decoded_candidates"] += len(chunk)
        print(f"features: context {number}/{len(groups)}, {len(features)} candidates", flush=True)
    if not features:
        raise ValueError("No feature samples selected.")
    timing["total_seconds"] = time.perf_counter() - overall_start
    print(f"features: finished in {timing['total_seconds']:.1f}s; "
          f"{timing['encoded_contexts']} contexts encoded, {timing['decoded_candidates']} candidates decoded, "
          f"{timing['reused_candidates']} candidates reused", flush=True)
    bundle = {"format": "gain_features_v1", "identity": identity, "spec": spec,
              "features": features, "timing": timing}
    if backbone_checkpoint is not None:
        bundle["backbone_path"] = relative_path(backbone_checkpoint, bundle_path.parent)
    atomic_save(bundle, bundle_path)
    bundle_path.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2))
    return bundle_path


class FeatureStore:
    def __init__(self, path):
        self.path = Path(path)
        self.bundle = torch.load(self.path, map_location="cpu", weights_only=True)
        if self.bundle.get("format") != "gain_features_v1" or digest(self.bundle["spec"]) != self.bundle["identity"]:
            raise ValueError("Invalid feature-cache format or metadata fingerprint.")
        spec = self.bundle["spec"]
        if not compatible_implementation(spec):
            raise ValueError("Feature cache predates the current input implementation. Re-extract features.")
        self.identity = self.bundle["identity"]
        self.spec = spec

    @property
    def backbone_path(self):
        reference = self.bundle.get("backbone_path")
        return resolve_reference(reference, self.path.resolve().parent) if reference else None

    def matrix(self, samples):
        hashes = image_hashes(samples)
        keys = [feature_identity(s, hashes) for s in samples]
        missing = [k for k in keys if k not in self.bundle["features"]]
        if missing:
            raise ValueError(f"Cache is missing {len(missing)} inputs (images/cameras may have changed). Re-extract features.")
        values = torch.stack([self.bundle["features"][k] for k in keys]).float()
        if values.ndim != 2 or not torch.isfinite(values).all():
            raise ValueError("Cache contains invalid feature tensors.")
        return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="lvsm")
    parser.add_argument("--lvsm-checkpoint", type=Path, default=Path(__file__).parent / LVSM_CHECKPOINT_NAME)
    parser.add_argument("--lvsm-url", default=LVSM_CHECKPOINT_URL)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lvsm-latent-tokens", type=int, default=3072)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--scene-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    samples = select_samples(load_manifest(args.dataset), args.num_samples, args.scene_id, args.experiment_id)
    factory = lambda: BACKBONES[args.backbone](args.lvsm_checkpoint, args.image_size,
                                               args.lvsm_latent_tokens, args.lvsm_url)
    args.device = resolve_device(args.device)
    print("features: checking/hashing backbone checkpoint", flush=True)
    output_bundle = args.output if args.output.suffix == ".pt" else args.output / "features.pt"
    initialized = None
    cached_sha = None
    if not args.lvsm_checkpoint.is_file() and output_bundle.is_file():
        existing = FeatureStore(output_bundle)
        if (existing.backbone_path is not None
                and existing.backbone_path.resolve() == args.lvsm_checkpoint.resolve()
                and existing.spec.get("name") == args.backbone):
            cached_sha = existing.spec.get("checkpoint_sha256")
    if args.lvsm_checkpoint.is_file():
        checkpoint_sha = file_sha256(args.lvsm_checkpoint)
    elif cached_sha:
        # Complete feature caches are useful even when backbone weights are not deployed.
        checkpoint_sha = cached_sha
    else:
        initialized = factory()  # Downloads only when neither weights nor usable cache metadata exist.
        checkpoint_sha = file_sha256(args.lvsm_checkpoint)
    spec = {"name": args.backbone, "checkpoint_sha256": checkpoint_sha,
            "latent_tokens": args.lvsm_latent_tokens}
    def checked_factory():
        backbone = initialized if initialized is not None else factory()
        if file_sha256(args.lvsm_checkpoint) != checkpoint_sha:
            raise ValueError("Backbone weights do not match the resumed feature cache.")
        return backbone
    path = extract_features(samples, args.output, checked_factory,
                            spec, args.image_size, args.candidate_batch_size, args.device, args.precision, args.lvsm_checkpoint)
    print(f"Saved features to {path}")


if __name__ == "__main__":
    main()
