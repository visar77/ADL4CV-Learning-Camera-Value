"""Cache frozen backbone embeddings for later gain-head training."""
from __future__ import annotations

from pathlib import Path
import argparse

from gain_predictor.artifacts import PROJECT_ROOT, file_sha256
from gain_predictor.data.manifest import load_manifest
from gain_predictor.data.splits import select_samples
from gain_predictor.devices import resolve_device
from gain_predictor.feature_cache import FeatureStore, extract_features
from gain_predictor.models import BACKBONES
from gain_predictor.models.lvsm import LVSM_CHECKPOINT_NAME, LVSM_CHECKPOINT_URL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="lvsm")
    parser.add_argument("--lvsm-checkpoint", type=Path, default=PROJECT_ROOT / LVSM_CHECKPOINT_NAME)
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
