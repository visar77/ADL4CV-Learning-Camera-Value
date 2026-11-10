"""Run a trained PSNR checkpoint on a preprocessed sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import GainSample, collate_gain_samples, load_processed_dataset
from model import ContextPoseCNN, LVSMGainPredictor


def _select(
    samples: list[GainSample],
    index: int,
    scene_id: str | None,
    experiment_id: str | None,
    candidate_id: str | None,
) -> ProcessedSample:
    selected = [
        sample
        for sample in samples
        if (scene_id is None or sample.scene_id == scene_id)
        and (experiment_id is None or sample.experiment_id == experiment_id)
        and (candidate_id is None or sample.candidate_id == candidate_id)
    ]
    if index < 0 or index >= len(selected):
        raise IndexError(f"--sample-index {index} is outside 0..{len(selected) - 1}")
    return selected[index]


def predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model_type = checkpoint.get("model_type")
    if model_type == "toy":
        model = ContextPoseCNN(image_size=int(checkpoint.get("image_size", 32)))
    elif model_type == "lvsm":
        if args.lvsm_checkpoint is None:
            raise ValueError("--lvsm-checkpoint is required for an LVSM checkpoint.")
        model = LVSMGainPredictor(
            args.lvsm_checkpoint, image_size=int(checkpoint.get("image_size", 512))
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}")
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    metadata, samples = load_processed_dataset(args.dataset)
    sample = _select(
        samples, args.sample_index, args.scene_id, args.experiment_id, args.candidate_id
    )
    default_image_size = 32 if model_type == "toy" else 512
    batch = collate_gain_samples(
        [sample], int(checkpoint.get("image_size", default_image_size))
    )
    with torch.no_grad():
        normalized = model(
            batch["images"].to(args.device),
            batch["context_poses"].to(args.device),
            batch["candidate_pose"].to(args.device),
            batch["mask"].to(args.device),
        ).item()
    prediction = normalized * float(checkpoint["target_std"]) + float(
        checkpoint["target_mean"]
    )
    predicted_psnr = sample.base_psnr * (1.0 + prediction)
    print(f"model image size:       {checkpoint.get('image_size', default_image_size)}")
    print(f"scene:                  {sample.scene_id}")
    print(f"experiment/candidate:   {sample.experiment_id}/{sample.candidate_id}")
    print(f"context views:          {len(sample.context_image_paths)}")
    print(f"ground-truth rel gain:  {sample.psnr_gain:.6f}")
    print(f"predicted rel gain:     {prediction:.6f}")
    print(f"absolute error:         {abs(prediction - sample.psnr_gain):.6f}")
    print(f"base PSNR:              {sample.base_psnr:.6f}")
    print(f"ground-truth PSNR:      {sample.candidate_psnr:.6f}")
    print(f"predicted candidate:    {predicted_psnr:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lvsm-checkpoint", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--scene-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
