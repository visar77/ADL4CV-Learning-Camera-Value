"""Score one candidate or rank an experiment's candidates using a saved head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import load_manifest, sample_key
from experiments import select_samples
from runtime import load_run, score_samples
from devices import resolve_device
from paths import resolve_reference
from predictor import GainEstimator, candidate_utility


def report_predictions(rows, gains, rank):
    utilities = candidate_utility(gains)
    for row, gain, utility in zip(rows, gains.tolist(), utilities.tolist(), strict=True):
        row.update(predicted_gain=gain, predicted_utility=utility)
    if rank:
        rows.sort(key=lambda row: row["predicted_utility"], reverse=True)
    print(json.dumps(rows, indent=2, allow_nan=False))
    return rows


def predict_request(args):
    request = json.loads(args.input.read_text())
    context = request.get("context", [])
    candidates = request.get("candidates", [])
    if args.candidate_id:
        candidates = [c for c in candidates if str(c.get("id")) == args.candidate_id]
    if args.num_samples is not None:
        if not 1 <= args.num_samples <= len(candidates):
            raise ValueError("--num-samples must be between one and the number of candidates.")
        candidates = candidates[:args.num_samples]
    if not context or not candidates:
        raise ValueError("Direct input needs nonempty context and candidates lists.")
    ids = [str(c.get("id", i)) for i, c in enumerate(candidates)]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique.")
    def pose(camera):
        value = camera.get("pose", camera.get("transform_matrix"))
        if value is None:
            raise ValueError("Each camera needs a pose (4x4 camera-to-world matrix).")
        return value
    def intrinsics(cameras):
        values = [c.get("intrinsics", request.get("intrinsics")) for c in cameras]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ValueError("Supply intrinsics for every camera or a common camera_angle_x.")
        return values
    images = [resolve_reference(c["image"], args.input.resolve().parent) for c in context]
    predictor = GainEstimator(args.checkpoint, backbone_checkpoint=args.lvsm_checkpoint,
                              device=args.device, candidate_batch_size=args.candidate_batch_size)
    values = predictor.predict(images, [pose(c) for c in context], [pose(c) for c in candidates],
                               context_intrinsics=intrinsics(context), candidate_intrinsics=intrinsics(candidates),
                               camera_angle_x=request.get("camera_angle_x"),
                               candidate_image_size=request.get("candidate_image_size"),
                               camera_convention=request.get("camera_convention", "blender"))
    rows = [{"candidate": identifier, "target": predictor.config["target"]} for identifier in ids]
    return report_predictions(rows, values, args.rank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset", type=Path, help="Offline labelled manifest")
    inputs.add_argument("--input", type=Path, help="Direct JSON request with context images/cameras and candidates")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--lvsm-checkpoint", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--scene-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    args.device = resolve_device(args.device)
    if args.input:
        if args.features:
            parser.error("--features is for offline manifests; direct inference prepares features automatically")
        return predict_request(args)
    checkpoint, model = load_run(args.checkpoint, args.device)
    samples = select_samples(load_manifest(args.dataset or Path(checkpoint["dataset_path"])),
                             count=args.num_samples, scene_id=args.scene_id, experiment_id=args.experiment_id,
                             duplicates=checkpoint["duplicates"])
    if args.candidate_id:
        samples = [s for s in samples if s.candidate_id == args.candidate_id or
                   any(source.endswith("\x1f" + args.candidate_id) for source in s.label_sources)]
    if not args.rank:
        if not 0 <= args.sample_index < len(samples):
            raise ValueError("Sample index is outside the selected candidates.")
        samples = [samples[args.sample_index]]
    if not samples:
        raise ValueError("No candidate samples selected.")
    if args.rank:
        from dataset import context_key
        if len({context_key(s) for s in samples}) != 1:
            raise ValueError("--rank requires one physical context; select --scene-id and --experiment-id.")
    values = score_samples(checkpoint, model, samples, args.features, args.device, args.lvsm_checkpoint).flatten()
    rows = [{"sample": sample_key(s), "candidate": s.candidate_id,
             "target": checkpoint["target"], "ground_truth_relative_gain": s.psnr_gain,
             "predicted_delta_db": value if checkpoint["target"] == "delta-db" else
                 value * abs(s.base_psnr) if s.base_psnr is not None else None}
            for s, value in zip(samples, values.tolist())]
    return report_predictions(rows, values, args.rank)


if __name__ == "__main__":
    main()
