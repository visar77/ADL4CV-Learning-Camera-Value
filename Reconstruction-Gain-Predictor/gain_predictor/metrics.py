"""Signed-gain regression errors and per-context camera-selection quality."""
from collections import defaultdict

import torch

from gain_predictor.data.batches import pose_features
from gain_predictor.data.records import DEFAULT_TARGET, context_key, input_key, target_values


def selection_metrics(samples, predictions, target=DEFAULT_TARGET):
    if not samples:
        return None
    predicted = torch.as_tensor(predictions, dtype=torch.float64).flatten()
    actual = target_values(samples, target).double().flatten()
    if len(predicted) != len(samples) or not torch.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match sample count.")
    error = predicted - actual
    result = {"samples": len(samples), "mae": error.abs().mean().item(),
              "rmse": error.square().mean().sqrt().item(), "target": target}
    is_psnr = all(s.metric == "PSNR" for s in samples)
    db_scales = torch.tensor([abs(s.base_psnr) if s.base_psnr is not None else float("nan") for s in samples])
    if target == "delta-db" and is_psnr:
        result["mae_db"] = result["mae"]
    elif target == "relative" and is_psnr and torch.isfinite(db_scales).all():
        result["mae_db"] = (error.abs() * db_scales).mean().item()
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[context_key(sample)].append(i)
    novelty = pose_features(samples)[:, 1].double()  # max-min positional novelty, like renderer blindspot.
    per_context = []
    def selected_gain(scores, gains):
        # Expected utility under uniform tie breaking; a constant predictor equals random.
        best = torch.isclose(scores, scores.max(), rtol=1e-7, atol=1e-9)
        return gains[best].mean().item()
    for key, indices in groups.items():
        # Repeated reconstructions are observations, not extra capture actions.
        actions = defaultdict(list)
        for i in indices:
            actions[input_key(samples[i])].append(i)
        ids = list(actions.values())
        gains = torch.tensor([actual[v].mean().item() for v in ids], dtype=torch.float64)
        scores = torch.tensor([predicted[v].mean().item() for v in ids], dtype=torch.float64)
        blindspot = torch.tensor([novelty[v].mean().item() for v in ids], dtype=torch.float64)
        picked = selected_gain(scores, gains)
        pair_scores = []
        for i in range(len(gains)):
            for j in range(i):
                delta = float(gains[i] - gains[j])
                if abs(delta) <= 1e-9:
                    continue
                diff = float(scores[i] - scores[j])
                pair_scores.append(.5 if abs(diff) <= 1e-9 else float(delta * diff > 0))
        per_context.append({"context": key, "scene": samples[indices[0]].scene_id,
                            "candidates": len(gains), "selected_gain": picked,
                            "oracle_gain": gains.max().item(), "regret": max(0., gains.max().item()-picked),
                            "random_gain": gains.mean().item(),
                            "blindspot_gain": selected_gain(blindspot, gains),
                            "pairwise_accuracy": sum(pair_scores)/len(pair_scores) if pair_scores else None})
    # Single-action contexts carry no ranking evidence.
    eligible = [g for g in per_context if g["candidates"] >= 2]
    names = ("selected_gain", "oracle_gain", "regret", "random_gain", "blindspot_gain", "pairwise_accuracy")
    scenes = defaultdict(list)
    for group in eligible:
        scenes[group["scene"]].append(group)
    scene_metrics = {}
    for scene, values in scenes.items():
        scene_metrics[scene] = {name: sum(g[name] for g in values if g[name] is not None) /
                              sum(g[name] is not None for g in values)
                              if any(g[name] is not None for g in values) else None for name in names}
    result["ranking_contexts"] = len(eligible)
    result["ranking"] = {name: sum(v[name] for v in scene_metrics.values() if v[name] is not None) /
                         sum(v[name] is not None for v in scene_metrics.values())
                         if any(v[name] is not None for v in scene_metrics.values()) else None for name in names}
    result["per_scene"] = scene_metrics
    result["per_context"] = per_context
    return result
