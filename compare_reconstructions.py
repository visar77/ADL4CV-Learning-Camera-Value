"""Compare matched reconstruction runs against one unchanged base per context."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

METRICS = ("PSNR", "SSIM", "LPIPS")


def metric_values(result, name):
    if name not in result.get("metrics", {}):
        raise ValueError(f"Missing metrics for {name}; run evaluate_metrics.py first")
    values = result["metrics"][name].get("averages", {})
    if any(key not in values or not math.isfinite(values[key]) for key in METRICS):
        raise ValueError(f"Missing/nonfinite PSNR, SSIM or LPIPS for {name}")
    if not result["metrics"][name].get("per_view"):
        raise ValueError(f"Missing per-view metrics for {name}")
    provenance = result.get("metrics_provenance", {}).get(name, {})
    if not provenance.get("checkpoint_sha256") or not provenance.get("test_data_sha256"):
        raise ValueError(f"Missing evaluation provenance for {name}; reevaluate with the current evaluator")
    expected = result["reconstructions"][name].get("iterations", 30000)
    if provenance.get("iteration") != expected:
        raise ValueError(f"Metrics for {name} do not match its final checkpoint iteration")
    return {key: float(values[key]) for key in METRICS}


def gains(metrics, baseline):
    return {key: (baseline[key] - metrics[key] if key == "LPIPS" else metrics[key] - baseline[key])
            for key in METRICS}


def timing(reconstruction):
    receipt = reconstruction.get("timing")
    if receipt is None:
        path = Path(reconstruction["reconstruction_path"]) / "training_run.json"
        receipt = json.loads(path.read_text()) if path.is_file() else {}
    seconds = receipt.get("wall_time_seconds") if receipt.get("status") == "complete" else None
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Training wall time must be finite and positive")
    return seconds, receipt.get("environment")


def candidates(result, allow_incomplete=False):
    output = {}
    for name, reconstruction in result.get("reconstructions", {}).items():
        index = reconstruction.get("candidate_lattice_index")
        if index is None or reconstruction.get("status") == "redundant" or name.startswith("redundancy-"):
            continue
        if reconstruction.get("status") != "complete":
            if allow_incomplete:
                continue
            raise ValueError(f"Candidate {name} is not complete")
        if allow_incomplete and name not in result.get("metrics", {}):
            continue
        if set(reconstruction["train_lattice_indices"]) != set(result["train_lattice_indices"]) | {index}:
            raise ValueError(f"Candidate {name} does not train on exactly the base views plus its candidate")
        if index in result["train_lattice_indices"]:
            raise ValueError(f"Non-redundant candidate {name} is already in the base")
        metric_values(result, name)
        if index in output:
            owner = output[index]
            if reconstruction["reconstruction_path"] != result["reconstructions"][owner]["reconstruction_path"]:
                raise ValueError(f"Multiple independent runs for candidate index {index}; use separate method IDs")
        else:
            output[index] = name
    return output


def rank_correlation(a, b):
    def ranks(values):
        return [sum(v < x for v in values) + (sum(v == x for v in values) - 1) / 2 for x in values]
    if len(a) < 2:
        return None
    a, b = ranks(a), ranks(b)
    av, bv = mean(a), mean(b)
    denominator = math.sqrt(sum((x-av)**2 for x in a) * sum((y-bv)**2 for y in b))
    return sum((x-av)*(y-bv) for x, y in zip(a, b)) / denominator if denominator else None


def summary(rows):
    if not rows:
        raise ValueError("No comparable candidate pairs")
    fresh_time = sum(row["reference_seconds"] for row in rows) if all(row["reference_seconds"] is not None for row in rows) else None
    method_time = sum(row["method_seconds"] for row in rows) if all(row["method_seconds"] is not None for row in rows) else None
    complete_timing = fresh_time is not None and method_time is not None
    fresh = [row["reference_gain"]["PSNR"] for row in rows]
    adapted = [row["method_gain"]["PSNR"] for row in rows]
    return {
        "paired_candidates": len(rows), "complete_timing": complete_timing,
        "reference_candidate_seconds": fresh_time, "method_candidate_seconds": method_time,
        "speedup": fresh_time / method_time if complete_timing else None,
        "time_saved_percent": 100 * (1-method_time/fresh_time) if complete_timing else None,
        "mean_reference_seconds": fresh_time / len(rows) if fresh_time is not None else None,
        "mean_method_seconds": method_time / len(rows) if method_time is not None else None,
        "mean_reference_metrics": {m: mean(r["reference_metrics"][m] for r in rows) for m in METRICS},
        "mean_method_metrics": {m: mean(r["method_metrics"][m] for r in rows) for m in METRICS},
        "mean_reference_gain": {m: mean(r["reference_gain"][m] for r in rows) for m in METRICS},
        "mean_method_gain": {m: mean(r["method_gain"][m] for r in rows) for m in METRICS},
        "mean_delta_gain": {m: mean(r["delta_gain"][m] for r in rows) for m in METRICS},
        "gain_mae": {m: mean(abs(r["delta_gain"][m]) for r in rows) for m in METRICS},
        "negative_psnr_gains_reference": sum(value < 0 for value in fresh),
        "negative_psnr_gains_method": sum(value < 0 for value in adapted),
        "psnr_gain_sign_agreement": mean((x >= 0) == (y >= 0) for x, y in zip(fresh, adapted)),
    }


def compare_context(reference, method, reference_id, method_id, allow_incomplete=False):
    if reference["train_lattice_indices"] != method["train_lattice_indices"]:
        raise ValueError("Base training cameras differ between methods")
    for result in (reference, method):
        if result.get("protocol", {}).get("gain_reference") != "original_base":
            raise ValueError("Comparison requires gain_reference=original_base for every method")
    baseline = metric_values(reference, "base")
    other_base = metric_values(method, "base")
    base_provenance = reference["metrics_provenance"]["base"]
    other_provenance = method["metrics_provenance"]["base"]
    if base_provenance["checkpoint_sha256"] != other_provenance["checkpoint_sha256"]:
        raise ValueError("Methods do not use the same original base PLY")
    if any(abs(baseline[m] - other_base[m]) > 1e-5 for m in METRICS):
        raise ValueError("The shared original base has inconsistent metrics")
    if base_provenance["test_data_sha256"] != other_provenance["test_data_sha256"]:
        raise ValueError("Held-out images/cameras differ between methods")
    if reference.get("protocol", {}).get("input_data_sha256") != method.get("protocol", {}).get("input_data_sha256"):
        raise ValueError("Training input fingerprints differ between methods")
    ref_candidates, new_candidates = candidates(reference, allow_incomplete), candidates(method, allow_incomplete)
    if ref_candidates.keys() != new_candidates.keys() and not allow_incomplete:
        raise ValueError(f"Candidate sets differ: reference={sorted(ref_candidates)}, method={sorted(new_candidates)}")
    rows = []
    for index in sorted(ref_candidates.keys() & new_candidates.keys()):
        ref_name, name = ref_candidates[index], new_candidates[index]
        old, new = reference["reconstructions"][ref_name], method["reconstructions"][name]
        if old["train_lattice_indices"] != new["train_lattice_indices"]:
            raise ValueError(f"Training cameras differ for candidate index {index}")
        for result, key in ((reference, ref_name), (method, name)):
            provenance = result["metrics_provenance"][key]
            if provenance["test_data_sha256"] != base_provenance["test_data_sha256"]:
                raise ValueError(f"Candidate {key} was evaluated on a different test corpus")
            if set(result["metrics"][key].get("per_view", {})) != set(reference["metrics"]["base"].get("per_view", {})):
                raise ValueError(f"Per-view metric sets differ for {key}")
        old_cameras = reference["metrics_provenance"][ref_name].get("training_cameras_sha256")
        new_cameras = method["metrics_provenance"][name].get("training_cameras_sha256")
        if not old_cameras or old_cameras != new_cameras:
            raise ValueError(f"Training camera provenance differs for candidate index {index}")
        ref_metrics, new_metrics = metric_values(reference, ref_name), metric_values(method, name)
        ref_gain, new_gain = gains(ref_metrics, baseline), gains(new_metrics, baseline)
        ref_seconds, ref_environment = timing(old)
        new_seconds, new_environment = timing(new)
        if ref_environment and new_environment:
            for key in ("gpu_name", "compute_capability", "torch_version", "cuda_version", "hostname"):
                if ref_environment.get(key) != new_environment.get(key):
                    raise ValueError(f"Timing environments differ ({key}); rerun methods on the same GPU environment")
        rows.append({"candidate": ref_name, "method_candidate": name, "candidate_lattice_index": index,
                     "reference_seconds": ref_seconds, "method_seconds": new_seconds,
                     "reference_metrics": ref_metrics, "method_metrics": new_metrics,
                     "reference_gain": ref_gain, "method_gain": new_gain,
                     "delta_gain": {m: new_gain[m] - ref_gain[m] for m in METRICS},
                     "timing_environment_verified": bool(ref_environment and new_environment)})
    aggregate = summary(rows)
    fresh_gains = [r["reference_gain"]["PSNR"] for r in rows]
    new_gains = [r["method_gain"]["PSNR"] for r in rows]
    aggregate["psnr_spearman"] = rank_correlation(fresh_gains, new_gains)
    best = max(new_gains)
    chosen = [r["reference_gain"]["PSNR"] for r in rows if r["method_gain"]["PSNR"] == best]
    aggregate["selected_view_regret_db"] = max(fresh_gains) - mean(chosen)
    base_seconds = timing(reference["reconstructions"]["base"])[0]
    preparation = (method.get("initialization") or {}).get("preparation") or {}
    preparation_seconds = preparation.get("wall_time_seconds", 0)
    control = method["reconstructions"].get("base_control")
    aggregate.update({"shared_base_seconds": base_seconds,
                      "initialization_preparation_seconds": preparation_seconds,
                      "base_control_seconds": timing(control)[0] if control else 0})
    if base_seconds is not None:
        fresh_seconds = aggregate["reference_candidate_seconds"]
        method_seconds = aggregate["method_candidate_seconds"]
        fresh_total = base_seconds + fresh_seconds if fresh_seconds is not None else None
        method_total = base_seconds + preparation_seconds + method_seconds if method_seconds is not None and preparation_seconds is not None else None
        aggregate["reference_total_with_base_seconds"] = fresh_total
        aggregate["method_total_with_base_seconds"] = method_total
        aggregate["production_speedup"] = fresh_total / method_total if fresh_total is not None and method_total else None
    return {"object_id": reference["object_id"], "experiment_id": reference["experiment_id"],
            "reference": reference_id, "method": method_id, "base_metrics": baseline,
            "original_base_sha256": base_provenance["checkpoint_sha256"],
            "missing_from_method": sorted(ref_candidates.keys() - new_candidates.keys()),
            "missing_from_reference": sorted(new_candidates.keys() - ref_candidates.keys()),
            "initialization": method.get("initialization"), "rows": rows, "summary": aggregate}


def build_report(results_root, reference_id, method_ids=None, allow_incomplete=False,
                 object_ids=None, experiment_ids=None, context_keys=None):
    root = Path(results_root)
    def load_contexts(method):
        records = {}
        for path in sorted((root / method).glob("*/*/results.json")):
            record = json.loads(path.read_text())
            key = (record["object_id"], record["experiment_id"])
            if context_keys is not None and key not in context_keys:
                continue
            if object_ids and key[0] not in object_ids:
                continue
            if experiment_ids and key[1] not in experiment_ids:
                continue
            records[key] = record
        return records
    reference = load_contexts(reference_id)
    if not reference:
        raise ValueError(f"No reference results found under {root / reference_id}")
    method_ids = method_ids or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != reference_id)
    comparisons, missing = [], []
    for method_id in method_ids:
        if method_id == reference_id:
            continue
        records = load_contexts(method_id)
        differences = reference.keys() ^ records.keys()
        if differences and not allow_incomplete:
            raise ValueError(f"Missing contexts in {method_id} or reference: {sorted(differences)}")
        missing.extend({"method": method_id, "object_id": k[0], "experiment_id": k[1]} for k in sorted(differences))
        for key in sorted(reference.keys() & records.keys()):
            try:
                comparisons.append(compare_context(reference[key], records[key], reference_id, method_id, allow_incomplete))
            except ValueError as error:
                raise ValueError(f"{key[0]}/{key[1]}/{method_id}: {error}") from error
    if not comparisons:
        raise ValueError("No matched method/context comparisons found")
    overall = {}
    for method_id in sorted({item["method"] for item in comparisons}):
        selected = [item for item in comparisons if item["method"] == method_id]
        overall[method_id] = {**summary([row for item in selected for row in item["rows"]]),
                              "contexts": len(selected),
                              "mean_selected_view_regret_db": mean(item["summary"]["selected_view_regret_db"] for item in selected)}
    return {"schema_version": 1, "reference": reference_id,
            "timing_definition": "training subprocess wall time; evaluation excluded",
            "gain_reference": "same unchanged original base PLY", "lpips_gain": "base minus candidate",
            "comparisons": comparisons, "overall": overall, "missing_contexts": missing}


def formatted(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def text_report(report):
    lines = ["Matched reconstruction benchmark", "Positive PSNR/SSIM/LPIPS delta means the method is better than fresh; LPIPS gain is a decrease.",
             "Times are training subprocess wall seconds (startup, loading, training and saving); evaluation excluded.", ""]
    for comparison in report["comparisons"]:
        lines.append(f"Object: {comparison['object_id']} | Experiment: {comparison['experiment_id']} | {comparison['method']} vs {comparison['reference']}")
        lines.append(f"Paired candidates: {len(comparison['rows'])} (redundancy controls and duplicate actions excluded); gains use the original base.")
        header = ["Candidate", "Fresh sec", "Method sec", "Method-fresh PSNR gain", "Method-fresh SSIM gain", "Method-fresh LPIPS gain"]
        values = [[r["candidate"], formatted(r["reference_seconds"], 2), formatted(r["method_seconds"], 2),
                   *(formatted(r["delta_gain"][m]) for m in METRICS)] for r in comparison["rows"]]
        s = comparison["summary"]
        values.append(["MEAN", formatted(s["mean_reference_seconds"], 2), formatted(s["mean_method_seconds"], 2),
                       *(formatted(s["mean_delta_gain"][m]) for m in METRICS)])
        widths = [max(len(row[i]) for row in [header, *values]) for i in range(len(header))]
        lines.append(" | ".join(value.ljust(widths[i]) for i, value in enumerate(header)))
        lines.append("-+-".join("-"*width for width in widths))
        lines.extend(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in values)
        lines.append(f"Candidate training wall time: {formatted(s['reference_candidate_seconds'], 2)}s -> {formatted(s['method_candidate_seconds'], 2)}s "
                     f"({formatted(s['time_saved_percent'], 1)}% saved; {formatted(s['speedup'], 2)}x speedup)")
        lines.append("Mean gain over original base (PSNR / SSIM / LPIPS): " + comparison["reference"] + " " +
                     " / ".join(formatted(s["mean_reference_gain"][m]) for m in METRICS) + "; " + comparison["method"] + " " +
                     " / ".join(formatted(s["mean_method_gain"][m]) for m in METRICS))
        lines.append(f"Negative PSNR gains: fresh {s['negative_psnr_gains_reference']}/{len(values)-1}; method {s['negative_psnr_gains_method']}/{len(values)-1}. "
                     f"PSNR gain MAE {formatted(s['gain_mae']['PSNR'])} dB; rank correlation {formatted(s['psnr_spearman'])}; selection regret {formatted(s['selected_view_regret_db'])} dB.")
        lines.append(f"Shared base: {formatted(s['shared_base_seconds'], 2)}s; initialization preparation: {formatted(s['initialization_preparation_seconds'], 2)}s; "
                     f"optional base-only control: {formatted(s['base_control_seconds'], 2)}s (separate from candidate totals).")
        preparation = (comparison.get("initialization") or {}).get("preparation")
        if preparation:
            lines.append(f"Initialization Gaussians: {preparation.get('original_gaussians', 'N/A')} base + "
                         f"{preparation.get('extra_gaussians', 'N/A')} extra = {preparation.get('initial_gaussians', 'N/A')} total.")
        if "reference_total_with_base_seconds" in s:
            lines.append(f"Production time including shared base and initialization: {formatted(s['reference_total_with_base_seconds'], 2)}s -> "
                         f"{formatted(s['method_total_with_base_seconds'], 2)}s ({formatted(s['production_speedup'], 2)}x speedup).")
        if not all(r["timing_environment_verified"] for r in comparison["rows"]):
            lines.append("Timing environment metadata unavailable for at least one pair; hardware comparability is unverified.")
        if comparison["missing_from_method"] or comparison["missing_from_reference"]:
            lines.append(f"INCOMPLETE: omitted indices method={comparison['missing_from_method']}, reference={comparison['missing_from_reference']}.")
        lines.append("")
    lines.append("All contexts (candidate-weighted means)")
    for method, s in report["overall"].items():
        lines.append(f"{method}: {s['contexts']} contexts, {s['paired_candidates']} pairs, {formatted(s['speedup'], 2)}x speedup, "
                     f"PSNR/SSIM/LPIPS gain delta " + " / ".join(formatted(s["mean_delta_gain"][m]) for m in METRICS))
    if report["missing_contexts"]:
        lines.append(f"INCOMPLETE: {len(report['missing_contexts'])} unmatched contexts omitted; see JSON.")
    return "\n".join(lines) + "\n"


def write_report(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    text = text_report(report)
    (output / "comparison.txt").write_text(text, encoding="utf-8")
    flat = []
    for context in report["comparisons"]:
        for row in context["rows"]:
            record = {key: context[key] for key in ("object_id", "experiment_id", "reference", "method")}
            record.update({key: row[key] for key in ("candidate", "method_candidate", "candidate_lattice_index", "reference_seconds", "method_seconds")})
            for group in ("reference_metrics", "method_metrics", "reference_gain", "method_gain", "delta_gain"):
                record.update({f"{group}_{metric}": row[group][metric] for metric in METRICS})
            flat.append(record)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="Root containing METHOD/OBJECT/EXPERIMENT/results.json")
    parser.add_argument("--reference", default="fresh-30k")
    parser.add_argument("--method", action="append")
    parser.add_argument("--object-id", action="append")
    parser.add_argument("--experiment-id", action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.results_root, args.reference, args.method, args.allow_incomplete,
                              args.object_id, args.experiment_id)
    except ValueError as error:
        parser.error(str(error))
    print(write_report(report, args.output_dir or args.results_root.parent / "report"), end="")


if __name__ == "__main__":
    main()
