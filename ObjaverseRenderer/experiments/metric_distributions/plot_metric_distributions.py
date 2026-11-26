#!/usr/bin/env python3
r"""Plot saved reconstruction metrics and signed, absolute gains; no GPU needed.

Run with the adjacent config.json, or select another with --config.

Requires NumPy and Matplotlib. Inputs are orchestrator result roots or exported
bundles containing results/<object>/<experiment>/results.json. Reconstruction-only
roots are inventoried, but cannot supply metrics. Source files are never modified.
"""

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("PSNR", "SSIM", "LPIPS")
FIELDS = METRICS + tuple(f"gain_{m}" for m in METRICS)
LABELS = ("PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓",
          "PSNR gain (dB) ↑", "SSIM gain ↑", "LPIPS gain ↑")
FORMULAS = {"gain_PSNR": "candidate PSNR - base PSNR",
            "gain_SSIM": "candidate SSIM - base SSIM",
            "gain_LPIPS": "base LPIPS - candidate LPIPS"}


def scores(value, location):
    result = {}
    for metric in METRICS:
        number = value["averages"][metric]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{location}: {metric} must be finite")
        result[metric] = float(number)
    return result


def discover(inputs):
    """Inventory expected contexts, including bundles with no metric files."""
    files, expected = set(), set()
    for root in inputs:
        root = root.expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Input does not exist: {root}")
        if root.is_file():
            files.add(root)
            continue
        search = root / "results" if (root / "results").is_dir() else root
        files.update(search.rglob("results.json"))
        recon = root / "reconstructions" if (root / "reconstructions").is_dir() else root
        for base in recon.glob("*/*/base"):
            if base.is_dir():
                expected.add((base.parent.parent.name, base.parent.name))
    return sorted(files), expected


def load_data(inputs, objects=None):
    files, expected = discover(inputs)
    selected = set(objects) if objects is not None else None
    available = {obj for obj, _ in expected}
    if selected is not None:
        expected = {context for context in expected if context[0] in selected}
    observations, bases, sources = [], [], []
    seen_contexts = set()
    missing_reasons = {}
    excluded = defaultdict(int)
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        obj = data.get("object_id", path.parent.parent.name)
        exp = data.get("experiment_id", path.parent.name)
        available.add(obj)
        if selected is not None and obj not in selected:
            continue
        if "reconstructions" not in data:
            raise ValueError(f"{path}: expected orchestrator results with reconstructions")
        context = (obj, exp)
        if context in seen_contexts:
            raise ValueError(f"Repeated context {context} in inputs. Supply each experimental run separately.")
        seen_contexts.add(context)
        expected.add(context)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        recon, metrics = data["reconstructions"], data.get("metrics", {})
        if "base" not in metrics or "base" not in recon:
            missing_reasons[context] = data.get("skip_reason", "Base reconstruction/metrics missing")
            excluded["contexts_without_base_metrics"] += 1
            continue
        base = scores(metrics["base"], f"{path}:base")
        base_indices = recon["base"]["train_lattice_indices"]
        base_set = set(base_indices)
        if len(base_set) != len(base_indices):
            raise ValueError(f"{path}: repeated base camera index")
        common = {"object": obj, "experiment": exp, "base_views": len(base_set),
                  "base_iterations": recon["base"].get("iterations"), "source": str(path)}
        bases.append({**common, **base})
        for name, item in recon.items():
            if name == "base":
                continue
            indices = item.get("train_lattice_indices", [])
            if name.startswith("redundancy") or item.get("status") == "redundant" or set(indices) == base_set:
                excluded["redundancy_controls"] += 1
                continue
            if item.get("status", "complete") != "complete" or name not in metrics:
                excluded["incomplete_or_missing_candidate_metrics"] += 1
                continue
            if not base_set < set(indices) or len(set(indices) - base_set) != 1 or len(indices) != len(base_set) + 1:
                raise ValueError(f"{path}:{name}: expected the same base plus exactly one new camera")
            provenance = data.get("metrics_provenance", {})
            bp, cp = provenance.get("base", {}), provenance.get(name, {})
            for key in ("test_data_sha256", "evaluator_sha256"):
                if bp.get(key) and cp.get(key) and bp[key] != cp[key]:
                    raise ValueError(f"{path}:{name}: base/candidate {key} mismatch")
            values = scores(metrics[name], f"{path}:{name}")
            row = {**common, "candidate": name, "candidate_iterations": item.get("iterations"),
                   "candidate_index": next(iter(set(indices) - base_set)),
                   "train_indices": sorted(indices), "provenance": cp, **values}
            for metric in METRICS:
                row[f"base_{metric}"] = base[metric]
                row[f"gain_{metric}"] = (values[metric] - base[metric]) * (-1 if metric == "LPIPS" else 1)
            observations.append(row)
    if selected is not None and selected - available:
        raise ValueError(f"Objects not found in inputs: {', '.join(sorted(selected - available))}")
    # Repeated training of one action contributes one mean, retaining all raw
    # observations and provenance separately. Never combine different budgets.
    grouped = defaultdict(list)
    for row in observations:
        grouped[(row["object"], row["experiment"], row["candidate_index"],
                 row["base_iterations"], row["candidate_iterations"],
                 row["provenance"].get("evaluator_sha256"))].append(row)
    actions = []
    for rows in grouped.values():
        row = dict(rows[0])
        row["observations"] = len(rows)
        row["candidate_names"] = [r["candidate"] for r in rows]
        for field in FIELDS:
            row[field] = float(np.mean([r[field] for r in rows]))
        actions.append(row)
    covered = {(r["object"], r["experiment"]) for r in actions}
    coverage = [{"object": obj, "experiment": exp, "has_candidate_metrics": (obj, exp) in covered,
                 "missing_reason": "" if (obj, exp) in covered else missing_reasons.get((obj, exp), "No candidate metrics found")}
                for obj, exp in sorted(expected)]
    return actions, observations, bases, coverage, sources, dict(excluded)


def summary(rows):
    result = {}
    for field in FIELDS:
        x = np.array([r[field] for r in rows])
        entry = dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                         np.quantile(x, [0, .05, .25, .5, .75, .95, 1]).tolist()))
        entry.update(n=len(x), mean=float(x.mean()), std_population=float(x.std()))
        if field.startswith("gain_"):
            entry.update(improved=int((x > 0).sum()), unchanged=int((x == 0).sum()), worsened=int((x < 0).sum()))
        result[field] = entry
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def ranks(x):
    """Average ranks for ties, without requiring SciPy."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation(matrix):
    out = np.full((matrix.shape[1], matrix.shape[1]), np.nan)
    ranked = np.array([ranks(col) for col in matrix.T])
    for i in range(len(ranked)):
        for j in range(len(ranked)):
            if np.std(ranked[i]) > 0 and np.std(ranked[j]) > 0:
                out[i, j] = np.corrcoef(ranked[i], ranked[j])[0, 1]
    return out


def make_plots(rows, bases, coverage, output, formats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160})
    objects = sorted({r["object"] for r in coverage})
    many_objects = len(objects) > 12
    view_counts = {r["experiment"]: r["base_views"] for r in bases}
    experiments = sorted({r["experiment"] for r in coverage},
                         key=lambda e: (view_counts.get(e, 0), e))
    palette = plt.get_cmap("tab20" if many_objects else "tab10").colors
    colors = dict(zip(objects, palette * (len(objects) // len(palette) + 1)))
    matrix = np.array([[r[f] for f in FIELDS] for r in rows])
    titles = []
    footer = f"{len(rows)} unique candidate actions · {len(bases)} bases · each point averages held-out test views"

    def save(fig, name, title, top=.95):
        fig.suptitle(title, fontsize=18, x=.04, ha="left")
        fig.text(.04, .012, footer, color="#555555", fontsize=9)
        fig.tight_layout(rect=(.015, .035, .995, top))
        for fmt in formats:
            fig.savefig(output / f"{name}.{fmt}", bbox_inches="tight")
        plt.close(fig)
        titles.append((name, title))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, ax in enumerate(axes.flat):
        x = matrix[:, i]
        bins = np.histogram_bin_edges(x, bins="fd")
        if len(bins) > 65:
            bins = np.linspace(x.min(), x.max(), 65)
        ax.hist(x, bins=bins, weights=np.ones(len(x)) / len(x), color="#416b92", alpha=.85)
        ax.axvline(np.median(x), color="#df9039", label=f"Median {np.median(x):.4g}")
        if i >= 3:
            ax.axvline(0, color="#a63c42", linestyle="--", linewidth=1)
            ax.text(.97, .94, f"{np.mean(x > 0):.1%} improve\n{np.mean(x < 0):.1%} worsen",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set(xlabel=LABELS[i], ylabel="Fraction of candidate actions")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, "01_distributions", "Six metrics · distributions and signed gains")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, ax in enumerate(axes.flat):
        for obj in objects:
            x = sorted(r[FIELDS[i]] for r in rows if r["object"] == obj)
            if x:
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=colors[obj], label=obj,
                        linestyle="--" if many_objects and objects.index(obj) >= len(palette) else "-")
        if i >= 3:
            ax.axvline(0, color="#a63c42", linestyle="--", linewidth=1)
        ax.set(xlabel=LABELS[i], ylabel="Fraction at or below value", ylim=(0, 1.02))
        ax.grid(alpha=.15)
    if many_objects:
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(.5, .955),
                   ncol=5, fontsize=8, frameon=False)
    else:
        axes[0, 0].legend(fontsize=8, frameon=False)
    save(fig, "02_ecdf", "Cumulative distributions · compare object shapes and tails",
         top=.77 if many_objects else .95)

    for group, names, filename, title in (
        ("object", objects, "03_by_object", "Six metrics · distributions by object"),
        ("experiment", experiments, "04_by_experiment", "Six metrics · distributions by base experiment"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(18, max(11, len(names) * .44)))
        for i, ax in enumerate(axes.flat):
            ticklabels = []
            for pos, name in enumerate(names, 1):
                values = [r[FIELDS[i]] for r in rows if r[group] == name]
                ticklabels.append(f"{name}  (n={len(values)})")
                if values:
                    box = ax.boxplot([values], positions=[pos], vert=False, widths=.55,
                                     whis=(5, 95), showfliers=False, patch_artist=True,
                                     medianprops={"color": "#222222"})
                    box["boxes"][0].set_facecolor(colors.get(name, "#83a7c5"))
                    box["boxes"][0].set_alpha(.6)
                    rng = np.random.default_rng(42 + pos)
                    ax.scatter(values, pos + rng.uniform(-.15, .15, len(values)),
                               s=5, color="#23435c", alpha=.3, rasterized=True)
                    if i < 3:
                        b = [r[FIELDS[i]] for r in bases if r[group] == name]
                        if b:
                            ax.scatter([np.median(b)], [pos], marker="D", s=28, color="#b95623", zorder=4)
            ax.set_yticks(range(1, len(names) + 1), ticklabels)
            if many_objects and group == "object" and i % 3 != 0:
                ax.tick_params(labelleft=False)
            ax.set_ylim(len(names) + .6, .4)
            ax.set_xlabel(LABELS[i])
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="x", alpha=.15)
            if i >= 3:
                ax.axvline(0, color="#a63c42", linestyle="--", linewidth=1)
        save(fig, filename, title + "\nBoxes: median/IQR; whiskers: 5–95%; dots: all actions; orange diamond: base median")

    fig, axes = plt.subplots(1, 3, figsize=(20, max(7, len(objects) * .39)))
    for i, ax in enumerate(axes):
        values = np.full((len(objects), len(experiments)), np.nan)
        for j, obj in enumerate(objects):
            for k, exp in enumerate(experiments):
                x = [r[f"gain_{METRICS[i]}"] for r in rows if r["object"] == obj and r["experiment"] == exp]
                if x:
                    values[j, k] = np.mean(np.array(x) > 0)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        experiment_labels = [e.removesuffix("_seed_77").replace("_", " ") for e in experiments] if many_objects else experiments
        ax.set_xticks(range(len(experiments)), experiment_labels, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(len(objects)), objects, fontsize=9)
        if many_objects and i > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(f"{METRICS[i]} · fraction improving")
        for j, k in np.ndindex(values.shape):
            v = values[j, k]
            ax.text(k, j, "—" if np.isnan(v) else f"{v:.0%}", ha="center", va="center",
                    color="white" if v > .65 else "#222222", fontsize=8)
    fig.colorbar(im, ax=list(axes), fraction=.015, pad=.015).set_label("Strictly positive gain")
    # Explicit spacing avoids tight_layout moving the shared colorbar over data.
    fig.suptitle("How often does an extra view help? · gray cells have no metric data", fontsize=18, y=.97)
    fig.subplots_adjust(left=.19 if many_objects else .12, right=.88,
                        bottom=.13 if many_objects else .3, top=.88,
                        wspace=.12 if many_objects else .65)
    fig.text(.04, .015, footer, fontsize=9, color="#555555")
    for fmt in formats:
        fig.savefig(output / f"05_improvement_rates.{fmt}", bbox_inches="tight")
    plt.close(fig)
    titles.append(("05_improvement_rates", "How often does an extra view help?"))

    # Pooling objects can hide within-context metric disagreement. Show both.
    pooled = correlation(matrix)
    context_ranks = np.full_like(matrix, np.nan)
    contexts = defaultdict(list)
    for index, row in enumerate(rows):
        contexts[(row["object"], row["experiment"])].append(index)
    for indices in contexts.values():
        if len(indices) < 2:
            continue
        for i in range(6):
            x = matrix[indices, i]
            if np.std(x) > 0:
                context_ranks[indices, i] = (ranks(x) - .5) / len(x) - .5
    within = np.full((6, 6), np.nan)
    for i in range(6):
        for j in range(6):
            keep = np.isfinite(context_ranks[:, i]) & np.isfinite(context_ranks[:, j])
            if keep.sum() >= 2:
                within[i, j] = np.corrcoef(context_ranks[keep, i], context_ranks[keep, j])[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    names = ["PSNR", "SSIM", "LPIPS ↓", "PSNR gain", "SSIM gain", "LPIPS gain"]
    for ax, mat, title in zip(axes, (pooled, within),
                              ("Pooled Spearman correlation", "Correlation of within-context percentile ranks")):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu")
        ax.set_xticks(range(6), names, rotation=40, ha="right")
        ax.set_yticks(range(6), names)
        ax.set_title(title)
        for j, k in np.ndindex(mat.shape):
            v = mat[j, k]
            ax.text(k, j, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > .65 else "#222222")
        fig.colorbar(im, ax=ax, fraction=.045)
    save(fig, "06_correlations", "Metric relationships · raw LPIPS is lower-is-better")
    return titles, {"pooled_spearman": pooled.tolist(), "within_context_rank_correlation": within.tolist()}


def report(output, actions, observations, bases, coverage, sources, excluded, formats):
    write_csv(output / "actions.csv", actions)
    write_csv(output / "observations.csv", observations)
    write_csv(output / "bases.csv", bases)
    write_csv(output / "coverage.csv", coverage)
    stats = summary(actions)
    titles, correlations = make_plots(actions, bases, coverage, output, formats)
    evaluators = defaultdict(int)
    for row in actions:
        evaluators[row["provenance"].get("evaluator_sha256", "unknown")] += 1
    payload = {"gain_formulas": FORMULAS, "summary": stats, "correlations": correlations,
               "sources": sources, "excluded": excluded, "coverage": coverage,
               "evaluator_hash_action_counts": dict(evaluators),
               "by_object": {obj: summary([r for r in actions if r["object"] == obj])
                             for obj in sorted({r["object"] for r in actions})}}
    # Constant distributions have undefined correlations; encode these as null.
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return value
    (output / "summary.json").write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False) + "\n")
    missing = [r for r in coverage if not r["has_candidate_metrics"]]
    warning = (f"Missing candidate metric data for {len(missing)} of {len(coverage)} contexts. "
               "Missing objects/contexts are not included in pooled distributions.") if missing else "All discovered contexts have candidate metrics."
    if len(evaluators) > 1:
        warning += (f" {len(evaluators)} evaluator script hashes are present (listed in summary.json). "
                    "Pooled figures are descriptive; equivalence of these evaluator versions has not been established.")
    table = "".join(f"<tr><td>{html.escape(LABELS[i])}</td><td>{stats[f]['mean']:.5g}</td>"
                    f"<td>{stats[f]['std_population']:.5g}</td><td>{stats[f]['median']:.5g}</td>"
                    f"<td>{stats[f]['q05']:.5g}</td><td>{stats[f]['q95']:.5g}</td></tr>"
                    for i, f in enumerate(FIELDS))
    obj_table = ""
    for obj in sorted({r["object"] for r in coverage}):
        cov = [r for r in coverage if r["object"] == obj]
        count = sum(r["object"] == obj for r in actions)
        obj_table += f"<tr><td>{html.escape(obj)}</td><td>{sum(r['has_candidate_metrics'] for r in cov)}/{len(cov)}</td><td>{count}</td></tr>"
    pictures = "".join(f'<h2>{html.escape(title)}</h2><a href="{name}.pdf">PDF</a>'
                       f'<img src="{name}.png" alt="{html.escape(title)}">' if "pdf" in formats else
                       f'<h2>{html.escape(title)}</h2><img src="{name}.png" alt="{html.escape(title)}">'
                       for name, title in titles)
    budget = sorted({(str(r["base_iterations"]), str(r["candidate_iterations"])) for r in actions})
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Reconstruction metric distributions</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:35px auto;padding:0 24px;color:#203344;background:#fafbfc}}
img{{width:100%;background:white}}table{{border-collapse:collapse;margin:20px 0}}td,th{{padding:8px 18px;border-bottom:1px solid #ccd3da;text-align:left}}
.notice{{padding:18px;background:#fff0cf}}h2{{margin-top:45px}}code{{background:#eef1f4;padding:2px}}</style>
<h1>Reconstruction metrics and candidate gains</h1><p>{len(actions)} physical candidate actions from {len(observations)} observations; {len(bases)} base reconstructions.</p>
<p class="notice">{warning}</p><p>Budgets (base → candidate): {html.escape(str(budget))} iterations. Missing budgets are shown as None.</p>
<p>Absolute metrics describe candidate reconstructions, averaged over saved held-out test views. Gains are signed absolute differences:
PSNR = candidate − base (dB); SSIM = candidate − base; LPIPS = base − candidate. Positive always means improvement.
These are not relative percentages. Base metrics are stored once per context.</p>
<p>Redundant base-view controls are excluded. Repeated reconstructions of the same physical action at the same budget are averaged;
all observations and provenance remain in <a href="observations.csv">observations.csv</a>. Pooled distributions weight each unique action equally.
The plots describe these saved experiments; gains may include extra optimization effects. They do not isolate the causal effect of a new view.
No confidence intervals or independent-sample claims are made.</p>
<table><tr><th>Object</th><th>Contexts with candidate metrics</th><th>Actions</th></tr>{obj_table}</table>
<table><tr><th>Metric</th><th>Mean</th><th>Population SD</th><th>Median</th><th>5th percentile</th><th>95th percentile</th></tr>{table}</table>
<p><a href="actions.csv">Action data</a> · <a href="bases.csv">Base data</a> · <a href="coverage.csv">Coverage</a> · <a href="summary.json">Summary and input hashes</a></p>
{pictures}</html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def load_config(path):
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    required = {"inputs", "objects", "output", "output_name", "formats"}
    missing = required - config.keys()
    extra = config.keys() - required
    if missing or extra:
        raise ValueError(f"Config keys mismatch; missing: {sorted(missing)}, unknown: {sorted(extra)}")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help="Override configured result roots or results.json files")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json",
                        help="JSON configuration file")
    parser.add_argument("--objects", nargs="+", help="Override the configured object selection")
    parser.add_argument("--output", type=Path, help="Override the configured output parent directory")
    parser.add_argument("--output-name", help="Override the configured output directory name")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        help="Plot formats; PNG is always included for the HTML report")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        inputs = config["inputs"]
        if not isinstance(inputs, list) or not inputs or any(not isinstance(p, str) or not p.strip() for p in inputs):
            raise ValueError("Config inputs must be a nonempty list of paths.")
        objects = args.objects if args.objects is not None else config["objects"]
        if objects is not None and (not isinstance(objects, list) or not objects
                                    or any(not isinstance(obj, str) or not obj.strip() for obj in objects)):
            raise ValueError("Objects must be null (all objects) or a nonempty list of object names.")
        output_setting = config["output"]
        if not isinstance(output_setting, str) or not output_setting.strip():
            raise ValueError("Config output must be a nonempty path.")
        name = args.output_name if args.output_name is not None else config["output_name"]
        if not isinstance(name, str) or not name.strip() or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("Output name must be a nonempty directory name, not a path.")
        formats = args.formats if args.formats is not None else config["formats"]
        if (not isinstance(formats, list) or not formats
                or any(not isinstance(fmt, str) or fmt not in ("png", "pdf", "svg") for fmt in formats)):
            raise ValueError("Formats must be a nonempty list containing png, pdf, or svg.")
        input_paths = args.inputs or [config_path.parent / Path(p).expanduser() for p in inputs]
        actions, observations, bases, coverage, sources, excluded = load_data(input_paths, objects)
        if not actions:
            raise ValueError("No candidate metrics found. Supply orchestrator evaluation results.json files; images/training logs alone are insufficient.")
        protocols = {(r["base_iterations"], r["candidate_iterations"]) for r in actions}
        if len(protocols) != 1:
            raise ValueError("Multiple iteration budgets found. Plot each budget separately rather than pooling them.")
        output_root = args.output if args.output is not None else config_path.parent / Path(output_setting).expanduser()
        output = (output_root.expanduser() / name).resolve()
        output.mkdir(parents=True, exist_ok=True)
        report(output, actions, observations, bases, coverage, sources, excluded,
               list(dict.fromkeys(["png", *formats])))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(f"Saved {len(actions)} actions ({len(observations)} observations), {len(bases)} bases: {output / 'index.html'}")
    missing = sum(not r["has_candidate_metrics"] for r in coverage)
    if missing:
        print(f"WARNING: {missing}/{len(coverage)} discovered contexts lack candidate metric data; see coverage.csv.")
    evaluators = {r["provenance"].get("evaluator_sha256", "unknown") for r in actions}
    if len(evaluators) > 1:
        print(f"NOTE: {len(evaluators)} evaluator script hashes are present; pooled plots are descriptive. See summary.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
