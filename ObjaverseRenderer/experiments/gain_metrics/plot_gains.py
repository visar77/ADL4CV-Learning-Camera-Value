#!/usr/bin/env python3
"""Gain histograms, per-object and per-experiment box plots, overview, and correlations.

Reads orchestrator result roots (results/<object>/<experiment>/results.json). Each action is
one candidate camera added to a base; gains are signed, so positive always means improvement.
Run with the adjacent config.json, or select another with --config.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"PSNR": "#276678", "SSIM": "#3C826C", "LPIPS": "#89649A"}
NAMES = {"personaje_vr": "Personaje VR", "vespa_HP": "Vespa HP",
         "Sci-fi _ Retro Future Car": "Sci-fi / Retro Future Car"}


def figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=.12, right=.94, bottom=.15, top=.79)
    fig.text(.12, .93, title, fontsize=23, weight="bold", color="#203344")
    fig.text(.12, .875, subtitle, fontsize=11, color="#65717C")
    fig.text(.12, .035, "Positive gain = improvement", fontsize=10, color="#65717C")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=8)
    return fig, ax


def save(fig, output, name, formats, *, bbox_inches=None, close=True):
    for extension in formats:
        fig.savefig(output / f"{name}.{extension}", dpi=220, facecolor="white",
                    bbox_inches=bbox_inches)
    if close:
        plt.close(fig)


def histogram(ax, values, color, label):
    counts, edges, bars = ax.hist(values, bins=100, color=color, edgecolor="white", linewidth=.7)
    for edge, bar in zip(edges[1:], bars):
        if edge <= 0:
            bar.set_facecolor("#C76F62")
    ax.axvline(0, color="#8A949C", linewidth=1, linestyle="--")
    ax.axvline(np.median(values), color="#C28E40", linewidth=1.8, linestyle="--")
    ax.text(.97, .94, f"Median {np.median(values):+.4g}\n{np.mean(values > 0):.1%} improve",
            transform=ax.transAxes, ha="right", va="top", linespacing=1.7, color="#40505D")
    ax.set(xlabel=label, ylabel="Candidate views", ylim=(0, max(counts) * 1.12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(axis="y", color="#E9EDF0", linewidth=.8)


def boxplot(ax, samples, names, color, label, precision):
    ax.boxplot(samples, positions=range(len(names)), vert=False, widths=.55,
               patch_artist=True,
               boxprops={"facecolor": color, "edgecolor": color, "alpha": .65},
               medianprops={"color": "#203344", "linewidth": 1.7},
               whiskerprops={"color": "#65717C"}, capprops={"color": "#65717C"},
               flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                           "markeredgecolor": color, "alpha": .6})
    ax.set_yticks(range(len(names)),
                  [f"{name}  (n={len(x)})" for name, x in zip(names, samples)], fontsize=10)
    ax.set_ylim(len(names) - .5, -.5)
    ax.axvline(0, color="#8A949C", linewidth=1)
    ax.set_xlabel(label)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:." + str(precision) + "f}"))
    ax.grid(axis="x", color="#E9EDF0", linewidth=.8)


def spearman(values):
    ranked = []
    for column in values.T:
        _, indices, counts = np.unique(column, return_inverse=True, return_counts=True)
        ranked.append((np.cumsum(counts) - (counts - 1) / 2)[indices])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(ranked)


def relative_psnr_gain(gain, base):
    gain, base = np.asarray(gain, dtype=float), np.asarray(base, dtype=float)
    if not np.isfinite(gain).all() or not np.isfinite(base).all() or np.any(base == 0):
        raise ValueError("PSNR gains and base PSNR must be finite; base PSNR must be nonzero.")
    with np.errstate(over="ignore", invalid="ignore"):
        relative = gain / base
    if not np.isfinite(relative).all():
        raise ValueError("Relative PSNR gains must be finite.")
    return relative


def load_config(path):
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    required = {"inputs", "objects", "output", "output_name", "formats"}
    missing, extra = required - config.keys(), config.keys() - required
    if missing or extra:
        raise ValueError(f"Config keys mismatch; missing: {sorted(missing)}, unknown: {sorted(extra)}")
    return config


def scores(value, location):
    result = {}
    for metric in COLORS:
        number = value["averages"][metric]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{location}: {metric} must be finite")
        result[metric] = float(number)
    return result


def result_files(inputs):
    files = set()
    for root in inputs:
        root = root.expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Input does not exist: {root}")
        if root.is_file():
            files.add(root)
        else:
            files.update((root / "results" if (root / "results").is_dir() else root).rglob("results.json"))
    return sorted(files)


def read_actions(paths, objects):
    """One row per unique candidate action; repeated trainings of an action are averaged."""
    selected = set(objects) if objects is not None else None
    available, seen, observations = set(), set(), []
    for path in result_files(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        obj = data.get("object_id", path.parent.parent.name)
        exp = data.get("experiment_id", path.parent.name)
        available.add(obj)
        if selected is not None and obj not in selected:
            continue
        if "reconstructions" not in data:
            raise ValueError(f"{path}: expected orchestrator results with reconstructions")
        if (obj, exp) in seen:
            raise ValueError(f"Repeated context {(obj, exp)} in inputs. Supply each experimental run separately.")
        seen.add((obj, exp))
        recon, metrics = data["reconstructions"], data.get("metrics", {})
        if "base" not in metrics or "base" not in recon:
            continue
        base = scores(metrics["base"], f"{path}:base")
        base_indices = recon["base"]["train_lattice_indices"]
        base_set = set(base_indices)
        if len(base_set) != len(base_indices):
            raise ValueError(f"{path}: repeated base camera index")
        for name, item in recon.items():
            if name == "base":
                continue
            indices = item.get("train_lattice_indices", [])
            if name.startswith("redundancy") or item.get("status") == "redundant" or set(indices) == base_set:
                continue
            if item.get("status", "complete") != "complete" or name not in metrics:
                continue
            if not base_set < set(indices) or len(set(indices) - base_set) != 1 or len(indices) != len(base_set) + 1:
                raise ValueError(f"{path}:{name}: expected the same base plus exactly one new camera")
            provenance = data.get("metrics_provenance", {})
            bp, cp = provenance.get("base", {}), provenance.get(name, {})
            for key in ("test_data_sha256", "evaluator_sha256"):
                if bp.get(key) and cp.get(key) and bp[key] != cp[key]:
                    raise ValueError(f"{path}:{name}: base/candidate {key} mismatch")
            values = scores(metrics[name], f"{path}:{name}")
            row = {"object": obj, "experiment": exp, "base_views": len(base_set),
                   "base_iterations": recon["base"].get("iterations"),
                   "candidate_iterations": item.get("iterations"),
                   "candidate_index": next(iter(set(indices) - base_set)),
                   "evaluator": cp.get("evaluator_sha256"), "base_PSNR": base["PSNR"]}
            for metric in COLORS:
                row[f"gain_{metric}"] = (values[metric] - base[metric]) * (-1 if metric == "LPIPS" else 1)
            observations.append(row)
    if selected is not None and selected - available:
        raise ValueError(f"Objects not found in inputs: {', '.join(sorted(selected - available))}")
    grouped = defaultdict(list)
    for row in observations:
        grouped[tuple(row[k] for k in ("object", "experiment", "candidate_index", "base_iterations",
                                       "candidate_iterations", "evaluator"))].append(row)
    rows = []
    for group in grouped.values():
        row = dict(group[0])
        for field in ["base_PSNR"] + [f"gain_{m}" for m in COLORS]:
            row[field] = float(np.mean([r[field] for r in group]))
        rows.append(row)
    if not rows:
        raise ValueError("No candidate metrics found. Supply orchestrator evaluation results.json files.")
    if len({(r["base_iterations"], r["candidate_iterations"]) for r in rows}) != 1:
        raise ValueError("Multiple iteration budgets found. Plot each budget separately rather than pooling them.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Override configured result roots or results.json files")
    parser.add_argument("--config", type=Path, default=HERE / "config.json", help="JSON configuration file")
    parser.add_argument("--objects", nargs="+", help="Override the configured object selection")
    parser.add_argument("--output", type=Path, help="Override the configured output parent directory")
    parser.add_argument("--output-name", help="Override the configured output directory name")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        help="Plot formats; PNG is always included")
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
        formats = list(dict.fromkeys(["png", *formats]))
        input_paths = args.inputs or [config_path.parent / Path(p).expanduser() for p in inputs]
        rows = read_actions(input_paths, objects)
        output_root = args.output if args.output is not None else config_path.parent / Path(output_setting).expanduser()
        args.output = (output_root.expanduser() / name).resolve()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    gains = np.array([[float(r[f"gain_{m}"]) for m in COLORS] for r in rows])
    if not np.isfinite(gains).all():
        parser.error("All gains must be finite.")
    base_psnr = np.array([float(r["base_PSNR"]) for r in rows])
    try:
        relative_psnr = relative_psnr_gain(gains[:, 0], base_psnr)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#40505D", "xtick.color": "#53616C",
                         "ytick.color": "#53616C", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "axes.spines.bottom": False, "pdf.fonttype": 42})
    objects = sorted({r["object"] for r in rows})
    base_views = {r["experiment"]: int(r["base_views"]) for r in rows}
    experiments = sorted(base_views, key=lambda e: (base_views[e], e))
    object_names = [NAMES.get(obj, obj) for obj in objects]
    experiment_names = [e.removesuffix("_seed_77").replace("_", " ") for e in experiments]
    by_experiment, experiment_axes = plt.subplots(
        1, 4, figsize=(23, max(6.5, len(experiments) * .55 + 2.5)), sharey=True)
    by_experiment.subplots_adjust(left=.14, right=.98, top=.82, bottom=.1, wspace=.1)
    by_experiment.text(.04, .94, "Reconstruction gains · by base experiment", fontsize=25,
                       weight="bold", color="#203344")
    by_experiment.text(.04, .89, f"{len(rows):,} unique candidate views · {len(experiments)} experiments · "
                       "positive gain = improvement", fontsize=12, color="#65717C")
    panels = [("PSNR", gains[:, 0], COLORS["PSNR"], "psnr"),
              ("Relative PSNR", relative_psnr, "#477DA8", "relative_psnr"),
              ("SSIM", gains[:, 1], COLORS["SSIM"], "ssim"),
              ("LPIPS", gains[:, 2], COLORS["LPIPS"], "lpips")]
    overview, axes = plt.subplots(2, 4, figsize=(23, 10), sharex="col", sharey="row",
                                  gridspec_kw={"height_ratios": [1, 1.4]})
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=8)
    overview.subplots_adjust(left=.15, right=.98, top=.85, bottom=.09, wspace=.14, hspace=.27)
    title = overview.text(.04, .95, "Reconstruction gains", fontsize=25, weight="bold", color="#203344")
    subtitle = overview.text(
        .04, .905, f"{len(rows):,} unique candidate views · {len(objects)} objects · positive gain = improvement",
        fontsize=12, color="#65717C",
    )
    for column, (metric, values, color, filename) in enumerate(panels):
        label = f"{metric} gain" + (" (dB)" if metric == "PSNR" else "")
        formula = "Base − candidate" if metric == "LPIPS" else "Candidate − base"
        if metric == "Relative PSNR":
            label = "PSNR gain / base PSNR"
            formula = "(Candidate − base) / base"
        fig, ax = figure(f"{metric} gain · distribution",
                         f"{formula}  ·  {len(rows):,} unique candidate views  ·  {len(objects)} objects")
        histogram(ax, values, color, label)
        save(fig, args.output, f"{filename}_histogram", formats)

        samples = [values[[r["object"] == obj for r in rows]] for obj in objects]
        fig, ax = figure(f"{metric} gain · by object",
                         "Median and middle 50% · whiskers: 1.5 × IQR · dots: outliers")
        fig.subplots_adjust(left=.32)
        precision = 2 if metric == "PSNR" else 3 if metric == "Relative PSNR" else 4
        boxplot(ax, samples, object_names, color, label, precision)
        save(fig, args.output, f"{filename}_boxplot", formats)
        histogram(axes[0, column], values, color, "")
        axes[0, column].set_title(f"{metric} gain", fontsize=16, color=color, weight="bold", pad=15)
        boxplot(axes[1, column], samples, object_names, color, label, precision)
        if column:
            axes[0, column].set_ylabel("")
            axes[1, column].tick_params(labelleft=False)

        experiment_samples = [values[[r["experiment"] == e for r in rows]] for e in experiments]
        fig, ax = figure(f"{metric} gain · by experiment",
                         "Median and middle 50% · whiskers: 1.5 × IQR · dots: outliers")
        fig.subplots_adjust(left=.32)
        boxplot(ax, experiment_samples, experiment_names, color, label, precision)
        save(fig, args.output, f"{filename}_boxplot_by_experiment", formats)
        experiment_axes[column].tick_params(axis="both", length=0, pad=8)
        boxplot(experiment_axes[column], experiment_samples, experiment_names, color, label, precision)
        experiment_axes[column].set_title(f"{metric} gain", fontsize=16, color=color, weight="bold", pad=15)
        if column:
            experiment_axes[column].tick_params(labelleft=False)
    save(by_experiment, args.output, "overview_by_experiment", formats)
    axes[0, 0].set_ylim(0, max(np.histogram(values, bins=100)[0].max() for _, values, _, _ in panels) * 1.12)
    save(overview, args.output, "overview", [f for f in formats if f != "png"], close=False)
    title.remove()
    subtitle.remove()
    overview.subplots_adjust(left=.16, top=.93, bottom=.115)
    for label in overview.findobj(match=Text):
        label.set_fontsize(label.get_fontsize() * 1.4)
    save(overview, args.output, "overview", ["png"])

    rho = spearman(np.column_stack([values for _, values, _, _ in panels]))
    fig, ax = figure("Gain metric correlations",
                     f"Spearman rank correlation · {len(rows):,} unique candidate views")
    fig.subplots_adjust(left=.23, right=.86, bottom=.22)
    masked = np.ma.masked_where(np.triu(np.ones(rho.shape, dtype=bool), k=1), rho)
    edges = np.arange(len(panels) + 1) - .5
    mesh = ax.pcolormesh(edges, edges, masked, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    names = ["PSNR gain", "Relative\nPSNR gain", "SSIM gain", "LPIPS gain"]
    ax.set_xticks(range(len(panels)), names)
    ax.set_yticks(range(len(panels)), names)
    for i in range(len(panels)):
        for j in range(i + 1):
            value = rho[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "n/a", ha="center", va="center",
                    fontsize=19, color="white" if abs(value) > .65 else "#203344")
    fig.colorbar(mesh, ax=ax, fraction=.05, pad=.07, ticks=[-1, -.5, 0, .5, 1]).set_label("Spearman ρ")
    fig.text(.12, .075, "Pooled across objects and experiments", fontsize=10, color="#65717C")
    text_left = ax.get_position().x0
    for label in fig.texts:
        label.set_x(text_left)
    save(fig, args.output, "correlations", formats, bbox_inches="tight")
    print(f"Saved figures ({', '.join(formats)}) in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
