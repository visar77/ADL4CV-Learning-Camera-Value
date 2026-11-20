#!/usr/bin/env python3
"""Gain histograms, box plots, overview, and correlations from the actions CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=220, facecolor="white")
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


def boxplot(ax, samples, objects, color, label, precision):
    ax.boxplot(samples, positions=range(len(objects)), vert=False, widths=.55,
               patch_artist=True,
               boxprops={"facecolor": color, "edgecolor": color, "alpha": .65},
               medianprops={"color": "#203344", "linewidth": 1.7},
               whiskerprops={"color": "#65717C"}, capprops={"color": "#65717C"},
               flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none",
                           "markeredgecolor": color, "alpha": .6})
    ax.set_yticks(range(len(objects)),
                  [f"{NAMES.get(obj, obj)}  (n={len(x)})" for obj, x in zip(objects, samples)], fontsize=10)
    ax.set_ylim(len(objects) - .5, -.5)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=HERE.parent / "metric_distributions/output/actions.csv")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("The input CSV has no candidate actions.")
    if len({(r["base_iterations"], r["candidate_iterations"]) for r in rows}) != 1:
        parser.error("Use one reconstruction iteration budget per input CSV.")
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
    overview.text(.04, .95, "Reconstruction gains", fontsize=25, weight="bold", color="#203344")
    overview.text(.04, .905, f"{len(rows):,} unique candidate views · {len(objects)} objects · positive gain = improvement",
                  fontsize=12, color="#65717C")
    overview.text(.04, .025, "Boxes: median and middle 50% · whiskers: 1.5 × IQR · dots: outliers",
                  fontsize=10, color="#65717C")
    for column, (metric, values, color, filename) in enumerate(panels):
        label = f"{metric} gain" + (" (dB)" if metric == "PSNR" else "")
        formula = "Base − candidate" if metric == "LPIPS" else "Candidate − base"
        if metric == "Relative PSNR":
            label = "PSNR gain / base PSNR"
            formula = "(Candidate − base) / base"
        fig, ax = figure(f"{metric} gain · distribution",
                         f"{formula}  ·  {len(rows):,} unique candidate views  ·  {len(objects)} objects")
        histogram(ax, values, color, label)
        save(fig, args.output, f"{filename}_histogram")

        samples = [values[[r["object"] == obj for r in rows]] for obj in objects]
        fig, ax = figure(f"{metric} gain · by object",
                         "Median and middle 50% · whiskers: 1.5 × IQR · dots: outliers")
        fig.subplots_adjust(left=.32)
        precision = 2 if metric == "PSNR" else 3 if metric == "Relative PSNR" else 4
        boxplot(ax, samples, objects, color, label, precision)
        save(fig, args.output, f"{filename}_boxplot")
        histogram(axes[0, column], values, color, "")
        axes[0, column].set_title(f"{metric} gain", fontsize=16, color=color, weight="bold", pad=15)
        boxplot(axes[1, column], samples, objects, color, label, precision)
        if column:
            axes[0, column].set_ylabel("")
            axes[1, column].tick_params(labelleft=False)
    axes[0, 0].set_ylim(0, max(np.histogram(values, bins=100)[0].max() for _, values, _, _ in panels) * 1.12)
    save(overview, args.output, "overview")

    rho = spearman(np.column_stack([values for _, values, _, _ in panels]))
    fig, ax = figure("Gain metric correlations",
                     f"Spearman rank correlation · {len(rows):,} unique candidate views")
    fig.subplots_adjust(left=.23, right=.86, bottom=.22)
    masked = np.ma.masked_where(np.triu(np.ones(rho.shape, dtype=bool), k=1), rho)
    im = ax.imshow(masked, cmap="RdBu", vmin=-1, vmax=1)
    names = ["PSNR gain", "Relative\nPSNR gain", "SSIM gain", "LPIPS gain"]
    ax.set_xticks(range(len(panels)), names)
    ax.set_yticks(range(len(panels)), names)
    for i in range(len(panels)):
        for j in range(i + 1):
            value = rho[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "n/a", ha="center", va="center",
                    fontsize=19, color="white" if abs(value) > .65 else "#203344")
    fig.colorbar(im, ax=ax, fraction=.05, pad=.07, ticks=[-1, -.5, 0, .5, 1]).set_label("Spearman ρ")
    fig.text(.12, .075, "Pooled across objects and experiments", fontsize=10, color="#65717C")
    save(fig, args.output, "correlations")
    print(f"Saved ten figures as PNG and PDF in {args.output.resolve()}")


if __name__ == "__main__":
    main()
