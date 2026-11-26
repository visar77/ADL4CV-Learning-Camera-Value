# Gain metrics

Ten figures: one histogram and one horizontal box plot for each of
PSNR gain, relative PSNR gain, SSIM gain, and LPIPS gain, plus an overview and
a combined four-metric correlation heatmap. Boxes show the **gain distribution by object**;
sample counts are shown beside object names. Each figure is exported as PNG
and vector PDF in [output/](output/). The overview is also exported as
[overview.svg](output/overview.svg).

[overview.png](output/overview.png) places the four histograms above the four
box plots, with consistent object ordering, shared count scales, and matched
gain axes within each column. Its columns are absolute PSNR, relative PSNR,
SSIM, and LPIPS gains. Relative PSNR gain is `gain_PSNR / base_PSNR`, displayed
as a dimensionless ratio. Its separate plots are
[histogram](output/relative_psnr_histogram.png) and
[box plot](output/relative_psnr_boxplot.png). Both retain signed values and
the full data range.

[correlations.png](output/correlations.png) shows
the lower half of the symmetric Spearman rank-correlation matrix for all four
gain metrics, in the same order as the overview. Correlations use all unique candidate actions pooled across
objects and experiments; they do not measure within-context ranking agreement.
Average ranks handle ties. Constant metrics have undefined correlations.
The separate relative-PSNR correlation experiment has been consolidated here;
this script generates every current figure.

From this folder, with NumPy and Matplotlib installed:

```bash
python3 plot_gains.py
```

The script reads the existing [deduplicated action CSV](../metric_distributions/output/actions.csv).
It does not run the earlier plotting script.
Use `--input path/to/actions.csv` to select another export. Default input/output
paths resolve relative to the script, so it can run from any working directory.

Absolute gains are differences: candidate minus base for PSNR (dB) and SSIM,
base minus candidate for LPIPS. Positive always means improvement. Histograms
retain the full range; the gold dashed line marks the median. Box plots show
the median and middle 50%; whiskers extend to the furthest observations within
1.5 times the interquartile range, and outliers are drawn as dots. All values
remain visible. Each unique physical candidate action has equal weight. Repeated actions were averaged
in the input export, whose source observations/provenance remain in the earlier
experiment. No extra data copies or metric recomputation are needed.

The figures use 891 actions across nine objects. Each object contributes
99 actions from ten contexts. The script excludes Dinavac DV-19 from all
statistics and plots because eight of its ten contexts failed. Its 19 completed
actions remain in the source CSV. The batches have different evaluator-script
hashes, so the pooled figures are descriptive. All
recorded budgets are 30k base / 3k candidate. Generated output is Git-ignored.

Run the correlation and relative-PSNR checks from this folder with
`python3 -m unittest discover -s tests -v`.
