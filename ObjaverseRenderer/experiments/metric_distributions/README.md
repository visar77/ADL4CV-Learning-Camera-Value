# Reconstruction metric distributions

This experiment keeps its script, tests, documentation, and generated outputs
in this folder. It reads existing result bundles without changing them or
running reconstruction/evaluation. Requires NumPy and Matplotlib.

From the workspace root:

```bash
python3 ObjaverseRenderer/experiments/metric_distributions/plot_metric_distributions.py \
  four_objects_views_results_reconstructions \
  six_objects_views_results_reconstructions
```

The default output is always [output/](output/), beside the script, regardless
of the working directory. Open [output/index.html](output/index.html).
`--output` explicitly overrides that location. Generated output is Git-ignored;
copy it explicitly when transferring the experiment's saved results.

The six figures are exported as PNG/PDF:

- Histograms of candidate PSNR/SSIM/LPIPS and their gains.
- Cumulative distributions by object.
- Boxplots by object, with base medians for absolute metrics.
- Boxplots by base experiment.
- Per-object/per-experiment fractions of candidates that improve each metric.
- Pooled Spearman and within-context percentile-rank correlations.

`--formats png pdf svg` also exports SVG. All tails remain visible. Boxplots
show the median and interquartile range, 5th–95th percentile whiskers, and all
candidate actions as dots. Orange diamonds show base medians.

Gains are **absolute** differences: candidate minus base for PSNR (dB) and SSIM,
base minus candidate for LPIPS. Positive always means improvement. Negative
gains are retained. These are not the predictor's relative-gain labels. Metrics
are saved averages over held-out views. Each unique candidate action receives
equal weight; base metrics are stored once per context.

Redundancy controls are excluded. Repeated reconstructions of the same physical
camera action within a context, budget, and evaluator are averaged. Every raw
observation and its provenance remains in `output/observations.csv`. Other
exports are `actions.csv`, `bases.csv`, `coverage.csv`, and `summary.json`, with
quantiles, sign counts, input hashes, evaluator hashes, and coverage.

Inputs can be exported bundles, result roots, or individual orchestrator
`results.json` files. Reconstruction-only inputs and failed contexts appear as
coverage gaps. Different iteration budgets are rejected; different evaluator
script hashes are explicitly flagged. Recorded base/candidate test-data and
evaluator fingerprints must agree. These are descriptive summaries: gains can
include additional optimization effects, and pooled statistics do not establish
causality or equivalence of different evaluator versions. Supply separate runs
of the same context in separate invocations.

## Current data and missing contexts

The 2026-11-18 report contains ten objects, 910 unique candidate actions from
920 observations, and 92 completed contexts. Eight Dinavac DV-19 contexts have
no candidate metrics:

- `furthest_4_seed_77`
- `random_3_seed_77`, `random_5_seed_77`, `random_10_seed_77`, `random_15_seed_77`
- `trajectory_2_seed_77`, `trajectory_6_seed_77`, `trajectory_12_seed_77`

Their source JSONs under
`six_objects_views_results_reconstructions/results/Dinavac DV-19/` record
base training as failed (subprocess exit code 1) and candidate training as
skipped because the shared base failed. Their exported reconstruction folders
contain no saved point-cloud checkpoints. Only `furthest_8_seed_77` and
`furthest_14_seed_77` completed for this object. The export includes no underlying
traceback or training stderr, so the cause of the trainer failures is unknown.
The gaps come from the input results, rather than plot filtering.

The two batches have different evaluator-script hashes, retained in the report.
Their metric implementation equivalence was not established.

## Validation

From this folder:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover signed gains, repeated-action aggregation, failed/missing
coverage, invalid metric/camera data, test fingerprint mismatches, duplicated
input runs, and rank ties. They do not require a GPU.

## 34 complete cluster objects

Run `python3 ObjaverseRenderer/experiments/metric_distributions/compare_cluster_34.py`
from the workspace root. This follows the 34-object feature manifest to
select 340 complete contexts from `cluster_34/results/` and exports the same
six figures plus shared-axis cohort CDFs for the original nine, the matching
nine object names in the cluster, the additional 25, and all 34. It leaves the
earlier archive under `output/` in place.

[Open the 34-object report](output_cluster_34/index.html) ·
[Cohort comparison data](output_cluster_34/comparison.csv) ·
[LVSM learning curves and fold analysis](../../../Reconstruction-Gain-Predictor/experiments/cluster_34_heads/README.md)
