# Next benchmark runs

Run the controlled comparison below, then share the reports and a separate
assessment of reconstruction quality and view-selection fidelity. The methods
are **`fresh-30k` and `base-cloud-3k`**, with a warm base-only control. Do not add
other methods or tune settings during this run. The
[README](README.md) describes the scripts, configuration, outputs, and metrics.

## 1. Prepare the checkout, environment, and inputs

Update the whole repository on the GPU machine, including `reconstruction/`,
`benchmark_config.py`, and `benchmark_contexts.py`. Use its existing FastGS
Python/CUDA environment. All commands below run from `ObjaverseRenderer/`,
with FastGS at `../FastGS` and corpora under `views/`. Add `--fastgs-dir`,
`--views-root`, and/or `--python` to benchmark commands if those paths differ.
For standalone checks, activate the GPU environment and adjust `FASTGS_DIR`.

The preset [benchmark_configs/validation.json](benchmark_configs/validation.json)
selects:

| Selection | IDs |
| --- | --- |
| Objects | `Table`, `LEGO Miss Liberty (small)`, `Wave Vase` |
| Experiments | `random_3_seed_77`, `random_5_seed_77`, `random_10_seed_77` |

**Every object runs every experiment**, giving nine contexts. Base-view counts
are 3, 5, and 10; the sampling seed is 77. Candidates are blindspot, next, and random.

For each object, provide both global train/test manifests and all referenced
images under `views/<object>/`. If a corpus is missing, render its GLB using
[the README's rendering instructions](README.md#objects-and-rendered-corpora).
Blender and source GLBs are unnecessary when the rendered corpus already exists.

Agree on one corpus version, check its images/lighting, and keep it unchanged
throughout the comparison and timing repeats. Rendering samples unseeded light
energies, so rerendering changes the data. Use a separate views root for a new
corpus version. Assign one GPU run owner; do not launch overlapping jobs on
the same object's root manifests or output directories.

Use the new output root below and let the runner create its own shared bases.
Do not pass an external base root for this controlled validation.

## 2. Verify the environment

```bash
python -m unittest discover -s tests -v
python gpu_preflight.py --fastgs-dir ../FastGS
FASTGS_RUN_GPU_TESTS=1 FASTGS_DIR=../FastGS python -m unittest discover \
  -s tests -p test_fastgs_adapter_gpu.py -v
```

Proceed when the CPU suite passes, preflight reports working CUDA/FastGS/KNN/
fused SSIM, and both explicit CUDA tests pass. The default suite skips the two
opt-in CUDA tests. Their tiny custom benchmark includes an extra-point case;
that is an integration check, not another method in this validation.

Install/build missing dependencies in the FastGS environment rather than
changing benchmark settings. Let any first-use LPIPS weight download complete
before measuring the real run. CPU-only test success is insufficient to start
collecting reconstruction results.

## 3. Inspect the full plan without training

```bash
python benchmark_reconstructions.py \
  --config benchmark_configs/validation.json \
  --base-control \
  --output-root benchmarks/validation_3_5_10 \
  --dry-run
```

Expected: **9 contexts, 2 methods**, with pending commands for each shared base,
fresh candidate, warm candidate, and warm base-only control. This checks corpus
inputs and FastGS source fingerprints without loading CUDA or writing outputs.
Resolve missing object/experiment/image errors before running training.

Per context, the maximum work is one 30k scratch base, three 30k fresh candidate
runs, three 3k warm candidate runs, and one 3k warm base-only control: eight
training subprocesses. Across the matrix this is **at most 72 training
subprocesses and 27 paired candidate results**. Duplicate physical candidate
cameras are trained and counted once per method and can reduce these totals.

## 4. Complete one context, inspect it, then expand

Run one complete context, including metrics:

```bash
python benchmark_reconstructions.py \
  --config benchmark_configs/validation.json \
  --object-ids "Table" \
  --experiment-ids random_3_seed_77 \
  --base-control \
  --output-root benchmarks/validation_3_5_10
```

Check that the report contains one context and up to three distinct candidate
pairs. Verify the shared base is at 30k and warm candidates/control are at 3k.
Inspect the evaluated images under each reconstruction's `test/ours_<budget>/`
output as well as its PSNR/SSIM/LPIPS values. Resolve broken renders, missing
metrics, or provenance errors before expanding.

Run the full matrix with the same output root:

```bash
python benchmark_reconstructions.py \
  --config benchmark_configs/validation.json \
  --base-control \
  --output-root benchmarks/validation_3_5_10
```

Matching completed work from the first context is reused. This is also the
resume command after interruption: retain `--base-control`. Failed/incomplete
reconstructions restart from their configured initialization, not an
intermediate optimization checkpoint.

Expected artifacts are `benchmark_config.json`, `benchmark_manifest.json`,
`environment.json`, checkpoints/receipts, per-method result records, and the
three report files under `benchmarks/validation_3_5_10/report/`. See
[the output reference](README.md#outputs-and-resume) for the directory layout.

Check `overall.base-cloud-3k.contexts` in `comparison.json` is **9**.
`paired_candidates` should be **27** when each context has three distinct
candidates; explain a smaller number using recorded duplicate-camera aliases.
Missing contexts are errors, not successful partial completion.

## 5. Review and share the results

For every object and base-view count, record these values from
`report/comparison.json` and inspect the corresponding images:

| Question | Fields to review |
| --- | --- |
| Is warm adaptation faster? | `summary.speedup`, candidate seconds, and production time including the shared base. |
| Is reconstruction quality acceptable? | Absolute PSNR/SSIM/LPIPS, `mean_delta_gain`, and negative-gain counts. |
| Does a candidate help beyond more optimization? | `summary.base_control_gain`, `rows[].method_gain_over_control`, and `summary.mean_method_gain_over_control`. |
| Do warm gains preserve fresh view utility? | `psnr_spearman`, `psnr_top_choice_agreement`, `selected_view_regret_db`, and `gain_mae`. |

Positive gains mean improvement for all metrics, including LPIPS. Controls and
candidate effects are in JSON/text; CSV retains the primary base/candidate
comparison. Use [the metric definitions](README.md#metrics-controls-and-report-interpretation)
when interpreting ties, means, and time totals.

Expect the shorter warm procedure to be faster, but **measure** the magnitude.
Quality gains, control effects, and agreement with fresh rankings are unknown
until evaluated. Do not assume improvement for every object or candidate.
Subtracting one shared control explains gain offsets but cannot change the
candidate ranking within a context.

Make two separate assessments:

- **Fast reconstruction:** judge warm adaptation using measured speed,
  absolute quality, and visual artifacts on the intended objects.
- **Fresh-training view utility:** review ranking agreement and regret against
  the application's accepted error tolerance. Better mean reconstruction
  quality alone does not justify substituting warm gains for fresh labels.

Share the per-context comparison table, `report/` directory, saved config,
manifest, environment record, and notes/images illustrating failures. Keep the
corpus and checkpoints available for reproducibility and debugging. The run
owner and reviewers should record both assessments before collecting predictor
labels or starting parameter sweeps.

## 6. Repeat runtime measurements

Repeat the same full matrix on the same GPU/machine and unchanged corpus with
a new output root:

```bash
python benchmark_reconstructions.py \
  --config benchmark_configs/validation.json \
  --base-control \
  --output-root benchmarks/validation_3_5_10_repeat2
```

Compare runtime variability and quality consistency. Reusing the first root
would reuse cached times. A new root does **not** create independent optimizer
seeds: FastGS seeds Python, NumPy, and PyTorch with 0. Experiment seed 77 selects
cameras. Independent training-seed support and a separate validation/test split
are implementation prerequisites for later statistical or tuning studies;
they are not part of this execution plan.

## Deferred evaluation and report regeneration

If GPU allocations require separate training/evaluation, add `--train-only` to
the full run and later execute:

```bash
python benchmark_reconstructions.py \
  --output-root benchmarks/validation_3_5_10 --evaluate-only
```

Use the same GPU Python/FastGS setup; retain experiment definitions and repeat
custom path overrides as needed. This reuses the saved methods, view root, and
exact context pairs, and rejects changed corpus inputs.

Regenerate reports from completed metrics without CUDA or source images:

```bash
python benchmark_reconstructions.py \
  --output-root benchmarks/validation_3_5_10 --report-only
```
