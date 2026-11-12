# Matched reconstruction benchmarks

All implementation changes are in **ObjaverseRenderer**, on branch
`experimental_base_inilialization`. Use your existing FastGS checkout and CUDA
environment; no FastGS source changes or special FastGS branch are required.

## Run the comparison shown in the example

From `ObjaverseRenderer/`, in the GPU FastGS environment:

```bash
python benchmark_reconstructions.py \
  --object-id "Wave Vase" \
  --experiment-id furthest_8_seed_77 \
  --method base-cloud-15k \
  --method base-extra-15k \
  --output-root benchmarks/wave_vase \
  --gpu-smoke-test
```

This runs three methods: the automatically included `fresh-30k` reference,
`base-cloud-15k`, and `base-extra-15k`. Each uses the same selected base views and
the same three independent candidate actions: one blindspot, one next, and one
random view. Each candidate trains on **base views plus that one candidate**.
Candidate selection is fixed across methods. Duplicate physical candidate
indices are reconstructed and counted once. Redundancy controls are excluded
from comparisons.

The original base is trained once, then reused unchanged. To reuse your existing
base outputs instead, add:

```bash
--base-reconstructions-root /data/reconstructions
```

That directory must contain
`<object>/<experiment>/base/point_cloud/iteration_30000/point_cloud.ply`.
Legacy base PLYs are accepted; their original settings cannot be verified and a
missing historical time is reported as `N/A`, never guessed. Candidate timings
are still available from new runs.

The benchmark automatically evaluates all outputs and prints a table per method,
object and context, followed by an aggregate across contexts. It writes:

```text
benchmarks/wave_vase/
  benchmark_config.json
  benchmark_manifest.json
  environment.json
  shared_base/<object>/<experiment>/base/...
  initializations/<object>/<experiment>/<content-hash>/...
  reconstructions/<method>/<object>/<experiment>/<candidate>/...
  results/<method>/<object>/<experiment>/results.json
  report/comparison.txt
  report/comparison.csv
  report/comparison.json
```

The report contains:

- Per-candidate reference/method seconds and PSNR, SSIM and LPIPS gain differences.
- Absolute quality scores and each method's gains over the same original base
  in the JSON and CSV outputs.
- Candidate totals, percentage time saved, speedup and mean gains.
- PSNR gain MAE, negative-gain counts, gain-sign agreement, rank correlation,
  and selected-view regret measured using the fresh reference's gains.
- Shared base time, extra-initialization preparation time and optional control
  time separately; production totals including the base when its time is known.
- Initial Gaussian counts for methods with extra points.

Positive PSNR/SSIM gain means an increase. Positive LPIPS gain means a **decrease**.
A positive method-minus-fresh gain difference means the method is better. Negative
observations are retained. A warm start does not guarantee nonnegative gains.

Candidate times use the same boundary for every method: wall time of its training
subprocess, including Python/CUDA startup, image/model loading, optimization and
saving. Rendering the source corpus and the separate quality evaluation are
excluded. Cached runs retain their original timing; they are not timed as zero.
Do not interpret subprocess wall time as just CUDA kernel time.

## GPU readiness and verification

The benchmark runs `gpu_preflight.py` using the selected interpreter before
training. It checks CUDA availability, imports and signatures of the selected
FastGS checkout, the KNN CUDA kernel, and fused SSIM forward/backward. It records
GPU type, compute capability, PyTorch/CUDA versions and host identity. Comparisons
reject inconsistent recorded timing environments.

`--gpu-smoke-test` additionally runs two self-contained CUDA tests before the
real experiment. They check direct PLY adaptation with densification and then
run an entire tiny benchmark: scratch training, plain warm starts, extra-Gaussian
warm starts, real rendering, real PSNR/SSIM/LPIPS evaluation and report generation.
These tiny tests check integration, not reconstruction quality. Their process
startup and metric/model loading can take a few minutes.

Run the checks separately if preferred:

```bash
python gpu_preflight.py --fastgs-dir ../FastGS
FASTGS_RUN_GPU_TESTS=1 python -m unittest discover \
  -s tests -p test_fastgs_adapter_gpu.py -v
```

Set `FASTGS_DIR=/path/to/FastGS` for the standalone GPU tests when the checkout
is not a sibling. The GPU tests need the same native extensions and pretrained
LPIPS/VGG weights as normal FastGS metrics. As with FastGS itself, the first VGG
use may download weights if they are not already cached.

The development machine has no available CUDA device. CPU tests verify orchestration,
PLY manipulation, resume/invalidation, evaluation command interoperability and
report arithmetic. Actual CUDA execution must be verified on the GPU machine by
the opt-in tests; passing the CPU checks alone is not a claim of measured GPU
speed or quality.

## Test the full matrix or customize it

Omit both `--method` options to run the full preset:

| Method | Candidate steps | Extra initial Gaussians |
|---|---:|---:|
| `fresh-30k` | 30000 | None; original random initialization |
| `base-cloud-3k` | 3000 | 0 |
| `base-cloud-5k` | 5000 | 0 |
| `base-cloud-15k` | 15000 | 0 |
| `base-extra-3k` | 3000 | 3000 |
| `base-extra-5k` | 5000 | 3000 |
| `base-extra-15k` | 15000 | 3000 |

Repeat `--object-id` and/or `--experiment-id` for more contexts. With no filters,
all existing view corpora and experiment definitions are selected. Use
`--max-contexts 2` or `--candidate-limit 1` for a smaller first test. The benchmark
uses existing rendered views; Blender and source GLBs are not required.

Inspect pending commands without writing files or loading CUDA:

```bash
python benchmark_reconstructions.py \
  --object-id "Wave Vase" --experiment-id furthest_8_seed_77 \
  --method base-cloud-15k --method base-extra-15k \
  --output-root benchmarks/wave_vase --dry-run
```

Copy `benchmark_configs/base_initialization.json` to customize the matrix and
pass it with `--config my_benchmark.json`. Its `candidates` list overrides the
candidate settings of every selected experiment, so every method sees matched
inputs. The experiment's base-view strategy/count/seed remain unchanged.

Each method supports `id`, `initialization` (`scratch` or `base`), `iterations`,
`lr_scale`, `densify_from_iter`, `densify_until_iter`, `densification_interval`,
`extra_gaussians`, `extra_gaussian_seed`, `extra_gaussian_jitter`,
`extra_gaussian_opacity` and `extra_gaussian_scale`. A method's values override
`adaptation_defaults`. Use unique IDs to keep different configurations side by
side. For example:

```json
{"id": "extra-5k-no-densify", "initialization": "base", "iterations": 5000,
 "extra_gaussians": 5000, "densify_until_iter": 0, "lr_scale": 0.05}
```

You can also add scratch methods with shorter budgets to isolate initialization
from iteration-budget effects. All methods still measure gains against the same
original base; `reference` chooses which scratch method is the report reference.

## Initialization and adaptation behavior

The renderer-owned `fastgs_adapter.py` imports the existing FastGS Scene,
GaussianModel, rasterizer, losses and densification routines. A local subclass
loads the PLY through Scene's existing initialization hook. The adapter owns the
short training loop and does not edit or monkey-patch FastGS source/modules.

Warm starts use fresh optimizer moments and gradient/radius buffers while
retaining the trained Gaussians and full SH degree. Position learning-rate decay
spans the adaptation budget. By default, learning rates are scaled by 0.1,
densification starts after step 100, stops at half the budget, and is considered
every 250 steps. An empty default densification window is disabled automatically.
`densify_until_iter: 0` explicitly disables it. Main parameters update every
step; SH features update every 16 steps and on the final step. Global opacity
resets and aggressive late scratch pruning are skipped. Final saves include the
final optimization step.

Extra initial Gaussians are created in a **separate cached PLY**. Base rows are
preserved exactly. Anchors are sampled uniformly from base Gaussians with a
fixed RNG seed (77 by default). Offsets are sampled in each anchor's rotated
covariance frame, with jitter 0.5; extra scales are multiplied by 0.5, appearance
and rotation are inherited, and opacity starts at 0.02. All are configurable.
No candidate or test image is used to place these initial points. The same
augmented PLY is shared across candidates and budgets with identical settings.

Extra capacity near reconstructed surfaces does not guarantee recovery of an
entirely unseen surface. Densification remains subject to FastGS's gradient and
consistency criteria. Compare the plain and augmented methods to measure whether
the extra points help on your data.

The default primary gain reference is now the **unchanged original base**.
`--base-control` adds a separate base-only adaptation to each warm method. Its
metrics and time are recorded separately and do not replace the reference or
enter candidate totals. This isolates improvement caused by additional
optimization without changing the main comparison.

## Resume, evaluation-only, and report-only

Rerunning the same command skips completed matching reconstructions and metrics.
Training receipts contain settings, commands, wall times and environment data.
The benchmark fingerprints input manifests/images and FastGS source; warm runs
also identify the renderer adapter and initialization PLY contents. Changed
settings invalidate affected runs. Changing padding settings only retrains that
method's candidates. An interrupted reconstruction restarts from its configured
initialization rather than an intermediate checkpoint.

Evaluation fingerprints checkpoint bytes, selected training cameras and held-out
camera/image bytes. This prevents stale quality scores after a checkpoint or
corpus changes. The reporter rejects different bases, camera sets, test images,
metric iterations or recorded timing environments. It matches actions by lattice
index, not just candidate names. Missing results fail explicitly by default.
Unknown historical times stay `N/A`.

Separate cluster training and evaluation if useful:

```bash
python benchmark_reconstructions.py <same experiment options> --train-only
python benchmark_reconstructions.py <same experiment options> --evaluate-only
```

Regenerate reports without CUDA, the source dataset or FastGS:

```bash
python benchmark_reconstructions.py \
  --output-root benchmarks/wave_vase --report-only
```

Report-only reuses the saved config and selected methods/contexts. For existing
result directories independently of the runner:

```bash
python compare_reconstructions.py \
  --results-root benchmarks/wave_vase/results \
  --reference fresh-30k --method base-cloud-15k --method base-extra-15k \
  --output-dir benchmarks/wave_vase/report
```

`--allow-incomplete` on this reporting command compares available candidate
intersections and explicitly lists omitted actions/contexts. It still rejects
incompatible provenance. `--redo-metrics` forces benchmark reevaluation.
`--force-retrain` retrains method candidates; `--force-base-retrain` rebuilds each
shared base once per context. Use a separate output root when you want a new
measured timing repeat rather than resuming cached results.

## Other device layouts

All paths default relative to the renderer checkout and can be overridden:

```bash
python benchmark_reconstructions.py \
  --object-id "Wave Vase" --experiment-id furthest_8_seed_77 \
  --method base-cloud-15k --method base-extra-15k \
  --views-root /data/views \
  --base-reconstructions-root /data/existing/reconstructions \
  --fastgs-dir /work/FastGS \
  --python /path/to/fastgs-env/bin/python \
  --output-root /data/benchmarks/wave_vase
```

Each object needs the two global transform manifests and their referenced PNGs.
The view directory must be writable because FastGS expects temporary root-level
manifests. These are restored after each training/evaluation call. Do not run
concurrent jobs against the same object's manifests or the same output namespace;
use separate dataset copies for concurrent comparisons.

Transfer source and datasets rather than a venv or compiled CUDA extensions.
The existing FastGS environment supplies NumPy, Pillow, plyfile, PyTorch and the
native extensions; this benchmark adds no new third-party dependency.

Standalone `orchestrator.py` also supports `--candidate-initialization base`,
`--candidate-iterations`, the `--adaptation-*` settings, `--extra-gaussians` and
`--extra-gaussian-*`, `--base-control`, `--run-name`, and all path overrides.
The former equal-budget baseline behavior remains explicitly available as
`--gain-reference equal-budget-base-control`, but it is not accepted as the
primary reference by the matched benchmark reporter.

Benchmark results record actual, potentially different baseline/candidate
iterations. The predictor's existing preprocessing rejects mixed metric iterations;
these benchmark records need protocol-aware import before being used for predictor
training. Do not relabel checkpoint iterations to bypass that check.

## CPU test suite

```bash
python -m unittest discover -s tests -v
```

Tests cover subprocess orchestration, independent candidates, original-base and
secondary-control semantics, deterministic Gaussian augmentation and base
preservation, cache invalidation, evaluation commands and manifest restoration,
metric signs, screenshot arithmetic, duplicate exclusion and ranking ties.
