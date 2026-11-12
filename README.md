# ObjaverseRenderer

ObjaverseRenderer renders shared multi-view corpora from `.glb` objects, trains
FastGS reconstructions from selected views, and compares reconstruction quality
and training time. Each candidate reconstruction adds **one view to the same
base view set**; candidates are independent, not a cumulative acquisition path.

**Collaborators: follow [BENCHMARK_NEXT_STEPS.md](BENCHMARK_NEXT_STEPS.md) for the
current execution plan, expected outputs, and decisions.** This README is the
functionality reference.

| Entry point | Purpose |
| --- | --- |
| [renderer.py](renderer.py) | Render an object's complete train/test corpus through Blender. |
| [orchestrator.py](orchestrator.py) | Optionally render missing corpora, then train base and candidate reconstructions. |
| [benchmark_reconstructions.py](benchmark_reconstructions.py) | Run matched methods across an object × experiment matrix, evaluate, and report. |
| [evaluate_metrics.py](evaluate_metrics.py) | Evaluate saved reconstructions independently of training. |
| [compare_reconstructions.py](compare_reconstructions.py) | Compare saved, evaluated method results without training or CUDA. |
| [experiment_viewer.py](experiment_viewer.py) | Visualize an experiment's selected training cameras. |

## Requirements and setup

Use the existing FastGS Python/CUDA environment for training and evaluation.
It supplies NumPy, Pillow, plyfile, PyTorch, torchvision, LPIPS, and the native
FastGS CUDA extensions. The renderer launcher, sampling, configuration, and
reporting use the Python standard library. The camera viewer additionally needs
NumPy and Matplotlib. No FastGS source modifications are required.

FastGS defaults to the sibling checkout `../FastGS`. Training/evaluation commands
accept `--fastgs-dir /path/to/FastGS` and `--python /path/to/gpu-env/bin/python`.
Activate that environment when invoking standalone GPU checks. Update the whole
repository checkout, including the helper modules, rather than copying a single
entry-point script.

Blender is needed only to generate or regenerate views. `renderer.py` finds
`blender` on `PATH`; otherwise set `BLENDER_EXE` in that file. It launches
`blender_worker.py` with `--background --factory-startup --python-exit-code 1`
and propagates worker failures. Existing rendered corpora need neither Blender
nor the source GLBs for training or benchmarking.

All example commands below run from `ObjaverseRenderer/`. Relative data paths
are examples; script defaults resolve from the repository directory.

## Objects and rendered corpora

Place GLBs in `objects/`; the filename without `.glb` becomes the object ID.
Quote IDs containing spaces. Alternatively, download source GLBs with
[download_gbuffer_objaverse.py](download_gbuffer_objaverse.py):

```bash
python download_gbuffer_objaverse.py --count 5 --seed 77
python download_gbuffer_objaverse.py --object-name 148/753974
```

The downloader retrieves source assets from the G-buffer Objaverse mapping,
uses their Objaverse names, and resumes existing GLBs. `--object-name` accepts
an ID or unique substring; duplicate names receive ID suffixes. `--output-dir`,
`--index`, and `--mapping` override the destination and metadata sources.
Metadata is streamed/held in memory rather than retained as a local dataset.

Render one object:

```bash
python renderer.py --glb-path objects/gamecube.glb --object-id gamecube
```

`--output-root` selects a separate corpus root. Renderer settings are constants
in `renderer.py`: 512 × 512 RGBA PNGs, 100 Fibonacci-lattice training cameras,
50° horizontal field of view, trajectory elevation 20°, and camera-distance
scale 1.15. The test set consists of **eight fixed cube-corner directions**,
independent of the training lattice. The camera points toward the object's
bounding-box center and uses a distance based on its bounding radius. Cycles
uses CUDA when available and otherwise falls back to CPU.

```text
views/<object>/
  images/train/000.png ...
  images/test/000.png ...
  global_transforms/transforms_train.json
  global_transforms/transforms_test.json
  metadata.json
  _worker_config.json
  rendered_scene.blend
```

`rendered_scene.blend` is optional (`SAVE_BLEND_FILE`). `metadata.json` records
the object, lattice size, trajectory elevation, lighting mode, and light
energies. Global manifests contain `camera_angle_x` and `frames`; each frame
has a relative `file_path` without `.png` and a 4×4 camera-to-world
`transform_matrix`. All experiments on an object reuse these images/cameras.

### Lighting and regeneration

The renderer uses the supplied Objaverse-XL four-SUN rig: white key, fill, rim,
and bottom lights, reference rotations/energy choices, shadows disabled, and
specular factor 1.0. It retains original GLB materials and the factory world.
Cycles uses 128 samples, diffuse/glossy bounces 1, transparent/transmission
bounces 3, filter width 0.01, denoising, and transparent film. See
[LIGHTING.md](LIGHTING.md) for the reference and optional Blender parity check.

Lighting is sampled once per object and fixed across that object's views.
There is **no explicit lighting RNG seed**: rerendering may change energies.
Reuse one unchanged corpus across methods and timing repeats. Render to a new
views root when changing geometry, camera settings, lattice size, or lighting;
the renderer does not remove stale images. `orchestrator.py --rerender` forces
regeneration, whereas changing experiment definitions only changes view subsets.

FastGS requires `transforms_train.json` and `transforms_test.json` temporarily
at the object root. Training/evaluation create these from the global manifests
and restore their previous contents afterward, including on failure; files
absent before the call are removed. Do not run concurrent jobs against the
same object's root manifests or the same output namespace. Use separate corpus
copies and output roots for parallel allocations.

## Experiments and candidate views

An experiment JSON in [experiments/](experiments/) defines base-view selection:

```json
{
  "id": "random_3_seed_77",
  "train": {"strategy": "random", "count": 3, "seed": 77},
  "candidates": [
    {"strategy": "blindspot"},
    {"strategy": "next", "count": 1},
    {"strategy": "random", "count": 1}
  ]
}
```

The file stem supplies `id` if omitted. IDs must be unique, single directory
names. The standalone orchestrator uses the experiment's candidate settings;
without candidates it trains only the base. The benchmark overrides candidates
with its config so every method receives matched actions.

| Base strategy | Selection |
| --- | --- |
| `random` | Seeded sampling without replacement from the training lattice. |
| `furthest` | Seeded farthest-point sampling over the sphere. |
| `trajectory` | A seeded ring, mapped to the nearest unused lattice points. More views preserve the earlier trajectory prefix. |

Trajectory orientation/start phase use the experiment seed; elevation comes
from renderer metadata, defaulting to 20°. The seed controls camera selection,
not training randomness.

| Candidate strategy | Action |
| --- | --- |
| `blindspot` | One unused lattice point with the largest minimum distance to the base views. |
| `random` | Seeded sampling of unused lattice points. |
| `next` | Continue the seeded trajectory after a trajectory base; otherwise start at the beginning and skip base cameras. |
| `redundancy` | Base-view aliases used as controls; excluded from matched candidate comparisons. |

A candidate `count` of 6 means six independent base-plus-one-view actions, not
one reconstruction adding six views. Names encode strategy, requested count,
and action number, such as `random-6-4`. Duplicate physical candidate indices
share one reconstruction/evaluation and count once. Candidate indices are the
matching identity across methods, not their labels.

View the base cameras without rendering or training:

```bash
python experiment_viewer.py \
  --experiment random_3_seed_77 --object-id gamecube \
  --show-lattice --show-test-views
```

`--experiment` accepts an ID or JSON path. The viewer saves a PNG by default;
`--output` selects its path and `--show` opens an interactive window. It uses
object metadata for lattice size/elevation when available; `--lattice-size`
overrides size. Candidate cameras are not displayed. Trajectory plots show
the seeded ring and mark the first/most recent base camera.

## Matched benchmarking

### Protocol and default methods

[benchmark_reconstructions.py](benchmark_reconstructions.py) trains one scratch
base per object/experiment context and reuses it unchanged across methods:

| Method | Candidate initialization | Candidate steps |
| --- | --- | ---: |
| `fresh-30k` | Scratch | 30000 |
| `base-cloud-3k` | Trained base PLY with fresh optimizer state | 3000 |

Both methods train on exactly the base views plus the same one candidate and
are evaluated on the same test corpus. The shared base uses 30000 steps.
The default config is
[benchmark_configs/base_initialization.json](benchmark_configs/base_initialization.json);
[benchmark_configs/validation.json](benchmark_configs/validation.json) fixes
the matrix used in the next-steps plan.

With `--base-control`, the warm method additionally adapts **base views alone**
for the same 3000 steps, from the same initializer. It is a separate diagnostic,
not another candidate or a replacement for the unchanged primary base.

### Selection and configuration

```bash
python benchmark_reconstructions.py \
  --object-ids "Table" "Wave Vase" \
  --experiment-ids random_3_seed_77 random_5_seed_77 \
  --base-control \
  --output-root benchmarks/example
```

This selects the Cartesian product: four contexts and both default methods.
The singular `--object-id` and `--experiment-id` flags also accept lists or
repetition. Duplicate requested IDs are deduplicated; unknown IDs and duplicate
experiment definitions fail before training. Without CLI filters, config
`objects`/`experiments` lists apply; without those, all rendered objects and all
experiment definitions are discovered. CLI filters override the corresponding
config lists. `--max-contexts N` takes the first N selected contexts;
`--candidate-limit N` caps distinct non-redundant candidates per context.

Copy a benchmark config and pass `--config my_config.json` to customize it:

| Top-level field | Meaning |
| --- | --- |
| `schema_version` | Must be 1. |
| `reference` | ID of the scratch reference method. Always included in a selected method set. |
| `base_iterations` | Shared base training budget. |
| `objects`, `experiments` | Optional nonempty ID lists. |
| `candidates` | Required candidate definitions; override every selected experiment's actions. |
| `adaptation_defaults` | Default warm learning-rate/densification options. |
| `methods` | At least a scratch reference and one comparison method, each with a unique ID. |

Method fields are `id`, `initialization` (`scratch` or `base`), and `iterations`.
Optional overrides are `lr_scale`, `densify_from_iter`, `densify_until_iter`,
`densification_interval`, `extra_gaussians`, `extra_gaussian_seed`,
`extra_gaussian_jitter`, `extra_gaussian_opacity`, and `extra_gaussian_scale`.
Method values override `adaptation_defaults`. Extra Gaussians require base
initialization. Custom scratch budgets or warm settings are supported but are
not part of the current two-method execution plan. `--method ID` can be repeated
to select configured methods; it does not define methods absent from the config.

### Warm adaptation behavior

[fastgs_adapter.py](fastgs_adapter.py) imports the selected FastGS implementation
and runs its losses, rasterizer, model, and densification routines without
editing or monkey-patching FastGS. It retains trained Gaussians/full SH degree
but resets optimizer moments and gradient/radius buffers. Position learning-rate
decay spans the adaptation budget. Default adaptation settings are:

- All configured learning rates scaled by 0.1.
- Densification starts after step 100, ends at half the budget (exclusive),
  and is considered every 250 steps. Set `densify_until_iter` to 0 to disable it.
- Main parameters update each step; SH features every 16 steps and on the final
  step. Final checkpoints include the final optimization update.
- Global opacity resets and aggressive late scratch pruning are skipped.

Changing the budget also changes the default decay/densification horizons.
Both scratch and warm training currently call FastGS `safe_state`, setting
Python, NumPy, and PyTorch seeds to 0. Independent optimizer seeds are not
exposed; new output directories repeat timings under the same seeds.

Optional extra Gaussians are written to a separate cached PLY; base rows are
preserved. Defaults sample anchors uniformly from base Gaussians with seed 77,
apply covariance-frame jitter 0.5, scale 0.5, inherited appearance/rotation, and
opacity 0.02. Placement uses no candidate/test RGB. Identical settings share the
augmented initializer across candidates/budgets. Local extra capacity does not
explicitly seed entirely unseen surfaces. The default methods add no points.

### Modes, preflight, and paths

The normal benchmark trains, evaluates, and writes reports. These modes are
mutually exclusive:

| Flag | Behavior |
| --- | --- |
| `--dry-run` | Validate inputs and print pending commands; no output writes, GPU loading, or training. FastGS source and corpora are still required. |
| `--train-only` | Train and save records; defer evaluation/reporting. |
| `--evaluate-only` | Evaluate saved reconstructions and report; do not train. |
| `--report-only` | Rebuild reports from saved evaluated records, without GPU or source corpora. |

Before real training/evaluation, preflight checks CUDA availability, FastGS
imports/API signatures, the KNN CUDA kernel, and fused SSIM forward/backward.
`environment.json` records the GPU, compute capability, PyTorch/CUDA/Python,
checkout location, and host. `--gpu-smoke-test` also runs the opt-in CUDA tests;
`--skip-preflight` disables the preflight, not the need for a valid environment.
LPIPS may download pretrained weights on first use.

Data/path overrides are `--views-root`, `--experiments-dir`, `--fastgs-dir`,
`--python`, `--output-root`, and `--base-reconstructions-root`. By default the
shared bases live under `<output-root>/shared_base`. An external base root must
contain `<object>/<experiment>/base/point_cloud/iteration_30000/point_cloud.ply`
for the default budget. PLYs without receipts are accepted only as 30k scratch
outputs; their original settings/time cannot be verified. New controlled runs
should create their own shared bases.

### Outputs and resume

```text
<output-root>/
  benchmark_config.json
  benchmark_manifest.json
  environment.json
  shared_base/<object>/<experiment>/base/...
  reconstructions/<method>/<object>/<experiment>/<candidate>/...
  reconstructions/base-cloud-3k/<object>/<experiment>/base_control/...
  results/<method>/<object>/<experiment>/results.json
  report/comparison.txt
  report/comparison.csv
  report/comparison.json
```

`base_control/` exists only with the flag; `initializations/<object>/<experiment>/
<content-hash>/` is created only for custom extra-Gaussian methods. Each training
directory contains `training_run.json` and
`point_cloud/iteration_<budget>/point_cloud.ply`. Receipts store completion,
settings, command, timing, and environment. Result records include protocol,
initializer provenance, base/candidate camera indices, aliases, status, metrics,
metric iterations, and evaluation provenance.

Rerun the **same command and output root**, retaining `--base-control` when used,
to reuse matching complete reconstructions and metrics. The benchmark checks
input manifests/images, selected FastGS source files, and warm adapter/initializer
contents. Setting/input changes invalidate affected runs. Interrupted training
restarts that reconstruction from its configured initialization, not an
intermediate checkpoint. `--force-retrain` rebuilds method candidates/controls;
`--force-base-retrain` rebuilds each shared base once before methods use it.
`--redo-metrics` reevaluates matching saved outputs.

```bash
python benchmark_reconstructions.py --output-root benchmarks/example --evaluate-only
python benchmark_reconstructions.py --output-root benchmarks/example --report-only
```

Both commands reuse the stored config, selected methods, and **exact recorded
context pairs**, including partial matrices from `--max-contexts`. Explicit
object/experiment filters can narrow them. Evaluation-only also restores the
recorded views root unless overridden; keep experiment definitions available,
and repeat `--experiments-dir` for a custom directory. Changed corpus inputs
are rejected. Reports reject mismatched cameras/test corpora, replaced original
bases, incompatible iterations, and inconsistent recorded timing environments.
Missing contexts/candidates fail by default. Report-only does not verify live
checkpoint files or images; it consumes saved metrics/provenance.

Run records contain absolute checkpoint paths. When transferring trained
outputs for evaluation, preserve those paths or update them consistently,
including the separately shared bases. For benchmark report-only, copy the
`results/` tree plus `benchmark_config.json` and `benchmark_manifest.json`;
checkpoints and source images are unnecessary. Transfer source/data rather
than compiled CUDA extensions or virtual environments, and use the target
device's FastGS setup.

### Metrics, controls, and report interpretation

For PSNR/SSIM, gain is candidate minus unchanged base. For LPIPS, gain is
unchanged base minus candidate, so **positive gain always means improvement**.
`delta_gain` is method gain minus fresh-reference gain. Negative gains are
retained; neither adding a view nor warm adaptation guarantees improvement.

The JSON report includes absolute quality, gains, MAE, negative PSNR-gain
counts, gain-sign agreement, PSNR rank correlation, top-choice agreement,
selection regret, and training times. The text report summarizes gain changes,
selection statistics, and timing. The CSV contains one row per distinct matched
candidate, with absolute scores, primary gains, deltas, camera index, and times.
With a base-only control, JSON additionally exposes:

- `base_control_metrics`: absolute control quality.
- `summary.base_control_gain`: control gain over the unchanged base.
- `rows[].method_gain_over_control`: each warm candidate's gain over the adapted
  base-only control, with the same positive-is-better convention.
- `summary.mean_method_gain_over_control`: mean effect beyond that control.

The text report prints control gains and mean candidate effects; these
additional control fields are not added to the CSV.

Subtracting one shared control does **not** change within-context rankings.
It helps distinguish new-view effects from further optimization. Controls must
match the warm candidate budget, base training cameras, and test corpus.

Top-choice agreement is the fraction of warm-best candidates that are also
fresh-best, equivalent to uniform tie-breaking among warm ties. Selection
regret is best fresh PSNR gain minus the mean fresh gain of those warm-best
candidates. Rank correlation is undefined for fewer than two candidates or
constant ranks. Overall quality/time summaries are candidate-weighted; overall
top-choice agreement and regret are context-weighted.

Candidate time measures the training subprocess, including startup, loading,
optimization, and saving. It excludes corpus rendering and separate evaluation.
Candidate speedup is summed fresh time divided by summed method time. Shared
base, initializer preparation, and control timings are separate; production
totals add shared base and initializer preparation but exclude the diagnostic
control. Cached outputs retain their measured times; unknown timing is `N/A`,
not zero. Use a **new output root** for a new timing measurement.

Compare evaluated method results directly, independently of the benchmark runner:

```bash
python compare_reconstructions.py \
  --results-root benchmarks/example/results \
  --reference fresh-30k --method base-cloud-3k \
  --output-dir benchmarks/example/report
```

`--allow-incomplete` compares available candidate/context intersections and
reports omissions while still rejecting incompatible provenance. The comparison
requires `gain_reference=original_base`. Benchmark records with mixed 30k/3k
iterations need protocol-aware preprocessing before predictor training; do not
relabel iterations to satisfy an equal-budget importer.

## Standalone orchestration and evaluation

Use `orchestrator.py` when generating views or training outside a matched
benchmark. By default it discovers `.glb` files in `objects/`, renders missing
corpora, and applies every experiment definition to every object, sequentially.
It uses scratch initialization and 30000 steps for both base and candidates.
With `--skip-render`, it discovers existing rendered objects instead.

```bash
python orchestrator.py \
  --skip-render --object-ids gamecube chair \
  --experiment-ids random_3_seed_77 --continue-on-error
```

Outputs default to `reconstructions/<object>/<experiment>/<base-or-candidate>/`
and `results/<object>/<experiment>/results.json`. A complete matching receipt
and final PLY enable reconstruction-level resume; 30k scratch PLYs without
receipts are also accepted. `--continue-on-error` continues after experiment
failures. `--rerender` forces corpus regeneration; `--force-retrain` forces run
reconstructions, while `--force-base-retrain` controls shared initializers.

Standalone warm adaptation:

```bash
python orchestrator.py \
  --skip-render --object-ids gamecube --experiment-ids random_3_seed_77 \
  --candidate-initialization base --candidate-iterations 3000 \
  --run-name base-cloud-3k --base-control
```

`--base-iterations` defaults to 30000. `--candidate-iterations` defaults to 3000
for base initialization, otherwise the base budget. `--run-name` adds an output
namespace; warm mode generates `base_init_<budget>` when omitted. Shared bases
remain outside that namespace. `--shared-base` enables shared bases for scratch
runs too. `--gain-reference original-base` is the default;
`equal-budget-base-control` changes the standalone gain reference and is not
accepted by the matched reporter.

Standalone overrides include `--views-root`, `--objects-root`,
`--experiments-dir`, `--reconstructions-root`, `--results-root`,
`--base-reconstructions-root`, `--initializations-root`, `--fastgs-dir`, and
`--python`. Adaptation uses `--adaptation-lr-scale`,
`--adaptation-densify-from-iter`, `--adaptation-densify-until-iter`, and
`--adaptation-densification-interval`. Extra-point options are
`--extra-gaussians`, `--extra-gaussian-seed`, `--extra-gaussian-jitter`,
`--extra-gaussian-opacity`, and `--extra-gaussian-scale`. `--candidate-limit`
and `--dry-run` support small/read-only checks; inspect `--help` for all options.

The scratch subprocess uses `--eval --optimizer_type default`, densification
interval 500, absolute gradient threshold 0.0012, and a final save at its budget.
Training-time evaluation is off (`--test_iterations -1` for scratch);
`--evaluate-during-training` enables final-step evaluation. The benchmark keeps
evaluation separate.

Evaluate standalone outputs later:

```bash
python evaluate_metrics.py \
  --object-id gamecube --experiment-id random_3_seed_77
```

The evaluator reads reconstruction camera indices, temporarily exposes the
manifests, runs FastGS rendering and PSNR/SSIM/LPIPS evaluation, and writes
per-view scores and averages under `metrics[<reconstruction-name>]`.
`metrics_iterations` records the budget for each reconstruction;
`metrics_provenance` identifies checkpoint bytes, training cameras, test
images/cameras, evaluator code, and iteration. The scalar `metrics_iteration`
is retained only while evaluated budgets agree. By default, each reconstruction's
recorded final iteration is used. An explicit `--iteration` overrides all of them.

`--results-root` selects another results tree; `--views-root` overrides dataset
locations. `--reconstructions-root` expects an object/experiment/run directory
layout and is useful for relocated standalone outputs; benchmark shared bases
need their own correct recorded paths. `--redo` forces evaluation; compatible
cached evaluations are reused and duplicate aliases are evaluated once. Without
filters, all result records under the chosen root are considered.

## Implementation layout

| Module | Responsibility |
| --- | --- |
| `reconstruction/config.py` | Standalone CLI defaults, validation, and path resolution. |
| `reconstruction/datasets.py` | Corpus discovery, experiment loading, view selection, temporary manifests. |
| `reconstruction/training.py` | FastGS commands, training specs, receipts, completion checks. |
| `reconstruction/experiments.py` | Shared initializers, candidate planning, result records, experiment execution. |
| `reconstruction/io.py` | Atomic JSON records and content hashes. |
| `benchmark_config.py` | Benchmark schema validation and method options. |
| `benchmark_contexts.py` | Matrix selection, input fingerprints, exact saved-context replay. |
| `fastgs_adapter.py`, `fastgs_schedule.py` | PLY initialization and the adaptation loop/schedule. |
| `gaussian_initialization.py` | Optional cached extra-Gaussian initialization. |
| `sampling.py` | Training lattice, fixed test directions, base/candidate selection. |

Generated objects, corpora, reconstruction/results trees, benchmark outputs,
and Python caches are excluded by `.gitignore` for the standard output names.

## Verification and limits

```bash
python -m unittest discover -s tests -v
python gpu_preflight.py --fastgs-dir ../FastGS
FASTGS_RUN_GPU_TESTS=1 FASTGS_DIR=../FastGS python -m unittest discover \
  -s tests -p test_fastgs_adapter_gpu.py -v
```

The CPU suite tests subprocess orchestration with a stand-in trainer, PLY
manipulation, controls, camera/metric matching, resume/invalidation, manifest
restoration, matrix selection, and reports. The two CUDA tests are opt-in;
they test actual adaptation/densification and a tiny train/evaluate/report
benchmark, including custom extra-point initialization. Passing CPU tests does
not establish GPU compatibility, reconstruction quality, or measured speedup.

The current implementation does not resume intermediate optimization states,
expose independent optimizer seeds, or create a separate validation split.
It evaluates one held-out test corpus. Parameter tuning and a final unbiased
quality assessment require separate validation/test data. Warm quality and
faithfulness to fresh candidate rankings must be assessed separately before
using warm gains as labels for fresh-training view selection.
