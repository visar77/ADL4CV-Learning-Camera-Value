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
| [comparison_base_cloud_and_fresh_candidates.py](benchmarks/comparison_base_cloud_and_fresh_candidates.py) | Compare fresh candidate training with 3k-step adaptation from a shared base cloud. |
| [evaluate_metrics.py](evaluate_metrics.py) | Evaluate saved reconstructions independently of training. |
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
`blender` on `PATH`; otherwise set `blender_exe` in `config.json`. It launches
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

Render one object by passing its GLB explicitly:

```bash
python renderer.py --glb-path objects/gamecube.glb --object-id gamecube
```

`--object-id` is required. If `--glb-path` is omitted, the renderer uses
`objects_root/<object-id>.glb` from `config.json`:

```bash
python renderer.py --object-id gamecube
```

`--output-root` selects a separate corpus root. Edit `config.json` to change
renderer settings without modifying the scripts. It controls the source-object
root, output root, Blender executable, image dimensions, lattice size,
horizontal field of view, trajectory elevation, camera-distance scale,
blend-file saving, and GPU use. The worker is always `blender_worker.py`.
Relative config paths are resolved from `ObjaverseRenderer/`. The experiment
viewer uses the configured output root, lattice size, and trajectory elevation
(object metadata still takes precedence when available). Defaults produce
512 × 512 RGBA PNGs and 100 Fibonacci-lattice training cameras, with 50°
horizontal field of view, trajectory elevation 20°, and camera-distance scale
1.15. The test set consists of **eight fixed cube-corner directions**,
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
one reconstruction adding six views. Standalone reconstruction directories use
strategy and action number, such as `random-4`. Candidate indices are the
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

The benchmark entry point is
[comparison_base_cloud_and_fresh_candidates.py](benchmarks/comparison_base_cloud_and_fresh_candidates.py).
Its matching configuration is
[comparison_base_cloud_and_fresh_candidates.json](benchmarks/comparison_base_cloud_and_fresh_candidates.json).
Each benchmark's generated data is kept in a same-named subfolder under
[benchmarks/](benchmarks/).

The comparison trains a shared base once (30,000 iterations by default), then
trains each selected candidate in two ways:

| Method | Candidate initialization | Candidate steps |
| --- | --- | ---: |
| `fresh` | Scratch | 30000 |
| `base-cloud-3k` | The exact shared base PLY, with a fresh optimizer | 3000 |

Both candidates use exactly the same base views, added view, and held-out test
corpus. The report includes absolute PSNR/SSIM/LPIPS, gains over the shared base,
paired gains versus fresh training, and training times. Candidate rankings and
selection regret are also reported when the context has multiple candidates.
Candidates are paired by their deterministic strategy/action IDs (for example
`random-1`); the benchmark verifies that both runs used the same training
cameras for each ID.

From `ObjaverseRenderer/`, validate a selected context and run it with:

```bash
python benchmarks/comparison_base_cloud_and_fresh_candidates.py \
  --object-id "Table" --experiment-id random_3_seed_77 --dry-run
python benchmarks/comparison_base_cloud_and_fresh_candidates.py \
  --object-id "Table" --experiment-id random_3_seed_77
```

Without filters, the `objects` and `experiments` lists in the matching config
are selected. The config must list only the experiments you want to run; its
default experiment is `random_3_seed_77`. Repeat `--object-id` or
`--experiment-id` to select more than one and override the corresponding config
list. The script accepts `--views-root`,
`--experiments-dir`, `--fastgs-dir`, and `--python` when the standard paths
differ. A dry run validates the selected corpora and prints the planned matrix
without writing outputs or starting training.

Edit the matching JSON config to change `base_iterations`,
`fresh_candidate_iterations`, `adapted_candidate_iterations`, the candidate
actions, the `objects` and `experiments` lists, and adaptation schedule. The
benchmark rejects redundant-view controls because they are not candidate
comparisons. Different settings should use a separately named benchmark script
and matching config rather than mixing runs in one output folder.

### FastGS run-to-run variability

[fastgs_reconstruction_variability.py](benchmarks/fastgs_reconstruction_variability.py)
repeats selected `base` or candidate reconstructions with both fresh 30k
training and base-cloud 3k adaptation. Configure `repetitions`, `objects`,
`experiments`, and `candidate_ids` in
[fastgs_reconstruction_variability.json](benchmarks/fastgs_reconstruction_variability.json).
The repetition count applies to every selected object/experiment/candidate for
both methods. FastGS uses its existing default RNG initialization for every
training process; this benchmark does not set per-repetition training seeds.

Each object/experiment gets one 30k base-cloud reconstruction, which is
preserved and reused to initialize every selected 3k candidate adaptation and
repetition. It is evaluated once. Selecting `base` includes this shared model
in the report; fresh-30k still independently repeats each selected target.
The report summarizes Gaussian counts and PSNR/SSIM/LPIPS means, population
standard deviations, and ranges for each selected target and method.

Run a plan first, then execute:

```bash
python benchmarks/fastgs_reconstruction_variability.py --dry-run
python benchmarks/fastgs_reconstruction_variability.py
```

Each execution writes to a new timestamped run directory under
`benchmarks/fastgs_reconstruction_variability/`. Reconstructions, per-run
`results.json` files, the config snapshot, `report.json`, and `report.txt` are
kept in that directory; an existing run directory is never reused or
overwritten.

Outputs are written under
`benchmarks/comparison_base_cloud_and_fresh_candidates/`:

```text
methods/<method>/reconstructions/<object>/<experiment>/...
methods/<method>/results/<object>/<experiment>/results.json
report/comparison.{json,csv,txt}
benchmark_config.json
benchmark_manifest.json
```

Rerunning the same command reuses complete checkpoints and metrics. Use
`--retrain` to rebuild checkpoints; changing the saved configuration is
rejected so that results from different protocols cannot be mixed. Evaluation uses the same test corpus and records checkpoint/camera provenance.
Rerunning the benchmark with the same configuration reuses completed training
and evaluation outputs and regenerates the report.

Positive PSNR/SSIM/LPIPS gain means improvement; LPIPS gain is base minus
candidate. The comparison's `delta_gain` is adapted gain minus fresh gain, so
negative values indicate worse gains after adaptation. Training times exclude
evaluation and rendering. Inspect rendered test images as well as metrics;
measured speed does not imply acceptable reconstruction quality or equivalent
fresh-training view-selection fidelity.

## Orchestration and evaluation

`orchestrator.py` reads its data/output paths from
[orchestrator_config.json](orchestrator_config.json). Paths are relative to
`ObjaverseRenderer/`. It discovers all GLBs and experiments when their IDs are
omitted, rendering missing corpora automatically.

```bash
python orchestrator.py \
  --object-id gamecube \
  --experiment_id random_3_seed_77 \
  --candidate_id random-1
```

The supported switches are `--retrain`, `--rerender`,
`--use-candidate-base [base-3k]`, `--evaluate_metrics`, and `--redo`. Existing
reconstructions are reused unless `--retrain` is specified; missing or
incomplete reconstructions are trained. `--rerender` forces corpus generation.
Candidate-base mode trains/reuses the regular `base/` reconstruction, then
adapts candidates for 3k steps from its PLY. If `base/` is missing, it is trained
first; a candidate-only run does not retrain an existing base. If
`base_cloud_ply` is set in `orchestrator_config.json`, that external PLY is used
directly instead and is never copied into the output tree or recorded in result
files. Without candidate-base mode, each candidate trains from scratch.

The result layout is:

```text
reconstructions/<object-id>/<experiment-id>/
  base/
  random-1/
  next-1/
  ...
results/<object-id>/<experiment-id>/results.json
```

FastGS locations, training budgets, and the small set of optimizer defaults
are in [training_config.json](training_config.json). `training_fastgs.py` uses
FastGS's native trainer for scratch training and the FastGS adapter for
PLY-initialized training. Add `--evaluate_metrics` to evaluate all result
records under the configured results root; add `--redo` to force recomputation.

## Implementation layout

| Module | Responsibility |
| --- | --- |
| `orchestrator.py`, `orchestrator_config.json` | Dataset discovery, candidate selection, manifests, paths, and experiment execution. |
| `training_fastgs.py`, `training_config.json` | FastGS scratch/adaptation training launcher and defaults. |
| `benchmarks/` | One executable script per benchmark, matching JSON config, report helpers, and same-named output folders. |
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

The CPU suite tests benchmark configuration/input selection, report matching,
training orchestration, PLY manipulation, and manifest restoration. The CUDA
adapter test is opt-in. Passing CPU tests does not establish GPU compatibility,
reconstruction quality, or measured speedup.

The current implementation does not resume intermediate optimization states,
expose independent optimizer seeds, or create a separate validation split.
It evaluates one held-out test corpus. Parameter tuning and a final unbiased
quality assessment require separate validation/test data. Warm quality and
faithfulness to fresh candidate rankings must be assessed separately before
using warm gains as labels for fresh-training view selection.
