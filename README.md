# ObjaverseRenderer

ObjaverseRenderer generates reproducible multi-view datasets from `.glb` assets
and prepares them for FastGS reconstruction experiments. It is designed for a
later view-selection/gain-prediction dataset:

- Render every pose in a global camera lattice once.
- Define reusable experiments as JSON files.
- Select a train subset by indices for each experiment.
- Train one FastGS reconstruction per object and experiment.
- Reuse the same rendered images and test views across all experiments.
- Record the exact images used by each reconstruction.
- Evaluate PSNR, SSIM, and LPIPS later, independently of reconstruction.

## TL;DR

### 1. Put objects in the objects directory

Place one or more GLB files in:

```text
ObjaverseRenderer/objects/
```

The filename becomes the object ID. For example:

```text
objects/gamecube.glb
objects/chair.glb
```

### 2. Configure experiments

Experiment definitions live in:

```text
ObjaverseRenderer/experiments/
```

Each JSON file describes a train-view sampling strategy and seed. For example:

```json
{
  "id": "furthest_8_seed_77",
  "train": {
    "strategy": "furthest",
    "count": 8,
    "seed": 77
  }
}
```

The supported strategies are currently:

- `random`
- `furthest`
- `trajectory`

Every JSON file in `experiments/` is run for every object.

### 3. Run the orchestrator

From the `ObjaverseRenderer` directory:

```bash
python3 orchestrator.py
```

This will:

1. Render the global view corpus for every object that needs it.
2. Create a temporary `transforms_train.json` for each experiment.
3. Run FastGS training.
4. Store the reconstruction and an experiment result record.
5. Delete the temporary train manifest.

To continue a previous cluster run:

```bash
python3 orchestrator.py --skip-render --continue-on-error
```

Completed reconstructions are skipped automatically.

### 4. Evaluate metrics later

After reconstructions are complete:

```bash
python3 evaluate_metrics.py
```

Or evaluate one object/experiment:

```bash
python3 evaluate_metrics.py \
    --object-id gamecube \
    --experiment-id furthest_8_seed_77
```

Metrics are written into:

```text
results/<object-id>/<experiment-id>/results.json
```

## Requirements

### Blender

The renderer must be run through Blender because
[blender_worker.py](./blender_worker.py) imports `bpy` and `mathutils`.

If Blender is available on `PATH`, no configuration is needed. Otherwise,
configure its executable in [renderer.py](./renderer.py):

```python
BLENDER_EXE = "/path/to/blender"
```

The default is `None`, which finds Blender through `PATH`.

### Python

The renderer/orchestrator use the Python standard library plus the local
sampling module. The FastGS loader and metrics pipeline require the dependencies
described in the FastGS repository, including NumPy, Pillow, plyfile, PyTorch,
torchvision, and LPIPS dependencies.

The FastGS native CUDA extensions are only needed for actual training and
rendering. They are not needed to inspect JSON manifests or run the sampling
logic.

FastGS must be available at the sibling path:

```text
/path/to/
├── FastGS/
└── ObjaverseRenderer/
```

The orchestrator defaults to:

```text
../FastGS
```

Override it with:

```bash
python3 orchestrator.py --fastgs-dir /path/to/FastGS
```

## Repository layout

```text
ObjaverseRenderer/
├── blender_worker.py
├── evaluate_metrics.py
├── orchestrator.py
├── renderer.py
├── sampling.py
├── experiments/
│   ├── furthest_4_seed_77.json
│   ├── random_5_seed_77.json
│   └── ...
├── objects/
│   ├── gamecube.glb
│   └── ...
├── views/
│   └── <object-id>/
├── reconstructions/
│   └── <object-id>/
└── results/
    └── <object-id>/
```

The source GLB files are inputs. Rendered views, reconstructions, and result
files are generated artifacts and should generally be excluded from version
control.

## Rendering global views

[renderer.py](./renderer.py) is the top-level Blender launcher. Its default
input is:

```text
objects/gamecube.glb
```

It can also render one explicit object:

```bash
python3 renderer.py \
    --glb-path objects/chair.glb \
    --object-id chair
```

The renderer creates:

```text
views/gamecube/
├── images/
│   ├── train/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   └── ...
│   └── test/
│       ├── 000.png
│       ├── 001.png
│       └── ...
├── global_transforms/
│   ├── transforms_train.json
│   └── transforms_test.json
├── metadata.json
├── rendered_scene.blend
└── transforms_test.json             # created by the orchestrator
```

The renderer does not create experiment-specific train manifests. It renders
the complete global lattice once so all experiments can reuse the same images.

### Global training views

`global_transforms/transforms_train.json` contains one frame for every
direction in the configured Fibonacci lattice. The image names are based on
their lattice index:

```json
{
  "camera_angle_x": 0.8726646259971648,
  "frames": [
    {
      "file_path": "images/train/000",
      "transform_matrix": [[...], [...], [...], [...]]
    }
  ]
}
```

FastGS appends `.png` to `file_path`, so paths intentionally omit the
extension.

### Global test views

`global_transforms/transforms_test.json` contains a fixed test set generated
from a separate Fibonacci lattice. The same test views are reused for every
experiment on an object, making metrics comparable.

## Experiment files

Each file in [experiments/](./experiments/) is applied to every object.

Example:

```json
{
  "id": "trajectory_6_seed_77",
  "train": {
    "strategy": "trajectory",
    "count": 6,
    "seed": 77
  }
}
```

The train configuration means:

- `strategy`: how lattice indices are selected.
- `count`: number of training views.
- `seed`: makes random and furthest-point selection reproducible.

`trajectory` selects views near evenly spaced azimuths at the configured
elevation. `furthest` performs farthest-point sampling over the sphere.
`random` samples without replacement.

Adding an experiment is as simple as adding a JSON file. The orchestrator
does not require a code change.

## Orchestrator workflow

[orchestrator.py](./orchestrator.py) is the normal entry point for generating
views and training FastGS.

For each object and experiment it:

1. Finds or creates the object’s global view corpus.
2. Loads the global train manifest.
3. Selects train lattice indices according to the experiment.
4. Writes `views/<object-id>/transforms_train.json`.
5. Ensures the shared `views/<object-id>/transforms_test.json` exists.
6. Creates a result record.
7. Runs FastGS with the object directory as `--source_path`.
8. Marks the result complete when training succeeds.
9. Deletes the temporary train manifest.

The temporary manifest is deliberately shared because FastGS requires the
exact filename `transforms_train.json` at the source root. It is safe to
delete after training because the global manifest and selected lattice indices
are recorded.

### FastGS command used

The orchestrator invokes FastGS approximately as:

```bash
python3 ../FastGS/train.py \
    --source_path views/gamecube \
    --model_path reconstructions/gamecube/<experiment-id> \
    --eval \
    --densification_interval 500 \
    --optimizer_type default \
    --test_iterations 7000 30000 \
    --save_iterations 7000 30000 \
    --grad_abs_thresh 0.0012
```

## Resuming cluster runs

The orchestrator is resumable between processes or cluster sessions.

An experiment is considered complete when this file exists:

```text
reconstructions/<object-id>/<experiment-id>/
└── point_cloud/iteration_30000/point_cloud.ply
```

On the next run, that experiment is skipped. Pending and failed experiments
are retried.

Recommended resume command:

```bash
python3 orchestrator.py \
    --skip-render \
    --continue-on-error
```

Options:

| Option | Meaning |
|---|---|
| `--skip-render` | Do not invoke Blender; require existing global transforms. |
| `--rerender` | Regenerate global views even if global transforms exist. |
| `--force-retrain` | Retrain experiments even if their final checkpoint exists. |
| `--continue-on-error` | Mark failures and continue with later experiments/objects. |
| `--object-id ID` | Process only one object. May be specified more than once. |
| `--fastgs-dir PATH` | Use a FastGS checkout at a custom path. |

Example for all objects:

```bash
python3 orchestrator.py \
    --skip-render \
    --continue-on-error
```

Example for selected objects:

```bash
python3 orchestrator.py \
    --object-id gamecube \
    --object-id chair \
    --skip-render \
    --continue-on-error
```

The current resume behavior is at reconstruction granularity. If a process
stops halfway through a 30,000-iteration FastGS run, the orchestrator retries
that experiment from the beginning. It does not currently resume from an
intermediate FastGS checkpoint.

## Multi-object processing

Without `--object-id`, the orchestrator discovers every `.glb` in `objects/`
and processes them sequentially:

```text
objects/
├── gamecube.glb
├── chair.glb
└── car.glb
```

Each object has its own:

```text
views/<object-id>/
reconstructions/<object-id>/
results/<object-id>/
```

FastGS still trains one object at a time. The later gain-prediction dataset can
combine records across all objects, but individual FastGS source datasets must
remain object-specific.

## Results files

The orchestrator creates a result file before training:

```text
results/gamecube/furthest_8_seed_77/results.json
```

This file is the mapping between an experiment and its reconstruction.
Initially it contains fields similar to:

```json
{
  "object_id": "gamecube",
  "experiment_id": "furthest_8_seed_77",
  "dataset_path": ".../views/gamecube",
  "reconstruction_path": ".../reconstructions/gamecube/furthest_8_seed_77",
  "train_lattice_indices": [3, 17, 42, 58],
  "train_image_paths": [
    "images/train/003",
    "images/train/017",
    "images/train/042",
    "images/train/058"
  ],
  "train_manifest": ".../views/gamecube/transforms_train.json",
  "test_manifest": ".../views/gamecube/transforms_test.json",
  "status": "pending",
  "metrics": {}
}
```

The `train_manifest` path is historical: the temporary file is deleted after
the reconstruction. The durable mapping is `train_lattice_indices` together
with `global_transforms/transforms_train.json`.

Statuses are:

- `pending`: result record exists but training has not completed.
- `running`: the orchestrator launched FastGS.
- `complete`: the expected final checkpoint exists.
- `failed`: FastGS returned an error.

## Metrics

[evaluate_metrics.py](./evaluate_metrics.py) is intentionally separate from
the orchestrator. This allows reconstruction to finish first and metrics to
be run later, possibly on a different GPU allocation.

Run all results for an object:

```bash
python3 evaluate_metrics.py --object-id gamecube
```

Run one experiment:

```bash
python3 evaluate_metrics.py \
    --object-id gamecube \
    --experiment-id random_10_seed_77
```

The metrics script:

1. Reads the result record.
2. Reconstructs the temporary train manifest from the recorded lattice
   indices.
3. Runs FastGS `render.py` on the test set.
4. Computes PSNR, SSIM, and LPIPS for every test view.
5. Stores per-view metrics and averages in `results.json`.
6. Deletes the temporary train manifest.

The resulting metrics section has this shape:

```json
{
  "metrics_iteration": 30000,
  "metrics": {
    "averages": {
      "PSNR": 25.1,
      "SSIM": 0.82,
      "LPIPS": 0.14
    },
    "per_view": {
      "00000": {
        "PSNR": 24.9,
        "SSIM": 0.81,
        "LPIPS": 0.15
      }
    }
  }
}
```

Metrics require a working CUDA-enabled FastGS environment because FastGS
rendering and LPIPS evaluation use CUDA tensors.

## FastGS compatibility rules

The generated dataset follows FastGS’s synthetic Blender loader:

- The source root contains `transforms_train.json` during training.
- The source root contains `transforms_test.json`.
- Every frame has `file_path` and `transform_matrix`.
- `file_path` is relative to the source root.
- Image extensions are omitted because FastGS appends `.png`.
- Images can be RGB or RGBA; FastGS composites RGBA against the configured
  background.
- Transform matrices are 4×4 camera-to-world matrices.

Do not rename the temporary manifests. FastGS specifically looks for:

```text
transforms_train.json
transforms_test.json
```

## Cleanup and rerendering

The renderer does not remove stale files. If the lattice size, camera setup,
object geometry, or test count changes, remove the object’s generated view
directory before rerendering:

```bash
rm -rf views/gamecube
python3 renderer.py --glb-path objects/gamecube.glb --object-id gamecube
```

Use `--rerender` when invoking the orchestrator if you intentionally want to
regenerate the global corpus. Do not use it as a substitute for changing
experiment definitions: experiment files affect subset selection, not the
rendered global images.

## Current limitations

- Interrupted FastGS training is not resumed from intermediate checkpoints.
- The completion check assumes the final iteration is `30000`.
- The orchestrator processes reconstructions sequentially.
- The metrics script assumes CUDA and FastGS’s metric dependencies are
  installed.
- Candidate-pose gain labels are not computed yet. The current pipeline records
  train subsets and reconstruction metrics; candidate augmentation and PSNR
  difference generation can be added as a later data-generation stage.
