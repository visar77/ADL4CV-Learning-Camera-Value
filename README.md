# Reconstruction Gain Predictor

Predict the reconstruction gain from adding a camera **before capturing its RGB image**, using existing context images and calibrated cameras.

## TL;DR — quick start

Run these commands from this directory, using an activated environment (`source ../.venv/bin/activate` for the existing workspace). For a fresh Linux setup, create one with `python3 -m venv .venv` and `source .venv/bin/activate`, then:

```bash
pip install -r requirements.txt

# 1. Assemble training records from existing renderer results and views.
python preprocess.py --output data/processed/table.jsonl

# 2. Check that the small CNN can memorize eight records on CPU.
python debug_train.py --dataset data/processed/table.jsonl \
  --num-samples 8 --epochs 200 --output checkpoints/quickstart/model.pt

# 3. Use that trained model to score cameras directly from images/camera metadata.
python inference.py --checkpoint checkpoints/quickstart/model.pt \
  --input examples/inference_request.json --rank
```

This assumes the Table images and reconstruction results exist under `data/views/Table/` and `data/results/Table/`. The last command demonstrates inference, not prediction quality on new contexts. **If you already have trained weights, inference needs neither training records nor feature extraction.**

To compare models on held-out contexts of the same object:

```bash
python compare.py --dataset data/processed/table.jsonl --scene-id Table \
  --mode context --models mean view-count geometry toy \
  --epochs 100 --output-dir checkpoints/table_comparison
```

For LVSM, first run `features.py` once, then train with `--model cached --features ...`; see step 2 below.

## How the pipeline fits together

```text
Training:
renderer results + RGB/cameras → preprocess.py → manifest.jsonl
  ├─ baselines/CNN → train.py → model.pt + report
  └─ frozen backbone → features.py → cache → train.py → model.pt + report
                                     compare.py runs several train.py
                                     experiments

Inference:
new RGB/cameras + trained weights → inference.py / predictor.py → candidate gains

Evaluation:
trained weights + labelled records → evaluate.py → regression and selection metrics
```

The command-line scripts are entry points. The shared Python modules listed below implement their common loading, model, and geometry logic.

### 1. Assemble training data: `preprocess.py`

Joins reconstruction scores with context image paths, camera poses, and intrinsics. It writes a JSONL manifest and `<manifest>.audit.json`; it does not resize/rewrite images or run a backbone. Defaults are `data/results/` and `data/views/`; override with `--results-root` and `--views-root`.

Records contain the context, candidate camera, and signed relative PSNR gain:

```text
(candidate PSNR - base PSNR) / abs(base PSNR)
```

Redundancy controls are retained when their candidate camera index is present. If a result has no distinct camera index for a redundant candidate, it is skipped and recorded in the audit. Repeated identical capture actions are averaged, preserving their original gain observations and source IDs. `--duplicates keep` preserves separate records; pass it to training too when studying raw observations. The audit reports duplicate spread and negative gains. Labels describe retraining reconstructions from scratch; they can be negative.

#### Use the full Crate reconstruction

The Crate results store context indices under `reconstructions.base` and each added camera in the candidate's `train_lattice_indices`. The preprocessor reads that format and derives the one added index. Run from this directory:

```bash
python preprocess.py \
  --results-root ../ObjaverseRenderer/results/Crate \
  --views-root ../ObjaverseRenderer/views \
  --output data/processed/crate_all.jsonl
```

This scans all experiment subdirectories under `results/Crate`; the camera images/transforms come from `views/Crate`. The Crate redundancy entries do not identify which existing camera was selected, so those entries cannot be assigned a candidate pose and are listed as skipped in `crate_all.audit.json`. The other candidate labels use the PSNR values recorded in each `results.json` directly:

```text
(candidate PSNR - base PSNR) / abs(base PSNR)
```

Start with a same-object, held-out-context run and a fast geometry baseline:

```bash
python train.py --dataset data/processed/crate_all.jsonl --scene-id Crate \
  --model geometry --mode context --epochs 100 \
  --output checkpoints/crate_geometry/model.pt \
  --log-dir runs/crate_geometry
```

Use the toy CNN to include the context RGB images:

```bash
python train.py --dataset data/processed/crate_all.jsonl --scene-id Crate \
  --model toy --mode context --epochs 100 \
  --output checkpoints/crate_toy/model.pt \
  --log-dir runs/crate_toy
```

The default context split keeps all candidates from one physical context together. This evaluates generalization to held-out Crate contexts, not to unseen objects. For cached LVSM-head training, first extract features for this manifest with `features.py`, then pass the resulting `features.pt` to `train.py --model cached`; this is substantially more compute-intensive.

Evaluate or rank predictions for a recorded experiment:

```bash
python evaluate.py --checkpoint checkpoints/crate_geometry/model.pt --split test
python inference.py --checkpoint checkpoints/crate_geometry/model.pt \
  --dataset data/processed/crate_all.jsonl --scene-id Crate \
  --experiment-id random_5_seed_77 --rank
```

The second command is an offline comparison against recorded candidate labels. For a new, unmeasured camera, use `inference.py --input` with a request containing the Crate context images and calibrated camera poses; the bundled `examples/inference_request.json` is for Table and is not the Crate request.

`table_v2.jsonl` was an example filename, not a separate dataset. Regenerated manifests include calibration, redundancy controls, and duplicate provenance: currently 119 unique Table records (100 candidate runs + 20 controls − one duplicate).

Image/source paths are **relative to the manifest**. New checkpoint references are relative to the checkpoint. Move the data/cache/run tree while preserving its layout; no username or checkout location is required.

**If you followed the quick start, skip migration:** it creates `data/processed/table.jsonl` with relative paths already. It does not create a file named `table_v2.jsonl`.

Only for an older manifest that actually exists and still contains absolute paths, run the optional migration with that file's name:

```bash
python preprocess.py --relativize data/processed/table.jsonl
```

This converts paths in place without changing labels. Use your existing filename if it differs; there is no separate migration step for newly generated data.

#### Existing datasets from before the calibration changes

Code updates do not update generated data: `data/` is Git-ignored. A legacy manifest can still have valid images/poses/labels but lack the new `context_intrinsics` and `candidate_intrinsics`. `--relativize` fixes paths **only** and does not add calibration.

Upgrade that existing manifest in place, using the camera transforms already beside its images:

```bash
python preprocess.py --upgrade data/processed/table.jsonl
# Only if the metadata lives separately:
# python preprocess.py --upgrade data/processed/table.jsonl --views-root /path/to/existing/views
```

This preserves record count, poses, signed gains, and duplicate provenance, and creates a `.jsonl.bak` before changing the file. It does not need reconstruction results, retraining, or candidate RGB. It matches camera poses against the existing transform JSONs and never guesses a field of view. A 100-row legacy manifest remains 100 rows; upgrade does not manufacture the extra controls from a full rebuild.

`preprocess.py --output ...` is a different operation: it rebuilds labels from `--results-root` (default `data/results`). A “no results.json files” error concerns that rebuild lookup, not whether the existing manifest/images were moved or whether CUDA works. A bare positional manifest filename is not accepted; use `--upgrade` for this compatibility repair.

### 2. Extract frozen features when needed: `features.py`

For cached-backbone training, encodes each distinct context once, decodes its candidate cameras, and saves embeddings. Repeating extraction reuses completed context/candidate files. **This step accelerates training; direct inference performs the backbone computation internally.** Baselines and the CNN skip it.

Production LVSM extraction on a CUDA machine, followed by head training:

```bash
python features.py --dataset data/processed/table.jsonl \
  --output data/features/lvsm512 --image-size 512 --lvsm-latent-tokens 3072 \
  --device cuda --precision bf16

python train.py --dataset data/processed/table.jsonl --model cached \
  --features data/features/lvsm512/features.pt --mode context --epochs 100
```

The **frozen backbone** turns context RGB/cameras and candidate cameras into a 768-dimensional candidate-conditioned embedding. The small trainable head maps that embedding to reconstruction gain; the backbone is not optimized during head training.

| Phase | Does LVSM run? |
|---|---|
| `preprocess.py` | No; assembles metadata/labels only. |
| First `features.py` run | Yes: one encoding per unique context, then candidate decoding. |
| Repeat extraction with complete cache | No; reuses cached embeddings, even if backbone weights are absent. |
| Cached-head training, including rolling validation | No; uses embeddings already loaded in memory. |
| Direct inference on new images/cameras | Yes; loads weights once and shares context encoding across candidates. |
| Offline evaluation/inference | Uses a valid cache when available; otherwise computes features online. |

The first cache build is the long step: it may download roughly 2 GB of weights, hash source files, then run encoding/decoding. At 512 resolution each image produces 4,096 tokens; the encoder processes `3072 + 4096 × context_views` tokens through full attention. Many views and CPU execution are especially expensive. Phase messages print **before** this work starts, and the timing JSON separates hashing, loading, encoding, and decoding. Head-training epochs do not rebuild this cache.

The official `lvsm_scene_encoder_decoder_res512.pt` weights download if missing. `--lvsm-checkpoint` selects another location/checkpoint. The 3,072 latent tokens are the pretrained model's learned scene-representation capacity; each token has 768 features. `--candidate-batch-size` controls decoder memory.

For a **CPU integration/overfitting smoke test**:

```bash
python features.py --dataset data/processed/table.jsonl --num-samples 8 \
  --output data/features/lvsm_smoke --image-size 16 --lvsm-latent-tokens 4 --device cpu

python train.py --dataset data/processed/table.jsonl --num-samples 8 \
  --model cached --features data/features/lvsm_smoke/features.pt --mode overfit \
  --epochs 200 --scheduler none --early-stopping-patience 0
```

Reduced resolution/latent counts change pretrained behavior; this smoke test does not validate production feature quality. Full-size CPU extraction is expensive. FP32 is the default portable precision. Optional BF16 also works on CPU; on a GPU without BF16 support, execution falls back to FP32 with a notice and possible small numerical differences.

The cache contains `features.pt`, phase timings, and resumable context/candidate files. Content/configuration hashes reject incompatible entries. Partial extraction covers only selected samples; match training selection flags or extract the whole manifest.

### 3. Train or compare: `train.py`, `debug_train.py`, `compare.py`

`train.py` selects records, creates splits, fits normalization on training samples, trains the chosen model, and writes the best checkpoint and report. Cached training optimizes only the head and never initializes LVSM. Use `--target delta-db` for absolute PSNR improvement instead of relative gain; measured PSNR is never a model input.

| Model | What is trained |
|---|---|
| `mean` | Constant mean training gain |
| `view-count` | Ridge regression on log context-view count |
| `geometry` | Ridge regression on camera distances, directions, count, and relative candidate position |
| `toy` | Small CNN with context/candidate pose conditioning |
| `cached` / `lvsm` | MLP head on frozen cached embeddings; both names use the same training path |

| Mode | Intended experiment |
|---|---|
| `overfit` | Memorize the selected records; only a training report |
| `context` (default) | Hold out entire contexts and all their candidates, including within one object |
| `scene` | Hold out entire objects |

Filter with `--scene-id`, `--experiment-id`, and `--num-samples` (a deterministic prefix per scene after duplicate handling). Use `overfit` for a single-context subset. Default splits are **80/10/10 by context/scene**, rounded for small datasets; the current Table data gives 8/1/1 contexts. Override fractions or supply `--split-file`; `--test-fraction 0` disables testing.

`debug_train.py` is a thin wrapper around `train.py`, defaulting to CPU, the toy CNN, batch size one, and overfit mode with no scheduler/early stopping. `compare.py` calls the same trainer for each requested model, verifies identical splits, and combines their reports. To include LVSM, add `cached` to `--models` and pass `--features`.

Common controls are `--epochs`, `--batch-size`, `--learning-rate`, `--scheduler`, and `--early-stopping-patience`; each script's `--help` lists the remaining options. Default dropout/weight decay are zero for overfitting experiments.

### 4. Inspect results: `evaluate.py`

Training writes `model.pt` and `model.report.json`; comparisons also write `comparison.json`. Unless overridden, run checkpoints and TensorBoard logs go to timestamped directories under `checkpoints/` and `runs/`.

`evaluate.py` reloads the exact saved split and verifies its observations, without training:

```bash
python evaluate.py --checkpoint checkpoints/table_comparison/geometry/model.pt --split test
# For an overfitting run, use --split train.
tensorboard --logdir runs --port 6006
```

Reports contain normalized MSE, MAE/RMSE, dB error when available, and per-context candidate-selection metrics: regret, selected/oracle gain, pairwise accuracy, random expected gain, and blindspot-heuristic gain. Ranking averages contexts within objects, then objects equally. Ties use expected utility under uniform tie-breaking. During neural training, validation runs **one minibatch per epoch** (up to 8 records), cycling through the validation split. A sample-weighted rolling mean of the last **10 validation batches** drives checkpoint selection, scheduling, and early stopping. This is a smoothed estimate across recent model states. Configure `--validation-batch-size`, `--validation-batches`, and `--validation-window`; history/TensorBoard include sample counts and timing. Full split reports run once at the end after restoring the selected checkpoint. Overfit mode uses the existing epoch training loss, with no extra validation pass. Overfit scores and within-object scores measure different things; neither establishes unseen-object generalization.

### 5. Predict on new inputs: `inference.py` and `predictor.py`

`inference.py --input examples/inference_request.json` is the direct CLI. The example has `context` entries with `image`, `pose`, and `intrinsics`, and `candidates` entries with `id`, `pose`, and `intrinsics`. Image paths are relative to the request file. No labels, training manifest, or feature cache is needed; `--rank` sorts candidates by predicted gain.

`predictor.py` provides the same behavior as a Python API:

```python
import math
from predictor import GainEstimator

predictor = GainEstimator("checkpoints/run/model.pt")  # Auto-selects CUDA when available.
scores = predictor.predict(
    context_images=images, context_poses=poses, candidate_poses=candidates,
    camera_angle_x=math.radians(50), camera_convention="blender",
)
```

Accepts image paths, PIL/NumPy RGB images, or PyTorch `[N,3,H,W]` tensors; floating input is `[0,1]`, uint8 is `[0,255]`. Poses are camera-to-world matrices; use `opencv` for OpenCV axes. Supply horizontal FOV in radians or context/candidate intrinsics as pixel K or normalized `[fx/W,fy/H,cx/W,cy/H]`. Pixel K refers to original dimensions; candidate size defaults to the first image.

Loading/resizing, camera conversion, rays, and scene normalization happen internally. The predictor loads the backbone once and shares one context encoding across candidates per call. `candidate_batch_size` controls decoder memory. Keep the head and matching backbone weights for deployment; their saved relative reference is used automatically, with `--lvsm-checkpoint` / `backbone_checkpoint=` as an override.

For offline inspection, `inference.py --dataset ... --scene-id ... --experiment-id ... --rank` instead scores labelled manifest records. That mode can reuse cached features and display ground truth.

### 6. Verify the implementation: `verify_lvsm.py` and tests

```bash
python -m unittest discover -s tests -v
python verify_lvsm.py --checkpoint lvsm_scene_encoder_decoder_res512.pt
```

`verify_lvsm.py` compares the real checkpoint on a small input against independent implementations of the [official LVSM equations](https://github.com/Haian-Jin/LVSM), using explicit attention and patch slicing. It checks numerical computation, not production-resolution GPU throughput or the original CUDA/xformers kernel.

## Moving to another Linux machine

Training, extraction, evaluation, and inference default to `--device auto`: CUDA when available, otherwise CPU. `debug_train.py` intentionally defaults to CPU; use `--device cpu` or `--device cuda:1` to override elsewhere. Install a CUDA-enabled PyTorch build and working GPU driver to use CUDA; the predictor requires no extra custom CUDA extensions.

Checkpoints and caches store **CPU tensors** and relative file references, without a GPU index or device-specific compiled code. Copy the required data/cache/weights with their relative layout intact. New inputs need the head and matching backbone weights, not the training manifest/cache. Floating-point results can differ slightly between hardware/precision settings. The tests include relocation and a CUDA→CPU round trip (skipped when CUDA hardware is unavailable).

## Shared modules and supporting files

These modules are imported by the scripts; they are not additional pipeline commands.

| File | Responsibility |
|---|---|
| `dataset.py` | Manifest records, label audits/duplicate handling, RGB loading, padded batches, and camera metadata. |
| `geometry.py` | Blender→OpenCV conversion, calibrated rays, context-only normalization, and pose features. |
| `model.py` | CNN, LVSM adapter, replaceable `FeatureBackbone` interface, and gain head. |
| `experiments.py` | Record selection, grouped splits, observation signatures, and selection metrics. |
| `runtime.py` | Shared model/checkpoint loading and prediction used by training/evaluation/inference. |
| `paths.py` | Resolve portable references relative to their owning file. |
| `devices.py` | Select runtime device/precision without binding artifacts to hardware. |
| `requirements.txt` | Python dependencies. |
| `examples/inference_request.json` | Direct-inference request using the existing Table images/cameras. |
| `tests/test_pipeline.py` | Basic manifest, image preparation, and model smoke tests. |
| `tests/test_research_pipeline.py` | Camera/backbone correctness, caching, splits, metrics, and training/CLI integration. |
| `tests/test_portability_and_inference.py` | Relocation, different working directories, and inference without training artifacts. |
| `tests/test_validation_and_devices.py` | Rolling validation budgets/weighting, 80/10/10 defaults, and device-independent saves. |
| `tests/test_manifest_upgrade.py` | Legacy calibration upgrade without raw results/candidate RGB; labels, backups, and failed-upgrade safety. |
| `.gitignore` | Excludes generated data, runs, checkpoints, and Python caches. |

For another backbone, implement `FeatureBackbone.encode_context()` and `decode_candidates()` in an adapter and register its factory, or pass it to `features.extract_features()`. Context state is a dictionary of tensors; candidate features are `[B,Q,D]`. Cached-head training accepts arbitrary `D`, so the experiment/training code stays shared.

Camera handling uses calibrated pixel-center rays and a context-only coordinate frame/scale, reused for every candidate. Padded views are masked in every encoder block. LVSM loading validates all required weight shapes and memory-maps model tensors. Predictor checkpoints from before these camera/mask corrections require retraining.
