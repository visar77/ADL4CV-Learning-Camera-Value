# Reconstruction Gain Predictor

Predict how much adding a camera will improve a reconstruction, **before capturing that camera's RGB image**. Inputs are existing context images, their calibrated cameras, and candidate cameras. Labels are signed reconstruction gains; negative gains are retained.

## Start here

The six scripts at the top level are commands. The `gain_predictor/` package contains their implementation.

| Command | Reads | Produces |
| --- | --- | --- |
| `prepare_data.py` | Renderer scores and camera/image metadata | Labelled `manifest.jsonl` and audit |
| `cache_features.py` | Manifest and frozen LVSM weights | Reusable image/camera embeddings |
| `train.py` | Manifest, plus embeddings for LVSM | Trained gain model and report |
| `compare.py` | Same inputs as training | Several models evaluated on identical splits |
| `evaluate.py` | Trained model and its saved data split | Regression and camera-selection metrics |
| `predict.py` | Trained model and new images/cameras (or a manifest) | Candidate gains and ranking |

**Preparing data** joins existing measurements into training records. **Caching features** runs a frozen neural network on the context images and cameras. Baselines and the small CNN skip caching. At prediction time, new images are encoded internally.

```text
renderer results + camera metadata → prepare_data.py → manifest
  ├─ baselines / CNN ────────────────────────→ train.py → model + report
  └─ context images + frozen LVSM → cache_features.py ──↗

new images + cameras + trained model → predict.py → candidate gains
```

## Quick start

Run from this directory in a Python environment with `pip install -r requirements.txt`.
Use the project's existing environment if available; for a fresh local environment,
run `python3 -m venv .venv` and `source .venv/bin/activate` first.

```bash
# Build labels from existing renderer outputs; adjust paths to your corpus.
python prepare_data.py \
  --results-root ../ObjaverseRenderer/results/Crate \
  --views-root ../ObjaverseRenderer/views \
  --output data/processed/crate.jsonl

# Fit a fast baseline, holding out entire contexts.
python train.py --dataset data/processed/crate.jsonl \
  --model geometry --mode context --output checkpoints/crate/model.pt

python evaluate.py --checkpoint checkpoints/crate/model.pt --split test

# Rank the recorded candidates in one context.
python predict.py --checkpoint checkpoints/crate/model.pt \
  --dataset data/processed/crate.jsonl --scene-id Crate \
  --experiment-id random_5_seed_77 --rank
```

For a CPU CNN smoke test on eight records:

```bash
python train.py --dataset data/processed/crate.jsonl \
  --model toy --mode overfit --device cpu --batch-size 1 --num-samples 8 \
  --epochs 200 --scheduler none --early-stopping-patience 0 \
  --output checkpoints/smoke/model.pt
```

Overfitting checks the pipeline; its scores do not establish generalization.

For LVSM, cache embeddings once and train a small head on them:

```bash
python cache_features.py --dataset data/processed/crate.jsonl \
  --output data/features/crate --device cuda --precision bf16

python train.py --dataset data/processed/crate.jsonl --model cached \
  --features data/features/crate/features.pt --head pose \
  --output checkpoints/crate_lvsm/model.pt
```

LVSM defaults to 512-pixel inputs and 3,072 latent tokens. The first extraction can download about 2 GB of weights and is compute-intensive; head training does not rerun LVSM. See [feature caching](docs/feature_cache.md) for resuming, memory controls, and reduced-size smoke tests.

## Where the code lives

| I want to change… | Start in `gain_predictor/` |
| --- | --- |
| How renderer outputs become labels | `data/manifest.py` and `data/records.py` |
| How inputs are batched or split | `data/batches.py` and `data/splits.py` |
| The networks | `models/lvsm.py`, `models/cnn.py`, or `models/heads.py` |
| How embeddings are extracted and reused | `feature_cache.py` |
| How the model is fitted or judged | `training.py` and `metrics.py` |
| How new cameras are scored | `inference.py` |
| How saved runs and labelled records are scored | `checkpoints.py` and `scoring.py` |

`models/` defines the neural networks and their tensor operations. `GainEstimator` is the user-facing API: it loads trained weights, prepares images/cameras, and returns gains. To change the gain head, edit `models/heads.py`; to change what happens when a user predicts, start at `inference.py`.

Generated `data/`, `checkpoints/`, and `runs/` remain separate from source code and are Git-ignored. Tests and the independent LVSM verifier live in `tests/`.

## Predict on new inputs

In Python, import `GainEstimator` from `gain_predictor` and call
`predict(context_images, context_poses, candidate_poses, ...)` with a common
horizontal field of view or per-camera intrinsics.

The CLI accepts the same inputs as JSON:

```bash
python predict.py --checkpoint checkpoints/crate_lvsm/model.pt \
  --input path/to/request.json --rank
```

[examples/inference_request.json](examples/inference_request.json) shows the format using Table images; replace its images and cameras for your context. Image references resolve relative to the request. Prediction needs the trained head and matching backbone weights, but no training manifest or feature cache.

`predict()` returns signed gains. `predict_utility()` returns `max(0, gain)`, allowing the option of keeping the base reconstruction. CLI ranking uses this utility, while regression and forced-capture evaluation retain signed gains. See [prediction details](docs/prediction.md) for tensor formats, calibration, and batching.

## Data and experiment rules

- New training and comparison runs default to signed PSNR change in dB (`candidate_psnr - base_psnr`). Use `--target relative` to reproduce the previous target, which divides that change by `abs(base_psnr)`. Candidate RGB and measured PSNR are never prediction inputs.
- Each candidate is one independent addition to its context. Labels retain the measured result, including negative gains. Keep different reconstruction protocols in separate datasets; the manifest builder does not enforce protocol separation.
- Repeated physical actions are averaged with their observations and provenance retained. Use `--duplicates keep` in both preparation and training to retain individual records. Redundancy controls have zero gain; missing repeated-camera IDs use a deterministic context-camera proxy recorded in the audit.
- `--mode context` holds out entire contexts; `--mode scene` holds out objects; `--mode overfit` uses only a training set. The default split is 80/10/10. See [training and evaluation](docs/training.md) for models, heads, validation budgets, and metrics.
- Manifests reference images relative to the manifest; checkpoints reference data/cache/weights relative to the checkpoint. Preserve that layout when copying artifacts. Tensor saves use CPU storage. Devices default to `auto` (CUDA when available, otherwise CPU).

To repair an **existing** old manifest, without reconstruction results or retraining:

```bash
python prepare_data.py --upgrade data/processed/old.jsonl
# If camera metadata is stored separately, add --views-root /path/to/views.
```

Upgrade recovers calibration from existing camera transforms, preserves labels and poses, and creates a `.jsonl.bak`. It does not guess a field of view. `--relativize` changes paths only; newly prepared manifests need neither operation. Preparation defaults to `data/results` and `data/views` when roots are omitted.

## Previous names

| Before | Now |
| --- | --- |
| `preprocess.py` | `prepare_data.py` |
| `features.py` | `cache_features.py`; cache implementation in `gain_predictor/feature_cache.py` |
| `inference.py` command | `predict.py` |
| `from predictor import GainEstimator` | `from gain_predictor import GainEstimator` |
| `model.py` | Separate networks in `gain_predictor/models/` |
| `dataset.py`, `experiments.py` | `gain_predictor/data/` and `metrics.py` |
| `runtime.py` | `checkpoints.py` and `scoring.py` inside the package |
| `debug_train.py` | Explicit `train.py` smoke-test command above |
| `verify_lvsm.py` | `python -m tests.verify_lvsm` |

Update old commands/imports to these names. Existing manifests and feature caches can be reused for dB training. The JSONL `psnr_gain` and `observed_gains` fields still store relative gains. Training reads the saved PSNR scores to derive dB labels, or recovers them from `psnr_gain` and `base_psnr` if the candidate score is absent. If a manifest lacks `base_psnr`, use `--target relative` to train from that file as-is.

Evaluation and prediction use the checkpoint's saved `target`. Report and prediction field names are unchanged. In particular, `predicted_gain` uses dB for `target: delta-db` and relative units for `target: relative`. Older relative models can still predict and evaluate with their original artifacts. Checkpoints from before the earlier camera/mask corrections still require retraining.

## Validation

```bash
python -m unittest discover -s tests -v
python -m tests.verify_lvsm --checkpoint lvsm_scene_encoder_decoder_res512.pt
```

The second command requires real LVSM weights and compares small inputs against independent attention and tokenization equations. CPU tests do not establish production GPU throughput or prediction quality.
