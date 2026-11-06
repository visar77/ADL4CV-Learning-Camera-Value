# Reconstruction Gain Predictor

This project trains a PyTorch predictor from reconstruction experiments. It
supports absolute reconstruction-quality prediction and true
reconstruction-gain prediction. The latter derives its labels dynamically from
the existing experiment records, so no additional gain files are needed.

The model is split into [model.py](./model.py): a frozen LVSM adapter and a
trainable MLP regression head that emits one scalar gain. The adapter is
currently metadata-compatible because the renderer records do not yet contain
the LVSM image-view tensor; it is the replacement boundary for the real LVSM
encoder.

## Requirements

Use Python 3.10 or newer. Install the dependencies into the environment used
to run the script:

```bash
pip install -r requirements.txt
```

The dependencies are:

- `torch` for the dataset, model, optimization, and checkpointing.
- `tensorboard` for monitoring training and validation loss.

## Input data

The renderer writes one result file per object and experiment:

```text
results/
└── gamecube/
    └── random_10_seed_77/
        └── results.json
```

The training script accepts either a single `results.json` file or a directory.
For a directory, every nested file named `results.json` is loaded recursively.
Each record must contain:

```json
{
  "object_id": "gamecube",
  "experiment_id": "random_10_seed_77",
  "train_lattice_indices": [3, 17, 42, 58],
  "metrics": {
    "averages": {
      "PSNR": 25.1,
      "SSIM": 0.82,
      "LPIPS": 0.14
    }
  }
}
```

Records without a non-empty `train_lattice_indices` list or the selected target
metric are skipped. If no complete labelled records remain, the script exits
with an actionable error. Run the renderer's metric evaluation first to fill
`metrics.averages`.

## Features, modes, and model

The `--mode` option selects the target:

- `quality` (default): predict the selected experiment's average metric.
- `gain`: predict the improvement caused by adding views to a reconstruction.

For each training example, the model converts the relevant view indices into seven
numeric features:

1. number of selected views;
2. minimum lattice index;
3. maximum lattice index;
4. mean index;
5. population standard deviation of indices;
6. index span;
7. selected-view density relative to the highest selected index.

Features and targets are normalized using statistics computed from the training
objects only. In gain mode, every label is first made dimensionless by dividing
the metric delta by the absolute baseline metric for that object. Thus labels
are comparable across objects despite different PSNR, SSIM, or LPIPS scales.
LPIPS deltas are inverted so a positive value always means improvement.

`GainPredictor` contains a frozen `LVSMBackbone` followed by a trainable MLP:

```text
7 inputs -> frozen LVSM adapter -> Linear(64) -> GELU -> Dropout -> Linear(1)
```

The split is by `object_id`, not by individual experiment. This prevents
experiments from the same object appearing in both training and validation
sets, which would otherwise make validation overly optimistic.

### Dynamic gain labels

Gain mode groups records by `object_id`. For each experiment, it finds the
largest strict subset of its `train_lattice_indices` among the same object's
experiments. That subset is the comparison reconstruction. The model input is
the set difference—the newly added view indices—and the target is:

```text
PSNR/SSIM gain  = larger-subset metric - baseline metric
LPIPS gain      = baseline metric - larger-subset metric
```

LPIPS is inverted so positive gain consistently means an improvement for all
metrics. Experiments with no strict-subset comparison are skipped. If multiple
strict subsets have the same size, selection is deterministic by experiment ID.

This derivation is cheap: it only loads JSON metadata and performs set
operations; it does not rerun rendering, reconstruction, or metric evaluation.
The expensive work remains the renderer's existing reconstruction and
evaluation stages.

## Training

From this directory, train against average PSNR:

```bash
python train.py \
  --data ../ObjaverseRenderer/results \
  --target PSNR \
  --output checkpoints/gain_predictor.pt \
  --log-dir runs/gain_predictor
```

To train the true dynamically derived gain target:

```bash
python train.py \
  --data ../ObjaverseRenderer/results \
  --mode gain \
  --target PSNR \
  --output checkpoints/psnr_gain_predictor.pt \
  --log-dir runs/psnr_gain
```

The first normal training run loads `--backbone-checkpoint`. If that file does
not exist, `--backbone-url` is downloaded to that path before the model is
constructed:

```bash
python train.py \
  --data ../ObjaverseRenderer/results \
  --mode gain \
  --backbone-checkpoint checkpoints/lvsm_backbone.pt \
  --backbone-url https://your-model-host/lvsm_backbone.pt
```

Tests replace `train.GainPredictor` with a mock model before `train()` runs, so
they never instantiate the downloader or access the network. A real LVSM
checkpoint must match the adapter's state dictionary until the image-based
LVSM implementation is wired in.

The available targets are `PSNR`, `SSIM`, and `LPIPS`. Useful options are:

```text
--epochs                 Number of training epochs (default: 50)
--batch-size             Batch size (default: 32)
--learning-rate          AdamW learning rate (default: 1e-3)
--validation-fraction    Fraction of objects used for validation (default: 0.2)
--seed                   Python and PyTorch random seed (default: 42)
--device                 Device such as cpu or cuda (auto-detected by default)
--checkpoint-every       Save an additional epoch-N checkpoint (default: 10)
--backbone-checkpoint     Local frozen LVSM checkpoint path
--backbone-url            URL used when the checkpoint path is missing
--object-id               Train only examples for one object
--max-samples             Train on the first N selected examples
--sample-index            Train on one zero-based selected example
```

The `--object-id` option trains on a single full object's available examples.
`--max-samples` selects a deterministic subset, and `--sample-index` selects
one sample. A one-sample run is supported; its validation set reuses that
sample because an object-level split cannot produce a separate validation
object. These controls can be combined, for example:

```bash
python train.py --data ../ObjaverseRenderer/results \
  --mode gain --object-id gamecube --max-samples 32
```

The script prints normalized training and validation MSE for every epoch and
saves the checkpoint with the best validation loss. It also writes periodic
snapshots such as `gain_predictor.epoch-10.pt` according to
`--checkpoint-every`. A checkpoint contains:

- `model`: the model state dictionary;
- `feature_mean` and `feature_std`: feature normalization values;
- `target_mean` and `target_std`: target normalization values;
- `target_metric`: the selected metric name;
- `mode`: `quality` or `gain`;
- `feature_names`: the feature ordering required at inference time.

## Monitoring with TensorBoard

Training writes scalar loss values beneath the configured log directory.
Start TensorBoard in another terminal:

```bash
tensorboard --logdir runs
```

Open the displayed local URL, then compare `loss/train` and
`loss/validation`. The validation curve is also the criterion used to select
the saved checkpoint.

## Tests

The test suite uses Python's standard-library `unittest`, so it does not
require pytest:

```bash
python -m unittest discover -s tests -v
```

The end-to-end test creates temporary renderer-style records and replaces the
production model with a tiny mock `nn.Module`, so it never downloads a
backbone. It verifies that:

- complete records load while incomplete records are skipped;
- a mock model receives the normalized feature input and trains for multiple
  epochs;
- a best checkpoint is written with the expected metadata;
- periodic epoch checkpoints are written;
- sample, subset, and object filters select the expected records;
- the monitoring writer receives both train and validation loss series.

No project data, checkpoint, or event files are created by the tests.

The tests also verify that nested experiments derive a single added-view
example and use the correct metric direction for both PSNR and LPIPS.

An image-based model can replace `GainPredictor` once rendered images or
learned backbone embeddings are available. The current pipeline does not
download a backbone or pretrained weights.
