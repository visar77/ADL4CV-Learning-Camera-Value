# Reconstruction Gain Predictor

The predictor estimates relative PSNR gain for adding a candidate camera pose:

```text
(candidate PSNR - base PSNR) / abs(base PSNR)
```

The model receives the existing context RGB views and their poses plus the
candidate pose. Candidate RGB is never used because it is unavailable at
decision time.

## Workflow

### 1. Preprocess once

Run this after renderer views and metric results exist:

```bash
cd Reconstruction-Gain-Predictor
../.venv/bin/python preprocess.py \
  --results-root data/results \
  --views-root data/views \
  --output data/processed/table.jsonl \
```

This creates one transparent JSONL dataset. It contains, per base/candidate
sample:

- paths to ordinary RGB context images;
- context camera poses;
- candidate camera pose;
- relative PSNR gain, base PSNR, and candidate PSNR;
- scene, experiment, and candidate identifiers.

It does not contain reconstruction paths or require reconstruction files.
The source folders are read during training as well. The model performs
conversion to tensors, scaling, and resizing before the backbone. The JSONL
dataset is stored wherever `--output` points; `data/processed/` is the
recommended project-local location.

For a one-sample debug dataset, add `--limit 1`.

### 2. Debug the pipeline with the toy CNN

`debug_train.py` uses the same processed-dataset format and training loop as
normal training, but forces CPU, the small CNN, and batch size one:

```bash
../.venv/bin/python debug_train.py \
  --dataset data/processed/table.jsonl \
  --num-samples 1 \
  --epochs 10
```

Both `train.py` and `debug_train.py` accept `--num-samples N`. Records are
selected deterministically from the beginning of the JSONL dataset, so
`--num-samples 1`, `--num-samples 10`, and so on are reproducible subsets.
The option applies before the scene split and is recorded in checkpoint
metadata.

### 3. Train the real model

`train.py` uses the LVSM adapter by default. It requires a downloaded LVSM
module checkpoint:

```bash
../.venv/bin/python train.py \
  --dataset data/processed/table.jsonl \
  --lvsm-checkpoint checkpoints/lvsm/lvsm-module.pt \
  --epochs 50
```

Adding more objects under `data/results/<scene>/` and `data/views/<scene>/`
works with the preprocessing step as long as each scene has matching
`results.json` records and `transforms_train.json` plus referenced train
images. Re-run preprocessing after adding objects.

However, `train.py` is not automatically plug-and-play with an arbitrary
LVSM download. The current adapter expects the downloaded checkpoint to be a
serialized `torch.nn.Module` implementing:

```python
forward(images, context_poses, candidate_pose, mask) -> [batch, embedding_dim]
```

If the LVSM download is a raw `state_dict`, a package checkpoint, or uses a
different input convention, an LVSM-specific wrapper still needs to be added
to `model.py`. The current workspace does not contain the LVSM implementation,
so downloading weights alone cannot establish that architecture.

### 4. Run inference

There is one inference entry point:

```bash
../.venv/bin/python inference.py \
  --checkpoint checkpoints/psnr_lvsm/<timestamp>/model.pt \
  --dataset data/processed/table.jsonl \
  --lvsm-checkpoint checkpoints/lvsm/lvsm-module.pt \
  --sample-index 0
```

Use `--scene-id`, `--experiment-id`, and `--candidate-id` to filter before
selecting `--sample-index`. The script prints ground-truth and predicted
relative gain and the corresponding candidate PSNR values.

## Outputs and monitoring

When output paths are omitted, every run gets a timestamp:

```text
runs/<model-name>/<YYYYMMDD-HHMMSS>/*              # TensorBoard files
checkpoints/<model-name>/<YYYYMMDD-HHMMSS>/model.pt
```

Start TensorBoard with:

```bash
../.venv/bin/tensorboard --logdir runs --port 6006
```

Open `http://localhost:6006`. Training logs `loss/train` and
`loss/validation` directly into the timestamped run directory.

## Tests

```bash
../.venv/bin/python -m unittest discover -s tests -v
```
