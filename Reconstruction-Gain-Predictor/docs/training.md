# Training and evaluation

Run commands from the project directory.

`train.py` selects records, creates splits, fits normalization on training samples, trains the chosen model, and writes the best checkpoint and report. Cached training optimizes only the head and never initializes LVSM. The default `--target delta-db` predicts signed PSNR change in dB, including negative gains. Use `--target relative` to reproduce the previous target. Both use the same feature cache.

| Model | What is trained |
|---|---|
| `mean` | Constant mean training gain |
| `view-count` | Ridge regression on log context-view count |
| `geometry` | Ridge regression on camera distances, directions, count, and relative candidate position |
| `toy` | Small CNN with context/candidate pose conditioning |
| `cached` / `lvsm` | Gain head on frozen candidate embeddings, optionally with relative camera poses |

**Use `--head pose` for experiments spanning orbit and free-moving cameras.** For each context camera the head computes its pose in the candidate camera's frame. Three translation components describe where it is, and two rotation-matrix columns encode its full orientation, including roll. The third column is determined by the first two. This representation requires calibrated camera poses and has no object-center or common look-at assumption.

Translations use the RMS distance of context camera centers from their centroid as a length scale, followed by elementwise `asinh` to compress large offsets. For one center or negligible spread the scale falls back to one input unit. Use consistent pose units for these trajectories. The geometry branch is independent of the world origin, world orientation, and context ordering. LVSM has its own existing first-context orientation convention.

A shared 16-unit layer encodes each relative pose jointly. Mean and max pooling over valid views, plus log view count, summarize the set. A linear pose score and a 32-unit correction conditioned on both the pose summary and the LVSM embedding predict signed gain. LVSM supplies image and intrinsic information relevant to visibility. The head has 25,890 trainable parameters with a 768-D embedding. It uses the same feature cache for all pose regimes. The two alternative heads are the default embedding-only `--head mlp` and the orbit-specific `--head coverage`, whose geometry branch summarizes only viewing-direction separation and count.

Add `--head pose` to the cached training command. Compare it with a width-matched MLP on the same scene splits using top-choice regret and signed-gain MAE. Orbit-only labels couple camera position and orientation. Testing transfer to free motion requires held-out scenes with motions such as lateral translation at fixed orientation and roll at fixed position, together with measured gains. These tests also probe whether the pooled LVSM embedding carries enough information about occlusion and scene depth.

| Mode | Intended experiment |
|---|---|
| `overfit` | Memorize the selected records; only a training report |
| `context` (default) | Hold out entire contexts and all their candidates, including within one object |
| `scene` | Hold out entire objects |

Filter with `--scene-id`, `--experiment-id`, and `--num-samples` (a deterministic prefix per scene after duplicate handling). Use `overfit` for a single-context subset. Default splits are **80/10/10 by context/scene**, rounded for small datasets; the current Table data gives 8/1/1 contexts. Override fractions or supply `--split-file`; `--test-fraction 0` disables testing.

`compare.py` calls the same trainer for each requested model, verifies identical splits, and combines their reports. To include LVSM, add `cached` to `--models` and pass `--features`.

Common controls are `--epochs`, `--batch-size`, `--learning-rate`, `--scheduler`, and `--early-stopping-patience`; each script's `--help` lists the remaining options. Default dropout/weight decay are zero for overfitting experiments.

## Reports and validation

Training writes `model.pt` and `model.report.json`; comparisons also write `comparison.json`. Unless overridden, run checkpoints and TensorBoard logs go to timestamped directories under `checkpoints/` and `runs/`.

`evaluate.py` reloads the exact saved split and verifies its observations, without training:

```bash
python evaluate.py --checkpoint checkpoints/table_comparison/geometry/model.pt --split test
# For an overfitting run, use --split train.
tensorboard --logdir runs --port 6006
```

Reports contain normalized MSE, MAE/RMSE, dB error when available, and per-context candidate-selection metrics such as regret, selected/oracle gain, pairwise accuracy, random expected gain, and blindspot-heuristic gain. Ranking averages contexts within objects, then objects equally. Ties use expected utility under uniform tie-breaking.

Cached and LVSM heads score **every validation record at the end of each epoch**. The sample-weighted MSE uses the current model weights and drives checkpoint selection, plateau scheduling, and early stopping. TensorBoard logs it as `loss/validation`. Set `--validation-batch-size` to control the size of each batch.

The toy CNN defaults to rolling validation to limit image loading. For cached and LVSM heads, supplying `--validation-batches` or `--validation-window` enables rolling validation. Its default budget is one batch of up to eight records per epoch, and the default window is ten batches. `loss/validation_rolling` can include different model states and subsets. History and TensorBoard include sample counts and timing in both modes. Full split reports run after restoring the selected checkpoint. Overfit mode monitors epoch training loss.
