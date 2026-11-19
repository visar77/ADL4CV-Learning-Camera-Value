# FisherRF baseline

Rank the same physical candidate cameras as the gain predictor, using the saved FastGS base cloud for each context. The script uses FisherRF's modified CUDA rasterizer. It scores one potential acquisition at a time, with the base fixed throughout.

## Setup and run

From the main project checkout, initialize the pinned FisherRF submodule:

```sh
cd "$(git rev-parse --show-toplevel)"
git submodule update --init --recursive -- FisherRF
```

The baseline loads that checkout automatically. Use the project's shared Conda environment, `adl4cv-learning`. On the cluster login node, run `bash conda_setup_env.sh`, then `mkdir -p logs` and `sbatch setup.sbatch` for the CUDA build. Activate the environment with `conda activate adl4cv-learning` when running interactively. FisherRF's CUDA 12.8 / RTX 5090 build and scoring parity still need a GPU run.

From inside the project checkout, run one context as follows. The first command switches to the project root. Omit the object and experiment filters for the whole bundle.

```sh
cd "$(git rev-parse --show-toplevel)"
python Reconstruction-Gain-Predictor/experiments/fisherrf/FisherRF_baseline.py \
  four_objects_views_results_reconstructions \
  --object Crate --experiment random_3_seed_77 --output /tmp/Crate_fisherrf.json
```

The run root contains sibling `results/`, `views/`, and `reconstructions/` directories. The scorer loads the base iteration and additive candidate camera sets from each result, maps physical camera indices into the object's global training transforms, and restores the SH degree, background, and camera dimensions from base-run metadata. It needs neither captured candidate RGB nor any candidate PLY. Failed bases and contexts without completed acquisitions are listed under `skipped`.

## Score and timing

For each observed camera, use the upstream modified render backward pass to obtain a diagonal information vector. Sum the base vectors once. Score each distinct candidate by summing its information divided elementwise by (base information + 1e-6). Use position, diffuse and higher-order SH features, scale, and opacity, matching the single-view selector's default exclusion of rotation. The largest score wins. This baseline uses a previously fitted reconstruction and performs no optimization during camera selection. [Upstream acquisition function](https://github.com/JiangWenPL/FisherRF/blob/b74732812b295189f230a192418375f56cec3bd6/active/H_reg.py) · [Selector defaults](https://github.com/JiangWenPL/FisherRF/blob/b74732812b295189f230a192418375f56cec3bd6/active_train.py)

The JSON gives each camera's raw Fisher score and signed measured PSNR gain in dB. Duplicate candidate runs for one camera share a score and their measured PSNR values are averaged. Exact score ties choose the smallest physical index. Each context reports the chosen camera, its signed gain, and regret in dB (best measured gain minus chosen gain). Match methods on object, experiment, and physical camera ID. Fisher scores are ranking values; their magnitudes have no dB interpretation.

Timings are wall-clock seconds with CUDA synchronization. `checkpoint_loading` includes restoring the base on the GPU. `base_information` and `candidate_scoring` measure the two scoring phases. `total` starts before checkpoint loading and includes camera preparation, ending when scores are available. Process startup, result evaluation, and the original base fit are outside this interval. For a comparison starting from posed RGB, report the actual base fitting time separately when available. Time LVSM from its corresponding starting point, including context encoding and candidate decoding. Comparing cached-head inference alone with end-to-end FisherRF answers a different question.

## Validation

From the project root:

```sh
python -m unittest discover -s Reconstruction-Gain-Predictor/experiments/fisherrf -v
```

To include the CUDA integration test on a GPU with the extensions built, run:

```sh
cd "$(git rev-parse --show-toplevel)"
FISHERRF_RUN_ROOT="$PWD/four_objects_views_results_reconstructions" \
  python -m unittest discover -s Reconstruction-Gain-Predictor/experiments/fisherrf -v
```

The integration test compares an ordinary upstream render of the Crate base against FastGS's saved render (mean absolute RGB error below 2/255) and checks the winning camera against upstream `HRegSelector.nbvs`, allowing numerical ties. Only that render check reads a held-out camera. CPU tests cover scoring, deduplication, camera IDs, signed gains, and regret.

## Saved comparison plan

Only FisherRF is implemented in this experiment. The following records the agreed broader plan for future work.

| Comparison | Protocol |
| --- | --- |
| NeU-NBV | Use frozen ShapeNet weights. Render a 60×60 uncertainty map for each candidate with up to five nearby observed cameras and select the highest mean RGB-channel uncertainty. Report zero-shot status. [Paper](https://arxiv.org/pdf/2303.01284) · [Code](https://github.com/dmar-bonn/neu-nbv) |
| Frequency-based selection | Render each candidate from the same base PLY with fixed background and resolution. Select the lowest median frequency using the authors' FFT statistic. [Paper](https://arxiv.org/html/2409.16470v1) · [Code](https://github.com/lemonci/gs-fft) |
| FisherRF | Use the single-view information criterion and timing protocol above. |
| Random and farthest camera | Random expected gain is the mean over unique actions. Farthest selects the candidate maximizing distance to its nearest acquired camera center. |
| Pose-only and LVSM-only controls | Train the same pose encoder and pooling without the image embedding. Compare separately to a 32-unit embedding-only MLP on the same cached LVSM features and training splits. |
| Later extensions | VIN-NBV estimates geometric improvement from a partial reconstruction and observed or inferred depth. Evaluate its own ranking objective first. POp-GS adds a second information-based Gaussian selector. [VIN-NBV](https://arxiv.org/html/2505.06219v1) · [POp-GS](https://openaccess.thecvf.com/content/CVPR2025/html/Wilson_POp-GS_Next_Best_View_in_3D-Gaussian_Splatting_with_P-Optimality_CVPR_2025_paper.html) |

Replay each base as an independent, forced one-camera decision. Exclude redundancy aliases and average repeated measurements of the same physical action. The existing bundles cover 910 actions across 92 completed contexts on ten objects, with 30k base and 3k candidate training steps. Retain each bundle's evaluator provenance. Use leave-one-object-out predictions for learned models and reserve a separate training-fold object for validation. Compute metrics within contexts, average within each object, then weight objects equally.

Report top-choice regret in dB, selected signed gain, and pairwise ranking accuracy. Use gain MAE only for calibrated gain predictors. For optional capture, report acquisition rate and realized signed gain, using zero for keeping the base. The gain predictor can abstain when every predicted utility is zero. A Fisher score does not by itself supply a calibrated abstention threshold.
