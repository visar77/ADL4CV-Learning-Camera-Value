# Prediction inputs and outputs

Run commands from the project directory.

`predict.py --input examples/inference_request.json` is the direct CLI. The example has `context` entries with `image`, `pose`, and `intrinsics`, and `candidates` entries with `id`, `pose`, and `intrinsics`. Image paths are relative to the request file.

Both inference modes report `predicted_gain` for using the candidate and `predicted_utility = max(0, predicted_gain)` for the choice between using it and keeping the base. The conversion happens in the checkpoint's target units after reversing target standardization. For example, a predicted gain of −0.3 dB has utility 0 dB. With `--rank`, equal utilities retain their input order. If all utilities are zero, keeping the base is tied for best. Training labels, regression errors, and the existing forced-capture ranking metrics use signed gains. Measured reconstruction quality can fall when FastGS uses the new view.

New checkpoints default to `target: delta-db`. Their `predicted_gain` and `predicted_delta_db` both report the signed difference `candidate_psnr - base_psnr` in dB, including negative values. Checkpoints trained with `--target relative` report their native relative gain in `predicted_gain`. Offline prediction also reports `ground_truth_relative_gain` and can convert relative predictions to `predicted_delta_db` when the manifest includes the base PSNR.

`gain_predictor` provides the same behavior as a Python API:

```python
import math
from gain_predictor import GainEstimator

predictor = GainEstimator("checkpoints/run/model.pt")  # Auto-selects CUDA when available.
utilities = predictor.predict_utility(
    context_images=images, context_poses=poses, candidate_poses=candidates,
    camera_angle_x=math.radians(50), camera_convention="blender",
)
```

Use `predict()` to retrieve signed gains. If you need both outputs, call `candidate_utility(gains)` from `gain_predictor` on that tensor to reuse the same predictions. Both API methods accept image paths, PIL/NumPy RGB images, or PyTorch `[N,3,H,W]` tensors. Floating input is `[0,1]` and uint8 is `[0,255]`. Poses are camera-to-world matrices. Use `opencv` for OpenCV axes. Supply horizontal FOV in radians or context/candidate intrinsics as pixel K or normalized `[fx/W,fy/H,cx/W,cy/H]`. Pixel K refers to original dimensions. Candidate size defaults to the first image.

Loading/resizing, camera conversion, rays, and scene normalization happen internally. The predictor loads the backbone once and shares one context encoding across candidates per call. `candidate_batch_size` controls decoder memory. Keep the head and matching backbone weights for deployment; their saved relative reference is used automatically, with `--lvsm-checkpoint` / `backbone_checkpoint=` as an override.

For offline inspection, `predict.py --dataset ... --scene-id ... --experiment-id ... --rank` instead scores labelled manifest records. That mode can reuse cached features and display ground truth.
