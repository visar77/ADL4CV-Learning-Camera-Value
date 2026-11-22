# Pipeline demo

Run from `ObjaverseRenderer/` in the FastGS environment:

```bash
python demo/demo_pipeline/run_pipeline.py --object-id vespa_HP
```

Use `--glb-path /path/to/model.glb` for a different source. `config.json`
selects a `run_id` for each run of an object, the Fibonacci lattice size,
experiment and exactly three distinct non-redundant candidate IDs from that
experiment. The configuration uses the 20-point full sphere and
`random_3_seed_77`. The eight held-out cameras are the renderer's standard
cube-corner test views. The pipeline renders all training and test images
without a floor, then runs the standard orchestrator and evaluation.
The `training` section controls base iterations, candidate adaptation iterations,
and the adaptation learning-rate and densification schedule. Set
`adaptation_densify_until_iter` to `null` to use half the candidate budget or
`0` to disable adaptation densification.
Rendering uses the standard Objaverse-XL lighting rig and Cycles settings,
with a fixed lighting seed and a required CUDA GPU.

Outputs are in `runs/<object-id>/<run-id>/`: `pipeline_1.png` shows the object,
`pipeline_2.png` the full blue lattice and translucent sampling sphere,
`pipeline_3.png` the blue base and labeled red candidates, and
`pipeline_4.png` the green test views and their camera-position cube. Image planes show
the actual rendered views. `selection.json` records the exact lattice indices
used by the figures and the reconstruction jobs. The corpus, reconstructions,
and metrics are also stored in the run directory.
Each overview also has a corresponding `pipeline_1.gif` through
`pipeline_4.gif`: a 60-frame looping turntable at the same elevation, showing
the same object and camera geometry as its PNG. Missing GIFs are generated when
rerunning an existing run.

After evaluation, the demo also renders `pipeline_reconstruct_base.png` and
`pipeline_reconstruct_candidate.png` from the FastGS checkpoints. The matching
`pipeline_reconstruct_evaluate_base.png` and
`pipeline_reconstruct_evaluate_candidate.png` show the held-out cameras with
their respective FastGS test renders. The Gaussian overview is composited as a
camera-facing image at the object center, so its overlap with camera geometry is
an approximation. The candidate defaults to `blindspot-1`; use
`--overview-candidate` to select another configured candidate. These four images
are regenerated on each run and require the evaluated test renders to be present.
The demo also creates `pipeline_reconstruct_base.gif` and
`pipeline_reconstruct_candidate.gif`: 60-frame, looping FastGS turntables using
Blender camera poses at the overview's fixed elevation and angle. The GIFs use
the same candidate as the PNGs and are regenerated on each run.
`pipeline_reconstruct_evaluate_base.gif` and
`pipeline_reconstruct_evaluate_candidate.gif` also show the held-out camera
geometry and evaluated renders throughout the turntable. Like the evaluation
PNGs, they composite the Gaussian render at the object center.

Rerunning reuses existing data and checkpoints. If the GLB, configuration,
experiment, or training settings change, use a new `run_id` in `config.json`.
Existing outputs directly under `runs/<object-id>/` are not moved or overwritten.
Do not run multiple processes for the same object and run ID concurrently.
