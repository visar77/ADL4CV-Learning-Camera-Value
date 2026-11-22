# Pipeline demo

Run from `ObjaverseRenderer/` in the FastGS environment:

```bash
python demo/demo_pipeline/run_pipeline.py --object-id vespa_HP
```

Use `--glb-path /path/to/model.glb` for a different source; the object ID
is the isolated run name. `config.json` selects the Fibonacci lattice size,
experiment and exactly three distinct non-redundant candidate IDs from that
experiment. The default uses the 20-point full sphere and
`random_5_seed_77`. The eight held-out cameras are the renderer's standard
cube-corner test views. The pipeline renders all training and test images
without a floor, then runs the standard orchestrator: a 30,000-step base,
three independent 3,000-step adaptations from that base, and evaluation.
Rendering uses the standard Objaverse-XL lighting rig and Cycles settings,
with a fixed lighting seed and a required CUDA GPU.

Outputs are in `runs/<object-id>/`: `pipeline_1.png` shows the object,
`pipeline_2.png` the full blue lattice and translucent sampling sphere,
`pipeline_3.png` the blue base and labeled red candidates, and
`pipeline_4.png` the green test views and their camera-position cube. Image planes show
the actual rendered views. `selection.json` records the exact lattice indices
used by the figures and the reconstruction jobs. The corpus, reconstructions,
and metrics are also stored in the run directory.

Rerunning reuses existing data and checkpoints. If the GLB, configuration,
experiment, or training settings change, use a new object ID. Do not run
multiple processes for the same object ID concurrently.
