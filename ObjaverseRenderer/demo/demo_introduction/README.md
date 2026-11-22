# Vespa street camera demo

The demo renders an editable double-facade street scene with the Vespa,
four blue **Captured** cameras and two red **Candidate** cameras. The captured
image planes show the actual rendered views; candidates show question marks.
Camera positions and Blender XYZ Euler rotations (degrees), including the
overview camera, are in `complex_scene/street_config.json`. The five additional
test views are used only for the optional reconstruction evaluation.

From `ObjaverseRenderer/`, run:

```bash
conda activate visarfastgs
python demo/demo_introduction/run_demo.py
```

Blender with a CUDA-capable GPU is required; all views and the overview use
Cycles GPU. The packed source scene is `assets/vespa_street.blend`. Outputs are
written to `runs/<object_id>/`: `camera_overview.png`, an editable
`camera_overview.blend`, the rendered views, and camera manifests. A rerun
reuses complete outputs. If the scene or configuration changes, select a new
`object_id` in the configuration to avoid overwriting existing work.

Optionally reconstruct and evaluate the base and candidate views using FastGS:

```bash
python demo/demo_introduction/complex_scene/run_street_reconstruction.py
```

Each candidate is independently adapted from the same four-view base and
evaluated on the same held-out views. The initialization is object-centered,
so reconstruction quality may be limited for the surrounding street.

The packed street uses Poly Haven's CC0
[Modular Urban Apartments Facade](https://polyhaven.com/a/modular_urban_apartments_facade),
[Street Lamp 01](https://polyhaven.com/a/street_lamp_01),
[Asphalt 04](https://polyhaven.com/a/asphalt_04), and
[Concrete Pavement 02](https://polyhaven.com/a/concrete_pavement_02).
