# Lighting

We use `_create_light()` and `randomize_lighting()` from the supplied official
[Objaverse-XL script](../blender_script.py) to follow an established research
protocol and reduce custom design choices.

The rig has four white SUN lights (key, fill, rim and bottom), with the reference
rotations and energy choices, shadows disabled and specular factor 1.0. Lighting
is sampled once per object and stays fixed across its views. Original GLB
materials and the factory world are retained. Cycles uses the reference's
128 samples and bounce settings; dataset camera sampling and FastGS output remain.

Run `python3 renderer.py --glb-path objects/gamecube.glb`. Regenerate existing
datasets to apply the change.

The renderer does not set an explicit lighting RNG seed. Rerendering can
therefore change the sampled light energies; the chosen energies are saved in
`metadata.json`. Reuse one unchanged rendered corpus across benchmark methods
and timing repeats. Use a separate views root for a new lighting dataset.

The optional Blender parity check requires the supplied reference file at
`../blender_script.py` (outside this repository):

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python tests/blender_lighting_check.py
```

It compares the actual light rigs for three seeds and checks Cycles settings
and preservation of the world/materials. The CPU suite's
`tests/test_lighting.py` checks launcher configuration and failure propagation;
it does not render images or validate CUDA.
