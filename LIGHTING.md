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
