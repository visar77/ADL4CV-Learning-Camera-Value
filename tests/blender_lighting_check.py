"""Compare actual Blender lights with the supplied official reference functions."""

import ast
from pathlib import Path
import random
import sys
from typing import Dict, Literal, Tuple

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blender_worker import configure_render, randomize_lighting


def light_state():
    return {
        obj.name: (
            obj.data.type,
            tuple(obj.location),
            tuple(obj.rotation_euler),
            obj.data.energy,
            obj.data.use_shadow,
            obj.data.specular_factor,
            tuple(obj.data.color),
            obj.data.angle,
        )
        for obj in bpy.context.scene.objects
        if obj.type == "LIGHT"
    }


reference = Path(__file__).resolve().parents[2] / "blender_script.py"
tree = ast.parse(reference.read_text())
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"_create_light", "randomize_lighting"}
]
assert len(functions) == 2
namespace = {
    "bpy": bpy,
    "random": random,
    "Dict": Dict,
    "Literal": Literal,
    "Tuple": Tuple,
}
exec(
    compile(ast.Module(body=functions, type_ignores=[]), str(reference), "exec"),
    namespace,
)
scene = bpy.context.scene
world = scene.world
materials = list(bpy.data.materials)
for seed in (0, 77, 78):
    random.seed(seed)
    namespace["randomize_lighting"]()
    expected = light_state()
    random.seed(seed)
    lights = randomize_lighting()
    assert len(lights) == 4
    assert light_state() == expected, f"Reference mismatch for seed {seed}"
    assert scene.world == world
    assert list(bpy.data.materials) == materials
configure_render(scene, {"image_width": 512, "image_height": 512, "use_gpu": False})
assert scene.cycles.samples == 128
assert scene.cycles.diffuse_bounces == scene.cycles.glossy_bounces == 1
assert scene.cycles.transparent_max_bounces == scene.cycles.transmission_bounces == 3
assert abs(scene.cycles.filter_width - 0.01) < 1e-6
assert scene.cycles.use_denoising and scene.render.film_transparent
assert scene.view_layers[0].material_override is None
print(
    "PASS official light parity for three seeds, original world/materials, and Cycles settings"
)
