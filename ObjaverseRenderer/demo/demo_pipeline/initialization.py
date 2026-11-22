"""Prepare the pipeline demo's initial point cloud."""


def prepare_points(object_dir, metadata, run_root):
    import numpy as np
    from plyfile import PlyData, PlyElement

    path = object_dir / "points3d.ply"
    if path.exists():
        return
    if (run_root / "reconstructions").exists() or (run_root / "results").exists():
        raise FileNotFoundError("Cannot replace the initialization of an existing demo run")
    center = np.asarray(metadata["center"])
    radius = metadata["bounding_radius"]
    xyz = np.random.default_rng(77).uniform(-1.5 * radius, 1.5 * radius, size=(100_000, 3)) + center
    vertices = np.empty(100_000, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, axis]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0
    for name in ("red", "green", "blue"):
        vertices[name] = 128
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
