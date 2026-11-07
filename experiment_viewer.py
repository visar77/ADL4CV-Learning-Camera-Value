"""Visualize an experiment's training views on the Fibonacci sphere."""

import argparse
import json
from pathlib import Path

from sampling import (
    fibonacci_lattice,
    select_indices,
    test_view_directions,
    trajectory_path,
)

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
VIEWS_DIR = ROOT / "views"
DEFAULT_LATTICE_SIZE = 100
TRAJECTORY_ELEVATION_DEG = 20.0


def load_experiment(value):
    path = Path(value)
    if not path.is_file():
        filename = value if value.endswith(".json") else f"{value}.json"
        path = EXPERIMENTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    with path.open(encoding="utf-8") as file:
        experiment = json.load(file)
    if "train" not in experiment:
        raise ValueError(f"Experiment has no 'train' configuration: {path}")
    experiment.setdefault("id", path.stem)
    return experiment


def resolve_lattice_size(object_id, override):
    if override is not None:
        return override
    if object_id:
        metadata_path = VIEWS_DIR / object_id / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            return int(metadata["lattice_size"])
    return DEFAULT_LATTICE_SIZE


def resolve_trajectory_elevation(object_id):
    if object_id:
        metadata_path = VIEWS_DIR / object_id / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            return float(
                metadata.get(
                    "trajectory_elevation_deg",
                    TRAJECTORY_ELEVATION_DEG,
                )
            )
    return TRAJECTORY_ELEVATION_DEG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show an experiment's training camera views on a sphere."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment JSON path or filename/id from the experiments directory.",
    )
    parser.add_argument(
        "--object-id",
        help="Optional object ID; uses its rendered lattice size when available.",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        help="Override the lattice size (defaults to object metadata or 100).",
    )
    parser.add_argument(
        "--show-lattice",
        action="store_true",
        help="Show every Fibonacci lattice point, in addition to the training views.",
    )
    parser.add_argument(
        "--show-test-views",
        action="store_true",
        help="Show the fixed test cameras in green.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the figure here (defaults to experiment_viewer_<id>.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive window (requires a working GUI backend).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def draw_camera(axes, position, np, color):
    """Draw a small wireframe camera whose lens points toward the sphere center."""
    outward = position / np.linalg.norm(position)
    forward = -outward
    reference_up = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(forward, reference_up)) > 0.95:
        reference_up = np.array((0.0, 1.0, 0.0))
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    def rectangle(depth, half_width, half_height):
        center = position + forward * depth
        return [
            center + right * x * half_width + up * y * half_height
            for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

    def draw_loop(vertices):
        for index in range(4):
            start, end = vertices[index], vertices[(index + 1) % 4]
            axes.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color=color,
                linewidth=1.6,
            )

    body_back = rectangle(-0.10, 0.055, 0.045)
    body_front = rectangle(-0.025, 0.055, 0.045)
    lens_base = rectangle(-0.02, 0.032, 0.027)
    lens_front = rectangle(0.065, 0.015, 0.013)

    draw_loop(body_back)
    draw_loop(body_front)
    draw_loop(lens_base)
    draw_loop(lens_front)
    for index in range(4):
        for rear, front in ((body_back, body_front), (lens_base, lens_front)):
            start, end = rear[index], front[index]
            axes.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color=color,
                linewidth=1.6,
            )


def main():
    args = parse_args()
    experiment = load_experiment(args.experiment)
    lattice_size = resolve_lattice_size(args.object_id, args.lattice_size)
    trajectory_elevation = resolve_trajectory_elevation(args.object_id)
    lattice = fibonacci_lattice(lattice_size)
    train = experiment["train"]
    train_indices = select_indices(
        strategy=train["strategy"],
        count=int(train["count"]),
        lattice=lattice,
        seed=int(train.get("seed", 0)),
        trajectory_elevation_deg=trajectory_elevation,
    )
    is_trajectory = train["strategy"].lower() in (
        "trajectory",
        "predefined_trajectory",
    )

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    figure = plt.figure(figsize=(9, 8))
    axes = figure.add_subplot(111, projection="3d")
    azimuth = np.linspace(0, 2 * np.pi, 64)
    elevation = np.linspace(0, np.pi, 32)
    sphere_x = np.outer(np.cos(azimuth), np.sin(elevation))
    sphere_y = np.outer(np.sin(azimuth), np.sin(elevation))
    sphere_z = np.outer(np.ones_like(azimuth), np.cos(elevation))
    axes.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#8fb9d8",
        alpha=0.16,
        linewidth=0,
        shade=True,
    )

    lattice_points = np.asarray(lattice)
    if args.show_lattice:
        axes.scatter(
            lattice_points[:, 0],
            lattice_points[:, 1],
            lattice_points[:, 2],
            color="#586b78",
            alpha=0.55,
            s=14,
            depthshade=False,
            label=f"Fibonacci lattice ({lattice_size})",
        )

    if is_trajectory:
        full_path = np.asarray(
            trajectory_path(
                720,
                seed=int(train.get("seed", 0)),
                trajectory_elevation_deg=trajectory_elevation,
            )
        )
        axes.plot(
            full_path[:, 0],
            full_path[:, 1],
            full_path[:, 2],
            color="#596b75",
            linewidth=1.5,
            linestyle="--",
            label="Randomized trajectory ring",
        )
        current_points = lattice_points[train_indices]
        axes.scatter(
            *current_points[0],
            color="#2878b5",
            marker="*",
            s=150,
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            zorder=10,
        )
        axes.scatter(
            *current_points[-1],
            color="#d1495b",
            marker="D",
            s=75,
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            zorder=10,
        )

    points = lattice_points[train_indices]
    for point in points:
        draw_camera(axes, point, np, "#d1495b")
    for index, (x, y, z) in zip(train_indices, points):
        axes.text(x * 1.17, y * 1.17, z * 1.17, f"{index:03d}", fontsize=8)

    if args.show_test_views:
        test_points = np.asarray(test_view_directions())
        for point in test_points:
            draw_camera(axes, point, np, "#26964b")
        for index, (x, y, z) in enumerate(test_points):
            axes.text(x * 1.17, y * 1.17, z * 1.17, f"T{index}", fontsize=8)

    legend_handles = [
        Line2D(
            [],
            [],
            color="#d1495b",
            linewidth=2,
            label=f"Training cameras ({len(train_indices)}; lenses point inward)",
        ),
    ]
    if args.show_lattice:
        legend_handles.insert(
            0,
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color="#586b78",
                label=f"Fibonacci lattice ({lattice_size})",
            ),
        )
    if args.show_test_views:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#26964b",
                linewidth=2,
                label="Fixed test cameras (8)",
            )
        )
    if is_trajectory:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#596b75",
                linewidth=1.5,
                linestyle="--",
                label="Randomized trajectory ring",
            )
        )
        legend_handles.extend(
            (
                Line2D(
                    [],
                    [],
                    marker="*",
                    linestyle="None",
                    color="#2878b5",
                    markersize=12,
                    label="First selected camera",
                ),
                Line2D(
                    [],
                    [],
                    marker="D",
                    linestyle="None",
                    color="#d1495b",
                    markersize=7,
                    label="Most recently selected camera",
                ),
            )
        )

    axes.set(
        xlim=(-1.25, 1.25),
        ylim=(-1.25, 1.25),
        zlim=(-1.25, 1.25),
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        title=(
            f"{experiment['id']} - camera views"
            if args.show_test_views
            else f"{experiment['id']} - training camera views"
        ),
    )
    axes.set_box_aspect((1, 1, 1))
    axes.legend(handles=legend_handles, loc="upper left")
    figure.tight_layout()

    output = args.output or Path(f"experiment_viewer_{experiment['id']}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved viewer figure to: {output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
