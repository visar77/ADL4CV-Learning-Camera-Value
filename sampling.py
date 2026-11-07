"""Select view directions from a shared Fibonacci lattice."""

import math
import random


def fibonacci_lattice(size):
    """Create the fixed set of candidate directions used by every strategy."""
    if size < 1:
        raise ValueError("Lattice size must be at least 1.")

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []

    for i in range(size):
        z = 1.0 - 2.0 * (i + 0.5) / size
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * i
        points.append((radius * math.cos(phi), radius * math.sin(phi), z))

    return points


def test_view_directions():
    """Return the fixed cube-corner camera directions used for test renders."""
    scale = 1.0 / math.sqrt(3.0)
    return [
        (-scale, -scale, -scale),
        (-scale, -scale, scale),
        (-scale, scale, -scale),
        (-scale, scale, scale),
        (scale, -scale, -scale),
        (scale, -scale, scale),
        (scale, scale, -scale),
        (scale, scale, scale),
    ]


def _distance_squared(a, b):
    return sum((a[k] - b[k]) ** 2 for k in range(3))


def _trajectory_parameters(seed, trajectory_elevation_deg):
    """Derive a repeatable ring orientation and phase from a seed."""
    rng = random.Random(seed)
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)

    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )
    return {
        "rotation": rotation,
        "azimuth_phase": rng.uniform(0.0, 2.0 * math.pi),
        "azimuth_step": 2.0 * math.pi * ((math.sqrt(5.0) - 1.0) / 2.0),
        "elevation_offset": math.radians(trajectory_elevation_deg),
    }


def _trajectory_direction(step, parameters):
    azimuth = parameters["azimuth_phase"] + step * parameters["azimuth_step"]
    elevation = parameters["elevation_offset"]
    local = (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )
    rotation = parameters["rotation"]
    return tuple(
        sum(rotation[row][column] * local[column] for column in range(3))
        for row in range(3)
    )


def trajectory_directions(
    count,
    seed=0,
    trajectory_elevation_deg=20.0,
):
    """Return the first `count` targets of a seeded, count-independent path."""
    if count < 1:
        raise ValueError("View count must be at least 1.")

    parameters = _trajectory_parameters(seed, trajectory_elevation_deg)
    return [_trajectory_direction(step, parameters) for step in range(count)]


def trajectory_path(count, seed=0, trajectory_elevation_deg=20.0):
    """Return points tracing a complete, smooth randomized ring."""
    if count < 1:
        raise ValueError("Path point count must be at least 1.")

    parameters = _trajectory_parameters(seed, trajectory_elevation_deg)
    parameters["azimuth_step"] = 2.0 * math.pi
    return [
        _trajectory_direction(step / count, parameters)
        for step in range(count + 1)
    ]


def select_indices(
    strategy,
    count,
    lattice,
    seed=0,
    excluded=(),
    trajectory_elevation_deg=20.0,
):
    """
    Select lattice indices only. `excluded` indices are unavailable, allowing
    a candidate view to be selected without duplicating a training view.
    """
    strategy = strategy.lower()
    excluded = set(excluded)
    available = [i for i in range(len(lattice)) if i not in excluded]

    if count < 1:
        raise ValueError("View count must be at least 1.")
    if count > len(available):
        raise ValueError(
            f"Requested {count} views, but only {len(available)} "
            "unused lattice points remain."
        )

    rng = random.Random(seed)

    if strategy == "random":
        return rng.sample(available, count)

    if strategy in ("furthest", "farthest", "furthest_point"):
        first = rng.choice(available)
        selected = [first]
        remaining = set(available)
        remaining.remove(first)

        while len(selected) < count:
            chosen = max(
                remaining,
                key=lambda i: min(
                    _distance_squared(lattice[i], lattice[j])
                    for j in selected
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)

        return selected

    if strategy in ("trajectory", "predefined_trajectory"):
        # Greedily map each count-independent path target to its nearest
        # unused lattice point, preserving the selected sequence as count grows.
        targets = trajectory_directions(
            count,
            seed=seed,
            trajectory_elevation_deg=trajectory_elevation_deg,
        )
        remaining = available.copy()
        selected = []

        for target in targets:
            chosen = min(
                remaining,
                key=lambda i: (_distance_squared(lattice[i], target), i),
            )
            selected.append(chosen)
            remaining.remove(chosen)

        return selected

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def select_candidate_index(
    strategy,
    lattice,
    training_indices,
    seed=0,
    trajectory_elevation_deg=20.0,
):
    """Select one unused candidate from the same lattice."""
    return select_indices(
        strategy=strategy,
        count=1,
        lattice=lattice,
        seed=seed,
        excluded=training_indices,
        trajectory_elevation_deg=trajectory_elevation_deg,
    )[0]
