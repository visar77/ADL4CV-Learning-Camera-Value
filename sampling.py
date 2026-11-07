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
        # Select lattice points closest to evenly spaced targets on a fixed
        # horizontal orbit, then return them in orbit order.
        elevation = math.radians(trajectory_elevation_deg)
        targets = []

        for j in range(count):
            azimuth = 2.0 * math.pi * j / count
            targets.append((
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ))

        remaining = set(available)
        selected = []

        for target in targets:
            chosen = min(
                remaining,
                key=lambda i: _distance_squared(lattice[i], target),
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
