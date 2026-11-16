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


def _trajectory_period(lattice_size):
    if lattice_size < 1:
        raise ValueError("Lattice size must be at least 1.")
    return min(lattice_size, max(1, round(math.sqrt(math.pi * lattice_size))))


def _trajectory_parameters(seed, trajectory_elevation_deg, trajectory_period=100):
    """Derive a repeatable ring orientation and phase from a seed."""
    if trajectory_period < 1:
        raise ValueError("Trajectory period must be at least 1.")
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
        "azimuth_step": 2.0 * math.pi / trajectory_period,
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
    trajectory_period=100,
):
    """Return the first `count` points on a seeded, count-independent ring."""
    if count < 1:
        raise ValueError("View count must be at least 1.")

    parameters = _trajectory_parameters(
        seed,
        trajectory_elevation_deg,
        trajectory_period,
    )
    return [_trajectory_direction(step, parameters) for step in range(count)]


def trajectory_path(count, seed=0, trajectory_elevation_deg=20.0):
    """Return points tracing a complete, smooth randomized ring."""
    if count < 1:
        raise ValueError("Path point count must be at least 1.")

    parameters = _trajectory_parameters(seed, trajectory_elevation_deg, count)
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
            trajectory_period=_trajectory_period(len(lattice)),
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
    """Select one candidate view using the candidate-specific strategy rules."""
    return select_candidate_indices(
        strategy=strategy,
        count=1,
        lattice=lattice,
        training_indices=training_indices,
        seed=seed,
        trajectory_elevation_deg=trajectory_elevation_deg,
    )[0]


def canonical_candidate_strategy(strategy):
    strategy = strategy.lower()
    aliases = {
        "furthest": "blindspot",
        "farthest": "blindspot",
        "furthest_point": "blindspot",
        "trajectory": "next",
        "predefined_trajectory": "next",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in ("redundancy", "blindspot", "random", "next"):
        raise ValueError(f"Unknown candidate strategy: {strategy}")
    return strategy


def candidate_configurations(config):
    """Normalize supported candidate JSON forms to unique (strategy, count) pairs."""
    if config is None:
        return []
    if isinstance(config, dict):
        if "strategy" in config:
            config = [config]
        else:
            expanded = []
            for strategy, settings in config.items():
                if isinstance(settings, list):
                    expanded.extend(
                        {"strategy": strategy, "count": count}
                        for count in settings
                    )
                elif isinstance(settings, dict):
                    expanded.append({"strategy": strategy, **settings})
                else:
                    expanded.append({"strategy": strategy, "count": settings})
            config = expanded
    if not isinstance(config, list):
        raise ValueError("'candidates' must be a list or strategy configuration.")

    configurations = []
    seen = set()
    for candidate in config:
        if not isinstance(candidate, dict) or "strategy" not in candidate:
            raise ValueError("Each candidate must have a strategy and count.")
        strategy = canonical_candidate_strategy(candidate["strategy"])
        if strategy != "blindspot" and "count" not in candidate:
            raise ValueError(f"Candidate strategy '{strategy}' requires a count.")
        count = 1 if strategy == "blindspot" else int(candidate["count"])
        if count < 1:
            raise ValueError("Candidate count must be at least 1.")
        key = (strategy, count)
        if key in seen:
            raise ValueError(
                f"Duplicate candidate configuration: {strategy}-{count}."
            )
        seen.add(key)
        configurations.append(key)
    return configurations


def select_candidate_indices(
    strategy,
    count,
    lattice,
    training_indices,
    seed=0,
    train_strategy=None,
    trajectory_elevation_deg=20.0,
):
    """Select independent candidate views without adding them to training together."""
    strategy = canonical_candidate_strategy(strategy)
    training_indices = list(training_indices)

    if strategy == "blindspot":
        count = 1
        if not training_indices:
            raise ValueError("Blindspot selection requires at least one training view.")
        available = [i for i in range(len(lattice)) if i not in training_indices]
        if not available:
            raise ValueError("Blindspot selection requires an unused lattice point.")
        return [
            max(
                available,
                key=lambda i: (
                    min(
                        _distance_squared(lattice[i], lattice[j])
                        for j in training_indices
                    ),
                    -i,
                ),
            )
        ]

    if count < 1:
        raise ValueError("Candidate count must be at least 1.")

    if strategy == "redundancy":
        if count > len(training_indices):
            raise ValueError(
                f"Requested {count} redundant candidates, but only "
                f"{len(training_indices)} training views are available."
            )
        return random.Random(seed).sample(training_indices, count)

    available = [i for i in range(len(lattice)) if i not in training_indices]
    if count > len(available):
        raise ValueError(
            f"Requested {count} candidates, but only {len(available)} "
            "unused lattice points remain."
        )

    if strategy == "random":
        return random.Random(seed).sample(available, count)

    # "Next" continues after trajectory training, but starts at the beginning
    # of the path for training strategies with no trajectory ordering.
    prefix_count = len(training_indices)
    schedule = select_indices(
        "trajectory",
        len(lattice),
        lattice,
        seed=seed,
        trajectory_elevation_deg=trajectory_elevation_deg,
    )
    if (
        train_strategy is not None
        and train_strategy.lower() in ("trajectory", "predefined_trajectory")
        and schedule[:prefix_count] != training_indices
    ):
        raise ValueError(
            "The base training indices do not match the configured trajectory."
        )

    is_trajectory_training = (
        train_strategy is not None
        and train_strategy.lower() in ("trajectory", "predefined_trajectory")
    )
    start = prefix_count % len(schedule) if is_trajectory_training else 0
    ordered_schedule = schedule[start:] + schedule[:start]
    selected = [
        index for index in ordered_schedule if index not in training_indices
    ][:count]
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} candidates, but only {len(selected)} "
            "unused lattice points remain."
        )
    return selected
