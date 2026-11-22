"""Configuration and paths for the separate street demo."""

import json
import math
import sys
from pathlib import Path

COMPLEX = Path(__file__).resolve().parent
INTRO = COMPLEX.parent
ROOT = INTRO.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(INTRO))
from gaussian_initialization import sha256


def run_dir(object_id):
    if (
        not isinstance(object_id, str)
        or not object_id.strip()
        or object_id in (".", "..")
        or "/" in object_id
        or "\\" in object_id
    ):
        raise ValueError("Object ID must be a nonempty single directory name")
    path = INTRO / "runs" / object_id
    if path.is_symlink():
        raise ValueError(f"Demo run directory must not be a symlink: {path}")
    return path


def load_street_config():
    config = json.loads((COMPLEX / "street_config.json").read_text(encoding="utf-8"))
    run_dir(config["object_id"])
    groups = [config["base_views"], config["candidate_views"], config["test_views"]]
    if [len(group) for group in groups] != [4, 2, 5]:
        raise ValueError("The street demo needs four base, two candidate, and five test views")
    views = [view for group in groups for view in group]
    ids = [view["id"] for view in views]
    if len(set(ids)) != len(ids) or any(
        not isinstance(name, str) or not name or not all(c.isalnum() or c in "-_" for c in name)
        for name in ids
    ):
        raise ValueError("Street view IDs must be distinct and contain only letters, numbers, - or _")
    positions = []
    for view in [*views, config["overview"]]:
        fields = ("position", "rotation_degrees") if "rotation_degrees" in view else ("position", "target")
        for field in fields:
            coordinates = view[field]
            if len(coordinates) != 3 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in coordinates
            ):
                raise ValueError(f"Invalid {field} for street view {view.get('id', 'overview')}")
        if "target" in view and view["position"] == view["target"]:
            raise ValueError(f"Camera and target coincide for street view {view['id']}")
        if view is not config["overview"]:
            positions.append(tuple(view["position"]))
    if len(set(positions)) != len(positions):
        raise ValueError("Street camera positions must be distinct")
    if any("rotation_degrees" not in view for view in config["base_views"] + config["candidate_views"]):
        raise ValueError("Captured and candidate cameras must specify Euler rotations")
    if "rotation_degrees" not in config["overview"] or config["overview"]["lens_mm"] <= 0:
        raise ValueError("Overview must specify an Euler rotation and positive lens")
    return config


def street_run_spec(config):
    scene = (INTRO / config["source_scene"]).resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"Street preview scene not found: {scene}")
    return {
        "config": config,
        "source_scene": str(scene),
        "source_sha256": sha256(scene),
        "training_config": json.loads((ROOT / "training_config.json").read_text(encoding="utf-8")),
    }
